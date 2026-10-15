package io.github.kernelin.compiler

import org.jetbrains.kotlin.backend.common.extensions.IrGenerationExtension
import org.jetbrains.kotlin.backend.common.extensions.IrPluginContext
import org.jetbrains.kotlin.backend.common.lower.DeclarationIrBuilder
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageLocation
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.cli.common.messages.MessageCollector
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.builders.irCallConstructor
import org.jetbrains.kotlin.ir.builders.irString
import org.jetbrains.kotlin.ir.declarations.IrFile
import org.jetbrains.kotlin.ir.declarations.IrFunction
import org.jetbrains.kotlin.ir.declarations.IrModuleFragment
import org.jetbrains.kotlin.ir.declarations.IrSimpleFunction
import org.jetbrains.kotlin.ir.expressions.IrCall
import org.jetbrains.kotlin.ir.expressions.IrExpression
import org.jetbrains.kotlin.ir.expressions.IrFunctionReference
import org.jetbrains.kotlin.ir.expressions.IrRichCallableReference
import org.jetbrains.kotlin.ir.util.hasAnnotation
import org.jetbrains.kotlin.ir.util.kotlinFqName
import org.jetbrains.kotlin.ir.visitors.IrElementTransformerVoid
import org.jetbrains.kotlin.ir.visitors.IrVisitorVoid
import org.jetbrains.kotlin.ir.visitors.acceptChildrenVoid
import org.jetbrains.kotlin.ir.visitors.transformChildrenVoid
import org.jetbrains.kotlin.name.ClassId
import org.jetbrains.kotlin.name.FqName
import org.jetbrains.kotlin.name.Name
import java.io.File

internal val KERNEL_ANNOTATION = FqName("io.github.kernelin.Kernel")
private val KERNEL_SOURCE_FN = FqName("io.github.kernelin.kernelSource")
private val KERNEL_SOURCE_CLASS = ClassId(FqName("io.github.kernelin"), Name.identifier("KernelSource"))

/**
 * Finds every `@Kernel` function and translates it to WGSL. The result is used two ways:
 * - on the JVM it is written next to the classes (`kernelin/<fq.name>.wgsl/.json`), so it ships in the jar;
 * - on every platform, `kernelSource(::kernel)` calls are replaced by a `KernelSource` holding the shader.
 */
class KernelinIrGenerationExtension(
    private val messages: MessageCollector,
    private val outputDir: File?,
) : IrGenerationExtension {

    override fun generate(moduleFragment: IrModuleFragment, pluginContext: IrPluginContext) {
        val shaders = HashMap<IrSimpleFunction, Pair<String, WgslShader>>()
        for (file in moduleFragment.files) {
            val kernels = mutableListOf<IrSimpleFunction>()
            file.acceptChildrenVoid(object : IrVisitorVoid() {
                override fun visitElement(element: IrElement) {
                    element.acceptChildrenVoid(this)
                }

                override fun visitSimpleFunction(declaration: IrSimpleFunction) {
                    if (declaration.hasAnnotation(KERNEL_ANNOTATION)) kernels += declaration
                    super.visitSimpleFunction(declaration)
                }
            })
            for (kernel in kernels) translate(file, kernel)?.let { shaders[kernel] = kernel.kotlinFqName.asString() to it }
        }
        for (file in moduleFragment.files) embedKernelSources(file, pluginContext, shaders)
    }

    private fun translate(file: IrFile, kernel: IrSimpleFunction): WgslShader? {
        val fqName = kernel.kotlinFqName.asString()
        val result = WgslKernelTranslator(kernel).translate()
        if (result.errors.isNotEmpty()) {
            for (error in result.errors) {
                messages.report(
                    CompilerMessageSeverity.ERROR,
                    "Kernelin: ${error.message} (in @Kernel $fqName)",
                    (error.file ?: file).location(error.element),
                )
            }
            return null
        }
        val shader = result.shader!!
        messages.report(
            CompilerMessageSeverity.INFO,
            "Kernelin: translated @Kernel $fqName to WGSL (${shader.wgsl.lines().size} lines)",
        )
        if (outputDir != null) {
            outputDir.mkdirs()
            outputDir.resolve("$fqName.wgsl").writeText(shader.wgsl)
            outputDir.resolve("$fqName.json").writeText(shader.manifestJson(fqName))
        }
        return shader
    }

    /** Replaces `kernelSource(::someKernel)` with `KernelSource("<fq name>", "<wgsl>", "<manifest json>")`. */
    private fun embedKernelSources(
        file: IrFile,
        context: IrPluginContext,
        shaders: Map<IrSimpleFunction, Pair<String, WgslShader>>,
    ) {
        file.transformChildrenVoid(object : IrElementTransformerVoid() {
            override fun visitCall(expression: IrCall): IrExpression {
                expression.transformChildrenVoid(this)
                val callee = expression.symbol.owner
                if (callee.kotlinFqName != KERNEL_SOURCE_FN) return expression

                val target = when (val ref = expression.arguments.firstOrNull()) {
                    is IrFunctionReference -> ref.symbol.owner
                    is IrRichCallableReference<*> -> ref.reflectionTargetSymbol?.owner as? IrFunction
                    else -> null
                }
                val error = when {
                    target == null -> "kernelSource() needs a direct reference to a kernel, like kernelSource(::myKernel)"
                    !target.hasAnnotation(KERNEL_ANNOTATION) -> "'${target.kotlinFqName}' isn't annotated with @Kernel"
                    target !in shaders -> "'${target.kotlinFqName}' must be a @Kernel from this module that translated without errors"
                    else -> null
                }
                if (error != null) {
                    messages.report(CompilerMessageSeverity.ERROR, "Kernelin: $error", file.location(expression))
                    return expression
                }
                val (fqName, shader) = shaders.getValue(target as IrSimpleFunction)
                val constructor = context.referenceClass(KERNEL_SOURCE_CLASS)?.owner?.declarations
                    ?.filterIsInstance<org.jetbrains.kotlin.ir.declarations.IrConstructor>()?.singleOrNull()
                    ?: return expression.also {
                        messages.report(CompilerMessageSeverity.ERROR, "Kernelin: KernelSource class not found; add kernelin-runtime", file.location(expression))
                    }
                val builder = DeclarationIrBuilder(context, callee.symbol, expression.startOffset, expression.endOffset)
                return builder.irCallConstructor(constructor.symbol, emptyList()).apply {
                    arguments[0] = builder.irString(fqName)
                    arguments[1] = builder.irString(shader.wgsl)
                    arguments[2] = builder.irString(shader.manifestJson(fqName))
                }
            }
        })
    }

    private fun IrFile.location(element: IrElement): CompilerMessageLocation? {
        if (element.startOffset < 0) return null
        val line = fileEntry.getLineNumber(element.startOffset)
        val column = fileEntry.getColumnNumber(element.startOffset)
        return CompilerMessageLocation.create(fileEntry.name, line + 1, column + 1, null)
    }
}
