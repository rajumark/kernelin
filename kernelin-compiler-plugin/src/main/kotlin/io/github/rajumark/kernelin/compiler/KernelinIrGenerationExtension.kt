package io.github.rajumark.kernelin.compiler

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

internal val KERNEL_ANNOTATION = FqName("io.github.rajumark.kernelin.Kernel")
private val KERNEL_SOURCE_FN = FqName("io.github.rajumark.kernelin.kernelSource")
private val KERNEL_SOURCE_CLASS = ClassId(FqName("io.github.rajumark.kernelin"), Name.identifier("KernelSource"))

/**
 * Finds every `@Kernel` function and translates it to WGSL. The result is used two ways:
 * - it is written to [outputDir] (`<fq.name>.wgsl/.json`). On the JVM that is next to the classes, so it ships
 *   in the jar; the Gradle plugin gives other platforms their own folder;
 * - on every platform, `kernelSource(::kernel)` calls are replaced by a `KernelSource` holding the shader.
 *
 * An incremental build can recompile a file calling `kernelSource(::kernel)` without the file holding the kernel
 * (for example after a library it uses changed). The kernel then has no body to translate here, so the shader
 * written to [outputDir] when the kernel was last compiled is used instead.
 */
class KernelinIrGenerationExtension(
    private val messages: MessageCollector,
    private val outputDir: File?,
) : IrGenerationExtension {

    override fun generate(moduleFragment: IrModuleFragment, pluginContext: IrPluginContext) {
        val shaders = HashMap<IrSimpleFunction, CompiledShader>()
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
            for (kernel in kernels) translate(file, kernel)?.let { shaders[kernel] = it }
        }
        for (file in moduleFragment.files) embedKernelSources(file, pluginContext, shaders)
    }

    private fun translate(file: IrFile, kernel: IrSimpleFunction): CompiledShader? {
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
        val shader = result.shader!!.let { CompiledShader(fqName, it.wgsl, it.manifestJson(fqName)) }
        // Only with the compiler's -verbose flag: a line per kernel is noise in a normal build log.
        messages.report(
            CompilerMessageSeverity.LOGGING,
            "Kernelin: translated @Kernel $fqName to WGSL (${shader.wgsl.lines().size} lines)",
        )
        if (outputDir != null) {
            outputDir.mkdirs()
            outputDir.resolve("$fqName.wgsl").writeText(shader.wgsl)
            outputDir.resolve("$fqName.json").writeText(shader.manifestJson)
        }
        return shader
    }

    /** The shader written when [kernel] was last compiled in this module (it isn't part of this compilation). */
    private fun savedShader(kernel: IrFunction): CompiledShader? {
        if (outputDir == null || kernel.body != null) return null
        val fqName = kernel.kotlinFqName.asString()
        val wgsl = outputDir.resolve("$fqName.wgsl").takeIf { it.isFile } ?: return null
        val manifest = outputDir.resolve("$fqName.json").takeIf { it.isFile } ?: return null
        return CompiledShader(fqName, wgsl.readText(), manifest.readText())
    }

    /** Replaces `kernelSource(::someKernel)` with `KernelSource("<fq name>", "<wgsl>", "<manifest json>")`. */
    private fun embedKernelSources(
        file: IrFile,
        context: IrPluginContext,
        shaders: Map<IrSimpleFunction, CompiledShader>,
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
                val shader = target?.let { shaders[it] ?: savedShader(it) }
                val error = when {
                    target == null -> "kernelSource() needs a direct reference to a kernel, like kernelSource(::myKernel)"
                    !target.hasAnnotation(KERNEL_ANNOTATION) -> "'${target.kotlinFqName}' isn't annotated with @Kernel"
                    shader == null -> "'${target.kotlinFqName}' must be a @Kernel from this module that translated without errors"
                    else -> null
                }
                if (error != null || shader == null) {
                    messages.report(CompilerMessageSeverity.ERROR, "Kernelin: $error", file.location(expression))
                    return expression
                }
                val constructor = context.referenceClass(KERNEL_SOURCE_CLASS)?.owner?.declarations
                    ?.filterIsInstance<org.jetbrains.kotlin.ir.declarations.IrConstructor>()?.singleOrNull()
                    ?: return expression.also {
                        messages.report(CompilerMessageSeverity.ERROR, "Kernelin: KernelSource class not found; add kernelin-runtime", file.location(expression))
                    }
                val builder = DeclarationIrBuilder(context, callee.symbol, expression.startOffset, expression.endOffset)
                return builder.irCallConstructor(constructor.symbol, emptyList()).apply {
                    arguments[0] = builder.irString(shader.fqName)
                    arguments[1] = builder.irString(shader.wgsl)
                    arguments[2] = builder.irString(shader.manifestJson)
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

/** A translated kernel, as embedded by `kernelSource()`. */
private class CompiledShader(val fqName: String, val wgsl: String, val manifestJson: String)
