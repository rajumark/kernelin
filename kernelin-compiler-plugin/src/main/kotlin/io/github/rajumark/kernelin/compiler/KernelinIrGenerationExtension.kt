package io.github.rajumark.kernelin.compiler

import org.jetbrains.kotlin.backend.common.extensions.IrGenerationExtension
import org.jetbrains.kotlin.backend.common.extensions.IrPluginContext
import org.jetbrains.kotlin.backend.common.lower.DeclarationIrBuilder
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageLocation
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.cli.common.messages.MessageCollector
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.builders.IrBuilderWithScope
import org.jetbrains.kotlin.ir.builders.irBlock
import org.jetbrains.kotlin.ir.builders.irCall
import org.jetbrains.kotlin.ir.builders.irCallConstructor
import org.jetbrains.kotlin.ir.builders.irGet
import org.jetbrains.kotlin.ir.builders.irInt
import org.jetbrains.kotlin.ir.builders.irString
import org.jetbrains.kotlin.ir.declarations.IrClass
import org.jetbrains.kotlin.ir.declarations.IrConstructor
import org.jetbrains.kotlin.ir.declarations.IrFile
import org.jetbrains.kotlin.ir.declarations.IrFunction
import org.jetbrains.kotlin.ir.declarations.IrModuleFragment
import org.jetbrains.kotlin.ir.declarations.IrParameterKind
import org.jetbrains.kotlin.ir.declarations.IrSimpleFunction
import org.jetbrains.kotlin.ir.declarations.IrValueParameter
import org.jetbrains.kotlin.ir.expressions.IrCall
import org.jetbrains.kotlin.ir.expressions.IrExpression
import org.jetbrains.kotlin.ir.expressions.IrFunctionExpression
import org.jetbrains.kotlin.ir.expressions.IrFunctionReference
import org.jetbrains.kotlin.ir.expressions.IrRichCallableReference
import org.jetbrains.kotlin.ir.types.classFqName
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
internal val KERNEL_LAUNCH_CLASS = ClassId(FqName("io.github.rajumark.kernelin"), Name.identifier("KernelLaunch"))

/**
 * Finds every `@Kernel` function and translates it to WGSL. The result is used two ways:
 * - it is written to [outputDir] (`<fq.name>.wgsl/.json`). On the JVM that is next to the classes, so it ships
 *   in the jar; the Gradle plugin gives other platforms their own folder;
 * - on every platform, `kernelSource(::kernel)` calls are replaced by a `KernelSource` holding the shader;
 * - a kernel call inside a launch block (a lambda with a `KernelLaunch` receiver, like `gpu.dispatch(n) { saxpy(2f, x, y) }`)
 *   is replaced by calls that record the shader and the arguments in that `KernelLaunch`.
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

    /**
     * Replaces `kernelSource(::someKernel)` with `KernelSource("<fq name>", "<wgsl>", "<manifest json>")`, and
     * `someKernel(a, b)` in a launch block with `kernelLaunch.argument(0, a); kernelLaunch.argument(1, b);
     * kernelLaunch.kernel(KernelSource(...))`.
     */
    private fun embedKernelSources(
        file: IrFile,
        context: IrPluginContext,
        shaders: Map<IrSimpleFunction, CompiledShader>,
    ) {
        file.transformChildrenVoid(object : IrElementTransformerVoid() {
            // The KernelLaunch receiver of each lambda we're inside of (null for other lambdas), innermost last.
            private val launchReceivers = ArrayList<IrValueParameter?>()

            override fun visitFunctionExpression(expression: IrFunctionExpression): IrExpression {
                val receiver = expression.function.parameters.firstOrNull { it.kind == IrParameterKind.ExtensionReceiver }
                launchReceivers += receiver?.takeIf { it.type.classFqName == KERNEL_LAUNCH_CLASS.asSingleFqName() }
                try {
                    return super.visitFunctionExpression(expression)
                } finally {
                    launchReceivers.removeAt(launchReceivers.lastIndex)
                }
            }

            override fun visitCall(expression: IrCall): IrExpression {
                expression.transformChildrenVoid(this)
                val callee = expression.symbol.owner
                val launch = launchReceivers.lastOrNull()
                return when {
                    callee.kotlinFqName == KERNEL_SOURCE_FN -> embedKernelSource(expression, callee)
                    launch != null && callee.hasAnnotation(KERNEL_ANNOTATION) -> recordLaunch(expression, callee, launch)
                    else -> expression
                }
            }

            private fun embedKernelSource(expression: IrCall, callee: IrSimpleFunction): IrExpression {
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
                if (error != null || shader == null) return expression.also { report("Kernelin: $error", it) }
                val builder = DeclarationIrBuilder(context, callee.symbol, expression.startOffset, expression.endOffset)
                return builder.newKernelSource(shader, expression) ?: expression
            }

            private fun recordLaunch(expression: IrCall, kernel: IrSimpleFunction, receiver: IrValueParameter): IrExpression {
                val shader = shaders[kernel] ?: savedShader(kernel) ?: return expression.also {
                    // A kernel of this module with translation errors already has its own errors.
                    if (kernel.body == null) report("Kernelin: '${kernel.kotlinFqName}' must be a @Kernel from this module to launch it", it)
                }
                val launchClass = context.referenceClass(KERNEL_LAUNCH_CLASS)?.owner
                    ?: return expression.also { report("Kernelin: KernelLaunch class not found; add kernelin-runtime", it) }
                val params = kernel.parameters.filter { it.kind == IrParameterKind.Regular }
                val values = params.map { p ->
                    expression.arguments[p.indexInParameters] ?: return expression.also {
                        report("Kernelin: pass a value for every kernel parameter ('${p.name}' has none); default values can't be launched", it)
                    }
                }
                val lambda = receiver.parent as IrFunction
                val builder = DeclarationIrBuilder(context, lambda.symbol, expression.startOffset, expression.endOffset)
                val source = builder.newKernelSource(shader, expression) ?: return expression
                // Arguments first, in Kotlin order (the call already holds them after any reordering temporaries).
                return builder.irBlock(resultType = context.irBuiltIns.unitType) {
                    params.forEachIndexed { i, p ->
                        val argument = launchClass.function("argument") { it.parameters[2].type.classFqName == p.type.classFqName }
                        +irCall(argument).apply {
                            arguments[0] = irGet(receiver)
                            arguments[1] = irInt(i)
                            arguments[2] = values[i]
                        }
                    }
                    +irCall(launchClass.function("kernel") { true }).apply {
                        arguments[0] = irGet(receiver)
                        arguments[1] = source
                    }
                }
            }

            private fun IrClass.function(name: String, accept: (IrSimpleFunction) -> Boolean): IrSimpleFunction =
                declarations.filterIsInstance<IrSimpleFunction>().single { it.name.asString() == name && accept(it) }

            private fun IrBuilderWithScope.newKernelSource(shader: CompiledShader, at: IrElement): IrExpression? {
                val constructor = context.referenceClass(KERNEL_SOURCE_CLASS)?.owner?.declarations
                    ?.filterIsInstance<IrConstructor>()?.singleOrNull()
                    ?: return null.also { report("Kernelin: KernelSource class not found; add kernelin-runtime", at) }
                return irCallConstructor(constructor.symbol, emptyList()).apply {
                    arguments[0] = irString(shader.fqName)
                    arguments[1] = irString(shader.wgsl)
                    arguments[2] = irString(shader.manifestJson)
                }
            }

            private fun report(message: String, at: IrElement) =
                messages.report(CompilerMessageSeverity.ERROR, message, file.location(at))
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
