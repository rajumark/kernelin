package io.github.kernelin.compiler

import org.jetbrains.kotlin.backend.common.extensions.IrGenerationExtension
import org.jetbrains.kotlin.backend.common.extensions.IrPluginContext
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageLocation
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.cli.common.messages.MessageCollector
import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.declarations.IrFile
import org.jetbrains.kotlin.ir.declarations.IrModuleFragment
import org.jetbrains.kotlin.ir.declarations.IrSimpleFunction
import org.jetbrains.kotlin.ir.util.hasAnnotation
import org.jetbrains.kotlin.ir.util.kotlinFqName
import org.jetbrains.kotlin.ir.visitors.IrVisitorVoid
import org.jetbrains.kotlin.ir.visitors.acceptChildrenVoid
import org.jetbrains.kotlin.name.FqName
import java.io.File

internal val KERNEL_ANNOTATION = FqName("io.github.kernelin.Kernel")

/** Finds every `@Kernel` function, translates it to WGSL and writes the result next to the classes. */
class KernelinIrGenerationExtension(
    private val messages: MessageCollector,
    private val outputDir: File?,
) : IrGenerationExtension {

    override fun generate(moduleFragment: IrModuleFragment, pluginContext: IrPluginContext) {
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
            for (kernel in kernels) translate(file, kernel)
        }
    }

    private fun translate(file: IrFile, kernel: IrSimpleFunction) {
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
            return
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
    }

    private fun IrFile.location(element: IrElement): CompilerMessageLocation? {
        if (element.startOffset < 0) return null
        val line = fileEntry.getLineNumber(element.startOffset)
        val column = fileEntry.getColumnNumber(element.startOffset)
        return CompilerMessageLocation.create(fileEntry.name, line + 1, column + 1, null)
    }
}
