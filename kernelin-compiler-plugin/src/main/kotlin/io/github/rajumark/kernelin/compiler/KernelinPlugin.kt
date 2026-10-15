package io.github.rajumark.kernelin.compiler

import org.jetbrains.kotlin.backend.common.extensions.IrGenerationExtension
import org.jetbrains.kotlin.compiler.plugin.AbstractCliOption
import org.jetbrains.kotlin.compiler.plugin.CliOption
import org.jetbrains.kotlin.compiler.plugin.CliOptionProcessingException
import org.jetbrains.kotlin.compiler.plugin.CommandLineProcessor
import org.jetbrains.kotlin.compiler.plugin.CompilerPluginRegistrar
import org.jetbrains.kotlin.config.CompilerConfiguration
import org.jetbrains.kotlin.config.CompilerConfigurationKey
import org.jetbrains.kotlin.config.JVMConfigurationKeys
import org.jetbrains.kotlin.config.MessageCollectorAccess
import org.jetbrains.kotlin.config.messageCollector
import java.io.File

const val KERNELIN_PLUGIN_ID = "io.github.rajumark.kernelin"

object KernelinConfigurationKeys {
    val OUTPUT_DIR = CompilerConfigurationKey<String>("Kernelin output directory")
}

class KernelinCommandLineProcessor : CommandLineProcessor {
    override val pluginId: String = KERNELIN_PLUGIN_ID

    override val pluginOptions: Collection<AbstractCliOption> = listOf(
        CliOption(
            optionName = "outputDir",
            valueDescription = "<path>",
            description = "Where generated .wgsl and .json files are written. Defaults to the class output directory.",
            required = false,
        ),
    )

    override fun processOption(option: AbstractCliOption, value: String, configuration: CompilerConfiguration) {
        when (option.optionName) {
            "outputDir" -> configuration.put(KernelinConfigurationKeys.OUTPUT_DIR, value)
            else -> throw CliOptionProcessingException("Unknown Kernelin option: ${option.optionName}")
        }
    }
}

class KernelinCompilerPluginRegistrar : CompilerPluginRegistrar() {
    override val pluginId: String = KERNELIN_PLUGIN_ID
    override val supportsK2: Boolean = true

    // Kernelin reports errors with exact source locations, which needs the collector itself.
    @OptIn(MessageCollectorAccess::class)
    override fun ExtensionStorage.registerExtensions(configuration: CompilerConfiguration) {
        val messages = configuration.messageCollector
        // Shaders go under kernelin/ so they ship inside the jar next to the classes.
        val outputDir = (configuration.get(KernelinConfigurationKeys.OUTPUT_DIR)?.let(::File)
            ?: configuration.get(JVMConfigurationKeys.OUTPUT_DIRECTORY))
            ?.resolve("kernelin")
        IrGenerationExtension.registerExtension(KernelinIrGenerationExtension(messages, outputDir))
    }
}
