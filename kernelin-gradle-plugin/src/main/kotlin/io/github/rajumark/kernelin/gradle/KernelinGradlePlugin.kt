package io.github.rajumark.kernelin.gradle

import org.gradle.api.Project
import org.gradle.api.provider.Property
import org.gradle.api.provider.Provider
import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension
import org.jetbrains.kotlin.gradle.plugin.KotlinCompilation
import org.jetbrains.kotlin.gradle.plugin.InternalSubpluginOption
import org.jetbrains.kotlin.gradle.plugin.KotlinCompilerPluginSupportPlugin
import org.jetbrains.kotlin.gradle.plugin.KotlinPlatformType
import org.jetbrains.kotlin.gradle.plugin.SubpluginArtifact
import org.jetbrains.kotlin.gradle.plugin.SubpluginOption

/**
 * Settings for Kernelin, written as `kernelin { ... }` in a build script.
 */
public abstract class KernelinExtension {
    /**
     * Runs the Kernelin compiler plugin (default `true`). When `false`, `@Kernel` functions are not translated
     * to WGSL and `kernelSource()` is not replaced, but `kernelin-runtime` stays on the classpath, so kernels
     * still run on the CPU with `KernelinCpu`.
     */
    public abstract val enabled: Property<Boolean>
}

/**
 * `plugins { id("io.github.rajumark.kernelin") }` sets up Kernelin in a Kotlin project:
 * - runs the Kernelin compiler plugin on every Kotlin compilation;
 * - adds `kernelin-runtime` (`@Kernel`, `globalId()`, `kernelSource()`, `KernelinCpu`) to Kotlin JVM, Android
 *   (with or without the Kotlin Android plugin) and Multiplatform projects;
 * - in Multiplatform projects with a `wasmJs` target, adds `kernelin-webgpu` to `wasmJsMain`.
 */
public class KernelinGradlePlugin : KotlinCompilerPluginSupportPlugin {
    override fun apply(target: Project) {
        val runtime = "io.github.rajumark:kernelin-runtime:$KERNELIN_VERSION"
        val webgpu = "io.github.rajumark:kernelin-webgpu:$KERNELIN_VERSION"

        target.extensions.create("kernelin", KernelinExtension::class.java).apply {
            enabled.convention(true)
        }

        target.plugins.withId("org.jetbrains.kotlin.jvm") {
            target.dependencies.add("implementation", runtime)
        }
        target.plugins.withId("org.jetbrains.kotlin.multiplatform") {
            val kotlin = target.extensions.getByType(KotlinMultiplatformExtension::class.java)
            kotlin.sourceSets.named("commonMain") { it.dependencies { implementation(runtime) } }
            kotlin.sourceSets.matching { it.name == "wasmJsMain" }.configureEach {
                it.dependencies { implementation(webgpu) }
            }
        }

        // Android: the Kotlin Android plugin, or AGP 9's built-in Kotlin (just com.android.application/library).
        // A Multiplatform project already gets the runtime in commonMain (this also covers
        // com.android.kotlin.multiplatform.library), so the check waits until all plugins are applied.
        var androidDone = false
        for (id in ANDROID_PLUGIN_IDS) {
            target.plugins.withId(id) {
                if (androidDone) return@withId
                androidDone = true
                target.configurations.named("implementation") { implementation ->
                    implementation.dependencies.addAllLater(target.provider {
                        if (target.plugins.hasPlugin("org.jetbrains.kotlin.multiplatform")) emptyList()
                        else listOf(target.dependencies.create(runtime))
                    })
                }
            }
        }
    }

    override fun isApplicable(kotlinCompilation: KotlinCompilation<*>): Boolean = true

    override fun getCompilerPluginId(): String = KERNELIN_PLUGIN_ID

    override fun getPluginArtifact(): SubpluginArtifact =
        SubpluginArtifact(groupId = "io.github.rajumark", artifactId = "kernelin-compiler-plugin", version = KERNELIN_VERSION)

    override fun applyToCompilation(kotlinCompilation: KotlinCompilation<*>): Provider<List<SubpluginOption>> {
        val project = kotlinCompilation.target.project
        val extension = project.extensions.getByType(KernelinExtension::class.java)
        // The JVM writes shaders next to the classes. Other platforms get their own folder, so that an incremental
        // build that recompiles kernelSource(::kernel) without the kernel's file can reuse the kernel's shader.
        val shaderDir = if (kotlinCompilation.platformType in KLIB_PLATFORMS) {
            project.layout.buildDirectory.dir("kernelin/${kotlinCompilation.target.name}/${kotlinCompilation.name}")
                .also { dir -> kotlinCompilation.compileTaskProvider.configure { it.outputs.dir(dir) } }
        } else null
        return extension.enabled.map { enabled ->
            listOfNotNull(
                SubpluginOption("enabled", enabled.toString()),
                // Internal: an absolute path must not be part of the task's build cache key.
                shaderDir?.let { InternalSubpluginOption("outputDir", it.get().asFile.absolutePath) },
            )
        }
    }

    private companion object {
        val KLIB_PLATFORMS = setOf(KotlinPlatformType.js, KotlinPlatformType.wasm, KotlinPlatformType.native)

        val ANDROID_PLUGIN_IDS = listOf(
            "org.jetbrains.kotlin.android",
            "com.android.application",
            "com.android.library",
            "com.android.dynamic-feature",
            "com.android.test",
        )
    }
}
