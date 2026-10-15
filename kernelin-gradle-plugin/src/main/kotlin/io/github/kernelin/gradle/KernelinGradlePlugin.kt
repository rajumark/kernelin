package io.github.kernelin.gradle

import org.gradle.api.Project
import org.gradle.api.provider.Provider
import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension
import org.jetbrains.kotlin.gradle.plugin.KotlinCompilation
import org.jetbrains.kotlin.gradle.plugin.KotlinCompilerPluginSupportPlugin
import org.jetbrains.kotlin.gradle.plugin.SubpluginArtifact
import org.jetbrains.kotlin.gradle.plugin.SubpluginOption

/**
 * `plugins { id("io.github.kernelin") }` sets up Kernelin in a Kotlin project:
 * - runs the Kernelin compiler plugin on every Kotlin compilation;
 * - adds `kernelin-runtime` (`@Kernel`, `globalId()`, `kernelSource()`, `KernelinCpu`);
 * - in Multiplatform projects with a `wasmJs` target, adds `kernelin-webgpu` to `wasmJsMain`.
 */
class KernelinGradlePlugin : KotlinCompilerPluginSupportPlugin {
    override fun apply(target: Project) {
        val runtime = "io.github.kernelin:kernelin-runtime:$KERNELIN_VERSION"
        val webgpu = "io.github.kernelin:kernelin-webgpu:$KERNELIN_VERSION"

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
    }

    override fun isApplicable(kotlinCompilation: KotlinCompilation<*>): Boolean = true

    override fun getCompilerPluginId(): String = "io.github.kernelin"

    override fun getPluginArtifact(): SubpluginArtifact =
        SubpluginArtifact(groupId = "io.github.kernelin", artifactId = "kernelin-compiler-plugin", version = KERNELIN_VERSION)

    override fun applyToCompilation(kotlinCompilation: KotlinCompilation<*>): Provider<List<SubpluginOption>> =
        kotlinCompilation.target.project.provider { emptyList() }
}
