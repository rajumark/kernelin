package io.github.kernelin.gradle

import org.gradle.api.Project
import org.gradle.api.provider.Provider
import org.jetbrains.kotlin.gradle.plugin.KotlinCompilation
import org.jetbrains.kotlin.gradle.plugin.KotlinCompilerPluginSupportPlugin
import org.jetbrains.kotlin.gradle.plugin.SubpluginArtifact
import org.jetbrains.kotlin.gradle.plugin.SubpluginOption

/**
 * `plugins { id("io.github.kernelin") }` adds the Kernelin compiler plugin and runtime to a Kotlin project.
 *
 * Not used inside this repository yet (the samples wire the compiler plugin directly);
 * it becomes the public entry point once Kernelin is published.
 */
class KernelinGradlePlugin : KotlinCompilerPluginSupportPlugin {
    override fun apply(target: Project) {
        target.plugins.withId("org.jetbrains.kotlin.jvm") {
            target.dependencies.add("implementation", "io.github.kernelin:kernelin-runtime:$VERSION")
        }
    }

    override fun isApplicable(kotlinCompilation: KotlinCompilation<*>): Boolean = true

    override fun getCompilerPluginId(): String = "io.github.kernelin"

    override fun getPluginArtifact(): SubpluginArtifact =
        SubpluginArtifact(groupId = "io.github.kernelin", artifactId = "kernelin-compiler-plugin", version = VERSION)

    override fun applyToCompilation(kotlinCompilation: KotlinCompilation<*>): Provider<List<SubpluginOption>> =
        kotlinCompilation.target.project.provider { emptyList() }

    private companion object {
        const val VERSION = "0.1.0-SNAPSHOT"
    }
}
