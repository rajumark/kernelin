plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.vanniktech.maven.publish)
    alias(libs.plugins.kover)
}

kotlin {
    jvmToolchain(21)
    compilerOptions {
        optIn.add("org.jetbrains.kotlin.compiler.plugin.ExperimentalCompilerApi")
    }
}

dependencies {
    compileOnly(libs.kotlin.compiler.embeddable)

    testImplementation(libs.kotlin.compiler.embeddable)
    testImplementation(project(":kernelin-runtime"))
    testImplementation(kotlin("test"))
    testImplementation(platform(libs.junit.bom))
    testRuntimeOnly(libs.junit.platform.launcher)
}

tasks.test {
    useJUnitPlatform()
    // Show the full CPU vs GPU difference in CI logs, not just "FAILED".
    testLogging {
        events("failed")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
    // Tests run the real Kotlin compiler with the packaged plugin jar, like a user's build would.
    val pluginJar = tasks.jar.flatMap { it.archiveFile }
    inputs.file(pluginJar)
    // GPU differential tests run the generated WGSL through this Deno script (skipped without Deno/GPU).
    val gpuRunner = layout.projectDirectory.file("src/test/js/gpu-run.mjs")
    inputs.file(gpuRunner)
    jvmArgumentProviders.add(CommandLineArgumentProvider {
        listOf(
            "-Dkernelin.pluginJar=${pluginJar.get().asFile.absolutePath}",
            "-Dkernelin.gpuRunner=${gpuRunner.asFile.absolutePath}",
        ) + listOf("kernelin.fuzz.count", "kernelin.fuzz.seed").mapNotNull { key ->
            providers.gradleProperty(key).orNull?.let { "-D$key=$it" }
        }
    })
    // Differences between CPU and GPU must never be served from the build cache without running.
    outputs.upToDateWhen { false }
    outputs.cacheIf { false }
}
