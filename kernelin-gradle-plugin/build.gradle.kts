plugins {
    `java-gradle-plugin`
    alias(libs.plugins.vanniktech.maven.publish)
    alias(libs.plugins.kotlin.jvm)
}

kotlin {
    jvmToolchain(21)
    explicitApi()
    // Public API is recorded in api/ and checked on every build, so a change can't break users by accident.
    @OptIn(org.jetbrains.kotlin.gradle.dsl.abi.ExperimentalAbiValidation::class)
    abiValidation()
}

dependencies {
    compileOnly(libs.kotlin.gradle.plugin)
}

gradlePlugin {
    plugins {
        create("kernelin") {
            id = "io.github.rajumark.kernelin"
            displayName = "Kernelin"
            description = "Write GPU kernels in plain Kotlin: compiles @Kernel functions to WGSL compute shaders."
            implementationClass = "io.github.rajumark.kernelin.gradle.KernelinGradlePlugin"
        }
    }
}

// The plugin adds runtime/compiler artifacts of its own version, so bake the version into the code.
val generateVersion by tasks.registering {
    val out = layout.buildDirectory.dir("generated/kernelin-version")
    val version = project.version.toString()
    inputs.property("version", version)
    outputs.dir(out)
    doLast {
        // Start clean so a renamed package never leaves a stale class behind in the jar.
        out.get().asFile.deleteRecursively()
        val file = out.get().file("io/github/rajumark/kernelin/gradle/KernelinVersion.kt").asFile
        file.parentFile.mkdirs()
        file.writeText("package io.github.rajumark.kernelin.gradle\n\ninternal const val KERNELIN_VERSION = \"$version\"\n")
    }
}
kotlin.sourceSets.main { kotlin.srcDir(generateVersion) }
