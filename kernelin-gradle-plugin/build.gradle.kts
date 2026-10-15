plugins {
    `java-gradle-plugin`
    `maven-publish`
    alias(libs.plugins.kotlin.jvm)
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    compileOnly(libs.kotlin.gradle.plugin)
}

gradlePlugin {
    plugins {
        create("kernelin") {
            id = "io.github.kernelin"
            displayName = "Kernelin"
            description = "Write GPU kernels in plain Kotlin: compiles @Kernel functions to WGSL compute shaders."
            implementationClass = "io.github.kernelin.gradle.KernelinGradlePlugin"
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
        val file = out.get().file("io/github/kernelin/gradle/KernelinVersion.kt").asFile
        file.parentFile.mkdirs()
        file.writeText("package io.github.kernelin.gradle\n\ninternal const val KERNELIN_VERSION = \"$version\"\n")
    }
}
kotlin.sourceSets.main { kotlin.srcDir(generateVersion) }
