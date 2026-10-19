import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    kotlin("multiplatform") version "2.4.21"
    // Adds kernelin-runtime to commonMain and kernelin-webgpu to wasmJsMain.
    id("io.github.rajumark.kernelin")
}

kotlin {
    jvmToolchain(21)
    jvm {
        mainRun {
            mainClass.set("hello.MainKt")
        }
    }
    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
        binaries.executable()
    }
    // Kotlin/JS and Kotlin/Native: the kernel runs on the CPU, and kernelSource() still gives its WGSL.
    js {
        nodejs()
        binaries.executable()
    }
    listOf(macosArm64(), linuxX64()).forEach {
        it.binaries.executable { entryPoint = "hello.main" }
    }
}
