import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    kotlin("multiplatform") version "2.4.21"
    // Adds kernelin-runtime to commonMain and kernelin-webgpu to wasmJsMain.
    id("io.github.kernelin")
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
}
