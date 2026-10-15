import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

kotlin {
    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser {
            testTask { enabled = false }
        }
    }

    sourceSets {
        wasmJsMain.dependencies {
            api(project(":kernelin-runtime"))
            api(libs.kotlinx.coroutines.core)
        }
    }
}
