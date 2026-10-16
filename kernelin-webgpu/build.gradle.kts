import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.vanniktech.maven.publish)
}

kotlin {
    // Library API: everything public must be declared public on purpose, with an explicit type.
    explicitApi()
    // Public API is recorded in api/ and checked on every build, so a change can't break users by accident.
    @OptIn(org.jetbrains.kotlin.gradle.dsl.abi.ExperimentalAbiValidation::class)
    abiValidation()
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
