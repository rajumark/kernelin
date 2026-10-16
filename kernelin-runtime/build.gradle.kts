import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.vanniktech.maven.publish)
    alias(libs.plugins.kover)
}

kotlin {
    // Library API: everything public must be declared public on purpose, with an explicit type.
    explicitApi()
    // Public API is recorded in api/ and checked on every build, so a change can't break users by accident.
    @OptIn(org.jetbrains.kotlin.gradle.dsl.abi.ExperimentalAbiValidation::class)
    abiValidation()
    jvmToolchain(21)
    jvm()
    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser {
            // Common tests already run on the JVM; browser tests would need a headless Chrome + npm.
            testTask { enabled = false }
        }
    }

    sourceSets {
        commonTest.dependencies {
            implementation(kotlin("test"))
        }
        jvmTest.dependencies {
            implementation(project.dependencies.platform(libs.junit.bom))
            runtimeOnly(libs.junit.platform.launcher)
        }
    }
}

tasks.withType<Test> {
    useJUnitPlatform()
}
