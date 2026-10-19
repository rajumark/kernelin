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
    // Kotlin/JS: kernels run on the CPU with KernelinCpu (the WebGPU host is Wasm only for now).
    js {
        browser {
            // Common tests run on Node.js; browser tests would need a headless Chrome.
            testTask { enabled = false }
        }
        nodejs()
    }
    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser {
            // Common tests already run on the JVM; browser tests would need a headless Chrome + npm.
            testTask { enabled = false }
        }
    }
    // Kotlin/Native: every tier 1 and 2 target, plus Windows. Apple targets build only on macOS;
    // Gradle skips them elsewhere and the api/ dump keeps their ABI.
    macosArm64()
    iosArm64()
    iosSimulatorArm64()
    iosX64()
    tvosArm64()
    tvosSimulatorArm64()
    watchosArm32()
    watchosArm64()
    watchosSimulatorArm64()
    linuxX64()
    linuxArm64()
    mingwX64()

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

// tvOS and watchOS share the native code tested on macOS and the iOS simulator; their simulators are often
// not installed (Xcode downloads them separately), so don't make the build depend on them.
tasks.matching { it.name == "tvosSimulatorArm64Test" || it.name == "watchosSimulatorArm64Test" }.configureEach {
    enabled = false
}
