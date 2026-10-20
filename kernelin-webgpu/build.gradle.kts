import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.android.kotlin.multiplatform.library)
    alias(libs.plugins.vanniktech.maven.publish)
}

kotlin {
    // Library API: everything public must be declared public on purpose, with an explicit type.
    explicitApi()
    // Public API is recorded in api/ and checked on every build, so a change can't break users by accident.
    @OptIn(org.jetbrains.kotlin.gradle.dsl.abi.ExperimentalAbiValidation::class)
    abiValidation()
    jvmToolchain(21)
    // Desktop JVM (macOS, Linux, Windows): native WebGPU through jWebGPU (wgpu over JNI), no browser needed.
    jvm()
    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser {
            testTask { enabled = false }
        }
        // Host tests run in Node against a small in-memory fake of WebGPU, so they need no GPU or browser.
        nodejs()
    }
    // Android: native WebGPU through AndroidX WebGPU (Dawn on Vulkan), no browser needed.
    // GPU tests run on a device or emulator: gradle :kernelin-webgpu:connectedAndroidDeviceTest
    android {
        namespace = "io.github.rajumark.kernelin.webgpu"
        compileSdk = 36
        // AndroidX WebGPU needs API 26.
        minSdk = 26
        withDeviceTest {
            instrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        }
    }

    sourceSets {
        commonMain.dependencies {
            api(project(":kernelin-runtime"))
            api(libs.kotlinx.coroutines.core)
        }
        androidMain.dependencies {
            implementation(libs.androidx.webgpu)
        }
        jvmMain.dependencies {
            implementation(libs.jwebgpu)
            // The native library for each desktop: jWebGPU loads the one that fits the running OS and CPU.
            runtimeOnly(libs.jwebgpu.wgpu.linux.x64)
            runtimeOnly(libs.jwebgpu.wgpu.mac.arm64)
            runtimeOnly(libs.jwebgpu.wgpu.mac.x64)
            runtimeOnly(libs.jwebgpu.wgpu.windows.x64)
        }
        jvmTest.dependencies {
            implementation(kotlin("test"))
            implementation(libs.kotlinx.coroutines.test)
            implementation(project.dependencies.platform(libs.junit.bom))
            runtimeOnly(libs.junit.platform.launcher)
        }
        wasmJsTest.dependencies {
            implementation(kotlin("test"))
            implementation(libs.kotlinx.coroutines.test)
        }
        getByName("androidDeviceTest").dependencies {
            implementation(kotlin("test"))
            implementation(libs.kotlinx.coroutines.test)
            implementation(libs.androidx.test.runner)
        }
    }
}

dependencies {
    // The host tests launch real @Kernel functions, so their compilation runs the compiler plugin.
    add("kotlinCompilerPluginClasspathWasmJsTest", project(":kernelin-compiler-plugin"))
    // The JVM GPU tests launch real @Kernel functions too.
    add("kotlinCompilerPluginClasspathJvmTest", project(":kernelin-compiler-plugin"))
    // The device tests write real @Kernel functions, so their compilation runs the compiler plugin.
    add("kotlinCompilerPluginClasspathAndroidDeviceTest", project(":kernelin-compiler-plugin"))
}

tasks.named<Test>("jvmTest") {
    useJUnitPlatform()
}
