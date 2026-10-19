// An Android app that benchmarks the sample kernels on the phone's GPU (native WebGPU, no browser) and on its CPU.
// Run on a USB-connected phone: scripts/bench-android-app.sh
plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "io.github.rajumark.kernelin.samples.android"
    compileSdk = 36
    defaultConfig {
        applicationId = "io.github.rajumark.kernelin.bench"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
    }
    buildTypes {
        // Benchmarks need optimized code; signed with the debug key so it installs without a keystore.
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    sourceSets {
        // The very same kernels as the JVM and browser benchmarks.
        getByName("main").kotlin.directories += "../samples/src/commonMain/kotlin"
    }
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    implementation(project(":kernelin-webgpu"))
    // Inside this repo the compiler plugin is wired directly; users apply the Gradle plugin instead.
    add("kotlinCompilerPluginClasspath", project(":kernelin-compiler-plugin"))
}
