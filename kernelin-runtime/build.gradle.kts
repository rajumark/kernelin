import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

kotlin {
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
