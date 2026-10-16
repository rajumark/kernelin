import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

kotlin {
    jvmToolchain(21)
    jvm()
    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        outputModuleName.set("kernelin-demo")
        browser {
            testTask { enabled = false }
        }
        binaries.executable()
    }

    sourceSets {
        // The kernels live in commonMain, so the same Kotlin compiles to the JVM, to Wasm and (via Kernelin) to WGSL.
        commonMain.dependencies {
            implementation(project(":kernelin-runtime"))
        }
        wasmJsMain.dependencies {
            implementation(project(":kernelin-webgpu"))
        }
        jvmTest.dependencies {
            implementation(kotlin("test"))
            implementation(project.dependencies.platform(libs.junit.bom))
            runtimeOnly(libs.junit.platform.launcher)
        }
    }
}

dependencies {
    // Inside this repo the compiler plugin is wired directly; users will apply the Gradle plugin instead.
    kotlinCompilerPluginClasspath(project(":kernelin-compiler-plugin"))
}

tasks.withType<Test> {
    useJUnitPlatform()
}

val jvmMain = kotlin.jvm().compilations["main"]

// gradle :samples:bench -Pout=<dir> writes inputs, expected outputs, shaders and CPU timings for the GPU runner.
tasks.register<JavaExec>("bench") {
    group = "kernelin"
    description = "Runs the CPU benchmarks and exports data for the WebGPU runner."
    dependsOn(jvmMain.compileTaskProvider)
    classpath = files(jvmMain.output.allOutputs, jvmMain.runtimeDependencyFiles)
    mainClass.set("io.github.rajumark.kernelin.samples.BenchmarkKt")
    maxHeapSize = "6g"
    val out = providers.gradleProperty("out").orElse(layout.buildDirectory.dir("kernelin-bench").map { it.asFile.path })
    argumentProviders.add(CommandLineArgumentProvider { listOf(out.get()) })
}

// Prints the benchmark's classpath, for scripts/bench-android-cpu.sh to turn into an Android dex.
tasks.register("printBenchClasspath") {
    group = "kernelin"
    description = "Prints the classpath of the CPU benchmark."
    dependsOn(jvmMain.compileTaskProvider)
    val classpath = files(jvmMain.output.classesDirs, jvmMain.runtimeDependencyFiles)
    doLast { println(classpath.files.filter { it.exists() }.joinToString("\n")) }
}

// Builds the Kotlin/Wasm demo and copies it, plus the generated shader and the kernel source,
// into docs/demo, which GitHub Pages serves. Run after changing the kernels: gradle :samples:exportDemo
tasks.register<Copy>("exportDemo") {
    group = "kernelin"
    description = "Exports the Kotlin/Wasm demo and its generated shader to docs/demo."
    val wasmDist = tasks.named("compileProductionExecutableKotlinWasmJsOptimize")
    dependsOn(wasmDist, jvmMain.compileTaskProvider)
    into(rootProject.layout.projectDirectory.dir("docs/demo"))
    from(layout.buildDirectory.dir("compileSync/wasmJs/main/productionExecutable/optimized")) {
        include("*.mjs", "*.wasm")
        into("wasm")
    }
    from(layout.buildDirectory.dir("classes/kotlin/jvm/main/kernelin")) {
        include("io.github.rajumark.kernelin.samples.mandelbrotView.*")
        into("kernels")
    }
    from("src/commonMain/kotlin/io/github/rajumark/kernelin/samples/Kernels.kt") {
        into("kernels")
    }
}
