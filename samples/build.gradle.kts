plugins {
    alias(libs.plugins.kotlin.jvm)
    application
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    implementation(project(":kernelin-runtime"))
    // Inside this repo the compiler plugin is wired directly; users will apply the Gradle plugin instead.
    kotlinCompilerPluginClasspath(project(":kernelin-compiler-plugin"))

    testImplementation(kotlin("test"))
    testImplementation(platform(libs.junit.bom))
    testRuntimeOnly(libs.junit.platform.launcher)
}

application {
    mainClass.set("io.github.kernelin.samples.BenchmarkKt")
    applicationDefaultJvmArgs = listOf("-Xmx6g")
}

tasks.test {
    useJUnitPlatform()
}

// Copies the shader Kernelin generated for the live demo (and its Kotlin source) into docs/demo,
// which GitHub Pages serves. Run after changing mandelbrotView: gradle :samples:exportDemo
tasks.register<Copy>("exportDemo") {
    group = "kernelin"
    description = "Exports the generated demo shader to docs/demo/kernels."
    dependsOn(tasks.named("compileKotlin"))
    from(layout.buildDirectory.dir("classes/kotlin/main/kernelin")) {
        include("io.github.kernelin.samples.mandelbrotView.*")
    }
    from("src/main/kotlin/io/github/kernelin/samples/Kernels.kt")
    into(rootProject.layout.projectDirectory.dir("docs/demo/kernels"))
}

// ./gradlew :samples:bench -Pout=<dir> writes inputs, expected outputs, shaders and CPU timings for the GPU runner.
tasks.register<JavaExec>("bench") {
    group = "kernelin"
    description = "Runs the CPU benchmarks and exports data for the WebGPU runner."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("io.github.kernelin.samples.BenchmarkKt")
    maxHeapSize = "6g"
    val out = providers.gradleProperty("out").orElse(layout.buildDirectory.dir("kernelin-bench").map { it.asFile.path })
    argumentProviders.add(CommandLineArgumentProvider { listOf(out.get()) })
}
