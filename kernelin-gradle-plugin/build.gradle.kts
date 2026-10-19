plugins {
    `java-gradle-plugin`
    alias(libs.plugins.vanniktech.maven.publish)
    alias(libs.plugins.kotlin.jvm)
}

kotlin {
    jvmToolchain(21)
    explicitApi()
    // Public API is recorded in api/ and checked on every build, so a change can't break users by accident.
    @OptIn(org.jetbrains.kotlin.gradle.dsl.abi.ExperimentalAbiValidation::class)
    abiValidation()
}

dependencies {
    compileOnly(libs.kotlin.gradle.plugin)
}

// The Gradle plugin and the compiler plugin share this id (KERNELIN_PLUGIN_ID in both).
val kernelinPluginId = "io.github.rajumark.kernelin"

gradlePlugin {
    plugins {
        create("kernelin") {
            id = kernelinPluginId
            displayName = "Kernelin"
            description = "Write GPU kernels in plain Kotlin: compiles @Kernel functions to WGSL compute shaders."
            implementationClass = "io.github.rajumark.kernelin.gradle.KernelinGradlePlugin"
        }
    }
}

// The plugin adds runtime/compiler artifacts of its own version, so bake the version (and the id) into the code.
val generateVersion by tasks.registering {
    val out = layout.buildDirectory.dir("generated/kernelin-version")
    val version = project.version.toString()
    inputs.property("version", version)
    inputs.property("pluginId", kernelinPluginId)
    outputs.dir(out)
    doLast {
        // Start clean so a renamed package never leaves a stale class behind in the jar.
        out.get().asFile.deleteRecursively()
        val file = out.get().file("io/github/rajumark/kernelin/gradle/KernelinVersion.kt").asFile
        file.parentFile.mkdirs()
        file.writeText(
            "package io.github.rajumark.kernelin.gradle\n\n" +
                "internal const val KERNELIN_VERSION = \"$version\"\n" +
                "internal const val KERNELIN_PLUGIN_ID = \"$kernelinPluginId\"\n",
        )
    }
}
kotlin.sourceSets.main { kotlin.srcDir(generateVersion) }

// Functional tests: Gradle TestKit builds small Kotlin JVM, Multiplatform and Android projects that apply this
// plugin. They resolve Kernelin from a local repository that the publish tasks below fill first.
val functionalTest by sourceSets.creating
gradlePlugin.testSourceSets(functionalTest)
dependencies {
    "functionalTestImplementation"(gradleTestKit())
    "functionalTestImplementation"(kotlin("test"))
    "functionalTestImplementation"(platform(libs.junit.bom))
    "functionalTestImplementation"(libs.junit.jupiter)
    "functionalTestRuntimeOnly"(libs.junit.platform.launcher)
}

val functionalTestTask = tasks.register<Test>("functionalTest") {
    description = "Builds sample projects with this plugin applied (Android ones need ANDROID_HOME)."
    group = "verification"
    testClassesDirs = functionalTest.output.classesDirs
    classpath = functionalTest.runtimeClasspath
    useJUnitPlatform()
    testLogging {
        events("failed", "skipped")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
    val modules = listOf(":kernelin-runtime", ":kernelin-compiler-plugin", ":kernelin-gradle-plugin")
    dependsOn(modules.map { "$it:publishAllPublicationsToFunctionalTestRepository" })
    val repo = rootProject.layout.buildDirectory.dir("functional-test-repo")
    val kotlinVersion = libs.versions.kotlin.get()
    // Reuse the normal Gradle user home, so Kotlin, AGP and their dependencies are downloaded only once.
    val userHome = gradle.gradleUserHomeDir
    inputs.dir(repo)
    inputs.property("kotlinVersion", kotlinVersion)
    systemProperty("kernelin.version", project.version.toString())
    systemProperty("kernelin.kotlinVersion", kotlinVersion)
    systemProperty("kernelin.agpVersion", libs.versions.agp.get())
    jvmArgumentProviders.add(CommandLineArgumentProvider {
        listOf("-Dkernelin.repo=${repo.get().asFile.absolutePath}", "-Dkernelin.testKitDir=${userHome.absolutePath}")
    })
    // Only fails or skips the Android tests; the result also depends on the installed Android SDK.
    inputs.property("androidHome", providers.environmentVariable("ANDROID_HOME").orElse(""))
}
tasks.check { dependsOn(functionalTestTask) }
