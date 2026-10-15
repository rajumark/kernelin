plugins {
    alias(libs.plugins.kotlin.jvm)
    `maven-publish`
}

kotlin {
    jvmToolchain(21)
    compilerOptions {
        optIn.add("org.jetbrains.kotlin.compiler.plugin.ExperimentalCompilerApi")
    }
}

dependencies {
    compileOnly(libs.kotlin.compiler.embeddable)

    testImplementation(libs.kotlin.compiler.embeddable)
    testImplementation(project(":kernelin-runtime"))
    testImplementation(kotlin("test"))
    testImplementation(platform(libs.junit.bom))
    testRuntimeOnly(libs.junit.platform.launcher)
}

tasks.test {
    useJUnitPlatform()
    // Tests run the real Kotlin compiler with the packaged plugin jar, like a user's build would.
    val pluginJar = tasks.jar.flatMap { it.archiveFile }
    inputs.file(pluginJar)
    jvmArgumentProviders.add(CommandLineArgumentProvider {
        listOf("-Dkernelin.pluginJar=${pluginJar.get().asFile.absolutePath}")
    })
}

java {
    withSourcesJar()
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])
        }
    }
}
