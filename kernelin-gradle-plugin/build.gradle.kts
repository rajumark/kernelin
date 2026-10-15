plugins {
    `java-gradle-plugin`
    alias(libs.plugins.kotlin.jvm)
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    compileOnly(libs.kotlin.gradle.plugin.api)
}

gradlePlugin {
    plugins {
        create("kernelin") {
            id = "io.github.kernelin"
            implementationClass = "io.github.kernelin.gradle.KernelinGradlePlugin"
        }
    }
}
