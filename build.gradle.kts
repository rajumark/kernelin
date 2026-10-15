plugins {
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.kotlin.multiplatform) apply false
}

allprojects {
    group = "io.github.kernelin"
    version = "0.1.0-SNAPSHOT"
}
