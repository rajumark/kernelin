rootProject.name = "hello-jvm"

pluginManagement {
    // Uses Kernelin straight from this repository. In your own project, drop both includeBuild lines and
    // write id("io.github.rajumark.kernelin") version "0.4.0": it then comes from Maven Central.
    includeBuild("../..")
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}

includeBuild("../..")
