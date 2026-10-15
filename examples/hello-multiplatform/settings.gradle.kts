rootProject.name = "hello-multiplatform"

pluginManagement {
    // Uses Kernelin straight from this repository. In your own project, drop the includeBuild lines.
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
