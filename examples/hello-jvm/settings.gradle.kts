rootProject.name = "hello-jvm"

pluginManagement {
    // Uses Kernelin straight from this repository. In your own project, drop the includeBuild lines;
    // the plugin and libraries then come from the repositories below.
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
