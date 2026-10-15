rootProject.name = "kernelin"

pluginManagement {
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

include(
    "kernelin-runtime",
    "kernelin-compiler-plugin",
    "kernelin-gradle-plugin",
    "kernelin-webgpu",
    "samples",
)
