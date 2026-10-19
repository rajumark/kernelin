rootProject.name = "kernelin"

pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
        google()
    }
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
        google()
    }
}

include(
    "kernelin-runtime",
    "kernelin-compiler-plugin",
    "kernelin-gradle-plugin",
    "kernelin-webgpu",
    "samples",
    "samples-android",
)
