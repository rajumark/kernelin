plugins {
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.kotlin.multiplatform) apply false
}

allprojects {
    group = "io.github.kernelin"
    version = "0.1.0-SNAPSHOT"
}

// Shared POM metadata for every published module (needed for Maven Central later).
subprojects {
    plugins.withId("maven-publish") {
        extensions.configure<PublishingExtension> {
            publications.withType<MavenPublication>().configureEach {
                pom {
                    name.set(project.name)
                    description.set("Kernelin: write GPU kernels in plain Kotlin, compiled to WGSL by a K2 compiler plugin.")
                    url.set("https://github.com/rajumark/kernelin")
                    licenses {
                        license {
                            name.set("Apache License 2.0")
                            url.set("https://www.apache.org/licenses/LICENSE-2.0")
                        }
                    }
                    developers {
                        developer {
                            id.set("rajumark")
                            name.set("rajumark")
                            url.set("https://github.com/rajumark")
                        }
                    }
                    scm {
                        url.set("https://github.com/rajumark/kernelin")
                        connection.set("scm:git:https://github.com/rajumark/kernelin.git")
                    }
                }
            }
        }
    }
}
