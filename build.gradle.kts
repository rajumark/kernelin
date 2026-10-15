plugins {
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.vanniktech.maven.publish) apply false
}

// Coordinates come from gradle.properties (GROUP, VERSION_NAME), the same values the publish plugin uses.
allprojects {
    group = providers.gradleProperty("GROUP").get()
    version = providers.gradleProperty("VERSION_NAME").get()
}

// Published modules: upload to the Maven Central Portal and sign. Credentials and the signing key live in
// ~/.gradle/gradle.properties (never in this repo). Nothing uploads unless a *MavenCentral* task runs.
subprojects {
    plugins.withId("com.vanniktech.maven.publish") {
        extensions.configure<com.vanniktech.maven.publish.MavenPublishBaseExtension> {
            publishToMavenCentral()
            if (providers.gradleProperty("signingInMemoryKey").isPresent) signAllPublications()
        }
    }
}
