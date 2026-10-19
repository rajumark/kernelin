package io.github.rajumark.kernelin.gradle

import org.gradle.testkit.runner.BuildResult
import org.gradle.testkit.runner.GradleRunner
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.File
import kotlin.test.assertContains
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Builds small real projects that apply `io.github.rajumark.kernelin`, one per kind of Kotlin project.
 * Kernelin comes from the local repository the build publishes to (see build.gradle.kts).
 * Android tests need an Android SDK and are skipped when ANDROID_HOME is not set.
 */
class KernelinGradlePluginFunctionalTest {
    @TempDir
    lateinit var dir: File

    private val version = System.getProperty("kernelin.version")
    private val kotlinVersion = System.getProperty("kernelin.kotlinVersion")
    private val agpVersion = System.getProperty("kernelin.agpVersion")
    private val repo = File(System.getProperty("kernelin.repo")).toURI()

    private val kernel = """
        package demo

        import io.github.rajumark.kernelin.Kernel
        import io.github.rajumark.kernelin.globalId

        @Kernel
        fun squares(out: IntArray) {
            val i = globalId()
            out[i] = i * i
        }
    """.trimIndent()

    // Uses the runtime (KernelinCpu) and the compiler plugin (kernelSource is replaced at compile time).
    private val main = """
        package demo

        import io.github.rajumark.kernelin.KernelinCpu
        import io.github.rajumark.kernelin.kernelSource

        fun runSquares(): String {
            val out = IntArray(4)
            KernelinCpu.dispatch(out.size) { squares(out) }
            return "squares=${'$'}{out.toList()} wgsl=${'$'}{kernelSource(::squares).wgsl.contains("@compute")}"
        }

        fun main() = println(runSquares())
    """.trimIndent()

    @Test
    fun `Kotlin JVM project gets the runtime and the compiler plugin`() {
        project(
            """
            plugins {
                kotlin("jvm") version "$kotlinVersion"
                id("io.github.rajumark.kernelin") version "$version"
                application
            }
            kotlin { jvmToolchain(21) }
            application { mainClass.set("demo.MainKt") }
            """,
        )
        source("src/main/kotlin")
        val result = gradle("run")
        assertContains(result.output, "squares=[0, 1, 4, 9] wgsl=true")
        assertTrue(dir.resolve("build/classes/kotlin/main/kernelin/demo.squares.wgsl").isFile)
    }

    @Test
    fun `kernelin enabled = false turns the compiler plugin off`() {
        project(
            """
            plugins {
                kotlin("jvm") version "$kotlinVersion"
                id("io.github.rajumark.kernelin") version "$version"
            }
            kotlin { jvmToolchain(21) }
            kernelin { enabled = false }
            """,
        )
        source("src/main/kotlin")
        gradle("compileKotlin")
        assertFalse(dir.resolve("build/classes/kotlin/main/kernelin").exists(), "no shaders when disabled")
    }

    @Test
    fun `Multiplatform project compiles kernels for JVM, JS and Native`() {
        project(
            """
            plugins {
                kotlin("multiplatform") version "$kotlinVersion"
                id("io.github.rajumark.kernelin") version "$version"
            }
            kotlin {
                jvmToolchain(21)
                jvm()
                js { nodejs() }
                linuxX64()
            }
            """,
        )
        source("src/commonMain/kotlin")
        gradle("compileKotlinJvm", "compileKotlinJs", "compileKotlinLinuxX64")
        assertKernelTranslated()
    }

    @Test
    fun `Android app with the Kotlin Android plugin gets the runtime`() {
        assumeAndroidSdk()
        // AGP 9 has Kotlin built in; the Kotlin Android plugin still works when that is turned off.
        dir.resolve("gradle.properties").appendText("android.builtInKotlin=false\nandroid.newDsl=false\n")
        androidProject(
            """
            plugins {
                id("com.android.application") version "$agpVersion"
                kotlin("android") version "$kotlinVersion"
                id("io.github.rajumark.kernelin") version "$version"
            }
            ${androidBlock()}
            kotlin { jvmToolchain(21) }
            """,
        )
        source("src/main/kotlin")
        androidManifest()
        gradle("assembleDebug")
        assertKernelTranslated()
    }

    @Test
    fun `Android app with AGP built-in Kotlin gets the runtime`() {
        assumeAndroidSdk()
        androidProject(
            """
            plugins {
                id("com.android.application") version "$agpVersion"
                id("io.github.rajumark.kernelin") version "$version"
            }
            ${androidBlock()}
            """,
        )
        source("src/main/kotlin")
        androidManifest()
        gradle("assembleDebug")
        assertKernelTranslated()
    }

    @Test
    fun `Android library with AGP built-in Kotlin gets the runtime`() {
        assumeAndroidSdk()
        androidProject(
            """
            plugins {
                id("com.android.library") version "$agpVersion"
                id("io.github.rajumark.kernelin") version "$version"
            }
            ${androidBlock(application = false)}
            """,
        )
        source("src/main/kotlin")
        gradle("assembleDebug")
        assertKernelTranslated()
    }

    @Test
    fun `Android Kotlin Multiplatform library gets the runtime in commonMain`() {
        assumeAndroidSdk()
        androidProject(
            """
            plugins {
                kotlin("multiplatform") version "$kotlinVersion"
                id("com.android.kotlin.multiplatform.library") version "$agpVersion"
                id("io.github.rajumark.kernelin") version "$version"
            }
            kotlin {
                jvmToolchain(21)
                android {
                    namespace = "demo.lib"
                    compileSdk = 36
                    minSdk = 24
                }
            }
            """,
        )
        source("src/commonMain/kotlin")
        gradle("compileAndroidMain")
        assertKernelTranslated()
    }

    // The compiler plugin writes the shader next to the JVM classes (Android hides its log message).
    // The build itself only passes when kernelin-runtime is on the classpath.
    private fun assertKernelTranslated() = assertTrue(
        dir.resolve("build").walk().any { it.name == "demo.squares.wgsl" },
        "the Kernelin compiler plugin did not translate demo.squares",
    )

    private fun assumeAndroidSdk() {
        val sdk = System.getenv("ANDROID_HOME")?.takeIf { it.isNotBlank() && File(it).isDirectory }
        assumeTrue(sdk != null, "ANDROID_HOME is not set: skipping the Android test")
        dir.resolve("local.properties").writeText("sdk.dir=${File(sdk!!).absolutePath.replace("\\", "/")}\n")
    }

    private fun androidManifest() =
        dir.resolve("src/main").apply { mkdirs() }.resolve("AndroidManifest.xml")
            .writeText("<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\" />\n")

    private fun androidBlock(application: Boolean = true) = """
        android {
            namespace = "demo.app"
            compileSdk = 36
            defaultConfig {
                minSdk = 24
                ${if (application) "targetSdk = 36" else ""}
            }
            compileOptions {
                sourceCompatibility = JavaVersion.VERSION_21
                targetCompatibility = JavaVersion.VERSION_21
            }
        }
    """.trimIndent()

    // AGP built-in Kotlin brings its own Kotlin version; pin the one Kernelin's compiler plugin is built for.
    private fun androidProject(build: String) = project(
        build,
        buildscript = """
            buildscript {
                repositories { mavenCentral() }
                dependencies { classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlinVersion") }
            }
        """,
    )

    private fun project(build: String, buildscript: String = "") {
        dir.resolve("settings.gradle.kts").writeText(
            """
            pluginManagement {
                repositories {
                    maven("$repo")
                    google()
                    gradlePluginPortal()
                    mavenCentral()
                }
            }
            dependencyResolutionManagement {
                repositories {
                    maven("$repo")
                    google()
                    mavenCentral()
                }
            }
            rootProject.name = "demo"
            """.trimIndent(),
        )
        dir.resolve("build.gradle.kts").writeText(buildscript.trimIndent() + "\n" + build.trimIndent())
        dir.resolve("gradle.properties").appendText(
            "org.gradle.jvmargs=-Xmx2g\nandroid.useAndroidX=true\n" +
                // Same as the repository's gradle.properties: Homebrew's JDK 21 (ignored where it doesn't exist).
                "org.gradle.java.installations.paths=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home\n",
        )
    }

    private fun source(root: String) {
        val pkg = dir.resolve("$root/demo").apply { mkdirs() }
        pkg.resolve("Kernels.kt").writeText(kernel)
        pkg.resolve("Main.kt").writeText(main)
    }

    private fun gradle(vararg tasks: String): BuildResult = GradleRunner.create()
        .withProjectDir(dir)
        .withTestKitDir(File(System.getProperty("kernelin.testKitDir")))
        .withArguments(*tasks, "--stacktrace")
        .forwardOutput()
        .build()
}
