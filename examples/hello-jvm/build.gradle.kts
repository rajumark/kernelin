plugins {
    kotlin("jvm") version "2.4.21"
    id("io.github.kernelin")
    application
}

kotlin {
    jvmToolchain(21)
}

application {
    mainClass.set("hello.MainKt")
}
