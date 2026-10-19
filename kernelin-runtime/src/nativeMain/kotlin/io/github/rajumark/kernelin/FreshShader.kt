package io.github.rajumark.kernelin

/** Native binaries have no resources; the embedded shader is always rebuilt with the module. */
internal actual fun freshShader(name: String): Pair<String, String>? = null
