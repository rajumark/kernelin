package io.github.rajumark.kernelin

/** What a GPU host needs to know to run a kernel: its buffers, scalar parameters and workgroup size. */
class KernelManifest(
    val kernel: String,
    val entryPoint: String,
    val workgroupSize: Int,
    /** Kernel parameters in Kotlin order. */
    val params: List<Param>,
) {
    class Param(
        val name: String,
        /** `"buffer"` or `"scalar"`. */
        val kind: String,
        /** Element/scalar type: `f32`, `i32`, or `bool` (a scalar sent as a u32 0/1 in the uniform). */
        val type: String,
        /** Bind group binding, for buffers. Binding 0 is always the uniform with the scalars. */
        val binding: Int = -1,
        /** `"read"` or `"read_write"`, for buffers. */
        val access: String = "",
    ) {
        val isBuffer get() = kind == "buffer"
        val isWritten get() = access == "read_write"
    }

    val buffers: List<Param> get() = params.filter { it.isBuffer }
    val scalars: List<Param> get() = params.filter { !it.isBuffer }

    /** Uniform size in bytes: `kernelin_size: u32` plus one 4-byte field per scalar, padded to 16. */
    val uniformBytes: Int get() = (1 + scalars.size) * 4 + 15 and 15.inv()

    companion object {
        fun parse(json: String): KernelManifest {
            @Suppress("UNCHECKED_CAST")
            val root = Json(json).parse() as Map<String, Any?>
            @Suppress("UNCHECKED_CAST")
            val params = (root["params"] as List<Map<String, Any?>>).map {
                Param(
                    name = it["name"] as String,
                    kind = it["kind"] as String,
                    type = it["type"] as String,
                    binding = (it["binding"] as Double?)?.toInt() ?: -1,
                    access = it["access"] as String? ?: "",
                )
            }
            return KernelManifest(
                kernel = root["kernel"] as String,
                entryPoint = root["entryPoint"] as String,
                workgroupSize = (root["workgroupSize"] as Double).toInt(),
                params = params,
            )
        }
    }
}

/** Tiny JSON reader for manifests (objects, arrays, strings, numbers, booleans, null). */
private class Json(private val s: String) {
    private var i = 0

    fun parse(): Any? = value().also { skip(); require(i == s.length) { "Trailing data in JSON at $i" } }

    private fun skip() {
        while (i < s.length && s[i].isWhitespace()) i++
    }

    private fun value(): Any? {
        skip()
        return when (val c = s[i]) {
            '{' -> obj()
            '[' -> arr()
            '"' -> str()
            't' -> word("true", true)
            'f' -> word("false", false)
            'n' -> word("null", null)
            else -> if (c == '-' || c.isDigit()) num() else error("Unexpected '$c' in JSON at $i")
        }
    }

    private fun obj(): Map<String, Any?> {
        val map = LinkedHashMap<String, Any?>()
        i++
        skip()
        if (s[i] == '}') return map.also { i++ }
        while (true) {
            skip()
            val key = str()
            skip()
            require(s[i++] == ':') { "Expected ':' in JSON at ${i - 1}" }
            map[key] = value()
            skip()
            when (s[i++]) {
                ',' -> continue
                '}' -> return map
                else -> error("Expected ',' or '}' in JSON at ${i - 1}")
            }
        }
    }

    private fun arr(): List<Any?> {
        val list = ArrayList<Any?>()
        i++
        skip()
        if (s[i] == ']') return list.also { i++ }
        while (true) {
            list += value()
            skip()
            when (s[i++]) {
                ',' -> continue
                ']' -> return list
                else -> error("Expected ',' or ']' in JSON at ${i - 1}")
            }
        }
    }

    private fun str(): String {
        require(s[i] == '"') { "Expected string in JSON at $i" }
        val sb = StringBuilder()
        i++
        while (s[i] != '"') {
            if (s[i] == '\\') {
                i++
                when (val e = s[i]) {
                    'n' -> sb.append('\n')
                    't' -> sb.append('\t')
                    'u' -> { sb.append(s.substring(i + 1, i + 5).toInt(16).toChar()); i += 4 }
                    else -> sb.append(e)
                }
            } else {
                sb.append(s[i])
            }
            i++
        }
        i++
        return sb.toString()
    }

    private fun num(): Double {
        val start = i
        while (i < s.length && (s[i].isDigit() || s[i] in "+-.eE")) i++
        return s.substring(start, i).toDouble()
    }

    private fun word(w: String, v: Any?): Any? {
        require(s.startsWith(w, i)) { "Unexpected token in JSON at $i" }
        i += w.length
        return v
    }
}
