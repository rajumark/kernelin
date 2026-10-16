package io.github.rajumark.kernelin.compiler

import kotlin.random.Random

/**
 * Generates random, valid, terminating Kotlin kernels for differential testing (CPU vs GPU).
 *
 * The generated code only uses operations whose results are exactly defined on both sides: Int arithmetic
 * (wrapping), bit operations with masked shift counts, division by a never-zero value, conversions through
 * Float only for values that are exactly representable, bounded loops, labeled jumps, `repeat`, `when`,
 * `in` ranges and helper functions. Any CPU/GPU difference is therefore a Kernelin bug.
 */
class KernelFuzzer(private val seed: Long) {
    private val r = Random(seed)
    private val out = StringBuilder()
    private var indent = 1
    private var counter = 0

    /** Variables in scope, innermost scope last. Each entry: name to mutable. */
    private val scopes = ArrayDeque<MutableList<Pair<String, Boolean>>>()

    /** Enclosing loops: a label for `for`/`while` loops ("L3"), or null for a `repeat` lambda. */
    private val loops = ArrayDeque<String?>()
    private val helpers = mutableListOf<Pair<String, Int>>()

    /** Helpers only see their own parameters (no kernel arrays, no globalId). */
    private var inHelper = false

    fun generate(name: String): String {
        val helperCode = StringBuilder()
        repeat(r.nextInt(0, 3)) { h ->
            val helperName = "${name}_h$h"
            val arity = r.nextInt(1, 3)
            val params = (0 until arity).map { "p$it" }
            scopes.addLast(params.map { it to false }.toMutableList())
            inHelper = true
            helperCode.appendLine("fun $helperName(${params.joinToString { "$it: Int" }}): Int = ${intExpr(2)}")
            inHelper = false
            scopes.removeLast()
            helpers += helperName to arity
        }
        scopes.addLast(mutableListOf("i" to false, "s" to false))
        repeat(r.nextInt(3, 8)) { statement(3) }
        val results = allVars().map { it.first }
        val combine = results.joinToString(" xor ") { "($it * ${r.nextInt(1, 1000)})" }
        line("outI[i] = $combine")
        scopes.removeLast()
        return buildString {
            append(helperCode)
            appendLine("// fuzz seed $seed")
            appendLine("@Kernel")
            appendLine("fun $name(a: IntArray, outI: IntArray, s: Int) {")
            appendLine("    val i = globalId()")
            append(out)
            appendLine("}")
        }
    }

    // ------------------------------------------------------------------ statements

    private fun line(text: String) {
        repeat(indent) { out.append("    ") }
        out.appendLine(text)
    }

    private inline fun block(header: String, body: () -> Unit) {
        line("$header {")
        indent++
        scopes.addLast(mutableListOf())
        body()
        scopes.removeLast()
        indent--
        line("}")
    }

    private fun allVars() = scopes.flatten()

    private fun mutableVars() = allVars().filter { it.second }.map { it.first }

    private fun fresh(prefix: String) = "$prefix${counter++}"

    private fun statement(depth: Int) {
        val choice = r.nextInt(if (depth <= 0) 4 else 11)
        when (choice) {
            0, 1 -> {
                val v = fresh("v")
                line("var $v = ${intExpr(3)}")
                scopes.last() += v to true
            }
            2, 3 -> {
                val target = mutableVars().randomOrNull(r)
                if (target == null) {
                    val v = fresh("v")
                    line("var $v = ${intExpr(2)}")
                    scopes.last() += v to true
                } else {
                    when (r.nextInt(4)) {
                        0 -> line("$target = ${intExpr(3)}")
                        1 -> line("$target += ${intExpr(2)}")
                        2 -> line("$target++")
                        else -> line("$target = $target xor ${intExpr(2)}")
                    }
                }
            }
            4 -> {
                block("if (${boolExpr(2)})") { repeat(r.nextInt(1, 3)) { statement(depth - 1) } }
                if (r.nextBoolean()) block("else") { repeat(r.nextInt(1, 3)) { statement(depth - 1) } }
            }
            5 -> forLoop(depth)
            6 -> whileLoop(depth)
            7 -> repeatLoop(depth)
            8 -> jump()
            9 -> {
                val target = mutableVars().randomOrNull(r) ?: return statement(depth)
                line("when (${intExpr(2)}) {")
                indent++
                line("in ${r.nextInt(-50, 0)}..${r.nextInt(0, 50)} -> $target += ${intExpr(1)}")
                line("${r.nextInt(-5, 5)}, ${r.nextInt(-5, 5)} -> $target = ${intExpr(1)}")
                line("else -> $target -= 1")
                indent--
                line("}")
            }
            else -> line("outI[i] = ${intExpr(2)}")
        }
    }

    private fun loopBody(depth: Int) = repeat(r.nextInt(1, 4)) { statement(depth - 1) }

    private fun forLoop(depth: Int) {
        val label = fresh("L")
        val k = fresh("k")
        val range = when (r.nextInt(5)) {
            0 -> "0 until ${r.nextInt(0, 5)}"
            1 -> "0..${r.nextInt(0, 4)}"
            2 -> "${r.nextInt(2, 6)} downTo 0"
            3 -> "0 until ${r.nextInt(2, 9)} step ${r.nextInt(1, 4)}"
            else -> "0..<${r.nextInt(1, 5)}"
        }
        loops.addLast(label)
        block("$label@ for ($k in $range)") {
            scopes.last() += k to false
            loopBody(depth)
        }
        loops.removeLast()
    }

    private fun whileLoop(depth: Int) {
        val w = fresh("w")
        val label = fresh("L")
        line("var $w = 0")
        scopes.last() += w to false
        loops.addLast(label)
        // The counter moves first, so a continue can never skip it: every loop terminates.
        block("$label@ while ($w < ${r.nextInt(0, 5)})") {
            line("$w++")
            loopBody(depth)
        }
        loops.removeLast()
    }

    private fun repeatLoop(depth: Int) {
        val p = fresh("t")
        loops.addLast(null)
        line("repeat(${r.nextInt(0, 5)}) { $p ->")
        indent++
        scopes.addLast(mutableListOf(p to false))
        loopBody(depth)
        scopes.removeLast()
        indent--
        line("}")
        loops.removeLast()
    }

    private fun jump() {
        if (loops.isEmpty()) return
        val cond = boolExpr(2)
        val labeled = loops.filterNotNull()
        val jump = when {
            // Directly inside a repeat lambda: move on to its next iteration.
            loops.last() == null && (labeled.isEmpty() || r.nextBoolean()) -> "return@repeat"
            labeled.isEmpty() -> return
            else -> (if (r.nextBoolean()) "break" else "continue") + "@" + labeled.random(r)
        }
        line("if ($cond) $jump")
    }

    // ------------------------------------------------------------------ expressions

    private fun intLeaf(): String = when (r.nextInt(7)) {
        0 -> listOf("0", "1", "-1", "2", "7", "Int.MAX_VALUE", "Int.MIN_VALUE", "65535", "-1024").random(r)
        1 -> r.nextInt(-100, 100).toString()
        2 -> if (inHelper) r.nextInt(-9, 9).toString() else "a[${index(allVars().random(r).first)}]"
        else -> allVars().random(r).first
    }

    /** A safe index into `a` (its size is a power of two, so masking keeps it in range). */
    private fun index(e: String) = "(($e) and (a.size - 1))"

    private fun intExpr(depth: Int): String {
        if (depth <= 0 || r.nextInt(10) < 3) return intLeaf()
        val x = { intExpr(depth - 1) }
        return when (r.nextInt(16)) {
            0 -> "(${x()} + ${x()})"
            1 -> "(${x()} - ${x()})"
            2 -> "(${x()} * ${x()})"
            3 -> "(${x()} / (${x()} or 1))"
            4 -> "(${x()} % (${x()} or 1))"
            5 -> "(${x()} and ${x()})"
            6 -> "(${x()} or ${x()})"
            7 -> "(${x()} xor ${x()})"
            8 -> "(${x()} ${listOf("shl", "shr", "ushr").random(r)} (${x()} and 31))"
            9 -> if (r.nextBoolean()) "(${x()}).inv()" else "(-(${x()}))"
            10 -> "(if (${boolExpr(depth - 1)}) ${x()} else ${x()})"
            11 -> "(when { ${boolExpr(depth - 1)} -> ${x()}; ${boolExpr(depth - 1)} -> ${x()}; else -> ${x()} })"
            12 -> listOf("min", "max").random(r).let { "$it(${x()}, ${x()})" }
            13 -> if (r.nextBoolean()) "abs(${x()})" else "${x()}.coerceIn(-1000, 1000)"
            // Through Float only with exactly representable values, so both sides must agree.
            14 -> "((${x()} and 65535).toFloat() / 4f).toInt()"
            else -> helpers.randomOrNull(r)?.let { (h, n) -> "$h(${(0 until n).joinToString { x() }})" } ?: x()
        }
    }

    private fun boolExpr(depth: Int): String {
        val x = { intExpr(depth - 1) }
        if (depth <= 0) return "(${intLeaf()} ${listOf("<", ">", "==", "!=").random(r)} ${intLeaf()})"
        return when (r.nextInt(7)) {
            0, 1 -> "(${x()} ${listOf("<", "<=", ">", ">=", "==", "!=").random(r)} ${x()})"
            2 -> "(${boolExpr(depth - 1)} && ${boolExpr(depth - 1)})"
            3 -> "(${boolExpr(depth - 1)} || ${boolExpr(depth - 1)})"
            4 -> "!${boolExpr(depth - 1)}"
            5 -> "(${x()} in ${r.nextInt(-100, 0)}..${r.nextInt(0, 100)})"
            else -> "(${x()} !in ${r.nextInt(-50, 0)} until ${r.nextInt(0, 50)})"
        }
    }
}
