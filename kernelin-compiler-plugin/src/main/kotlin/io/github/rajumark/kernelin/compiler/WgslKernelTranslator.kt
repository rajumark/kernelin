package io.github.rajumark.kernelin.compiler

import org.jetbrains.kotlin.ir.IrElement
import org.jetbrains.kotlin.ir.IrStatement
import org.jetbrains.kotlin.ir.declarations.IrClass
import org.jetbrains.kotlin.ir.declarations.IrDeclarationOrigin
import org.jetbrains.kotlin.ir.declarations.IrFile
import org.jetbrains.kotlin.ir.declarations.IrPackageFragment
import org.jetbrains.kotlin.ir.declarations.IrParameterKind
import org.jetbrains.kotlin.ir.declarations.IrSimpleFunction
import org.jetbrains.kotlin.ir.declarations.IrValueDeclaration
import org.jetbrains.kotlin.ir.declarations.IrValueParameter
import org.jetbrains.kotlin.ir.declarations.IrVariable
import org.jetbrains.kotlin.ir.expressions.IrBlock
import org.jetbrains.kotlin.ir.expressions.IrBlockBody
import org.jetbrains.kotlin.ir.expressions.IrBreak
import org.jetbrains.kotlin.ir.expressions.IrBreakContinue
import org.jetbrains.kotlin.ir.expressions.IrCall
import org.jetbrains.kotlin.ir.expressions.IrConst
import org.jetbrains.kotlin.ir.expressions.IrContainerExpression
import org.jetbrains.kotlin.ir.expressions.IrContinue
import org.jetbrains.kotlin.ir.expressions.IrDoWhileLoop
import org.jetbrains.kotlin.ir.expressions.IrElseBranch
import org.jetbrains.kotlin.ir.expressions.IrExpression
import org.jetbrains.kotlin.ir.expressions.IrExpressionBody
import org.jetbrains.kotlin.ir.expressions.IrFunctionExpression
import org.jetbrains.kotlin.ir.expressions.IrGetValue
import org.jetbrains.kotlin.ir.expressions.IrLoop
import org.jetbrains.kotlin.ir.expressions.IrReturn
import org.jetbrains.kotlin.ir.expressions.IrSetValue
import org.jetbrains.kotlin.ir.expressions.IrStatementOrigin
import org.jetbrains.kotlin.ir.expressions.IrStringConcatenation
import org.jetbrains.kotlin.ir.expressions.IrThrow
import org.jetbrains.kotlin.ir.expressions.IrTry
import org.jetbrains.kotlin.ir.expressions.IrTypeOperator
import org.jetbrains.kotlin.ir.expressions.IrTypeOperatorCall
import org.jetbrains.kotlin.ir.expressions.IrWhen
import org.jetbrains.kotlin.ir.expressions.IrWhileLoop
import org.jetbrains.kotlin.ir.types.IrType
import org.jetbrains.kotlin.ir.types.classFqName
import org.jetbrains.kotlin.ir.types.isMarkedNullable
import org.jetbrains.kotlin.ir.util.fileOrNull
import org.jetbrains.kotlin.ir.util.hasAnnotation
import org.jetbrains.kotlin.ir.util.kotlinFqName
import kotlin.math.pow
import kotlin.math.sign

internal const val WORKGROUP_SIZE = 64

internal enum class WType(val wgsl: String, val kotlin: String) {
    I32("i32", "Int"),
    F32("f32", "Float"),
    BOOL("bool", "Boolean"),
    UNIT("", "Unit"),
    ;

    /** Type of this value inside the uniform buffer: bool isn't host-shareable, so it's a u32 (0 or 1). */
    val uniformWgsl: String get() = if (this == BOOL) "u32" else wgsl
}

internal class KernelError(val element: IrElement, val message: String, val file: IrFile?)

/** A kernel parameter: an array ([BufferParam]) or a single value ([ScalarParam]). */
internal sealed interface KernelParam {
    val name: String
}

internal class BufferParam(override val name: String, val elementType: WType, val binding: Int) : KernelParam {
    var written = false
}

internal class ScalarParam(override val name: String, val type: WType) : KernelParam

internal class TranslationResult(val shader: WgslShader?, val errors: List<KernelError>)

/** A translated kernel: WGSL source plus the information a GPU host needs to run it. */
internal class WgslShader(
    val wgsl: String,
    /** Kernel parameters in Kotlin order. */
    val params: List<KernelParam>,
) {
    val buffers get() = params.filterIsInstance<BufferParam>()
    val scalars get() = params.filterIsInstance<ScalarParam>()

    fun manifestJson(fqName: String): String = buildString {
        appendLine("{")
        appendLine("  \"kernel\": ${json(fqName)},")
        appendLine("  \"entryPoint\": \"main\",")
        appendLine("  \"workgroupSize\": $WORKGROUP_SIZE,")
        appendLine("  \"uniform\": { \"binding\": 0, \"fields\": [")
        val fields = listOf("{ \"name\": \"kernelin_size\", \"type\": \"u32\" }") +
            scalars.map { "{ \"name\": ${json(it.name)}, \"type\": ${json(it.type.uniformWgsl)} }" }
        appendLine(fields.joinToString(",\n") { "    $it" })
        appendLine("  ] },")
        appendLine("  \"params\": [")
        appendLine(params.joinToString(",\n") { p ->
            when (p) {
                is BufferParam -> "    { \"name\": ${json(p.name)}, \"kind\": \"buffer\", \"type\": ${json(p.elementType.wgsl)}, " +
                    "\"binding\": ${p.binding}, \"access\": \"${if (p.written) "read_write" else "read"}\" }"
                is ScalarParam -> "    { \"name\": ${json(p.name)}, \"kind\": \"scalar\", \"type\": ${json(p.type.wgsl)} }"
            }
        })
        appendLine("  ]")
        appendLine("}")
    }
}

/** [s] as a JSON string literal. Kernel names in backticks may hold quotes, backslashes and other characters. */
internal fun json(s: String): String = buildString {
    append('"')
    for (c in s) {
        when {
            c == '"' -> append("\\\"")
            c == '\\' -> append("\\\\")
            c == '\n' -> append("\\n")
            c == '\t' -> append("\\t")
            c < ' ' || c == '\u2028' || c == '\u2029' -> append("\\u").append(c.code.toString(16).padStart(4, '0'))
            else -> append(c)
        }
    }
    append('"')
}

/**
 * Translates one `@Kernel` function (plus any plain helper functions it calls) to a WGSL compute shader.
 *
 * Works on IR right after FIR-to-IR conversion, before any lowering, so loops, `&&`, `++` and
 * compound assignments still have their recognisable shapes (marked by statement origins).
 */
internal class WgslKernelTranslator(private val kernel: IrSimpleFunction) {
    private val errors = mutableListOf<KernelError>()
    private val names = NameAllocator()
    private val buffers = LinkedHashMap<IrValueParameter, BufferParam>()
    private val scalars = LinkedHashMap<IrValueParameter, ScalarParam>()
    private val params = mutableListOf<KernelParam>()
    private val helpers = LinkedHashMap<Any, Helper>()
    private val helperStack = ArrayDeque<IrSimpleFunction>()
    private var usesFloatToInt = false
    private var usesFloatChecks = false
    private var usesPow = false

    /** Kernelin's own WGSL functions this shader calls (see [RUNTIME_FUNCTIONS]), in a fixed order. */
    private val runtime = sortedSetOf<String>()

    /** Marks runtime function [name] (and the NaN check it builds on) as used and returns [name]. */
    private fun use(name: String): String {
        runtime += name
        usesFloatChecks = true
        usesSpecialFloats = true
        return name
    }

    /**
     * Counts emitted side effects: array writes, and calls to helpers that write arrays. Comparing it before and
     * after translating an expression tells whether that expression has side effects.
     */
    private var sideEffects = 0

    /** [writes] is true when the helper (or a helper it calls) writes to an array. */
    private class Helper(val name: String, var code: String = "", var writes: Boolean = false)

    /** A helper that takes arrays is generated once per distinct set of kernel buffers passed to it. */
    private data class HelperKey(val function: IrSimpleFunction, val arrays: List<BufferParam>)

    /** A translated expression. [const] is its value when it's a compile-time constant (Int, Float or Boolean). */
    private data class Expr(val code: String, val type: WType, val const: Any? = null)

    private var usesSpecialFloats = false

    /**
     * WGSL literal for a value computed with Kotlin's rules. WGSL evaluates all-constant expressions at shader
     * creation and rejects overflow, division by zero and non-finite floats there, while Kotlin wraps or produces
     * Infinity/NaN, so Kernelin folds constants itself and only emits the result.
     */
    private fun lit(v: Any): Expr = when (v) {
        is Int -> Expr(if (v == Int.MIN_VALUE) "(-2147483647i - 1i)" else "${v}i", WType.I32, v)
        is Float -> if (v.isFinite()) {
            Expr(floatLiteral(v), WType.F32, v)
        } else {
            // Not a constant to WGSL (it reads a private variable), so it may be Infinity or NaN.
            usesSpecialFloats = true
            Expr("bitcast<f32>(${java.lang.Float.floatToRawIntBits(v).toUInt()}u | kernelin_zero)", WType.F32, v)
        }
        is Boolean -> Expr(v.toString(), WType.BOOL, v)
        else -> error("unexpected constant $v")
    }

    /** Kotlin's result for operator [name] on constants, or null when it can't be folded. */
    private fun fold(name: String, r: Any, args: List<Any>): Any? {
        val b = args.firstOrNull()
        fun num(x: Any?) = x as? Number
        return when (r) {
            is Int -> when (name) {
                "plus" -> when (b) { is Int -> r + b; is Float -> r + b; else -> null }
                "minus" -> when (b) { is Int -> r - b; is Float -> r - b; else -> null }
                "times" -> when (b) { is Int -> r * b; is Float -> r * b; else -> null }
                "div" -> when (b) { is Int -> if (b == 0) null else r / b; is Float -> r / b; else -> null }
                "rem" -> when (b) { is Int -> if (b == 0) null else r % b; is Float -> r % b; else -> null }
                "and" -> (b as? Int)?.let { r and it }
                "or" -> (b as? Int)?.let { r or it }
                "xor" -> (b as? Int)?.let { r xor it }
                "shl" -> (b as? Int)?.let { r shl it }
                "shr" -> (b as? Int)?.let { r shr it }
                "ushr" -> (b as? Int)?.let { r ushr it }
                "inv" -> r.inv()
                "unaryMinus" -> -r
                "unaryPlus" -> r
                "inc" -> r + 1
                "dec" -> r - 1
                "toInt" -> r
                "toFloat" -> r.toFloat()
                "compareTo" -> when (b) { is Int -> r.compareTo(b); is Float -> r.compareTo(b); else -> null }
                "equals" -> r == b
                else -> null
            }
            is Float -> when (name) {
                "plus" -> num(b)?.let { r + it.toFloat() }
                "minus" -> num(b)?.let { r - it.toFloat() }
                "times" -> num(b)?.let { r * it.toFloat() }
                "div" -> num(b)?.let { r / it.toFloat() }
                "rem" -> num(b)?.let { r % it.toFloat() }
                "unaryMinus" -> -r
                "unaryPlus" -> r
                "inc" -> r + 1f
                "dec" -> r - 1f
                "toInt" -> r.toInt()
                "toFloat" -> r
                "compareTo" -> num(b)?.let { r.compareTo(it.toFloat()) }
                "equals" -> r == b
                else -> null
            }
            is Boolean -> when (name) {
                "not" -> !r
                "and" -> (b as? Boolean)?.let { r and it }
                "or" -> (b as? Boolean)?.let { r or it }
                "xor" -> (b as? Boolean)?.let { r xor it }
                "equals" -> r == b
                else -> null
            }
            else -> null
        }
    }

    fun translate(): TranslationResult {
        checkKernelSignature()
        val body = if (errors.isEmpty()) BodyTranslator(kernel, isKernel = true).translate() else ""
        if (errors.isNotEmpty()) return TranslationResult(null, errors.distinctBy { it.element to it.message })

        val wgsl = buildString {
            appendLine("// Generated by Kernelin from ${kernel.kotlinFqName} - do not edit.")
            appendLine()
            appendLine("struct KernelinParams {")
            appendLine("  kernelin_size: u32,")
            for (s in scalars.values) appendLine("  ${s.name}: ${s.type.uniformWgsl},")
            appendLine("}")
            appendLine()
            appendLine("@group(0) @binding(0) var<uniform> kernelin_params: KernelinParams;")
            for (b in buffers.values) {
                val access = if (b.written) "read_write" else "read"
                appendLine("@group(0) @binding(${b.binding}) var<storage, $access> ${b.name}: array<${b.elementType.wgsl}>;")
            }
            appendLine()
            appendLine("var<private> kernelin_id: u32;")
            if (usesSpecialFloats) {
                // Always 0. Reading it keeps Infinity/NaN constants out of WGSL's compile-time evaluation.
                appendLine("var<private> kernelin_zero: u32;")
            }
            if (usesFloatToInt || usesFloatChecks) {
                // Checked on the bits: GPU compilers may assume no NaNs and drop a plain `x != x`.
                appendLine()
                appendLine("fn kernelin_is_nan(x: f32) -> bool {")
                appendLine("  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;")
                appendLine("}")
                appendLine()
                appendLine("fn kernelin_is_inf(x: f32) -> bool {")
                appendLine("  return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u;")
                appendLine("}")
            }
            if (usesFloatToInt) {
                // Kotlin's Float.toInt(): NaN -> 0, out of range -> Int.MIN_VALUE / MAX_VALUE. Plain i32() clamps the
                // top to 2^31 - 128, and Vulkan drivers leave out-of-range conversions undefined, so handle both ends.
                appendLine()
                appendLine("fn kernelin_f2i(x: f32) -> i32 {")
                appendLine("  if (kernelin_is_nan(x)) {")
                appendLine("    return 0i;")
                appendLine("  }")
                appendLine("  if (x >= 2147483648.0f) {")
                appendLine("    return 2147483647i;")
                appendLine("  }")
                appendLine("  if (x <= -2147483648.0f) {")
                appendLine("    return i32(-2147483647) - 1i;")
                appendLine("  }")
                appendLine("  return i32(x);")
                appendLine("}")
            }
            if (usesPow) {
                // Kotlin's Float.pow (Java's Math.pow). WGSL's pow is undefined for x < 0 and for x == 0, y <= 0,
                // so only |x| > 0 goes to it and the sign and the special cases are handled here.
                appendLine()
                appendLine("fn kernelin_pow(x: f32, y: f32) -> f32 {")
                appendLine("  if (y == 0.0f) {")
                appendLine("    return 1.0f;")
                appendLine("  }")
                appendLine("  if (kernelin_is_nan(x) || kernelin_is_nan(y)) {")
                appendLine("    return bitcast<f32>(0x7fc00000u | kernelin_zero);")
                appendLine("  }")
                appendLine("  var r: f32;")
                appendLine("  if (x == 0.0f) {")
                appendLine("    r = select(0.0f, bitcast<f32>(0x7f800000u | kernelin_zero), y < 0.0f);")
                appendLine("  } else {")
                appendLine("    r = pow(abs(x), y);")
                appendLine("  }")
                appendLine("  if (x < 0.0f && trunc(y) != y) {")
                appendLine("    return bitcast<f32>(0x7fc00000u | kernelin_zero);")
                appendLine("  }")
                appendLine("  // A negative base (or -0) with an odd integer exponent keeps its sign; floats of 2^24 and more are even.")
                appendLine("  if ((bitcast<u32>(x) & 0x80000000u) != 0u && fract(abs(y) * 0.5f) == 0.5f) {")
                appendLine("    r = -r;")
                appendLine("  }")
                appendLine("  return r;")
                appendLine("}")
            }
            for (name in runtime) {
                appendLine()
                append(RUNTIME_FUNCTIONS.getValue(name).trimIndent())
                appendLine()
            }
            for (h in helpers.values) {
                appendLine()
                append(h.code)
            }
            appendLine()
            appendLine("@compute @workgroup_size($WORKGROUP_SIZE)")
            appendLine("fn main(@builtin(global_invocation_id) kernelin_gid: vec3<u32>, @builtin(num_workgroups) kernelin_nwg: vec3<u32>) {")
            appendLine("  kernelin_id = kernelin_gid.x + kernelin_gid.y * kernelin_nwg.x * ${WORKGROUP_SIZE}u;")
            appendLine("  if (kernelin_id >= kernelin_params.kernelin_size) {")
            appendLine("    return;")
            appendLine("  }")
            append(body)
            appendLine("}")
        }
        return TranslationResult(WgslShader(wgsl, params.toList()), emptyList())
    }

    private fun error(element: IrElement, message: String, function: IrSimpleFunction = kernel) {
        errors += KernelError(element, message, function.fileOrNull)
    }

    private fun checkKernelSignature() {
        if (scalarType(kernel.returnType) != WType.UNIT) {
            error(kernel, "a @Kernel function must return Unit; write results into an array parameter instead")
        }
        if (kernel.typeParameters.isNotEmpty()) error(kernel, "a @Kernel function can't have type parameters")
        if (kernel.isSuspend) error(kernel, "a @Kernel function can't be suspend")
        var binding = 1
        for (p in kernel.parameters) {
            if (p.kind != IrParameterKind.Regular) {
                error(p, "a @Kernel function must be a top-level function without receivers")
                continue
            }
            val name = names.allocate(p.name.asString())
            val element = arrayElementType(p.type)
            val scalar = scalarType(p.type)
            when {
                element != null -> BufferParam(name, element, binding++).also { buffers[p] = it; params += it }
                scalar == WType.I32 || scalar == WType.F32 || scalar == WType.BOOL ->
                    ScalarParam(name, scalar).also { scalars[p] = it; params += it }
                else -> error(p, "parameter '${p.name}' has type ${p.type.describe()}; kernels accept FloatArray, IntArray, Int, Float and Boolean${p.type.suggestion()}")
            }
        }
    }

    /** Translates the body of the kernel or of one helper function. */
    private inner class BodyTranslator(private val function: IrSimpleFunction, private val isKernel: Boolean) {
        private val out = StringBuilder()
        private var indent = 1
        private val locals = HashMap<IrValueDeclaration, Expr>()
        private val arrayAliases = HashMap<IrValueDeclaration, BufferParam>()
        private val loops = ArrayDeque<LoopEntry>()

        /**
         * One loop being emitted. [key] is the IrLoop (or the lambda of a `repeat`). WGSL has no labels, so a
         * `break@outer` / `continue@outer` from a nested loop sets a flag on the target and breaks out level by level.
         */
        private inner class LoopEntry(val key: Any, val declPos: Int, val declIndent: Int) {
            var breakFlag: String? = null
            var continueFlag: String? = null
            /** Jumps that leave this loop for an outer one: (target, isBreak). */
            val jumpsOut = LinkedHashSet<Pair<LoopEntry, Boolean>>()
        }

        private inline fun loop(key: Any, emit: () -> Unit) {
            val entry = LoopEntry(key, out.length, indent)
            loops.addLast(entry)
            emit()
            loops.removeLast()
            val decls = listOfNotNull(entry.breakFlag, entry.continueFlag)
                .joinToString("") { "  ".repeat(entry.declIndent) + "var $it: bool = false;\n" }
            if (decls.isNotEmpty()) out.insert(entry.declPos, decls)
            val parent = loops.lastOrNull()
            for ((target, isBreak) in entry.jumpsOut) {
                val flag = if (isBreak) target.breakFlag!! else target.continueFlag!!
                when {
                    target !== parent || isBreak -> line("if ($flag) { break; }")
                    else -> line("if ($flag) { $flag = false; continue; }")
                }
            }
        }

        private fun jumpTo(target: LoopEntry, isBreak: Boolean) {
            if (target === loops.last()) {
                line(if (isBreak) "break;" else "continue;")
                return
            }
            val flag = if (isBreak) {
                target.breakFlag ?: names.allocate("kernelin_break", internal = true).also { target.breakFlag = it }
            } else {
                target.continueFlag ?: names.allocate("kernelin_continue", internal = true).also { target.continueFlag = it }
            }
            line("$flag = true;")
            line("break;")
            for (e in loops.reversed()) {
                if (e === target) break
                e.jumpsOut += target to isBreak
            }
        }

        private fun jump(j: IrBreakContinue) {
            val target = loops.lastOrNull { it.key == j.loop }
                ?: return run { fail(j, "this break/continue target isn't supported") }
            jumpTo(target, j is IrBreak)
        }

        fun translate(): String {
            when (val body = function.body) {
                is IrBlockBody -> body.statements.forEach(::statement)
                is IrExpressionBody -> statement(body.expression)
                null -> error(function, "function '${function.name}' has no body")
                else -> error(body, "unsupported function body")
            }
            return out.toString()
        }

        fun bindParameter(param: IrValueParameter, name: String, type: WType) {
            locals[param] = Expr(name, type)
        }

        fun bindArray(param: IrValueParameter, buffer: BufferParam) {
            arrayAliases[param] = buffer
        }

        private fun fail(element: IrElement, message: String): Expr {
            error(element, message, function)
            return Expr("0", WType.I32)
        }

        private fun line(text: String) {
            repeat(indent) { out.append("  ") }
            out.appendLine(text)
        }

        private inline fun block(header: String, body: () -> Unit) {
            line(if (header.isEmpty()) "{" else "$header {")
            indent++
            body()
            indent--
            line("}")
        }

        // ---------------------------------------------------------------- statements

        private fun statement(s: IrStatement) {
            when (s) {
                is IrVariable -> variable(s, null)
                is IrSetValue -> {
                    val target = locals[s.symbol.owner]
                        ?: return run { fail(s, "only local variables can be assigned") }
                    val pre = mutableListOf<String>()
                    val compound = compoundOperand(s.value) { it is IrGetValue && it.symbol == s.symbol }
                    if (compound != null) {
                        val rhs = expr(compound.second, pre)
                        pre.forEach(::line)
                        line("${target.code} ${compound.first}= ${cast(rhs, target.type)};")
                        return
                    }
                    val value = expr(s.value, pre)
                    pre.forEach(::line)
                    line("${target.code} = ${cast(value, target.type)};")
                }
                is IrCall -> callStatement(s)
                is IrWhen -> whenStatement(s)
                is IrBlock -> when (s.origin) {
                    IrStatementOrigin.FOR_LOOP -> forLoop(s)
                    IrStatementOrigin.POSTFIX_INCR, IrStatementOrigin.PREFIX_INCR,
                    IrStatementOrigin.POSTFIX_DECR, IrStatementOrigin.PREFIX_DECR ->
                        if (!incrementStatement(s)) s.statements.forEach(::statement)
                    else -> s.statements.forEach(::statement)
                }
                is IrContainerExpression -> s.statements.forEach(::statement)
                is IrWhileLoop -> whileLoop(s)
                is IrDoWhileLoop -> doWhileLoop(s)
                is IrBreakContinue -> jump(s)
                is IrReturn -> returnStatement(s)
                is IrTypeOperatorCall -> when (s.operator) {
                    IrTypeOperator.IMPLICIT_COERCION_TO_UNIT -> statement(s.argument)
                    else -> expressionStatement(s)
                }
                // A bare value as a statement (e.g. the result of `i++`) has no effect.
                is IrGetValue, is IrConst -> Unit
                is IrExpression -> expressionStatement(s)
                else -> fail(s, "this kind of declaration can't be used inside a kernel")
            }
        }

        private fun expressionStatement(e: IrExpression) {
            val pre = mutableListOf<String>()
            val value = expr(e, pre)
            pre.forEach(::line)
            when {
                // A value nobody uses (e.g. sqrt(x) on its own line): WGSL needs a phony assignment.
                value.type != WType.UNIT -> line("_ = ${value.code};")
                // A call to a helper that returns nothing (e.g. bump(out, i)) is a statement of its own.
                value.code.isNotEmpty() -> line("${value.code};")
            }
        }

        private fun variable(v: IrVariable, pre: MutableList<String>?) {
            val emit: (String) -> Unit = pre?.let { p -> { p += it } } ?: ::line
            val elementType = arrayElementType(v.type)
            if (elementType != null) {
                val buffer = v.initializer?.let(::arrayBuffer)
                if (buffer == null || v.isVar) {
                    fail(v, "array variables must be a 'val' that refers to a kernel parameter")
                } else {
                    arrayAliases[v] = buffer
                }
                return
            }
            val type = scalarType(v.type)
            if (type == null || type == WType.UNIT) {
                fail(v, "variable '${v.name}' has type ${v.type.describe()}; only Int, Float and Boolean are supported${v.type.suggestion()}")
                return
            }
            val init = v.initializer
            if (v.origin == IrDeclarationOrigin.IR_TEMPORARY_VARIABLE && !v.isVar && init != null && init.isStable()) {
                locals[v] = expr(init, pre ?: mutableListOf())
                return
            }
            val name = names.allocate(v.name.asString())
            val localPre = pre ?: mutableListOf()
            val value = init?.let { expr(it, localPre) }
            if (pre == null) localPre.forEach(::line)
            locals[v] = Expr(name, type)
            when {
                value == null -> emit("var $name: ${type.wgsl};")
                v.isVar -> emit("var $name: ${type.wgsl} = ${cast(value, type)};")
                else -> emit("let $name: ${type.wgsl} = ${cast(value, type)};")
            }
        }

        private fun callStatement(call: IrCall) {
            val fn = call.symbol.owner
            val elementType = fn.parentClassFq()?.let(::arrayElementTypeOf)
            if (elementType != null && fn.name.asString() == "set") {
                val pre = mutableListOf<String>()
                val buffer = arrayBuffer(call.receiver()!!) ?: return run { fail(call, "arrays must be kernel parameters") }
                buffer.written = true
                sideEffects++
                val index = expr(call.regularArgs()[0], pre)
                val target = "${buffer.name}[${cast(index, WType.I32)}]"
                val compound = compoundOperand(call.regularArgs()[1]) { get ->
                    get is IrCall && get.symbol.owner.name.asString() == "get" &&
                        get.receiver()?.let(::arrayBuffer) == buffer &&
                        cast(expr(get.regularArgs()[0], mutableListOf()), WType.I32) == cast(index, WType.I32)
                }
                if (compound != null) {
                    val rhs = expr(compound.second, pre)
                    pre.forEach(::line)
                    line("$target ${compound.first}= ${cast(rhs, buffer.elementType)};")
                    return
                }
                val value = expr(call.regularArgs()[1], pre)
                pre.forEach(::line)
                line("$target = ${cast(value, buffer.elementType)};")
                return
            }
            val lambda = call.regularArgs().getOrNull(1) as? IrFunctionExpression
            if (lambda != null && fn.kotlinFqName.asString() == "kotlin.repeat") {
                repeatLoop(call, lambda)
                return
            }
            expressionStatement(call)
        }

        /** `repeat(n) { body }` becomes `for (var it = 0; it < n; it++) { body }`. */
        private fun repeatLoop(call: IrCall, lambda: IrFunctionExpression) {
            val pre = mutableListOf<String>()
            val times = cast(expr(call.regularArgs()[0], pre), WType.I32)
            val function = lambda.function
            val param = function.parameters.firstOrNull { it.kind == IrParameterKind.Regular }
            val i = names.allocate(param?.name?.asString() ?: "it")
            param?.let { locals[it] = Expr(i, WType.I32) }
            val body = function.body as? IrBlockBody ?: return run { fail(lambda, "this repeat body isn't supported") }
            block("") {
                pre.forEach(::line)
                val end = names.allocate("kernelin_end", internal = true)
                line("let $end: i32 = $times;")
                loop(function) {
                    block("for (var $i: i32 = 0i; $i < $end; $i = $i + 1i)") {
                        body.statements.forEach(::statement)
                    }
                }
            }
        }

        private fun whenStatement(w: IrWhen) {
            if (w.origin == IrStatementOrigin.ANDAND || w.origin == IrStatementOrigin.OROR) {
                expressionStatement(w)
                return
            }
            ifChain(w.branches, 0)
        }

        /** Emits `branches[start..]` as one `if / else if / else` chain. */
        private fun ifChain(branches: List<org.jetbrains.kotlin.ir.expressions.IrBranch>, start: Int) {
            val first = branches[start]
            if (first.isElse()) {
                statement(first.result)
                return
            }
            val firstPre = mutableListOf<String>()
            val firstCond = expr(first.condition, firstPre)
            firstPre.forEach(::line)
            line("if (${cast(firstCond, WType.BOOL)}) {")
            indent++
            statement(first.result)
            indent--
            var i = start + 1
            while (i < branches.size) {
                val branch = branches[i]
                if (branch.isElse()) {
                    line("} else {")
                    indent++
                    statement(branch.result)
                    indent--
                    break
                }
                val pre = mutableListOf<String>()
                val cond = expr(branch.condition, pre)
                if (pre.isNotEmpty()) {
                    // The condition needs setup statements, so it can't be an `else if`.
                    line("} else {")
                    indent++
                    ifChain(branches, i)
                    indent--
                    break
                }
                line("} else if (${cast(cond, WType.BOOL)}) {")
                indent++
                statement(branch.result)
                indent--
                i++
            }
            line("}")
        }

        private fun org.jetbrains.kotlin.ir.expressions.IrBranch.isElse() =
            this is IrElseBranch || condition.isTrueConst()

        private fun whileLoop(loop: IrWhileLoop) = loop(loop) {
            val pre = mutableListOf<String>()
            val cond = expr(loop.condition, pre)
            if (pre.isEmpty()) {
                block("while (${cast(cond, WType.BOOL)})") { loop.body?.let(::statement) }
            } else {
                block("loop") {
                    pre.forEach(::line)
                    block("if (!(${cast(cond, WType.BOOL)}))") { line("break;") }
                    loop.body?.let(::statement)
                }
            }
        }

        private fun doWhileLoop(loop: IrDoWhileLoop) = loop(loop) {
            block("loop") {
                loop.body?.let(::statement)
                val pre = mutableListOf<String>()
                val cond = expr(loop.condition, pre)
                block("continuing") {
                    pre.forEach(::line)
                    line("break if !(${cast(cond, WType.BOOL)});")
                }
            }
        }

        /** `x = x op y` / `a[i] = a[i] op y` -> the operator and `y`, so we can emit `x op= y`. */
        private fun compoundOperand(value: IrExpression, isSelf: (IrExpression) -> Boolean): Pair<String, IrExpression>? {
            val call = value as? IrCall ?: return null
            val fn = call.symbol.owner
            if (fn.parentClassFq() !in setOf("kotlin.Int", "kotlin.Float")) return null
            val op = mapOf("plus" to "+", "minus" to "-", "times" to "*", "div" to "/", "rem" to "%")[fn.name.asString()] ?: return null
            val receiver = call.receiver() ?: return null
            if (!isSelf(receiver) || scalarType(receiver.type) != scalarType(call.type)) return null
            return op to call.regularArgs()[0]
        }

        /** `i++`, `--i` etc. used as a statement on a local variable become WGSL `i++;` / `i--;`. */
        private fun incrementStatement(b: IrBlock): Boolean {
            val set = b.statements.filterIsInstance<IrSetValue>().singleOrNull() ?: return false
            val target = locals[set.symbol.owner] ?: return false
            if (b.statements.any { it !is IrSetValue && it !is IrVariable && it !is IrGetValue }) return false
            val op = if (b.origin == IrStatementOrigin.POSTFIX_INCR || b.origin == IrStatementOrigin.PREFIX_INCR) "++" else "--"
            line("${target.code}$op;")
            return true
        }

        private fun returnStatement(r: IrReturn) {
            loops.lastOrNull { it.key == r.returnTargetSymbol.owner }?.let { target ->
                // `return@repeat` (or the end of a repeat lambda) moves on to the next iteration.
                if (!r.value.isUnitValue()) statement(r.value)
                jumpTo(target, isBreak = false)
                return
            }
            if (isKernel || function.returnType.let(::scalarType) == WType.UNIT) {
                if (!r.value.isUnitValue()) statement(r.value)
                line("return;")
                return
            }
            val pre = mutableListOf<String>()
            val value = expr(r.value, pre)
            pre.forEach(::line)
            line("return ${cast(value, scalarType(function.returnType)!!)};")
        }

        /**
         * `for (i in range) body` arrives as
         * `{ val it = range.iterator(); while (it.hasNext()) { val i = it.next(); body } }`.
         */
        private fun forLoop(b: IrBlock) {
            val iteratorVar = b.statements.getOrNull(0) as? IrVariable
            val whileLoop = b.statements.getOrNull(1) as? IrWhileLoop
            val iteratorCall = iteratorVar?.initializer as? IrCall
            val whileBody = whileLoop?.body as? IrContainerExpression
            val loopVar = whileBody?.statements?.firstOrNull() as? IrVariable
            val rangeExpr = iteratorCall?.receiver()
            if (whileLoop == null || rangeExpr == null || loopVar == null || iteratorCall.symbol.owner.name.asString() != "iterator") {
                fail(b, "this for loop shape isn't supported; loop over an Int range like 0 until n")
                return
            }
            val pre = mutableListOf<String>()
            val range = range(rangeExpr, pre) ?: return
            if (scalarType(loopVar.type) != WType.I32) {
                fail(loopVar, "for loops must use an Int loop variable")
                return
            }
            val i = names.allocate(loopVar.name.asString())
            locals[loopVar] = Expr(i, WType.I32)
            val body = { whileBody.statements.drop(1).forEach(::statement) }
            block("") {
                pre.forEach(::line)
                if (range.cannotOverflow()) {
                    // The plain loop: `i + step` stays within Int whenever the loop goes on.
                    val end = names.allocate("kernelin_end", internal = true)
                    line("let $end: i32 = ${cast(range.end, WType.I32)};")
                    val step = range.step.const?.let { constant(it as Int) } ?: names.allocate("kernelin_step", internal = true).also {
                        line("let $it: i32 = ${cast(range.step, WType.I32)};")
                    }
                    val cmp = when {
                        range.descending -> if (range.inclusive) ">=" else ">"
                        else -> if (range.inclusive) "<=" else "<"
                    }
                    val update = if (range.descending) "$i - $step" else "$i + $step"
                    loop(whileLoop) {
                        block("for (var $i: i32 = ${cast(range.start, WType.I32)}; $i $cmp $end; $i = $update)") { body() }
                    }
                    return@block
                }
                // Like Kotlin's IntProgression: count the steps up front, so the loop always ends, even when the
                // last element is next to Int.MAX_VALUE / MIN_VALUE (where `i + step` would wrap around).
                val first = names.allocate("kernelin_first", internal = true)
                val end = names.allocate("kernelin_end", internal = true)
                line("let $first: i32 = ${cast(range.start, WType.I32)};")
                line("let $end: i32 = ${cast(range.end, WType.I32)};")
                val stepConst = range.step.const as Int?
                val step = stepConst?.let { constant(it) } ?: names.allocate("kernelin_step", internal = true).also {
                    line("let $it: i32 = ${cast(range.step, WType.I32)};")
                }
                val stepU = stepConst?.let { "${it}u" } ?: "u32($step)"
                // Kotlin throws for a step of 0 or less; the GPU can't throw, so the loop runs no iterations.
                val nonEmpty = when {
                    range.descending -> "$first >= $end"
                    range.inclusive -> "$first <= $end"
                    else -> "$first < $end"
                }
                val last = names.allocate("kernelin_last", internal = true)
                val count = names.allocate("kernelin_count", internal = true)
                val done = names.allocate("kernelin_done", internal = true)
                // Distance from the first to the last value as u32: it fits even for Int.MIN_VALUE..Int.MAX_VALUE.
                val distance = when {
                    range.descending -> "(u32($first) - u32($end))"
                    range.inclusive -> "(u32($end) - u32($first))"
                    else -> "(u32($end) - u32($first) - 1u)"
                }
                val bits = "bitcast<u32>($step)"
                block(if (stepConst == null) "if ($step > 0i && $nonEmpty)" else "if ($nonEmpty)") {
                    line("var $i: i32 = $first;")
                    line("let $last: u32 = $distance / $stepU;")
                    line("var $count: u32 = 0u;")
                    line("var $done: bool = false;")
                    loop(whileLoop) {
                        block("loop") {
                            body()
                            block("continuing") {
                                // Wraps only after the last iteration, when the value is no longer used.
                                line("$i = bitcast<i32>(bitcast<u32>($i) ${if (range.descending) "-" else "+"} $bits);")
                                // Count up and compare before adding: a counter that counts down to a wrap made
                                // Mesa's loop analysis divide INT_MIN by -1 at shader compile time (SIGFPE in
                                // lavapipe for constant ranges like 0..Int.MAX_VALUE). [done] is declared outside:
                                // naga's Metal output got a `let` inside `continuing` wrong.
                                line("$done = $count == $last;")
                                line("$count = $count + 1u;")
                                line("break if $done;")
                            }
                        }
                    }
                }
            }
        }

        /**
         * A range to loop over. [start], [end] and [step] are Int expressions; [step] is positive for both
         * directions (as written after `step`), [descending] for `downTo`.
         */
        private inner class Range(
            val start: Expr,
            val end: Expr,
            val step: Expr,
            val inclusive: Boolean,
            val descending: Boolean,
            val stepped: Boolean = false,
        ) {
            /** True when a plain `for (i = start; i < end; i += step)` can never wrap around (and step > 0). */
            fun cannotOverflow(): Boolean {
                val s = step.const as? Int ?: return false
                val e = end.const as? Int
                return when {
                    descending -> e != null && e >= Int.MIN_VALUE + s
                    // i < end <= Int.MAX_VALUE, so i + 1 always fits.
                    !inclusive && s == 1 -> true
                    !inclusive -> e != null && e.toLong() - 1 + s <= Int.MAX_VALUE
                    else -> e != null && e.toLong() + s <= Int.MAX_VALUE
                }
            }
        }

        private fun range(e: IrExpression, pre: MutableList<String>): Range? {
            val call = e as? IrCall ?: return null.also { fail(e, "for loops must iterate over an Int range like 0 until n") }
            val fn = call.symbol.owner
            val name = fn.name.asString()
            fun int(arg: IrExpression) = expr(arg, pre).let { if (it.type == WType.I32) it else Expr(cast(it, WType.I32), WType.I32) }
            val one = lit(1)
            return when (name) {
                "until", "rangeUntil" -> Range(int(call.receiver()!!), int(call.regularArgs()[0]), one, inclusive = false, descending = false)
                "rangeTo" -> Range(int(call.receiver()!!), int(call.regularArgs()[0]), one, inclusive = true, descending = false)
                "downTo" -> Range(int(call.receiver()!!), int(call.regularArgs()[0]), one, inclusive = true, descending = true)
                "<get-indices>" -> {
                    val buffer = arrayBuffer(call.receiver()!!)
                        ?: return null.also { fail(call, "indices must be read from a kernel array parameter") }
                    Range(lit(0), Expr("i32(arrayLength(&${buffer.name}))", WType.I32), one, inclusive = false, descending = false)
                }
                "step" -> {
                    val inner = range(call.receiver()!!, pre) ?: return null
                    if (inner.stepped) return null.also { fail(call, "only one step per range is supported; combine the steps into one") }
                    val step = int(call.regularArgs()[0])
                    val s = step.const as? Int
                    if (s != null && s <= 0) {
                        return null.also { fail(call, "step must be positive, but it is $s (Kotlin throws IllegalArgumentException)") }
                    }
                    Range(inner.start, inner.end, step, inner.inclusive, inner.descending, stepped = true)
                }
                else -> null.also { fail(call, "for loops must iterate over an Int range (until, .., ..<, downTo, step, indices)") }
            }
        }

        // ---------------------------------------------------------------- expressions

        private fun expr(e: IrExpression, pre: MutableList<String>): Expr = when (e) {
            is IrConst -> constant(e)
            is IrGetValue -> getValue(e)
            is IrCall -> call(e, pre)
            is IrWhen -> whenExpression(e, pre)
            is IrTypeOperatorCall -> when (e.operator) {
                IrTypeOperator.IMPLICIT_CAST, IrTypeOperator.IMPLICIT_NOTNULL,
                IrTypeOperator.IMPLICIT_INTEGER_COERCION -> expr(e.argument, pre)
                IrTypeOperator.IMPLICIT_COERCION_TO_UNIT -> expr(e.argument, pre).copy(type = WType.UNIT)
                else -> fail(e, "type checks and casts (is/as) aren't supported on the GPU")
            }
            is IrContainerExpression -> blockExpression(e, pre)
            is IrFunctionExpression -> fail(e, "lambdas aren't supported in kernels yet (this includes repeat { } and forEach { }); use a for loop")
            is IrStringConcatenation -> fail(e, "strings aren't supported on the GPU")
            is IrThrow -> fail(e, "throwing exceptions isn't supported on the GPU")
            is IrTry -> fail(e, "try/catch isn't supported on the GPU")
            else -> fail(e, "this expression (${e.javaClass.simpleName.removePrefix("Ir").removeSuffix("Impl")}) isn't supported in kernels yet")
        }

        private fun constant(c: IrConst): Expr = when (val v = c.value) {
            is Int, is Float, is Boolean -> lit(v)
            is Double -> fail(c, "Double isn't supported on most GPUs; use Float (add an f suffix: 1.0f)")
            is Long -> fail(c, "Long isn't supported on the GPU; use Int")
            else -> fail(c, "this constant type isn't supported on the GPU")
        }

        private fun constant(v: Int) = if (v == Int.MIN_VALUE) "(-2147483647i - 1i)" else "${v}i"

        private fun getValue(e: IrGetValue): Expr {
            val owner = e.symbol.owner
            locals[owner]?.let { return it }
            scalars[owner]?.let {
                // Booleans travel as u32 (bool isn't allowed in uniform buffers).
                return if (it.type == WType.BOOL) Expr("(kernelin_params.${it.name} != 0u)", WType.BOOL)
                else Expr("kernelin_params.${it.name}", it.type)
            }
            if (owner in buffers || owner in arrayAliases) {
                return fail(e, "an array can only be used for indexing (a[i]), a.size or a.indices")
            }
            return fail(e, "'${owner.name}' can't be used here")
        }

        private fun blockExpression(b: IrContainerExpression, pre: MutableList<String>): Expr {
            val statements = b.statements
            if (statements.isEmpty()) return Expr("", WType.UNIT)
            for (s in statements.dropLast(1)) {
                if (s is IrVariable) variable(s, pre) else return fail(s, "this statement can't be used inside an expression")
            }
            val last = statements.last() as? IrExpression ?: return fail(b, "a block used as a value must end with an expression")
            return expr(last, pre)
        }

        private fun whenExpression(w: IrWhen, pre: MutableList<String>): Expr {
            val b = w.branches
            if (w.origin == IrStatementOrigin.ANDAND && b.size == 2) {
                // a && b  ==  if (a) b else false
                return shortCircuit(b[0].condition, b[0].result, isAnd = true, pre)
            }
            if (w.origin == IrStatementOrigin.OROR && b.size == 2) {
                // a || b  ==  if (a) true else b
                return shortCircuit(b[0].condition, b[1].result, isAnd = false, pre)
            }
            val type = scalarType(w.type)
            if (type == null || type == WType.UNIT) return fail(w, "this if/when must produce an Int, Float or Boolean value")
            val elseBranch = b.lastOrNull()?.takeIf { it is IrElseBranch || it.condition.isTrueConst() }
                ?: return fail(w, "an if/when used as a value needs an else branch")
            val branches = b.dropLast(1)
            if (branches.isEmpty()) return expr(elseBranch.result, pre)
            // The first condition always runs; everything else runs only when Kotlin would run it.
            val conds = listOf(Part(emptyList(), expr(branches[0].condition, pre), false)) +
                branches.drop(1).map { part(it.condition) }
            val values = branches.map { part(it.result) }
            val elseValue = part(elseBranch.result)
            val conditional = conds.drop(1) + values + elseValue
            if (conditional.any { it.pre.isNotEmpty() || it.sideEffects }) {
                // select() evaluates every side, so a branch with setup statements or side effects (a helper that
                // writes an array) becomes a real if/else that assigns a temporary.
                val tmp = names.allocate("kernelin_value", internal = true)
                pre += "var $tmp: ${type.wgsl};"
                fun emit(k: Int, indent: String) {
                    if (k == branches.size) {
                        elseValue.pre.forEach { pre += indent + it }
                        pre += "$indent$tmp = ${cast(elseValue.value, type)};"
                        return
                    }
                    conds[k].pre.forEach { pre += indent + it }
                    pre += "${indent}if (${cast(conds[k].value, WType.BOOL)}) {"
                    values[k].pre.forEach { pre += "$indent  $it" }
                    pre += "$indent  $tmp = ${cast(values[k].value, type)};"
                    pre += "$indent} else {"
                    emit(k + 1, "$indent  ")
                    pre += "$indent}"
                }
                emit(0, "")
                return Expr(tmp, type)
            }
            // Plain values: one select() chain, which is cheaper on the GPU than branching.
            var result = cast(elseValue.value, type)
            var resultConst: Any? = elseValue.value.const.takeIf { elseValue.value.type == type }
            for (k in branches.indices.reversed()) {
                val cond = conds[k].value
                val value = values[k].value
                when (cond.const) {
                    true -> { result = cast(value, type); resultConst = value.const.takeIf { value.type == type } }
                    false -> Unit
                    else -> { result = "select($result, ${cast(value, type)}, ${cast(cond, WType.BOOL)})"; resultConst = null }
                }
            }
            return resultConst?.let(::lit) ?: Expr(result, type)
        }

        /** A translated expression with its own setup statements, and whether it has side effects. */
        private inner class Part(val pre: List<String>, val value: Expr, val sideEffects: Boolean)

        private fun part(e: IrExpression): Part {
            val pre = mutableListOf<String>()
            val before = sideEffects
            val value = expr(e, pre)
            return Part(pre, value, sideEffects != before)
        }

        /**
         * `l && r` / `l || r`. WGSL's && and || skip the right side too, but setup statements for it can't go
         * inside the expression, so then the right side moves into an if.
         */
        private fun shortCircuit(left: IrExpression, right: IrExpression, isAnd: Boolean, pre: MutableList<String>): Expr {
            val l = expr(left, pre)
            val r = part(right)
            if (l.const == !isAnd) return lit(!isAnd)
            if (l.const == isAnd) return r.value.also { pre += r.pre }
            if (r.pre.isEmpty()) {
                return Expr("(${cast(l, WType.BOOL)} ${if (isAnd) "&&" else "||"} ${cast(r.value, WType.BOOL)})", WType.BOOL)
            }
            val tmp = names.allocate(if (isAnd) "kernelin_and" else "kernelin_or", internal = true)
            pre += "var $tmp: bool = ${cast(l, WType.BOOL)};"
            pre += if (isAnd) "if ($tmp) {" else "if (!$tmp) {"
            r.pre.forEach { pre += "  $it" }
            pre += "  $tmp = ${cast(r.value, WType.BOOL)};"
            pre += "}"
            return Expr(tmp, WType.BOOL)
        }

        private fun call(call: IrCall, pre: MutableList<String>): Expr {
            val fn = call.symbol.owner
            val name = fn.name.asString()
            val owner = fn.parentClassFq() ?: fn.packageFq()
            val resultType = scalarType(call.type)

            if (owner == "io.github.rajumark.kernelin" && name == "globalId") return Expr("i32(kernelin_id)", WType.I32)
            if (owner == "io.github.rajumark.kernelin" && name in RANDOM_FUNCTIONS) return randomCall(call, name, pre)

            // Built-in comparison operators.
            if (owner == "kotlin.internal.ir") return builtinOperator(call, name, pre)

            // FloatArray / IntArray members.
            owner?.let(::arrayElementTypeOf)?.let { element ->
                val buffer = arrayBuffer(call.receiver()!!) ?: return fail(call, "arrays must be kernel parameters")
                return when (name) {
                    "get" -> Expr("${buffer.name}[${cast(expr(call.regularArgs()[0], pre), WType.I32)}]", element)
                    "<get-size>" -> Expr("i32(arrayLength(&${buffer.name}))", WType.I32)
                    "set" -> fail(call, "array assignment can't be used as a value")
                    else -> fail(call, "FloatArray/IntArray.$name() isn't supported on the GPU")
                }
            }

            if (owner == "kotlin.Double" && (name == "toFloat" || name == "toInt")) return doubleConstant(call, name)
            if (owner == "kotlin" && name in setOf("isNaN", "isInfinite", "isFinite")) {
                val x = expr(call.receiver()!!, pre)
                if (x.type != WType.F32) return fail(call, "$name() is supported for Float only")
                usesFloatChecks = true
                return when (name) {
                    "isNaN" -> Expr("kernelin_is_nan(${x.code})", WType.BOOL)
                    "isInfinite" -> Expr("kernelin_is_inf(${x.code})", WType.BOOL)
                    else -> Expr("(!kernelin_is_nan(${x.code}) && !kernelin_is_inf(${x.code}))", WType.BOOL)
                }
            }
            if (owner == "kotlin.Int" || owner == "kotlin.Float" || owner == "kotlin.Boolean") {
                return primitiveMember(call, name, pre)
            }
            if (name == "contains" && (owner?.startsWith("kotlin.ranges") == true || owner == "kotlin.collections")) {
                return rangeContains(call, pre)
            }
            if (owner == "kotlin.math") return mathFunction(call, name, pre)
            if (owner == "kotlin.ranges" && name in setOf("coerceIn", "coerceAtLeast", "coerceAtMost")) {
                val t = resultType ?: return fail(call, "unsupported coerce type")
                val xe = expr(call.receiver()!!, pre)
                val ae = call.regularArgs().map { expr(it, pre) }
                if (xe.const is Number && ae.all { it.const is Number }) {
                    val v = (listOf(xe) + ae).map { it.const as Number }
                    val folded: Any? = if (t == WType.I32) {
                        val n = v.map { it.toInt() }
                        when (name) { "coerceIn" -> if (n[1] <= n[2]) n[0].coerceIn(n[1], n[2]) else null; "coerceAtLeast" -> n[0].coerceAtLeast(n[1]); else -> n[0].coerceAtMost(n[1]) }
                    } else {
                        val n = v.map { it.toFloat() }
                        when (name) { "coerceIn" -> if (n[1] <= n[2]) n[0].coerceIn(n[1], n[2]) else null; "coerceAtLeast" -> n[0].coerceAtLeast(n[1]); else -> n[0].coerceAtMost(n[1]) }
                    }
                    folded?.let { return lit(it) }
                }
                val x = cast(xe, t)
                val a = ae.map { cast(it, t) }
                // Float: Kotlin's comparisons, so NaN and -0f come out as on the CPU (WGSL's clamp/min/max don't say).
                if (t == WType.F32) return when (name) {
                    "coerceIn" -> Expr("${use("kernelin_coerce_in")}($x, ${a[0]}, ${a[1]})", t)
                    "coerceAtLeast" -> Expr("${use("kernelin_at_least")}($x, ${a[0]})", t)
                    else -> Expr("${use("kernelin_at_most")}($x, ${a[0]})", t)
                }
                return when (name) {
                    "coerceIn" -> Expr("clamp($x, ${a[0]}, ${a[1]})", t)
                    "coerceAtLeast" -> Expr("max($x, ${a[0]})", t)
                    else -> Expr("min($x, ${a[0]})", t)
                }
            }
            return helperCall(call, fn, pre)
        }

        /** randomInt/randomFloat/randomIntBelow/randomNormal: Kernelin's counter-based random numbers. */
        private fun randomCall(call: IrCall, name: String, pre: MutableList<String>): Expr {
            // A left-out `stream` is null here (default arguments aren't filled in yet); it means 0.
            val args = call.arguments.map { it?.let { a -> cast(expr(a, pre), WType.I32) } ?: "0i" }
            val wgsl = RANDOM_FUNCTIONS.getValue(name)
            runtime += "kernelin_random_bits"
            runtime += wgsl
            val type = if (name == "randomInt" || name == "randomIntBelow") WType.I32 else WType.F32
            return Expr("$wgsl(${args.joinToString(", ")})", type)
        }

        /** `x in a..b`, `x in a until b`, `x in a downTo b` (also `!in` and `when (x) { in a..b -> }`). */
        private fun rangeContains(call: IrCall, pre: MutableList<String>): Expr {
            val range = call.receiver() as? IrCall
            val kind = range?.symbol?.owner?.name?.asString()
            if (range == null || kind !in setOf("rangeTo", "until", "rangeUntil", "downTo")) {
                return fail(call, "'in' supports ranges written in place with .., ..<, until or downTo (like x in 0 until n)")
            }
            val value = expr(call.regularArgs()[0], pre)
            val low = expr(range.receiver()!!, pre)
            val high = expr(range.regularArgs()[0], pre)
            val t = if (WType.F32 in setOf(value.type, low.type, high.type)) WType.F32 else WType.I32
            var x = cast(value, t)
            if (!x.matches(SIMPLE_VALUE)) {
                val tmp = names.allocate("kernelin_in", internal = true)
                pre += "let $tmp: ${t.wgsl} = $x;"
                x = tmp
            }
            val a = cast(low, t)
            val b = cast(high, t)
            return when (kind) {
                "rangeTo" -> Expr("(($x >= $a) && ($x <= $b))", WType.BOOL)
                "downTo" -> Expr("(($x <= $a) && ($x >= $b))", WType.BOOL)
                else -> Expr("(($x >= $a) && ($x < $b))", WType.BOOL)
            }
        }

        /** Doubles can't run on most GPUs, but a Double constant converted right away (PI.toFloat()) is fine. */
        private fun doubleConstant(call: IrCall, name: String): Expr {
            val receiver = call.receiver()
            val value: Double? = when {
                receiver is IrConst -> receiver.value as? Double
                receiver is IrCall && receiver.symbol.owner.packageFq() == "kotlin.math" ->
                    when (receiver.symbol.owner.name.asString()) { "<get-PI>" -> kotlin.math.PI; "<get-E>" -> kotlin.math.E; else -> null }
                else -> null
            }
            if (value == null || !value.isFinite()) {
                return fail(call, "Double isn't supported on most GPUs; use Float (only Double constants converted right away, like PI.toFloat(), are allowed)")
            }
            return if (name == "toFloat") lit(value.toFloat()) else lit(value.toInt())
        }

        private fun builtinOperator(call: IrCall, name: String, pre: MutableList<String>): Expr {
            val args = call.regularArgs()
            // `a < b` with mixed types arrives as `less(a.compareTo(b), 0)`.
            val compareTo = (args.getOrNull(0) as? IrCall)?.takeIf {
                it.symbol.owner.name.asString() == "compareTo" && args[1].isIntLiteral(0)
            }
            val (l, r) = if (compareTo != null) {
                promote(expr(compareTo.receiver()!!, pre), expr(compareTo.regularArgs()[0], pre))
            } else {
                promote(expr(args[0], pre), expr(args[1], pre))
            }
            if (l.const is Number && r.const is Number) {
                // Operators compare floats the IEEE 754 way (-0f == 0f, NaN is unordered), unlike compareTo.
                val folded = if (l.type == WType.F32 || r.type == WType.F32) {
                    val a = (l.const as Number).toFloat()
                    val b = (r.const as Number).toFloat()
                    when (name) {
                        "less" -> a < b
                        "lessOrEqual" -> a <= b
                        "greater" -> a > b
                        "greaterOrEqual" -> a >= b
                        "EQEQ", "EQEQEQ", "ieee754equals" -> a == b
                        else -> null
                    }
                } else {
                    val a = l.const as Int
                    val b = r.const as Int
                    when (name) {
                        "less" -> a < b
                        "lessOrEqual" -> a <= b
                        "greater" -> a > b
                        "greaterOrEqual" -> a >= b
                        "EQEQ", "EQEQEQ", "ieee754equals" -> a == b
                        else -> null
                    }
                }
                if (folded != null) return lit(folded)
            }
            if (l.const is Boolean && r.const is Boolean && name in setOf("EQEQ", "EQEQEQ")) return lit(l.const == r.const)
            val op = when (name) {
                "less" -> "<"
                "lessOrEqual" -> "<="
                "greater" -> ">"
                "greaterOrEqual" -> ">="
                "EQEQ", "EQEQEQ", "ieee754equals" -> "=="
                else -> return fail(call, "operator '$name' isn't supported on the GPU")
            }
            return Expr("(${l.code} $op ${r.code})", WType.BOOL)
        }

        private fun primitiveMember(call: IrCall, name: String, pre: MutableList<String>): Expr {
            val receiver = expr(call.receiver()!!, pre)
            val args = call.regularArgs().map { expr(it, pre) }
            val resultType = scalarType(call.type) ?: return fail(call, "operator result type ${call.type.describe()} isn't supported")
            if (receiver.const != null && args.all { it.const != null }) {
                if (name in setOf("div", "rem") && receiver.type == WType.I32 && args[0].const == 0) {
                    return fail(call, "integer division by zero (Kotlin would throw ArithmeticException here)")
                }
                fold(name, receiver.const, args.map { it.const!! })?.let { return lit(it) }
            }
            fun binary(op: String): Expr {
                val t = if (resultType == WType.BOOL && receiver.type != WType.BOOL) promote(receiver, args[0]).first.type else resultType
                return Expr("(${cast(receiver, t)} $op ${cast(args[0], t)})", resultType)
            }
            return when (name) {
                "plus" -> binary("+")
                "minus" -> binary("-")
                "times" -> binary("*")
                "div" -> binary("/")
                "rem" -> binary("%")
                "unaryMinus" -> Expr("(-${receiver.code})", resultType)
                "unaryPlus" -> receiver
                "inc" -> Expr("(${receiver.code} + ${one(resultType)})", resultType)
                "dec" -> Expr("(${receiver.code} - ${one(resultType)})", resultType)
                "and" -> binary("&")
                "or" -> binary("|")
                "xor" -> if (resultType == WType.BOOL) binary("!=") else binary("^")
                "not" -> Expr("(!${receiver.code})", WType.BOOL)
                "inv" -> Expr("(~${receiver.code})", WType.I32)
                // Kotlin uses only the low 5 bits of the shift count; say so explicitly for every GPU.
                "shl" -> Expr("(${receiver.code} << (u32(${args[0].code}) & 31u))", WType.I32)
                "shr" -> Expr("(${receiver.code} >> (u32(${args[0].code}) & 31u))", WType.I32)
                "ushr" -> Expr("i32(u32(${receiver.code}) >> (u32(${args[0].code}) & 31u))", WType.I32)
                "toInt" -> if (receiver.type == WType.F32) {
                    usesFloatToInt = true
                    Expr("kernelin_f2i(${receiver.code})", WType.I32)
                } else {
                    Expr(cast(receiver, resultType), resultType)
                }
                "toFloat" -> Expr(cast(receiver, resultType), resultType)
                "compareTo" -> {
                    val (l, r) = promote(receiver, args[0])
                    // Float.compareTo is a total order: -0f < 0f, and NaN is above everything and equal to itself.
                    if (l.type == WType.F32) Expr("${use("kernelin_compare")}(${l.code}, ${r.code})", WType.I32)
                    else Expr("select(select(0i, 1i, ${l.code} > ${r.code}), -1i, ${l.code} < ${r.code})", WType.I32)
                }
                "equals" -> {
                    val (l, r) = promote(receiver, args[0])
                    Expr("(${l.code} == ${r.code})", WType.BOOL)
                }
                else -> fail(call, "${receiver.type.kotlin}.$name() isn't supported on the GPU")
            }
        }

        private fun mathFunction(call: IrCall, name: String, pre: MutableList<String>): Expr {
            val resultType = scalarType(call.type)
            val allArgs = listOfNotNull(call.receiver()) + call.regularArgs()
            if (resultType == null || allArgs.any { scalarType(it.type) == null }) {
                return fail(call, "kotlin.math.$name with Double isn't supported on the GPU; use the Float version")
            }
            val args = allArgs.map { expr(it, pre) }
            if (args.all { it.const is Number }) foldMath(name, args.map { it.const as Number }, resultType)?.let { return lit(it) }
            fun f(i: Int) = cast(args[i], WType.F32)
            fun t(i: Int) = cast(args[i], resultType)
            val floatOnly = mapOf(
                "sqrt" to "sqrt", "sin" to "sin", "cos" to "cos", "tan" to "tan", "asin" to "asin",
                "acos" to "acos", "atan" to "atan", "sinh" to "sinh", "cosh" to "cosh", "tanh" to "tanh",
                "exp" to "exp", "ln" to "log", "log2" to "log2", "floor" to "floor", "ceil" to "ceil",
                "round" to "round", "truncate" to "trunc",
            )
            floatOnly[name]?.let { return Expr("$it(${f(0)})", WType.F32) }
            return when (name) {
                "abs" -> Expr("abs(${t(0)})", resultType)
                "sign" -> if (resultType == WType.F32) Expr("${use("kernelin_sign")}(${t(0)})", resultType) else Expr("sign(${t(0)})", resultType)
                "min" -> Expr("${if (resultType == WType.F32) use("kernelin_min") else "min"}(${t(0)}, ${t(1)})", resultType)
                "max" -> Expr("${if (resultType == WType.F32) use("kernelin_max") else "max"}(${t(0)}, ${t(1)})", resultType)
                "atan2" -> Expr("atan2(${f(0)}, ${f(1)})", WType.F32)
                "pow" -> {
                    usesPow = true
                    usesFloatChecks = true
                    usesSpecialFloats = true
                    Expr("kernelin_pow(${f(0)}, ${f(1)})", WType.F32)
                }
                "hypot" -> Expr("${use("kernelin_hypot")}(${f(0)}, ${f(1)})", WType.F32)
                "log10" -> Expr("${use("kernelin_log10")}(${f(0)})", WType.F32)
                else -> fail(call, "kotlin.math.$name isn't supported on the GPU yet")
            }
        }

        /** kotlin.math on constants, computed with the Float versions Kotlin itself would call. */
        private fun foldMath(name: String, a: List<Number>, resultType: WType): Any? {
            val x = a[0].toFloat()
            val y = a.getOrNull(1)?.toFloat()
            if (resultType == WType.I32) {
                val i = a[0].toInt()
                val j = a.getOrNull(1)?.toInt()
                return when (name) {
                    "abs" -> kotlin.math.abs(i)
                    "sign" -> i.sign
                    "min" -> j?.let { kotlin.math.min(i, it) }
                    "max" -> j?.let { kotlin.math.max(i, it) }
                    else -> null
                }
            }
            return when (name) {
                "sqrt" -> kotlin.math.sqrt(x); "sin" -> kotlin.math.sin(x); "cos" -> kotlin.math.cos(x)
                "tan" -> kotlin.math.tan(x); "asin" -> kotlin.math.asin(x); "acos" -> kotlin.math.acos(x)
                "atan" -> kotlin.math.atan(x); "sinh" -> kotlin.math.sinh(x); "cosh" -> kotlin.math.cosh(x)
                "tanh" -> kotlin.math.tanh(x); "exp" -> kotlin.math.exp(x); "ln" -> kotlin.math.ln(x)
                "log2" -> kotlin.math.log2(x); "log10" -> kotlin.math.log10(x); "floor" -> kotlin.math.floor(x)
                "ceil" -> kotlin.math.ceil(x); "round" -> kotlin.math.round(x); "truncate" -> kotlin.math.truncate(x)
                "abs" -> kotlin.math.abs(x); "sign" -> kotlin.math.sign(x)
                "min" -> y?.let { kotlin.math.min(x, it) }; "max" -> y?.let { kotlin.math.max(x, it) }
                "atan2" -> y?.let { kotlin.math.atan2(x, it) }; "pow" -> y?.let { x.pow(it) }
                "hypot" -> y?.let { kotlin.math.hypot(x, it) }
                else -> null
            }
        }

        private fun helperCall(call: IrCall, fn: IrSimpleFunction, pre: MutableList<String>): Expr {
            if (call.arguments.any { it is IrFunctionExpression }) {
                return fail(call, "lambdas aren't supported in kernels yet ('${fn.name}' takes one); use a for loop")
            }
            if (fn.body == null) {
                return fail(call, "'${fn.kotlinFqName}' can't run on the GPU (only kotlin.math, Int/Float operators and your own functions are supported)")
            }
            if (fn.hasAnnotation(KERNEL_ANNOTATION)) return fail(call, "a kernel can't call another @Kernel function")
            if (fn in helperStack) return fail(call, "recursion isn't allowed on the GPU ('${fn.name}' calls itself)")
            val arrays = LinkedHashMap<Int, BufferParam>()
            fn.parameters.forEachIndexed { i, p ->
                if (arrayElementType(p.type) != null) {
                    arrays[i] = call.arguments[i]?.let(::arrayBuffer)
                        ?: return fail(call, "arrays passed to '${fn.name}' must be the kernel's own array parameters")
                }
            }
            val key: Any = if (arrays.isEmpty()) fn else HelperKey(fn, arrays.values.toList())
            val helper = helpers[key] ?: translateHelper(call, fn, key, arrays) ?: return Expr("0", scalarType(call.type) ?: WType.I32)
            if (helper.writes) sideEffects++
            val args = fn.parameters.mapIndexedNotNull { i, p ->
                if (i in arrays) return@mapIndexedNotNull null
                val arg = call.arguments[i] ?: return fail(call, "default arguments aren't supported in kernel helpers yet")
                cast(expr(arg, pre), scalarType(p.type) ?: WType.I32)
            }
            val type = scalarType(fn.returnType) ?: WType.I32
            return Expr("${helper.name}(${args.joinToString(", ")})", type)
        }

        private fun translateHelper(call: IrCall, fn: IrSimpleFunction, key: Any, arrays: Map<Int, BufferParam>): Helper? {
            val returnType = scalarType(fn.returnType)
            var ok = returnType != null
            if (returnType == null) fail(call, "helper '${fn.name}' returns ${fn.returnType.describe()}; helpers must return Int, Float, Boolean or Unit")
            if (fn.typeParameters.isNotEmpty()) { ok = false; fail(call, "helper '${fn.name}' can't have type parameters") }
            val paramTypes = fn.parameters.mapIndexed { i, p ->
                val t = scalarType(p.type)
                if (i in arrays) {
                    null
                } else if (p.kind == IrParameterKind.DispatchReceiver || p.kind == IrParameterKind.Context) {
                    ok = false; fail(call, "helper '${fn.name}' must be a top-level function"); null
                } else if (t == null || t == WType.UNIT) {
                    ok = false; fail(call, "helper '${fn.name}' parameter '${p.name}' must be Int, Float, Boolean, FloatArray or IntArray"); null
                } else t
            }
            if (!ok) return null

            val helper = Helper(names.allocate(fn.name.asString()))
            helperStack.addLast(fn)
            val translator = BodyTranslator(fn, isKernel = false)
            arrays.forEach { (i, buffer) -> translator.bindArray(fn.parameters[i], buffer) }
            val paramDecls = fn.parameters.mapIndexedNotNull { i, p ->
                if (i in arrays) return@mapIndexedNotNull null
                val pname = names.allocate(if (p.kind == IrParameterKind.ExtensionReceiver) "receiver" else p.name.asString())
                translator.bindParameter(p, pname, paramTypes[i]!!)
                "$pname: ${paramTypes[i]!!.wgsl}"
            }
            val effectsBefore = sideEffects
            val body = translator.translate()
            helper.writes = sideEffects != effectsBefore
            helperStack.removeLast()
            val ret = if (returnType == WType.UNIT) "" else " -> ${returnType!!.wgsl}"
            helper.code = "fn ${helper.name}(${paramDecls.joinToString(", ")})$ret {\n$body}\n"
            helpers[key] = helper
            return helper
        }

        /** Resolves an expression that refers to one of the kernel's array parameters. */
        private fun arrayBuffer(e: IrExpression): BufferParam? {
            val owner = (e as? IrGetValue)?.symbol?.owner ?: return null
            return buffers[owner] ?: arrayAliases[owner]
        }
    }

    // -------------------------------------------------------------------- helpers

    private fun promote(a: Expr, b: Expr): Pair<Expr, Expr> =
        if (a.type == b.type) a to b
        else if (a.type == WType.F32 || b.type == WType.F32) Expr(cast(a, WType.F32), WType.F32) to Expr(cast(b, WType.F32), WType.F32)
        else a to b

    private fun cast(e: Expr, to: WType): String =
        if (e.type == to || to == WType.UNIT || e.type == WType.UNIT) e.code else "${to.wgsl}(${e.code})"

    private fun one(t: WType) = if (t == WType.F32) "1.0f" else "1i"

    private fun floatLiteral(v: Float): String {
        val s = v.toString().lowercase()
        return if ('.' in s || 'e' in s) "${s}f" else "$s.0f"
    }
}

internal fun scalarType(type: IrType): WType? {
    if (type.isMarkedNullable()) return null
    return when (type.classFqName?.asString()) {
        "kotlin.Int" -> WType.I32
        "kotlin.Float" -> WType.F32
        "kotlin.Boolean" -> WType.BOOL
        "kotlin.Unit" -> WType.UNIT
        else -> null
    }
}

internal fun arrayElementType(type: IrType): WType? =
    if (type.isMarkedNullable()) null else type.classFqName?.asString()?.let(::arrayElementTypeOf)

internal fun arrayElementTypeOf(fqName: String): WType? = when (fqName) {
    "kotlin.FloatArray" -> WType.F32
    "kotlin.IntArray" -> WType.I32
    else -> null
}

private fun IrType.suggestion(): String = when (classFqName?.asString()) {
    "kotlin.Double" -> " (Double isn't supported on most GPUs; use Float)"
    "kotlin.Long" -> " (Long isn't supported on the GPU; use Int)"
    "kotlin.String" -> " (strings aren't supported on the GPU)"
    else -> ""
}

private fun IrType.describe(): String =
    (classFqName?.shortName()?.asString() ?: "this type") + if (isMarkedNullable()) "?" else ""

private fun IrSimpleFunction.parentClassFq(): String? = (parent as? IrClass)?.kotlinFqName?.asString()

private fun IrSimpleFunction.packageFq(): String? = (parent as? IrPackageFragment)?.packageFqName?.asString()

private fun IrCall.receiver(): IrExpression? {
    val params = symbol.owner.parameters
    val i = params.indexOfFirst { it.kind == IrParameterKind.DispatchReceiver || it.kind == IrParameterKind.ExtensionReceiver }
    return if (i >= 0) arguments[i] else null
}

private fun IrCall.regularArgs(): List<IrExpression> =
    symbol.owner.parameters.mapIndexedNotNull { i, p -> if (p.kind == IrParameterKind.Regular) arguments[i] else null }

/** A value that can't change while a compound expression runs: a constant, a parameter or a `val`. */
private fun IrExpression.isStable(): Boolean = this is IrConst ||
    this is IrGetValue && symbol.owner.let { it is IrValueParameter || it is IrVariable && !it.isVar }

private fun IrExpression.isTrueConst() = this is IrConst && value == true

private fun IrExpression.isIntLiteral(v: Int) = this is IrConst && value == v

private fun IrExpression.isUnitValue(): Boolean =
    this is org.jetbrains.kotlin.ir.expressions.IrGetObjectValue && type.classFqName?.asString() == "kotlin.Unit"

/** A name or a number literal: safe to repeat in an expression without a temporary. */
private val SIMPLE_VALUE = Regex("[A-Za-z_][A-Za-z0-9_.]*|-?[0-9.]+[if]?")

/** Gives every WGSL identifier a unique, legal name. */
internal class NameAllocator {
    private val used = HashSet<String>()

    /** [internal] names belong to Kernelin itself; user names never start with `kernelin`. */
    fun allocate(kotlinName: String, internal: Boolean = false): String {
        var base = kotlinName.replace(NOT_IDENTIFIER, "_").trim('_').ifEmpty { "v" }
        if (base[0].isDigit()) base = "v$base"
        if (base in WGSL_RESERVED || !internal && base.startsWith("kernelin")) base += "_k"
        var name = base
        var n = 1
        while (!used.add(name)) name = "${base}_${n++}"
        return name
    }

    private companion object {
        val NOT_IDENTIFIER = Regex("[^A-Za-z0-9_]")

        val WGSL_RESERVED = setOf(
            // keywords and types
            "alias", "array", "atomic", "bitcast", "bool", "break", "case", "const", "const_assert", "continue",
            "continuing", "default", "diagnostic", "discard", "else", "enable", "f16", "f32", "false", "fn", "for",
            "i32", "if", "let", "loop", "mat2x2", "mat3x3", "mat4x4", "override", "ptr", "requires", "return",
            "sampler", "struct", "switch", "true", "u32", "var", "vec2", "vec3", "vec4", "while", "main",
            // reserved words
            "NULL", "Self", "abstract", "active", "alignas", "alignof", "as", "asm", "async", "attribute", "auto",
            "await", "become", "cast", "catch", "class", "coherent", "common", "compile", "concept", "constexpr",
            "crate", "debugger", "decltype", "delete", "demote", "do", "enum", "explicit", "export", "extends",
            "extern", "external", "fallthrough", "filter", "final", "finally", "friend", "from", "get", "goto",
            "highp", "impl", "implements", "import", "inline", "instanceof", "interface", "layout", "lowp", "macro",
            "match", "mediump", "meta", "mod", "module", "move", "mut", "mutable", "namespace", "new", "nil",
            "noexcept", "noinline", "null", "nullptr", "of", "operator", "package", "packoffset", "partition",
            "pass", "patch", "precise", "precision", "priv", "protected", "pub", "public", "readonly", "ref",
            "regardless", "register", "require", "resource", "restrict", "self", "set", "shared", "sizeof", "smooth",
            "snorm", "static", "std", "subroutine", "super", "target", "template", "this", "throw", "trait", "try",
            "type", "typedef", "typeid", "typename", "typeof", "union", "unless", "unorm", "unsafe", "unsized",
            "use", "using", "varying", "virtual", "volatile", "wgsl", "where", "with", "writeonly", "yield",
            // predeclared address spaces, access modes, built-in values and type aliases: a helper function with one
            // of these names would shadow it at module scope (Chrome rejects `fn uniform()`: "cannot use function
            // 'uniform' as address space")
            "function", "private", "workgroup", "uniform", "storage", "handle", "read", "write", "read_write",
            "global_invocation_id", "local_invocation_id", "local_invocation_index", "workgroup_id", "num_workgroups",
            "vec2f", "vec3f", "vec4f", "vec2i", "vec3i", "vec4i", "vec2u", "vec3u", "vec4u", "vec2h", "vec3h", "vec4h",
            "mat2x2f", "mat3x3f", "mat4x4f", "texture_2d", "sampler_comparison",
            // built-in functions, so user names never shadow them
            "all", "any", "cross", "degrees", "determinant", "distance", "dot", "exp2", "faceForward", "fma", "fract",
            "frexp", "inverseSqrt", "ldexp", "mix", "modf", "normalize", "quantizeToF16", "radians", "reflect",
            "refract", "reverseBits", "saturate", "smoothstep", "step", "transpose", "countLeadingZeros",
            "countOneBits", "countTrailingZeros", "extractBits", "insertBits", "firstLeadingBit", "firstTrailingBit",
            "storageBarrier", "workgroupBarrier", "atomicAdd", "atomicLoad", "atomicStore",
            "abs", "acos", "arrayLength", "asin", "atan", "atan2", "ceil", "clamp", "cos", "cosh", "exp", "floor",
            "length", "log", "log2", "max", "min", "pow", "round", "select", "sign", "sin", "sinh", "sqrt", "tan",
            "tanh", "trunc",
        )
    }
}

/** Kernelin's random functions (in Random.kt of the runtime) and the WGSL function each one becomes. */
private val RANDOM_FUNCTIONS = mapOf(
    "randomInt" to "kernelin_random_int",
    "randomFloat" to "kernelin_random_float",
    "randomIntBelow" to "kernelin_random_below",
    "randomNormal" to "kernelin_random_normal",
)

/**
 * Kernelin's own WGSL functions for Kotlin semantics that WGSL leaves open. NaN is always checked on the bits
 * (kernelin_is_nan), because GPU compilers may assume there are no NaNs and drop a plain `x != x`.
 */
private val RUNTIME_FUNCTIONS = mapOf(
    // The random functions must give the CPU's bits: the same u32 math as Random.kt in kernelin-runtime, where
    // Int arithmetic wraps like u32 arithmetic and `ushr` is `>>` on u32.
    "kernelin_random_bits" to """
        fn kernelin_pcg(v: u32) -> u32 {
          let state = v * 747796405u + 2891336453u;
          let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
          return (word >> 22u) ^ word;
        }

        fn kernelin_random_bits(seed: i32, index: i32, stream: i32) -> u32 {
          return kernelin_pcg(bitcast<u32>(index) + kernelin_pcg(bitcast<u32>(stream) + kernelin_pcg(bitcast<u32>(seed))));
        }
    """,
    "kernelin_random_int" to """
        fn kernelin_random_int(seed: i32, index: i32, stream: i32) -> i32 {
          return bitcast<i32>(kernelin_random_bits(seed, index, stream));
        }
    """,
    // Top 24 bits times 2^-24: exact in f32, so the same Float as on the CPU.
    "kernelin_random_float" to """
        fn kernelin_random_float(seed: i32, index: i32, stream: i32) -> f32 {
          return f32(kernelin_random_bits(seed, index, stream) >> 8u) * 5.9604645e-8f;
        }
    """,
    // The high 32 bits of the 64-bit product bits * bound, from 16-bit halves (WGSL has no 64-bit integers).
    // The CPU throws for bound <= 0; the GPU can't, so it returns 0.
    "kernelin_random_below" to """
        fn kernelin_random_below(bound: i32, seed: i32, index: i32, stream: i32) -> i32 {
          if (bound <= 0i) {
            return 0i;
          }
          let a = kernelin_random_bits(seed, index, stream);
          let b = u32(bound);
          let lo_lo = (a & 0xffffu) * (b & 0xffffu);
          let hi_lo = (a >> 16u) * (b & 0xffffu);
          let lo_hi = (a & 0xffffu) * (b >> 16u);
          let hi_hi = (a >> 16u) * (b >> 16u);
          let cross = (lo_lo >> 16u) + (hi_lo & 0xffffu) + lo_hi;
          return i32(hi_hi + (hi_lo >> 16u) + (cross >> 16u));
        }
    """,
    // Box-Muller, as in Random.kt. ln, sqrt and cos make it close to the CPU, not bit-identical.
    "kernelin_random_normal" to """
        fn kernelin_random_normal(seed: i32, index: i32, stream: i32) -> f32 {
          let bits = kernelin_random_bits(seed, index, stream);
          let u1 = 1.0f - f32(bits >> 8u) * 5.9604645e-8f;
          let u2 = f32(kernelin_pcg(bits) >> 8u) * 5.9604645e-8f;
          return sqrt(-2.0f * log(u1)) * cos(u2 * 6.2831855f - 3.1415927f);
        }
    """,
    // Kotlin's min/max (Java's Math.min/max): NaN if either side is NaN, and -0f is less than 0f.
    "kernelin_min" to """
        fn kernelin_min(a: f32, b: f32) -> f32 {
          if (kernelin_is_nan(a) || kernelin_is_nan(b)) {
            return bitcast<f32>(0x7fc00000u | kernelin_zero);
          }
          if (a == b) {
            return bitcast<f32>(bitcast<u32>(a) | bitcast<u32>(b));
          }
          return select(b, a, a < b);
        }
    """,
    "kernelin_max" to """
        fn kernelin_max(a: f32, b: f32) -> f32 {
          if (kernelin_is_nan(a) || kernelin_is_nan(b)) {
            return bitcast<f32>(0x7fc00000u | kernelin_zero);
          }
          if (a == b) {
            return bitcast<f32>(bitcast<u32>(a) & bitcast<u32>(b));
          }
          return select(b, a, a > b);
        }
    """,
    // coerceIn/AtLeast/AtMost as Kotlin writes them: `if (x < min) min else if (x > max) max else x`, where a
    // comparison with NaN is false (so a NaN value or bound changes nothing). Kotlin throws when min > max.
    "kernelin_coerce_in" to """
        fn kernelin_coerce_in(x: f32, lo: f32, hi: f32) -> f32 {
          if (kernelin_is_nan(x)) {
            return x;
          }
          if (!kernelin_is_nan(lo) && x < lo) {
            return lo;
          }
          if (!kernelin_is_nan(hi) && x > hi) {
            return hi;
          }
          return x;
        }
    """,
    "kernelin_at_least" to """
        fn kernelin_at_least(x: f32, lo: f32) -> f32 {
          if (kernelin_is_nan(x) || kernelin_is_nan(lo)) {
            return x;
          }
          return select(x, lo, x < lo);
        }
    """,
    "kernelin_at_most" to """
        fn kernelin_at_most(x: f32, hi: f32) -> f32 {
          if (kernelin_is_nan(x) || kernelin_is_nan(hi)) {
            return x;
          }
          return select(x, hi, x > hi);
        }
    """,
    // Kotlin's sign (Java's Math.signum): NaN stays NaN and -0f stays -0f.
    "kernelin_sign" to """
        fn kernelin_sign(x: f32) -> f32 {
          if (kernelin_is_nan(x) || (bitcast<u32>(x) & 0x7fffffffu) == 0u) {
            return x;
          }
          return select(-1.0f, 1.0f, x > 0.0f);
        }
    """,
    // Kotlin's hypot without overflow or underflow: length(vec2(x, y)) squares x and y, so it gave Infinity for
    // 1e20f and 0 for 1e-25f. Scaling by the bigger side keeps the square near 1. Infinity wins over NaN, like Java.
    "kernelin_hypot" to """
        fn kernelin_hypot(x: f32, y: f32) -> f32 {
          let a = abs(x);
          let b = abs(y);
          if (kernelin_is_inf(a) || kernelin_is_inf(b)) {
            return bitcast<f32>(0x7f800000u | kernelin_zero);
          }
          if (kernelin_is_nan(a) || kernelin_is_nan(b)) {
            return bitcast<f32>(0x7fc00000u | kernelin_zero);
          }
          let big = max(a, b);
          if (big == 0.0f) {
            return 0.0f;
          }
          let r = min(a, b) / big;
          return big * sqrt(1.0f + r * r);
        }
    """,
    // Kotlin's log10, accurate to a few ULPs. log(x) * log10(e) loses most digits near x = 1 (where WGSL's log
    // has only an absolute error bound), so split x = m * 2^e with m in [0.707, 1.414) and compute
    // ln(m) = 2 atanh((m - 1) / (m + 1)) as a short series; log10(2) is split in two so e * log10(2) is exact.
    "kernelin_log10" to """
        fn kernelin_log10(x: f32) -> f32 {
          if (kernelin_is_nan(x) || x < 0.0f) {
            return bitcast<f32>(0x7fc00000u | kernelin_zero);
          }
          if (x == 0.0f) {
            return bitcast<f32>(0xff800000u | kernelin_zero);
          }
          if (kernelin_is_inf(x)) {
            return x;
          }
          let parts = frexp(x);
          var m = parts.fract;
          var e = parts.exp;
          if (m < 0.70710677f) {
            m = m * 2.0f;
            e = e - 1i;
          }
          let t = (m - 1.0f) / (m + 1.0f);
          let t2 = t * t;
          let ln_m = 2.0f * t * (1.0f + t2 * (0.33333334f + t2 * (0.2f + t2 * (0.14285715f + t2 * (0.11111111f + t2 * 0.09090909f)))));
          let fe = f32(e);
          return fe * 0.30102539f + (fe * 4.6050391e-6f + ln_m * 0.43429449f);
        }
    """,
    // Float.compareTo (Java's Float.compare): a total order. Every NaN becomes the one NaN Java uses, then the
    // bits of a negative float are flipped (except the sign), so comparing them as Ints gives the float order.
    "kernelin_compare" to """
        fn kernelin_float_order(x: f32) -> i32 {
          var bits = bitcast<i32>(x);
          if (kernelin_is_nan(x)) {
            bits = 0x7fc00000i;
          }
          return bits ^ ((bits >> 31u) & 0x7fffffffi);
        }

        fn kernelin_compare(a: f32, b: f32) -> i32 {
          let x = kernelin_float_order(a);
          let y = kernelin_float_order(b);
          return select(select(0i, 1i, x > y), -1i, x < y);
        }
    """,
)
