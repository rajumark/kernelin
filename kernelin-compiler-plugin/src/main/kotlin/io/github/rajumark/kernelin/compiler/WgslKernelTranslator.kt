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

internal const val WORKGROUP_SIZE = 64

internal enum class WType(val wgsl: String, val kotlin: String) {
    I32("i32", "Int"),
    F32("f32", "Float"),
    BOOL("bool", "Boolean"),
    UNIT("", "Unit"),
}

internal class KernelError(val element: IrElement, val message: String, val file: IrFile?)

internal class BufferParam(val name: String, val elementType: WType, val binding: Int) {
    var written = false
}

internal class ScalarParam(val name: String, val type: WType)

internal class TranslationResult(val shader: WgslShader?, val errors: List<KernelError>)

/** A translated kernel: WGSL source plus the information a GPU host needs to run it. */
internal class WgslShader(
    val wgsl: String,
    /** Kernel parameters in Kotlin order; each is a [BufferParam] or a [ScalarParam]. */
    val params: List<Any>,
) {
    val buffers get() = params.filterIsInstance<BufferParam>()
    val scalars get() = params.filterIsInstance<ScalarParam>()

    fun manifestJson(fqName: String): String = buildString {
        appendLine("{")
        appendLine("  \"kernel\": \"$fqName\",")
        appendLine("  \"entryPoint\": \"main\",")
        appendLine("  \"workgroupSize\": $WORKGROUP_SIZE,")
        appendLine("  \"uniform\": { \"binding\": 0, \"fields\": [")
        val fields = listOf("{ \"name\": \"kernelin_size\", \"type\": \"u32\" }") +
            scalars.map { "{ \"name\": \"${it.name}\", \"type\": \"${it.type.wgsl}\" }" }
        appendLine(fields.joinToString(",\n") { "    $it" })
        appendLine("  ] },")
        appendLine("  \"params\": [")
        appendLine(params.joinToString(",\n") { p ->
            when (p) {
                is BufferParam -> "    { \"name\": \"${p.name}\", \"kind\": \"buffer\", \"type\": \"${p.elementType.wgsl}\", " +
                    "\"binding\": ${p.binding}, \"access\": \"${if (p.written) "read_write" else "read"}\" }"
                is ScalarParam -> "    { \"name\": \"${p.name}\", \"kind\": \"scalar\", \"type\": \"${p.type.wgsl}\" }"
                else -> error("unexpected param $p")
            }
        })
        appendLine("  ]")
        appendLine("}")
    }
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
    private val params = mutableListOf<Any>()
    private val helpers = LinkedHashMap<IrSimpleFunction, Helper>()
    private val helperStack = ArrayDeque<IrSimpleFunction>()

    private class Helper(val name: String, var code: String = "")

    private data class Expr(val code: String, val type: WType)

    fun translate(): TranslationResult {
        checkKernelSignature()
        val body = if (errors.isEmpty()) BodyTranslator(kernel, isKernel = true).translate() else ""
        if (errors.isNotEmpty()) return TranslationResult(null, errors.distinctBy { it.element to it.message })

        val wgsl = buildString {
            appendLine("// Generated by Kernelin from ${kernel.kotlinFqName} - do not edit.")
            appendLine()
            appendLine("struct KernelinParams {")
            appendLine("  kernelin_size: u32,")
            for (s in scalars.values) appendLine("  ${s.name}: ${s.type.wgsl},")
            appendLine("}")
            appendLine()
            appendLine("@group(0) @binding(0) var<uniform> kernelin_params: KernelinParams;")
            for (b in buffers.values) {
                val access = if (b.written) "read_write" else "read"
                appendLine("@group(0) @binding(${b.binding}) var<storage, $access> ${b.name}: array<${b.elementType.wgsl}>;")
            }
            appendLine()
            appendLine("var<private> kernelin_id: u32;")
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
                scalar == WType.I32 || scalar == WType.F32 -> ScalarParam(name, scalar).also { scalars[p] = it; params += it }
                scalar == WType.BOOL -> error(p, "Boolean kernel parameters aren't supported yet; pass an Int (0 or 1)")
                else -> error(p, "parameter '${p.name}' has type ${p.type.describe()}; kernels accept FloatArray, IntArray, Int and Float${p.type.suggestion()}")
            }
        }
    }

    /** Translates the body of the kernel or of one helper function. */
    private inner class BodyTranslator(private val function: IrSimpleFunction, private val isKernel: Boolean) {
        private val out = StringBuilder()
        private var indent = 1
        private val locals = HashMap<IrValueDeclaration, Expr>()
        private val arrayAliases = HashMap<IrValueDeclaration, BufferParam>()
        private val loops = ArrayDeque<IrLoop>()

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
                is IrBreak -> { checkLoopTarget(s); line("break;") }
                is IrContinue -> { checkLoopTarget(s); line("continue;") }
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
            if (value.type != WType.UNIT) line("_ = ${value.code};")
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
            expressionStatement(call)
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

        private fun whileLoop(loop: IrWhileLoop) {
            loops.addLast(loop)
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
            loops.removeLast()
        }

        private fun doWhileLoop(loop: IrDoWhileLoop) {
            loops.addLast(loop)
            block("loop") {
                loop.body?.let(::statement)
                val pre = mutableListOf<String>()
                val cond = expr(loop.condition, pre)
                block("continuing") {
                    pre.forEach(::line)
                    line("break if !(${cast(cond, WType.BOOL)});")
                }
            }
            loops.removeLast()
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

        private fun checkLoopTarget(jump: IrBreakContinue) {
            if (loops.lastOrNull() != jump.loop) {
                fail(jump, "break/continue to an outer loop label isn't supported yet")
            }
        }

        private fun returnStatement(r: IrReturn) {
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
            block("") {
                pre.forEach(::line)
                val end = names.allocate("kernelin_end", internal = true)
                line("let $end: i32 = ${range.end};")
                val step = if (range.step.isIntLiteral()) range.step else names.allocate("kernelin_step", internal = true).also {
                    line("let $it: i32 = ${range.step};")
                }
                val cmp = when {
                    range.descending -> if (range.inclusive) ">=" else ">"
                    else -> if (range.inclusive) "<=" else "<"
                }
                val update = if (range.descending) "$i - $step" else "$i + $step"
                loops.addLast(whileLoop)
                block("for (var $i: i32 = ${range.start}; $i $cmp $end; $i = $update)") {
                    whileBody.statements.drop(1).forEach(::statement)
                }
                loops.removeLast()
            }
        }

        private inner class Range(
            val start: String,
            val end: String,
            val step: String,
            val inclusive: Boolean,
            val descending: Boolean,
        )

        private fun range(e: IrExpression, pre: MutableList<String>): Range? {
            val call = e as? IrCall ?: return null.also { fail(e, "for loops must iterate over an Int range like 0 until n") }
            val fn = call.symbol.owner
            val name = fn.name.asString()
            fun int(arg: IrExpression) = cast(expr(arg, pre), WType.I32)
            return when (name) {
                "until", "rangeUntil" -> Range(int(call.receiver()!!), int(call.regularArgs()[0]), "1i", inclusive = false, descending = false)
                "rangeTo" -> Range(int(call.receiver()!!), int(call.regularArgs()[0]), "1i", inclusive = true, descending = false)
                "downTo" -> Range(int(call.receiver()!!), int(call.regularArgs()[0]), "1i", inclusive = true, descending = true)
                "<get-indices>" -> {
                    val buffer = arrayBuffer(call.receiver()!!)
                        ?: return null.also { fail(call, "indices must be read from a kernel array parameter") }
                    Range("0i", "i32(arrayLength(&${buffer.name}))", "1i", inclusive = false, descending = false)
                }
                "step" -> {
                    val inner = range(call.receiver()!!, pre) ?: return null
                    Range(inner.start, inner.end, int(call.regularArgs()[0]), inner.inclusive, inner.descending)
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
            is Int -> Expr(if (v == Int.MIN_VALUE) "(-2147483647i - 1i)" else "${v}i", WType.I32)
            is Float -> if (v.isFinite()) Expr(floatLiteral(v), WType.F32) else fail(c, "NaN and infinite Float literals aren't supported")
            is Boolean -> Expr(v.toString(), WType.BOOL)
            is Double -> fail(c, "Double isn't supported on most GPUs; use Float (add an f suffix: 1.0f)")
            is Long -> fail(c, "Long isn't supported on the GPU; use Int")
            else -> fail(c, "this constant type isn't supported on the GPU")
        }

        private fun getValue(e: IrGetValue): Expr {
            val owner = e.symbol.owner
            locals[owner]?.let { return it }
            scalars[owner]?.let { return Expr("kernelin_params.${it.name}", it.type) }
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
                val l = expr(b[0].condition, pre)
                val r = expr(b[0].result, pre)
                return Expr("(${cast(l, WType.BOOL)} && ${cast(r, WType.BOOL)})", WType.BOOL)
            }
            if (w.origin == IrStatementOrigin.OROR && b.size == 2) {
                // a || b  ==  if (a) true else b
                val l = expr(b[0].condition, pre)
                val r = expr(b[1].result, pre)
                return Expr("(${cast(l, WType.BOOL)} || ${cast(r, WType.BOOL)})", WType.BOOL)
            }
            val type = scalarType(w.type)
            if (type == null || type == WType.UNIT) return fail(w, "this if/when must produce an Int, Float or Boolean value")
            val elseBranch = b.lastOrNull()?.takeIf { it is IrElseBranch || it.condition.isTrueConst() }
                ?: return fail(w, "an if/when used as a value needs an else branch")
            // Both sides of select() are evaluated, which is fine because kernel expressions have no side effects.
            var result = cast(expr(elseBranch.result, pre), type)
            for (branch in b.dropLast(1).asReversed()) {
                val cond = cast(expr(branch.condition, pre), WType.BOOL)
                val value = cast(expr(branch.result, pre), type)
                result = "select($result, $value, $cond)"
            }
            return Expr(result, type)
        }

        private fun call(call: IrCall, pre: MutableList<String>): Expr {
            val fn = call.symbol.owner
            val name = fn.name.asString()
            val owner = fn.parentClassFq() ?: fn.packageFq()
            val resultType = scalarType(call.type)

            if (owner == "io.github.rajumark.kernelin" && name == "globalId") return Expr("i32(kernelin_id)", WType.I32)

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

            if (owner == "kotlin.Int" || owner == "kotlin.Float" || owner == "kotlin.Boolean") {
                return primitiveMember(call, name, pre)
            }
            if (owner == "kotlin.math") return mathFunction(call, name, pre)
            if (owner == "kotlin.ranges" && name in setOf("coerceIn", "coerceAtLeast", "coerceAtMost")) {
                val t = resultType ?: return fail(call, "unsupported coerce type")
                val x = cast(expr(call.receiver()!!, pre), t)
                val a = call.regularArgs().map { cast(expr(it, pre), t) }
                return when (name) {
                    "coerceIn" -> Expr("clamp($x, ${a[0]}, ${a[1]})", t)
                    "coerceAtLeast" -> Expr("max($x, ${a[0]})", t)
                    else -> Expr("min($x, ${a[0]})", t)
                }
            }
            return helperCall(call, fn, pre)
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
                "shl" -> Expr("(${receiver.code} << u32(${args[0].code}))", WType.I32)
                "shr" -> Expr("(${receiver.code} >> u32(${args[0].code}))", WType.I32)
                "ushr" -> Expr("i32(u32(${receiver.code}) >> u32(${args[0].code}))", WType.I32)
                "toFloat", "toInt" -> Expr(cast(receiver, resultType), resultType)
                "compareTo" -> {
                    val (l, r) = promote(receiver, args[0])
                    Expr("select(select(0i, 1i, ${l.code} > ${r.code}), -1i, ${l.code} < ${r.code})", WType.I32)
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
                "sign" -> Expr("sign(${t(0)})", resultType)
                "min" -> Expr("min(${t(0)}, ${t(1)})", resultType)
                "max" -> Expr("max(${t(0)}, ${t(1)})", resultType)
                "atan2" -> Expr("atan2(${f(0)}, ${f(1)})", WType.F32)
                "pow" -> Expr("pow(${f(0)}, ${f(1)})", WType.F32)
                "hypot" -> Expr("length(vec2<f32>(${f(0)}, ${f(1)}))", WType.F32)
                "log10" -> Expr("(log(${f(0)}) * 0.4342944819f)", WType.F32)
                else -> fail(call, "kotlin.math.$name isn't supported on the GPU yet")
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
            val helper = helpers[fn] ?: translateHelper(call, fn) ?: return Expr("0", scalarType(call.type) ?: WType.I32)
            val args = fn.parameters.mapIndexed { i, p ->
                val arg = call.arguments[i] ?: return fail(call, "default arguments aren't supported in kernel helpers yet")
                cast(expr(arg, pre), scalarType(p.type) ?: WType.I32)
            }
            val type = scalarType(fn.returnType) ?: WType.I32
            return Expr("${helper.name}(${args.joinToString(", ")})", type)
        }

        private fun translateHelper(call: IrCall, fn: IrSimpleFunction): Helper? {
            val returnType = scalarType(fn.returnType)
            var ok = returnType != null
            if (returnType == null) fail(call, "helper '${fn.name}' returns ${fn.returnType.describe()}; helpers must return Int, Float, Boolean or Unit")
            if (fn.typeParameters.isNotEmpty()) { ok = false; fail(call, "helper '${fn.name}' can't have type parameters") }
            val paramTypes = fn.parameters.map { p ->
                val t = scalarType(p.type)
                if (p.kind == IrParameterKind.DispatchReceiver || p.kind == IrParameterKind.Context) {
                    ok = false; fail(call, "helper '${fn.name}' must be a top-level function"); null
                } else if (t == null || t == WType.UNIT) {
                    ok = false; fail(call, "helper '${fn.name}' parameter '${p.name}' must be Int, Float or Boolean (arrays can't be passed yet)"); null
                } else t
            }
            if (!ok) return null

            val helper = Helper(names.allocate(fn.name.asString()))
            helperStack.addLast(fn)
            val translator = BodyTranslator(fn, isKernel = false)
            val paramDecls = fn.parameters.mapIndexed { i, p ->
                val pname = names.allocate(if (p.kind == IrParameterKind.ExtensionReceiver) "receiver" else p.name.asString())
                translator.bindParameter(p, pname, paramTypes[i]!!)
                "$pname: ${paramTypes[i]!!.wgsl}"
            }
            val body = translator.translate()
            helperStack.removeLast()
            val ret = if (returnType == WType.UNIT) "" else " -> ${returnType!!.wgsl}"
            helper.code = "fn ${helper.name}(${paramDecls.joinToString(", ")})$ret {\n$body}\n"
            helpers[fn] = helper
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

private fun String.isIntLiteral() = matches(Regex("-?\\d+i"))

/** Gives every WGSL identifier a unique, legal name. */
internal class NameAllocator {
    private val used = HashSet<String>()

    /** [internal] names belong to Kernelin itself; user names never start with `kernelin`. */
    fun allocate(kotlinName: String, internal: Boolean = false): String {
        var base = kotlinName.replace(Regex("[^A-Za-z0-9_]"), "_").trim('_').ifEmpty { "v" }
        if (base[0].isDigit()) base = "v$base"
        if (base in WGSL_RESERVED || !internal && base.startsWith("kernelin")) base += "_k"
        var name = base
        var n = 1
        while (!used.add(name)) name = "${base}_${n++}"
        return name
    }

    private companion object {
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
            // built-in functions Kernelin emits, so user names never shadow them
            "abs", "acos", "arrayLength", "asin", "atan", "atan2", "ceil", "clamp", "cos", "cosh", "exp", "floor",
            "length", "log", "log2", "max", "min", "pow", "round", "select", "sign", "sin", "sinh", "sqrt", "tan",
            "tanh", "trunc",
        )
    }
}
