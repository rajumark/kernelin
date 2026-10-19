package io.github.rajumark.kernelin.compiler

import org.jetbrains.kotlin.KtFakeSourceElementKind
import org.jetbrains.kotlin.KtSourceElement
import org.jetbrains.kotlin.diagnostics.AbstractSourceElementPositioningStrategy
import org.jetbrains.kotlin.diagnostics.DiagnosticReporter
import org.jetbrains.kotlin.diagnostics.KtDiagnosticFactoryToRendererMap
import org.jetbrains.kotlin.diagnostics.KtDiagnosticsContainer
import org.jetbrains.kotlin.diagnostics.SourceElementPositioningStrategies
import org.jetbrains.kotlin.diagnostics.error1
import org.jetbrains.kotlin.diagnostics.rendering.BaseDiagnosticRendererFactory
import org.jetbrains.kotlin.diagnostics.rendering.CommonRenderers
import org.jetbrains.kotlin.diagnostics.reportOn
import org.jetbrains.kotlin.fir.FirElement
import org.jetbrains.kotlin.fir.FirSession
import org.jetbrains.kotlin.fir.analysis.checkers.MppCheckerKind
import org.jetbrains.kotlin.fir.analysis.checkers.context.CheckerContext
import org.jetbrains.kotlin.fir.analysis.checkers.declaration.DeclarationCheckers
import org.jetbrains.kotlin.fir.analysis.checkers.declaration.FirDeclarationChecker
import org.jetbrains.kotlin.fir.analysis.checkers.expression.ExpressionCheckers
import org.jetbrains.kotlin.fir.analysis.checkers.expression.FirExpressionChecker
import org.jetbrains.kotlin.fir.analysis.extensions.FirAdditionalCheckersExtension
import org.jetbrains.kotlin.fir.declarations.DirectDeclarationsAccess
import org.jetbrains.kotlin.fir.declarations.FirDeclarationOrigin
import org.jetbrains.kotlin.fir.declarations.FirFile
import org.jetbrains.kotlin.fir.declarations.FirResolvePhase
import org.jetbrains.kotlin.fir.declarations.FirNamedFunction
import org.jetbrains.kotlin.fir.declarations.FirProperty
import org.jetbrains.kotlin.fir.declarations.hasAnnotation
import org.jetbrains.kotlin.fir.declarations.utils.isSuspend
import org.jetbrains.kotlin.fir.expressions.FirAnnotation
import org.jetbrains.kotlin.fir.expressions.FirCheckNotNullCall
import org.jetbrains.kotlin.fir.expressions.FirElvisExpression
import org.jetbrains.kotlin.fir.expressions.FirFunctionCall
import org.jetbrains.kotlin.fir.expressions.FirLiteralExpression
import org.jetbrains.kotlin.fir.expressions.FirSafeCallExpression
import org.jetbrains.kotlin.fir.expressions.FirStatement
import org.jetbrains.kotlin.fir.expressions.FirStringConcatenationCall
import org.jetbrains.kotlin.fir.expressions.FirThrowExpression
import org.jetbrains.kotlin.fir.expressions.FirTryExpression
import org.jetbrains.kotlin.fir.expressions.FirTypeOperatorCall
import org.jetbrains.kotlin.fir.extensions.FirExtensionRegistrar
import org.jetbrains.kotlin.fir.references.toResolvedCallableSymbol
import org.jetbrains.kotlin.fir.resolve.providers.firProvider
import org.jetbrains.kotlin.fir.symbols.SymbolInternals
import org.jetbrains.kotlin.fir.symbols.lazyResolveToPhase
import org.jetbrains.kotlin.fir.symbols.impl.FirConstructorSymbol
import org.jetbrains.kotlin.fir.symbols.impl.FirNamedFunctionSymbol
import org.jetbrains.kotlin.fir.types.ConeKotlinType
import org.jetbrains.kotlin.fir.types.classId
import org.jetbrains.kotlin.fir.types.coneType
import org.jetbrains.kotlin.fir.types.functionTypeKind
import org.jetbrains.kotlin.fir.types.isMarkedNullable
import org.jetbrains.kotlin.fir.visitors.FirVisitorVoid
import org.jetbrains.kotlin.name.ClassId
import org.jetbrains.kotlin.name.FqName
import org.jetbrains.kotlin.name.Name
import org.jetbrains.kotlin.psi.KtElement
import org.jetbrains.kotlin.types.ConstantValueKind

/*
 * Frontend (K2/FIR) checks for @Kernel functions. They run while you type in IntelliJ IDEA and Android Studio,
 * so code the GPU can't run is underlined right away, not only when you build.
 *
 * They catch the common mistakes early and only report what the WGSL translator (which runs later, on IR) would
 * reject anyway, so valid code is never blocked. Everything else is still checked by the translator. The same
 * checks run on the helper functions a kernel calls (see the end of this file).
 */

private val KERNEL_CLASS_ID = ClassId.topLevel(KERNEL_ANNOTATION)

object KernelinErrors : KtDiagnosticsContainer() {
    /** The one Kernelin diagnostic; the argument is the message, same wording as the translator's errors. */
    val KERNELIN_ERROR by error1<KtElement, String>()

    override fun getRendererFactory(): BaseDiagnosticRendererFactory = Renderers

    private object Renderers : BaseDiagnosticRendererFactory() {
        override val MAP by KtDiagnosticFactoryToRendererMap("Kernelin") { map ->
            map.put(KERNELIN_ERROR, "Kernelin: {0}", CommonRenderers.STRING)
        }
    }
}

class KernelinFirExtensionRegistrar : FirExtensionRegistrar() {
    override fun ExtensionRegistrarContext.configurePlugin() {
        +::KernelinCheckers
    }
}

class KernelinCheckers(session: FirSession) : FirAdditionalCheckersExtension(session) {
    override val declarationCheckers = object : DeclarationCheckers() {
        override val namedFunctionCheckers = setOf(KernelSignatureChecker, KernelHelperCallChecker, HelperFunctionChecker)
        override val propertyCheckers = setOf(KernelVariableChecker)
    }

    override val expressionCheckers = object : ExpressionCheckers() {
        override val basicExpressionCheckers = setOf(KernelExpressionChecker)
        override val functionCallCheckers = setOf(KernelCallChecker)
    }
}

/** The @Kernel function we're inside of, if any (also inside a `repeat { }` lambda in it). */
private fun CheckerContext.enclosingKernel(): FirNamedFunctionSymbol? =
    containingDeclarations.asReversed().firstNotNullOfOrNull { symbol ->
        (symbol as? FirNamedFunctionSymbol)?.takeIf { it.hasAnnotation(KERNEL_CLASS_ID, session) }
    }

context(context: CheckerContext, reporter: DiagnosticReporter)
private fun report(element: FirElement, message: String, kernel: FirNamedFunctionSymbol) {
    reporter.reportOn(element.source, KernelinErrors.KERNELIN_ERROR, "$message (in @Kernel ${kernel.name})")
}

private val SCALARS = setOf("kotlin.Int", "kotlin.Float", "kotlin.Boolean")
private val ARRAYS = setOf("kotlin.FloatArray", "kotlin.IntArray")

private fun ConeKotlinType.fq(): String? = classId?.asSingleFqName()?.asString()

private fun ConeKotlinType.describe(): String =
    (classId?.shortClassName?.asString() ?: "this type") + if (isMarkedNullable) "?" else ""

private fun ConeKotlinType.suggestion(): String = when (fq()) {
    "kotlin.Double" -> " (Double isn't supported on most GPUs; use Float)"
    "kotlin.Long" -> " (Long isn't supported on the GPU; use Int)"
    "kotlin.String" -> " (strings aren't supported on the GPU)"
    else -> ""
}

/** Return type, type parameters, receivers and parameter types of the @Kernel function itself. */
object KernelSignatureChecker : FirDeclarationChecker<FirNamedFunction>(MppCheckerKind.Common) {
    context(context: CheckerContext, reporter: DiagnosticReporter)
    override fun check(declaration: FirNamedFunction) {
        if (!declaration.hasAnnotation(KERNEL_CLASS_ID, context.session)) return
        val kernel = declaration.symbol
        val name = declaration.source
        if (declaration.returnTypeRef.coneType.fq() != "kotlin.Unit") {
            reporter.reportOn(
                name, KernelinErrors.KERNELIN_ERROR,
                "a @Kernel function must return Unit; write results into an array parameter instead (in @Kernel ${kernel.name})",
                SourceElementPositioningStrategies.DECLARATION_NAME,
            )
        }
        if (declaration.typeParameters.isNotEmpty()) report(declaration.typeParameters.first(), "a @Kernel function can't have type parameters", kernel)
        if (declaration.isSuspend) {
            reporter.reportOn(name, KernelinErrors.KERNELIN_ERROR, "a @Kernel function can't be suspend (in @Kernel ${kernel.name})",
                SourceElementPositioningStrategies.DECLARATION_NAME)
        }
        val receiver = declaration.receiverParameter
        if (receiver != null) report(receiver, "a @Kernel function must be a top-level function without receivers", kernel)
        if (declaration.contextParameters.isNotEmpty()) report(declaration.contextParameters.first(), "a @Kernel function must be a top-level function without receivers", kernel)
        if (context.containingDeclarations.any { it !is org.jetbrains.kotlin.fir.symbols.impl.FirFileSymbol }) {
            reporter.reportOn(name, KernelinErrors.KERNELIN_ERROR, "a @Kernel function must be a top-level function without receivers (in @Kernel ${kernel.name})",
                SourceElementPositioningStrategies.DECLARATION_NAME)
        }
        for (p in declaration.valueParameters) {
            val type = p.returnTypeRef.coneType
            val fq = type.fq()
            if (!type.isMarkedNullable && (fq in SCALARS || fq in ARRAYS)) continue
            report(p, "parameter '${p.name}' has type ${type.describe()}; kernels accept FloatArray, IntArray, Int, Float and Boolean${type.suggestion()}", kernel)
        }
    }
}

/** Local variables in a kernel must be Int, Float or Boolean (or a FloatArray/IntArray alias of a parameter). */
object KernelVariableChecker : FirDeclarationChecker<FirProperty>(MppCheckerKind.Common) {
    context(context: CheckerContext, reporter: DiagnosticReporter)
    override fun check(declaration: FirProperty) {
        val kernel = context.enclosingKernel() ?: return
        val message = variableError(declaration) ?: return
        reporter.reportOn(
            declaration.source, KernelinErrors.KERNELIN_ERROR, "$message (in @Kernel ${kernel.name})",
            SourceElementPositioningStrategies.DECLARATION_NAME,
        )
    }
}

private fun variableError(declaration: FirProperty): String? {
    if (!declaration.isLocal) return null
    // Variables the compiler made up (the iterator of a for loop, ...) aren't the user's to fix.
    if (declaration.source?.kind is KtFakeSourceElementKind) return null
    val type = declaration.returnTypeRef.coneType
    val fq = type.fq()
    if (!type.isMarkedNullable && (fq in SCALARS || fq in ARRAYS)) return null
    return "variable '${declaration.name}' has type ${type.describe()}; only Int, Float and Boolean are supported${type.suggestion()}"
}

/** Expressions that never work on the GPU: strings, Double/Long constants, exceptions, casts, nullable values. */
object KernelExpressionChecker : FirExpressionChecker<FirStatement>(MppCheckerKind.Common) {
    context(context: CheckerContext, reporter: DiagnosticReporter)
    override fun check(expression: FirStatement) {
        val kernel = context.enclosingKernel() ?: return
        val message = expressionError(expression, context.containingElements) ?: return
        report(expression, message, kernel)
    }
}

/** What's wrong with [expression] on the GPU, or null. [elements] lead from the declaration to [expression]. */
private fun expressionError(expression: FirStatement, elements: List<FirElement>): String? {
    if (expression.source?.kind is KtFakeSourceElementKind) return null
    return when (expression) {
        is FirStringConcatenationCall -> "strings aren't supported on the GPU"
        is FirLiteralExpression -> literalError(expression, elements)
        is FirThrowExpression -> "throwing exceptions isn't supported on the GPU"
        is FirTryExpression -> "try/catch isn't supported on the GPU"
        is FirTypeOperatorCall -> "type checks and casts (is/as) aren't supported on the GPU"
        is FirSafeCallExpression, is FirElvisExpression, is FirCheckNotNullCall -> "nullable values aren't supported on the GPU"
        else -> null
    }
}

private fun literalError(literal: FirLiteralExpression, elements: List<FirElement>): String? {
    // Annotation arguments like @Suppress("...") aren't kernel code.
    if (elements.any { it is FirAnnotation }) return null
    return when (literal.kind) {
        ConstantValueKind.String -> if (elements.any { it is FirStringConcatenationCall }) null
            else "strings aren't supported on the GPU"
        // A Double constant converted right away (2.0.toFloat()) is fine; the translator folds it.
        ConstantValueKind.Double -> if (isConvertedRightAway(literal, elements)) null
            else "Double isn't supported on most GPUs; use Float (add an f suffix: 1.0f)"
        ConstantValueKind.Long -> if (isConvertedRightAway(literal, elements)) null else "Long isn't supported on the GPU; use Int"
        else -> null
    }
}

private fun isConvertedRightAway(literal: FirLiteralExpression, elements: List<FirElement>): Boolean {
    val parent = elements.getOrNull(elements.lastIndex - 1)
    // Also skip `val d = 1.0`: the variable is already reported, one error per mistake is enough.
    if (parent is FirProperty) return true
    return parent is FirFunctionCall && parent.explicitReceiver === literal &&
        parent.calleeReference.name.asString().let { it == "toFloat" || it == "toInt" }
}

private val KOTLIN = FqName("kotlin")
private val COLLECTIONS = FqName("kotlin.collections")
private val CONTAINS = Name.identifier("contains")
private val ALLOWED_LIBRARY_PACKAGES = setOf(KOTLIN, FqName("kotlin.math"), FqName("kotlin.ranges"), KERNEL_ANNOTATION.parent())

/** Top-level functions in package kotlin that the GPU can't run (scope functions, preconditions, array builders). */
private val REJECTED_KOTLIN_FUNCTIONS = setOf(
    "let", "also", "apply", "run", "with", "takeIf", "takeUnless", "error", "require", "requireNotNull", "check",
    "checkNotNull", "TODO", "lazy", "arrayOf", "floatArrayOf", "intArrayOf", "booleanArrayOf", "doubleArrayOf",
    "longArrayOf", "emptyArray", "arrayOfNulls", "to", "synchronized", "assert", "runCatching",
).mapTo(HashSet(), Name::identifier)

/** Calls: other kernels, object creation, library functions, and functions taking lambdas (except `repeat`). */
object KernelCallChecker : FirExpressionChecker<FirFunctionCall>(MppCheckerKind.Common) {
    context(context: CheckerContext, reporter: DiagnosticReporter)
    override fun check(expression: FirFunctionCall) {
        val kernel = context.enclosingKernel() ?: return
        val message = callError(expression, context.session) ?: return
        report(expression, message, kernel)
    }
}

private fun callError(expression: FirFunctionCall, session: FirSession): String? {
    if (expression.source?.kind is KtFakeSourceElementKind) return null
    val callee = expression.calleeReference.toResolvedCallableSymbol() ?: return null
    return when {
        callee is FirConstructorSymbol ->
            "creating objects or arrays isn't supported in kernels; pass arrays in as kernel parameters"
        callee.hasAnnotation(KERNEL_CLASS_ID, session) -> "a kernel can't call another @Kernel function"
        callee !is FirNamedFunctionSymbol -> null
        callee.origin == FirDeclarationOrigin.Source -> lambdaError(callee, session)
        else -> libraryError(callee, session)
    }
}

private fun lambdaError(callee: FirNamedFunctionSymbol, session: FirSession): String? =
    if (takesLambda(callee, session)) "lambdas aren't supported in kernels yet ('${callee.name}' takes one); use a for loop" else null

private fun takesLambda(callee: FirNamedFunctionSymbol, session: FirSession) =
    callee.valueParameterSymbols.any { it.resolvedReturnType.functionTypeKind(session) != null }

private fun libraryError(callee: FirNamedFunctionSymbol, session: FirSession): String? {
    val id = callee.callableId
    val pkg = id.packageName
    val rejected = when {
        // `x in 10 downTo 0` resolves to Iterable.contains; the translator handles it.
        pkg == COLLECTIONS && callee.name == CONTAINS -> false
        pkg !in ALLOWED_LIBRARY_PACKAGES -> true
        pkg == KOTLIN && id.classId == null -> callee.name in REJECTED_KOTLIN_FUNCTIONS
        else -> false
    }
    if (!rejected) return null
    return lambdaError(callee, session)
        ?: "'${id.asSingleFqName()}' can't run on the GPU (only kotlin.math, Int/Float operators and your own functions are supported)"
}

// ------------------------------------------------------------------------------------------------ helpers

/*
 * A kernel's helper functions (your own functions it calls, and the ones they call) run on the GPU too, so the same
 * checks apply to them. A helper in the kernel's file gets the errors on its own code. A helper in another file
 * gets one error on the call in the kernel, because the IDE shows a file's errors only while checking that file.
 */

/** Something the GPU can't run, found in [helper]. */
private class HelperProblem(
    val source: KtSourceElement?,
    val message: String,
    val helper: FirNamedFunctionSymbol,
    val file: FirFile?,
    val positioning: AbstractSourceElementPositioningStrategy = SourceElementPositioningStrategies.DEFAULT,
)

/** The helper a call runs on the GPU: a function from this module's sources, or null for anything else. */
private fun helperCallee(call: FirFunctionCall, session: FirSession): FirNamedFunctionSymbol? {
    val callee = call.calleeReference.toResolvedCallableSymbol() as? FirNamedFunctionSymbol ?: return null
    if (callee.origin != FirDeclarationOrigin.Source || callee.callableId.isLocal) return null
    // These calls are errors of their own (reported by KernelCallChecker); their bodies don't matter.
    if (callee.hasAnnotation(KERNEL_CLASS_ID, session) || takesLambda(callee, session)) return null
    return callee
}

/**
 * Walks helper functions, each once (so recursion can't loop; the translator reports it at build time).
 * With [findProblems] false it only follows calls, to see which helpers a kernel reaches.
 */
@OptIn(SymbolInternals::class)
private class HelperScanner(private val session: FirSession, private val findProblems: Boolean = true) {
    val problems = mutableListOf<HelperProblem>()
    val reached = LinkedHashSet<FirNamedFunctionSymbol>()

    /** Follows the helper calls in a kernel's [body] (the kernel's own code is checked by the checkers above). */
    fun scanCallsOf(body: FirElement) = body.accept(Walker(null, null))

    /** Checks [helper] and, when [follow] is true, every helper it calls. */
    fun scanHelper(helper: FirNamedFunctionSymbol, follow: Boolean = true) {
        if (!reached.add(helper)) return
        helper.lazyResolveToPhase(FirResolvePhase.BODY_RESOLVE)
        val declaration = helper.fir
        val file = if (findProblems) session.firProvider.getFirCallableContainerFile(helper) else null
        if (findProblems) signatureProblems(declaration, file)
        declaration.body?.accept(Walker(helper, file).also { it.follow = follow })
    }

    private fun signatureProblems(f: FirNamedFunction, file: FirFile?) {
        val helper = f.symbol
        fun problem(source: KtSourceElement?, message: String, positioning: AbstractSourceElementPositioningStrategy = SourceElementPositioningStrategies.DEFAULT) {
            problems += HelperProblem(source, message, helper, file, positioning)
        }
        val returnType = f.returnTypeRef.coneType
        if (returnType.isMarkedNullable || returnType.fq() !in SCALARS + "kotlin.Unit") {
            problem(f.source, "helper '${f.name}' returns ${returnType.describe()}; helpers must return Int, Float, Boolean or Unit",
                SourceElementPositioningStrategies.DECLARATION_NAME)
        }
        if (f.typeParameters.isNotEmpty()) problem(f.typeParameters.first().source, "helper '${f.name}' can't have type parameters")
        if (f.dispatchReceiverType != null || f.contextParameters.isNotEmpty()) {
            problem(f.source, "helper '${f.name}' must be a top-level function", SourceElementPositioningStrategies.DECLARATION_NAME)
        }
        val params = listOfNotNull(f.receiverParameter?.let { it.source to (it.typeRef.coneType to "receiver") }) +
            f.valueParameters.map { it.source to (it.returnTypeRef.coneType to it.name.asString()) }
        for ((source, typeAndName) in params) {
            val (type, name) = typeAndName
            if (!type.isMarkedNullable && (type.fq() in SCALARS || type.fq() in ARRAYS)) continue
            problem(source, "helper '${f.name}' parameter '$name' must be Int, Float, Boolean, FloatArray or IntArray")
        }
    }

    private inner class Walker(private val helper: FirNamedFunctionSymbol?, private val file: FirFile?) : FirVisitorVoid() {
        var follow = true
        private val elements = ArrayList<FirElement>()

        override fun visitElement(element: FirElement) {
            elements += element
            if (findProblems && helper != null) {
                val message = when (element) {
                    is FirProperty -> variableError(element)
                    is FirFunctionCall -> callError(element, session)
                    is FirStatement -> expressionError(element, elements)
                    else -> null
                }
                if (message != null) {
                    val positioning = if (element is FirProperty) SourceElementPositioningStrategies.DECLARATION_NAME
                        else SourceElementPositioningStrategies.DEFAULT
                    problems += HelperProblem(element.source, message, helper, file, positioning)
                }
            }
            if (follow && element is FirFunctionCall && element.source?.kind !is KtFakeSourceElementKind) {
                helperCallee(element, session)?.let { scanHelper(it) }
            }
            element.acceptChildren(this)
            elements.removeAt(elements.lastIndex)
        }
    }
}

private fun FirElement.accept(visitor: FirVisitorVoid) = accept(visitor, null)

/** In a kernel: unsupported code in helpers from other files, reported on the call. */
object KernelHelperCallChecker : FirDeclarationChecker<FirNamedFunction>(MppCheckerKind.Common) {
    context(context: CheckerContext, reporter: DiagnosticReporter)
    override fun check(declaration: FirNamedFunction) {
        if (!declaration.hasAnnotation(KERNEL_CLASS_ID, context.session)) return
        val body = declaration.body ?: return
        val kernelFile = context.containingFileSymbol
        val calls = LinkedHashMap<FirNamedFunctionSymbol, FirFunctionCall>()
        body.accept(object : FirVisitorVoid() {
            override fun visitElement(element: FirElement) {
                if (element is FirFunctionCall && element.source?.kind !is KtFakeSourceElementKind) {
                    helperCallee(element, context.session)?.let { calls.putIfAbsent(it, element) }
                }
                element.acceptChildren(this)
            }
        })
        for ((helper, call) in calls) {
            val scanner = HelperScanner(context.session)
            scanner.scanHelper(helper)
            val problem = scanner.problems.firstOrNull { it.file?.symbol != kernelFile } ?: continue
            val where = problem.file?.name?.let { " in $it" }.orEmpty()
            reporter.reportOn(
                call.source, KernelinErrors.KERNELIN_ERROR,
                "${problem.message} (in helper '${problem.helper.name}'$where, called from @Kernel ${declaration.name})",
            )
        }
    }
}

/** In a file with kernels: unsupported code in the helpers they call, reported on the helper's own code. */
object HelperFunctionChecker : FirDeclarationChecker<FirNamedFunction>(MppCheckerKind.Common) {
    @OptIn(SymbolInternals::class, DirectDeclarationsAccess::class)
    context(context: CheckerContext, reporter: DiagnosticReporter)
    override fun check(declaration: FirNamedFunction) {
        val session = context.session
        if (declaration.hasAnnotation(KERNEL_CLASS_ID, session) || declaration.symbol.callableId.isLocal) return
        val file = context.containingFileSymbol ?: return
        val kernels = file.fir.declarations
            .filterIsInstance<FirNamedFunction>()
            .filter { it.hasAnnotation(KERNEL_CLASS_ID, session) }
        val kernel = kernels.firstOrNull { kernel ->
            kernel.symbol.lazyResolveToPhase(FirResolvePhase.BODY_RESOLVE)
            val body = kernel.body ?: return@firstOrNull false
            HelperScanner(session, findProblems = false).apply { scanCallsOf(body) }.reached.contains(declaration.symbol)
        } ?: return
        val scanner = HelperScanner(session)
        scanner.scanHelper(declaration.symbol, follow = false)
        for (problem in scanner.problems) {
            reporter.reportOn(
                problem.source, KernelinErrors.KERNELIN_ERROR,
                "${problem.message} (in helper '${declaration.name}', called from @Kernel ${kernel.name})",
                problem.positioning,
            )
        }
    }
}
