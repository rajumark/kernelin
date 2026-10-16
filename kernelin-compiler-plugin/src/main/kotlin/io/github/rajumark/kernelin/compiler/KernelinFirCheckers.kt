package io.github.rajumark.kernelin.compiler

import org.jetbrains.kotlin.KtFakeSourceElementKind
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
import org.jetbrains.kotlin.fir.declarations.FirDeclarationOrigin
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
import org.jetbrains.kotlin.fir.symbols.impl.FirConstructorSymbol
import org.jetbrains.kotlin.fir.symbols.impl.FirNamedFunctionSymbol
import org.jetbrains.kotlin.fir.types.ConeKotlinType
import org.jetbrains.kotlin.fir.types.classId
import org.jetbrains.kotlin.fir.types.coneType
import org.jetbrains.kotlin.fir.types.functionTypeKind
import org.jetbrains.kotlin.fir.types.isMarkedNullable
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
 * reject anyway, so valid code is never blocked. Everything else is still checked by the translator.
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
        override val namedFunctionCheckers = setOf(KernelSignatureChecker)
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
        if (!declaration.isLocal) return
        // Variables the compiler made up (the iterator of a for loop, ...) aren't the user's to fix.
        if (declaration.source?.kind is KtFakeSourceElementKind) return
        val kernel = context.enclosingKernel() ?: return
        val type = declaration.returnTypeRef.coneType
        val fq = type.fq()
        if (!type.isMarkedNullable && (fq in SCALARS || fq in ARRAYS)) return
        reporter.reportOn(
            declaration.source, KernelinErrors.KERNELIN_ERROR,
            "variable '${declaration.name}' has type ${type.describe()}; only Int, Float and Boolean are supported${type.suggestion()} (in @Kernel ${kernel.name})",
            SourceElementPositioningStrategies.DECLARATION_NAME,
        )
    }
}

/** Expressions that never work on the GPU: strings, Double/Long constants, exceptions, casts, nullable values. */
object KernelExpressionChecker : FirExpressionChecker<FirStatement>(MppCheckerKind.Common) {
    context(context: CheckerContext, reporter: DiagnosticReporter)
    override fun check(expression: FirStatement) {
        if (expression.source?.kind is KtFakeSourceElementKind) return
        val kernel = context.enclosingKernel() ?: return
        val message = when (expression) {
            is FirStringConcatenationCall -> "strings aren't supported on the GPU"
            is FirLiteralExpression -> literalError(expression)
            is FirThrowExpression -> "throwing exceptions isn't supported on the GPU"
            is FirTryExpression -> "try/catch isn't supported on the GPU"
            is FirTypeOperatorCall -> "type checks and casts (is/as) aren't supported on the GPU"
            is FirSafeCallExpression, is FirElvisExpression, is FirCheckNotNullCall -> "nullable values aren't supported on the GPU"
            else -> null
        } ?: return
        report(expression, message, kernel)
    }

    context(context: CheckerContext)
    private fun literalError(literal: FirLiteralExpression): String? {
        // Annotation arguments like @Suppress("...") aren't kernel code.
        if (context.containingElements.any { it is FirAnnotation }) return null
        return when (literal.kind) {
            ConstantValueKind.String -> if (context.containingElements.any { it is FirStringConcatenationCall }) null
                else "strings aren't supported on the GPU"
            // A Double constant converted right away (2.0.toFloat()) is fine; the translator folds it.
            ConstantValueKind.Double -> if (isConvertedRightAway(literal)) null
                else "Double isn't supported on most GPUs; use Float (add an f suffix: 1.0f)"
            ConstantValueKind.Long -> if (isConvertedRightAway(literal)) null else "Long isn't supported on the GPU; use Int"
            else -> null
        }
    }

    context(context: CheckerContext)
    private fun isConvertedRightAway(literal: FirLiteralExpression): Boolean {
        val elements = context.containingElements
        val parent = elements.getOrNull(elements.lastIndex - 1)
        // Also skip `val d = 1.0`: the variable is already reported, one error per mistake is enough.
        if (parent is FirProperty) return true
        return parent is FirFunctionCall && parent.explicitReceiver === literal &&
            parent.calleeReference.name.asString().let { it == "toFloat" || it == "toInt" }
    }
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
        if (expression.source?.kind is KtFakeSourceElementKind) return
        val kernel = context.enclosingKernel() ?: return
        val callee = expression.calleeReference.toResolvedCallableSymbol() ?: return
        val message = when {
            callee is FirConstructorSymbol ->
                "creating objects or arrays isn't supported in kernels; pass arrays in as kernel parameters"
            callee.hasAnnotation(KERNEL_CLASS_ID, context.session) -> "a kernel can't call another @Kernel function"
            callee !is FirNamedFunctionSymbol -> null
            callee.origin == FirDeclarationOrigin.Source -> lambdaError(callee)
            else -> libraryError(callee)
        } ?: return
        report(expression, message, kernel)
    }

    context(context: CheckerContext)
    private fun lambdaError(callee: FirNamedFunctionSymbol): String? =
        if (callee.valueParameterSymbols.any { it.resolvedReturnType.functionTypeKind(context.session) != null }) {
            "lambdas aren't supported in kernels yet ('${callee.name}' takes one); use a for loop"
        } else null

    context(context: CheckerContext)
    private fun libraryError(callee: FirNamedFunctionSymbol): String? {
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
        return lambdaError(callee)
            ?: "'${id.asSingleFqName()}' can't run on the GPU (only kotlin.math, Int/Float operators and your own functions are supported)"
    }
}
