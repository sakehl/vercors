package vct.col.rewrite

import hre.util.ScopedStack
import vct.col.ast._
import vct.col.origin.{
  Blame,
  DiagnosticOrigin,
  FrontendAdditiveError,
  InstanceInvocationFailure,
  InstanceNull,
  InvocationFailure,
  Origin,
  PlusProviderInvocationFailed,
  PlusProviderNull,
  WithContractFailure,
}
import vct.col.ref.Ref
import vct.col.rewrite.Disambiguate.{
  MissingSizeInformation,
  OperatorToInvocation,
}
import vct.col.rewrite.{Generation, Rewriter, RewriterBuilder}
import vct.col.typerules.CoercionUtils
import vct.col.util.AstBuildHelpers.withResult
import vct.col.util.SuccessionMap
import vct.result.VerificationError.{SystemError, Unreachable}

case object Disambiguate extends RewriterBuilder {
  override def key: String = "disambiguate"
  override def desc: String =
    "Translate ambiguous operators into concrete operators."

  case class OperatorToInvocation(blame: Blame[FrontendAdditiveError])
      extends Blame[InstanceInvocationFailure] {
    override def blame(error: InstanceInvocationFailure): Unit =
      error match {
        case InstanceNull(node) => blame.blame(PlusProviderNull(node))
        case failure: WithContractFailure =>
          blame.blame(PlusProviderInvocationFailed(failure))
      }
  }

  private case class MissingSizeInformation(node: Node[_]) extends SystemError {

    override def text: String =
      node.o.messageInContext(
        "Missing size information for this node, this is a bug!"
      )
  }
}

case class Disambiguate[Pre <: Generation]() extends Rewriter[Pre] {
  val functionSucc
      : SuccessionMap[InstanceOperatorFunction[Pre], InstanceFunction[Post]] =
    SuccessionMap()
  val methodSucc
      : SuccessionMap[InstanceOperatorMethod[Pre], InstanceMethod[Post]] =
    SuccessionMap()

  val currentResult: ScopedStack[Result[Post]] = ScopedStack()

  override def dispatch(decl: Declaration[Pre]): Unit =
    decl match {
      case f: InstanceOperatorFunction[Pre] =>
        functionSucc(f) =
          withResult { result: Result[Post] =>
            currentResult.having(result) {
              classDeclarations.declare(
                new InstanceFunction[Post](
                  dispatch(f.returnType),
                  variables.collect(f.args.map(dispatch(_)))._1,
                  variables.collect(f.typeArgs.map(dispatch(_)))._1,
                  f.body.map(dispatch(_)),
                  dispatch(f.contract),
                  f.inline,
                  f.threadLocal,
                )(f.blame)(f.o)
              )
            }
          }(DiagnosticOrigin)
      case m: InstanceOperatorMethod[Pre] =>
        methodSucc(m) =
          withResult { result: Result[Post] =>
            currentResult.having(result) {
              classDeclarations.declare(
                new InstanceMethod[Post](
                  dispatch(m.returnType),
                  variables.collect(m.args.map(dispatch(_)))._1,
                  variables.collect(m.outArgs.map(dispatch(_)))._1,
                  variables.collect(m.typeArgs.map(dispatch(_)))._1,
                  m.body.map(dispatch(_)),
                  dispatch(m.contract),
                  m.inline,
                  m.pure,
                )(m.blame)(m.o)
              )
            }
          }(DiagnosticOrigin)
      case _ => rewriteDefault(decl)
    }

  override def dispatch(e: Expr[Pre]): Expr[Post] = {
    implicit val o: Origin = e.o
    e match {
      case op @ AmbiguousDiv(left, right) =>
        if (op.isVectorIntOp)
          VectorFloorDiv(dispatch(left), dispatch(right))(op.blame)
        else if (op.isVectorOp)
          VectorFloatDiv(dispatch(left), dispatch(right))(op.blame)
        else if (op.isIntOp)
          FloorDiv(dispatch(left), dispatch(right))(op.blame)
        else
          FloatDiv(dispatch(left), dispatch(right))(op.blame)
      case op @ AmbiguousTruncDiv(left, right) =>
        if (op.isVectorIntOp)
          VectorTruncDiv(dispatch(left), dispatch(right))(op.blame)
        else if (op.isVectorOp)
          VectorFloatDiv(dispatch(left), dispatch(right))(op.blame)
        else if (op.isIntOp)
          TruncDiv(dispatch(left), dispatch(right))(op.blame)
        else
          FloatDiv(dispatch(left), dispatch(right))(op.blame)
      case op @ AmbiguousMod(left, right) =>
        if (op.isVectorOp)
          VectorMod(dispatch(left), dispatch(right))(op.blame)
        else
          Mod(dispatch(left), dispatch(right))(op.blame)
      case op @ AmbiguousTruncMod(left, right) =>
        if (op.isVectorOp)
          VectorTruncMod(dispatch(left), dispatch(right))(op.blame)
        else
          TruncMod(dispatch(left), dispatch(right))(op.blame)
      case op @ AmbiguousMult(left, right) =>
        if (op.isProcessOp)
          ProcessSeq(dispatch(left), dispatch(right))
        else if (op.isSetOp)
          SetIntersection(dispatch(left), dispatch(right))
        else if (op.isBagOp)
          BagLargestCommon(dispatch(left), dispatch(right))
        else if (op.isVectorOp)
          VectorMult(dispatch(left), dispatch(right))
        else
          Mult(dispatch(left), dispatch(right))
      case op @ AmbiguousPlus(left, right) =>
        if (op.isProcessOp)
          ProcessChoice(dispatch(left), dispatch(right))
        else if (op.isPointerOp)
          unfoldPointerAdd(
            PointerAdd(dispatch(left), dispatch(right))(op.blame)
          )
        else if (op.isSeqOp)
          Concat(dispatch(left), dispatch(right))
        else if (op.isSetOp)
          SetUnion(dispatch(left), dispatch(right))
        else if (op.isBagOp)
          BagAdd(dispatch(left), dispatch(right))
        else if (op.isStringOp)
          StringConcat(dispatch(left), dispatch(right))
        else if (op.getCustomPlusType(OperatorLeftPlus[Pre]()).isDefined)
          rewritePlusOf(OperatorLeftPlus[Pre](), op)
        else if (op.getCustomPlusType(OperatorRightPlus[Pre]()).isDefined)
          rewritePlusOf(OperatorRightPlus[Pre](), op)
        else if (op.isVectorOp)
          VectorPlus(dispatch(left), dispatch(right))
        else
          Plus(dispatch(left), dispatch(right))
      case op @ AmbiguousMinus(left, right) =>
        if (op.isSetOp)
          SetMinus(dispatch(left), dispatch(right))
        else if (op.isPointerOp)
          PointerAdd(dispatch(left), dispatch(UMinus(right)))(op.blame)
        else if (op.isBagOp)
          BagMinus(dispatch(left), dispatch(right))
        else if (op.isVectorOp)
          VectorMinus(dispatch(left), dispatch(right))
        else
          Minus(dispatch(left), dispatch(right))
      case op @ AmbiguousOr(left, right) =>
        if (op.isProcessOp)
          ProcessPar(dispatch(left), dispatch(right))
        else
          Or(dispatch(left), dispatch(right))
      case op: BitOp[Pre] =>
        val cons =
          if (op.isBoolOp)
            op match {
              case _: AmbiguousComputationalOr[Pre] => Or[Post](_, _)
              case _: AmbiguousComputationalXor[Pre] => Neq[Post](_, _)
              case _: AmbiguousComputationalAnd[Pre] => And[Post](_, _)
            }
          else
            op match {
              case _: AmbiguousComputationalOr[Pre] =>
                ComputationalOr[Post](_, _)
              case _: AmbiguousComputationalXor[Pre] =>
                ComputationalXor[Post](_, _)
              case _: AmbiguousComputationalAnd[Pre] =>
                ComputationalAnd[Post](_, _)
            }

        cons(dispatch(op.left), dispatch(op.right))
      case op @ AmbiguousSubscript(collection, index) =>
        if (op.isPointerOp)
          PointerSubscript(dispatch(collection), dispatch(index))(op.blame)
        else if (op.isMapOp)
          MapGet(dispatch(collection), dispatch(index))(op.blame)
        else if (op.isArrayOp)
          ArraySubscript(dispatch(collection), dispatch(index))(op.blame)
        else if (op.isSeqOp)
          SeqSubscript(dispatch(collection), dispatch(index))(op.blame)
        else if (op.isVectorOp)
          VectorSubscript(dispatch(collection), dispatch(index))(op.blame)
        else
          throw Unreachable(
            "AmbiguousSubscript must subscript a pointer, map, array, or seq because of the type check."
          )
      case op @ AmbiguousMember(x, xs) =>
        if (op.isMapOp)
          MapMember(dispatch(x), dispatch(xs))
        else if (op.isSetOp)
          SetMember(dispatch(x), dispatch(xs))
        else if (op.isBagOp)
          BagMemberCount(dispatch(x), dispatch(xs))
        else if (op.isSeqOp)
          SeqMember(dispatch(x), dispatch(xs))
        else
          throw Unreachable(
            "AmbiguousMember must query a map, set, bag, or seq because of the type check."
          )
      case cmp: AmbiguousComparison[Pre] =>
        if (cmp.isMapOp)
          cmp match {
            case AmbiguousEq(left, right, _, _) =>
              MapEq(dispatch(left), dispatch(right))
            case AmbiguousNeq(left, right, _, _) =>
              Not(MapEq(dispatch(left), dispatch(right)))
          }
        else if (cmp.isVectorOp)
          cmp match {
            case AmbiguousEq(left, right, _, _) =>
              VectorEq(dispatch(left), dispatch(right))
            case AmbiguousNeq(left, right, _, _) =>
              VectorNeq(dispatch(left), dispatch(right))
          }
        else if (cmp.isPointerOp)
          cmp match {
            case AmbiguousEq(left, right, _, Some(size)) =>
              PointerEq(dispatch(left), dispatch(right), dispatch(size))
            case e @ AmbiguousEq(_, _, _, None) =>
              throw MissingSizeInformation(e)
            case AmbiguousNeq(left, right, _, Some(size)) =>
              PointerNeq(dispatch(left), dispatch(right), dispatch(size))
            case e @ AmbiguousNeq(_, _, _, None) =>
              throw MissingSizeInformation(e)
          }
        else
          cmp match {
            case AmbiguousEq(left, right, _, _) =>
              Eq(dispatch(left), dispatch(right))
            case AmbiguousNeq(left, right, _, _) =>
              Neq(dispatch(left), dispatch(right))
          }
      case cmp: AmbiguousOrderOp[Pre] =>
        if (cmp.isBagOp)
          cmp match {
            case AmbiguousGreater(left, right, _) =>
              SubBag(dispatch(right), dispatch(left))
            case AmbiguousLess(left, right, _) =>
              SubBag(dispatch(left), dispatch(right))
            case AmbiguousGreaterEq(left, right, _) =>
              SubBagEq(dispatch(right), dispatch(left))
            case AmbiguousLessEq(left, right, _) =>
              SubBagEq(dispatch(left), dispatch(right))
          }
        else if (cmp.isSetOp)
          cmp match {
            case AmbiguousGreater(left, right, _) =>
              SubSet(dispatch(right), dispatch(left))
            case AmbiguousLess(left, right, _) =>
              SubSet(dispatch(left), dispatch(right))
            case AmbiguousGreaterEq(left, right, _) =>
              SubSetEq(dispatch(right), dispatch(left))
            case AmbiguousLessEq(left, right, _) =>
              SubSetEq(dispatch(left), dispatch(right))
          }
        else if (cmp.isPointerOp)
          cmp match {
            case AmbiguousGreater(left, right, Some(size)) =>
              PointerGreater(dispatch(left), dispatch(right), dispatch(size))
            case e @ AmbiguousGreater(_, _, None) =>
              throw MissingSizeInformation(e)
            case AmbiguousLess(left, right, Some(size)) =>
              PointerLess(dispatch(left), dispatch(right), dispatch(size))
            case e @ AmbiguousLess(_, _, None) =>
              throw MissingSizeInformation(e)
            case AmbiguousGreaterEq(left, right, Some(size)) =>
              PointerGreaterEq(dispatch(left), dispatch(right), dispatch(size))
            case e @ AmbiguousGreaterEq(_, _, None) =>
              throw MissingSizeInformation(e)
            case AmbiguousLessEq(left, right, Some(size)) =>
              PointerLessEq(dispatch(left), dispatch(right), dispatch(size))
            case e @ AmbiguousLessEq(_, _, None) =>
              throw MissingSizeInformation(e)
          }
        else
          cmp match {
            case AmbiguousGreater(left, right, _) =>
              Greater(dispatch(left), dispatch(right))
            case AmbiguousLess(left, right, _) =>
              Less(dispatch(left), dispatch(right))
            case AmbiguousGreaterEq(left, right, _) =>
              GreaterEq(dispatch(left), dispatch(right))
            case AmbiguousLessEq(left, right, _) =>
              LessEq(dispatch(left), dispatch(right))
          }
      case r @ Result(_) if currentResult.nonEmpty =>
        Result(currentResult.top.applicable)(r.o)
      case other => rewriteDefault(other)
    }
  }

  def rewritePlusOf(
      operator: Operator[Pre],
      plus: AmbiguousPlus[Pre],
  ): Expr[Post] = {
    val (subject, other) =
      if (operator == OperatorLeftPlus[Pre]())
        (plus.left, plus.right)
      else
        (plus.right, plus.left)
    val validOperators = plus.getValidOperatorsOf(operator).get
    validOperators match {
      case Seq(m: InstanceOperatorMethod[Pre]) =>
        MethodInvocation[Post](
          dispatch(subject),
          methodSucc.ref(m),
          Seq(dispatch(other)),
          Seq(),
          Seq(),
          Seq(),
          Seq(),
        )(OperatorToInvocation(plus.blame))(plus.o)
      case Seq(f: InstanceOperatorFunction[Pre]) =>
        InstanceFunctionInvocation[Post](
          dispatch(subject),
          functionSucc.ref(f),
          Seq(dispatch(other)),
          Seq(),
          Seq(),
          Seq(),
        )(OperatorToInvocation(plus.blame))(plus.o)
      case _ => ???
    }
  }

  def unfoldPointerAdd[G](e: PointerAdd[G]): PointerAdd[G] =
    e.pointer match {
      case inner @ PointerAdd(_, _) =>
        val PointerAdd(pointerInner, offsetInner) = unfoldPointerAdd(inner)
        PointerAdd(pointerInner, Plus(offsetInner, e.offset)(e.o))(e.blame)(e.o)
      case _ => e
    }
}
