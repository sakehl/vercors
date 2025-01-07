package vct.rewrite

import vct.col.ast._
import vct.col.origin.{LabelContext, Origin, PreferredName}
import vct.col.ref.Ref
import vct.col.rewrite.{Generation, Rewriter, RewriterBuilder, Rewritten}
import vct.col.util.AstBuildHelpers.assignLocal
import vct.col.util.StatementToExpression
import vct.result.Message
import vct.result.VerificationError.SystemError
import vct.rewrite.InlinePallasWrappers.{
  InlineArgAssignOrigin,
  WrapperInlineFailed,
}

case object InlinePallasWrappers extends RewriterBuilder {
  override def key: String = "inlinePallasWrapper"
  override def desc: String =
    "Inline calls to wrapper-function in pallas specifications."

  val InlineArgAssignOrigin: Origin = Origin(Seq(
    PreferredName(Seq("inlineArgAssign")),
    LabelContext("Assign of argument during inlining"),
  ))

  private def WrapperInliningOrigin(wrapperDef: Origin, inv: Node[_]): Origin =
    Origin(
      (LabelContext("inlining of ") +: inv.o.originContents) ++
        (LabelContext("definition") +: wrapperDef.originContents)
    )

  case class WrapperInlineFailed(inv: ProcedureInvocation[_], msg: String = "")
      extends SystemError {
    override def text: String = {
      Message.messagesInContext((
        inv.o,
        "Inlining of wrapper-function in pallas specification failed. " + msg,
      ))
    }
  }
}

case class InlinePallasWrappers[Pre <: Generation]() extends Rewriter[Pre] {

  override def dispatch(decl: Declaration[Pre]): Unit = {

    // Drop all definitions of wrapper functions, since they will be inlined.
    decl match {
      case proc: Procedure[Pre] if proc.pallasWrapper =>
      case other => rewriteDefault(other)
    }
  }

  override def dispatch(node: Expr[Pre]): Expr[Rewritten[Pre]] = {

    node match {
      case res: LLVMIntermediaryResult[Pre] =>
        implicit val o: Origin = res.o
        res.sretArg match {
          case Some(Ref(retArg)) => Local[Post](ref = succ(retArg))
          case None => Result[Post](applicable = succ(res.applicable.decl))
        }
      case inv: ProcedureInvocation[Pre] if inv.ref.decl.pallasWrapper =>
        // TODO: Implement inlining of pallas wrappers
        val wFunc = inv.ref.decl

        if (wFunc.body.isEmpty) {
          throw WrapperInlineFailed(inv, "Cannot inline function without body")
        }
        if (wFunc.args.size != inv.args.size) {
          throw WrapperInlineFailed(
            inv,
            "Number of arguments differs between definition and invocation.",
          )
        }

        // Declare variables to substitute the vars from the function definition.
        val newArgs = localHeapVariables.scope {
          variables.scope {
            wFunc.args.map(arg => new Variable[Pre](arg.t)(arg.o))
          }
        }

        val assigns = newArgs.zip(inv.args).map { case (v, e) =>
          assignLocal(new Local[Pre](v.ref)(v.o), e)(InlineArgAssignOrigin)
        }

        /*
        TODO: The function body is a scope, hence the assignments are not in the
        same scope as the rest of the instructions of the function-body
         */
        val bodyWithAssign = Block(assigns ++ wFunc.body)(inv.o)

        val inlinedBody = StatementToExpression.toExpression(
          this,
          (s: String) => WrapperInlineFailed(inv, s),
          bodyWithAssign,
          None,
        )

        inlinedBody match {
          case Some(e) => e
          case None =>
            throw WrapperInlineFailed(
              inv,
              "Wrapper could not be converted to expression.",
            )
        }
      case other => rewriteDefault(other)
    }

  }
}
