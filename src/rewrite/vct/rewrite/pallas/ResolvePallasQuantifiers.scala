package vct.rewrite.pallas

import hre.util.ScopedStack
import vct.col.ast._
import vct.col.origin.Origin
import vct.col.rewrite.{Generation, Rewriter, RewriterBuilder}
import vct.result.VerificationError.UserError
import vct.rewrite.pallas.ResolvePallasQuantifiers.InvalidBoundVariableUsage

object ResolvePallasQuantifiers extends RewriterBuilder {
  override def key: String = "resolvePallasQuantifiers"
  override def desc: String = "Encodes pallas-quantifiers into col-quantifiers."

  private final case class InvalidBoundVariableUsage(bv: LLVMBoundVar[_])
      extends UserError {
    override def code: String = "llvmInvalidBoundVarUsage"

    override def text: String =
      bv.o.messageInContext(f"Unsupported use of a bound variable: $bv")
  }
}

case class ResolvePallasQuantifiers[Pre <: Generation]() extends Rewriter[Pre] {

  // Tracks the bound-variables that are available in the current quantifier.
  private val boundVars: ScopedStack[Map[(String, Type[Pre]), Variable[Post]]] =
    ScopedStack()

  override def dispatch(expr: Expr[Pre]): Expr[Post] = {
    expr match {
      case LLVMForall(bindingExpr, bodyExpr) =>
        implicit val o: Origin = expr.o
        val oldBVMap = boundVars.topOption.getOrElse(Map.empty)
        // Gather all new bound variables from the binding expression
        val bVars = bindingExpr.collect { case LLVMBoundVar(id, vType) =>
          (id, vType)
        }.filterNot(t => oldBVMap.contains(t))

        // Declare variables for new BVs & update stack
        var newBVMap = oldBVMap
        var newBindings: Seq[Variable[Post]] = Seq.empty
        variables.scope {
          localHeapVariables.scope {
            bVars.foreach { case (id, vType) =>
              val v = new Variable[Post](dispatch(vType))
              newBVMap = newBVMap.updated((id, vType), v)
              newBindings = newBindings.appended(v)
            }
          }
        }
        boundVars.having(newBVMap) {
          Forall[Post](
            newBindings,
            Seq.empty,
            Implies(dispatch(bindingExpr), dispatch(bodyExpr)),
          )
        }
      case bv @ LLVMBoundVar(id, varType) =>
        implicit val o: Origin = bv.o
        val bvMap = boundVars.topOption.getOrElse(Map.empty)
        val bindingVar = bvMap
          .getOrElse((id, varType), throw InvalidBoundVariableUsage(bv))
        Local(bindingVar.ref)
      case _ => expr.rewriteDefault()
    }
  }

}
