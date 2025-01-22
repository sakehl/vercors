package vct.col.util

import hre.util.ScopedStack
import vct.col.ast._
import vct.col.origin.{DiagnosticOrigin, Origin}
import vct.col.ref.Ref
import vct.col.rewrite.{Generation, NonLatchingRewriter}
import vct.result.VerificationError

/** @param inlinePermLet
  *   Specified if let-expressions that contain Perm(...) should be inlined.
  *   This should only be relevant for the wrapper-inlining of pallas
  *   specifications
  */
case class StatementToExpression[Pre <: Generation, Post <: Generation](
    rw: NonLatchingRewriter[Pre, Post],
    errorBuilder: String => VerificationError,
    inlinePermLet: Boolean = false,
) {

  private val letSubstitutions
      : ScopedStack[Map[Ref[Pre, Variable[Pre]], Expr[Pre]]] = ScopedStack()

  def toExpression(
      stat: Statement[Pre],
      alt: => Option[Expr[Post]],
  ): Option[Expr[Post]] = {
    implicit val o: Origin = DiagnosticOrigin
    stat match {
      case Return(e) =>
        alt match {
          case Some(_) =>
            throw errorBuilder("Dead code after return is not allowed.")
          case None => Some(dispatchExpr(e))
        }
      case Block(Nil) => alt
      case Block(stat +: tail) =>
        toExpression(stat, toExpression(Block(tail), alt))
      case Branch(Nil) => alt
      case Branch((BooleanValue(true), impl) +: _) => toExpression(impl, alt)
      case Branch((cond, impl) +: branches) =>
        Some(Select(
          dispatchExpr(cond),
          toExpression(impl, alt).getOrElse(return None),
          toExpression(Branch(branches), alt).getOrElse(return None),
        ))
      case Scope(locals, impl) =>
        if (!locals.forall(countAssignments(_, impl).exists(_ <= 1))) {
          throw errorBuilder("Variables may only be assigned once.")
        }
        toExpression(impl, alt)
      case Assign(Local(ref), e) =>
        rw.localHeapVariables.scope {
          rw.variables.scope {
            e match {
              case Perm(_, _) if inlinePermLet =>
                // Inline the assignment
                letSubstitutions.having(
                  letSubstitutions.topOption.getOrElse(Map.empty)
                    .updated(ref, e)
                ) { alt }
              case _ =>
                // Turn into let-expression
                alt match {
                  case Some(exprAlt) =>
                    Some(Let[Post](
                      rw.variables.collect { rw.dispatch(ref.decl) }._1.head,
                      dispatchExpr(e),
                      exprAlt,
                    ))
                  case None =>
                    throw errorBuilder("Assign may not be the last statement")
                }
            }

          }
        }

      case _ => None
    }
  }

  private def dispatchExpr(e: Expr[Pre]): Expr[Post] = {
    if (!inlinePermLet || letSubstitutions.isEmpty) { return rw.dispatch(e) }
    val subMap = letSubstitutions.top.map[Expr[Pre], Expr[Pre]] {
      case (ref, expr) => (Local[Pre](ref)(DiagnosticOrigin), expr)
    }
    val sub = Substitute(subMap)
    rw.dispatch(sub.dispatch(e))
  }

  private def countAssignments[G <: Generation](
      v: Variable[G],
      s: Statement[G],
  ): Option[Int] =
    s match {
      case Return(_) => Some(0)
      case Block(stats) =>
        val x =
          stats.map(countAssignments(v, _)).fold(Some(0)) {
            case (Some(a), Some(b)) => Some(a + b)
            case _ => None
          }
        x
      case Branch(conds) =>
        val assignmentCounts = conds.map(_._2).map(countAssignments(v, _))
          .collect {
            case Some(n) => n
            case None => return None
          }
        if (assignmentCounts.forall(_ <= 1)) {
          assignmentCounts.maxOption.orElse(Some(0))
        } else { None }
      case Assign(Local(ref), _) =>
        Some(
          if (ref.decl == v)
            1
          else
            0
        )
      case Assign(_, _) => None
      case _ => None
    }

}
