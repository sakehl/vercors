package vct.rewrite

import hre.util.ScopedStack
import vct.col.ast._
import vct.col.origin.{Origin, PanicBlame}
import vct.col.rewrite.{Generation, Rewriter, RewriterBuilder}
import vct.col.util.AstBuildHelpers._

case object EncodePointerComparison extends RewriterBuilder {
  override def key: String = "pointerComparison"
  override def desc: String =
    "Encodes comparison between pointers taking into account pointer provenance."

  // Probably overkill to keep track of all of these separately, but this might be useful in the future for disambiguating contexts
  private sealed trait Context
  private final case class InAxiom() extends Context
  private final case class InPrecondition() extends Context
  private final case class InPostcondition() extends Context
  private final case class InLoopInvariant() extends Context
  private final case class InAssertion() extends Context
  private final case class InExhale() extends Context
  private final case class InInhale() extends Context
  private final case class InQuantifier() extends Context
  private final case class InPredicate() extends Context
}

case class EncodePointerComparison[Pre <: Generation]() extends Rewriter[Pre] {
  import EncodePointerComparison._

  private val context: ScopedStack[Context] = ScopedStack()

  override def dispatch(decl: Declaration[Pre]): Unit =
    decl match {
      case axiom: ADTAxiom[Pre] =>
        context.having(InAxiom()) {
          allScopes.anySucceed(axiom, axiom.rewriteDefault())
        }
      case resource: AbstractPredicate[Pre] => context.having(InPredicate()) {
        allScopes.anySucceed(resource, resource.rewriteDefault())
      }
      case decl => allScopes.anySucceed(decl, decl.rewriteDefault())
    }

  override def dispatch(
      contract: ApplicableContract[Pre]
  ): ApplicableContract[Post] = {
    contract.rewrite(
      requires =
        context.having(InPrecondition()) { dispatch(contract.requires) },
      ensures = context.having(InPostcondition()) { dispatch(contract.ensures) },
    )
  }

  override def dispatch(contract: LoopContract[Pre]): LoopContract[Post] = {
    implicit val o: Origin = contract.o
    contract match {
      case inv @ LoopInvariant(invariant, _) =>
        inv.rewrite(invariant =
          context.having(InLoopInvariant()) { dispatch(invariant) }
        )
      case contract @ IterationContract(requires, ensures, _) =>
        contract.rewrite(
          requires = context.having(InPrecondition()) { dispatch(requires) },
          ensures = context.having(InPostcondition()) { dispatch(ensures) },
        )
    }
  }

  override def dispatch(stat: Statement[Pre]): Statement[Post] = stat match {
    case assert@Assert(res) => assert.rewrite(res=context.having(InAssertion()) {dispatch(res)})
    case exhale@Exhale(res) => exhale.rewrite(res=context.having(InExhale()) {dispatch(res)})
    case inhale@Inhale(res) => inhale.rewrite(res=context.having(InInhale()) {dispatch(res)})
    case assume@Assume(assn) => assume.rewrite(assn=context.having(InInhale()) {dispatch(assn)})
    case _ => super.dispatch(stat)
  }

  override def dispatch(e: Expr[Pre]): Expr[Post] = {
    implicit val o: Origin = e.o
    e match {
      case PointerEq(Null(), r) => dispatch(r) === Null()
      case PointerEq(l, Null()) => dispatch(l) === Null()
      case e @ PointerEq(l, r) =>
        dispatchPointerComparison(
          l,
          r,
          Eq(_, _),
          eitherNull = Eq(_, _),
          compareAddress = false,
        )
      case PointerNeq(Null(), r) => dispatch(r) !== Null()
      case PointerNeq(l, Null()) => dispatch(l) !== Null()
      case e @ PointerNeq(l, r) =>
        dispatchPointerComparison(
          l,
          r,
          Neq(_, _),
          eitherNull = Neq(_, _),
          compareAddress = false,
        )
      case PointerGreater(Null(), _) => ff
      case PointerGreater(l, Null()) => dispatch(l) !== Null()
      case e @ PointerGreater(l, r) =>
        dispatchPointerComparison(
          l,
          r,
          Greater(_, _),
          eitherNull = (l, r) => And(r === Null(), l !== Null()),
          compareAddress = true,
        )
      case PointerLess(Null(), r) => dispatch(r) !== Null()
      case PointerLess(_, Null()) => ff
      case e @ PointerLess(l, r) =>
        dispatchPointerComparison(
          l,
          r,
          Less(_, _),
          eitherNull = (l, r) => And(l === Null(), r !== Null()),
          compareAddress = true,
        )
      case PointerGreaterEq(Null(), r) => dispatch(r) === Null()
      case PointerGreaterEq(_, Null()) => tt
      case e @ PointerGreaterEq(l, r) =>
        dispatchPointerComparison(
          l,
          r,
          GreaterEq(_, _),
          eitherNull = (_, r) => r === Null(),
          compareAddress = true,
        )
      case PointerLessEq(Null(), _) => tt
      case PointerLessEq(l, Null()) => dispatch(l) === Null()
      case e @ PointerLessEq(l, r) =>
        dispatchPointerComparison(
          l,
          r,
          LessEq(_, _),
          eitherNull = (l, _) => l === Null(),
          compareAddress = true,
        )

      case asserting@Asserting(condition, _) => asserting.rewrite(condition=context.having(InAssertion()) {dispatch(condition)})
      // Match on all quantifiers
      case binder: Binder[Pre] => binder match {
        case exists@Exists(_, _, body) => exists.rewrite(body = context.having(InQuantifier()) {
          dispatch(body)
        })
        case forall@Forall(_, _, body) => forall.rewrite(body = context.having(InQuantifier()) {
          dispatch(body)
        })
        case starall@Starall(_, _, body) => starall.rewrite(body = context.having(InQuantifier()) {
          dispatch(body)
        })
        case forperm@ForPerm(_, _, body) => forperm.rewrite(body = context.having(InQuantifier()) {
          dispatch(body)
        })
        case forperm@ForPermWithValue(_, body) => forperm.rewrite(body = context.having(InQuantifier()) {
          dispatch(body)
        })
        case sum@Sum(_, condition, main) => sum.rewrite(condition=context.having(InQuantifier()){
          dispatch(condition)
        }, main = context.having(InQuantifier()) {
          dispatch(main)
        })
        case product@Product(_, condition, main) => product.rewrite(condition=context.having(InQuantifier()){
          dispatch(condition)
        }, main = context.having(InQuantifier()) {
          dispatch(main)
        })
        case let@Let(_, _, _) => super.dispatch(let)
      }
      case e => super.dispatch(e)
    }
  }

  private def dispatchPointerComparison(
      l: Expr[Pre],
      r: Expr[Pre],
      comparison: (Expr[Post], Expr[Post]) => Expr[Post],
      eitherNull: (Expr[Post], Expr[Post]) => Expr[Post],
      compareAddress: Boolean,
  )(implicit o: Origin): Expr[Post] = {
    val newL = dispatch(l)
    val newR = dispatch(r)
    val addressComp =
      (l: Expr[Post], r: Expr[Post]) => {
        comparison(
          PointerAddress(l)(PanicBlame("Can not be null")),
          PointerAddress(r)(PanicBlame("Can not be null")),
        )
      }
    val comp =
      if (compareAddress) { addressComp }
      else { comparison }

    val blockEq =
      PointerBlock(newL)(PanicBlame("Can not be null")) ===
        PointerBlock(newR)(PanicBlame("Can not be null"))

    Select(
      Or(newL === Null(), newR === Null()),
      eitherNull(newL, newR), {
        // Change based on context
        context.topOption match {
          case None =>
            Select(
              ChooseFresh[Post](LiteralSet(TBool(), Seq(tt, ff)))(PanicBlame(
                "Set is never empty"
              )),
              if (compareAddress) { And(blockEq, addressComp(newL, newR)) }
              else { comp(newL, newR) },
              addressComp(newL, newR),
            )
          case Some(
                InAxiom() | InPrecondition() | InPostcondition() |
                InLoopInvariant() | InAssertion() | InInhale() | InExhale() |  InQuantifier() | InPredicate()
              ) =>
            if (compareAddress) { And(blockEq, addressComp(newL, newR)) }
            else { comp(newL, newR) }
        }
      },
    )
  }
}
