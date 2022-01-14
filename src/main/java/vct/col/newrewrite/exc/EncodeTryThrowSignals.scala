package vct.col.newrewrite.exc

import hre.util.ScopedStack
import vct.col.ast._
import vct.col.util.AstBuildHelpers._
import RewriteHelpers._
import vct.col.newrewrite.error.ExcludedByPassOrder
import vct.col.newrewrite.util.Substitute
import vct.col.origin.{AssertFailed, Blame, FramedGetLeft, FramedGetRight, Origin, ThrowNull}
import vct.col.ref.Ref
import vct.col.rewrite.{Generation, Rewriter, RewriterBuilder, Rewritten}
import vct.col.util.AstBuildHelpers
import RewriteHelpers._
import vct.result.VerificationResult.Unreachable

import scala.collection.mutable

case object EncodeTryThrowSignals extends RewriterBuilder {
  case class ThrowNullAssertFailed(t: Throw[_]) extends Blame[AssertFailed] {
    override def blame(error: AssertFailed): Unit =
      t.blame.blame(ThrowNull(t))
  }

  case object ExcVar extends Origin {
    override def preferredName: String = "exc"
    override def messageInContext(message: String): String =
      s"[At variable generated to contain thrown exception]: $message"
  }

  case object CurrentlyHandling extends Origin {
    override def preferredName: String = "currently_handling_exc"
    override def messageInContext(message: String): String =
      s"[At variable generated to remember exception currently being handled]: $message"
  }

  case object ReturnPoint extends Origin {
    override def preferredName: String = "bubble"
    override def messageInContext(message: String): String =
      s"[At label generated to bubble an exception]: $message"
  }

  case object CatchLabel extends Origin {
    override def preferredName: String = "catches"
    override def messageInContext(message: String): String =
      s"[At label generated for catch blocks]: $message"
  }

  case object FinallyLabel extends Origin {
    override def preferredName: String = "finally"
    override def messageInContext(message: String): String =
      s"[At label generated for finally]: $message"
  }
}

case class EncodeTryThrowSignals[Pre <: Generation]() extends Rewriter[Pre] {
  import EncodeTryThrowSignals._

  val currentException: ScopedStack[Variable[Post]] = ScopedStack()
  val exceptionalHandlerEntry: ScopedStack[LabelDecl[Post]] = ScopedStack()
  val returnHandler: ScopedStack[LabelDecl[Post]] = ScopedStack()

  val needCurrentExceptionRestoration: ScopedStack[Boolean] = ScopedStack()
  needCurrentExceptionRestoration.push(false)

  val signalsBinding: ScopedStack[(Variable[Pre], Expr[Post])] = ScopedStack()
  val catchBindings: mutable.Set[Variable[Pre]] = mutable.Set()

  val rootClass: ScopedStack[Ref[Post, Class[Post]]] = ScopedStack()

  def getExc(implicit o: Origin): Local[Post] =
    currentException.top.get

  override def dispatch(program: Program[Pre]): Program[Rewritten[Pre]] =
    program.rootClass match {
      case Some(TClass(Ref(cls))) =>
        rootClass.having(succ[Class[Post]](cls)) {
          program.rewrite()
        }
      case _ => throw Unreachable("Root class unknown or not a class.")
    }

  override def dispatch(stat: Statement[Pre]): Statement[Post] = {
    implicit val o: Origin = stat.o
    stat match {
      case TryCatchFinally(body, after, catches) =>
        val handlersEntry = new LabelDecl[Post]()(CatchLabel)
        val finallyEntry = new LabelDecl[Post]()(FinallyLabel)

        val newBody = exceptionalHandlerEntry.having(handlersEntry) {
          needCurrentExceptionRestoration.having(false) {
            dispatch(body)
          }
        }

        val catchImpl = Block[Post](catches.map {
          case CatchClause(decl, body) =>
            decl.drop()
            catchBindings += decl
            Branch(Seq((
              (getExc !== Null[Post]()) && InstanceOf(getExc, TypeValue(dispatch(decl.t))),
              Block(Seq(
                exceptionalHandlerEntry.having(finallyEntry) {
                  needCurrentExceptionRestoration.having(true) {
                    dispatch(body)
                  }
                },
                assignLocal(getExc, Null()),
              ),
            ))))
        })

        val finallyImpl = Block[Post](Seq(
          Label(finallyEntry, Block(Nil)),
          needCurrentExceptionRestoration.having(true) { dispatch(after) },
          Branch(Seq((
            getExc !== Null(),
            Goto(exceptionalHandlerEntry.top.ref),
          ))),
        ))

        val (store: Statement[Post], restore: Statement[Post], vars: Seq[Variable[Post]]) = if(needCurrentExceptionRestoration.top) {
          val tmp = new Variable[Post](TClass(rootClass.top))(CurrentlyHandling)
          (
            Block[Post](Seq(
              assignLocal(tmp.get, getExc),
              assignLocal(getExc, Null()),
            )),
            assignLocal[Post](getExc, tmp.get),
            Seq(tmp),
          )
        } else (Block[Post](Nil), Block[Post](Nil), Nil)

        Scope(vars, Block(Seq(
          store,
          newBody,
          Label(handlersEntry, Block(Nil)),
          catchImpl,
          finallyImpl,
          restore,
        )))

      case t @ Throw(obj) =>
        Block(Seq(
          assignLocal(getExc, dispatch(obj)),
          Assert(getExc !== Null())(ThrowNullAssertFailed(t)),
          Goto(exceptionalHandlerEntry.top.ref),
        ))

      case inv: InvokeProcedure[Pre] =>
        Block(Seq(
          inv.rewrite(outArgs = currentException.top.ref +: inv.outArgs.map(succ[Variable[Post]])),
          Branch(Seq((
            getExc !== Null(),
            Goto(exceptionalHandlerEntry.top.ref),
          ))),
        ))

      case inv: InvokeMethod[Pre] =>
        Block(Seq(
          inv.rewrite(outArgs = currentException.top.ref +: inv.outArgs.map(succ[Variable[Post]])),
          Branch(Seq((
            getExc !== Null(),
            Goto(exceptionalHandlerEntry.top.ref),
          ))),
        ))

      case other => rewriteDefault(other)
    }
  }

  override def dispatch(decl: Declaration[Pre]): Unit = decl match {
    case method: AbstractMethod[Pre] =>
      implicit val o: Origin = method.o

      val exc = new Variable[Post](TClass(rootClass.top))(ExcVar)

      currentException.having(exc) {
        val body = method.body.map(body => {
          val bubble = new LabelDecl[Post]()(ReturnPoint)

          Block(Seq(
            assignLocal(exc.get, Null()),
            exceptionalHandlerEntry.having(bubble) {
              currentException.having(exc) {
                dispatch(body)
              }
            },
            Label(bubble, Block(Nil)),
          ))
        })

        val ensures: Expr[Post] =
          ((exc.get !== Null()) ==> foldOr(method.contract.signals.map {
            case SignalsClause(binding, _) => InstanceOf(exc.get, TypeValue(dispatch(binding.t)))
          })) &*
          ((exc.get === Null()) ==> dispatch(method.contract.ensures)) &*
          AstBuildHelpers.foldStar(method.contract.signals.map {
            case SignalsClause(binding, assn) =>
              binding.drop()
              ((exc.get !== Null()) && InstanceOf(exc.get, TypeValue(dispatch(binding.t)))) ==>
                signalsBinding.having((binding, exc.get)) { dispatch(assn) }
          })

        method.rewrite(
          body = body,
          outArgs = collectInScope(variableScopes) { exc.declareDefault(this); method.outArgs.foreach(dispatch) },
          contract = method.contract.rewrite(ensures = ensures, signals = Nil),
        ).succeedDefault(this, method)
      }

    case other => rewriteDefault(other)
  }

  override def dispatch(e: Expr[Pre]): Expr[Post] = e match {
    case Local(Ref(v)) if signalsBinding.nonEmpty && signalsBinding.top._1 == v =>
      implicit val o: Origin = e.o
      signalsBinding.top._2

    case Local(Ref(v)) if catchBindings.contains(v) =>
      implicit val o: Origin = e.o
      Cast(getExc, TypeValue(dispatch(v.t)))

    case other => rewriteDefault(other)
  }
}
