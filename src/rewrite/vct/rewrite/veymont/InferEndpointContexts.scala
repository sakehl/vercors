package vct.rewrite.veymont

import com.typesafe.scalalogging.LazyLogging
import hre.util.ScopedStack
import vct.col.ast.{
  AbstractRewriter,
  AmbiguousLocation,
  ApplicableContract,
  Assert,
  Assign,
  Block,
  BooleanValue,
  Branch,
  ChorPerm,
  ChorRun,
  Choreography,
  Class,
  ClassDeclaration,
  Communicate,
  CommunicateStatement,
  CommunicateX,
  ConstructorInvocation,
  Declaration,
  Deref,
  Endpoint,
  EndpointExpr,
  EndpointName,
  EndpointStatement,
  Eval,
  Expr,
  FieldLocation,
  Fork,
  InstanceField,
  InstanceMethod,
  JavaClass,
  JavaConstructor,
  JavaInvocation,
  JavaLocal,
  JavaMethod,
  JavaNamedType,
  JavaParam,
  JavaPublic,
  JavaTClass,
  Join,
  Local,
  Location,
  Loop,
  MethodInvocation,
  NewObject,
  Node,
  Null,
  Perm,
  PredicateLocation,
  Procedure,
  Program,
  ReadPerm,
  RunMethod,
  Scope,
  Statement,
  TClass,
  TVeyMontChannel,
  TVoid,
  ThisChoreography,
  ThisObject,
  Type,
  UnitAccountedPredicate,
  Value,
  Variable,
  VeyMontAssignExpression,
}
import vct.col.origin.{AssignLocalOk, Origin, PanicBlame}
import vct.col.ref.Ref
import vct.col.resolve.ctx.RefJavaMethod
import vct.col.rewrite.adt.ImportADTImporter
import vct.col.rewrite.{Generation, Rewriter, RewriterBuilder, Rewritten}
import vct.col.util.SuccessionMap
import vct.col.util.AstBuildHelpers._
import vct.result.VerificationError.{SystemError, Unreachable, UserError}
import vct.rewrite.veymont.InferEndpointContexts.{
  EndpointInferenceUndefined,
  MultipleImplicitEndpointsError,
  NoImplicitEndpointError,
  getEndpoint,
}

import scala.collection.mutable

object InferEndpointContexts extends RewriterBuilder {
  override def key: String = "inferEndpointContexts"
  override def desc: String =
    "Infer endpoint context for ChorStatement nodes that require one but do not have it yet, such as assignment."

  case class NoImplicitEndpointError(expr: Expr[_]) extends UserError {
    override def code: String = "noImplicitEndpoint"
    override def text: String =
      expr.o.messageInContext(
        "Cannot infer an endpoint context for this expression."
      )
  }

  case class MultipleImplicitEndpointsError(expr: Expr[_]) extends UserError {
    override def code: String = "multipleImplicitEndpoints"
    override def text: String =
      expr.o.messageInContext(
        "This expression references multiple distinct endpoints, whereas only one is expected."
      )
  }

  case class EndpointInferenceUndefined(stmt: Node[_]) extends SystemError {
    override def text: String =
      stmt.o.messageInContext(
        "It is not defined whether an endpoint context should be inferred for this node"
      )
  }

  def getEndpoints[G](expr: Expr[G]): Seq[Endpoint[G]] =
    mutable.LinkedHashSet.from(expr.collect {
      case EndpointName(Ref(endpoint)) => endpoint
    }).toSeq

  def getEndpoint[G](expr: Expr[G]): Endpoint[G] =
    getEndpoints(expr) match {
      case Seq(endpoint) => endpoint
      case Seq() => throw NoImplicitEndpointError(expr)
      case _ => throw MultipleImplicitEndpointsError(expr)
    }

  def getEndpoint[G](loc: Location[G]): Endpoint[G] =
    loc match {
      case FieldLocation(obj, _) => getEndpoint(obj)
      case AmbiguousLocation(deref) => getEndpoint(deref)
      case _ => throw EndpointInferenceUndefined(loc)
    }
}

case class InferEndpointContexts[Pre <: Generation]()
    extends Rewriter[Pre] with LazyLogging {
  val inChor = ScopedStack[Boolean]()
  val inEndpointExpr = ScopedStack[Endpoint[Pre]]()

  override def dispatch(decl: Declaration[Pre]): Unit =
    decl match {
      case chor: Choreography[Pre] =>
        chor.rewrite(
          preRun = inChor.having(true) { chor.preRun.map(dispatch) },
          run = inChor.having(true) { dispatch(chor.run) },
        ).succeed(chor)
      case comm: Communicate[Pre] =>
        implicit val o = comm.o
        comm.rewrite(
          receiver = comm.receiver.map(_.decl)
            .orElse(Some(getEndpoint[Pre](comm.target)))
            .map(succ[Endpoint[Post]](_)),
          sender = comm.sender.map(_.decl)
            .orElse(Some(getEndpoint[Pre](comm.msg)))
            .map(succ[Endpoint[Post]](_)),
        ).succeed(comm)
      case _ => super.dispatch(decl)
    }

  override def dispatch(stmt: Statement[Pre]): Statement[Post] =
    stmt match {
      // Whitelist statements that do not need a context
      case s @ EndpointStatement(None, assign: Assign[Pre]) =>
        val endpoint: Endpoint[Pre] = getEndpoint(assign.target)
        s.rewrite(endpoint = Some(succ(endpoint)))
      case s @ EndpointStatement(None, Eval(invoke: MethodInvocation[Pre])) =>
        val endpoint: Endpoint[Pre] = getEndpoint(invoke.obj)
        s.rewrite(endpoint = Some(succ(endpoint)))
      case s @ EndpointStatement(None, _) => throw EndpointInferenceUndefined(s)
      case comm: CommunicateStatement[Pre] =>
        // Make inChor false because we don't want to infer endpoint contexts for expressions in the channel invariant
        // These should remain plain
        inChor.having(false) { comm.rewriteDefault() }
      case s => s.rewriteDefault()
    }

  override def dispatch(expr: Expr[Pre]): Expr[Post] =
    expr match {
      case p @ Perm(loc, perm)
          if inChor.topOption.contains(true) && inEndpointExpr.nonEmpty =>
        ChorPerm[Post](succ(inEndpointExpr.top), dispatch(loc), dispatch(perm))(
          p.o
        )
      case v @ Value(loc)
          if inChor.topOption.contains(true) && inEndpointExpr.nonEmpty =>
        ChorPerm[Post](
          succ(inEndpointExpr.top),
          dispatch(loc),
          ReadPerm()(v.o),
        )(v.o)
      case p @ Perm(loc, perm) if inChor.topOption.contains(true) =>
        ChorPerm[Post](succ(getEndpoint(loc)), dispatch(loc), dispatch(perm))(
          p.o
        )
      case v @ Value(loc) if inChor.topOption.contains(true) =>
        ChorPerm[Post](succ(getEndpoint(loc)), dispatch(loc), ReadPerm()(v.o))(
          v.o
        )
      case expr @ EndpointExpr(Ref(endpoint), _) =>
        inEndpointExpr.having(endpoint) { expr.rewriteDefault() }
      case _ => expr.rewriteDefault()
    }
}
