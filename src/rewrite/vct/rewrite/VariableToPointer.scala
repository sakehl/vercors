package vct.rewrite

import hre.util.ScopedStack
import vct.col.ast._
import vct.col.ref._
import vct.col.origin._
import vct.col.rewrite.{Generation, Rewriter, RewriterBuilder, Rewritten}
import vct.col.util.AstBuildHelpers._
import vct.col.util.SuccessionMap
import vct.result.VerificationError.UserError

import scala.collection.mutable

case object VariableToPointer extends RewriterBuilder {
  override def key: String = "variableToPointer"

  override def desc: String =
    "Translate every local and field to a pointer such that it can have its address taken"

  case class UnsupportedAddrOf(loc: Expr[_]) extends UserError {
    override def code: String = "unsupportedAddrOf"

    override def text: String =
      loc.o.messageInContext(
        "Taking an address of this expression is not supported"
      )
  }

  private case class CannotTakeAddressInFunction(arg: Variable[_])
      extends UserError {
    override def code: String = "addrOfFuncArg"

    override def text: String =
      arg.o.messageInContext(
        "Taking the address of a pure function's argument is not supported"
      )
  }
}

case class VariableToPointer[Pre <: Generation]() extends Rewriter[Pre] {

  import VariableToPointer._

  val addressedSet: mutable.Set[Node[Pre]] = new mutable.HashSet[Node[Pre]]()
  val heapVariableMap: SuccessionMap[HeapVariable[Pre], HeapVariable[Post]] =
    SuccessionMap()
  val variableMap: SuccessionMap[Variable[Pre], Variable[Post]] =
    SuccessionMap()
  val fieldMap: SuccessionMap[InstanceField[Pre], InstanceField[Post]] =
    SuccessionMap()
  val noTransform: ScopedStack[scala.collection.Set[Variable[Pre]]] =
    ScopedStack()

  override def dispatch(program: Program[Pre]): Program[Rewritten[Pre]] = {
    // TODO: Replace the asByReferenceClass checks with something that more clearly communicates that we want to exclude all reference types
    addressedSet.addAll(program.collect {
      case AddrOf(Local(Ref(v))) if v.t.asByReferenceClass.isEmpty => v
      case AddrOf(DerefHeapVariable(Ref(v)))
          if v.t.asByReferenceClass.isEmpty =>
        v
      case AddrOf(Deref(o, Ref(f)))
          if f.t.asByReferenceClass.isEmpty && o.t.asByValueClass.isEmpty =>
        f
    })
    super.dispatch(program)
  }

  override def dispatch(decl: Declaration[Pre]): Unit =
    decl match {
      case func: Function[Pre] => {
        val arg = func.args.find(addressedSet.contains(_))
        if (arg.nonEmpty) { throw CannotTakeAddressInFunction(arg.get) }
        globalDeclarations.succeed(func, func.rewriteDefault())
      }
      case proc: Procedure[Pre] => {
        val skipVars = mutable.Set[Variable[Pre]]()
        val extraVars = mutable.ArrayBuffer[(Variable[Post], Variable[Post])]()
        // Relies on args being evaluated before body
        allScopes.anySucceed(
          proc,
          proc.rewrite(
            args =
              variables.collect {
                proc.args.map { v =>
                  val newV = variables.succeed(v, v.rewriteDefault())
                  if (addressedSet.contains(v)) {
                    variableMap(v) =
                      new Variable(TNonNullPointer(dispatch(v.t), None))(v.o)
                    skipVars += v
                    extraVars += ((newV, variableMap(v)))
                  }
                }
              }._1,
            body = {
              if (proc.body.isEmpty) { None }
              else {
                if (extraVars.isEmpty) { Some(dispatch(proc.body.get)) }
                else {
                  variables.scope {
                    val locals =
                      variables.collect {
                        extraVars.map { case (_, pointer) =>
                          variables.declare(pointer)
                        }
                      }._1
                    val block =
                      Block(extraVars.map { case (normal, pointer) =>
                        Assign(
                          DerefPointer(pointer.get(normal.o))(PanicBlame(
                            "Non-null pointer should always be initialized successfully"
                          ))(normal.o),
                          normal.get(normal.o),
                        )(AssignLocalOk)(proc.o)
                      }.toSeq :+ dispatch(proc.body.get))(proc.o)
                    Some(Scope(locals, block)(proc.o))
                  }
                }
              }
            },
            contract = {
              noTransform.having(skipVars) { dispatch(proc.contract) }
            },
          ),
        )
      }
      case v: HeapVariable[Pre] if addressedSet.contains(v) =>
        heapVariableMap(v) = globalDeclarations
          .succeed(v, new HeapVariable(TNonNullPointer(dispatch(v.t), None))(v.o))
      case v: Variable[Pre] if addressedSet.contains(v) =>
        variableMap(v) = variables
          .succeed(v, new Variable(TNonNullPointer(dispatch(v.t), None))(v.o))
      case f: InstanceField[Pre] if addressedSet.contains(f) =>
        fieldMap(f) = classDeclarations.succeed(
          f,
          new InstanceField(
            TNonNullPointer(dispatch(f.t), None),
            f.flags.map { it => dispatch(it) },
          )(f.o),
        )
      case other => allScopes.anySucceed(other, other.rewriteDefault())
    }

  override def dispatch(stat: Statement[Pre]): Statement[Post] = {
    implicit val o: Origin = stat.o
    stat match {
      case s: Scope[Pre] =>
        s.rewrite(
          locals = variables.dispatch(s.locals),
          body = Block(s.locals.filter { local => addressedSet.contains(local) }
            .map { local =>
              implicit val o: Origin = local.o
              Assign(
                Local[Post](variableMap.ref(local)),
                NewNonNullPointerArray(
                  variableMap(local).t.asPointer.get.element,
                  const(1),
                  None
                )(PanicBlame("Size is > 0")),
              )(PanicBlame("Initialisation should always succeed"))
            } ++ Seq(dispatch(s.body))),
        )
      case i @ Instantiate(cls, out)
          if cls.decl.isInstanceOf[ByValueClass[Pre]] =>
        Block(Seq(i.rewriteDefault()) ++ cls.decl.declarations.flatMap {
          case f: InstanceField[Pre] =>
            if (f.t.asClass.isDefined) {
              Seq(
                Assign(
                  Deref[Post](dispatch(out), fieldMap.ref(f))(PanicBlame(
                    "Initialisation should always succeed"
                  )),
                  NewNonNullPointerArray(
                    fieldMap(f).t.asPointer.get.element,
                    const(1),
                    None
                  )(PanicBlame("Size is > 0")),
                )(PanicBlame("Initialisation should always succeed")),
                Assign(
                  PointerSubscript(
                    Deref[Post](dispatch(out), fieldMap.ref(f))(PanicBlame(
                      "Initialisation should always succeed"
                    )),
                    const[Post](0),
                  )(PanicBlame("Size is > 0")),
                  dispatch(NewObject[Pre](f.t.asClass.get.cls)),
                )(PanicBlame("Initialisation should always succeed")),
              )
            } else if (addressedSet.contains(f)) {
              Seq(
                Assign(
                  Deref[Post](dispatch(out), fieldMap.ref(f))(PanicBlame(
                    "Initialisation should always succeed"
                  )),
                  NewNonNullPointerArray(
                    fieldMap(f).t.asPointer.get.element,
                    const(1),
                    None
                  )(PanicBlame("Size is > 0")),
                )(PanicBlame("Initialisation should always succeed"))
              )
            } else { Seq() }
          case _ => Seq()
        })
      case other => other.rewriteDefault()
    }
  }

  override def dispatch(expr: Expr[Pre]): Expr[Post] = {
    implicit val o: Origin = expr.o
    expr match {
      case deref @ DerefHeapVariable(Ref(v)) if addressedSet.contains(v) =>
        DerefPointer(
          DerefHeapVariable[Post](heapVariableMap.ref(v))(deref.blame)
        )(PanicBlame("Should always be accessible"))
      case Local(Ref(v))
          if addressedSet.contains(v) && !noTransform.exists(_.contains(v)) =>
        DerefPointer(Local[Post](variableMap.ref(v)))(PanicBlame(
          "Should always be accessible"
        ))
      case deref @ Deref(obj, Ref(f)) if addressedSet.contains(f) =>
        DerefPointer(Deref[Post](dispatch(obj), fieldMap.ref(f))(deref.blame))(
          PanicBlame("Should always be accessible")
        )
      case newObject @ NewObject(Ref(cls: ByValueClass[Pre])) =>
        val obj = new Variable[Post](TByValueClass(succ(cls), Seq()))
        ScopedExpr(
          Seq(obj),
          With(
            Block(
              Seq(assignLocal(obj.get, newObject.rewriteDefault())) ++
                cls.declarations.flatMap {
                  case f: InstanceField[Pre] =>
                    if (f.t.asClass.isDefined) {
                      Seq(
                        Assign(
                          Deref[Post](obj.get, anySucc(f))(PanicBlame(
                            "Initialisation should always succeed"
                          )),
                          dispatch(NewObject[Pre](f.t.asClass.get.cls)),
                        )(PanicBlame("Initialisation should always succeed"))
                      )
                    } else if (addressedSet.contains(f)) {
                      Seq(
                        Assign(
                          Deref[Post](obj.get, fieldMap.ref(f))(PanicBlame(
                            "Initialisation should always succeed"
                          )),
                          NewNonNullPointerArray(
                            fieldMap(f).t.asPointer.get.element,
                            const(1),
                            None
                          )(PanicBlame("Size is > 0")),
                        )(PanicBlame("Initialisation should always succeed"))
                      )
                    } else { Seq() }
                  case _ => Seq()
                }
            ),
            obj.get,
          ),
        )
      case Perm(PointerLocation(AddrOf(DerefHeapVariable(Ref(v)))), perm)
          if addressedSet.contains(v) =>
        val newPerm = dispatch(perm)
        Star(
          Perm(HeapVariableLocation[Post](heapVariableMap.ref(v)), newPerm),
          Perm(
            PointerLocation(DerefHeapVariable[Post](heapVariableMap.ref(v))(
              PanicBlame("Access is framed")
            ))(PanicBlame("Cannot be null")),
            newPerm,
          ),
        )
      case Value(PointerLocation(AddrOf(DerefHeapVariable(Ref(v)))))
          if addressedSet.contains(v) =>
        Star(
          Value(HeapVariableLocation[Post](heapVariableMap.ref(v))),
          Value(
            PointerLocation(DerefHeapVariable[Post](heapVariableMap.ref(v))(
              PanicBlame("Access is framed")
            ))(PanicBlame("cannot be null"))
          ),
        )
      case other => other.rewriteDefault()
    }
  }

  override def dispatch(loc: Location[Pre]): Location[Post] = {
    implicit val o: Origin = loc.o
    loc match {
      case HeapVariableLocation(Ref(v)) if addressedSet.contains(v) =>
        PointerLocation(
          DerefHeapVariable[Post](heapVariableMap.ref(v))(PanicBlame(
            "Should always be accessible"
          ))
        )(PanicBlame("Should always be accessible"))
      case FieldLocation(obj, Ref(f)) if addressedSet.contains(f) =>
        PointerLocation(Deref[Post](dispatch(obj), fieldMap.ref(f))(PanicBlame(
          "Should always be accessible"
        )))(PanicBlame("Should always be accessible"))
      case PointerLocation(AddrOf(Deref(obj, Ref(f))))
          if addressedSet.contains(f) =>
        FieldLocation[Post](dispatch(obj), fieldMap.ref(f))
      case PointerLocation(AddrOf(DerefHeapVariable(Ref(v))))
          if addressedSet.contains(v) =>
        HeapVariableLocation[Post](heapVariableMap.ref(v))
      case PointerLocation(AddrOf(local @ Local(_))) =>
        throw UnsupportedAddrOf(local)
      case other => other.rewriteDefault()
    }
  }
}
