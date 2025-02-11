package vct.col.rewrite.adt

import vct.col.ast._
import ImportADT.typeText
import hre.util.ScopedStack
import vct.col.origin._
import vct.col.ref.Ref
import vct.col.rewrite.Generation
import vct.col.util.AstBuildHelpers.{functionInvocation, _}
import vct.col.util.SuccessionMap

import scala.collection.mutable

case object ImportPointer extends ImportADTBuilder("pointer") {
  private def PointerField(t: Type[_]): Origin =
    Origin(Seq(PreferredName(Seq(typeText(t))), LabelContext("pointer field")))

  private val PointerCreationOrigin: Origin = Origin(
    Seq(LabelContext("adtPointer, pointer creation method"))
  )

  private val AsTypeOrigin: Origin = Origin(
    Seq(LabelContext("classToRef, asType function"))
  )

  case class PointerNullOptNone(inner: Blame[PointerNull], expr: Expr[_])
      extends Blame[OptionNone] {
    override def blame(error: OptionNone): Unit = inner.blame(PointerNull(expr))
  }

  case class PointerBoundsPreconditionFailed(
      inner: Blame[PointerBounds],
      expr: Node[_],
  ) extends Blame[PreconditionFailed] {
    override def blame(error: PreconditionFailed): Unit =
      inner.blame(PointerBounds(expr))
  }

  case class DerefPointerBoundsPreconditionFailed(
      inner: Blame[PointerDerefError],
      expr: Expr[_],
  ) extends Blame[PreconditionFailed] {
    override def blame(error: PreconditionFailed): Unit =
      inner.blame(PointerInsufficientPermission(expr))
  }

  case class PointerFieldInsufficientPermission(
      inner: Blame[PointerInsufficientPermission],
      expr: Expr[_],
  ) extends Blame[InsufficientPermission] {
    override def blame(error: InsufficientPermission): Unit =
      inner.blame(PointerInsufficientPermission(expr))
  }
}

case class ImportPointer[Pre <: Generation](importer: ImportADTImporter)
    extends ImportADT[Pre](importer) {
  import ImportPointer._

  private lazy val pointerFile = parse("pointer")

  private lazy val blockAdt = find[AxiomaticDataType[Post]](
    pointerFile,
    "block",
  )
  private lazy val blockBase = find[ADTFunction[Post]](blockAdt, "base_addr")
  private lazy val blockLength = find[ADTFunction[Post]](
    blockAdt,
    "block_length",
  )
  private lazy val blockLoc = find[ADTFunction[Post]](blockAdt, "loc")
  private lazy val pointerAdt = find[AxiomaticDataType[Post]](
    pointerFile,
    "pointer",
  )
  private lazy val pointerOf = find[ADTFunction[Post]](pointerAdt, "pointer_of")
  private lazy val pointerBlock = find[ADTFunction[Post]](
    pointerAdt,
    "pointer_block",
  )
  private lazy val pointerOffset = find[ADTFunction[Post]](
    pointerAdt,
    "pointer_offset",
  )
  private lazy val pointerDeref = find[Function[Post]](pointerFile, "ptr_deref")
  private lazy val pointerAdd = find[Function[Post]](pointerFile, "ptr_add")

  private val pointerField: mutable.Map[Type[Post], SilverField[Post]] = mutable
    .Map()

  private val pointerCreationMethods
      : SuccessionMap[Type[Pre], Procedure[Post]] = SuccessionMap()

  private val asTypeFunctions: mutable.Map[Type[Pre], Function[Post]] = mutable
    .Map()
  private val inAxiom: ScopedStack[Unit] = ScopedStack()
  private var casts: Set[(Type[Pre], Type[Pre])] = Set.empty

  private def makeAsTypeFunction(
      typeName: String,
      adt: Ref[Post, AxiomaticDataType[Post]] = pointerAdt.ref,
  ): Function[Post] = {
    val value =
      new Variable[Post](TAxiomatic(adt, Nil))(
        AsTypeOrigin.where(name = "value")
      )
    globalDeclarations.declare(
      function[Post](
        AbstractApplicable,
        TrueSatisfiable,
        returnType = TAxiomatic(adt, Nil),
        args = Seq(value),
      )(AsTypeOrigin.where(name = "as_" + typeName))
    )
  }

  private def makePointerCreationMethod(
      t: Type[Pre],
      newT: Type[Post],
  ): Procedure[Post] = {
    implicit val o: Origin = PointerCreationOrigin
      .where(name = "create_nonnull_pointer_" + newT.toString)

    val result =
      new Variable[Post](TAxiomatic(pointerAdt.ref, Nil))(o.where(name = "res"))
    globalDeclarations.declare(procedure[Post](
      blame = AbstractApplicable,
      contractBlame = TrueSatisfiable,
      returnType = TVoid(),
      outArgs = Seq(result),
      ensures = UnitAccountedPredicate(
        (ADTFunctionInvocation[Post](
          typeArgs = Some((blockAdt.ref, Nil)),
          ref = blockLength.ref,
          args = Seq(ADTFunctionInvocation[Post](
            typeArgs = Some((pointerAdt.ref, Nil)),
            ref = pointerBlock.ref,
            args = Seq(result.get),
          )),
        ) === const(1)) &*
          (ADTFunctionInvocation[Post](
            typeArgs = Some((pointerAdt.ref, Nil)),
            ref = pointerOffset.ref,
            args = Seq(result.get),
          ) === const(0)) &* Perm(
            SilverFieldLocation(
              obj =
                FunctionInvocation[Post](
                  ref = pointerDeref.ref,
                  args = Seq(result.get),
                  typeArgs = Nil,
                  Nil,
                  Nil,
                )(PanicBlame("ptr_deref requires nothing.")),
              field =
                pointerField.getOrElseUpdate(
                  newT, {
                    globalDeclarations
                      .declare(new SilverField(newT)(PointerField(newT)))
                  },
                ).ref,
            ),
            WritePerm(),
          ) &* (asType(t, result.get) === result.get)
      ),
      decreases = Some(DecreasesClauseNoRecursion[Post]()),
    ))
  }

  private def getPointerField(ptr: Expr[Pre]): Ref[Post, SilverField[Post]] = {
    val tElement = dispatch(ptr.t.asPointer.get.element)
    pointerField.getOrElseUpdate(
      tElement, {
        globalDeclarations
          .declare(new SilverField(tElement)(PointerField(tElement)))
      },
    ).ref
  }

  private def unwrapOption(
      ptr: Expr[Pre],
      blame: Blame[PointerNull],
  ): Expr[Post] = {
    ptr.t match {
      case TPointer(_) =>
        dispatch(ptr) match {
          case OptSome(inner) => inner
          case newPtr => OptGet(newPtr)(PointerNullOptNone(blame, ptr))(ptr.o)
        }
      case TNonNullPointer(_) => dispatch(ptr)
    }
  }

  override def applyCoercion(e: => Expr[Post], coercion: Coercion[Pre])(
      implicit o: Origin
  ): Expr[Post] =
    coercion match {
      case CoerceNullPointer(_) => OptNone()
      case CoerceNonNullPointer(_) => OptSome(e)
      case other => super.applyCoercion(e, other)
    }

  override def postCoerce(program: Program[Pre]): Program[Post] = {
    casts =
      program.collect {
        case Cast(a, TypeValue(b))
            if a.t.asPointer.isDefined && b.asPointer.isDefined =>
          (a.t.asPointer.get.element, b.asPointer.get.element)
      }.toSet
    super.postCoerce(program)
  }

  override def postCoerce(decl: Declaration[Pre]): Unit = {
    decl match {
      case axiom: ADTAxiom[Pre] =>
        inAxiom.having(()) {
          allScopes.anySucceed(axiom, axiom.rewriteDefault())
        }
      // TODO: This is an ugly way to exempt this one bit of generated code from having ptrAdd's added
      case proc: Procedure[Pre]
          if proc.o.find[LabelContext]
            .exists(_.label == "classToRef cast helpers") =>
        inAxiom.having(()) { allScopes.anySucceed(proc, proc.rewriteDefault()) }
      case adt: AxiomaticDataType[Pre]
          if adt.o.find[SourceName].exists(_.name == "pointer") =>
        implicit val o: Origin = adt.o
        inAxiom.having(()) {
          globalDeclarations.succeed(
            adt,
            adt.rewrite(decls = {
              aDTDeclarations.collect {
                adt.decls.foreach(dispatch)
                casts.collect {
                  // TODO: Should we be doing Pointer(TAny) here instead of Pointer(TVoid)?
                  case (TVoid(), other) if other != TVoid[Pre]() => other
                  case (other, TVoid()) if other != TVoid[Pre]() => other
                }.foreach { t =>
                  val trigger: Local[Post] => Expr[Post] =
                    p => asType(t, asType(TVoid(), p, succ(adt)), succ(adt))
                  aDTDeclarations.declare(new ADTAxiom[Post](forall(
                    TAxiomatic(succ(adt), Nil),
                    body = p => { trigger(p) === asType(t, p, succ(adt)) },
                    triggers = p => Seq(Seq(trigger(p))),
                  )))
                }
              }._1
            }),
          )
        }
      case _ => super.postCoerce(decl)
    }
  }

  override def postCoerce(t: Type[Pre]): Type[Post] =
    t match {
      case TPointer(_) => TOption(TAxiomatic(pointerAdt.ref, Nil))
      case TNonNullPointer(_) => TAxiomatic(pointerAdt.ref, Nil)
      case other => super.postCoerce(other)
    }

  override def postCoerce(location: Location[Pre]): Location[Post] = {
    implicit val o: Origin = location.o
    location match {
      case loc @ PointerLocation(pointer) =>
        val arg =
          unwrapOption(pointer, loc.blame) match {
            case inv @ FunctionInvocation(ref, _, _, _, _)
                if ref.decl == pointerAdd.ref.decl =>
              inv
            case ptr =>
              FunctionInvocation[Post](
                ref = pointerAdd.ref,
                args = Seq(ptr, const(0)),
                typeArgs = Nil,
                Nil,
                Nil,
              )(PanicBlame("ptrAdd(ptr, 0) should be infallible"))
          }
        SilverFieldLocation(
          obj =
            FunctionInvocation[Post](
              ref = pointerDeref.ref,
              args = Seq(arg),
              typeArgs = Nil,
              Nil,
              Nil,
            )(PanicBlame("ptr_deref requires nothing."))(pointer.o),
          field = getPointerField(pointer),
        )
      case other => other.rewriteDefault()
    }
  }

  override def postCoerce(s: Statement[Pre]): Statement[Post] = {
    implicit val o: Origin = s.o
    s match {
      case scope: Scope[Pre] =>
        scope.rewrite(body = Block(scope.locals.collect {
          case v if v.t.isInstanceOf[TNonNullPointer[Pre]] => {
            val firstUse = scope.body.collectFirst {
              case l @ Local(Ref(variable)) if variable == v => l
            }
            if (
              firstUse.isDefined && scope.body.collectFirst {
                case Assign(l @ Local(Ref(variable)), _) if variable == v =>
                  System.identityHashCode(l) !=
                    System.identityHashCode(firstUse.get)
              }.getOrElse(true)
            ) {
              val oldT = v.t.asInstanceOf[TNonNullPointer[Pre]].element
              val newT = dispatch(oldT)
              Seq(
                InvokeProcedure[Post](
                  pointerCreationMethods.getOrElseUpdate(
                    oldT,
                    makePointerCreationMethod(oldT, newT),
                  ).ref,
                  Nil,
                  Seq(Local(succ(v))),
                  Nil,
                  Nil,
                  Nil,
                )(TrueSatisfiable)
              )
            } else { Nil }
          }
        }.flatten :+ dispatch(scope.body)))
      case _ => s.rewriteDefault()
    }
  }

  def rewriteTopLevelPointerSubscriptInTrigger(e: Expr[Pre]): Expr[Post] = {
    implicit val o: Origin = e.o
    e match {
      case sub @ PointerSubscript(pointer, index) =>
        FunctionInvocation[Post](
          ref = pointerDeref.ref,
          args = Seq(
            FunctionInvocation[Post](
              ref = pointerAdd.ref,
              args = Seq(unwrapOption(pointer, sub.blame), dispatch(index)),
              typeArgs = Nil,
              Nil,
              Nil,
            )(NoContext(PointerBoundsPreconditionFailed(sub.blame, index)))
          ),
          typeArgs = Nil,
          Nil,
          Nil,
        )(PanicBlame("ptr_deref requires nothing."))
      case deref @ DerefPointer(pointer) =>
        FunctionInvocation[Post](
          ref = pointerDeref.ref,
          args = Seq(
            if (
              inAxiom.isEmpty &&
              !deref.o.find[LabelContext]
                .exists(_.label == "classToRef cast helpers")
            ) {
              FunctionInvocation[Post](
                ref = pointerAdd.ref,
                // Always index with zero, otherwise quantifiers with pointers do not get triggered
                args = Seq(unwrapOption(pointer, deref.blame), const(0)),
                typeArgs = Nil,
                Nil,
                Nil,
              )(NoContext(
                DerefPointerBoundsPreconditionFailed(deref.blame, pointer)
              ))
            } else { unwrapOption(pointer, deref.blame) }
          ),
          typeArgs = Nil,
          Nil,
          Nil,
        )(PanicBlame("ptr_deref requires nothing."))
      case other => dispatch(other)
    }
  }

  override def postCoerce(e: Expr[Pre]): Expr[Post] = {
    implicit val o: Origin = e.o
    e match {
      case f @ Forall(_, triggers, _) =>
        f.rewrite(triggers =
          triggers.map(_.map(rewriteTopLevelPointerSubscriptInTrigger))
        )
      case s @ Starall(_, triggers, _) =>
        s.rewrite(triggers =
          triggers.map(_.map(rewriteTopLevelPointerSubscriptInTrigger))
        )
      case e @ Exists(_, triggers, _) =>
        e.rewrite(triggers =
          triggers.map(_.map(rewriteTopLevelPointerSubscriptInTrigger))
        )
      case sub @ PointerSubscript(pointer, index) =>
        SilverDeref(
          obj =
            FunctionInvocation[Post](
              ref = pointerDeref.ref,
              args = Seq(
                FunctionInvocation[Post](
                  ref = pointerAdd.ref,
                  args = Seq(unwrapOption(pointer, sub.blame), dispatch(index)),
                  typeArgs = Nil,
                  Nil,
                  Nil,
                )(NoContext(PointerBoundsPreconditionFailed(sub.blame, index)))
              ),
              typeArgs = Nil,
              Nil,
              Nil,
            )(PanicBlame("ptr_deref requires nothing.")),
          field = getPointerField(pointer),
        )(PointerFieldInsufficientPermission(sub.blame, sub))
      case add @ PointerAdd(pointer, offset) =>
        val inv =
          FunctionInvocation[Post](
            ref = pointerAdd.ref,
            args = Seq(unwrapOption(pointer, add.blame), dispatch(offset)),
            typeArgs = Nil,
            Nil,
            Nil,
          )(NoContext(PointerBoundsPreconditionFailed(add.blame, pointer)))
        pointer.t match {
          case TPointer(_) => OptSome(inv)
          case TNonNullPointer(_) => inv
        }
      case deref @ DerefPointer(pointer) =>
        SilverDeref(
          obj =
            FunctionInvocation[Post](
              ref = pointerDeref.ref,
              args = Seq(
                if (
                  inAxiom.isEmpty &&
                  !deref.o.find[LabelContext]
                    .exists(_.label == "classToRef cast helpers")
                ) {
                  FunctionInvocation[Post](
                    ref = pointerAdd.ref,
                    // Always index with zero, otherwise quantifiers with pointers do not get triggered
                    args = Seq(unwrapOption(pointer, deref.blame), const(0)),
                    typeArgs = Nil,
                    Nil,
                    Nil,
                  )(NoContext(
                    DerefPointerBoundsPreconditionFailed(deref.blame, pointer)
                  ))
                } else { unwrapOption(pointer, deref.blame) }
              ),
              typeArgs = Nil,
              Nil,
              Nil,
            )(PanicBlame("ptr_deref requires nothing.")),
          field = getPointerField(pointer),
        )(PointerFieldInsufficientPermission(deref.blame, deref))
      case len @ PointerBlockLength(pointer) =>
        ADTFunctionInvocation[Post](
          typeArgs = Some((blockAdt.ref, Nil)),
          ref = blockLength.ref,
          args = Seq(ADTFunctionInvocation[Post](
            typeArgs = Some((pointerAdt.ref, Nil)),
            ref = pointerBlock.ref,
            args = Seq(unwrapOption(pointer, len.blame)),
          )),
        )
      case off @ PointerBlockOffset(pointer) =>
        ADTFunctionInvocation[Post](
          typeArgs = Some((pointerAdt.ref, Nil)),
          ref = pointerOffset.ref,
          args = Seq(unwrapOption(pointer, off.blame)),
        )
      case pointerLen @ PointerLength(pointer) =>
        postCoerce(
          PointerBlockLength(pointer)(pointerLen.blame) -
            PointerBlockOffset(pointer)(pointerLen.blame)
        )
      case Cast(value, typeValue) if value.t.asPointer.isDefined =>
        // TODO: Check if types are compatible
        val targetType = typeValue.t.asInstanceOf[TType[Pre]].t
        val innerType = targetType.asPointer.get.element
        val newValue = dispatch(value)
        (targetType, value.t) match {
          case (TPointer(_), TPointer(_)) =>
            Select[Post](
              OptEmpty(newValue),
              OptNoneTyped(TAxiomatic(pointerAdt.ref, Nil)),
              OptSome(applyAsTypeFunction(
                innerType,
                value,
                OptGet(newValue)(PanicBlame(
                  "Can never be null since this is ensured in the conditional expression"
                )),
              )),
            )
          case (TNonNullPointer(_), TPointer(_)) =>
            applyAsTypeFunction(
              innerType,
              value,
              OptGet(newValue)(PanicBlame(
                "Casting a pointer to a non-null pointer implies the pointer must be statically known to be non-null"
              )),
            )
          case (TPointer(_), TNonNullPointer(_)) =>
            OptSome(applyAsTypeFunction(innerType, value, newValue))
          case (TNonNullPointer(_), TNonNullPointer(_)) =>
            applyAsTypeFunction(innerType, value, newValue)
        }
      case other => super.postCoerce(other)
    }
  }

  private def applyAsTypeFunction(
      innerType: Type[Pre],
      preExpr: Expr[Pre],
      postExpr: Expr[Post],
  )(implicit o: Origin): Expr[Post] = {
    asType(
      innerType,
      preExpr match {
        case PointerAdd(_, _) => postExpr
        // Don't add ptrAdd in an ADT axiom since we cannot use functions with preconditions there
        case _
            if inAxiom.nonEmpty ||
              !preExpr.o.find[LabelContext]
                .exists(_.label == "classToRef cast helpers") =>
          postExpr
        case _ =>
          FunctionInvocation[Post](
            ref = pointerAdd.ref,
            // Always index with zero, otherwise quantifiers with pointers do not get triggered
            args = Seq(postExpr, const(0)),
            typeArgs = Nil,
            Nil,
            Nil,
          )(PanicBlame(
            "Pointer out of bounds in pointer cast (no appropriate blame available)"
          ))
      },
    )
  }

  private def asType(
      t: Type[Pre],
      expr: Expr[Post],
      adt: Ref[Post, AxiomaticDataType[Post]] = pointerAdt.ref,
  )(implicit o: Origin): Expr[Post] = {
    functionInvocation[Post](
      PanicBlame("as_type requires nothing"),
      asTypeFunctions
        .getOrElseUpdate(t, makeAsTypeFunction(t.toString, adt = adt)).ref,
      Seq(expr),
    )
  }
}
