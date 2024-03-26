package vct.col.rewrite.adt

import vct.col.ast._
import ImportADT.typeText
import vct.col.origin._
import vct.col.ref.Ref
import vct.col.rewrite.Generation
import vct.col.rewrite.adt.ImportPointer.{DerefPointerBoundsPreconditionFailed, PointerBoundsPreconditionFailed, PointerFieldInsufficientPermission, PointerNullOptNone}
import vct.col.util.AstBuildHelpers.{ExprBuildHelpers, const}

import scala.collection.mutable

case object ImportPointer extends ImportADTBuilder("pointer") {
  private def PointerField(t: Type[_]): Origin = Origin(
    Seq(
      PreferredName(Seq(typeText(t))),
      LabelContext("pointer field"),
    )
  )

  case class PointerNullOptNone(inner: Blame[PointerNull], expr: Expr[_]) extends Blame[OptionNone] {
    override def blame(error: OptionNone): Unit =
      inner.blame(PointerNull(expr))
  }

  case class PointerBoundsPreconditionFailed(inner: Blame[PointerBounds], expr: Node[_]) extends Blame[PreconditionFailed] {
    override def blame(error: PreconditionFailed): Unit =
      inner.blame(PointerBounds(expr))
  }

  case class DerefPointerBoundsPreconditionFailed(inner: Blame[PointerDerefError], expr: Expr[_]) extends Blame[PreconditionFailed] {
    override def blame(error: PreconditionFailed): Unit =
      inner.blame(PointerInsufficientPermission(expr))
  }


  case class PointerFieldInsufficientPermission(inner: Blame[PointerInsufficientPermission], expr: Expr[_]) extends Blame[InsufficientPermission] {
    override def blame(error: InsufficientPermission): Unit =
      inner.blame(PointerInsufficientPermission(expr))
  }
}

case class ImportPointer[Pre <: Generation](importer: ImportADTImporter) extends ImportADT[Pre](importer) {
  import ImportPointer._

  private lazy val pointerFile = parse("pointer")

  private lazy val blockAdt = find[AxiomaticDataType[Post]](pointerFile, "block")
  private lazy val blockBase = find[ADTFunction[Post]](blockAdt, "base_addr")
  private lazy val blockLength = find[ADTFunction[Post]](blockAdt, "block_length")
  private lazy val blockLoc = find[ADTFunction[Post]](blockAdt, "loc")
  private lazy val pointerAdt = find[AxiomaticDataType[Post]](pointerFile, "pointer")
  private lazy val pointerOf = find[ADTFunction[Post]](pointerAdt, "pointer_of")
  private lazy val pointerBlock = find[ADTFunction[Post]](pointerAdt, "pointer_block")
  private lazy val pointerOffset = find[ADTFunction[Post]](pointerAdt, "pointer_offset")
  private lazy val pointerDeref = find[Function[Post]](pointerFile, "ptr_deref")
  private lazy val pointerAdd = find[Function[Post]](pointerFile, "ptr_add")

  val pointerField: mutable.Map[Type[Post], SilverField[Post]] = mutable.Map()

  private def getPointerField(ptr: Expr[Pre]): Ref[Post, SilverField[Post]] = {
    val tElement = dispatch(ptr.t.asPointer.get.element)
    pointerField.getOrElseUpdate(tElement, {
      globalDeclarations.declare(new SilverField(tElement)(PointerField(tElement)))
    }).ref
  }

  override def applyCoercion(e: => Expr[Post], coercion: Coercion[Pre])(implicit o: Origin): Expr[Post] = coercion match {
    case CoerceNullPointer(_) => OptNone()
    case other => super.applyCoercion(e, other)
  }

  override def postCoerce(t: Type[Pre]): Type[Post] = t match {
    case TPointer(_) => TOption(TAxiomatic(pointerAdt.ref, Nil))
    case other => rewriteDefault(other)
  }

  override def postCoerce(location: Location[Pre]): Location[Post] = location match {
    case loc@PointerLocation(pointer) =>
      SilverFieldLocation(
        obj = FunctionInvocation[Post](
          ref = pointerDeref.ref,
          args = Seq(OptGet(dispatch(pointer))(PointerNullOptNone(loc.blame, pointer))(pointer.o)),
          typeArgs = Nil, Nil, Nil,
        )(PanicBlame("ptr_deref requires nothing."))(pointer.o),
        field = getPointerField(pointer),
      )(loc.o)
    case other => rewriteDefault(other)
  }

  override def postCoerce(e: Expr[Pre]): Expr[Post] = {
    implicit val o: Origin = e.o
    e match {
      case sub@PointerSubscript(pointer, index) =>
        SilverDeref(
          obj = FunctionInvocation[Post](
            ref = pointerDeref.ref,
            args = Seq(FunctionInvocation[Post](
              ref = pointerAdd.ref,
              args = Seq(OptGet(dispatch(pointer))(PointerNullOptNone(sub.blame, pointer)), dispatch(index)),
              typeArgs = Nil, Nil, Nil)(NoContext(PointerBoundsPreconditionFailed(sub.blame, index)))),
            typeArgs = Nil, Nil, Nil,
          )(PanicBlame("ptr_deref requires nothing.")),
          field = getPointerField(pointer),
        )(PointerFieldInsufficientPermission(sub.blame, sub))
      case add@PointerAdd(pointer, offset) =>
        OptSome(FunctionInvocation[Post](
          ref = pointerAdd.ref,
          args = Seq(OptGet(dispatch(pointer))(PointerNullOptNone(add.blame, pointer)), dispatch(offset)),
          typeArgs = Nil, Nil, Nil,
        )(NoContext(PointerBoundsPreconditionFailed(add.blame, pointer))))
      case deref@DerefPointer(pointer) =>
        SilverDeref(
          obj = FunctionInvocation[Post](
            ref = pointerDeref.ref,
            args = Seq(FunctionInvocation[Post](
              ref = pointerAdd.ref,
              // Always index with zero, otherwise quantifiers with pointers do not get triggered
              args = Seq(OptGet(dispatch(pointer))(PointerNullOptNone(deref.blame, pointer)), const(0)),
              typeArgs = Nil, Nil, Nil)(NoContext(DerefPointerBoundsPreconditionFailed(deref.blame, pointer)))),
            typeArgs = Nil, Nil, Nil,
          )(PanicBlame("ptr_deref requires nothing.")),
          field = getPointerField(pointer),
        )(PointerFieldInsufficientPermission(deref.blame, deref))
      case len@PointerBlockLength(pointer) =>
        ADTFunctionInvocation[Post](
          typeArgs = Some((blockAdt.ref, Nil)),
          ref = blockLength.ref,
          args = Seq(ADTFunctionInvocation[Post](
            typeArgs = Some((pointerAdt.ref, Nil)),
            ref = pointerBlock.ref,
            args = Seq(OptGet(dispatch(pointer))(PointerNullOptNone(len.blame, pointer)))
          ))
        )
      case off@PointerBlockOffset(pointer) =>
        ADTFunctionInvocation[Post](
          typeArgs = Some((pointerAdt.ref, Nil)),
          ref = pointerOffset.ref,
          args = Seq(OptGet(dispatch(pointer))(PointerNullOptNone(off.blame, pointer)))
        )
      case pointerLen@PointerLength(pointer) =>
        postCoerce(PointerBlockLength(pointer)(pointerLen.blame) - PointerBlockOffset(pointer)(pointerLen.blame))
      case other => rewriteDefault(other)
    }
  }
}


object ImportArrayPointer extends ImportADTBuilder("array") {
  private def ArrayField(t: Type[_]): Origin = Origin(
    Seq(
      PreferredName(Seq(typeText(t))),
      LabelContext("array pointer field"),
    )
  )

  case class ArrayNullOptNone(inner: Blame[ArrayNull], expr: Expr[_]) extends Blame[OptionNone] {
    override def blame(error: OptionNone): Unit =
      inner.blame(ArrayNull(expr))
  }

  case class ArrayBoundsPreconditionFailed(inner: Blame[ArrayBounds], subscript: Node[_]) extends Blame[PreconditionFailed] {
    override def blame(error: PreconditionFailed): Unit =
      inner.blame(ArrayBounds(subscript))
  }

  case class ArrayFieldInsufficientPermission(inner: Blame[ArrayInsufficientPermission], expr: Expr[_]) extends Blame[InsufficientPermission] {
    override def blame(error: InsufficientPermission): Unit =
      inner.blame(ArrayInsufficientPermission(expr))
  }
}

case class ImportArrayPointer[Pre <: Generation](importer: ImportADTImporter) extends ImportADT[Pre](importer) {
  import ImportArrayPointer._

  var inTriggers: Boolean = false

  private lazy val arrayFile = parse("array")

  private lazy val arrayAdt = find[AxiomaticDataType[Post]](arrayFile, "array")
  private lazy val arrayAxLoc = find[ADTFunction[Post]](arrayAdt, "array_loc")
  private lazy val arrayLen = find[ADTFunction[Post]](arrayAdt, "alen")
  private lazy val arrayLoc = find[Function[Post]](arrayFile, "aloc")

  val arrayField: mutable.Map[Type[Post], SilverField[Post]] = mutable.Map()

  private def getArrayField(ptr: Expr[Pre]): Ref[Post, SilverField[Post]] = {
    val tElement = dispatch(ptr.t.asPointer.get.element)
    arrayField.getOrElseUpdate(tElement, {
      globalDeclarations.declare(new SilverField(tElement)(ArrayField(tElement)))
    }).ref
  }

  override def applyCoercion(e: => Expr[Post], coercion: Coercion[Pre])(implicit o: Origin): Expr[Post] = coercion match {
    case CoerceNullPointer(_) => OptNoneTyped(TAxiomatic(arrayAdt.ref, Nil))
    case other => super.applyCoercion(e, other)
  }

  override def postCoerce(t: Type[Pre]): Type[Post] = t match {
    case TPointer(_) => TOption(TAxiomatic(arrayAdt.ref, Nil))
    case other => rewriteDefault(other)
  }

  override def postCoerce(location: Location[Pre]): Location[Post] = location match {
    case loc@PointerLocation(pointer) =>
      val pointerInner = pointer match {
        case ApplyCoercion(inner, CoerceIdentity(_)) => inner
        case _ => ???
      }

      val (arr, index, boundsBlame) = pointerInner match {
        case add@PointerAdd(arr, index) => (arr, index, PointerBoundsPreconditionFailed(add.blame, pointer)) // Maybe just do not support PointerAdd, since that could lead to problems
        case AddrOf(deref@DerefPointer(arr)) => (arr, const[Pre](0)(location.o), DerefPointerBoundsPreconditionFailed(deref.blame, pointer))
        case AddrOf(sub@PointerSubscript(arr, index)) => (arr, index, PointerBoundsPreconditionFailed(sub.blame, index))
        case l: Local[Pre] => (l, const[Pre](0)(location.o), PanicBlame("TODO: PointerLocation zero"))
        case _ => ???
      }

      SilverFieldLocation(
        obj = FunctionInvocation[Post](
          ref = arrayLoc.ref,
          args = Seq(
            OptGet(dispatch(arr))(PointerNullOptNone(loc.blame, pointer))(arr.o),
            dispatch(index)),
          typeArgs = Nil, Nil, Nil)(NoContext(boundsBlame))(loc.o),
        field = getArrayField(arr),
      )(loc.o)
    case other => rewriteDefault(other)
  }
  override def postCoerce(e: Expr[Pre]): Expr[Post] = {
    implicit val o: Origin = e.o
    e match {
      case sub@PointerSubscript(arr, index) =>
        SilverDeref(
          FunctionInvocation[Post](
            ref = arrayLoc.ref,
            args = Seq(
              OptGet(dispatch(arr))(PointerNullOptNone(sub.blame, arr))(arr.o),
              dispatch(index)),
            typeArgs = Nil, Nil, Nil)(NoContext(PointerBoundsPreconditionFailed(sub.blame, sub))),
          field = getArrayField(arr))(PointerFieldInsufficientPermission(sub.blame, sub))
      case s@Starall(bindings, triggers, body) =>
        inTriggers = true
        val newTriggers = triggers.map(s => s.map(t => dispatch(t)))
        inTriggers = false
        Starall(variables.dispatch(bindings), newTriggers, dispatch(body))(s.blame)(s.o)
      case f@Forall(bindings, triggers, body) =>
        inTriggers = true
        val newTriggers = triggers.map(s => s.map(t => dispatch(t)))
        inTriggers = false
        Forall(variables.dispatch(bindings), newTriggers, dispatch(body))(f.o)

      case add@PointerAdd(pointer, offset) if inTriggers =>
        SilverDeref(
          FunctionInvocation[Post](
            ref = arrayLoc.ref,
            args = Seq(
              OptGet(dispatch(pointer))(PointerNullOptNone(add.blame, pointer))(pointer.o),
              dispatch(offset)),
            typeArgs = Nil, Nil, Nil)(NoContext(PointerBoundsPreconditionFailed(add.blame, add))),
          field = getArrayField(pointer))(PanicBlame("TODO: Get array field no permission?"))
      case add@PointerAdd(pointer, offset) =>
        ???
      case deref@DerefPointer(arr) => SilverDeref(
        FunctionInvocation[Post](
          ref = arrayLoc.ref,
          args = Seq(
            OptGet(dispatch(arr))(PointerNullOptNone(deref.blame, arr))(arr.o),
            dispatch(const(0))),
          typeArgs = Nil, Nil, Nil)(NoContext(DerefPointerBoundsPreconditionFailed(deref.blame, deref))),
        field = getArrayField(arr))(PointerFieldInsufficientPermission(deref.blame, deref))
      case length@PointerLength(arr) =>
        ADTFunctionInvocation(None, arrayLen.ref, Seq(
          OptGet(dispatch(arr))(PointerNullOptNone(length.blame, arr))(arr.o)
        ))
      case length@PointerBlockLength(arr) =>
        ADTFunctionInvocation(None, arrayLen.ref, Seq(
          OptGet(dispatch(arr))(PointerNullOptNone(length.blame, arr))(arr.o)
        ))
      case length@PointerBlockOffset(arr) =>
        const[Post](0)(length.o)
      case other => rewriteDefault(other)
    }
  }

}
