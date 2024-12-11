package vct.rewrite.adt

import hre.util.ScopedStack
import vct.col.ast.{
  Applicable,
  Asserting,
  BitAnd,
  BitNot,
  BitOr,
  BitShl,
  BitShr,
  BitUShr,
  BitXor,
  Declaration,
  Expr,
  Function,
  FunctionInvocation,
  Node,
  ProverFunction,
  ProverFunctionInvocation,
  SuccessorsProvider,
  SuccessorsProviderTrafo,
}
import vct.col.origin.{
  AssertFailed,
  Blame,
  IntegerOutOfBounds,
  Origin,
  PanicBlame,
  SourceName,
  TrueSatisfiable,
}
import vct.col.ref.Ref
import vct.col.rewrite.{Generation, NonLatchingRewriter}
import vct.col.rewrite.adt.{ImportADT, ImportADTBuilder, ImportADTImporter}
import vct.col.util.AstBuildHelpers.functionInvocation
import vct.result.VerificationError.UserError

import scala.collection.mutable
import scala.reflect.ClassTag

case object ImportBitVector extends ImportADTBuilder("bitvec") {
  private case class UnsupportedBitVectorSize(origin: Origin, bits: Int)
      extends UserError {

    override def code: String = "unsupportedBVSize"

    override def text: String =
      origin.messageInContext(
        s"Unsupported bit vector size `$bits` supported sizes are: 8, 16, 32, 64"
      )
  }

  private case class OutOfBoundsBlame(
      node: Node[_],
      blame: Blame[IntegerOutOfBounds],
  )(implicit val bits: Int)
      extends Blame[AssertFailed] {
    override def blame(error: AssertFailed): Unit =
      blame.blame(IntegerOutOfBounds(node, bits))
  }
}

case class ImportBitVector[Pre <: Generation](importer: ImportADTImporter)
    extends ImportADT[Pre](importer) {
  import ImportBitVector._

  private val functionMap: mutable.HashMap[String, Applicable[Post]] = mutable
    .HashMap()

  // TODO: These could easily be programmatically generated instead of fixing the sizes
  private lazy val bv8 = parse("bv8")
  private lazy val bv16 = parse("bv16")
  private lazy val bv32 = parse("bv32")
  private lazy val bv64 = parse("bv64")

  private lazy val u8_require_inbounds = find[Function[Post]](
    bv8,
    "u8_require_inbounds",
  )
  private lazy val u16_require_inbounds = find[Function[Post]](
    bv16,
    "u16_require_inbounds",
  )
  private lazy val u32_require_inbounds = find[Function[Post]](
    bv32,
    "u32_require_inbounds",
  )
  private lazy val u64_require_inbounds = find[Function[Post]](
    bv64,
    "u64_require_inbounds",
  )

  private lazy val u8_is_inbounds = find[Function[Post]](bv8, "u8_is_inbounds")
  private lazy val u16_is_inbounds = find[Function[Post]](
    bv16,
    "u16_is_inbounds",
  )
  private lazy val u32_is_inbounds = find[Function[Post]](
    bv32,
    "u32_is_inbounds",
  )
  private lazy val u64_is_inbounds = find[Function[Post]](
    bv64,
    "u64_is_inbounds",
  )

  private lazy val u8_assume_inbounds = find[Function[Post]](
    bv8,
    "u8_assume_inbounds",
  )
  private lazy val u16_assume_inbounds = find[Function[Post]](
    bv16,
    "u16_assume_inbounds",
  )
  private lazy val u32_assume_inbounds = find[Function[Post]](
    bv32,
    "u32_assume_inbounds",
  )
  private lazy val u64_assume_inbounds = find[Function[Post]](
    bv64,
    "u64_assume_inbounds",
  )

  private lazy val s8_is_inbounds = find[Function[Post]](bv8, "s8_is_inbounds")
  private lazy val s16_is_inbounds = find[Function[Post]](
    bv16,
    "s16_is_inbounds",
  )
  private lazy val s32_is_inbounds = find[Function[Post]](
    bv32,
    "s32_is_inbounds",
  )
  private lazy val s64_is_inbounds = find[Function[Post]](
    bv64,
    "s64_is_inbounds",
  )

  private lazy val s8_assume_inbounds = find[Function[Post]](
    bv8,
    "s8_assume_inbounds",
  )
  private lazy val s16_assume_inbounds = find[Function[Post]](
    bv16,
    "s16_assume_inbounds",
  )
  private lazy val s32_assume_inbounds = find[Function[Post]](
    bv32,
    "s32_assume_inbounds",
  )
  private lazy val s64_assume_inbounds = find[Function[Post]](
    bv64,
    "s64_assume_inbounds",
  )

  private def function[T <: Applicable[Post]](
      name: String
  )(implicit o: Origin, bits: Int, tag: ClassTag[T]): Ref[Post, T] = {
    val func = s"bv${bits}_$name"
    bits match {
      case 8 =>
        functionMap.getOrElseUpdate(func, find[T](bv8, func)).asInstanceOf[T]
          .ref
      case 16 =>
        functionMap.getOrElseUpdate(func, find[T](bv16, func)).asInstanceOf[T]
          .ref
      case 32 =>
        functionMap.getOrElseUpdate(func, find[T](bv32, func)).asInstanceOf[T]
          .ref
      case 64 =>
        functionMap.getOrElseUpdate(func, find[T](bv64, func)).asInstanceOf[T]
          .ref
      case _ => throw UnsupportedBitVectorSize(o, bits)
    }
  }

  private def to(
      e: Expr[Post],
      blame: Blame[IntegerOutOfBounds],
  )(implicit bits: Int, signed: Boolean): Expr[Post] = {
    implicit val o: Origin = e.o
    ProverFunctionInvocation(
      function(
        if (signed)
          "from_sint"
        else
          "from_uint"
      ),
      Seq(ensureInRange(e, blame)),
    )
  }

  private def from(
      e: Expr[Post]
  )(implicit bits: Int, signed: Boolean): Expr[Post] = {
    implicit val o: Origin = e.o
    if (signed) {
      FunctionInvocation(
        function[Function[Post]]("to_sint"),
        Seq(assumeInRange(e)),
        Nil,
        Nil,
        Nil,
      )(TrueSatisfiable)
    } else {
      ProverFunctionInvocation(function("to_uint"), Seq(assumeInRange(e)))
    }
  }

  private case class StripAsserting[G]() extends NonLatchingRewriter[G, G]() {
    case class SuccOrIdentity()
        extends SuccessorsProviderTrafo[G, G](allScopes) {
      override def postTransform[T <: Declaration[G]](
          pre: Declaration[G],
          post: Option[T],
      ): Option[T] = Some(post.getOrElse(pre.asInstanceOf[T]))
    }
    override def succProvider: SuccessorsProvider[G, G] = SuccOrIdentity()

    override def dispatch(e: Expr[G]): Expr[G] =
      e match {
        case Asserting(_, body) => dispatch(body)
        case _ => e.rewriteDefault()
      }
  }

  private def simplifyBV(e: Expr[Post])(implicit signed: Boolean) =
    e match {
      case ProverFunctionInvocation(
            Ref(r1),
            Seq(
              Asserting(
                _,
                ProverFunctionInvocation(Ref(r2), Seq(Asserting(_, inner))),
              )
            ),
          )
          if r1.o.find[SourceName].map(_.name).exists(n =>
            functionMap.contains(n) &&
              ((signed && n.endsWith("_from_sint")) ||
                (!signed && n.endsWith("_from_uint")))
          ) && r2.o.find[SourceName].map(_.name).exists(n =>
            functionMap.contains(n) &&
              ((signed && n.endsWith("_to_sint")) ||
                (!signed && n.endsWith("_to_uint")))
          ) =>
        inner
      case _ => e
    }

  private def ensureInRange(
      e: Expr[Post],
      blame: Blame[IntegerOutOfBounds],
  )(implicit bits: Int, signed: Boolean): Expr[Post] = {
    implicit val o: Origin = e.o
    val stripped = StripAsserting().dispatch(e)
    bits match {
      case 8 =>
        Asserting(
          functionInvocation[Post](
            TrueSatisfiable,
            if (signed)
              s8_is_inbounds.ref
            else
              u8_is_inbounds.ref,
            args = Seq(stripped),
          ),
          e,
        )(OutOfBoundsBlame(e, blame))
      case 16 =>
        Asserting(
          functionInvocation[Post](
            TrueSatisfiable,
            if (signed)
              s16_is_inbounds.ref
            else
              u16_is_inbounds.ref,
            args = Seq(stripped),
          ),
          e,
        )(OutOfBoundsBlame(e, blame))
      case 32 =>
        Asserting(
          functionInvocation[Post](
            TrueSatisfiable,
            if (signed)
              s32_is_inbounds.ref
            else
              u32_is_inbounds.ref,
            args = Seq(stripped),
          ),
          e,
        )(OutOfBoundsBlame(e, blame))
      case 64 =>
        Asserting(
          functionInvocation[Post](
            TrueSatisfiable,
            if (signed)
              s64_is_inbounds.ref
            else
              u64_is_inbounds.ref,
            args = Seq(stripped),
          ),
          e,
        )(OutOfBoundsBlame(e, blame))
      case _ => throw UnsupportedBitVectorSize(o, bits)
    }
  }

  private def assumeInRange(
      e: Expr[Post]
  )(implicit bits: Int, signed: Boolean): Expr[Post] = {
    implicit val o: Origin = e.o
    val stripped = StripAsserting().dispatch(e)
    bits match {
      case 8 =>
        Asserting(
          functionInvocation[Post](
            TrueSatisfiable,
            if (signed)
              s8_assume_inbounds.ref
            else
              u8_assume_inbounds.ref,
            args = Seq(stripped),
          ),
          e,
        )(PanicBlame("Assert true"))
      case 16 =>
        Asserting(
          functionInvocation[Post](
            TrueSatisfiable,
            if (signed)
              s16_assume_inbounds.ref
            else
              u16_assume_inbounds.ref,
            args = Seq(stripped),
          ),
          e,
        )(PanicBlame("Assert true"))
      case 32 =>
        Asserting(
          functionInvocation[Post](
            TrueSatisfiable,
            if (signed)
              s32_assume_inbounds.ref
            else
              u32_assume_inbounds.ref,
            args = Seq(stripped),
          ),
          e,
        )(PanicBlame("Assert true"))
      case 64 =>
        Asserting(
          functionInvocation[Post](
            TrueSatisfiable,
            if (signed)
              s64_assume_inbounds.ref
            else
              u64_assume_inbounds.ref,
            args = Seq(stripped),
          ),
          e,
        )(PanicBlame("Assert true"))
      case _ => throw UnsupportedBitVectorSize(o, bits)
    }
  }

  private def binOp(
      name: String,
      l: Expr[Pre],
      r: Expr[Pre],
      b: Int,
      s: Boolean,
      blame: Blame[IntegerOutOfBounds],
  )(implicit o: Origin): Expr[Post] = {
    implicit val bits: Int = b
    implicit val signed: Boolean = s
    from(ProverFunctionInvocation(
      function(name),
      Seq(
        simplifyBV(to(dispatch(l), blame)),
        simplifyBV(to(dispatch(r), blame)),
      ),
    ))
  }

  override def postCoerce(e: Expr[Pre]): Expr[Post] = {
    implicit val o: Origin = e.o
    e match {
      case op @ BitAnd(l, r, b, s) => binOp("and", l, r, b, s, op.blame)
      case op @ BitOr(l, r, b, s) => binOp("or", l, r, b, s, op.blame)
      case op @ BitXor(l, r, b, s) => binOp("xor", l, r, b, s, op.blame)
      case op @ BitShl(l, r, b, s) => binOp("shl", l, r, b, s, op.blame)
      case op @ BitShr(l, r, b) => binOp("shr", l, r, b, s = true, op.blame)
      case op @ BitUShr(l, r, b, s) => binOp("ushr", l, r, b, s, op.blame)
      case op @ BitNot(arg, b, s) =>
        implicit val bits: Int = b
        implicit val signed: Boolean = s
        from(ProverFunctionInvocation(
          function("not"),
          Seq(simplifyBV(to(dispatch(arg), op.blame))),
        ))
      case _ => super.postCoerce(e)
    }
  }
}
