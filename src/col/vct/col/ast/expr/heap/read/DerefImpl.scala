package vct.col.ast.expr.heap.read

import vct.col.ast.expr.ExprImpl
import vct.col.ast.{
  Deref,
  EndpointName,
  Expr,
  FieldLocation,
  TClass,
  TClassUnique,
  TPointer,
  TUnique,
  Type,
  Value,
}
import vct.col.check.{Check, CheckContext, CheckError, SeqProgReceivingEndpoint}
import vct.col.print.{Ctx, Doc, Group, Precedence}
import vct.col.ref.Ref
import vct.col.ast.ops.DerefOps

trait DerefImpl[G] extends ExprImpl[G] with DerefOps[G] {
  this: Deref[G] =>
  override def t: Type[G] = obj.t match {
      case tc@TClassUnique(inner, uniqueMap) =>
        uniqueMap.collectFirst {case (fieldRef, unique) if ref.decl == fieldRef.decl => addUniquePointer(inner, unique) }
          .getOrElse(getT(tc))
      case t => getT(t)
    }

  def getT(classT: Type[G]): Type[G] = {
    classT.asClass.map(_.instantiate(ref.decl.t)).getOrElse(ref.decl.t)
  }

  def addUniquePointer(cls: Ref[G, vct.col.ast.Class[G]], unique: BigInt): Type[G] = {
    getT(TClass(cls, Seq())) match {
      case TPointer(inner) => TPointer(TUnique(inner, unique))
      case _ => ???
    }
  }

  override def check(context: CheckContext[G]): Seq[CheckError] =
    Check.inOrder(
      super.check(context),
      obj.t.asClass.get.cls.decl.checkDefines(ref.decl, this),
    )

  override def precedence: Int = Precedence.POSTFIX
  override def layout(implicit ctx: Ctx): Doc =
    assoc(obj) <> "." <> ctx.name(ref)

  def value: Value[G] = Value(FieldLocation(obj, ref))
}
