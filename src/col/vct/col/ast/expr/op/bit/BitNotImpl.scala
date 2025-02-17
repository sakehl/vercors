package vct.col.ast.expr.op.bit

import vct.col.ast.{BitNot, TCInt, TInt, Type}
import vct.col.print.{Ctx, Doc, Precedence, Text}
import vct.col.typerules.CoercionUtils
import vct.col.ast.ops.BitNotOps

trait BitNotImpl[G] extends BitNotOps[G] {
  this: BitNot[G] =>
  override def t: Type[G] =
    arg.t match {
      case cint @ TCInt() => cint
      case _ if CoercionUtils.getCoercion(arg.t, TCInt()).isDefined => TCInt()
      case _ => TInt()
    }

  override def precedence: Int = Precedence.PREFIX
  override def layout(implicit ctx: Ctx): Doc = Text("~") <> assoc(arg)
}
