package vct.col.ast.expr.literal.constant

import vct.col.ast.{CIntegerValue, TCInt, CPrimitiveType, Type}
import vct.col.print.{Ctx, Doc, Precedence, Text}
import vct.col.ast.ops.CIntegerValueOps

trait CIntegerValueImpl[G] extends CIntegerValueOps[G] {
  this: CIntegerValue[G] =>

  assert(t.isInstanceOf[TCInt[G]] || t.isInstanceOf[CPrimitiveType[G]])

  override def precedence: Int = Precedence.ATOMIC
  override def layout(implicit ctx: Ctx): Doc = Text(value.toString())
}
