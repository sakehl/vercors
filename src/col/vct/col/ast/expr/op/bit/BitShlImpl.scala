package vct.col.ast.expr.op.bit

import vct.col.ast.{BitShl, TCInt, TInt, Type}
import vct.col.print.{Ctx, Doc, Precedence}
import vct.col.ast.ops.BitShlOps

trait BitShlImpl[G] extends BitShlOps[G] {
  this: BitShl[G] =>
  override def t: Type[G] =
    getNumericType match {
      case t: TCInt[G] if t.bits.isEmpty && bits != 0 =>
        t.bits = Some(bits)
        t
      case t => t
    }

  override def precedence: Int = Precedence.SHIFT
  override def layout(implicit ctx: Ctx): Doc = lassoc(left, "<<", right)
}
