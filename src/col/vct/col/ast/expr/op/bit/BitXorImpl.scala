package vct.col.ast.expr.op.bit

import vct.col.ast.{BitXor, TCInt, TInt, Type}
import vct.col.print.{Ctx, Doc, Precedence}
import vct.col.ast.ops.BitXorOps

trait BitXorImpl[G] extends BitXorOps[G] {
  this: BitXor[G] =>
  override def t: Type[G] =
    getNumericType match {
      case t: TCInt[G] if t.bits.isEmpty && bits != 0 =>
        t.bits = Some(bits)
        t
      case t => t
    }

  override def precedence: Int = Precedence.BIT_XOR
  override def layout(implicit ctx: Ctx): Doc = lassoc(left, "^", right)
}
