package vct.col.ast.expr.op.bool

import vct.col.ast.{ComputationalXor, TBool, Type}
import vct.col.print.{Ctx, Doc, Precedence}
import vct.col.ast.ops.ComputationalXorOps

trait ComputationalXorImpl[G] extends ComputationalXorOps[G] { this: ComputationalXor[G] =>
  override def t: Type[G] = TBool()

  override def precedence: Int = Precedence.BIT_XOR
  override def layout(implicit ctx: Ctx): Doc = lassoc(left, "^", right)
}