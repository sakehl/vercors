package vct.col.ast.expr.heap.read

import vct.col.ast.{AddrOf, TPointer, Type}
import vct.col.print._
import vct.col.ast.ops.AddrOfOps

trait AddrOfImpl[G] extends AddrOfOps[G] {
  this: AddrOf[G] =>
  // TODO LvdH: Need to do something with uniqueness here probably
  override def t: Type[G] = TPointer(e.t, None)

  override def precedence: Int = Precedence.PREFIX
  override def layout(implicit ctx: Ctx): Doc = Text("&") <> assoc(e)
}
