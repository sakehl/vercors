package vct.col.ast.`type`

import vct.col.ast.TChar
import vct.col.print.{Ctx, Doc, Group, Text}
import vct.col.ast.ops.TCharOps
import vct.col.typerules.TypeSize

trait TCharImpl[G] extends TCharOps[G] {
  this: TChar[G] =>
  // TODO: Parameterize on target
  override def byteSize: TypeSize = TypeSize.Exact(BigInt.int2bigInt(1))
  override def layout(implicit ctx: Ctx): Doc = Text("char")
}
