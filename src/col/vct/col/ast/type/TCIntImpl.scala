package vct.col.ast.`type`

import vct.col.ast.TCInt
import vct.col.print.{Ctx, Doc, Text}
import vct.col.ast.ops.TCIntOps
import vct.col.typerules.TypeSize

trait TCIntImpl[G] extends TCIntOps[G] {
  this: TCInt[G] =>
  override def byteSize: TypeSize =
    if (bits == 0) { TypeSize.Unknown() }
    else { TypeSize.Exact(BigInt.int2bigInt(bits / 8)) }
  override def layout(implicit ctx: Ctx): Doc = Text("int")
}
