package vct.col.ast.lang.c

import vct.col.ast.CTArray
import vct.col.print.{Ctx, Doc, Group}
import vct.col.ast.ops.CTArrayOps
import vct.col.typerules.TypeSize

trait CTArrayImpl[G] extends CTArrayOps[G] {
  this: CTArray[G] =>
  // TODO: Parameterize on target
  override def byteSize: TypeSize = TypeSize.Exact(BigInt.int2bigInt(8))
  override def layout(implicit ctx: Ctx): Doc =
    Group(innerType.show <> "[" <> Doc.args(size.toSeq) <> "]")
}
