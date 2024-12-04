package vct.col.ast.lang.c

import vct.col.ast.CTPointer
import vct.col.print.{Ctx, Doc, Text}
import vct.col.ast.ops.CTPointerOps
import vct.col.typerules.TypeSize

trait CTPointerImpl[G] extends CTPointerOps[G] {
  this: CTPointer[G] =>
  // TODO: Parameterize on target
  override def byteSize: TypeSize = TypeSize.Exact(BigInt.int2bigInt(8))
  override def layout(implicit ctx: Ctx): Doc =
    Text("pointer") <> open <> innerType <> close
}
