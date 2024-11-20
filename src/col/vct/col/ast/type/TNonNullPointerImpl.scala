package vct.col.ast.`type`

import vct.col.ast.TNonNullPointer
import vct.col.ast.ops.TNonNullPointerOps
import vct.col.print._
import vct.col.typerules.TypeSize

trait TNonNullPointerImpl[G] extends TNonNullPointerOps[G] {
  this: TNonNullPointer[G] =>
  // TODO: Parameterize on target
  override def byteSize(): TypeSize = TypeSize.Exact(BigInt.int2bigInt(8))
  override def layoutSplitDeclarator(implicit ctx: Ctx): (Doc, Doc) = {
    val (spec, decl) = element.layoutSplitDeclarator
    (spec, decl <> "*")
  }

  override def layout(implicit ctx: Ctx): Doc =
    Group(Text("NonNull") <> open <> element <> close)
}
