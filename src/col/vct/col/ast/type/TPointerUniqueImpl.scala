package vct.col.ast.`type`

import vct.col.ast.TPointerUnique
import vct.col.ast.ops.TPointerUniqueOps
import vct.col.print._

trait TPointerUniqueImpl[G] extends TPointerUniqueOps[G]{
  this: TPointerUnique[G] =>
  override def layoutSplitDeclarator(implicit ctx: Ctx): (Doc, Doc) = {
    val (spec, decl) = element.layoutSplitDeclarator
    (spec, decl <> "*")
  }

  override def layout(implicit ctx: Ctx): Doc =
    Group(Text("unique_pointer") <> open <> element <> "," <> id.toString <> close)
}
