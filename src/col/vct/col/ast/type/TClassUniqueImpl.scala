package vct.col.ast.`type`

import vct.col.ast.TClassUnique
import vct.col.ast.ops.TClassUniqueOps
import vct.col.print._

trait TClassUniqueImpl[G] extends TClassUniqueOps[G] { this: TClassUnique[G] =>
  override def layout(implicit ctx: Ctx): Doc = {
    uniqueMap.foldLeft[Doc](Text(ctx.name(cls))){case (i, (fref, unique)) =>
      Text("unique_class_field<") <> ctx.name(fref.decl) <> "," <> unique.toString <> ">" <+> i}
  }
}
