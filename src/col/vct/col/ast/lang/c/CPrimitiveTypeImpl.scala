package vct.col.ast.lang.c

import vct.col.ast.CPrimitiveType
import vct.col.print.{Ctx, Doc}
import vct.col.ast.ops.CPrimitiveTypeOps
import vct.col.resolve.lang.C
import vct.col.typerules.TypeSize

trait CPrimitiveTypeImpl[G] extends CPrimitiveTypeOps[G] {
  this: CPrimitiveType[G] =>
  // TODO: Parameterize on target
  override def byteSize: TypeSize =
    C.getIntSize(C.LP64, specifiers).map(n => TypeSize.Exact(n / 8))
      .getOrElse(TypeSize.Unknown())
  override def layout(implicit ctx: Ctx): Doc = Doc.spread(specifiers)
}
