package vct.col.ast.`type`

import vct.col.ast.{TConst, TNonNullConstPointer, Type}
import vct.col.ast.ops.TNonNullConstPointerOps
import vct.col.print._

trait TNonNullConstPointerImpl[G] extends TNonNullConstPointerOps[G] { this: TNonNullConstPointer[G] =>
  val element: Type[G] = TConst[G](pureElement)
  val unique = None
  override def layout(implicit ctx: Ctx): Doc =
    Text("constNonNullPointer") <> open <> pureElement <> close
}
