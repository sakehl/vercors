package vct.col.ast.expr.veymont

import vct.col.ast.ops.EndpointExprOps
import vct.col.ast.{EndpointExpr, Type}
import vct.col.print._

trait EndpointExprImpl[G] extends EndpointExprOps[G] {
  this: EndpointExpr[G] =>
  override def layout(implicit ctx: Ctx): Doc =
    Text("(\\[") <> ctx.name(endpoint) <> "]" <+> expr <> ")"
  override def precedence: Int = Precedence.ATOMIC

  override def t: Type[G] = expr.t
}
