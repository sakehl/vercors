package vct.col.ast.lang

import vct.col.ast.{CLiteralArray, TNotAValue, Type}
import vct.col.print.Doc.{arg, fold}
import vct.col.print.{Ctx, Doc, Precedence, Text}

trait CLiteralArrayImpl[G] { this: CLiteralArray[G] =>
  override def t: Type[G] = new TNotAValue()

  override def precedence = Precedence.ATOMIC

  override def layout(implicit ctx: Ctx): Doc = Text("{") <> arg(fold(exprs)(_ <> "," <+> _)) <> "}"
}