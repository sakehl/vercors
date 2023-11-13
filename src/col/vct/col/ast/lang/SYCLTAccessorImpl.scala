package vct.col.ast.lang

import vct.col.ast.SYCLTAccessor
import vct.col.print.{Ctx, Doc, Group, Text}

trait SYCLTAccessorImpl[G] { this: SYCLTAccessor[G] =>
  override def layout(implicit ctx: Ctx): Doc =
    Group(Text("sycl::accessor") <> "<" <> typ <> ", " <> Text(dimCount.toString) <> ">")

  override val namespacePath = "sycl::accessor"
}