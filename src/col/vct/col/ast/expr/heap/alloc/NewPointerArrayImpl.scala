package vct.col.ast.expr.heap.alloc

import vct.col.ast.{NewPointerArray, TPointer, TPointerUnique, Type}
import vct.col.print._
import vct.col.ast.ops.NewPointerArrayOps

trait NewPointerArrayImpl[G] extends NewPointerArrayOps[G] {
  this: NewPointerArray[G] =>
  override lazy val t: Type[G] = unique.map(TPointerUnique[G](element,_))
    .getOrElse(TPointer[G](element))
}
