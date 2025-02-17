package vct.col.ast.family.coercion

import vct.col.ast.{CoerceToUniquePointer, TPointerUnique, Type}
import vct.col.ast.ops.CoerceToUniquePointerOps
import vct.col.print._

trait CoerceToUniquePointerImpl[G] extends CoerceToUniquePointerOps[G] { this: CoerceToUniquePointer[G] =>

}
