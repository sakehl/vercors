package vct.col.ast.lang.llvm

import vct.col.ast.node.NodeFamilyImpl
import vct.col.ast.ops.LLVMResultOps
import vct.col.ast.{LLVMResult, Type}
import vct.col.print.Precedence

trait LLVMResultImpl[G] extends NodeFamilyImpl[G] with LLVMResultOps[G] {
  this: LLVMResult[G] =>
  override def t: Type[G] = func.decl.returnType

  override def precedence: Int = Precedence.ATOMIC
}
