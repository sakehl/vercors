package vct.col.ast.`type`

import scala.jdk.CollectionConverters._
import vct.col.ast.stmt.decl.ProgramUnit
import vct.col.ast.util.{ASTMapping, ASTMapping1, ASTVisitor, TypeMapping, VisitorHelper}
import vct.col.ast.util.VisitorHelper

object TypeExpression {
  /* These operators don't affect the type that VerCors thinks the inner type is, but rather specify something about
    the way they are stored (const, static) or are ignored (short, unsigned) */
  val LEAKY_OPERATORS = Set(
    TypeOperator.Const,
    TypeOperator.Short,
    TypeOperator.Signed,
    TypeOperator.Long,
    TypeOperator.Unsigned,
    TypeOperator.Global,
    TypeOperator.Local,
    TypeOperator.Extern,
    TypeOperator.Static,
  )
}

case class TypeExpression(val operator:TypeOperator, val types:List[Type]) extends Type with VisitorHelper {
  require(types != null, "The types list is null")

  /** Constructs a new type expression from an array of types */
  def this(operator:TypeOperator, types:Array[Type]) = this(operator, types.toList)

  private def isLeaky: Boolean = TypeExpression.LEAKY_OPERATORS.contains(operator)

  /** Gives the heading type in the type list */
  def firstType = types.head

  /** Provides a Java wrapper (as `java.util.List`) over the types list (`types`) */
  def typesJava = types.asJava

  override def isPrimitive(fraction:PrimitiveSort): Boolean =
    if (isLeaky) firstType.isPrimitive(fraction:PrimitiveSort) else false

  override def isFraction =
    if (isLeaky) firstType.isFraction else false

  override def isBoolean =
    if (isLeaky) firstType.isBoolean else false

  override def isInteger =
    if (isLeaky) firstType.isInteger else false

  override def isDouble =
    if (isLeaky) firstType.isDouble else false

  override def isVoid =
    if (isLeaky) firstType.isVoid else false

  override def isNull =
    if (isLeaky) firstType.isNull else false

  override def isIntegerType =
    if (isLeaky) firstType.isIntegerType else false

  override def isNumeric =
    if (isLeaky) firstType.isNumeric else false

  override def isResource =
    if (isLeaky) firstType.isResource else false

  override def supertypeof(context: ProgramUnit, t: Type): Boolean =
    if (isLeaky) firstType.supertypeof(context, t) else false

  override def accept_simple[T,A](m:ASTMapping1[T,A], arg:A) = m.map(this, arg)
  override def accept_simple[T](v:ASTVisitor[T]) = handle_standard(() => v.visit(this))
  override def accept_simple[T](m:ASTMapping[T]) = handle_standard(() => m.map(this))
  override def accept_simple[T](m:TypeMapping[T]) = handle_standard(() => m.map(this))

  override def debugTreeChildrenFields: Iterable[String] = Seq("types", "args")
  override def debugTreePropertyFields: Iterable[String] = Seq("operator")
}
