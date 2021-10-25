package vct.col.rewrite

import vct.col.ast.`type`.{ASTReserved, PrimitiveSort, Type}
import vct.col.ast.expr.{Binder, BindingExpression, OperatorExpression, StandardOperator}
import vct.col.ast.generic.ASTNode
import vct.col.ast.stmt.decl.{DeclarationStatement, ProgramUnit}
import vct.col.ast.util.{ASTUtils, AbstractRewriter, SequenceUtils}

import scala.collection.mutable
import scala.jdk.CollectionConverters.IterableHasAsScala

class DesugarValidPointer(source: ProgramUnit) extends AbstractRewriter(source) {
  override def visit(exp: OperatorExpression): Unit = {
    exp.operator match {
      case StandardOperator.ValidPointer =>
        val t = exp.arg(0).getType
        val array = rewrite(exp.arg(0))
        val size = rewrite(exp.arg(1))
        val perm = rewrite(exp.arg(2))
        result = validPointerFor(array, t, size, perm)
      case StandardOperator.ValidPointerIndex =>
        val t = exp.arg(0).getType
        val array = rewrite(exp.arg(0))
        val index = rewrite(exp.arg(1))
        val perm = rewrite(exp.arg(2))
        result = validPointerIndexFor(array, t, index, perm)
      case StandardOperator.PointsTo =>
        val variable = rewrite(exp.first)
        val permission = rewrite(exp.second)
        val value = rewrite(exp.third)
        result = create expression(StandardOperator.Star,
          create.expression(StandardOperator.Perm, variable, permission),
          create.expression(StandardOperator.EQ, variable, value)
        )
      case _ =>
        super.visit(exp)
    }
  }

  override def visit(e: BindingExpression): Unit = {
    e.binder match {
      case Binder.Forall | Binder.Star =>
        val new_main = rewrite(e.main)
        val res_type = rewrite(e.result_type)
        val res_decl = rewrite(e.getDeclarations)
        val res_triggers = rewrite(e.javaTriggers)
        val res_select = rewrite(e.select)
        var res_forall: ASTNode = null
        for (clause <- ASTUtils.conjuncts(new_main, StandardOperator.Star).asScala) {
          if (res_forall == null)
            res_forall = create.binder(e.binder, res_type, res_decl, res_triggers, res_select, clause)
          else
            res_forall = create expression(StandardOperator.Star,
              res_forall,
              create.binder(e.binder, res_type, res_decl, res_triggers, res_select, clause)
              )
        }
        result = res_forall
      case _ => super.visit(e)
    }
  }

  def validPointerFor(input: ASTNode, t: Type, size: ASTNode, perm: ASTNode): ASTNode = {
    val conditions: mutable.ListBuffer[ASTNode] = mutable.ListBuffer()
    val seqInfo = SequenceUtils.expectArrayType(t, "Expected an array type here, but got %s")

    if(!seqInfo.isOpt || !seqInfo.isCell) {
      Fail("Expected a pointer type here, but got %s", t)
    }

    val value = input

    conditions += neq(value, create.reserved_name(ASTReserved.OptionNone))
    conditions += lte(size, create.dereference(value, "length"))

    conditions += create.starall(
      and(lte(constant(0), name("__i")), less(name("__i"), size)),
      create.expression(StandardOperator.Perm,
        create.pattern(create.expression(StandardOperator.Subscript, value, name("__i"))),
        perm),
      List(new DeclarationStatement("__i", create.primitive_type(PrimitiveSort.Integer))):_*
    )

    conditions.reduce(star)
  }

  def validPointerIndexFor(input: ASTNode, t: Type, index: ASTNode, perm: ASTNode): ASTNode = {
    val conditions: mutable.ListBuffer[ASTNode] = mutable.ListBuffer()
    val seqInfo = SequenceUtils.expectArrayType(t, "Expected an array type here, but got %s")

    if(!seqInfo.isOpt || !seqInfo.isCell) {
      Fail("Expected a pointer type here, but got %s", t)
    }

    val value = input
    conditions += neq(value, create.reserved_name(ASTReserved.OptionNone))
    conditions += less(index, create.dereference(value, "length"))

    conditions += create.expression(StandardOperator.Perm,
      create.expression(StandardOperator.Subscript, value, index),
      perm)

    conditions.reduce(star)
  }
}
