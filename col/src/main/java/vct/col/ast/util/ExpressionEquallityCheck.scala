package vct.col.ast.util

import vct.col.ast.expr.StandardOperator._
import vct.col.ast.expr.constant.{ConstantExpression, IntegerValue}
import vct.col.ast.expr.{NameExpression, OperatorExpression}
import vct.col.ast.generic.ASTNode

object ExpressionEquallityCheck {
  def is_constant_int(e: ASTNode): Option[Int] = e match {
    case ConstantExpression(IntegerValue(value)) => Some(value)
    case OperatorExpression(Plus, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int(e1); i2 <- is_constant_int(e2)} yield i1 + i2
    case OperatorExpression(Minus, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int(e1); i2 <- is_constant_int(e2)} yield i1 - i2
    case OperatorExpression(Mult, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int(e1); i2 <- is_constant_int(e2)} yield i1 * i2
    case OperatorExpression(FloorDiv, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int(e1); i2 <- is_constant_int(e2)} yield i1 / i2
    case OperatorExpression(Mod, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int(e1); i2 <- is_constant_int(e2)} yield i1 % i2
    case OperatorExpression(Exp, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int(e1); i2 <- is_constant_int(e2)} yield scala.math.pow(i1, i2).toInt
    case OperatorExpression(BitAnd, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int(e1); i2 <- is_constant_int(e2)} yield i1 & i2
    case OperatorExpression(BitOr, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int(e1); i2 <- is_constant_int(e2)} yield i1 | i2
    case OperatorExpression(BitXor, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int(e1); i2 <- is_constant_int(e2)} yield i1 ^ i2
    case _ => None
  }

  def is_constant_int_java(e: ASTNode): Option[Integer] = is_constant_int(e).map(Integer.valueOf)

  def equal_expressions(lhs: ASTNode, rhs: ASTNode): Boolean = {
    (is_constant_int(lhs), is_constant_int(lhs)) match {
      case (Some(i1), Some(i2)) => return i1 == i2
      case (None, None) => ()
      //If one is a constant expression, and the other is not, this cannot be the same
      case _ => return false
    }

    (lhs, rhs) match {
      case (OperatorExpression(op1, lhs1 :: lhs2 :: Nil), OperatorExpression(op2, rhs1 :: rhs2 :: Nil)) if op1 == op2 && (op1 == And || op1 == Plus) =>
        if (equal_expressions(lhs1, rhs1) && equal_expressions(lhs2, rhs2))
          true
        else
          equal_expressions(lhs1, rhs2) && equal_expressions(lhs2, rhs1)
      case (OperatorExpression(op1, args1), OperatorExpression(op2, args2)) if op1 == op2 && args1.length == args2.length =>
        (args1 zip args2).foldRight(true)((xy, rest) => rest && equal_expressions(xy._1, xy._2))
      case (NameExpression(name1, _, _), NameExpression(name2, _, _)) => name1 == name2
      // In the general case, we are just interested in syntactic equality
      case (e1, e2) => e1 == e2
    }

  }
}
