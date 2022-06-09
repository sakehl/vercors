package vct.col.ast.util

import hre.lang.System.Warning
import vct.col.ast.expr.StandardOperator._
import vct.col.ast.expr.constant.{ConstantExpression, IntegerValue}
import vct.col.ast.expr.{NameExpression, OperatorExpression, StandardOperator}
import vct.col.ast.generic.ASTNode
import vct.col.ast.stmt.decl.Contract
import vct.col.ast.util.ExpressionEqualityCheck.is_constant_int

import scala.collection.mutable
import scala.jdk.CollectionConverters.IterableHasAsScala
import scala.util.Failure

object ExpressionEqualityCheck {
  def apply(info: Option[AnnotationVariableInfo] = None): ExpressionEqualityCheck = new ExpressionEqualityCheck(info)

  def is_constant_int(e: ASTNode): Option[Int] = {
    ExpressionEqualityCheck().is_constant_int(e)
  }

  def is_constant_int_java(e: ASTNode): Option[Integer] = is_constant_int(e).map(Integer.valueOf)

  def equal_expressions(lhs: ASTNode, rhs: ASTNode): Boolean = {
    ExpressionEqualityCheck().equal_expressions(lhs, rhs)
  }

}

class ExpressionEqualityCheck(info: Option[AnnotationVariableInfo]) {
  var replacer_depth = 0
  var replacer_depth_int = 0
  val max_depth = 100

  def is_constant_int(e: ASTNode): Option[Int] = {
    replacer_depth_int = 0
    is_constant_int_(e)
  }

  def is_constant_int_(e: ASTNode): Option[Int] = e match {
    case NameExpression(name, _, _) =>
      // Does it have a direct int value?
      info.flatMap(_.variable_values.get(name)) match {
        case Some(x) => Some(x)
        case None =>
          info.flatMap(_.variable_equalities.get(name)) match{
            case None => None
          case Some(equals) =>
            for(eq <- equals){
              // Make sure we do not loop indefinitely by keep replacing the same expressions somehow
              if(replacer_depth_int > max_depth) return None
              replacer_depth_int += 1
              val res = is_constant_int_(eq)
              if(res.isDefined) return res
            }
            None
        }
      }

    case ConstantExpression(IntegerValue(value)) => Some(value)
    case OperatorExpression(Plus, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int_(e1); i2 <- is_constant_int_(e2)} yield i1 + i2
    case OperatorExpression(Minus, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int_(e1); i2 <- is_constant_int_(e2)} yield i1 - i2
    case OperatorExpression(Mult, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int_(e1); i2 <- is_constant_int_(e2)} yield i1 * i2
    case OperatorExpression(FloorDiv, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int_(e1); i2 <- is_constant_int_(e2)} yield i1 / i2
    case OperatorExpression(Mod, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int_(e1); i2 <- is_constant_int_(e2)} yield i1 % i2
    case OperatorExpression(Exp, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int_(e1); i2 <- is_constant_int_(e2)} yield scala.math.pow(i1, i2).toInt
    case OperatorExpression(BitAnd, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int_(e1); i2 <- is_constant_int_(e2)} yield i1 & i2
    case OperatorExpression(BitOr, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int_(e1); i2 <- is_constant_int_(e2)} yield i1 | i2
    case OperatorExpression(BitXor, e1 :: e2 :: Nil) =>
      for {i1 <- is_constant_int_(e1); i2 <- is_constant_int_(e2)} yield i1 ^ i2
    case _ => None
  }

  def equal_expressions(lhs: ASTNode, rhs: ASTNode): Boolean = {
    replacer_depth = 0
    equal_expressions_(lhs: ASTNode, rhs: ASTNode)
  }

  def equal_expressions_(lhs: ASTNode, rhs: ASTNode): Boolean = {
    (is_constant_int(lhs), is_constant_int(rhs)) match {
      case (Some(i1), Some(i2)) => return i1 == i2
      case (None, None) => ()
      //If one is a constant expression, and the other is not, this cannot be the same
      case _ => return false
    }

    (lhs, rhs) match {
      case (OperatorExpression(op1, lhs1 :: lhs2 :: Nil), OperatorExpression(op2, rhs1 :: rhs2 :: Nil)) if op1 == op2 && (op1 == And || op1 == Plus) =>
        if (equal_expressions_(lhs1, rhs1) && equal_expressions_(lhs2, rhs2))
          true
        else
          equal_expressions_(lhs1, rhs2) && equal_expressions_(lhs2, rhs1)
      case (OperatorExpression(op1, args1), OperatorExpression(op2, args2)) if op1 == op2 && args1.length == args2.length =>
        (args1 zip args2).foldRight(true)((xy, rest) => rest && equal_expressions_(xy._1, xy._2))
      case (NameExpression(name1, _, _), NameExpression(name2, _, _)) =>
        if(name1 == name2)  true
        else if(info.isDefined) {
          // Check if the variables are synonyms
          (info.get.variable_synonyms.get(name1), info.get.variable_synonyms.get(name2)) match {
            case (Some(x), Some(y)) => x == y
            case _ => false
          }
        } else false
      case (NameExpression(name1, _, _), e2) =>
        replace_variable(name1, e2)
      case (e1, NameExpression(name2, _, _)) =>
        replace_variable(name2, e1)
      // In the general case, we are just interested in syntactic equality
      case (e1, e2) => e1 == e2
    }

  }

  def replace_variable(name: String, other_e: ASTNode): Boolean ={
    if(info.isDefined){
      info.get.variable_equalities.get(name) match{
        case None => false
        case Some(equals) =>
          for(eq <- equals){
            // Make sure we do not loop indefinitely by keep replacing the same expressions somehow
            if(replacer_depth > max_depth) return false
            replacer_depth += 1
            if(equal_expressions_(eq, other_e)) return true
          }
          false
      }
    } else {
      false
    }
  }
}


case class AnnotationVariableInfo(variable_equalities: Map[String, List[ASTNode]], variable_values: Map[String, Int],
                                  variable_synonyms: Map[String, Int])
/** This class gathers information about variables, such as:
  * `requires x == 0` and stores that x is equal to the value 0.
  * Which we can use in simplify steps
  * This information is returned with get_info(c: Contract)
  */
class AnnotationVariableInfoGetter() {
  val variable_equalities: mutable.Map[String, mutable.ListBuffer[ASTNode]] =
    mutable.Map()
  val variable_values: mutable.Map[String, Int] = mutable.Map()
  // We put synonyms in the same group and give them a group number, to identify the same synonym groups
  val variable_synonyms: mutable.Map[String, Int] = mutable.Map()
  var current_synonym_group = 0;

  def extract_equalities(e: ASTNode): Unit = {
    e match {
      case OperatorExpression(EQ, e1 :: e2 :: Nil) => {
        (e1, e2) match{
          case (NameExpression(n1, _, _), NameExpression(n2, _, _))  => add_synonym(n1, n2)
          case (NameExpression(n1, _, _), _)  => add_name(n1, e2)
          case (_, NameExpression(n2, _, _))  => add_name(n2, e2)
          case _ =>
        }
      }
      case _ =>
    }
  }

  def add_synonym(n1: String, n2: String): Unit = {
    (variable_synonyms.get(n1), variable_synonyms.get(n1)) match {
      // We make a new group
      case (None, None) =>
        variable_synonyms(n1) = current_synonym_group
        variable_synonyms(n2) = current_synonym_group
        current_synonym_group += 1
      // Add to the found group
      case (Some(id1), None) => variable_synonyms(n2) = id1
      case (None, Some(id2)) => variable_synonyms(n1) = id2
      // Merge the groups, give every synonym group member of id2 value id1
      case (Some(id1), Some(id2)) =>
        variable_synonyms.mapValuesInPlace((_, group) => if (group == id2) id1 else group)
    }
  }

  def add_name(name :String , expr: ASTNode): Unit ={
    // Add to constant list
    is_constant_int(expr) match {
      case Some(x) => variable_values.get(name) match {
        case Some(x_) => if (x!=x_) Warning("Value of %s is required to be both %d and %d", name, x, x_);
        case None => variable_values(name) = x
      }
      case None =>
        val list = variable_equalities.getOrElseUpdate(name, mutable.ListBuffer())
        list.addOne(expr)
    }
  }

  def get_info(annotations: Iterable[ASTNode]): AnnotationVariableInfo = {
    variable_equalities.clear()
    variable_values.clear()

    //visit(c)
    for(clause <- annotations){
      extract_equalities(clause)
    }

    distribute_info()
    AnnotationVariableInfo(variable_equalities.view.mapValues(_.toList).toMap, variable_values.toMap,
      variable_synonyms.toMap)
  }

  def distribute_info(): Unit = {
    // First distribute value knowledge over the rest of the map
    val begin_size = variable_values.size

    for((name, equals) <- variable_equalities){
      if(!variable_values.contains(name))
        for(equal <- equals){
          equal match {
            case n : NameExpression =>
              variable_values.get(n.name).foreach(variable_values(name) = _)
            case _ =>
          }
        }
    }

    // If sizes are not the same, we know more, so distribute again!
    if(variable_values.size != begin_size) distribute_info()
  }
}
