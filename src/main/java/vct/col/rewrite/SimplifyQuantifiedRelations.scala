package vct.col.rewrite

import vct.col.ast.`type`.{PrimitiveSort, PrimitiveType, Type}
import vct.col.ast.expr.StandardOperator._
import vct.col.ast.expr._
import vct.col.ast.expr.constant.{ConstantExpression, IntegerValue}
import vct.col.ast.generic.ASTNode
import vct.col.ast.stmt.decl.ProgramUnit
import vct.col.ast.util.{AbstractRewriter, NameScanner, RecursiveVisitor, Substituter}

import scala.annotation.nowarn
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * This rewrite pass simplifies expressions of roughly this form:
  * forall i: Int . lower <= i <= higher ==> left {<=|<|==|>|>=} right
  * where left or right is independent of all quantified names
  */
class SimplifyQuantifiedRelations(source: ProgramUnit) extends AbstractRewriter(source) {

  def getNames(node: ASTNode): Set[String] = {
    val scanner = new NameScanner()
    node.accept(scanner)
    scanner.accesses
  }

  def substituteNode(substitute_vars: Map[String, ASTNode], node: ASTNode): ASTNode = {
    val sub = new Substituter(null, substitute_vars)
    node.apply(sub)
  }

  /**
    * a && b && c --> Seq(a, b, c)
    */
  def getConjuncts(node: ASTNode): Seq[ASTNode] = node match {
    case op: OperatorExpression if op.operator == And =>
      getConjuncts(op.first) ++ getConjuncts(op.second)
    case other => Seq(other)
  }

  def splitSelect(select: ASTNode, main: ASTNode): (Seq[ASTNode], ASTNode) = {
    var left: ArrayBuffer[ASTNode] = ArrayBuffer()
    var right = main

    left ++= getConjuncts(select)

    while(right.isa(StandardOperator.Implies)) {
      left ++= getConjuncts(right.asInstanceOf[OperatorExpression].first)
      right = right.asInstanceOf[OperatorExpression].second
    }

    (left.toSeq, right)
  }

  def independentOf(names: Set[String], node: ASTNode): Boolean =
    getNames(node).intersect(names).isEmpty

  def isNameIn(names: Set[String], node: ASTNode): Boolean = node match {
    case name: NameExpression if names.contains(name.getName) => true
    case _ => false
  }

  def mapOp(op: StandardOperator, args: Option[ASTNode]*): Option[ASTNode] = {
    if(args.forall(_.isDefined)) {
      Some(create expression(op, args.map(_.get):_*))
    } else {
      None
    }
  }

  class Bounds(val names: Set[String]) {
    val lowerBounds: mutable.Map[String, Seq[ASTNode]] = mutable.Map()
    val upperBounds: mutable.Map[String, Seq[ASTNode]] = mutable.Map()
    val upperExclusiveBounds: mutable.Map[String, Seq[ASTNode]] = mutable.Map()

    def addLowerBound(name: String, bound: ASTNode): Unit =
      lowerBounds(name) = lowerBounds.getOrElse(name, Seq()) :+ bound

    def addUpperBound(name: String, bound: ASTNode): Unit =
      upperBounds(name) = upperBounds.getOrElse(name, Seq()) :+ bound

    def addUpperExclusiveBound(name: String, bound: ASTNode): Unit =
      upperExclusiveBounds(name) = upperExclusiveBounds.getOrElse(name, Seq()) :+ bound

    def extremeValue(name: String, maximizing: Boolean): Option[ASTNode] =
      (if(maximizing) upperBounds else lowerBounds).get(name) match {
        case None => None
        case Some(bounds) => Some(SimplifyQuantifiedRelations.this.extremeValue(bounds, !maximizing))
      }

    def selectNonEmpty: Seq[ASTNode] =
      lowerBounds.flatMap {
        case (name, lowerBounds) =>
          upperBounds.getOrElse(name, Seq()).flatMap(upperBound =>
            lowerBounds.map(lowerBound => create.expression(StandardOperator.LTE, lowerBound, upperBound))
          )
      }.toSeq
  }

  class RewriteLinearArrayAccesses(source: ProgramUnit, bounds: Bounds) extends AbstractRewriter(source) {
    var rewritten : Boolean = false
    var substitute_forall : Option[SubstituteForall] = None

    override def visit(expr: OperatorExpression): Unit = {
      // Make sure we only rewrite one array index here
      if (!rewritten) {
        expr.operator match {
          case Subscript =>
            val linear_expr_finder = new FindLinearExpressions(bounds)
            expr.second.accept(linear_expr_finder)
            linear_expr_finder.can_rewrite() match {
              case Some(substitute_forall) =>
                this.substitute_forall = Some(substitute_forall)
                result =create expression(Subscript, expr.first, create identifier substitute_forall.new_forall_var)
                rewritten = true;
              case None => super.visit(expr)
            }
          case _ => super.visit(expr)
        }
      }
    }
  }

  def is_constant_int(e: ASTNode) : Option[Int] = {
    e match {
      case ConstantExpression(IntegerValue(value)) => Some(value)
      case OperatorExpression(Plus,  e1 ::e2 :: Nil) =>
        for { i1 <- is_constant_int(e1); i2 <- is_constant_int(e2) } yield i1+i2
      case OperatorExpression(Minus, e1 :: e2 :: Nil) =>
        for { i1 <- is_constant_int(e1); i2 <- is_constant_int(e2) } yield i1-i2
      case OperatorExpression(Mult, e1 :: e2 :: Nil) =>
        for { i1 <- is_constant_int(e1); i2 <- is_constant_int(e2) } yield i1*i2
      case OperatorExpression(FloorDiv, e1 :: e2 :: Nil) =>
        for { i1 <- is_constant_int(e1); i2 <- is_constant_int(e2) } yield i1/i2
      case OperatorExpression(Mod, e1 :: e2 :: Nil) =>
        for { i1 <- is_constant_int(e1); i2 <- is_constant_int(e2) } yield i1%i2
      case OperatorExpression(Exp, e1 :: e2 :: Nil) =>
        for { i1 <- is_constant_int(e1); i2 <- is_constant_int(e2) } yield scala.math.pow(i1,i2).toInt
      case OperatorExpression(BitAnd, e1 :: e2 :: Nil) =>
        for { i1 <- is_constant_int(e1); i2 <- is_constant_int(e2) } yield i1 & i2
      case OperatorExpression(BitOr, e1 :: e2 :: Nil) =>
        for { i1 <- is_constant_int(e1); i2 <- is_constant_int(e2) } yield i1 | i2
      case OperatorExpression(BitXor, e1 :: e2 :: Nil) =>
        for { i1 <- is_constant_int(e1); i2 <- is_constant_int(e2) } yield i1 ^ i2
      case _ => None
    }
  }

  def equal_expressions(lhs : ASTNode, rhs: ASTNode): Boolean = {
    (is_constant_int(lhs), is_constant_int(lhs)) match{
      case (Some(i1), Some(i2)) => return i1 == i2
      case (None, None) => ()
      //If one is a constant expression, and the other is not, this cannot be the same
      case _ => return false
    }

    (lhs, rhs) match {
      case (OperatorExpression(op1, lhs1 ::lhs2 :: Nil), OperatorExpression(op2, rhs1 ::rhs2 :: Nil)) if op1 == op2 && (op1 == And || op1 == Plus) =>
        if(equal_expressions(lhs1, rhs1) && equal_expressions(lhs2, rhs2))
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

  // The `new_forall_var` will be the name of variable of the newly made forall.
  // The `new_bounds`, will contain all the new equations for "select" part of the forall.
  // The `substitute_old_vars` contains a map, so we can replace the old forall variables with new expressions
  // We also store the `linear_expression`, so if we ever come across it, we can replace it with the new variable.
  case class SubstituteForall(new_forall_var: String, new_bounds: ASTNode, substitute_old_vars: Map[String, ASTNode])

  // This class should be called on array indices, were the index expression has at least one variable in the variables set.
  // We should not encounter anymore nested forall's.
  class FindLinearExpressions(variable_bounds: Bounds) extends RecursiveVisitor[AnyRef](null, null) {
    val linear_expressions: mutable.Map[String, ASTNode] = mutable.Map()
    var constant_expression: Option[ASTNode] = None
    var is_linear: Boolean  = true
    var current_multiplier: Option[ASTNode] = None

    def can_rewrite(): Option[SubstituteForall] = {
      if(!is_linear) {
        return None
      }

      //We are looking at an expression like:
      // xs[a*i + b]
      // We can rewrite this to:
      // xs[i_fresh], and were we substitute every occurrence of i with:
      // i := (i_fresh - b) / a
      // We need to add i_fresh % a == b to the forall precondition
      // We assume that a>0
      // TODO: Check for a or generalize
      def rewrite_single_variable(): SubstituteForall ={
        val i_old = variable_bounds.names.head
        //Since we have only one variable, we reuse the old one
        val i_fresh = create identifier i_old
        //Only consider the linear factor, if it is not equal to 1
        val lin = linear_expressions(i_old)
        val linear_factor = if(is_value(lin, 1)) None else Some(lin)
        val base = (linear_factor, constant_expression) match{
          // (i_fresh-b) / a
          case (Some(a), Some(b)) => create expression(FloorDiv, create expression(Minus, i_fresh, b), a)
          // i_fresh / a
          case (Some(a), None) => create expression(FloorDiv, i_fresh, a)
          // i_fresh - b
          case (None, Some(b)) => create expression(Minus, i_fresh, b)
          // i_fresh
          case (None, None) => i_fresh
        }

        val substitutions = Map(
          // i := ((i_fresh - b) / a)
          i_old -> base
        )

        var bounds = (linear_factor, constant_expression) match {
          case (Some(a), Some(b)) =>
            create expression(EQ, create expression(Mod, i_fresh, a), b)
          case (Some(a), None) =>
            create expression(EQ, create expression(Mod, i_fresh, a), create constant 0)
          case _ => create constant true
        }

        for(old_upper_bound <- variable_bounds.upperExclusiveBounds(i_old))
          bounds = create expression(And, create expression(LT, base, old_upper_bound), bounds)

        for(old_lower_bound <- variable_bounds.lowerBounds(i_old))
          bounds = create expression(And, create expression(LTE, old_lower_bound, base), bounds)

        SubstituteForall(i_old, bounds, substitutions)
      }

      // We are looking for patterns: 0 <= j < m && 0 <= i < n: xs[a*n*j + a*i + b]
      // This can be rewritten to: b <= i_fresh <= a*n*m + b && i_fresh % a == b: xs[i_fresh]
      // We can then substitute i and j:
      // i := ((i_fresh - b) / a) % n
      // j := ((i_fresh - b) /a) / n
      // Given that n > 0, m > 0 and a > 0
      // TODO: We are not checking n, m and a on this
      // inner_var := i, outer_var := j
      def check_vars(inner_var: String , outer_var /*j*/: String): Option[SubstituteForall] = {
        // lin_inner := a
        val lin_inner = linear_expressions(inner_var)
        // lin_outer =?= a*n
        val lin_outer = linear_expressions(outer_var)
        for (bounds <- variable_bounds.upperExclusiveBounds.get(inner_var); upper <- bounds) {
          if (is_value(lin_inner, 1)) {
            if (equal_expressions(upper, lin_outer)) {
                return Some(make_substitute(inner_var, outer_var, upper, None))
              }
          }
          else {
            if (equal_expressions(create expression(Mult, lin_inner, upper), lin_outer)) {
              return Some(make_substitute(inner_var, outer_var, upper, Some(lin_inner)))
            }
          }
        }
        None
      }

      // inner_var := i, outer_var := j, max_bound_inner := n
      def make_substitute(inner_var: String, outer_var: String, max_bound_inner: ASTNode,
                          linear_factor: Option[ASTNode]): SubstituteForall ={

        val i_fresh_string = inner_var ++ "_" ++ outer_var
        val i_fresh = create identifier i_fresh_string
        val base = (linear_factor, constant_expression) match{
            // (i_fresh-b) / a
          case (Some(a), Some(b)) => create expression(FloorDiv, create expression(Minus, i_fresh, b), a)
            // i_fresh / a
          case (Some(a), None) => create expression(FloorDiv, i_fresh, a)
            // i_fresh - b
          case (None, Some(b)) => create expression(Minus, i_fresh, b)
            // i_fresh
          case (None, None) => i_fresh
        }

        val substitutions = Map(
          // i := ((i_fresh - b) / a) % n
          inner_var -> (create expression(Mod, base, max_bound_inner)),
          // j := ((i_fresh - b) /a) / n
          outer_var -> (create expression(FloorDiv, base, max_bound_inner)),
          )

        val max_bound_outer = variable_bounds.upperExclusiveBounds(outer_var).head
        val new_max_bound = linear_factor match {
          case Some(a) => create expression(Mult, a, create expression(Mult, max_bound_inner, max_bound_outer))
          case None => create expression(Mult, max_bound_inner, max_bound_outer)
        }

        var bounds = (linear_factor, constant_expression) match {
          // b <= i_fresh && i_fresh < a * m * n + b && i_fresh % a == b
          case (Some(a), Some(b)) => create expression (And,
              create expression(LTE, b, i_fresh),
              create expression(And,
                create expression(LT, i_fresh, create expression(Plus, new_max_bound, b)),
                create expression(EQ, create expression(Mod, i_fresh, a), b)
              )
            )
          // 0 <= i_fresh && i_fresh < a * m * n && i_fresh % a == 0
          case (Some(a), None) => create expression (And,
            create expression(LTE, create constant 0, i_fresh),
            create expression(And,
              create expression(LT, i_fresh, new_max_bound),
              create expression(EQ, create expression(Mod, i_fresh, a), create constant 0)
            )
          )
          // b <= i_fresh && i_fresh < m * n + b
          case (None, Some(b)) => create expression (And,
            create expression(LTE, b, i_fresh),
            create expression(LT, i_fresh, create expression(Plus, new_max_bound, b))
          )
          // 0 <= i_fresh && i_fresh < m * n
          case (None, None) => create expression (And,
            create expression(LTE, create constant 0, i_fresh),
            create expression(LT, i_fresh, new_max_bound)
          )
        }

        // The old bounds are independent, so we do not have to substitute them
        for(old_upper_bound <- variable_bounds.upperExclusiveBounds(inner_var))
          if(old_upper_bound != max_bound_inner)
            bounds = create expression(And, create expression(LT, substitutions(inner_var), old_upper_bound), bounds)

        for(old_upper_bound <- variable_bounds.upperExclusiveBounds(outer_var).tail)
            bounds = create expression(And, create expression(LT, substitutions(outer_var), old_upper_bound), bounds)

        for(old_lower_bound <- variable_bounds.lowerBounds(inner_var))
          if(!is_value(old_lower_bound, 0))
            bounds = create expression(And, create expression(LTE, old_lower_bound, substitutions(inner_var)), bounds)

        for(old_lower_bound <- variable_bounds.lowerBounds(outer_var))
          if(!is_value(old_lower_bound, 0))
            bounds = create expression(And, create expression(LTE, old_lower_bound, substitutions(outer_var)), bounds)

        // Since we know the lower bound was also 0, and the we multiply the upper bounds,
        // we do have to require that each upper bound is at least bigger than 0.
        // TODO: Generalize for non-zero lower bounds
        bounds = create expression(And, create expression (LT, create constant 0, max_bound_inner), bounds)
        bounds = create expression(And, create expression (LT, create constant 0, max_bound_outer), bounds)
        SubstituteForall(i_fresh_string, bounds, substitutions)
      }

      if(variable_bounds.names.size == 1){
        return Some(rewrite_single_variable())
      } else if(variable_bounds.names.size != 2){
        //Only support 1 and 2 variables at the moment
        // TODO: Generalize this
        return None
      }

      val v1 = variable_bounds.names.head
      val v2 = variable_bounds.names.tail.head

      check_vars(v1, v2) orElse check_vars(v2, v1)
    }

    def is_value(e: ASTNode, x: Int): Boolean =
      is_constant_int(e) match {
        case None => false
        case Some(y) => y == x
      }

    override def visit(expr: OperatorExpression): Unit = {
      expr.operator match {
        case Plus =>
          // if the first is constant, the second argument cannot be
          if (isConstant(expr.first)) {
            addToConstant(expr.first, Plus)
            expr.second.accept(this)
          } else if (isConstant(expr.second)) {
            addToConstant(expr.second, Plus)
            expr.first.accept(this)
          } else { // Both arguments contain linear information
            expr.first.accept(this)
            expr.second.accept(this)
          }
        case Minus =>
          // if the first is constant, the second argument cannot be
          if (isConstant(expr.first)) {
            addToConstant(expr.first, Plus)
            val old_multiplier = current_multiplier
            multiplyMultiplier(create constant (-1))
            expr.second.accept(this)
            current_multiplier = old_multiplier
          } else if (isConstant(expr.second)) {
            addToConstant(expr.second, Minus)
            expr.first.accept(this)
          } else { // Both arguments contain linear information
            expr.first.accept(this)
            val old_multiplier = current_multiplier
            multiplyMultiplier(create constant (-1))
            expr.second.accept(this)
            current_multiplier = old_multiplier
          }
        case Mult =>
          if (isConstant(expr.first)) {
            val old_multiplier = current_multiplier
            multiplyMultiplier(expr.first)
            expr.second.accept(this)
            current_multiplier = old_multiplier
          } else if (isConstant(expr.second)) {
            val old_multiplier = current_multiplier
            multiplyMultiplier(expr.second)
            expr.first.accept(this)
            current_multiplier = old_multiplier
          } else {
            is_linear = false
          }
        case _ =>
          is_linear = false;
      }
    }

    override def visit(e: NameExpression): Unit = {
      if(variable_bounds.names.contains(e.getName)){
        linear_expressions get e.getName match {
          case None => linear_expressions(e.getName) = current_multiplier.getOrElse(create constant 1); //s
          case Some(old) => linear_expressions(e.getName) = create expression(Plus, old, current_multiplier.getOrElse(create constant 1));
        }
      } else {
        Abort("We should not end up here, the precondition of \'FindLinearExpressions\' was not uphold." )
      }
    }
    def isConstant(node: ASTNode): Boolean = independentOf(variable_bounds.names, node)

    def addToConstant(node : ASTNode, operator: StandardOperator = Plus): Unit = {
      val added_node: ASTNode = current_multiplier match  {
        case None => node;
        case Some(expr) => create expression(Mult, expr, node);
      }
      @nowarn("msg=not.*?exhaustive")
      val _ = constant_expression match {
        case None if operator == Plus => constant_expression = Some(added_node);
        case None if operator == Minus => constant_expression
          = Some(create expression(Mult, create constant(-1), added_node));
        case Some(expr) => constant_expression = Some(create expression(operator, expr, added_node)) ;
      }
    }

    def multiplyMultiplier(node : ASTNode): Unit ={
      current_multiplier match {
        case None => current_multiplier = Some(node);
        case Some(expr) => current_multiplier = Some(create expression(Mult, expr, node));
      }
    }
  }

  /* This function tries to look if we come across any linear terms in array accesses, which we can rewrite.
     Potentially new ideas or important issues (TODO):
     * We require that there is always a lower bound of zero present. This can be generalized. E.g then s <= i < n,
       whenever we then compare things with n (the length of the interval), we should compare with "n-s".
     * For two variables, we can only handle the form: forall i, j: 0 <= j < m && 0 <= i < n: xs[a*(n*j + i)+b]
       I believe we can be a bit more general, and can rewrite anything in the form xs[a*(c*j + i)+b], when c > n
       or when we have xs[a*j + c*i+b] and not sure if a and c have a relation.
     * We have some preconditions we should really check, for instance: "a>0" (Think about a<0?)
     * We cannot rewrite: forall i, j: 0 <= j < m && 0 <= i < n: xs[a*(32*j + i)+b], even if somewhere else we have
       "require n==32"
     * We evaluate some checks if bounds are 0, or a==1 and other stuff. We wrote very simple "is_constant_int"
       and "equal_expressions" functions. Most likely, the equal expression function also could call a black-box
       sat solver, to determine equality in more cases.
     * This must be possible for forall's that have more than two variables, but we did not implement it yet.
     * The rewrite pass beforehand, handles the case when something like forall(i; 0<=i && i < n; (term!i)). We
       could potentially handle more here.
   */
  def rewriteLinearArray(bounds: Bounds, main: ASTNode, old_select: Seq[ASTNode], binder: Binder, result_type: Type): Option[ASTNode] = {
    // For now we only allow forall's with one or two variables
    // TODO: Generalize this
    if(bounds.names.size != 1 && bounds.names.size != 2){
      return None
    }

    //We need to check some preconditions for 2 vars
    if(bounds.names.size == 2) {

      val scanner = new NameScanner()
      main.accept(scanner)
      // Check if we can make a new nice name, which we will use, but this is only possible when it is not a free variable
      val possible_names = Set(bounds.names.head ++ "_" ++ bounds.names.tail.head, bounds.names.tail.head ++ "_" ++ bounds.names.head)
      if (scanner.freeNames.keySet.intersect(possible_names).nonEmpty)
        return None
    }

    // This allows only forall's to be rewritten, if they have at least one lower bound of zero
    // TODO: Generalize this, so we don't have this restriction
    for (name <- bounds.names) {
      var one_zero = false
      bounds.lowerBounds.getOrElse(name, Seq())
        .foreach(lower => is_constant_int(lower) match {
          case Some(0) => one_zero = true
          case _ =>
        })
      //Exit when notAt least one zero, or no upper bounds
      if (!one_zero || bounds.upperBounds.getOrElse(name, Seq()).isEmpty) {
        return None
      }
    }

    val linear_accesses = new RewriteLinearArrayAccesses(source, bounds)
    var new_main = linear_accesses.rewrite(main)
    linear_accesses.substitute_forall match {
      case Some(substitute_forall) =>
        new_main = substituteNode(substitute_forall.substitute_old_vars, new_main)
        val select = (Seq(substitute_forall.new_bounds) ++ old_select.map(substituteNode(substitute_forall.substitute_old_vars,_))).reduce(and)
        val declaration = create field_decl(substitute_forall.new_forall_var, new PrimitiveType(PrimitiveSort.Integer))
        val forall = create binder(binder, result_type, Array(declaration), Array(), select, new_main)
        Some(forall)
      case None => None
    }
  }

  /**
   * If we want the maximum/minimum of a list of nodes, that is encoded as an ITE. If we want to compose with our own
   * pass (i.e. have nested forall's) it is nice to be able to transform something like (a < b ? a : b) back to min(a,b)
   * if we encounter it again. We could/should make nodes min/max(a, b, c, ...), but this also works for now.
   */
  val extremeOfListNode: mutable.Map[ASTNode, (Seq[ASTNode], Boolean)] = mutable.Map()

  def extremeValue(values: Seq[ASTNode], maximizing: Boolean): ASTNode = {
    val result = if(values.size == 1) {
      values.head
    } else {
      val preferFirst = if(maximizing) GT else LT
      create.expression(ITE,
        create.expression(preferFirst, values(0), values(1)),
        extremeValue(values(0) +: values.drop(2), maximizing),
        extremeValue(values(1) +: values.drop(2), maximizing),
      )
    }
    extremeOfListNode(result) = (values, maximizing)
    result
  }

  def independentBounds(bounds: Bounds, nodes: ASTNode*): Boolean = {
    nodes.map(getNames(_).intersect(bounds.names)).foldLeft[Either[Unit, Set[String]]](Right(Set.empty)) {
      case (Right(used), nameSet) =>
        if(nameSet.intersect(used).nonEmpty) {
          Left(())
        } else {
          Right(nameSet ++ used)
        }
      case (Left(()), _) => Left(())
    }.isRight
  }

  def extremeValue(bounds: Bounds, node: ASTNode, maximizing: Boolean): Option[ASTNode] = node match {
    case op: OperatorExpression => op.operator match {
      case Plus =>
        if(!independentBounds(bounds, op.first, op.second)) None
        else mapOp(Plus, extremeValue(bounds, op.first, maximizing), extremeValue(bounds, op.second, maximizing))
      case Minus =>
        if(!independentBounds(bounds, op.first, op.second)) None
        else mapOp(Minus, extremeValue(bounds, op.first, maximizing), extremeValue(bounds, op.second, !maximizing))
      case Mult | Div | FloorDiv =>
        val (left, right) = (op.first, op.second)
        if(!independentBounds(bounds, left, right)) return None

        val leftA = extremeValue(bounds, left, maximizing)
        val leftB = extremeValue(bounds, left, !maximizing)
        val rightA = extremeValue(bounds, right, maximizing)
        val rightB = extremeValue(bounds, right, !maximizing)
        val maybeValues = Seq(
          mapOp(op.operator, leftA, rightA),
          mapOp(op.operator, leftB, rightB),
        )
        if(maybeValues.exists(_.isEmpty)) return None
        // We take distinct here in case both sides are constant
        val values = maybeValues.map(_.get).distinct
        Some(extremeValue(values, maximizing))
      case UMinus => extremeValue(bounds, node, !maximizing)
      case ITE if extremeOfListNode.contains(node) =>
        val (nodes, nodeIsMaximizing) = extremeOfListNode(node)

        /* When maximizing == nodeIsMaximizing, e.g.:
         * max_i { max(f(i), g(i)) | bounds }
         * = max(max_i { f(i) | bounds }, max_i { g(i) | bounds })
         *
         * When maximizing == !nodeIsMaximizing, e.g.:
         * max_{i,j} { min(f(i), g(j)) | bounds }
         * = min(max_{i,j} { f(i) }, max_{i,j} { g(j) })
         * check that the arguments to inner function are mutually independent over the bound variables
         */

        if(maximizing != nodeIsMaximizing && !independentBounds(bounds, nodes:_*)) {
          return None
        }

        val extremeNodes = nodes.map(extremeValue(bounds, _, maximizing))
        if(extremeNodes.exists(_.isEmpty)) return None
        Some(extremeValue(extremeNodes.map(_.get), nodeIsMaximizing))
      case _ => None
    }
    case name: NameExpression if bounds.names.contains(name.getName) => bounds.extremeValue(name.getName, maximizing)
    case other: NameExpression => Some(other)
    case const: ConstantExpression => Some(const)
    case _ => None
  }

  /**
    * Try to derive set bounds for all the quantified variables
    * @param names The names for which to derive bounds
    * @param select The "select" portion of a BindingExpression, e.g. of the form 0 <= i && i < n && 0 <= j && j < m
    * @return When successful, a map of each name in `names` to its inclusive lower bound and exclusive upper bound
    */
  def getBounds(names: Set[String], select: Seq[ASTNode]): (Bounds, Seq[ASTNode]) = {
    val bounds = new Bounds(names)
    var dependent_bounds : Seq[ASTNode] = Seq()

    def process_bound(expression: ASTNode): Unit =
    {
      expression match {
        case expr: OperatorExpression if Set(LT, LTE, GT, GTE, EQ).contains(expr.operator) =>
          val (variable, op, bound) = if (isNameIn(names, expr.first) && independentOf(names, expr.second)) {
            // If the quantified variable is the first argument: keep it as is
            (expr.first, expr.operator, expr.second)
          } else if (isNameIn(names, expr.second) && independentOf(names, expr.first)) {
            // If the quantified variable is the second argument: flip the relation
            @nowarn("msg=not.*?exhaustive")
            val op = expr.operator match {
              case LT => GT
              case LTE => GTE
              case GT => LT
              case GTE => LTE
              case EQ => EQ
            }
            (expr.second, op, expr.first)
          } else {
            dependent_bounds = dependent_bounds :+ expression; return
          }

          val name = variable.asInstanceOf[NameExpression].getName

          @nowarn("msg=not.*?exhaustive")
          val _: Unit = op match {
            case LT =>
              bounds.addUpperBound(name, create expression(Minus, bound, create constant 1))
              bounds.addUpperExclusiveBound(name, bound)
            case LTE =>
              bounds.addUpperBound(name, bound)
              bounds.addUpperExclusiveBound(name, create expression(Plus, bound, create constant 1))
            case GT =>
              bounds.addLowerBound(name, create expression(Plus, bound, create constant 1))
            case GTE =>
              bounds.addLowerBound(name, bound)
            case EQ =>
              bounds.addUpperBound(name, bound)
              bounds.addUpperExclusiveBound(name, create expression(Plus, bound, create constant 1))
              bounds.addLowerBound(name, bound)
          }
        case OperatorExpression(Member, List(elem, OperatorExpression(RangeSeq, List(low, high)))) =>
          val name = elem match {
            case expr: NameExpression if names.contains(expr.getName) =>
              expr.getName
            case _ => dependent_bounds = dependent_bounds :+ expression; return;
          }

          if (independentOf(names, low) && independentOf(names, high)) {
            bounds.addUpperBound(name, create expression(Minus, high, create constant 1))
            bounds.addUpperExclusiveBound(name, high)
            bounds.addLowerBound(name, low)
          } else {
            dependent_bounds = dependent_bounds :+ expression
          }
        case _ => dependent_bounds = dependent_bounds :+ expression
      }
    }
    select.foreach(process_bound)
    (bounds, dependent_bounds)
  }

  def rewriteMain(bounds: Bounds, main: ASTNode): Option[ASTNode] = {
    val (left, op, right) = main match {
      case exp: OperatorExpression if Set(LT, LTE, GT, GTE).contains(exp.operator) =>
        (exp.first, exp.operator, exp.second)
      case _ => return None
    }

    /* If one side is independent of the quantified variables, only emit the strongest statement.
     * e.g. forall i: 0<=i<n ==> len > i
     * equivalent to: (0<n) ==> len >= n
     */
    if(independentOf(bounds.names, left)) {
      if(Set(LT, LTE).contains(op)) {
        // constant <= right
        extremeValue(bounds, right, maximizing = false)
          .map(min => create expression(op, left, min))
      } else /* GT, GTE */ {
        // constant >= right
        extremeValue(bounds, right, maximizing = true)
          .map(max => create expression(op, left, max))
      }
    } else if(independentOf(bounds.names, right)) {
      if(Set(LT, LTE).contains(op)) {
        // left <= constant
        extremeValue(bounds, left, maximizing = true)
          .map(max => create expression(op, max, right))
      } else /* GT, GTE */ {
        // left >= constant
        extremeValue(bounds, left, maximizing = false)
          .map(min => create expression(op, min, right))
      }
    } else {
      None
    }
  }

  override def visit(expr: BindingExpression): Unit = {
    expr.binder match {
      case Binder.Forall | Binder.Star =>
        val bindings = expr.getDeclarations.map(_.name).toSet
        val (select, main) = splitSelect(rewrite(expr.select), rewrite(expr.main))
        val (independentSelect, potentialBounds) = select.partition(independentOf(bindings, _))
        val (bounds, dependent_bounds) = getBounds(bindings, potentialBounds)
        rewriteLinearArray(bounds, main, independentSelect ++ dependent_bounds, expr.binder, expr.result_type) match {
          case Some(new_forall) =>
            result = new_forall; return
          case None =>
        }
        //Only rewrite main, when the dependent bounds are not existing
        if(dependent_bounds.isEmpty){
          rewriteMain(bounds, main) match {
            case Some(main) =>
              result = create expression(Implies, (independentSelect ++ bounds.selectNonEmpty).reduce(and), main); return
            case None =>
          }
        }
        super.visit(expr)
      case _ =>
        super.visit(expr)
    }
  }
}
