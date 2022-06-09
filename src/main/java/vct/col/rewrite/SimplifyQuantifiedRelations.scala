package vct.col.rewrite

import vct.col.ast.`type`.{PrimitiveSort, PrimitiveType, Type}
import vct.col.ast.expr.StandardOperator._
import vct.col.ast.expr._
import vct.col.ast.expr.constant.ConstantExpression
import vct.col.ast.generic.ASTNode
import vct.col.ast.stmt.composite.{ForEachLoop, LoopStatement}
import vct.col.ast.stmt.decl.{ASTClass, ASTSpecial, Contract, DeclarationStatement, Method, ProgramUnit}
import vct.col.ast.util.{ASTUtils, AbstractRewriter, AnnotationVariableInfoGetter, ContractBuilder, ExpressionEqualityCheck, NameScanner, RecursiveVisitor, Substituter}
import vct.col.ast.util.ExpressionEqualityCheck.{equal_expressions, is_constant_int}

import scala.annotation.nowarn
import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.jdk.CollectionConverters.IterableHasAsScala

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
    val sub = new Substituter(source, substitute_vars)
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

  class Bounds(val names: Set[String], val lowerBounds: mutable.Map[String, Seq[ASTNode]],
               val upperBounds: mutable.Map[String, Seq[ASTNode]],
               val upperExclusiveBounds: mutable.Map[String, Seq[ASTNode]] ) {
    def this(names: Set[String]) = {
      this(names, mutable.Map(), mutable.Map(), mutable.Map())
    }

    def addLowerBound(name: String, bound: ASTNode): Unit =
      lowerBounds(name) = lowerBounds.getOrElse(name, Seq()) :+ bound

    def addUpperBound(name: String, bound: ASTNode): Unit =
      upperBounds(name) = upperBounds.getOrElse(name, Seq()) :+ bound

    def addUpperExclusiveBound(name: String, bound: ASTNode): Unit =
      upperExclusiveBounds(name) = upperExclusiveBounds.getOrElse(name, Seq()) :+ bound

    def extremeValue(name: String, maximizing: Boolean, add_to_extremes: Boolean = true): Option[ASTNode] =
      (if(maximizing) upperBounds else lowerBounds).get(name) match {
        case None => None
        case Some(bounds) => Some(SimplifyQuantifiedRelations.this.extremeValue(bounds, !maximizing, add_to_extremes))
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
            if(getNames(expr.second).intersect(bounds.names).isEmpty) {
              super.visit(expr)
              return
            }
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
      } else{
        super.visit(expr)
      }
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

      /**
        * This function determines if the vars in this specific order allow the forall to be rewritten to one
        * forall.
        *
        * Precondition:
        *   * At least one var in `vars`
        *   * linear_expressions has an expression for all `vars`
        *   * variable_bounds.upperExclusiveBounds has a non-empty list for all `vars`
        *
        * We are looking for patterns:
        *   /\_{0<=i<k} {0 <= x_i < n_i} : ... ar[Sum_{0<=i<k} {a_i * x_i} + b] ...
        * and we require that for i>0
        *   a_i == a_{i-1} * n_{i-1}
        *   (or equivalent a_i == Prod_{0<=j<i} {n_j} * a_0 )
        *
        * Further more we require that n_i>0 and a_i>0 (although I think a_0<0 is also valid)
        * TODO: We are not checking n_i and a_i on this
        * We can than replace the forall with
        *   b <= x_new < a_{k-1} * n_{k-1} + b && (x_new - b) % a_0 == 0 : ... ar[x_new] ...
        * and each x_i gets replaced by
        *   x_i -> ((x_new - b) / a_i) % n_i
        *   And since we never go past a_{k-1} * n_{k-1} + b, no modulo needed here
        * x_{k-1} -> (x_new - b) / a_{k-1} */
      def check_vars_list(vars: List[String]): Option[SubstituteForall] = {
        val x_0 = vars.head
        val a_0 = linear_expressions(x_0)
        // x_{i-1}, a_{i-1}, n_{i-1}
        var x_i_last = x_0
        var a_i_last = a_0
        var n_i_last: ASTNode = null
        val ns : mutable.Map[String, ASTNode] = mutable.Map();

        val x_new = vars.mkString("_")
        // x_base == (x_new -b)
        val x_base = constant_expression match {
          case None => create identifier x_new
          case Some(b) => create expression(Minus, create identifier x_new, b)
        }
        val replace_map:  mutable.Map[String, ExpressionNode] = mutable.Map()

        for(x_i <- vars.tail){
          val a_i = linear_expressions(x_i)
          var found_valid_n = false

          // Find a suitable upper bound
          for (n_i_last_candidate <- variable_bounds.upperExclusiveBounds(x_i_last)) {
            if( !found_valid_n && equality_checker.equal_expressions(a_i, simplified_mult(a_i_last, n_i_last_candidate)) ) {
              found_valid_n = true
              n_i_last = n_i_last_candidate
              ns(x_i_last) = n_i_last_candidate
            }
          }

          if(!found_valid_n) return None
          // We now know the valid bound of x_{i-1}
          //  x_{i-1} -> ((x_new -b) / a_{i-1}) % n_{i-1}
          replace_map(x_i_last) =
            if(is_value(a_i_last, 1))
              create expression (Mod, x_base, n_i_last)
          else
              create expression (Mod, create expression(FloorDiv, x_base, a_i_last), n_i_last)

          // Yay we are good up to now, go check out the next i
          x_i_last = x_i
          a_i_last = a_i
          n_i_last = null
        }
        // Add the last value, no need to do modulo
        replace_map(x_i_last) = create expression(FloorDiv, x_base, a_i_last)
        // Get a random upperbound for x_i_last;
        n_i_last = variable_bounds.upperExclusiveBounds(x_i_last).head
        ns(x_i_last) = n_i_last
        // 0 <= x_new - b < a_{k-1} * n_{k-1}
        var new_bounds = create expression(And,
            create expression(LTE, create constant 0, x_base),
            create expression(LT, x_base, simplified_mult(a_i_last, n_i_last))
        )
        // && (x_new - b) % a_0 == 0
        new_bounds = if(is_value(a_0, 1)) new_bounds else
          create expression(And, new_bounds,
            create expression(EQ, create expression (Mod, x_base, a_0),
              create constant 0)
          )

        for(x_i <- vars){
          val n_i = ns(x_i)
          // Remove the upper bound we used, but keep the others
          for(old_upper_bound <- variable_bounds.upperExclusiveBounds(x_i)){
            if(old_upper_bound != n_i){
              new_bounds = create expression(And, create expression(LT, replace_map(x_i), old_upper_bound), new_bounds)
            }
          }

          // Remove the lower zero bound, but keep the others
          for(old_lower_bound <- variable_bounds.lowerBounds(x_i))
            if(!is_value(old_lower_bound, 0))
              new_bounds = create expression(And, create expression(LTE, old_lower_bound, replace_map(x_i)), new_bounds)

          // Since we know the lower bound was also 0, and the we multiply the upper bounds,
          // we do have to require that each upper bound is at least bigger than 0.
          new_bounds = create expression(And, create expression (LT, create constant 0, n_i), new_bounds)
        }

        Some(SubstituteForall(x_new, new_bounds, replace_map.toMap))
      }

      def simplified_mult(lhs: ASTNode, rhs: ASTNode): ASTNode = {
        if (is_value(lhs, 1)) rhs
        else if (is_value(rhs, 1)) lhs
        else create expression(Mult, lhs, rhs)
      }

      // Checking the preconditions
      if(variable_bounds.names.isEmpty) return None
      for(v <- variable_bounds.names){
        if(!(linear_expressions.contains(v) &&
          variable_bounds.upperExclusiveBounds.contains(v) &&
          variable_bounds.upperExclusiveBounds(v).nonEmpty)
        ) {
          return None
        }
      }

      for(vars <- variable_bounds.names.toList.reverse.permutations){
        check_vars_list(vars) match {
          case Some(subst) => return Some(subst)
          case None =>
        }
      }
      None
    }

    def is_value(e: ASTNode, x: Int): Boolean =
      equality_checker.is_constant_int(e) match {
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

  class rewritePerm(factor: ASTNode) extends AbstractRewriter(source) {
    override def visit(expr: OperatorExpression): Unit = {
      expr.operator match {
        case Perm => result = create expression(Perm, expr.first, create expression(Mult, factor, expr.second))
        case Value =>
        case ArrayPerm | HistoryPerm | ActionPerm | CurrentPerm | PointsTo
             => Fail("Permission operator not supported in rewriting quantified relations: %s", expr)
        case _ => super.visit(expr)
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
  def rewriteLinearArray(old_bounds: Bounds, old_main: ASTNode, old_indep_select: Seq[ASTNode], old_dep_select: Seq[ASTNode],
                         binder: Binder, result_type: Type): Option[ASTNode] = {
    var main = old_main
    var bounds = old_bounds
    var indep_select = old_indep_select
    var dep_select = old_dep_select

    var changed = false
    // Check if a variable has an equal upper and lower bound, meaning it just takes on one value
    for(name <- old_bounds.names){
      val equal_bounds = bounds.lowerBounds(name).toSet.intersect(bounds.upperBounds(name).toSet)
      if(equal_bounds.nonEmpty){
        changed = true
        val new_value = equal_bounds.head
        val replacer = substituteNode(Map(name -> new_value), _)
        main = replacer(main)
        indep_select = indep_select.map(replacer)
        // Some dependent selects, might now have become independent or even bounds
        val replaced_dep_select = dep_select.map(replacer)
        val remaining_names = bounds.names - name

        val (additional_indep_select, potentialBounds) = replaced_dep_select.partition(independentOf(remaining_names, _))
        val (new_bounds, new_dep_select) = getBounds(remaining_names, potentialBounds)
        dep_select = new_dep_select
        indep_select ++= additional_indep_select
        // Add the other bounds from the replaced variables to the indep_select
        bounds.lowerBounds(name).foreach( lb =>
          if(lb != new_value) indep_select ++= Seq(create expression(LTE, lb, new_value)) )
        bounds.upperBounds(name).foreach( ub =>
          if(ub != new_value) indep_select ++= Seq(create expression(LTE, new_value, ub)) )
        // Finally add the bounds from the other variables again to the new bounds
        for(other_name <- bounds.names){
          if(name != other_name){
            bounds.lowerBounds(other_name).foreach(new_bounds.addLowerBound(other_name, _))
            bounds.upperBounds(other_name).foreach(new_bounds.addUpperBound(other_name, _))
            bounds.upperExclusiveBounds(other_name).foreach(new_bounds.addUpperExclusiveBound(other_name, _))
          }
        }
        bounds = new_bounds
      }
    }


    // Check if variables are independent in the main
    // This means we can remove variables from the forall (e.g. forall int i,j... to forall int i)
    val new_old_bounds = bounds
    for(name <- new_old_bounds.names){
      if(independentOf(Set(name), main)){
        var independent = true
        dep_select.foreach(s => if(!independentOf(Set(name), s)) independent = false)
        if(independent){
          // We can freely remove this named variable
          val max_bound = bounds.extremeValue(name, maximizing = true)
          val min_bound = bounds.extremeValue(name, maximizing = false)
          (max_bound,min_bound) match {
            case (Some(max_bound), Some(min_bound)) =>
              changed = true
              bounds = new Bounds(bounds.names - name, bounds.lowerBounds -= name,
                bounds.upperBounds -= name, bounds.upperExclusiveBounds -= name)
              // We remove the forall variable i, but need to rewrite some expressions
              // (forall i; a <= i <= b; ...Perm(ar, x)...) =====> b>=a ==> ...Perm(ar, x*(b-a+1))...
              indep_select = indep_select.appended(create expression(StandardOperator.GTE, max_bound, min_bound))
              val rp = new rewritePerm(
                create expression(Plus, constant(1), create expression(Minus, max_bound, min_bound)))
              main = rp.rewrite(main)
            case _ =>
          }
        }
      }
    }
    // Always return result, even if we could not rewrite further
    val result : Option[ASTNode] = if(changed){
      if(bounds.names.isEmpty){
        val select = (indep_select ++ dep_select)
        if (select.isEmpty) Some(main) else Some(create expression(Implies, select.reduce(and), main))
      } else{
        var select = indep_select ++ dep_select
        bounds.upperExclusiveBounds.foreach {
          case (n: String, upperBounds: Seq[ASTNode]) =>
            val i = create local_name n
            upperBounds.foreach(upperBound =>
              select = select.appended(create expression(LT, i, upperBound))
            )
        }
        bounds.lowerBounds.foreach {
          case (n: String, lowerBounds: Seq[ASTNode]) =>
            val i = create local_name n
            lowerBounds.foreach(lowerBound =>
              select = select.appended(create expression(LTE, lowerBound, i))
            )
        }
        val declarations = bounds.names.toArray.map(create field_decl(_, new PrimitiveType(PrimitiveSort.Integer)))
        val select_reduced = if(select.nonEmpty) select.reduce(and) else create constant(true)
        val forall = create binder(binder, result_type, declarations, Array(), select_reduced, main)
        Some(forall)
      }
    } else {
      None
    }


    // For now we only allow forall's with one or two variables
    // TODO: Generalize this
    if(bounds.names.isEmpty){
      return result
    }

    //We need to check some preconditions for more than 2 vars
    if(bounds.names.size >= 2) {

      val scanner = new NameScanner()
      main.accept(scanner)
      // Check if we can make a new nice name, which we will use, but this is only possible when it is not a free variable
      val possible_names = bounds.names.toList.permutations.map(_.mkString("_")).toSet
      if (scanner.freeNames.keySet.intersect(possible_names).nonEmpty)
        return result
    }

    // This allows only forall's to be rewritten, if they have at least one lower bound of zero
    // TODO: Generalize this, so we don't have this restriction
    for (name <- bounds.names) {
      var one_zero = false
      bounds.lowerBounds.getOrElse(name, Seq())
        .foreach(lower => equality_checker.is_constant_int(lower) match {
          case Some(0) => one_zero = true
          case _ =>
        })
      //Exit when notAt least one zero, or no upper bounds
      if (!one_zero || bounds.upperBounds.getOrElse(name, Seq()).isEmpty) {
        return result
      }
    }

    val linear_accesses = new RewriteLinearArrayAccesses(source, bounds)
    main = linear_accesses.rewrite(main)
    linear_accesses.substitute_forall match {
      case Some(substitute_forall) =>
        main = substituteNode(substitute_forall.substitute_old_vars, main)
        val select = (Seq(substitute_forall.new_bounds) ++ indep_select ++
            dep_select.map(substituteNode(substitute_forall.substitute_old_vars,_)))
        val select_non_empty = if (select.nonEmpty) select.reduce(and) else create constant(true)
        val declaration = create field_decl(substitute_forall.new_forall_var, new PrimitiveType(PrimitiveSort.Integer))
        val forall = create binder(binder, result_type, Array(declaration), Array(), select_non_empty, main)
        Some(forall)
      case None => result
    }
  }

  /**
   * If we want the maximum/minimum of a list of nodes, that is encoded as an ITE. If we want to compose with our own
   * pass (i.e. have nested forall's) it is nice to be able to transform something like (a < b ? a : b) back to min(a,b)
   * if we encounter it again. We could/should make nodes min/max(a, b, c, ...), but this also works for now.
   */
  val extremeOfListNode: mutable.Map[ASTNode, (Seq[ASTNode], Boolean)] = mutable.Map()

  def extremeValue(values: Seq[ASTNode], maximizing: Boolean, add_to_extremes : Boolean = true): ASTNode = {
    val result = if(values.size == 1) {
      values.head
    } else {
      val preferFirst = if(maximizing) GT else LT
      create.expression(ITE,
        create.expression(preferFirst, values(0), values(1)),
        extremeValue(values(0) +: values.drop(2), maximizing, add_to_extremes),
        extremeValue(values(1) +: values.drop(2), maximizing, add_to_extremes),
      )
    }
    if(add_to_extremes) extremeOfListNode(result) = (values, maximizing)
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

  var equality_checker: ExpressionEqualityCheck = ExpressionEqualityCheck()

  override def visit(special: ASTSpecial): Unit = {
    if(special.kind == ASTSpecial.Kind.Inhale){
      val info_getter = new AnnotationVariableInfoGetter()
      val annotations = ASTUtils.conjuncts(special.args(0), StandardOperator.Star).asScala
      equality_checker = ExpressionEqualityCheck(Some(info_getter.get_info(annotations)))

      result = create special(special.kind, rewrite(special.args):_*)

      equality_checker = ExpressionEqualityCheck()

    } else {
      result = create special(special.kind, rewrite(special.args): _*)
    }
  }

  override def visit(c: ASTClass): Unit = { //checkPermission(c);
    val name = c.getName
    if (name == null) Abort("illegal class without name")
    else {
      Debug("rewriting class " + name)
      val new_pars = rewrite(c.parameters)
      val new_supers = rewrite(c.super_classes)
      val new_implemented = rewrite(c.implemented_classes)
      val res = new ASTClass(name, c.kind, new_pars, new_supers, new_implemented)
      res.setOrigin(c.getOrigin)
      currentTargetClass = res
      val contract = c.getContract
      if (currentContractBuilder == null) currentContractBuilder = new ContractBuilder
      if (contract != null) {
        val info_getter = new AnnotationVariableInfoGetter()
        val annotations = LazyList(ASTUtils.conjuncts(contract.pre_condition, StandardOperator.Star).asScala
          , ASTUtils.conjuncts(contract.invariant, StandardOperator.Star).asScala).flatten

        equality_checker = ExpressionEqualityCheck(Some(info_getter.get_info(annotations)))
        rewrite(contract, currentContractBuilder)
        equality_checker = ExpressionEqualityCheck()
      }
      res.setContract(currentContractBuilder.getContract)
      currentContractBuilder = null

      for (i <- 0 until c.size()) {
        res.add(rewrite(c.get(i)))
      }
      result = res
      currentTargetClass = null
    }
  }

  override def visit(s: ForEachLoop): Unit = {
    val new_decl = rewrite(s.decls)
    val res = create.foreach(new_decl, rewrite(s.guard), rewrite(s.body))

    val mc = s.getContract
    if (mc != null) {
      val info_getter = new AnnotationVariableInfoGetter()
      val annotations = LazyList(ASTUtils.conjuncts(mc.pre_condition, StandardOperator.Star).asScala
        , ASTUtils.conjuncts(mc.invariant, StandardOperator.Star).asScala).flatten

      equality_checker = ExpressionEqualityCheck(Some(info_getter.get_info(annotations)))
      res.setContract(rewrite(mc))
      equality_checker = ExpressionEqualityCheck()
    } else {
      res.setContract(rewrite(mc))
    }


    res.set_before(rewrite(s.get_before))
    res.set_after(rewrite(s.get_after))
    result = res
  }

  override def visit(s: LoopStatement): Unit = { //checkPermission(s);
    val res = new LoopStatement
    var tmp = s.getInitBlock
    if (tmp != null) res.setInitBlock(tmp.apply(this))
    tmp = s.getUpdateBlock
    if (tmp != null) res.setUpdateBlock(tmp.apply(this))
    tmp = s.getEntryGuard
    if (tmp != null) res.setEntryGuard(tmp.apply(this))
    tmp = s.getExitGuard
    if (tmp != null) res.setExitGuard(tmp.apply(this))
    val mc = s.getContract
    if (mc != null) {
      val info_getter = new AnnotationVariableInfoGetter()
      val annotations = LazyList(ASTUtils.conjuncts(mc.pre_condition, StandardOperator.Star).asScala
        , ASTUtils.conjuncts(mc.invariant, StandardOperator.Star).asScala).flatten

      equality_checker = ExpressionEqualityCheck(Some(info_getter.get_info(annotations)))
      res.appendContract(rewrite(mc))
      equality_checker = ExpressionEqualityCheck()
    } else {
      res.appendContract(rewrite(mc))
    }


    tmp = s.getBody
    res.setBody(tmp.apply(this))
    res.set_before(rewrite(s.get_before))
    res.set_after(rewrite(s.get_after))
    res.setOrigin(s.getOrigin)
    result = res
  }

  override def visit(m: Method): Unit = { //checkPermission(m);
    val name = m.getName
    if (currentContractBuilder == null) {
      currentContractBuilder = new ContractBuilder
    }
    val args = rewrite(m.getArgs)
    val mc = m.getContract

    var c: Contract = null
    // Ensure we maintain the type of emptiness of mc
    // If the contract was null previously, the new contract can also be null
    // If the contract was non-null previously, the new contract cannot be null
    if (mc != null) {
      val info_getter = new AnnotationVariableInfoGetter()
      val annotations = LazyList(ASTUtils.conjuncts(mc.pre_condition, StandardOperator.Star).asScala
        , ASTUtils.conjuncts(mc.invariant, StandardOperator.Star).asScala).flatten

      equality_checker = ExpressionEqualityCheck(Some(info_getter.get_info(annotations)))

      rewrite(mc, currentContractBuilder)
      c = currentContractBuilder.getContract(false)
      equality_checker = ExpressionEqualityCheck()
    }
    else {
      c = currentContractBuilder.getContract(true)
    }
    if (mc != null && c != null && c.getOrigin == null) {
      c.setOrigin(mc.getOrigin)
    }
    currentContractBuilder = null
    val kind = m.kind
    val rt = rewrite(m.getReturnType)
    val signals = rewrite(m.signals)
    val body = rewrite(m.getBody)
    result = create.method_kind(kind, rt, signals, c, name, args, m.usesVarArgs, body)
  }

  override def visit(expr: BindingExpression): Unit = {
    expr.binder match {
      case Binder.Forall | Binder.Star =>
        val bindings = expr.getDeclarations.map(_.name).toSet
        val (select, main) = splitSelect(rewrite(expr.select), rewrite(expr.main))
        val (independentSelect, potentialBounds) = select.partition(independentOf(bindings, _))
        val (bounds, dependent_bounds) = getBounds(bindings, potentialBounds)
        //Only rewrite main, when the dependent bounds are not existing
        if(dependent_bounds.isEmpty && expr.binder != Binder.Star){
          rewriteMain(bounds, main) match {
            case Some(main) =>
              result = create expression(Implies, (independentSelect ++ bounds.selectNonEmpty).reduce(and), main); return
            case None =>
          }
        }
        rewriteLinearArray(bounds, main, independentSelect, dependent_bounds, expr.binder, expr.result_type) match {
          case Some(new_forall) =>
            result = new_forall;
            return
          case None =>
        }
        super.visit(expr)
      case _ =>
        super.visit(expr)
    }
  }
}
