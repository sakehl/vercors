package vct.parsers.rewrite

import vct.col.ast.`type`.{ASTReserved, PrimitiveSort, PrimitiveType, Type}
import vct.col.ast.expr._
import vct.col.ast.generic.ASTNode
import vct.col.ast.stmt.composite.{BlockStatement, ParallelBarrier}
import vct.col.ast.stmt.decl.{Contract, DeclarationStatement, Method, ProgramUnit}
import vct.col.ast.util._

import scala.collection.mutable
import scala.jdk.CollectionConverters._


class KernelBodyRewriter(override val source: ProgramUnit,
                         static_shared_mem_decl: mutable.Buffer[DeclarationStatement],
                         dynamic_shared_mem_declaration: mutable.Buffer[DeclarationStatement],
                         global_mem_names: Set[String]
                        ) extends AbstractRewriter(source) {

  def this(source: ProgramUnit, static_shared_mem_decl: java.util.List[DeclarationStatement],
           dynamic_shared_mem_declaration: java.util.List[DeclarationStatement],
           global_mem_names: java.util.Set[String]) =
    this(source, static_shared_mem_decl.asScala, dynamic_shared_mem_declaration.asScala, global_mem_names.asScala.toSet)

  //  class NameScanner extends RecursiveVisitor[AnyRef](null, null)
  private val shared_mem_names: mutable.Set[String] = mutable.Set()
  private val shared_mem_scalar_names: mutable.Map[String, ASTNode] = mutable.Map()
  private val shared_mem_decl: mutable.ListBuffer[DeclarationStatement] = mutable.ListBuffer()
  private var dynamic_memory_arrays = 0

  prepare_shared_mem()

  private def is_simple_type(t: Type): Boolean = {
    t match {
      case e: PrimitiveType => e.isSimple()
      case _ => false
    }
  }

  private def hasSharedMemNames(node: ASTNode): Boolean = {
    val scanner = new NameScanner
    node.accept(scanner)
    val free_names = scanner.freeNames.keySet
    free_names.intersect(shared_mem_names).nonEmpty
  }

  private def prepare_shared_mem(): Unit = {
    dynamic_memory_arrays = dynamic_shared_mem_declaration.size

    for (decl <- static_shared_mem_decl) {
      val t = decl.getType
      t match {
        case e: PrimitiveType =>
          e.sort match {
            case PrimitiveSort.Pointer => Fail("Pointer not supported as static shared memory in VerCors: %s", decl)
            case PrimitiveSort.Array =>
              if (e.nrOfArguments != 2) Fail("Not enough arguments for shared array type: %s", decl)
              if (!e.firstarg.isInstanceOf[Type]) Fail("Type not supported as shared memory: %s",decl)
              val main_type = e.firstarg.asInstanceOf[Type]
              if (!is_simple_type(main_type)) Fail("Type not supported as shared memory: %s", decl)
              if (ExpressionEquallityCheck.is_constant_int(e.secondarg).isEmpty)
                Fail("Not a constant argument to shared array type: %s", decl)

              val new_type = create.primitive_type(PrimitiveSort.Option, create.primitive_type(PrimitiveSort.Array,
                create.primitive_type(PrimitiveSort.Cell, main_type)))
              shared_mem_names.add(decl.name)
              val new_decl = create.field_decl(decl.getOrigin,
                decl.name, new_type, create.expression(StandardOperator.NewArray, new_type, e.secondarg))
              shared_mem_decl += new_decl
            case _ if e.isSimple =>
              val new_type = create primitive_type(PrimitiveSort.Option, create primitive_type(PrimitiveSort.Array,
                create.primitive_type(PrimitiveSort.Cell, e)))
              val new_decl = create field_decl(decl.getOrigin,
                decl.name, new_type, create expression(StandardOperator.NewArray, new_type, create constant(1)))
              shared_mem_decl += new_decl
              val new_replacement = create expression(StandardOperator.Subscript, create local_name(decl.name) ,create constant(0))
              shared_mem_names.add(decl.name)
              shared_mem_scalar_names.addOne(decl.name, new_replacement)
            case _ => Fail("Wrong type for shared memory: %s", decl)
          }
      }
    }
    var i = 0

    for (decl <- dynamic_shared_mem_declaration) {
      i += 1
      val t = decl.getType
      t match {
        case e: PrimitiveType if (e.sort eq PrimitiveSort.Pointer) || (e.sort eq PrimitiveSort.Array) =>
          if (e.nrOfArguments != 1) Fail("To many arguments for dynamic shared array type: %s", decl)

          val main_type: Type = e.firstarg match {
            case t: Type => t
            case _ => Fail("Type not supported as shared memory: %s", decl); return
          }

          if (!is_simple_type(main_type)) Fail("Type not supported as shared memory: %s", decl)

          shared_mem_names.add(decl.name)
          val new_type = create.primitive_type(PrimitiveSort.Option, create.primitive_type(PrimitiveSort.Array,
            create.primitive_type(PrimitiveSort.Cell, main_type)))
          val new_decl = create.field_decl(decl.name, new_type,
            create.expression(StandardOperator.NewArray, new_type,
              create.local_name("shared_mem_size_" + i)))
          shared_mem_decl += new_decl
        case _ => Fail("Wrong type for dynamic shared memory: %s", decl)
      }
    }
  }

  override def visit(e: MethodInvokation): Unit = {
    e.method match {
      case "get_global_id" =>
        val arg = e.getArg(0)
        if (arg.isConstant(0)) {
          result = plus(mult(create.local_name("opencl_gid"), create.local_name("opencl_gsize")), create.local_name("opencl_lid"))
        }
        else Fail("bad dimension: %s", arg)
      case "get_local_id" =>
        val arg = e.getArg(0)
        if (arg.isConstant(0)) result = create.local_name("opencl_lid")
        else Fail("bad dimension: %s", arg)
      case "get_group_id" =>
        val arg = e.getArg(0)
        if (arg.isConstant(0)) result = create.local_name("opencl_gid")
        else Fail("bad dimension: %s", arg)
      case "get_local_size" =>
        val arg = e.getArg(0)
        if (arg.isConstant(0)) result = create.local_name("opencl_gsize")
        else Fail("bad dimension: %s", arg)
      case "get_num_groups" =>
        val arg = e.getArg(0)
        if (arg.isConstant(0)) result = create.local_name("opencl_gcount")
        else Fail("bad dimension: %s", arg)
      case _ =>
        super.visit(e)
    }
  }

  override def visit(n: NameExpression): Unit = {
    if (n.kind ne NameExpressionKind.Reserved) {
      super.visit(n)
      return
    }
    n.reserved match {
      case ASTReserved.GlobalThreadId =>
        result = plus(mult(create.local_name("opencl_gid"), create.local_name("opencl_gsize")),
          create.local_name("opencl_lid"))
      case ASTReserved.LocalThreadId =>
        result = create.local_name("opencl_lid")
      case _ =>
        super.visit(n)
    }
  }

  override def visit(old_m: Method): Unit = {
    val decls = mutable.ListBuffer[DeclarationStatement]()
    val inner_decl = create.field_decl("opencl_lid", create.primitive_type(PrimitiveSort.Integer),
      create.expression(StandardOperator.RangeSeq, create.constant(0), create.local_name("opencl_gsize")))
    val outer_decl = create.field_decl("opencl_gid", create.primitive_type(PrimitiveSort.Integer),
      create.expression(StandardOperator.RangeSeq, create.constant(0), create.local_name("opencl_gcount")))
    val icb = new ContractBuilder // thread contract
    val gcb = new ContractBuilder // group contract
    gcb.requires(create.constant(true))
    val kcb = new ContractBuilder // kernel contract
    kcb given create.field_decl("opencl_gcount", create.primitive_type(PrimitiveSort.Integer))
    kcb given create.field_decl("opencl_gsize", create.primitive_type(PrimitiveSort.Integer))
    for (i <- 1 to dynamic_memory_arrays) {
      kcb given create.field_decl("shared_mem_size_" + i, create.primitive_type(PrimitiveSort.Integer))
    }

    val m : Method = if(shared_mem_scalar_names.nonEmpty){
      val subst = new Substituter(source, shared_mem_scalar_names.toMap)
      old_m.apply(subst).asInstanceOf[Method]
    } else {
      old_m;
    }

    for (shared_array <- shared_mem_names) {
      icb.requires(create.non_null(create.local_name(shared_array)))
    }
    val returns = rewrite(m.getReturnType)
    for (d <- m.getArgs) {
      decls += rewrite(d)
    }
    var c = m.getContract
    if (c == null) c = new ContractBuilder().getContract(false)
    rewrite(c, icb)
    icb.clearKernelInvariant()
    icb.clearGivenYields()
    gcb.appendInvariant(rewrite(c.invariant))
    kcb.appendInvariant(rewrite(c.invariant))
    kcb.context(rewrite(c.kernelInvariant))

    for(clause <- ASTUtils.conjuncts(c.pre_condition, StandardOperator.Star).asScala ) {
      val new_clause = rewrite(clause)
      val has_shared_mem_var = hasSharedMemNames(new_clause)
      if (!has_shared_mem_var) {
        val group = create.starall(create.expression(StandardOperator.Member, create.local_name("opencl_lid"),
          create.expression(StandardOperator.RangeSeq, create.constant(0), create.local_name("opencl_gsize"))),
          new_clause, create.field_decl("opencl_lid", create.primitive_type(PrimitiveSort.Integer)))
        gcb.requires(group)
        kcb.requires(create.starall(create.expression(StandardOperator.Member, create.local_name("opencl_gid"),
          create.expression(StandardOperator.RangeSeq, create.constant(0), create.local_name("opencl_gcount"))),
          group, create.field_decl("opencl_gid", create.primitive_type(PrimitiveSort.Integer))))
      }
    }
    kcb.given(rewrite(c.given):_*)
    kcb.yields(rewrite(c.yields):_*)
    var body = rewrite(m.getBody).asInstanceOf[BlockStatement]

    val group_block = create.region(null, create.parallel_block("group_block",
      icb.getContract(false), Array(inner_decl), body))
    val inner_region: Array[ASTNode]  = shared_mem_decl.toArray.appended(group_block)

    body = create.block(inner_region:_*)

    body = create.block(create.region(null, create.parallel_block("kernel_block",
      gcb.getContract(false), Array(outer_decl), body)))
    body = create.block(create.invariant_block("__vercors_kernel_invariant__", rewrite(c.kernelInvariant), body))
    result = create.method_decl(returns, kcb.getContract(false), m.name, decls.toArray, body)
  }

  override def visit(e: OperatorExpression): Unit = e.operator match {
    case StandardOperator.StructSelect =>
      if (e.arg(1).isName("x"))
        if (e.arg(0).isName("threadIdx")) result = name("opencl_lid")
        else if (e.arg(0).isName("blockIdx")) result = name("opencl_gid")
        else if (e.arg(0).isName("blockDim")) result = name("opencl_gsize")
        else if (e.arg(0).isName("gridDim")) result = name("opencl_gcount")
        else super.visit(e)
      else super.visit(e)
    case _ =>
      super.visit(e)
  }

  override def visit(pb: ParallelBarrier): Unit = if ((pb.label eq "group_block") && pb.gpu_specifier != null) {
    //The gpu specifier has the value for the type of memory fence we can get.
    // 1 is for shared fence, 2 is for global fence. They can be ORed together (|), giving the value 3
    // Actually the user could give an integer value at the moment, and shouldn't but this is okay for now.
    val specifier = ExpressionEquallityCheck.is_constant_int(pb.gpu_specifier)
    if (specifier.isEmpty) Fail("The barrier should have arguments like `CLK_GLOBAL_MEM_FENCE`: `barrier(%s)", pb.gpu_specifier)
    val specifier_value = specifier.get
    var global_fence = false
    var shared_fence = false
    specifier_value match {
      case 1 =>
        shared_fence = true
      case 2 =>
        global_fence = true
      case 3 =>
        global_fence = true
        shared_fence = true
      case _ =>
        Fail("The barrier should have arguments like `CLK_GLOBAL_MEM_FENCE`: `barrier(%s)", pb.gpu_specifier)
    }
    val cb = new ContractBuilder()

    cb.requires(create.expression(StandardOperator.LTE, create.constant(0), create.local_name("opencl_lid")))
    cb.requires(create.expression(StandardOperator.LT, create.local_name("opencl_lid"),
      create.local_name("opencl_gsize")))
    cb.requires(create.expression(StandardOperator.LTE, create.constant(0), create.local_name("opencl_gid")))
    cb.requires(create.expression(StandardOperator.LT, create.local_name("opencl_gid"),
      create.local_name("opencl_gcount")))
    for (shared_array <- shared_mem_names) {
      cb.requires(create.non_null(create.local_name(shared_array)))
    }

    rewrite(pb.contract, cb)

    val new_contract = cb.getContract()

    val scanner = new PermissionScanner()
    new_contract.accept(scanner)
    val permission_names = scanner.permission_names
    if (!shared_fence && permission_names.intersect(shared_mem_names).nonEmpty)
      Fail("A CLK_LOCAL_MEM_FENCE barrier is needed to redistribute local memory permissions.\n If permissions are not redistributed but are needed, use `context`.\n %s", pb)
    if (!global_fence && permission_names.intersect(global_mem_names).nonEmpty)
      Fail("A CLK_GLOBAL_MEM_FENCE barrier is needed to redistribute global memory permissions. \n If permissions are not redistributed but are needed, use `context`.\n %s", pb)

    result = create.barrier("group_block", new_contract, pb.invs, rewrite(pb.body))
  }
  else Fail("Incorrect barrier specified: %s", pb)
}

// In this class we scan the permissions of a barrier contract
// We only record the names of the arrays/variables for which the permissions are changed (e.g. not context permissions).
// Therefore, we can see if a local memory fence of global memory fence is needed for the barrier
class PermissionScanner() extends RecursiveVisitor[AnyRef](null, null) {
  val permission_names: mutable.Set[String] = mutable.Set()

  val perm_operators : Set[StandardOperator] =
    Set(StandardOperator.Perm, StandardOperator.Value, StandardOperator.PointsTo)

  val not_supported : Set[StandardOperator] =
    Set(StandardOperator.ArrayPerm, StandardOperator.HistoryPerm,
      StandardOperator.ActionPerm, StandardOperator.CurrentPerm)

  override def visit(e: OperatorExpression): Unit = {
    if(perm_operators contains e.operator) e.first match {
      case OperatorExpression(StandardOperator.Subscript, args) =>
        args.head match {
          case NameExpression(name, _, _) =>
            permission_names.add(name)
          case _ => Fail("Not a sort of permission that is supported, %s", e)
        }
      case NameExpression(name, _, _) =>
        permission_names.add(name);
      case _ => Fail("Not a sort of permission that is supported, %s", e)
    }
    else if(not_supported contains e.operator)
      Fail ("This permission operator is not (yet) supported with GPU kernels: %s", e)
    else
      super.visit(e)
  }

  def my_dispatch[R <: ASTNode](objects: Array[R]): Unit = {
    for (obj <- objects) {
      if (obj != null) obj.accept(this)
    }
  }

  def my_dispatch(obj: ASTNode): Unit = {
      if (obj != null) obj.accept(this)
  }

  override def visit (c: Contract): Unit = {
    my_dispatch(c.given)
    my_dispatch(c.yields)
    if (c.modifies != null) {
      my_dispatch(c.modifies)
    }
    if (c.accesses != null) {
      my_dispatch(c.accesses)
    }
    my_dispatch(c.invariant)
    //my_dispatch(c.pre_condition)
    //my_dispatch(c.post_condition)

    // We do not record permission accesses for context elements, since these do not swap permissions.
    val contextElems : mutable.ListBuffer[ASTNode] = mutable.ListBuffer()
    for (pre <- ASTUtils.conjuncts(c.pre_condition, StandardOperator.Star).asScala) {
      var added = false
      for (post <- ASTUtils.conjuncts(c.post_condition, StandardOperator.Star).asScala) {
        if (pre == post) {
          contextElems.addOne(pre)
          added = true
        }
      }
      if (!added) my_dispatch(pre)
    }
    for (post <- ASTUtils.conjuncts(c.post_condition, StandardOperator.Star).asScala) {
      if (!contextElems.contains(post)) my_dispatch(post)
    }

    my_dispatch(c.signals)
  }
}