package vct.parsers.rewrite;

import hre.ast.MessageOrigin;
import vct.col.ast.expr.KernelInvocation;
import vct.col.ast.expr.MethodInvokation;
import vct.col.ast.generic.ASTNode;
import vct.col.ast.stmt.composite.BlockStatement;
import vct.col.ast.stmt.decl.ASTClass.ClassKind;
import vct.col.ast.stmt.decl.*;
import vct.col.ast.type.PrimitiveSort;
import vct.col.ast.type.PrimitiveType;
import vct.col.ast.type.Type;
import vct.col.ast.type.TypeExpression;
import vct.col.ast.util.AbstractRewriter;
import vct.col.ast.util.ContractBuilder;

import java.util.*;

import static vct.col.ast.type.PrimitiveSort.*;

public class ConvertTypeExpressions extends AbstractRewriter {
  private Boolean in_kernel = false;
  private Boolean in_kernel_parameters = false;
  private List<DeclarationStatement> static_shared_mem_declarations;// = new ArrayList<>();
  private List<DeclarationStatement> dynamic_shared_mem_declaration;// = new ArrayList<>();
  private Set<String> global_mem_names;// = new HashSet<>();
  private Boolean opencl = false;
  private Boolean cuda = false;

  private final ASTClass kernels;
  
  public ConvertTypeExpressions(ProgramUnit source) {
    super(source);
    switch(source.sourceLanguage){
      case OpenCLorC:
        opencl = true;
        break;
      case CUDA:
        cuda = true;
        break;
      default:
    }
    create.enter();
    create.setOrigin(new MessageOrigin("Collected kernels"));
    kernels=create.ast_class("OpenCL",ClassKind.Kernel,null,null,null);
    create.leave();
  }

  // Check a declaration if it is a shared memory declaration. If this is the case, it is added to the
  // `static_shared_mem_declarations` or `dynamic_shared_mem_declaration` class attributes and it will make the result null
  // (so it can be filtered, Block statements will do this automatically), to indicate it can be removed from a GPU kernel.
  // The removed declarations will be added to the correct place during the KernelBodyRewriter rewrite step.
  @Override
  public void visit(DeclarationStatement d){
    boolean extern=false;
    boolean static_type=false;
    boolean local=false;
    boolean global=false;
    boolean pointer = false;
    boolean array = false;
    Type t=d.getType();
    Stack<PrimitiveSort> pointer_stack = new Stack<>();
    Stack<List<ASTNode>> array_arg_stack = new Stack<>();

    while(t instanceof PrimitiveType){
      PrimitiveType e = (PrimitiveType)t;
      if(e.sort == Pointer || e.sort == Array) {
        t = (Type)e.firstarg();
        if(e.sort == Pointer) pointer = true; else array = true;
        pointer_stack.push(e.sort);
        List<ASTNode> tail = new ArrayList<>(e.argsJava().subList(1, e.argsJava().size()));
        array_arg_stack.push(tail);
      } else if(e.sort == Cell){
        t = (Type)e.firstarg();
      } else if (e.sort == Option) {
        t = (Type) e.firstarg();
      } else break;
    }

    while(t instanceof TypeExpression){
      TypeExpression e=(TypeExpression)t;
      switch (e.operator()) {
        case Static:
          t=e.firstType();
          static_type = true;
          break;
        case Extern:
          extern=true;
          t=e.firstType();
          break;
        case Local:
          local=true;
          t=e.firstType();
          break;
        case Global:
          global = true;
          t=e.firstType();
          break;
        default:
          Fail("cannot deal with type operator %s", e.operator());
      }
    }

    while(!pointer_stack.empty()){
      List<ASTNode> args = array_arg_stack.pop();
      args.add(0, t);
      t = new PrimitiveType(pointer_stack.pop(), args);
    }

    DeclarationStatement res = create.field_decl(d.name(), rewrite(t), rewrite(d.initJava()));
    if (extern) {
      res.setFlag(ASTFlags.EXTERN, true);
    }
    result = res;

    if(!in_kernel) {
      if (!(global || local)) return;
      else Fail("Cannot specify the `global` or `local` address space in a non GPU kernel: '%s'", d);
    }

    // We are now in a GPU kernel (in_kernel == True)
    if(opencl){
      if(extern || static_type){
        Fail("VerCors does not support `static` or `extern` type qualifiers in OpenCL kernels: '%s'",d);
      }

      if(in_kernel_parameters && array)
        Fail("Cannot have a array parameter for a kernel in OpenCL: '%s'",d);

      if(!(global || local)){
        if(in_kernel_parameters && pointer)
          Fail("Cannot have a kernel pointer parameter without `global` or `local` address space in OpenCL: '%s'",d);
        return;
      }
      // We now have a global or local variable

      if(global && local) Fail("Cannot have a a type with both `global` and `local` address space: '%s'",d);

      if(d.initJava() != null) Fail("A local or global type cannot have an initialized value: '%s'", d);
      // We can not have local or global scalar types in the parameters
      if(in_kernel_parameters && !pointer)
        Fail("Cannot have a scalar kernel parameter with `global` or `local` address space in OpenCL: '%s'",d);


      if(global){
        if(in_kernel_parameters){
          global_mem_names.add(d.name());
          return;
        }
        Fail("The `global` type operator is not allowed in a kernel body: '%s'", d);
      } else {
        // We have a local variable
        if(in_kernel_parameters){
          dynamic_shared_mem_declaration.add(res);
          result = null;
          return;
        } else{

          if(is_in_kernel_function_scope(d)){
            static_shared_mem_declarations.add(res);
            result = null;
            return;
          }
          Fail("A local memory declaration is only allowed in the (top) kernel function scope: '%s'", d);
        }
      }
    } else if(cuda){
      if(global) Fail("The `global` type operator is not allowed in Cuda: '%s'", d);
      if(in_kernel_parameters && array) Fail("Cannot have a array parameter for a kernel in OpenCL: '%s'",d);

      if(!local){
        if(extern || static_type)
          Fail("VerCors does not support `static` or `extern` type qualifiers in CUDA kernels: '%s'",d);
        if(in_kernel_parameters && pointer) global_mem_names.add(d.name());
        return;
      }
      // We have a local variable
      if(in_kernel_parameters)
        Fail("The `local` (__shared__) type operator is not allowed as kernel parameter in Cuda: '%s'", d);
      if(static_type)
        Fail("VerCors does not support `static local` type qualifiers in CUDA kernels: '%s'",d);

      // Apparently this is not needed in CUDA, but we didn't build the logic around this thus disallow it
      if(!is_in_kernel_function_scope(d))
        Fail("A local (__shared__) memory declaration is only supported in the (top) kernel function scope: '%s'", d);

      if(d.initJava() != null) Fail("A local (__shared__) type cannot have an initialized value: '%s'", d);

      if(extern){
        if(dynamic_shared_mem_declaration.size() != 0)
          Fail("Cannot have multiple `extern __shared__` declarations in a CUDA kernel: %s", d);
        dynamic_shared_mem_declaration.add(res);
        result = null;
        return;
      }

      static_shared_mem_declarations.add(res);
      result = null;
    }
  }

  private boolean is_in_kernel_function_scope(ASTNode node){
    if(!in_kernel) return false;

    boolean in_kernel_function_scope = false;

    ASTNode parent = node.getParent();
    if(parent instanceof BlockStatement) parent = parent.getParent();
    // We are in a kernel, thus if the parent is a method, we are at top level
    if(parent instanceof Method) in_kernel_function_scope = true;

    return in_kernel_function_scope;
  }

  @Override
  public void visit(Method m){

    Type t=m.getReturnType();
    boolean kernel = false;
    boolean extern = false;
    boolean static_type = false;
    while(t instanceof TypeExpression){
      TypeExpression e=(TypeExpression)t;
      switch (e.operator()) {
      case Static:
        static_type = true;
        t=e.firstType();
        break;
      case Extern:
        extern = true;
        t=e.firstType();
        break;        
      case Kernel:
        kernel = true;
        in_kernel = true;
        in_kernel_parameters = true;
        t=e.firstType();
        break;        
      default:
        Fail("cannot deal with type operator %s", e.operator());
      }
    }
    Debug("remaining type of %s is %s",m.getReturnType(),t);
    if(kernel){
      static_shared_mem_declarations = new ArrayList<>();
      dynamic_shared_mem_declaration = new ArrayList<>();
      global_mem_names = new HashSet<>();
    }

    // This is copied from AbstractRewriter, since we need to indicate when we rewrite the kernel parameters
    String name=m.getName();
    if (currentContractBuilder==null) currentContractBuilder=new ContractBuilder();

    DeclarationStatement[] args = m.getArgs();
    args = rewrite(args);
    // Filter out the removed local (shared) memory args
    args = Arrays.stream(args).filter(Objects::nonNull).toArray(DeclarationStatement[]::new);
    in_kernel_parameters = false;

    Contract mc=m.getContract();
    Contract c;
    // Ensure we maintain the type of emptiness of mc
    // If the contract was null previously, the new contract can also be null
    // If the contract was non-null previously, the new contract cannot be null
    if (mc!=null) {
      rewrite(mc,currentContractBuilder);
      c = currentContractBuilder.getContract(false);
    } else {
      c = currentContractBuilder.getContract(true);
    }

    if (mc != null && c != null && c.getOrigin() == null) {
      c.setOrigin(mc.getOrigin());
    }
    currentContractBuilder=null;

    Method.Kind kind=m.kind;
    Type[] signals = rewrite(m.signals);
    ASTNode body=rewrite(m.getBody());

    Method out=create.method_kind(kind, t, signals, c, name, args, m.usesVarArgs(), body);
    out.copyMissingFlags(m);

    if(static_type) out.setStatic(true);
    if(extern) out.setFlag(ASTFlags.EXTERN,true);
    in_kernel = false;
    if(kernel){
      KernelBodyRewriter kbr = new KernelBodyRewriter(source(), static_shared_mem_declarations,
              dynamic_shared_mem_declaration, global_mem_names);
      result=kbr.rewrite(out);

    } else {
      result=out;
    }
  }

  @Override
  public void visit(KernelInvocation ki) {
    MethodInvokation res = create.invokation(null, null, ki.method(), rewrite(ki.javaArgs()));

    res.get_before().addStatement(
            create.assignment(create.unresolved_name("opencl_gcount"), rewrite(ki.blockCount())));
    res.get_before().addStatement(
            create.assignment(create.unresolved_name("opencl_gsize"), rewrite(ki.threadCount())));

    if(ki.sharedMemorySize() != null) {
      res.get_before().addStatement(
              create.assignment(create.unresolved_name("shared_mem_size_1"), rewrite(ki.sharedMemorySize())));
    }

    result = res;
  }

  @Override
  public
  ProgramUnit rewriteAll(){
    ProgramUnit pu=super.rewriteAll();
    if (kernels.size()>0){
      pu.add(kernels);
    }
    return pu;
  }
}
