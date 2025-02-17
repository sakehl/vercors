package vct.rewrite

import vct.col.ast.{Expr, Type, _}
import vct.col.origin.Origin
import vct.col.rewrite.{Generation, Rewriter, RewriterBuilder}
import vct.col.typerules.CoercingRewriter
import vct.result.VerificationError.UserError
import hre.util.ScopedStack
import vct.col.ref.Ref

import scala.collection.mutable

case class DisallowedConstAssignment(target: Node[_]) extends UserError {
  override def code: String = "disallowedConstAssignment"
  override def text: String =
    target.o.messageInContext("Cannot assign to constant target.")
}

case class DisallowedQualifiedType(target: Node[_]) extends UserError {
  override def code: String = "disallowedQualifiedType"
  override def text: String =
    target.o.messageInContext("This qualified type is not allowed.")
}

case class DisallowedQualifiedMethodCoercion(calledOrigin: Origin) extends UserError {
  override def code: String = "disallowedQualifiedMethodCoercion"
  override def text: String =
    calledOrigin.messageInContext("The coercion of args with qualifiers for this method call is not allowed.")
}

case class DisallowedQualifiedCoercion(calledOrigin: Origin, source: Type[_], target: Type [_]) extends UserError {
  override def code: String = "disallowedQualifiedCoercion"
  override def text: String =
    calledOrigin.messageInContext(s"The coercion of $source to $target is not allowed.")
}

case object TypeQualifierCoercion extends RewriterBuilder {
  override def key: String = "TypeQualifierCoercion"
  override def desc: String = "Removes qualifiers from types."

  def getUniqueMap[G](t: TClassUnique[G]): Map[InstanceField[G], BigInt] = {
    t.uniqueMap.map {case (ref, unique) => (ref.decl, unique)}.toMap
  }


}

case class TypeQualifierCoercion[Pre <: Generation]()
  extends CoercingRewriter[Pre] {

  val uniqueClasses: mutable.Map[(Class[Pre], Map[InstanceField[Pre], BigInt]), Class[Post]] = mutable.Map()
  val uniqueField: mutable.Map[(InstanceField[Pre], Map[InstanceField[Pre], BigInt]), InstanceField[Post]] = mutable.Map()

  def createUniqueClassCopy(original: Class[Pre], pointerInstanceFields: Map[InstanceField[Pre], BigInt]): Class[Post] = {
      globalDeclarations.declare({
        classDeclarations.scope({
          val decls = classDeclarations.collect { original.decls.foreach { d =>
            classDeclarations.declare[InstanceField[Post]] { d match {
              case field: InstanceField[Pre] if pointerInstanceFields.contains(field) =>
                val unique = pointerInstanceFields(field)
                val it = field.t match {
                  case TPointer(it) => it
                  case _ => ??? // Not allowed
                }
                val (info, innerResType) = getUnqualified(it)
                if (info.const) ??? // Not allowed
                val resType = TPointerUnique(innerResType, unique)
                val resField = field.rewrite(t = resType)
                uniqueField((field, pointerInstanceFields)) = resField
                resField
              case field: InstanceField[Pre] =>
                val resField = field.rewrite()
                uniqueField((field, pointerInstanceFields)) = resField
                resField
              case _ => ??? // Not allowed
            }}
          }}._1
          original.rewrite(decls=decls)
        })
      })
  }



  override def applyCoercion(e: => Expr[Post], coercion: Coercion[Pre])(
    implicit o: Origin
  ): Expr[Post] = {
    coercion match {
      case CoerceFromConst(_) =>
      case CoerceToConst(_) =>
      case CoerceToUnique(_, _) =>
      case CoerceFromUnique(_, _) =>
      case CoerceBetweenUnique(_, _, _) =>
      case CoerceToUniquePointer(_, t) => return UniquePointerCoercion(e, dispatch(t))
      case CoerceFromUniquePointer(_, t) => return UniquePointerCoercion(e, dispatch(t))
      case CoerceBetweenUniquePointer(_, t) => return UniquePointerCoercion(e, dispatch(t))
      case CoerceBetweenUniqueClass(_, t) =>  return UniquePointerCoercion(e, dispatch(t))
      case _ =>
    }
    e
  }

  override def postCoerce(t: Type[Pre]): Type[Post] =
    t match {
      case TConst(t) => dispatch(t)
      case TUnique(_, _) => throw DisallowedQualifiedType(t)
      case TPointer(it) => makePointer(it)
      case tu: TClassUnique[Pre] =>
        val map = TypeQualifierCoercion.getUniqueMap(tu)
        val c = tu.cls.decl
        val uniqueClass = uniqueClasses.getOrElseUpdate((c,map), createUniqueClassCopy(c, map))
        TClass[Post](uniqueClass.ref, Seq())
      case other => other.rewriteDefault()
    }

  override def postCoerce(e: Expr[Pre]): Expr[Post] = {
    implicit val o: Origin = e.o
    e match {
      case PreAssignExpression(target, _) if target.t.isInstanceOf[TConst[Pre]] => throw DisallowedConstAssignment(target)
      case PostAssignExpression(target, _) if target.t.isInstanceOf[TConst[Pre]] => throw DisallowedConstAssignment(target)
      case npa @ NewPointerArray(t, size, _) =>
        val (info, newT) = getUnqualified(t)
        if(info.const) NewConstPointerArray(newT, dispatch(size))(npa.blame)
        else NewPointerArray(newT, dispatch(size), info.unique)(npa.blame)
      case newO@NewObjectUnique(cls, _) =>
        val map = TypeQualifierCoercion.getUniqueMap(newO.t.asInstanceOf[TClassUnique[Pre]])
        val c = cls.decl
        val uniqueClass = uniqueClasses.getOrElseUpdate((c,map), createUniqueClassCopy(c, map))
        NewObject[Post](uniqueClass.ref)
      case d@Deref(obj, ref)  =>
        obj match {
          // Always has an CoerceClassAnyClassCoercion
          case ApplyCoercion(e, _) if e.t.isInstanceOf[TClassUnique[Pre]] =>
            val source = e.t.asInstanceOf[TClassUnique[Pre]]
            val map = TypeQualifierCoercion.getUniqueMap(source)
            if (!uniqueField.contains(ref.decl, map)) createUniqueClassCopy(source.cls.decl, map)
            d.rewrite(ref = uniqueField(ref.decl, map).ref)
          case _ => d.rewriteDefault()
        }

      case other => other.rewriteDefault()
    }
  }

  override def postCoerce(s: Statement[Pre]): Statement[Post] =
    s match {
      case Assign(target, _) if getUnqualified(target.t)._1.const => throw DisallowedConstAssignment(target)
      case a@AssignInitial(target, value) => Assign(dispatch(target), dispatch(value))(a.blame)(a.o)
      case other => other.rewriteDefault()
    }

  case class InnerInfo(){
    var unique: Option[BigInt] = None
    var const: Boolean = false
  }

  def getUnqualified(t: Type[Pre], info: InnerInfo = InnerInfo()): (InnerInfo, Type[Post]) = t match {
    case TConst(_) | TUnique(_, _) if info.const || info.unique.isDefined =>
      throw DisallowedQualifiedType(t)
    case TConst(it) =>
      info.const = true
      getUnqualified(it, info)
    case TUnique(it, id) =>
      info.unique = Some(id)
      getUnqualified(it, info)
    case _ => (info, dispatch(t))
  }

  def makePointer(t: Type[Pre]): Type[Post] = {
    implicit val o: Origin = t.o
    val (info, resType) = getUnqualified(t)
    if(info.const) TConstPointer(resType)
    else if (info.unique.isDefined) TPointerUnique(resType, info.unique.get)
    else TPointer(resType)
  }
}

case object MakeUniqueMethodCopies extends RewriterBuilder {
  override def key: String = "MakeUniqueMethodCopies"
  override def desc: String =
    "Makes copies of called function that are specialized for unique pointers."
}

case class MakeUniqueMethodCopies[Pre <: Generation]()
    extends Rewriter[Pre] {
  val methodCopyTypes: ScopedStack[Map[Type[Pre], Type[Post]]] = ScopedStack()

  val seenClassConversions: ScopedStack[mutable.Set[(Type[Pre], Type[Pre])]] = ScopedStack()
  val seenClasses: ScopedStack[mutable.Set[Type[Pre]]] = ScopedStack()

  val abstractFunction: mutable.Map[(Function[Pre], Map[Type[Pre], Type[Post]]), Function[Post]] = mutable.Map()
  val abstractProcedure: mutable.Map[(Procedure[Pre], Map[Type[Pre], Type[Post]]), Procedure[Post]] = mutable.Map()
  def getCopyType(t: Type[Pre]): Option[Type[Post]] = methodCopyTypes.topOption.flatMap(m => m.get(t))

  override def dispatch(t: Type[Pre]): Type[Post] = getCopyType(t).getOrElse(t.rewriteDefault())

  case class CoercedArg(originalParamT: Type[Pre], givenArgT: Type[Pre])

  def getPointers(t: Type[Pre]): Seq[PointerType[Pre]] = {
    def getPointersRec(n: Node[Pre]): Seq[PointerType[Pre]] = n match {
      case t: PointerType[Pre] => Seq(t)
      case t@TClassUnique(cls, _) =>
        val m = TypeQualifierCoercion.getUniqueMap(t)
        cls.decl.decls.flatMap {
          case field: InstanceField[Pre] if m.contains(field) =>
            // Field which are in the unique map should be pointers
            val element = field.t.asPointer.get.element
            // But also consider the element type of the pointer
            Seq(TPointerUnique(element, m(field))) ++ getPointers(element)
          case field: InstanceField[Pre] => getPointers(field.t)
          case _: JavaClassDeclaration[Pre] | _: PVLClassDeclaration[Pre] => ???
          case _ => Seq()
        }
      case tc@TClass(cls, Seq()) =>
        if(seenClasses.top.contains(tc)) return Seq()
        seenClasses.top.add(tc)

        cls.decl.decls.flatMap {
          case field: InstanceField[Pre] => getPointers(field.t)
          case _: JavaClassDeclaration[Pre] | _: PVLClassDeclaration[Pre] => ???
          case _ => Seq()
        }
      // Do not support type args yet. We should instantiate them or something?
      case TClass(cls, typeArgs) => ???
      case _ => Seq()
    }
    // Just go over all subnodes of the type. Only TClass(Unique) is a special instance, since
    // flatCollect does not visit references (and TClass contains a reference towards the class)
    t.flatCollect(getPointersRec(_))
  }

  /* A class/struct can create pointer coercions.
   * Given a class coercion, we capture all the pointer coercions which are made.
   * We also capture all pointer types, which are not coerced in any way.
   * E.g. struct v { int* xs, unique<2> int* ys} can get get turned into
   * struct v { unique<1> int* xs, unique<2> int* ys}, then not only struct v is coerced, but also int* -> unique<1> int*
   * is a coercion we need to capture. It is also useful to capture that unique<2> int* is not coerced.
  */
  def getPointerCoercionFromClassCoercion(originalParam: Type[Pre], givenT: Type[Pre]):
  (Seq[CoercedArg], Seq[PointerType[Pre]]) =
    (originalParam, givenT) match {
      case (TClass(clsParam, Seq()), TClass(clsGiven, Seq())) =>
        if(clsParam.decl.decls.length != clsGiven.decl.decls.length) ???
        clsParam.decl.decls.zip(clsGiven.decl.decls).map{
          case (fieldP, fieldG) => ???
        }
        ???
      case (originalT@TClassUnique(inner, _), TClass(cls, typeArgs)) =>
        // No type args
        if(!(typeArgs.isEmpty && inner == cls)) ???
        val m = TypeQualifierCoercion.getUniqueMap(originalT)

        val (resCoercions, resPointers) = m.map{case (field, u) =>
          field.t match {
            case t: PointerType[Pre] =>
              // This pointer gets coerced. But the element type of the pointer could contain further
              // pointers, so we need to check that.
              (CoercedArg(TPointerUnique(t.element, u), t), getPointers(t.element))
            case _ => ???
          }
        }.foldLeft( (Seq[CoercedArg](), Seq[PointerType[Pre]]()) ){
          case ((resCoercions, resPointers), (coerce, pointers)) => (resCoercions :+ coerce, resPointers ++ pointers)
        }

        val resPointers2 = cls.decl.decls.flatMap {
          // All fields which were not in the map, are not coerced, but could still contain pointers
          case field: InstanceField[Pre] if !m.contains(field) =>
            getPointers(field.t)
          case _ => Seq()
        }
        (resCoercions, resPointers++resPointers2)
      case (TClass(cls, typeArgs), givenT@TClassUnique(inner, _)) =>
        // No type args
        if(!(typeArgs.isEmpty && inner == cls)) ???
        val m = TypeQualifierCoercion.getUniqueMap(givenT)

        val (resCoercions, resPointers) = m.map{case (field, u) =>
          field.t match {
            case t: PointerType[Pre] =>
              // Any fields which are in the resulting unique map, are pointers that are coerced.
              // And again we check the element type of the pointer, since that could contain pointers.
              (CoercedArg(t, TPointerUnique(t.element, u)), getPointers(t.element))
            case _ => ???
          }
        }.foldLeft( (Seq[CoercedArg](), Seq[PointerType[Pre]]()) ){
          case ((resCoercions, resPointers), (coerce, pointers)) => (resCoercions :+ coerce, resPointers ++ pointers)
        }

        // Again, anything not in the unique map is not coerced. So check those fields for pointers
        val resPointers2 = cls.decl.decls.flatMap {
          case field: InstanceField[Pre] if !m.contains(field) =>
            getPointers(field.t)
          case _ => Seq()
        }
        (resCoercions, resPointers++resPointers2)

      case (originalParam@TClassUnique(innerO, _), givenT@TClassUnique(innerG, _)) =>
        // No type args
        if(!(innerO == innerG)) ???
        val originalM = TypeQualifierCoercion.getUniqueMap(originalParam)
        val givenM = TypeQualifierCoercion.getUniqueMap(givenT)

        /* The most interesting case! We have two unique map
         * 1)If the unique maps contain the same field:
         * a) the unique numbers are the same, there is no coercion
         * b) the unique numbers are differnet, there is a coercion
         * 2) For each field which isn't contained in the other map:
         *    There is a coercion!
         * 3) Each field which is not contained in either map:
         *    There is no coercion! But still needs to be checked for pointers
         */


        val (resCoercions1, resPointers1) =  originalM.map { case (field, uO) =>
          field.t match {
            case t: PointerType[Pre] =>
              val coercion = givenM.get(field) match {
                case Some(uG) if uO == uG => Seq()
                case Some(uG) => Seq(CoercedArg(TPointerUnique(t.element, uO), TPointerUnique(t.element, uG)))
                case None => Seq(CoercedArg(TPointerUnique(t.element, uO), TPointer(t.element)))
              }
              (coercion, getPointers(t))
            case _ => ???
          }
        }.foldLeft( (Seq[CoercedArg](), Seq[PointerType[Pre]]()) ){
          case ((resCoercions, resPointers), (coerce, pointers)) => (resCoercions ++ coerce, resPointers ++ pointers)
        }

        val (resCoercions2, resPointers2) =  givenM.map { case (field, uG) =>
          field.t match {
            case t: PointerType[Pre] =>
              givenM.get(field) match {
                case None => (Seq(CoercedArg(TPointer(t.element), TPointerUnique(t.element, uG))), getPointers(t))
                // If it was both unique in given and original, it was already covered above
                case Some(_) => (Seq(), Seq())
              }
            case _ => ???
          }
        }.foldLeft( (Seq[CoercedArg](), Seq[PointerType[Pre]]()) ){
          case ((resCoercions, resPointers), (coerce, pointers)) => (resCoercions ++ coerce, resPointers ++ pointers)
        }
        // And check all the fields which were not in any unique maps
        val resPointers3 = innerO.decl.decls.flatMap {
          case field: InstanceField[Pre] if !originalM.contains(field) && !givenM.contains(field) =>
            getPointers(field.t)
          case _ => Seq()
        }
        (resCoercions1 ++ resCoercions2, resPointers1++resPointers2++resPointers3)
      // Should only work for class types
      case _ => ???
  }

  def checkCoercion(original: Type[Pre], result: Type[Pre], coercionMap: Map[Type[Pre], Type[Pre]],
                    calledOrigin: Origin): Unit = {
    val rec = (l, r) => checkCoercion(l, r, coercionMap, calledOrigin)
    (original, result) match {
    case (l: PointerType[Pre], r: PointerType[Pre]) =>
      if(coercionMap.contains(l) && coercionMap(l) != r) throw DisallowedQualifiedMethodCoercion(calledOrigin)
      rec(l.element, r.element)
  }
  }

  def removeCoercions(args: Seq[Expr[Pre]]): Seq[Expr[Post]] = args.map({
    case UniquePointerCoercion(e, _) => dispatch(e)
    case e => dispatch(e)
  })

  // Instead of the regular procedure, we create an abstract procedure, which is the same, but with different types
  def createAbstractProcedureCopy(original: Procedure[Pre], typeCoerced: Map[Type[Pre], Type[Post]]): Procedure[Post] = {
    methodCopyTypes.having(typeCoerced) {
      globalDeclarations.declare({
        // Subtle, need to create variable scope, otherwise variables are already 'succeeded' in different copies.
        variables.scope({
          original.rewrite(body = None)
        })
      })
    }
  }

  // Same for functions
  def createFunctionCopy(original: Function[Pre], typeCoerced: Map[Type[Pre], Type[Post]]): Function[Post] = {
    methodCopyTypes.having(typeCoerced) {
      globalDeclarations.declare({
        // Subtle, need to create variable scope, otherwise variables are already 'succeeded' in different copies.
        variables.scope({
          // We do copy body, otherwise functions could be different.
          original.rewrite()
        })
      })
    }
  }

  def combine[A, B](a: (Seq[A], Seq[B]), b: (Seq[A], Seq[B])): (Seq[A], Seq[B]) = {
    (a._1 ++ b._1, a._2 ++ b._2)
  }

  def addFirst[A, B](a: A, b: (Seq[A], Seq[B])): (Seq[A], Seq[B]) = {
    (b._1 :+ a, b._2)
  }

  def getCoercionPerParam(paramT: Type[Pre], argT: Type[Pre]): (Seq[CoercedArg], Seq[PointerType[Pre]]) = (paramT, argT) match {
    // Unpack pointers first, since the outer pointer is structurally the same
    case (p@TPointer(paramT), a@TPointer(argT)) =>
      addFirst(CoercedArg(p, a), getCoercionPerParam(paramT, argT))
    case (p@TPointerUnique(paramT, paramU), a@TPointerUnique(argT, argU)) if paramU == argU =>
      addFirst(CoercedArg(p, a), getCoercionPerParam(paramT, argT))
    // Now we should have two different pointers
    case (p: PointerType[Pre], a: PointerType[Pre]) =>
      (Seq(CoercedArg(p, a)), getPointers(p.element))
    // Other case can only be if it was a class
    case (p, a) =>
      // Should be class type
      if(p.asClass.isEmpty || a.asClass.isEmpty) ???
      // Classes can refer to itself. So we do not want to keep repeating ourselves
      if(seenClassConversions.top.contains(p, a)) return (Seq(), Seq())
      seenClassConversions.top.add((p, a))

      val pClass = p.asClass.get.cls.decl.decls
      val aClass = a.asClass.get.cls.decl.decls
      if(pClass.size != aClass.size) ???

      pClass.zip(aClass).map {
        case (pf: InstanceField[Pre], af: InstanceField[Pre]) if pf.t == af.t =>
          if(!(pf.flags == af.flags)) ???
          (Seq(), getPointers(pf.t))
        case (pf: InstanceField[Pre], af: InstanceField[Pre]) =>
          if(!(pf.flags.isEmpty && af.flags.isEmpty)) ???
          // Should be different types. That can only be because of pointers or classes!
          getCoercionPerParam(pf.t, af.t)
        // Not instance field, so we do not care
        case _ => (Seq(), Seq())
      }.foldLeft[(Seq[CoercedArg],Seq[PointerType[Pre]])](
        // We still need to add our own coercion
        (Seq(CoercedArg(p, a)), Seq()) ){
        case (r, m) => combine(r, m)
      }
  }

  def getCoercionAndPointers(params: Seq[Variable[Pre]], args: Seq[Expr[Pre]]): (Seq[CoercedArg], Seq[PointerType[Pre]]) = {
    if(params.length != args.length) ???
    params.map(_.t).zip(args).map {
      case (originalT, UniquePointerCoercion(e, _)) =>
        getCoercionPerParam(originalT, e.t)
      case (originalT, _) =>
        // No coercions, just get pointers
        (Seq(), getPointers(originalT))
    }.foldLeft[(Seq[CoercedArg],Seq[PointerType[Pre]])]( (Seq(), Seq()) ){
      case (r, m) => combine(r, m)
    }
  }

  def createMapAndCheck(args: Seq[(Seq[Variable[Pre]], Seq[Expr[Pre]])], returnCoercion: Option[CoercedArg],
                        returnType: Option[Type[Pre]], calledOrigin: Origin): Map[Type[Pre], Type[Post]] = {

    if(returnCoercion.isEmpty && args.flatMap(_._2).collectFirst{ case _: UniquePointerCoercion[Pre] => () }.isEmpty){
      // No coercions, return empty map
      return Map()
    }
    seenClassConversions.push(mutable.Set())
    seenClasses.push(mutable.Set())


    var (coercions, nonCoercedPointers) = args.foldLeft(Seq[CoercedArg](), Seq[PointerType[Pre]]()){
      case (res, (fArgs, invArgs)) =>  combine(res, getCoercionAndPointers(fArgs, invArgs))
    }
    coercions = coercions ++ returnCoercion
    // If we care about the return type it is given as argument
    if(returnType.nonEmpty && returnCoercion.isEmpty){
      nonCoercedPointers = nonCoercedPointers ++ getPointers(returnType.get)
    }

    val m = coercions.groupMapReduce(_.originalParamT)(c => dispatch(c.givenArgT))(
      // For any duplicates, we exit if they do not resolve to the same type
      (l, r) => if(l == r) l else throw DisallowedQualifiedMethodCoercion(calledOrigin)
    )
    // If any nonCoercedPointer is in the coercion set, invocation is wrong
    if(m.keySet.intersect(nonCoercedPointers.toSet).nonEmpty)
      throw DisallowedQualifiedMethodCoercion(calledOrigin)

    seenClassConversions.pop()
    seenClasses.pop()
    m
  }



  def rewriteProcedureInvocation(inv: ProcedureInvocation[Pre], returnCoercion: Option[CoercedArg], anyReturnPointer: Boolean = false): ProcedureInvocation[Post] = {
    val f = inv.ref.decl
    val m = createMapAndCheck(
      Seq(
        (f.args, inv.args),
        (f.outArgs, inv.outArgs),
        (inv.givenMap.map(_._1.decl), inv.givenMap.map(_._2)),
        (inv.yields.map(_._2.decl), inv.yields.map(_._1))
      ),
      returnCoercion,
      if(anyReturnPointer) None else Some(f.returnType),
      inv.o
    )
    // No coercions, so do nothing
    if(m.isEmpty){
      return inv.rewriteDefault()
    }

    val newProc: Procedure[Post] =
      abstractProcedure.getOrElseUpdate((f, m), createAbstractProcedureCopy(f, m))

    val newArgs = removeCoercions(inv.args)
    val newOutArgs = removeCoercions(inv.outArgs)
    val givenArgs = removeCoercions(inv.givenMap.map(_._2))
    val newGivenMap = newProc.contract.givenArgs.map(_.ref).zip(givenArgs)
    val yieldsArgs = removeCoercions(inv.yields.map(_._1))
    val newYieldsMap = yieldsArgs.zip( newProc.contract.yieldsArgs.map(_.ref) )
    inv.rewrite(ref = newProc.ref, args=newArgs, outArgs=newOutArgs, givenMap= newGivenMap, yields=newYieldsMap)
  }

  def rewriteFunctionInvocation(inv: FunctionInvocation[Pre], returnCoercion: Option[CoercedArg], anyReturnPointer: Boolean = false): FunctionInvocation[Post] = {
    val f = inv.ref.decl
    val m = createMapAndCheck(
      Seq(
        (f.args, inv.args),
        (inv.givenMap.map(_._1.decl), inv.givenMap.map(_._2)),
        (inv.yields.map(_._2.decl), inv.yields.map(_._1))
      ),
      returnCoercion,
      if(anyReturnPointer) None else Some(f.returnType),
      inv.o
    )
    // No coercions, but since it is a function invocation, this could take place whilst we are making a copy of a method
    // or function.
    if(m.isEmpty){
      if(methodCopyTypes.nonEmpty){
        val map = methodCopyTypes.top
        // So we are already coercing. Let's see if we need to change anything.
        if(f.args.exists(v => map.contains(v.t)) || map.contains(f.returnType) ) {
          // So yes, we just use the same map we were already using
          val newFunc: Function[Post] = abstractFunction.getOrElseUpdate((f, map), createFunctionCopy(f, map))
          val newArgs = removeCoercions(inv.args)
          return inv.rewrite(ref = newFunc.ref, args=newArgs)
        }
      }
      // Otherwise business as usual
      return inv.rewriteDefault()
    }

    // Coercing a function call, whilst we are already coercing seems quite complicated.
    // So let's not do that for now.
    if(methodCopyTypes.nonEmpty) throw DisallowedQualifiedMethodCoercion(inv.o)

    val newFunc: Function[Post] = abstractFunction.getOrElseUpdate((f, m), createFunctionCopy(f, m))
    val newArgs = removeCoercions(inv.args)
    inv.rewrite(ref = newFunc.ref, args=newArgs)
  }

  // For AmbiguousSubscript / DerefPointer we do not care about how the return type is coerced
  // so if we encounter invocations we communicate that.
  def rewriteAnyPointerReturn(e: Expr[Pre]): Expr[Post] = e match {
    case inv: ProcedureInvocation[Pre] =>
      rewriteProcedureInvocation(inv, None, anyReturnPointer=true)
    case inv: FunctionInvocation[Pre] =>
      rewriteFunctionInvocation(inv, None, anyReturnPointer=true)
    case e => dispatch(e)
  }

  // If the coerce contains an invocation, maybe it is valid, otherwise not
  def rewriteCoerce(e: Expr[Pre], target: Type[Pre]): Expr[Post] = e match {
    case inv: ProcedureInvocation[Pre] =>
      // This seems the other way around, but the invocation has the return type of the function/method (original type)
      // And the target type is what we want to coerce towards.
      val returnCoercion = Some(CoercedArg(inv.t, target))
      rewriteProcedureInvocation(inv, returnCoercion)
    case inv: FunctionInvocation[Pre] =>
      val returnCoercion = Some(CoercedArg(inv.t, target))
          rewriteFunctionInvocation(inv, returnCoercion)
    case am@AmbiguousMinus(pointer, _) => am.rewrite(left=rewriteCoerce(pointer, target))
    case am@AmbiguousPlus(pointer, _) => am.rewrite(left=rewriteCoerce(pointer, target))
    case e => throw DisallowedQualifiedCoercion(e.o, e.t, target)
  }

  def getNewField(originalT: Type[Pre], originalField: InstanceField[Pre]): Option[InstanceField[Post]] =
    methodCopyTypes.topOption.flatMap(
      m => m.get(originalT).map(t => {
        val idx = originalT.asClass.get.cls.decl.decls.indexOf(originalField)
        if(idx == -1) ???
        t.asClass.get.cls.decl.decls(idx).asInstanceOf[InstanceField[Post]]
      })
    )


  override def dispatch(e: Location[Pre]): Location[Post] = e match {
    case loc: FieldLocation[Pre] =>
      val newField = getNewField(loc.obj.t, loc.field.decl)
      if(newField.isDefined){
        loc.rewrite(field=newField.get.ref)
      } else {
        loc.rewriteDefault()
      }
    case other => other.rewriteDefault()
  }

  override def dispatch(e: Expr[Pre]): Expr[Post] = {
    implicit val o: Origin = e.o
    e match {
      case inv: FunctionInvocation[Pre] =>
        rewriteFunctionInvocation(inv, None)
      case inv: ProcedureInvocation[Pre] =>
        rewriteProcedureInvocation(inv, None)
      // So this is awkward, but...
      // A lot of times we just coerced to 'pointer', as with subscripting. In this case we don't care if the return gets
      // coerced.
      case e@AmbiguousSubscript(p, _) => e.rewrite(collection=rewriteAnyPointerReturn(p))
      case d@Deref(obj, ref) =>
        val newField = getNewField(obj.t, ref.decl)
        if(newField.isDefined){
          d.rewrite(ref=newField.get.ref)
        } else {
          d.rewriteDefault()
        }
      case e@DerefPointer(p) =>
        e.rewrite(pointer=rewriteAnyPointerReturn(p))
      case e@FreePointer(p) => e.rewrite(pointer=rewriteAnyPointerReturn(p))
      case e@PointerBlockLength(p) => e.rewrite(pointer=rewriteAnyPointerReturn(p))
      case e@PointerLength(p) => e.rewrite(pointer=rewriteAnyPointerReturn(p))
      case e@PointerBlockOffset(p) => e.rewrite(pointer=rewriteAnyPointerReturn(p))
      case e@SharedMemSize(p) => e.rewrite(pointer=rewriteAnyPointerReturn(p))
      // We store the coercion for the return type
      case u: UniquePointerCoercion[Pre] => rewriteCoerce(u.e, u.t)
      case other => other.rewriteDefault()
    }
  }

  override def dispatch(s: Statement[Pre]): Statement[Post] = {
    implicit val o: Origin = s.o
    s match {
      case ev@Eval(e) =>
        ev.rewrite(rewriteAnyPointerReturn(e))
      case other => other.rewriteDefault()
    }
  }
}
