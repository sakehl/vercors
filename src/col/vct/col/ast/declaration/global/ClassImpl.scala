package vct.col.ast.declaration.global

import vct.col.ast.{
  Class,
  ClassDeclaration,
  Declaration,
  Expr,
  InstanceField,
  TByReferenceClass,
  TClass,
  TVar,
  Type,
  Variable,
}
import vct.col.ast.util.Declarator
import vct.col.print._
import vct.col.util.AstBuildHelpers.tt

trait ClassImpl[G] extends Declarator[G] {
  this: Class[G] =>
  def typeArgs: Seq[Variable[G]]
  def decls: Seq[ClassDeclaration[G]]
  def supports: Seq[Type[G]]
  def intrinsicLockInvariant: Expr[G]

  def classType(typeArgs: Seq[Type[G]]): TClass[G]

  def transSupportArrowsHelper(
      seen: Set[TClass[G]]
  ): Seq[(TClass[G], TClass[G])] = {
    // TODO: Does this break things if we have a ByValueClass with supers?
    val t: TClass[G] = classType(
      typeArgs.map((v: Variable[G]) => TVar(v.ref[Variable[G]]))
    )
    if (seen.contains(t))
      Nil
    else
      supers.map(sup => (t, sup)) ++
        supers.flatMap(sup => sup.transSupportArrowsHelper(Set(t) ++ seen))
  }

  def transSupportArrows: Seq[(TClass[G], TClass[G])] =
    transSupportArrowsHelper(Set.empty)

  def supers: Seq[TClass[G]] = supports.map(_.asClass.get)

  def fields: Seq[InstanceField[G]] =
    decls.collect { case field: InstanceField[G] => field }

  override def declarations: Seq[Declaration[G]] = decls ++ typeArgs

  def layoutLockInvariant(implicit ctx: Ctx): Doc =
    Text("lock_invariant") <+> Nest(intrinsicLockInvariant.show) <> ";" <+/>
      Empty

  def layoutLock(implicit ctx: Ctx): Doc =
    Text("Lock") <+> "intrinsicLock$" <+> "=" <+> "new" <+>
      "ReentrantLock(true);" <+/> "Condition" <+> "condition$" <+> "=" <+>
      "intrinsicLock$" <> "." <> "newCondition()" <> ";"

  def layoutJava(implicit ctx: Ctx): Doc =
    (if (intrinsicLockInvariant == tt[G])
       Empty
     else
       Doc.spec(Show.lazily(layoutLockInvariant(_)))) <+/> Group(
      Text("class") <+> ctx.name(this) <>
        (if (typeArgs.nonEmpty)
           Text("<") <> Doc.args(typeArgs) <> ">"
         else
           Empty) <>
        (if (supports.isEmpty)
           // Inheritance still needs work anyway
           Text(" ") <> "extends" <+> "Thread"
         else
           Text(" ") <> "extends" <+> Doc.args(
             supports.map(supp => ctx.name(supp.asClass.get.cls)).map(Text)
           )) <+> "{"
    ) <>> Doc.stack2(layoutLock +: decls) <+/> "}"

  def layoutPvl(implicit ctx: Ctx): Doc =
    (if (intrinsicLockInvariant == tt[G])
       Empty
     else
       Doc.spec(Show.lazily(layoutLockInvariant(_)))) <+/> Group(
      Text("class") <+> ctx.name(this) <>
        (if (typeArgs.nonEmpty)
           Text("<") <> Doc.args(typeArgs) <> ">"
         else
           Empty) <>
        (if (supports.isEmpty)
           Empty
         else
           Text(" implements") <+> Doc.args(
             supports.map(supp => ctx.name(supp.asClass.get.cls)).map(Text)
           )) <+> "{"
    ) <>> Doc.stack2(decls) <+/> "}"

  override def layout(implicit ctx: Ctx): Doc =
    ctx.syntax match {
      case Ctx.Java => layoutJava
      case _ => layoutPvl
    }
}
