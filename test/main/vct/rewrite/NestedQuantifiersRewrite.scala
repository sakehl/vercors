package vct.rewrite

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import vct.col.ast._
import vct.col.rewrite.SimplifyNestedQuantifiers
import vct.col.origin._
import vct.col.rewrite.InitialGeneration
import vct.col.util.AstBuildHelpers._
import vct.helper.ColHelper


class NestedQuantifiersRewrite extends AnyFlatSpec with Matchers {
  type G = InitialGeneration
  implicit val o: Origin = DiagnosticOrigin
  implicit val blame: Blame[VerificationFailure] = PanicBlame("")
  val xs_ = new Variable[G](TArray[G](TInt[G]()))
  val xs = Local[G](xs_.ref)

  val ns_ =
    for (n <- 0 until 10)
      yield (new Variable[G](TInt[G]())(o.where(name=s"ns$n")))
  val ns: Seq[Expr[G]] =
    for (n <- ns_)
      yield (Local[G](n.ref))
  val as_ =
    for (a <- 0 until 10)
      yield (new Variable[G](TInt[G]())(o.where(name=s"as$a")))
  val as: Seq[Expr[G]] =
    for (n <- as_)
      yield (Local[G](n.ref))


  def c(i: Int) = const[G](i)
  def block(e: Expr[G], c: Option[Expr[G]]=None) =
    Scope[G]((xs_ +: ns_) ++ as_,
      Inhale[G](if(c.isEmpty) e else And(c.get, e))
    )

  def notRewrite(bounds: Seq[(Expr[G], Expr[G])], as: Seq[Expr[G]]
                    ){
    val vars = Seq.fill(bounds.size)(TInt[G])
    val before = block(foralls[G](
      vars,
      {
        case locals =>
          val domain = locals.lazyZip(bounds).map(
            {case (x, (min, max)) => min <= x && x < max }
          ) . reduceLeft[Expr[G]]({case (l, r) => l && r})
          val idx = locals.lazyZip(as).map(
            {case (x, a) => a*x }
          ) . reduceLeft[Expr[G]]({case (l, r) => l + r})
          domain ==> ArraySubscript(xs, idx)(blame)
      },
    ))

    val rw = SimplifyNestedQuantifiers[G]()
    ColHelper.assertEquals(rw.labelDecls.scope { rw.dispatch(before) }, before)
  }

  def correctNumbers(bounds: Seq[(Expr[G], Expr[G])],
                     index: Seq[Local[G]] => Expr[G],
                     res_domain: Local[G] => Expr[G],
                     extra_cond: Option[Seq[Local[G]] => Expr[G]] = None,
                     c: Option[Expr[G]]=None
                    ){
    val vars = Seq.fill(bounds.size)(TInt[G])

    val before = block(foralls[G](vars,
      {
        case locals =>
          var domain = locals.lazyZip(bounds).map(
            {case (x, (min, max)) => min <= x && x < max }
          ) . reduceLeft[Expr[G]]({case (l, r) => l && r})
          domain = extra_cond.map(c => domain && c(locals)).getOrElse(domain)
          domain ==> ArraySubscript(xs, index(locals))(blame)
      },
    ), c)

    val after = block(foralls[G](
      Seq(TInt()),
      {
        case Seq(xyz) =>
          res_domain(xyz) ==> ArraySubscript(xs, xyz)(blame)
      },
      {
        case Seq(xyz) => Seq(Seq(ArraySubscript(xs, xyz)(blame)))
      }
    ), c)

    val rw = SimplifyNestedQuantifiers[G]()
    ColHelper.assertEquals(rw.labelDecls.scope { rw.dispatch(before) }, after)
  }

  it should "Rewrite x=0..3, y=0..5, z=0..2  ..x+3*y+15*z.. to xyz=0..15*2 ..xyz.." in correctNumbers(
    Seq((c(0), c(3)), (c(0), c(5)), (c(0), c(2))),
    {case Seq(x, y, z) => c(1)*x + c(3)*y + c(15)*z},
    (xyz: Local[G]) => c(0) <= xyz && xyz < c(15)*c(2)
  )

  it should "Not rewrite x=0..3, y=0..5, z=0..2  ..x+4*y+15*z.." in notRewrite(
    Seq((c(0), c(3)), (c(0), c(5)), (c(0), c(2))),
    Seq(c(1), c(4), c(15))
  )

  it should "Rewrite x=0..3, y=0..4, z=0..2  ..x+3*y+15*z.. to xyz=0..15*2 (xyz%15/3 < 4) ==> ..xyz.." in correctNumbers(
    Seq((c(0), c(3)), (c(0), c(4)), (c(0), c(2))),
    {case Seq(x, y, z) => c(1)*x + c(3)*y + c(15)*z},
    (xyz: Local[G]) => ((xyz % c(15)) / c(3) < c(4)) && (c(0) <= xyz && xyz < c(15)*c(2))
  )

  it should "Rewrite x=0..2, y=0..5, z=0..2  ..x+3*y+15*z.. to xyz=0..15*2 (xyz%15%3<2) ==> ..xyz.." in correctNumbers(
    Seq((c(0), c(2)), (c(0), c(5)), (c(0), c(2))),
    {case Seq(x, y, z) => c(1)*x + c(3)*y + c(15)*z},
    (xyz: Local[G]) => ((xyz % c(15)) % c(3) < c(2)) && (c(0) <= xyz && xyz < c(15)*c(2))
  )

  it should "Rewrite x=0..3, y=0..5, z=0..2  ..-x+-3*y+-15*z.. to -xyz=0..-15*2 ..xyz.." in correctNumbers(
    Seq((c(0), c(3)), (c(0), c(5)), (c(0), c(2))),
    {case Seq(x, y, z) => c(-1)*x + c(-3)*y + c(-15)*z},
    (xyz: Local[G]) => c(-15)*c(2) < xyz && xyz <= c(0)
  )

  it should "Rewrite x=0..n  ..n-x-1.. to -(x'-(-1+n))=0..n ..x'.." in correctNumbers(
    Seq((c(0), ns(0))),
    {case Seq(x) => ns(0) - x + c(-1)},
    (xyz: Local[G]) => c(-1) * ns(0) < (xyz - (c(-1) + ns(0))) && (xyz - (c(-1) + ns(0))) <= c(0)
  )

  it should "Not rewrite x=0..nx, y=0..ny, z=0..nz  ..a0*x+a0*n0*y+a0*n0*n1*z.." in notRewrite(
    Seq((c(0), ns(0)), (c(0), ns(1)), (c(0), ns(2))),
    Seq(as(0), as(0)*ns(1), as(0)*ns(1)*ns(2))
  )

  it should "Rewrite x=0..nx, y=0..ny, z=0..nz  ..a0*x + a0*n0*y + a0*n0*n1*z.. given that a0!=0" in correctNumbers(
    Seq((c(0), ns(0)), (c(0), ns(1)), (c(0), ns(2))),
    {case Seq(x,y,z) => as(0)*x + as(0)*ns(0)*y + as(0)*ns(0)*ns(1)*z},
    (xyz: Local[G]) => (c(0) <= xyz && xyz < as(0)*ns(0)*ns(1)*ns(2)) &&
      xyz % (as(0)*ns(0)*ns(1)) % (as(0)*ns(0)) % as(0) === c(0),
    None,
    Some(as(0) > c(0))
  )


}
