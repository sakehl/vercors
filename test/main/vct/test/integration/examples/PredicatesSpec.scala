package vct.test.integration.examples

import vct.test.integration.helper.VercorsSpec

class PredicatesSpec extends VercorsSpec {
  vercors should verify using silicon example
    "concepts/predicates/IntegerList.java"
  vercors should verify using silicon example
    "concepts/predicates/minmax-list.pvl"
  vercors should error withCode "cyclicInline" example
    "concepts/predicates/MutuallyRecursiveInlinePredicates.pvl"
  vercors should error withCode "cyclicInline" example
    "concepts/predicates/RecursiveInlinePredicate.pvl"
  vercors should verify using silicon example
    "concepts/predicates/ScaleInlinePredicate.pvl"
  // https://github.com/utwente-fmt/vercors/discussions/842
  // vercors should verify using silicon example "concepts/predicates/TreeRecursive.java"

  /*
  vercors should verify using anyBackend in "using predicate (with scale) in a trigger" pvl """
resource x_perm(int x, int n, int[] data) =  (data != null && data.length == n && n>0 && x >= 0 && x < n) ** Perm(data[x], write);

  context (\forall* int i=0..n; {:x_perm(i, n, xs):});
  context (\forall* int i=0..n; {:[1\2]x_perm(i, n, ys):});
void main(int[] xs, int[] ys, int n){
  if(n>0){
    unfold x_perm(0, n, xs);
    unfold [1\2]x_perm(0, n, ys);
    xs[0] = ys[0];
    fold [1\2]x_perm(0, n, ys);
    fold x_perm(0, n, xs);
  }
}
  """
   */

  vercors should verify using silicon in
    "Opening a present predicate should succeed" pvl """
    class C {
      int x;
    }

    resource P(C c) = true;

    requires P(c);
    void m(C c) {
      unfold P(c);
    }
    """

  vercors should fail withCode "unfoldFailed" using silicon in
    "Opening a non-present predicate should fail" pvl """
    class C { int x; }

    resource P(C c) = true;

    void m(C c) {
      unfold P(c);
    }
    """

  vercors should fail withCode "unfoldFailed:false" using silicon in
    "Opening a non-present inline predicate should fail" pvl """
    inline resource P(int x, int y) = x == 1 && y == 2;

    requires x == 1;
    void m(int x) {
      unfold P(x, 0);
    }
    """

  vercors should verify using silicon in
    "Opening a present inline predicate should succeed" pvl """
    inline resource P(int x, int y) = x == 1 && y == 2;

    requires x == 1;
    void m(int x) {
      unfold P(x, 2);
    }
    """
}
