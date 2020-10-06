// -*- tab-width:4 ; indent-tabs-mode:nil -*-
//:: cases TutorialSyntaxJava
//:: suite TutorialExamples
//:: tools silicon
//:: verdict Pass

/*
    This file contains the Examples from the Chapter "Syntax" of the VerCors tutorial 
    (see https://github.com/utwente-fmt/vercors/wiki/Tutorial-Syntax ).
*/

class Examples {

    /// example 1
    
    //@ requires x>0;
    public int increment(int x);

    /*@
    requires x>0;
    requires y>0;
    @*/
    public int add1(int x, int y);
    
    
    /// example 2
    
    /*@
    requires x >= 0;
    requires y >= 0;
    ensures \result == x+y;
    ensures \result >= 0;
    @*/
    public int add2(int x, int y) {
        return x+y;
    }

    /*@
    requires xs != null;
    ensures \result != null;
    ensures \result.length == \old(xs.length)+1;
    ensures \result.length > 0;
    @*/
    public int[] append(int[] xs, int y) {
        return new int[xs.length+1];
    }


    /// example 3
    
    //@ requires a>0 && b>0;
    //@ ensures \result == a*b;
    public int mult3(int a, int b) {
        int res = 0;
        //@ loop_invariant res == i*a;
        //@ loop_invariant i <= b;
        for (int i=0; i<b; i++) {
            res = res + a;
        }
        return res;
    }


    /// example 4
    
    //@ ensures a==0 ? \result == 0 : \result >= 10;
    public int mult10(int a) {
        int res = a;
        if (res < 0) {
            res = 0-res;
        }
        //@ assert res >= 0;
        return 10*res;
    }


    /// example 5
    
    void example5() {
        int x = 2;
        //@ assume x == 1;
        int y = x + 3;
        //@ assert y == 4;
    }


    /// example 6
    
    /*@
    requires x>=0 && y>=0;
    ensures \result == (x<y ? 5*x : 5*y);
    @*/
    public int minTimes5(int x, int y) {
        //@ ghost int min = (x<y ? x : y);
        int res = 0;
        //@ loop_invariant i<=5;
        //@ loop_invariant res == i*min;
        //@ loop_invariant min<=x && min<=y && (min==x || min==y); 
        for (int i=0; i<5; i++) {
            if (x<y) {
                res = res + x;
            } else {
                res = res + y;
            }
        }
        /*@ 
        ghost if (x<y) {
            assert res == 5*x;
        } else {
            assert res == 5*y;
        }
        @*/
        return res;
    }


    /// example 7
    
    /*@
    requires x > 0;
    ensures \result == (cond ? x+y : x);
    ghost static int cond_add7(bool cond, int x, int y) {
        if (cond) {
            return x+y;
        } else {
            return x;
        }
    }
    @*/

    //@ requires val1 > 0 && val2>0 && z==val1+val2;
    void some_method7(int val1, int val2, int z) {
        //@ ghost int z2 = cond_add7(val2>0, val1, val2);
        //@ assert z == z2;
    }


    /// example 8
    
    /*@
    requires x > 0;
    static pure int cond_add8(bool cond, int x, int y) 
        = cond ? x+y : x;
    @*/
    

    /// example 9
    
    /*@
    requires x > 0;
    @*/
    static /*@ pure @*/ int cond_add9(boolean cond, int x, int y) {
        if (cond) {
            return x+y;
        } else {
            return x;
        }
    }


    /// example 10
    
    //@ requires a>0 && b>0;
    //@ ensures \result == a*b;
    public int mult2(int a, int b);
    // commented out body for the sake of speeding up the analysis:
    //{
    //  int res = 0;
    //  // loop_invariant res == i*a;
    //  // loop_invariant i <= b;
    //  for (int i=0; i<b; i++) {
    //    res += a;
    //  }
    //  return res;
    //}

    /*@ 
    requires a>0 && b>0;
    ensures \result == a+b;
    public pure int add10(int a, int b);
    @*/


    /// example 11
    
    //@ pure inline int min(int x, int y) = (x<y ? x : y);
    
    
    /// example 12
    
    /*@
    given int x;
    given int y2;
    yields int modified_x;
    requires x > 0;
    ensures modified_x > 0;
    @*/
    int some_method12(boolean real_arg) {
        int res = 0;
        /// ...
        //@ ghost modified_x = x + 1;
        /// ...
        return res;
    }

    void other_method12() {
        //@ ghost int some_ghost;
        int some_result = some_method12(true) /*@ with {y2=3; x=2;} then {some_ghost=modified_x;} @*/;
    }


    /// example 13

    /*@
        /// requirements to make assertion actually pass
        requires x==1 && y==5 && z!=null;
    @*/
    public void example13(int x, int y, Object z) {
        //@ assert (\let int abs_x = (x<0 ? -x : x); y==(z==null ? abs_x : 5*abs_x));
    }
    
    
    /// example 14
    
    //@ requires arr != null;
    //@ requires Perm(arr[*], read);
    //@ requires (\forall int i ; 0<=i && i<arr.length ; arr[i]>0);
    void foo14(int[] arr) {
        /// example of a quantifier using an interval
        //@ assert (\forall int i = 0 .. arr.length ; arr[i]>0);
    }
    
    
    /// example 15
    
    //@ requires arr != null;
    //@ requires Perm(arr[*], read);
    //@ requires (\exists int i ; 0<=i && i<arr.length ; arr[i]>0);
    void foo15(int[] arr);
    

    /// example 16
    
    //@ requires arr != null;
    //@ requires Perm(arr[*], read);
    //@ requires (\forall int i ; 0<=i && i<arr.length ; {: arr[i] :} > 0);
    void foo16(int[] arr);
    


}
