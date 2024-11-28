#include <stdint.h>
#include <stdbool.h>

int main() {
    int a[] = {5, 6, 7, 8};
    int b[] = {1, 2, 3, 4};
    intptr_t c = (intptr_t)&a[3];
    int *d = (int *)(c + 4);
    // UB! comparison d == b only guarantees that the address of d and b are equal but not that they have the same provenance
    // As far as the compiler is concerned it may assume that d == b is false even if it isn't at runtime
    if (d == b) {
        /*[/expect ptrPerm]*/
        //@ assert *d == 1;
        /*[/end]*/
        return 1;
    } else {
        return 0;
    }
}

// We can attempt some trickery to see if pointers are equal without knowing the provenance, but it won't work luckily
/*[/expect postFailed:false]*/
//@ requires (p == q) || (p != q);
//@ ensures \result == (p == q);
bool pointerEq(int *p, int *q) {
    return p == q;
}
/*[/end]*/
