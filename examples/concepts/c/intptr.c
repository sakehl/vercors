#include <stdint.h>

int main() {
    int a[] = {5, 6, 7, 8};
    int b[] = {1, 2, 3, 4};
    intptr_t c = (intptr_t)&a[3];
    int *d = (int *)(c + 4);
    // This is (non-catastrophic) UB, gcc with -O1 returns 0, and with -O2 returns 1
    if (c + 4 == (intptr_t)&b[0]) {
        //@ assert *d == 1;
        return 1;
    } else {
        return 0;
    }
}
