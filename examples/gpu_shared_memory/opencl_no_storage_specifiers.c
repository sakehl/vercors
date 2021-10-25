//:: cases opencl_no_storage_specifiers
//:: tools silicon
//:: verdict Error

// Pointer argument without storage specifier (local or global)
#include "opencl.h"
/*@
  context Perm(x[get_local_id(0)], write);
  ensures x[get_local_id(0)] == get_local_id(0) + 1;
@*/
__kernel void f(int* x){
  int i = get_local_id(0);
  x[i] = i;
  x[i] = x[i] + 1;
}