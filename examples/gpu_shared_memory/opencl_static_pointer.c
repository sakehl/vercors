//:: cases opencl_static_pointer
//:: tools silicon
//:: verdict Error

// Statically allocated shared memory in OpenCL, but pointer not allowed
#include "opencl.h"
/*@
  requires get_local_size(0) == 32;
  context Perm(x[get_local_id(0)], write);
  ensures x[get_local_id(0)] == get_local_id(0) + 1;
@*/
__kernel void f(){
  __local int* x;
  int i = get_local_id(0);
  x[i] = i;
  x[i] = x[i] + 1;
}