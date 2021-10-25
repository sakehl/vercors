//:: cases opencl_static_static_keyword
//:: tools silicon
//:: verdict Error

// Statically allocated shared memory in OpenCL, but static keyword not allowed
#include "opencl.h"
/*@
  requires get_local_size(0) == 32;
  context Perm(x[get_local_id(0)], write);
  ensures x[get_local_id(0)] == get_local_id(0) + 1;
@*/
__kernel void f(){
  static __local int x[32];
  int i = get_local_id(0);
  x[i] = i;
  x[i] = x[i] + 1;
}