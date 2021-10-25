//:: cases opencl_mixed_shared_memory
//:: tools silicon
//:: verdict Pass

// Mixed static and dynamically allocated shared memory in OpenCL
#include "opencl.h"
/*@
  requires shared_mem_size_1 == 32;
  requires shared_mem_size_2 == 32;
  requires get_local_size(0) == 32;
  context Perm(x[get_local_id(0)], write);
  context Perm(y[get_local_id(0)], write);
  context Perm(z[get_local_id(0)], write);
  ensures x[get_local_id(0)] == get_local_id(0) + get_group_id(0) + 2;
@*/
__kernel void f(local int* x, __local int* y){
  local int z[32];
  int i = get_local_id(0);
  x[i] = i;
  y[i] = get_group_id(0);
  z[i] = 2;
  x[i] = x[i] + y[i] + z[i];
}