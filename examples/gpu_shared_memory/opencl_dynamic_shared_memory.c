//:: cases opencl_dynamic_shared_memory
//:: tools silicon
//:: verdict Pass

// Dynamically allocated shared memory in OpenCL
#include "opencl.h"
/*@
  requires shared_mem_size_1 == get_local_size(0);
  context Perm(x[get_local_id(0)], write);
  ensures x[get_local_id(0)] == get_local_id(0) + 1;
@*/
__kernel void f(__local int* x){
  int i = get_local_id(0);
  x[i] = i;
  x[i] = x[i] + 1;
}