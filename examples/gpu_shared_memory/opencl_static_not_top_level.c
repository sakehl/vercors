//:: cases opencl_static_not_top_level
//:: tools silicon
//:: verdict Error

// Statically allocated shared memory in OpenCL, but can only declare at top level
#include "opencl.h"
/*@
  requires get_local_size(0) == 32;
@*/
__kernel void f(){
  if(1 == 1){
    __local int x[32];
    int i = get_local_id(0);
    x[i] = i;
    x[i] = x[i] + 1;
  }
}