//:: cases opencl_static_extern_keyword
//:: tools silicon
//:: verdict Error

// Statically allocated shared memory in OpenCL, but not allowed extern keyword (is allowed in cuda)
#include "opencl.h"
/*@
  requires get_local_size(0) == 32;
  context Perm(x[get_local_id(0)], write);
  ensures x[get_local_id(0)] == get_local_id(0) + 1;
@*/
__kernel void f(){
  extern __local int x[];
  int i = get_local_id(0);
  x[i] = i;
  x[i] = x[i] + 1;
}