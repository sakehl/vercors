//:: cases opencl_shared_nested_array
//:: tools silicon
//:: verdict Error

// Statically allocated shared memory in OpenCL, but cannot be an array array
// Lars van den Haak: I'm unsure if this is actually allowed in OpenCL, but we disallow it in VerCors for now
#include "opencl.h"
/*@
  requires get_local_size(0) == 32;
@*/
__kernel void f(){
  __local int x[32][2];
  int i = get_local_id(0);
}