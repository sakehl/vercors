//:: cases opencl_global_keyword_body
//:: tools silicon
//:: verdict Error

// Cannot declare something global in the body
#include "opencl.h"
/*@
  requires get_local_size(0) == 32;
@*/
__kernel void f(){
  __global int x[32];
  int i = get_local_id(0);
}