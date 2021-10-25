//:: cases opencl_shared_scalar
//:: tools silicon
//:: verdict Pass

// Statically allocated scalar shared memory scalar in OpenCL
#include "opencl.h"
/*@
  context \ltid == 0 ==> Perm(x, write);
  ensures \ltid == 0 ==> x == get_global_id(0) + 1;
@*/
__kernel void f(){
  __local int x;
  int i = get_local_id(0);
  if(i == 0){
     x = get_global_id(0) + 1;
  }
}