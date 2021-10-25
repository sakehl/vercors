//:: cases opencl_dynamic_array
//:: tools silicon
//:: verdict Error

// Local Array argument is not allowed in kernel parameters
#include "opencl.h"
/*@
  context Perm(x[get_local_id(0)], write);
  ensures x[get_local_id(0)] == get_local_id(0) + 1;
@*/
__kernel void f(__local int x[]){
  int i = get_local_id(0);
  x[i] = i;
  x[i] = x[i] + 1;
}