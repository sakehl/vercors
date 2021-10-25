//:: cases cuda_kernel_parameters_storage_specifiers
//:: tools silicon
//:: verdict Error

// Pointer argument without storage specifier (local or global)
#include "cuda.h"
/*@
  context Perm(x[\ltid], write);
  ensures x[\ltid] == \ltid + 1;
@*/
__global__ void f(global int* x){
  int i = threadIdx.x;
  x[i] = i;
  x[i] = x[i] + 1;
}