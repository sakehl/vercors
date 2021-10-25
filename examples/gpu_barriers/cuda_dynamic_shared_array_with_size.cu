//:: cases cuda_dynamic_shared_array_with_size
//:: tools silicon
//:: verdict Error

#include "cuda.h"

/*@
  context_everywhere n == 64;
  context_everywhere shared_mem_size_1 == n;
  context_everywhere blockDim.x == n;
  context_everywhere gridDim.x == 1;
  context_everywhere d != NULL;

  context Perm(d[\gtid], write);
  requires Perm(s[\ltid], write);
  ensures Perm(s[n-\ltid-1], write);
@*/
__global__ void dynamicReverse(int *d, int n)
{
  extern __shared__ int s[64];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  /*@
  context Perm(d[\gtid], write);
  requires Perm(s[\ltid], write);
  ensures Perm(s[n-\ltid-1], write);
  @*/
  __syncthreads();
  d[t] = s[tr];
}