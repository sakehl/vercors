//:: cases cuda_dynamic_shared_memory_no_size
//:: tools silicon
//:: verdict Fail

#include "cuda.h"

/*@
  // Cannot give enough permissions without the following line
  // context_everywhere shared_mem_size_1 == n;
  context_everywhere blockDim.x == n;
  context_everywhere gridDim.x == 1;
  context_everywhere d != NULL;

  context Perm(d[\gtid], write);
  requires Perm(s[\ltid], write);
  ensures Perm(s[n-\ltid-1], write);
@*/
__global__ void dynamicReverse(int *d, int n)
{
  extern __shared__ int s[];
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