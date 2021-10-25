//:: cases cuda_dynamic_wrong_barrier_permissions
//:: tools silicon
//:: verdict Fail

#include "cuda.h"

/*@
  context_everywhere shared_mem_size_1 == n;
  context_everywhere blockDim.x == n;
  context_everywhere gridDim.x == 1;
  context_everywhere d != NULL;

  context Perm(d[\gtid], write);
  requires Perm(s[\ltid], write);
  ensures Perm(s[n-\ltid-2], write);
@*/
__global__ void dynamicReverse(int *d, int n)
{
  extern __shared__ int s[];
  int t = threadIdx.x;
  //Should be n-t-1
  // We are now indexing in the range [-1,59) with tr
  int tr = n-t-2;
  s[t] = d[t];
  /*@
  context Perm(d[\gtid], write);
  requires Perm(s[\ltid], write);
  ensures Perm(s[n-\ltid-2], write);
  @*/
  __syncthreads();
  d[t] = s[tr];
}