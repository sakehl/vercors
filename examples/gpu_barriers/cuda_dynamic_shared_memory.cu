//:: cases cuda_dynamic_shared_memory.c
//:: tools silicon
//:: verdict Pass

#include "cuda.h"

/*@
  context_everywhere shared_mem_size_1 == n;
  context_everywhere get_local_size(0) == n;
  context_everywhere get_num_groups(0) == 1;
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