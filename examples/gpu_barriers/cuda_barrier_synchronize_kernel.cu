//:: cases cuda_barrier_synchronize_kernel
//:: tools silicon
//:: verdict Fail

// Global mem fence barrier in cuda
#include "cuda.h"
/*@
  //Changing blockDim to 4 and gridDim to 1 should let this work
  context_everywhere blockDim.x == 2;
  context_everywhere gridDim.x == 2;
  context_everywhere x != NULL;

  requires Perm(x[\gtid], write);
  ensures \gtid == 0 ==> Perm(x[\gtid], write);
  ensures \gtid == 0 ==> Perm(x[\gtid+1], write);
  ensures \gtid == 0 ==> Perm(x[\gtid+2], write);
  ensures \gtid == 0 ==> Perm(x[\gtid+3], write);
  ensures \gtid == 0 ==> x[\gtid] == 10;
@*/
__global__ void f(int* x){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  // 0 1 2 3
  x[i] = i;
  // 1 2 3 4
  x[i] = x[i] + 1;
  /*@
  requires Perm(x[\gtid], write);
  requires x[\gtid] == \gtid + 1;
  ensures \gtid == 0 ==> Perm(x[\gtid], write);
  ensures \gtid == 0 ==> Perm(x[\gtid+1], write);
  ensures \gtid == 0 ==> Perm(x[\gtid+2], write);
  ensures \gtid == 0 ==> Perm(x[\gtid+3], write);
  ensures \gtid == 0 ==> x[\gtid] == 1;
  ensures \gtid == 0 ==> x[\gtid+1] == 2;
  ensures \gtid == 0 ==> x[\gtid+2] == 3;
  ensures \gtid == 0 ==> x[\gtid+3] == 4;
  @*/
  __syncthreads();
  if(i == 0){
    //10 2 3 4
    x[0] = x[0] + x[1] + x[2] + x[3];
  }
}