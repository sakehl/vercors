//:: cases opencl_barrier_synchronize_kernel
//:: tools silicon
//:: verdict Fail

// Global mem fence barrier in OpenCL
#include "opencl.h"
/*@
  //Changing local size to 4 and groups to 1 should let this work
  context_everywhere get_local_size(0) == 2;
  context_everywhere get_num_groups(0) == 2;
  context_everywhere x != NULL;

  requires Perm(x[\gtid], write);
  ensures \gtid == 0 ==> Perm(x[\gtid], write);
  ensures \gtid == 0 ==> Perm(x[\gtid+1], write);
  ensures \gtid == 0 ==> Perm(x[\gtid+2], write);
  ensures \gtid == 0 ==> Perm(x[\gtid+3], write);
  ensures \gtid == 0 ==> x[\gtid] == 10;
@*/
__kernel void f(__global int* x){
  int i = get_global_id(0);
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
  barrier(CLK_GLOBAL_MEM_FENCE)
  if(get_global_id(0) == 0){
    //10 2 3 4
    x[0] = x[0] + x[1] + x[2] + x[3];
  }
}