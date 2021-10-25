//:: cases opencl_no_global_barrier
//:: tools silicon
//:: verdict Error

// Global mem fence barrier in OpenCL needed, but not given.
#include "opencl.h"
/*@
  context_everywhere get_local_size(0) == 2;
  context_everywhere get_num_groups(0) == 2;
  context_everywhere x != NULL;

  requires Perm(x[\gtid], write);
  ensures \ltid == 0 ==> Perm(x[\gtid], write);
  ensures \ltid == 0 ==> Perm(x[\gtid + 1], write);
  ensures \ltid == 0 ==> x[\gtid] == \gtid*2 + 3;
  ensures \ltid == 0 ==> x[\gtid+1] == \gtid + 2;
@*/
__kernel void f(__global int* x){
  int i = get_global_id(0);
  // 0 1 2 3
  x[i] = i;
  // 1 2 3 4
  x[i] = x[i] + 1;
  /*@
  requires \ltid >= 0 && \ltid < 2;
  requires get_group_id(0) >= 0 && get_group_id(0) < 2;

  requires Perm(x[\gtid], write);
  requires x[\gtid] == \gtid + 1;
  ensures \ltid == 0 ==> Perm(x[\gtid], write);
  ensures \ltid == 0 ==> Perm(x[\gtid + 1], write);
  ensures \ltid == 0 ==> x[\gtid] == \gtid + 1;
  ensures \ltid == 0 ==> x[\gtid+1] == \gtid + 2;
  @*/
  barrier(CLK_LOCAL_MEM_FENCE)
  if(get_local_id(0) == 0){
    //3 2 7 4
    x[i] = x[i] + x[i+1];
  }
}