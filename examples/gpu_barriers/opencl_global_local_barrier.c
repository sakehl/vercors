//:: cases opencl_global_local_barrier.c
//:: tools silicon
//:: verdict Pass

// Global and local mem fence barrier in OpenCL
#include "opencl.h"
/*@
  context_everywhere get_local_size(0) == 2;
  context_everywhere get_num_groups(0) == 2;
  context_everywhere x != NULL;
  context_everywhere y != NULL;

  context Perm(x[\gtid], read);
  context \ltid == 0 ==> Perm(x[\gtid+1], read);
  requires \ltid ==0 ==> Perm(y[\gtid], write);
  requires \ltid ==0 ==> Perm(y[\gtid+1], write);
  requires Perm(x_sh[\ltid], write);

  ensures Perm(y[\gtid], write);
  ensures Perm(x_sh[0], 1\2) ** Perm(x_sh[1], 1\2);
  ensures \ltid == 0 ==> y[\gtid] == \old(x[\gtid]) + \old(x[\gtid+1]) + \gtid;
  ensures \ltid == 1 ==> y[\gtid] == \old(x[\gtid]) + \gtid;
@*/
__kernel void f(__global int* x, __global int* y){
  int i = get_global_id(0);
  int lid = get_local_id(0);
  local int x_sh[2];

  x_sh[lid] = x[i];
  if(get_local_id(0) == 0){
    y[i] = i;
    y[i+1] = i+1;
  }

  /*@
  context Perm(x[\gtid], read);
  context \ltid == 0 ==> Perm(x[\gtid+1], read);
  requires \ltid == 0 ==> Perm(y[\gtid], write);
  requires \ltid ==0 ==> Perm(y[\gtid+1], write);
  requires Perm(x_sh[\ltid], write);
  requires x_sh[\ltid] == x[\gtid];
  requires \ltid == 0 ==> y[\gtid] == \gtid;
  requires \ltid == 0 ==> y[\gtid+1] == \gtid+1;

  ensures Perm(y[\gtid], write);
  ensures Perm(x_sh[0], 1\2) ** Perm(x_sh[1], 1\2);
  ensures x_sh[\ltid] == x[\gtid];
  ensures \ltid == 0 ==> x_sh[\ltid+1] == x[\gtid+1];
  ensures y[\gtid] == \gtid;
  @*/
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
  if(get_local_id(0) == 0){
    y[i] += x_sh[lid] + x_sh[lid+1];
  } else {
    y[i] += x_sh[lid];
  }
}