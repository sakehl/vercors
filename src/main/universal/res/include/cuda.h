// Custom VerCors header for CUDA
#ifndef CUDA_H
#define CUDA_H

// Keywords for CUDA
#define __global__ __vercors_kernel__
#define __shared__ __vercors_local_memory__
// Barriers
#define __syncthreads() __vercors_barrier__(1|2)

#define cudaEvent_t int
#define cudaMemcpyHostToDevice 0
#define cudaMemcpyDeviceToHost 1

#define atomicMin(tgt, val) __vercors_atomic__ { (tgt)[0] = (tgt)[0] < (val) ? (tgt)[0] : (val); }
#define atomicAdd(tgt, val) __vercors_atomic__ { (tgt)[0] += (val); }
#define atomicRelax(tgt, w, s) __vercors_atomic__ { (tgt)[0] = (s != -1 && ((tgt)[0] == -1 || s+w <= (tgt)[0])) ? s+w : (tgt)[0]; }

extern /*@ pure @*/ int get_work_dim(); // Number of dimensions in use

extern /*@ pure @*/ int get_global_size(int dimindx); // Number of global work-items

extern /*@ pure @*/ int get_global_id (int dimindx); // Global work-item ID value

extern /*@ pure @*/ int get_local_size (int dimindx); //

extern /*@ pure @*/ int get_enqueued_local_size (int dimindx); // Number of local work-items

extern /*@ pure @*/ int get_local_id (int dimindx); // Local work-item ID

extern /*@ pure @*/ int get_num_groups (int dimindx); // Number of work-groups

extern /*@ pure @*/ int get_group_id (int dimindx); // Work-group ID

extern /*@ pure @*/ int get_global_offset (int dimindx); // Global offset

extern /*@ pure @*/ int get_global_linear_id (); // Work-items, 1-dimensional global ID

extern /*@ pure @*/ int get_local_linear_id (); // Work-items, 1-dimensional local ID

extern /*@ pure @*/ int get_sub_group_size (); // Number of work-items in the subgroup

extern /*@ pure @*/ int get_max_sub_group_size (); // Maximum size of a subgroup

extern /*@ pure @*/ int get_num_sub_groups (); // Number of subgroups

extern /*@ pure @*/ int get_enqueued_num_sub_groups (); //

extern /*@ pure @*/ int get_sub_group_id (); // Sub-group ID

extern /*@ pure @*/ int get_sub_group_local_id (); // Unique work-item ID

cudaEvent_t cudaEventCreate();
void cudaEventDestroy(cudaEvent_t e);
void cudaEventRecord(cudaEvent_t e, int i);
void cudaEventSynchronize(cudaEvent_t e);
int cudaEventElapsedTime(cudaEvent_t begin, cudaEvent_t end);

#endif