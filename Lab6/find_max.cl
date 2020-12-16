/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length) {


    //each group(of 256 threads) reduce in local memory and setting each first-in-group value to max of its group.

    int thread_id = get_global_id(0);
    int localthread_id = get_local_id(0);
    int local_size = get_local_size(0);
    __local unsigned int shared_mem[1024];

    // load in shared memory
    shared_mem[localthread_id] = data[thread_id];
    barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);

    for(int stride=1; stride < local_size; stride *= 2) {
               if (localthread_id % (2*stride) == 0) {
                   unsigned int lhs = shared_mem[localthread_id];
                   unsigned int rhs = shared_mem[localthread_id + stride];
                   shared_mem[localthread_id] = lhs < rhs ? rhs : lhs;
               }
               barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
           }

       if (localthread_id == 0) data[0] = shared_mem[0];

}
