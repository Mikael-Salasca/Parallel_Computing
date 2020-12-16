/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length) {


    //each group(of 256 threads) reduce in local memory and setting each first-in-group value to max of its group.

    int thread_id = get_global_id(0);
    int localthread_id = get_local_id(0);
    int local_size = get_local_size(0);
    __local unsigned int shared_mem[1024];
    shared_mem[localthread_id] = data[thread_id];
    barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);

    for(int i = local_size/2; i >= 1;i /= 2)
    {
        if(localthread_id < i)
        {
            if(shared_mem[localthread_id] < shared_mem[localthread_id+i])
                shared_mem[localthread_id] = shared_mem[localthread_id+i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // if (localthread_id == 0) {
    //   printf("dbg \n");
    //
    //   for (int i = 0; i < local_size; ++i){
    //     printf("%d - ", shared_mem[localthread_idi]);
    //   }
    //   printf(" \n");

    //}
    // if(localthread_id == 0)
    //     data[thread_id]=shared_mem[localthread_id];
    //
    // barrier(CLK_LOCAL_MEM_FENCE);

    // assign the max to the first
    if(thread_id == 0){
      int val = shared_mem[0];
      for(int i = local_size; i < length; i+= local_size){
        if(shared_mem[i] > val){
          val = data[i];
        }
      }
      data[0]=val;

    }



}
