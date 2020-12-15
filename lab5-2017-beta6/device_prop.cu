#include <stdio.h>
// nvcc device_prop.cu -o dP
// 
// struct cudaDeviceProp {
//     char name[256];
//     size_t totalGlobalMem;
//     size_t sharedMemPerBlock;
//     int regsPerBlock;
//     int warpSize;
//     size_t memPitch;
//     int maxThreadsPerBlock;
//     int maxThreadsDim[3];
//     int maxGridSize[3];
//     size_t totalConstMem;
//     int major;
//     int minor;
//     int clockRate;
//     size_t textureAlignment;
//     int deviceOverlap;
//     int multiProcessorCount;
//     int kernelExecTimeoutEnabled;
//     int integrated;
//     int canMapHostMemory;
//     int computeMode;
//     int concurrentKernels;
//     int ECCEnabled;
//     int pciBusID;
//     int pciDeviceID;
//     int tccDriver;
// }

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
   printf("  Shared Mem per block (KB): %lu\n",
          prop.sharedMemPerBlock / 1000);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}
