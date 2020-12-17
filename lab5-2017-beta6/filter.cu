// Lab 5, image filters with CUDA.

// Compile with a command-line similar to Lab 4:
// nvcc filter.cu -c -arch=sm_30 -o filter.o
  // g++ filter.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
// or (multicore lab)
// nvcc filter.cu -c -arch=sm_20 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -L/usr/local/cuda/lib64 -lcudart -lglut -o filter

// 2017-11-27: Early pre-release, dubbed "beta".
// 2017-12-03: First official version! Brand new lab 5 based on the old lab 6.
// Better variable names, better prepared for some lab tasks. More changes may come
// but I call this version 1.0b2.
// 2017-12-04: Two fixes: Added command-lines (above), fixed a bug in computeImages
// that allocated too much memory. b3
// 2017-12-04: More fixes: Tightened up the kernel with edge clamping.
// Less code, nicer result (no borders). Cleaned up some messed up X and Y. b4

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
#else
  #include <GL/glut.h>
#endif
#include "readppm.h"
#include "milli.h"

// Use these for setting shared memory size.
#define maxKernelSizeX 10
#define maxKernelSizeY 10

#define TILE_W 8
#define TILE_H 8
#define KS_X 3
#define KS_Y 3
#define VERSION 5// 1 naive, 2 shared, 3 shared sep, 4 gaussian 5x5 sep, 5 median


__managed__
int kernelsize_pad = 0;
__managed__
int BLOCK_W = 0;
__managed__
int BLOCK_H = 0;

__global__ void filter(unsigned char * image, unsigned char * out,
    const unsigned int imagesizex,
        const unsigned int imagesizey,
        const int kernelsizex,
          const int kernelsizey) {
    // map from blockIdx to pixel position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int dy, dx;
    unsigned int sumR, sumG, sumB;

    int divby = (2 * kernelsizex + 1) * (2 * kernelsizey + 1); // Works for box filters only!

    if (x < imagesizex && y < imagesizey) // If inside image
    {
        // Filter kernel (simple box filter)
        sumR = 0;
        sumG = 0;
        sumB = 0;
        for (dy = -kernelsizey; dy <= kernelsizey; dy++)
            for (dx = -kernelsizex; dx <= kernelsizex; dx++) {
                // Use max and min to avoid branching!
                int yy = min(max(y + dy, 0), imagesizey - 1);
                int xx = min(max(x + dx, 0), imagesizex - 1);

                sumR += image[((yy) * imagesizex + (xx)) * 3 + 0];
                sumG += image[((yy) * imagesizex + (xx)) * 3 + 1];
                sumB += image[((yy) * imagesizex + (xx)) * 3 + 2];
            }
        out[(y * imagesizex + x) * 3 + 0] = sumR / divby;
        out[(y * imagesizex + x) * 3 + 1] = sumG / divby;
        out[(y * imagesizex + x) * 3 + 2] = sumB / divby;
    }
}

// Filter optimized with shared memory
__global__ void filter_shared(unsigned char * image, unsigned char * out,
    const unsigned int imagesizex,
        const unsigned int imagesizey,
          const int kernelsizex,
            const int kernelsizey) {

    // Statically allocated shared memory
    __shared__ int s_i[(TILE_W + 2*maxKernelSizeX) * (TILE_H + 2*maxKernelSizeY) * 3];

    int x = blockIdx.x * TILE_W + threadIdx.x - kernelsizex;
    int y = blockIdx.y * TILE_H + threadIdx.y - kernelsizey;

    // clamp to edge of image
    x = min(max(x, 0), imagesizex-1);
    y = min(max(y,0), imagesizey-1);

    int index = x + y * imagesizex;
    int b_index = threadIdx.y * blockDim.y + threadIdx.x;

    // eah thread copies its pixel of the block to shared memory
    s_i[b_index * 3 + 0] = image[index * 3 + 0];
    s_i[b_index * 3 + 1] = image[index * 3 + 1];
    s_i[b_index * 3 + 2] = image[index * 3 + 2];

    __syncthreads();

    unsigned int sumR, sumG, sumB;

    int divby = (2 * kernelsizex + 1) * (2 * kernelsizey + 1); // Works for box filters only!

    int dy, dx;

    // Only threads inside the apron will write results
    if ((threadIdx.x >= kernelsizex)
          && (threadIdx.x < BLOCK_W - kernelsizex)
              && (threadIdx.y >= kernelsizey)
                && (threadIdx.y < BLOCK_H - kernelsizey))
    {
        // Filter kernel (simple box filter)
        sumR = 0;
        sumG = 0;
        sumB = 0;
        for (dy = -kernelsizey; dy <= kernelsizey; ++dy) {
          for (dx = -kernelsizex; dx <= kernelsizex; ++dx) {
                sumR += s_i[(b_index + dy*blockDim.x + dx) * 3 + 0];
                sumG += s_i[(b_index + dy*blockDim.x + dx) * 3 + 1];
                sumB += s_i[(b_index + dy*blockDim.x + dx) * 3 + 2];
            }
        }
        out[index * 3 + 0] = sumR / divby;
        out[index * 3 + 1] = sumG / divby;
        out[index * 3 + 2] = sumB / divby;
    } // end if
} // end filter shared

// Filter GAUSS 5x5
__global__ void filter_gauss_horizontal(unsigned char * image, unsigned char * out,
    const unsigned int imagesizex,
        const unsigned int imagesizey,
          const int kernelsizex,
            const int kernelsizey) {

    // Statically allocated shared memory
    __shared__ int s_i[(TILE_W + 2*maxKernelSizeX) * (TILE_H + 2*maxKernelSizeY) * 3];

    int x = blockIdx.x * TILE_W + threadIdx.x - kernelsizex;
    int y = blockIdx.y * TILE_H + threadIdx.y - kernelsizey;

    // clamp to edge of image
    x = min(max(x, 0), imagesizex-1);
    y = min(max(y,0), imagesizey-1);

    int index = x + y * imagesizex;
    int b_index = threadIdx.y * blockDim.y + threadIdx.x;

    // eah thread copies its pixel of the block to shared memory
    s_i[b_index * 3 + 0] = image[index * 3 + 0];
    s_i[b_index * 3 + 1] = image[index * 3 + 1];
    s_i[b_index * 3 + 2] = image[index * 3 + 2];

    __syncthreads();

    unsigned int sumR, sumG, sumB;

    int gauss_weight[5] = {1,4,6,4,1};
    int gauss_index;
    int divby = 16; // Works for 5x5 gauss

    int dy, dx;

    // Only threads inside the apron will write results
    if ((threadIdx.x >= kernelsizex)
          && (threadIdx.x < BLOCK_W - kernelsizex)
              && (threadIdx.y >= kernelsizey)
                && (threadIdx.y < BLOCK_H - kernelsizey))
    {
        // Filter kernel (simple box filter)
        sumR = 0;
        sumG = 0;
        sumB = 0;
        gauss_index = 0;

        for (dy = -kernelsizey; dy <= kernelsizey; ++dy) {
          for (dx = -kernelsizex; dx <= kernelsizex; ++dx) {
                sumR += s_i[(b_index + dy*blockDim.x + dx) * 3 + 0] * gauss_weight[gauss_index];
                sumG += s_i[(b_index + dy*blockDim.x + dx) * 3 + 1] * gauss_weight[gauss_index];
                sumB += s_i[(b_index + dy*blockDim.x + dx) * 3 + 2] * gauss_weight[gauss_index];
            }
            gauss_index++;
        }
        out[index * 3 + 0] = sumR / divby;
        out[index * 3 + 1] = sumG / divby;
        out[index * 3 + 2] = sumB / divby;
    } // end if

}

// Filter GAUSS 5x5
__global__ void filter_gauss_vertical(unsigned char * image, unsigned char * out,
    const unsigned int imagesizex,
        const unsigned int imagesizey,
          const int kernelsizex,
            const int kernelsizey) {

    // Statically allocated shared memory
    __shared__ int s_i[(TILE_W + 2*maxKernelSizeX) * (TILE_H + 2*maxKernelSizeY) * 3];

    int x = blockIdx.x * TILE_W + threadIdx.x - kernelsizex;
    int y = blockIdx.y * TILE_H + threadIdx.y - kernelsizey;

    // clamp to edge of image
    x = min(max(x, 0), imagesizex-1);
    y = min(max(y,0), imagesizey-1);

    int index = x + y * imagesizex;
    int b_index = threadIdx.y * blockDim.y + threadIdx.x;

    // eah thread copies its pixel of the block to shared memory
    s_i[b_index * 3 + 0] = image[index * 3 + 0];
    s_i[b_index * 3 + 1] = image[index * 3 + 1];
    s_i[b_index * 3 + 2] = image[index * 3 + 2];

    __syncthreads();

    unsigned int sumR, sumG, sumB;

    int gauss_weight[5] = {1,4,6,4,1};
    int gauss_index;
    int divby = 16; // Works for 5x5 gauss

    int dy, dx;

    // Only threads inside the apron will write results
    if ((threadIdx.x >= kernelsizex)
          && (threadIdx.x < BLOCK_W - kernelsizex)
              && (threadIdx.y >= kernelsizey)
                && (threadIdx.y < BLOCK_H - kernelsizey))
    {
        // Filter kernel (simple box filter)
        sumR = 0;
        sumG = 0;
        sumB = 0;
        gauss_index = 0;

          for (dx = -kernelsizex; dx <= kernelsizex; ++dx) {
                sumR += s_i[(b_index + dy*blockDim.x + dx) * 3 + 0] * gauss_weight[gauss_index];
                sumG += s_i[(b_index + dy*blockDim.x + dx) * 3 + 1] * gauss_weight[gauss_index];
                sumB += s_i[(b_index + dy*blockDim.x + dx) * 3 + 2] * gauss_weight[gauss_index];
                ++gauss_index;
            }
        out[index * 3 + 0] = sumR / divby;
        out[index * 3 + 1] = sumG / divby;
        out[index * 3 + 2] = sumB / divby;
    } // end if
}

__device__ void exchange(unsigned int *i, unsigned int *j) {
     int k;
     k = *i;
     *i = *j;
     *j = k;
}

__global__ void filter_median_horizontal(unsigned char * image, unsigned char * out,
    const unsigned int imagesizex,
        const unsigned int imagesizey,
          const int kernelsizex,
            const int kernelsizey) {

    // Statically allocated shared memory
    __shared__ int s_i[(TILE_W + 2*maxKernelSizeX) * (TILE_H + 2*maxKernelSizeY) * 3];

    int x = blockIdx.x * TILE_W + threadIdx.x - kernelsizex;
    int y = blockIdx.y * TILE_H + threadIdx.y - kernelsizey;

    // clamp to edge of image
    x = min(max(x, 0), imagesizex-1);
    y = min(max(y,0), imagesizey-1);

    int index = x + y * imagesizex;
    int b_index = threadIdx.y * blockDim.y + threadIdx.x;

    // eah thread copies its pixel of the block to shared memory
    s_i[b_index * 3 + 0] = image[index * 3 + 0];
    s_i[b_index * 3 + 1] = image[index * 3 + 1];
    s_i[b_index * 3 + 2] = image[index * 3 + 2];

    __syncthreads();

    unsigned int arr_R[maxKernelSizeX*2+1], arr_G[maxKernelSizeX*2+1], arr_B[maxKernelSizeX*2+1];
    unsigned int arr_index;

    int dy, dx;

    // Only threads inside the apron will write results
    if ((threadIdx.x >= kernelsizex)
          && (threadIdx.x < BLOCK_W - kernelsizex)
              && (threadIdx.y >= kernelsizey)
                && (threadIdx.y < BLOCK_H - kernelsizey))
    {
        arr_index = 0;
        for (dy = -kernelsizey; dy <= kernelsizey; ++dy) {
          for (dx = -kernelsizex; dx <= kernelsizex; ++dx) {
                arr_R[arr_index] = s_i[(b_index + dy*blockDim.x + dx) * 3 + 0];
                arr_G[arr_index] = s_i[(b_index + dy*blockDim.x + dx) * 3 + 1];
                arr_B[arr_index] = s_i[(b_index + dy*blockDim.x + dx) * 3 + 2];
            }
            arr_index++;
        }

        // sort arrays
        unsigned int i,j,k;

        //if (threadIdx.x == 0) {

          for (k=2;k<=arr_index;k=2*k) // Outer loop, double size for each step
          {
            for (j=k>>1;j>0;j=j>>1) // Inner loop, half size for each step
            {
              for (i=0;i<arr_index;i++) // Loop over ARR
              {
                int ixj=i^j; // Calculate indexing!
                if ((ixj)>i)
                {
                  if ((i&k)==0 && arr_R[i]>arr_R[ixj]) exchange(&arr_R[i],&arr_R[ixj]);
                  if ((i&k)!=0 && arr_R[i]<arr_R[ixj]) exchange(&arr_R[i],&arr_R[ixj]);

                  if ((i&k)==0 && arr_G[i]>arr_G[ixj]) exchange(&arr_G[i],&arr_G[ixj]);
                  if ((i&k)!=0 && arr_G[i]<arr_G[ixj]) exchange(&arr_G[i],&arr_G[ixj]);

                  if ((i&k)==0 && arr_B[i]>arr_B[ixj]) exchange(&arr_B[i],&arr_B[ixj]);
                  if ((i&k)!=0 && arr_B[i]<arr_B[ixj]) exchange(&arr_B[i],&arr_B[ixj]);
                }
              }
            }
          }
        //}

        //__syncthreads();

        out[index * 3 + 0] = arr_R[(arr_index-1)/2];
        out[index * 3 + 1] = arr_G[(arr_index-1)/2];
        out[index * 3 + 2] = arr_B[(arr_index-1)/2];
    } // end if
} // end filter median horizontal

// Filter optimized with shared memory
__global__ void filter_median_vertical(unsigned char * image, unsigned char * out,
    const unsigned int imagesizex,
        const unsigned int imagesizey,
          const int kernelsizex,
            const int kernelsizey) {

    // Statically allocated shared memory
    __shared__ int s_i[(TILE_W + 2*maxKernelSizeX) * (TILE_H + 2*maxKernelSizeY) * 3];

    int x = blockIdx.x * TILE_W + threadIdx.x - kernelsizex;
    int y = blockIdx.y * TILE_H + threadIdx.y - kernelsizey;

    // clamp to edge of image
    x = min(max(x, 0), imagesizex-1);
    y = min(max(y,0), imagesizey-1);

    int index = x + y * imagesizex;
    int b_index = threadIdx.y * blockDim.y + threadIdx.x;

    // eah thread copies its pixel of the block to shared memory
    s_i[b_index * 3 + 0] = image[index * 3 + 0];
    s_i[b_index * 3 + 1] = image[index * 3 + 1];
    s_i[b_index * 3 + 2] = image[index * 3 + 2];

    __syncthreads();

    unsigned int arr_R[maxKernelSizeX*2+1], arr_G[maxKernelSizeX*2+1], arr_B[maxKernelSizeX*2+1];
    unsigned int arr_index;

    int dy, dx;

    // Only threads inside the apron will write results
    if ((threadIdx.x >= kernelsizex)
          && (threadIdx.x < BLOCK_W - kernelsizex)
              && (threadIdx.y >= kernelsizey)
                && (threadIdx.y < BLOCK_H - kernelsizey))
    {
        arr_index = 0;
        for (dy = -kernelsizey; dy <= kernelsizey; ++dy) {
          for (dx = -kernelsizex; dx <= kernelsizex; ++dx) {
                arr_R[arr_index] = s_i[(b_index + dy*blockDim.x + dx) * 3 + 0];
                arr_G[arr_index] = s_i[(b_index + dy*blockDim.x + dx) * 3 + 1];
                arr_B[arr_index] = s_i[(b_index + dy*blockDim.x + dx) * 3 + 2];
                arr_index++;
            }
        }

        // sort arrays
        unsigned int i,j,k;

        //if (threadIdx.x == 0) {

          for (k=2;k<=arr_index;k=2*k) // Outer loop, double size for each step
          {
            for (j=k>>1;j>0;j=j>>1) // Inner loop, half size for each step
            {
              for (i=0;i<arr_index;i++) // Loop over ARR
              {
                int ixj=i^j; // Calculate indexing!
                if ((ixj)>i)
                {
                  if ((i&k)==0 && arr_R[i]>arr_R[ixj]) exchange(&arr_R[i],&arr_R[ixj]);
                  if ((i&k)!=0 && arr_R[i]<arr_R[ixj]) exchange(&arr_R[i],&arr_R[ixj]);

                  if ((i&k)==0 && arr_G[i]>arr_G[ixj]) exchange(&arr_G[i],&arr_G[ixj]);
                  if ((i&k)!=0 && arr_G[i]<arr_G[ixj]) exchange(&arr_G[i],&arr_G[ixj]);

                  if ((i&k)==0 && arr_B[i]>arr_B[ixj]) exchange(&arr_B[i],&arr_B[ixj]);
                  if ((i&k)!=0 && arr_B[i]<arr_B[ixj]) exchange(&arr_B[i],&arr_B[ixj]);
                }
              }
            }
          }
        //}

        //__syncthreads();

        out[index * 3 + 0] = arr_R[(arr_index-1)/2];
        out[index * 3 + 1] = arr_G[(arr_index-1)/2];
        out[index * 3 + 2] = arr_B[(arr_index-1)/2];
    } // end if
} // end filter median horizontal



// Global variables for image data
unsigned char * image, * pixels, * dev_bitmap, * dev_input;
unsigned int imagesizey, imagesizex; // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////

void computeImages(int kernelsizex, int kernelsizey) {
    if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY) {
        printf("Kernel size out of bounds!\n");
        return;
    }

    pixels = (unsigned char * ) malloc(imagesizex * imagesizey * 3);
    cudaMalloc((void ** ) & dev_input, imagesizex * imagesizey * 3);
    cudaMemcpy(dev_input, image, imagesizey * imagesizex * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void ** ) & dev_bitmap, imagesizex * imagesizey * 3);
    dim3 grid(imagesizex, imagesizey);
    cudaEvent_t start;
    cudaEvent_t end;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    filter << < grid, 1 >>> (dev_input, dev_bitmap, imagesizex, imagesizey,kernelsizex,kernelsizey); // Awful load balance

    cudaDeviceSynchronize();
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    printf("Naive version - Elapsed time (ms): %f \n", time);    //	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaMemcpy(pixels, dev_bitmap, imagesizey * imagesizex * 3, cudaMemcpyDeviceToHost);
    cudaFree(dev_bitmap);
    cudaFree(dev_input);
}

// ## compute image optimized with shared memory

inline unsigned int iDivUp( const unsigned int &a, const unsigned int &b ) { return ( a%b != 0 ) ? (a/b+1):(a/b); }

void computeImages_shared(int kernelsizex, int kernelsizey) {

  pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);

   kernelsize_pad = max(kernelsizex,kernelsizey);
   BLOCK_W = TILE_W + 2*kernelsizex;
   BLOCK_H = TILE_H + 2*kernelsizey;


  const dim3 grid( iDivUp( imagesizex, TILE_W ), iDivUp( imagesizey, TILE_H ) );
  printf("gx=%d\n", grid.x);
  printf("gy=%d\n", grid.y);
  const dim3 threadBlock( BLOCK_W, BLOCK_W );
  printf("bx=%d\n", threadBlock.x);
  printf("by=%d\n", threadBlock.y);

  cudaEvent_t start;
  cudaEvent_t end;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

	filter_shared<<<grid,threadBlock>>>(dev_input, dev_bitmap, imagesizey, imagesizex, kernelsizex, kernelsizey);

  cudaDeviceSynchronize();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  printf("Shared memory version - Elapsed time (ms): %f \n", time);

	cudaThreadSynchronize();
//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
	cudaFree( dev_bitmap );
	cudaFree( dev_input );
} // end compute __shared__

void computeImages_shared_separable(int kernelsizex, int kernelsizey) {

  pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);

   kernelsize_pad = max(kernelsizex,kernelsizey);
   BLOCK_W = TILE_W + 2*kernelsizex;
   BLOCK_H = TILE_H + 2*kernelsizey;


  const dim3 grid( iDivUp( imagesizex, TILE_W ), iDivUp( imagesizey, TILE_H ) );
  printf("gx=%d\n", grid.x);
  printf("gy=%d\n", grid.y);
  const dim3 threadBlock( BLOCK_W, BLOCK_W );
  printf("bx=%d\n", threadBlock.x);
  printf("by=%d\n", threadBlock.y);

  cudaEvent_t start;
  cudaEvent_t end;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

	filter_shared<<<grid,threadBlock>>>(dev_input, dev_bitmap, imagesizey, imagesizex, kernelsizex, 1);
  cudaDeviceSynchronize();
  filter_shared<<<grid,threadBlock>>>(dev_bitmap, dev_bitmap, imagesizey, imagesizex, 1, kernelsizey);

  cudaDeviceSynchronize();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  printf("Separable version - Elapsed time (ms): %f \n", time);

	cudaThreadSynchronize();
//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
	cudaFree( dev_bitmap );
	cudaFree( dev_input );
}// end compute shared separable


void computeImages_gauss_separable(int kernelsizex, int kernelsizey) {

  pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);

   kernelsize_pad = max(kernelsizex,kernelsizey);
   BLOCK_W = TILE_W + 2*2;
   BLOCK_H = TILE_H + 2*2;


  const dim3 grid( iDivUp( imagesizex, TILE_W ), iDivUp( imagesizey, TILE_H ) );
  printf("gx=%d\n", grid.x);
  printf("gy=%d\n", grid.y);
  const dim3 threadBlock( BLOCK_W, BLOCK_W );
  printf("bx=%d\n", threadBlock.x);
  printf("by=%d\n", threadBlock.y);

  cudaEvent_t start;
  cudaEvent_t end;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  filter_gauss_horizontal<<<grid,threadBlock>>>(dev_bitmap, dev_bitmap, imagesizey, imagesizex, 1, 2);
  cudaDeviceSynchronize();
	filter_gauss_vertical<<<grid,threadBlock>>>(dev_input, dev_bitmap, imagesizey, imagesizex, 2, 1);

  cudaDeviceSynchronize();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  printf("Gaussian version - Elapsed time (ms): %f \n", time);

	cudaThreadSynchronize();
//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
	cudaFree( dev_bitmap );
	cudaFree( dev_input );
}// end compute shared gaussian separable


void computeImages_median(int kernelsizex, int kernelsizey) {

  pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);

   kernelsize_pad = max(kernelsizex,kernelsizey);
   BLOCK_W = TILE_W + 2*kernelsizex;
   BLOCK_H = TILE_H + 2*kernelsizey;


  const dim3 grid( iDivUp( imagesizex, TILE_W ), iDivUp( imagesizey, TILE_H ) );
  printf("gx=%d\n", grid.x);
  printf("gy=%d\n", grid.y);
  const dim3 threadBlock( BLOCK_W, BLOCK_W );
  printf("bx=%d\n", threadBlock.x);
  printf("by=%d\n", threadBlock.y);

  cudaEvent_t start;
  cudaEvent_t end;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  filter_median_horizontal<<<grid,threadBlock>>>(dev_bitmap, dev_bitmap, imagesizey, imagesizex, 1, kernelsizey);
  cudaDeviceSynchronize();
  filter_median_vertical<<<grid,threadBlock>>>(dev_input, dev_bitmap, imagesizey, imagesizex, kernelsizex, 1);

  cudaDeviceSynchronize();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  printf("Median version - Elapsed time (ms): %f \n", time);

	cudaThreadSynchronize();
//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
	cudaFree( dev_bitmap );
	cudaFree( dev_input );
} // end compute median


// Display images
void Draw() {
    // Dump the whole picture onto the screen.
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    if (imagesizey >= imagesizex) { // Not wide - probably square. Original left, result right.
        glRasterPos2f(-1, -1);
        glDrawPixels(imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image);
        glRasterPos2i(0, -1);
        glDrawPixels(imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    } else { // Wide image! Original on top, result below.
        glRasterPos2f(-1, -1);
        glDrawPixels(imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels);
        glRasterPos2i(-1, 0);
        glDrawPixels(imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image);
    }
    glFlush();
}

// Main program, inits
int main(int argc, char ** argv) {
    glutInit( & argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);

    if (argc > 1)
        image = readppm(argv[1], (int * ) & imagesizex, (int * ) & imagesizey);
    else
        image = readppm((char * )
            "maskros512.ppm", (int * ) & imagesizex, (int * ) & imagesizey);

    if (imagesizey >= imagesizex)
        glutInitWindowSize(imagesizex * 2, imagesizey);
    else
        glutInitWindowSize(imagesizex, imagesizey * 2);
    glutCreateWindow("Lab 5");
    glutDisplayFunc(Draw);

    ResetMilli();

    printf("Kernel size X=%d\n",KS_X );
    printf("Kernel size Y=%d\n",KS_Y );
    printf("Kernel size =%d\n", ((KS_X*2+1) * (KS_Y*2+1)));

    int v = VERSION;
    switch (v) {
      case 1:
        computeImages(KS_X,KS_Y);
        break;
      case 2:
        computeImages_shared(KS_X,KS_Y);
        break;
      case 3:
        computeImages_shared_separable(KS_X,KS_Y);
        break;
      case 4:
        computeImages_gauss_separable(2,2);
        break;
      case 5:
        computeImages_median(KS_X,KS_Y);
        break;
      default:
        computeImages(KS_X,KS_Y);
        break;
    }


    // You can save the result to a file like this:
    //	writeppm("out.ppm", imagesizey, imagesizex, pixels);

    glutMainLoop();
    return 0;
}
