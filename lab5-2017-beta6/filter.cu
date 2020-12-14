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

#define TILE_W 16
#define TILE_H 16
#define kernelsizex 3
#define kernelsizey 2
const int BLOCK_W = TILE_W + 2*kernelsizex;
const int BLOCK_H = TILE_H + 2*kernelsizey;


__global__ void filter(unsigned char * image, unsigned char * out,
    const unsigned int imagesizex,
        const unsigned int imagesizey) {
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
__global__ void filter_optimized(unsigned char * image, unsigned char * out,
    const unsigned int imagesizex,
        const unsigned int imagesizey) {

    // Statically allocated shared memory
    __shared__ int s_i[BLOCK_W * BLOCK_H * 3];

    // Compute each thread's global row and column index
    int x = blockIdx.x * TILE_W + threadIdx.x - kernelsizex;
    int y = blockIdx.y * TILE_H + threadIdx.y - kernelsizey;

    // clamp to edge of image
    x = min(max(x, 0), imagesizex-1);
    y = min(max(y,0), imagesizey-1);

    int index = x + y * imagesizex;
    int b_index = threadIdx.y * blockDim.y + threadIdx.x;

    // only read the part of the image that is relevant for your computation. ie each thread loads one pixel
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
            && (threadIdx.x < BLOCK_W - kernelsizex)
              && (threadIdx.y >= kernelsizey)
                && (threadIdx.y < BLOCK_H - kernelsizey))
    {
        // Filter kernel (simple box filter)
        sumR = 0;
        sumG = 0;
        sumB = 0;
        for (dx = -kernelsizex; dx <= kernelsizex; ++dx) {
            for (dy = -kernelsizey; dy <= 2; ++dy) {
                sumR += s_i[(b_index + dy*blockDim.x + dx) * 3 + 0];
                sumG += s_i[(b_index + dy*blockDim.x + dx) * 3 + 1];
                sumB += s_i[(b_index + dy*blockDim.x + dx) * 3 + 2];
            }
        }
        out[index * 3 + 0] = sumR / divby;
        out[index * 3 + 1] = sumG / divby;
        out[index * 3 + 2] = sumB / divby;
    } // end if
} // end filter opti

// Global variables for image data
unsigned char * image, * pixels, * dev_bitmap, * dev_input;
unsigned int imagesizey, imagesizex; // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////

void computeImages() {
    if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY) {
        printf("Kernel size out of bounds!\n");
        return;
    }

    pixels = (unsigned char * ) malloc(imagesizex * imagesizey * 3);
    cudaMalloc((void ** ) & dev_input, imagesizex * imagesizey * 3);
    cudaMemcpy(dev_input, image, imagesizey * imagesizex * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void ** ) & dev_bitmap, imagesizex * imagesizey * 3);
    dim3 grid(imagesizex, imagesizey);
    filter << < grid, 1 >>> (dev_input, dev_bitmap, imagesizex, imagesizey); // Awful load balance
    cudaThreadSynchronize();
    //	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaMemcpy(pixels, dev_bitmap, imagesizey * imagesizex * 3, cudaMemcpyDeviceToHost);
    cudaFree(dev_bitmap);
    cudaFree(dev_input);
}

// ## compute image optimized with shared memory

inline unsigned int iDivUp( const unsigned int &a, const unsigned int &b ) { return ( a%b != 0 ) ? (a/b+1):(a/b); }

void computeImages_optimized() {

  pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);

  const unsigned int blockW = 16;
  const unsigned int blockH = 16;
  const unsigned int tileW = blockW + 2 * kernelsizex;
  const unsigned int tileH = blockH + 2 * kernelsizey;
  const unsigned int threadBlockH = 8;
  const dim3 grid( iDivUp( imagesizex, blockW ), iDivUp( imagesizey, blockH ) );
  printf("gx=%d\n", grid.x);
  printf("gy=%d\n", grid.y);
  const dim3 threadBlock( tileW, tileH );
  printf("bx=%d\n", threadBlock.x);
  printf("by=%d\n", threadBlock.y);

	filter_optimized<<<grid,threadBlock>>>(dev_input, dev_bitmap, imagesizey, imagesizex); // Awful load balance

	cudaThreadSynchronize();
//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
	cudaFree( dev_bitmap );
	cudaFree( dev_input );
}

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

    //computeImages();
    computeImages_optimized();

    // You can save the result to a file like this:
    //	writeppm("out.ppm", imagesizey, imagesizex, pixels);

    glutMainLoop();
    return 0;
}
