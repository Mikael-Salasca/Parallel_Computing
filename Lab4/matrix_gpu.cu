// Matrix addition, CPU version
// nvcc matrix_gpu.cu milli.c -L /usr/local/cuda/lib -lcudart -o matrix

#include <stdio.h>
#include "milli.h"

__global__
void add_matrix_gpu(float *a, float *b, float *c, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = i + j*N;
	c[index] = a[index] + b[index];
}

void add_matrix_cpu(float *a, float *b, float *c, int N) {
	int index;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			index = i + j*N;
			c[index] = a[index] + b[index];
		}
}

#define BLOCK_X_SIZE 16
#define BLOCK_Y_SIZE 16

#define N 8192


int main() {
	float *a = new float[N*N];
	float *b = new float[N*N];
	float *c = new float[N*N];

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}

		const int size = N*N*sizeof(float);

		//GPU
		float *a_gpu;
		float *b_gpu;
		float *c_gpu;
		cudaEvent_t start_event, later_event;
		cudaEventCreate(&start_event);
		cudaEventCreate(&later_event);

		cudaMalloc((void**)&a_gpu, size);
		cudaMalloc((void**)&b_gpu, size);
		cudaMalloc((void**)&c_gpu, size);

		cudaMemcpy(a_gpu,(void*)a, size, cudaMemcpyHostToDevice);
		cudaMemcpy(b_gpu,(void*)b, size, cudaMemcpyHostToDevice);

		dim3 threadsPerBlock(BLOCK_X_SIZE, BLOCK_Y_SIZE); // 16*16, 256 threads
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

		cudaEventRecord(start_event, 0);

		add_matrix_gpu<<<numBlocks, threadsPerBlock>>>(a_gpu,b_gpu,c_gpu,N);

		cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

		cudaDeviceSynchronize();
		cudaMemcpy( c, c_gpu, size, cudaMemcpyDeviceToHost);
		cudaFree( a_gpu );
		cudaFree( b_gpu );
		cudaFree( c_gpu );

		cudaEventRecord(later_event, 0);
	  cudaEventSynchronize(later_event);
		float time = 0.0;
	  cudaEventElapsedTime(&time, start_event, later_event);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)	{
			// printf("%0.2f ", c[i+j*N]);
		}
		// printf("\n");
	}
	printf("N: %d \n", N);
	printf("Block size: %d \n", BLOCK_X_SIZE);
	printf("Nb blocks: %d \n", N / threadsPerBlock.x);
	printf("GPU - Elapsed time (ms): %f \n", time);

	// CPU
	int start = GetMicroseconds();
	add_matrix_cpu(a, b, c, N);
	float end = (GetMicroseconds() - (float)start ) / 1000;

	for (int i = 0; i < N; i++)	{
		for (int j = 0; j < N; j++)	{
			//printf("%0.2f ", c[i+j*N]);
		}
		//printf("\n");
	}

	printf("CPU - Elapsed Time (ms): %f\n ", end);

	delete[] a;
	delete[] b;
	delete[] c;

	printf("done\n");

	return EXIT_SUCCESS;

}
