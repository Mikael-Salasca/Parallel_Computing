// Matrix addition, CPU version
// nvcc matrix_gpu.cu -L /usr/local/cuda/lib -lcudart -o matrix_gpu

#include <stdio.h>

__global__
void add_matrix(float *a, float *b, float *c, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = i + j*N;
	c[index] = a[index] + b[index];
}

#define BLOCK_SIZE 16
#define N 32


int main() {
	float a[N*N];
	float b[N*N];
	float c[N*N];
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}

		const int size = N*N*sizeof(float);

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
		dim3 dimGrid(N/BLOCK_SIZE, N/BLOCK_SIZE);
		dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
		cudaEventRecord(start_event, 0);
		add_matrix<<<dimGrid, dimBlock>>>(a_gpu,b_gpu,c_gpu,N);

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

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%0.2f ", c[i+j*N]);
		}
		printf("\n");
	}

	printf("done\n");

	printf("Elapsed time: %f \n", time/1000);

	return EXIT_SUCCESS;

}
