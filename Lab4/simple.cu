// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>

const int N = 16;
const int blocksize = 16;
const float input[16] = { 1.0, 2.0, 4.0, 8.0,	16.0, 32.0, 64.0, 128.0,256.0, 512.0, 1024.0, 2048.0,4096.0, 8192.0, 16284.0, 32568.0 };

__global__
void simple(float *c) {
	c[threadIdx.x] = sqrtf(c[threadIdx.x]);
}

int main() {
	float *c = new float[N];
	float *cd;
	const int size = N*sizeof(float);
	cudaMalloc( (void**)&cd, size );
	cudaMemcpy(cd, (void*)input, size, cudaMemcpyHostToDevice);
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	simple<<<dimGrid, dimBlock>>>(cd);
	cudaDeviceSynchronize();
	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost );
	cudaFree( cd );

	printf("GPU \n");
	for (int i = 0; i < N; i++)
		printf("%f ", c[i]);
	printf("\n");

	printf("CPU \n");
	for (int i = 0; i < N; i++)
		printf("%f ", sqrtf(input[i]));

	delete[] c;
	printf("done\n");
	return EXIT_SUCCESS;
}
