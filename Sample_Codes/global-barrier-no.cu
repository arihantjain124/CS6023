#include<stdio.h>
#include<cuda.h>

#define N		1024
#define BLOCKSIZE	64

__device__ unsigned binary[N];

__global__ void K() {
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
	binary[id] = id;
	__syncthreads();
	if (binary[N-1 - id] != N-1 - id)
		printf("Error: There is no global barrier.\n");
}

int main() {
	K<<<N / BLOCKSIZE, BLOCKSIZE>>>();
	cudaDeviceSynchronize();

	return 0;
}
