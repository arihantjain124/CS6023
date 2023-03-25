#include<stdio.h>
#include<cuda.h>

#define N		1024
#define BLOCKSIZE	64

__device__ volatile unsigned k2counter;

__global__ void K2init() {
	k2counter = 0;
}
__global__ void K2() {
	printf("This is before: %d\n", id);

	// global barrier start
	__syncthreads();	// synchronized with all the threads in this block.

	if (threadIdx.x == 0) {	// representative
		atomicInc((unsigned *)&k2counter, gridDim.x + 1);
		while (k2counter != gridDim.x)
			;
	}
	__syncthreads();
	// global barrier end

	printf("This is after the global barrier: %d\n", id);
}
int main() {
	K2init<<<1, 1>>>();
	K2<<<N / BLOCKSIZE, BLOCKSIZE>>>();
	cudaDeviceSynchronize();

	return 0;
}
