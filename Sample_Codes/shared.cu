#include <stdio.h>
#include <cuda.h>

#define BLOCKSIZE	26

__global__ void dkernel() {
	__shared__ char str[BLOCKSIZE+1];
	str[threadIdx.x] = 'A' + (threadIdx.x + blockIdx.x) % BLOCKSIZE;
	if (threadIdx.x == 0) {
		str[BLOCKSIZE] = '\0';
	}
	//__syncthreads();
	if (threadIdx.x == 0) {
		printf("%d: %s\n", blockIdx.x, str);
	}
}
int main() {
	dkernel<<<10, BLOCKSIZE>>>();
	cudaDeviceSynchronize();
}
