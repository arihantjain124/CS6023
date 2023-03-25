#include <stdio.h>
#include <cuda.h>

#define BLOCKSIZE	1024

__global__ void dkernel() {
	__shared__ unsigned data[12*1024];
	data[threadIdx.x] = threadIdx.x;
}
int main() {
	cudaFuncSetCacheConfig(dkernel, cudaFuncCachePreferL1);
	//cudaFuncSetCacheConfig(dkernel, cudaFuncCachePreferShared);
	dkernel<<<1, BLOCKSIZE>>>();
	cudaDeviceSynchronize();
}
