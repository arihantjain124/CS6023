#include <stdio.h>
#include <cuda.h>

__device__ int sumg = 0;
__global__ void K() {
	int num = blockIdx.x * blockDim.x + threadIdx.x;
	num += num;
	++num;
	atomicAdd(&sumg, num);
	__shared__ int sum;
	sum = 0;
	__syncthreads();

	sum += num;
}
int main() {
	K<<<100, 32*5>>>();
	cudaDeviceSynchronize();
	return 0;
}
