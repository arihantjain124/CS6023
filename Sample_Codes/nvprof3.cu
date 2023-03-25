#include <stdio.h>
#include <cuda.h>

__device__ int sumg = 0;
__global__ void K(int num) {
	num += num;
	++num;
	atomicAdd(&sumg, num);
	__shared__ int sum;
	sum = 0;
	__syncthreads();

	sum += num;
}
int main() {
	for (unsigned ii = 0; ii < 100; ++ii) {
		K<<<5, 32>>>(ii);
		cudaDeviceSynchronize();
	}
	return 0;
}
