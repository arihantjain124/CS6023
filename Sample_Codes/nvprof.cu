#include <stdio.h>
#include <cuda.h>

__global__ void K1(int num) {
	num += num;
	++num;
}
__device__ int sum = 0;
__global__ void K2(int num) {
	atomicAdd(&sum, num);
}
__global__ void K3(int num) {
	__shared__ int sum;
	sum = 0;
	__syncthreads();

	sum += num;
}
int main() {
	for (unsigned ii = 0; ii < 100; ++ii) {
		K1<<<5, 32>>>(ii);
		cudaDeviceSynchronize();
	}
	for (unsigned ii = 0; ii < 100; ++ii) {
		K2<<<5, 32>>>(ii);
		cudaDeviceSynchronize();
	}
	for (unsigned ii = 0; ii < 100; ++ii) {
		K3<<<5, 32>>>(ii);
		cudaDeviceSynchronize();
	}
	return 0;
}
