#include <cuda.h>
#include <stdio.h>

__constant__ unsigned meta[1];

__global__ void dkernel(unsigned *data) {
	data[threadIdx.x] = meta[0];
}
__global__ void print(unsigned *data) {
	printf("%d %d\n", threadIdx.x, data[threadIdx.x]);
}
int main() {

	unsigned hmeta = 10;
	cudaMemcpyToSymbol(meta, &hmeta, sizeof(unsigned));
	unsigned *data;
	cudaMalloc(&data, 32 * sizeof(unsigned));
	dkernel<<<1, 32>>>(data);
	cudaDeviceSynchronize();
	print<<<1, 32>>>(data);
	cudaDeviceSynchronize();
	return 0;
}
