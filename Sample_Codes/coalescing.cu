#include <stdio.h>
#include <cuda.h>
#define N 1024
__global__ void dkernel(unsigned *a, unsigned chunksize) {
	unsigned start = chunksize * threadIdx.x;
	for (unsigned nn = start; nn < start + chunksize; ++nn) {
		a[nn]++;
	}
}
int main() {
	unsigned *a, chunksize = 32;
	cudaMalloc(&a, sizeof(unsigned) * N);
	dkernel<<<1, N/chunksize>>>(a, chunksize);
	cudaDeviceSynchronize();
    return 0;
}
