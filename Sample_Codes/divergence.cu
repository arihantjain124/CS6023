#include <stdio.h>
#include <cuda.h>
__global__ void dkernel(unsigned *vector, unsigned vectorsize) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id % 2) vector[id] = id;
	else vector[id] = vectorsize * vectorsize;
}
#define BLOCKSIZE	10
#define N		BLOCKSIZE
int main(int nn, char *str[]) {
	unsigned *vector, *hvector;
	cudaMalloc(&vector, N * sizeof(unsigned));
	hvector = (unsigned *)malloc(N * sizeof(unsigned));

	unsigned nblocks = ceil((float)N / BLOCKSIZE);

    	dkernel<<<nblocks, BLOCKSIZE>>>(vector, N);
	cudaMemcpy(hvector, vector, N * sizeof(unsigned), cudaMemcpyDeviceToHost);
	for (unsigned ii = 0; ii < N; ++ii) {
		printf("%4d ", hvector[ii]);
	}
	printf("\n");
    	return 0;
}
