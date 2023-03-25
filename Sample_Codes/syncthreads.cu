#include <stdio.h>
#include <cuda.h>
__global__ void dkernel(unsigned *vector, unsigned vectorsize) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	vector[id] = id;
	__syncthreads();

	if (id < vectorsize - 1 && vector[id + 1] != id + 1) printf("syncthreads does not work.\n");
}
#define BLOCKSIZE	1000
#define N		BLOCKSIZE
int main(int nn, char *str[]) {
	unsigned *vector, *hvector;
	cudaMalloc(&vector, N * sizeof(unsigned));
	hvector = (unsigned *)malloc(N * sizeof(unsigned));

    	dkernel<<<100, BLOCKSIZE>>>(vector, N);
	cudaMemcpy(hvector, vector, N * sizeof(unsigned), cudaMemcpyDeviceToHost);
	for (unsigned ii = 0; ii < N; ++ii) {
		printf("%4d ", hvector[ii]);
	}
	printf("\n");
    	return 0;
}
