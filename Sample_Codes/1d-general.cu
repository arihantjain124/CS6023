#include <stdio.h>
#include <cuda.h>
__global__ void dkernel(unsigned *vector, unsigned vectorsize) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < vectorsize) vector[id] = id;
}
#define BLOCKSIZE	1024
int main(int nn, char *str[]) {
	unsigned N = atoi(str[1]);
	unsigned *vector, *hvector;
	cudaMalloc(&vector, N * sizeof(unsigned));
	hvector = (unsigned *)malloc(N * sizeof(unsigned));

	unsigned nblocks = ceil((float)N / BLOCKSIZE);
	printf("nblocks = %d\n", nblocks);

    	dkernel<<<nblocks, BLOCKSIZE>>>(vector, N);
	cudaMemcpy(hvector, vector, N * sizeof(unsigned), cudaMemcpyDeviceToHost);
	for (unsigned ii = 0; ii < N; ++ii) {
		printf("%4d ", hvector[ii]);
		if (ii % 1000 == 0) printf("\n");
	}
    	return 0;
}
