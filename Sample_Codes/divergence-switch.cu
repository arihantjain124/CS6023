#include <stdio.h>
#include <cuda.h>
__global__ void dkernel(unsigned *vector, unsigned vectorsize) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	switch(id) {
	case 0: vector[id] = 0; break;
	case 1: vector[id] = vector[id]; break;
	case 2: vector[id] = vector[id - 2]; break;
	case 3: vector[id] = vector[id + 3]; break;
	case 4: vector[id] = 4 + 4 + vector[id]; break;
	case 5: vector[id] = 5 - vector[id]; break;
	case 6: vector[id] = vector[6]; break;
	case 7: vector[id] = 7 + 7; break;
	case 8: vector[id] = vector[id] + 8; break;
	case 9: vector[id] = vector[id] * 9; break;
	}
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
