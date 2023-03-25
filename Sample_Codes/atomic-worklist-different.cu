#include <stdio.h>
#include <cuda.h>

#define N		500
#define BLOCKSIZE	64
#define ELEPERTHREAD	5

__device__ unsigned wlsize;
__device__ unsigned worklist[N * ELEPERTHREAD];

__global__ void k1(unsigned *nelements) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned index = atomicAdd(&wlsize, nelements[id]);
	for (unsigned ii = 0; ii < nelements[id]; ++ii)
		worklist[index + ii] = id;
}
__global__ void k2() {
	printf("Number of threads = %d, worklist size = %d\n", N, wlsize);

	for (unsigned ii = 0; ii < wlsize; ++ii)
		printf("%d ", worklist[ii]);
	printf("\n");
}
int main() {
	cudaMemset(&wlsize, 0, sizeof(unsigned));	// initialization.

	unsigned hnelements[N];
	for (unsigned ii = 0; ii < N; ++ii) {
		hnelements[ii] = rand() % ELEPERTHREAD;
	}

	unsigned *nelements;
	cudaMalloc(&nelements, N * sizeof(unsigned));
	cudaMemcpy(nelements, hnelements, N * sizeof(unsigned), cudaMemcpyHostToDevice);

	unsigned nblocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;
	k1<<<nblocks, BLOCKSIZE>>>(nelements);
	cudaDeviceSynchronize();
	k2<<<1, 1>>>();
	cudaDeviceSynchronize();

	return 0;
}
