#include <stdio.h>
#include <cuda.h>

#define N	64
__global__ void exscan() {
	__shared__ unsigned a[N]; //= {4, 3, 9, 3, 5, 7, 3, 2};
	a[threadIdx.x] = threadIdx.x;
	__syncthreads();

	unsigned n = sizeof(a) / sizeof (*a);
	__syncthreads();

	if (threadIdx.x == 0) {
		for (unsigned ii = 0; ii < n; ++ii)
			printf("%d ", a[ii]);
		printf("\n");
	}
	__syncthreads();
	
	int tmp;
	for (int off = 1; off < n; off *= 2) {
		if (threadIdx.x >= off) {
			tmp = a[threadIdx.x - off];
		}
		__syncthreads();
		if (threadIdx.x >= off) {
			a[threadIdx.x] += tmp;
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		for (unsigned ii = 0; ii < n; ++ii)
			printf("%d ", a[ii]);
		printf("\n");
	}
}

int main() {
	//cudaSetDevice(5);
	exscan<<<1, N>>>();
	cudaThreadSynchronize();
	return 0;
}

