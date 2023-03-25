#include <stdio.h>
#include <cuda.h>

__global__ void dynshared(int sz) {
	extern __shared__ int s[];
	if (threadIdx.x < sz) s[threadIdx.x] = threadIdx.x;
	__syncthreads();
	if (threadIdx.x < sz && threadIdx.x % 2) printf("%d\n", s[threadIdx.x]);
}
int main() {
	int sz;
	scanf("%d", &sz);
	dynshared<<<1, 32, sz * sizeof(int)>>>(sz);
	cudaDeviceSynchronize();

	return 0;
}

