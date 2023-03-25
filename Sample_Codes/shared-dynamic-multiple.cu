#include <stdio.h>
#include <cuda.h>

__global__ void dynshared(int sz, int n1) {
	extern __shared__ int s[];
	int *s1 = s;
	int *s2 = s + n1;
	if (threadIdx.x < n1) s1[threadIdx.x] = threadIdx.x;
	if (threadIdx.x < (sz - n1)) s2[threadIdx.x] = threadIdx.x * 100;
	__syncthreads();
	if (threadIdx.x < sz && threadIdx.x % 2) printf("%d\n", s1[threadIdx.x]);
}
int main() {
	int sz;
	scanf("%d", &sz);
	dynshared<<<1, 1024, sz * sizeof(int)>>>(sz, sz / 2);
	cudaDeviceSynchronize();

	return 0;
}

