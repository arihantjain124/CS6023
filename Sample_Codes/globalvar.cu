#include <stdio.h>
#include <cuda.h>
unsigned int N = 32;

__global__ void dkernel(int m) {
	unsigned id = threadIdx.x;
    printf("id = %d, N = %d, m = %d.\n", N, m);
}
int main() {
	unsigned id = 1;
	dkernel<<<N, id>>>(N);
	cudaDeviceSynchronize();
    return 0;
}
