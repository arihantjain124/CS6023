#include <cuda.h>
#include <stdio.h>

__global__ void K(int *p) {
	*p = 0;
	printf("%d\n", *p);
}
int main() {
	int *x, *y;
	cudaMalloc(&x, sizeof(int));

	K<<<2, 10>>>(x);
	cudaDeviceSynchronize();

	y = x;
	cudaFree(y);

	K<<<2, 10>>>(x);
	cudaDeviceSynchronize();
	//cudaError_t err = cudaGetLastError();
	//printf("error=%d, %s, %s\n", err, cudaGetErrorName(err), cudaGetErrorString(err));

	return 0;
}
