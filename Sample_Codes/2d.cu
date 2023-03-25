#include <stdio.h>
#include <cuda.h>
__global__ void dkernel(unsigned *matrix) {
	unsigned id = threadIdx.x * blockDim.y + threadIdx.y;
	matrix[id] = id;
}
#define N	5
#define M	6
int main() {
	dim3 block(N, M, 1);
	unsigned *matrix, *hmatrix;
	cudaMalloc(&matrix, N * M * sizeof(unsigned));
	hmatrix = (unsigned *)malloc(N * M * sizeof(unsigned));
    	dkernel<<<1, block>>>(matrix);
	cudaMemcpy(hmatrix, matrix, N * M * sizeof(unsigned), cudaMemcpyDeviceToHost);
	for (unsigned ii = 0; ii < N; ++ii) {
		for (unsigned jj = 0; jj < M; ++jj) {
			printf("%2d ", hmatrix[ii * M + jj]);
		}
		printf("\n");
	}
    return 0;
}
