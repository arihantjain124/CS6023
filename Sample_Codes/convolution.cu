#include <stdio.h>
#include <cuda.h>


#define N		100
#define	BLOCKSIZE	32

__global__ void init(int *input) {
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < N) input[id] = id + 1;
}
__global__ void print(int *output) {
	for (unsigned ii = 0; ii < N; ++ii)
		printf("%d ", output[ii]);
	printf("\n");
}
__global__ void convolution(int *input, int *filter, int *output, int fsize) {
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= N) return;

	//int *filteroutput = (int *)malloc(fsize * sizeof(int));
	int sum = 0;
	int halff = fsize / 2;
	int istart = id - halff, iend = id + halff + 1;
	int fstart = 0, fend = fsize;

	if (istart < 0) {
		fstart -= istart;
		istart = 0;
	}
	if (iend > N) {
		fend -= (iend - N);
		iend = N;
	}
	
	for (unsigned ii = fstart; ii < fend; ++ii) {
		// filteroutput[ii] = input[id + ii] * filter[ii];
		sum += input[istart + ii - fstart] * filter[ii];
	}
	output[id] = sum;
}
int main() {
	int *input, *filter, *output;
	int hf[] = {3, 4, 5, 4, 3};
	int fsize = sizeof(hf) / sizeof(*hf);

	if (fsize % 2 == 0) {
		printf("Error: Filter size (%d) is even.\n", fsize);
		exit(1);
	}
	cudaMalloc(&input, N * sizeof(int));
	cudaMalloc(&filter, fsize * sizeof(int));
	cudaMalloc(&output, N * sizeof(int));

	cudaMemcpy(filter, hf, fsize * sizeof(int), cudaMemcpyHostToDevice);
	
	int nblocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;
	init<<<nblocks, BLOCKSIZE>>>(input);

	convolution<<<nblocks, BLOCKSIZE>>>(input, filter, output, fsize);

	print<<<1, 1>>>(output);
	cudaDeviceSynchronize();

	return 0;
}
