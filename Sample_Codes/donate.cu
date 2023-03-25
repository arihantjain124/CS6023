#include <stdio.h>
#include <cuda.h>

#define N		500
#define BLOCKSIZE	64
#define ELEPERTHREAD	20


__device__ const unsigned delta = ELEPERTHREAD / 5;

__global__ void k1(unsigned *nelements) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ unsigned sum;
	__shared__ unsigned avg;
	__shared__ unsigned donationbox[N], donationboxindex;

	if (id == 0) {
		sum = 0;
		donationboxindex = 0;
	}
	__syncthreads();
	
	// compute sum.
	atomicAdd(&sum, nelements[id]);
	__syncthreads();

	// compute average.
	if (id == 0) avg = sum / blockDim.x;
	__syncthreads();

	// check if I need to donate.
	unsigned surplus = nelements[id] - avg;
	if (surplus > delta) {
		// donate.
		unsigned index = atomicAdd(&donationboxindex, surplus);
		for (unsigned ii = 0; ii < surplus; ++ii) {
			donationbox[index + ii] = id;	// some work.
		}
	}

	// process.
	// some processing here.
	//__syncthreads();	// this is wrong.

	// empty donation box.
	while (donationboxindex < N * ELEPERTHREAD) {
		unsigned index = atomicDec(&donationboxindex, N * ELEPERTHREAD + blockDim.x);	// to ensure that wrap-around does not cause confusion.
		if (index < N * ELEPERTHREAD) {
			unsigned work = donationbox[index];
			// process with work.
		}
	}
}

int main() {
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
	//k2<<<1, 1>>>();
	//cudaDeviceSynchronize();

	return 0;
}
