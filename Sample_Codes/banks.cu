#include <stdio.h>
#include <cuda.h>
#include "mytime.h"


__global__ void bankcheck() {
	__shared__ unsigned s[1024];
	s[1 * threadIdx.x] = threadIdx.x;
}
__global__ void bankcheck2() {
	__shared__ unsigned s[1024];
	s[32 * threadIdx.x] = threadIdx.x;
}
int main() {
	int ii;
	double start, end;

	bankcheck<<<1, 32>>>();	// dummy for warmup.
	cudaDeviceSynchronize();

	start = rtclock();
	for (ii = 0; ii < 1000; ++ii) {
		bankcheck<<<1, 32>>>();
		cudaDeviceSynchronize();
	}
	end = rtclock();
	printtime("bank consecutive: ", start, end);

	start = rtclock();
	for (ii = 0; ii < 1000; ++ii) {
		bankcheck2<<<1, 32>>>();
		cudaDeviceSynchronize();
	}
	end = rtclock();
	printtime("bank strided: ", start, end);
	return 0;
}
