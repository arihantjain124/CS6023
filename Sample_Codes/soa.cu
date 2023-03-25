#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>
#define N 1024
struct nodeAOS {
	int a;
	double b;
	char c;
} *allnodesAOS;
struct nodeSOA {
	int *a;
	double *b;
	char *c;
} allnodesSOA;
__global__ void dkernelaos(struct nodeAOS *allnodesAOS) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	allnodesAOS[id].a = id;
	allnodesAOS[id].b = 0.0;
	allnodesAOS[id].c = 'c';
}
__global__ void dkernelsoa(int *a, double *b, char *c) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	a[id] = id;
	b[id] = 0.0;
	c[id] = 'd';
}

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d", stat);
  return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}
void printtime(const char *str, double starttime, double endtime) {
	printf("%s%3f seconds\n", str, endtime - starttime);
}

#define BLOCKSIZE	1024
int main(int nn, char *str[]) {
	cudaMalloc(&allnodesAOS, N * sizeof(struct nodeAOS));
	cudaMalloc(&allnodesSOA.a, N * sizeof(int));
	cudaMalloc(&allnodesSOA.b, N * sizeof(double));
	cudaMalloc(&allnodesSOA.c, N * sizeof(char));

	unsigned nblocks = ceil((float)N / BLOCKSIZE);

	double starttime = rtclock();
    	dkernelaos<<<nblocks, BLOCKSIZE>>>(allnodesAOS);
	cudaThreadSynchronize();
	double endtime = rtclock();
	printtime("AoS time: ", starttime, endtime);

	starttime = rtclock();
    	dkernelsoa<<<nblocks, BLOCKSIZE>>>(allnodesSOA.a, allnodesSOA.b, allnodesSOA.c);
	cudaThreadSynchronize();
	endtime = rtclock();
	printtime("SoA time: ", starttime, endtime);

    	return 0;
}
