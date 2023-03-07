#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;


__global__ void computeKernel(int p, int q, int r, int *A, int *B, 
	         int *C, int *D, int *E){

    __shared__ int As[32][32];
    __shared__ int Bs[32][32];
    __shared__ int Cs[32][32];
    __shared__ int Ds[32][32];

    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;

    int temp = 0;
	int iter = ceil(((float(q)) / 32));
    for (int i = 0; i < iter ; ++i) 
	{
		int idx = i * 32 + threadIdx.x;

        if (row < p && idx < q) {
            As[threadIdx.y][threadIdx.x] = A[(row * q) + idx];
            Cs[threadIdx.y][threadIdx.x] = C[(row * q) + idx];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
            Cs[threadIdx.y][threadIdx.x] = 0.0;
        }

		idx = i * 32 + threadIdx.y;

        if (col < r &&  idx < q) {
            Bs[threadIdx.y][threadIdx.x] = B[(idx) * r + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }
		
		idx = i * 32 + threadIdx.x;

		if ((blockIdx.x * 32 + threadIdx.y) < r && idx < q) {
            Ds[threadIdx.x][threadIdx.y] = D[(blockIdx.x * 32 + threadIdx.y)*q + idx];
        } else {
            Ds[threadIdx.x][threadIdx.y] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < 32; ++j) {
            temp += (As[threadIdx.y][j] * Bs[j][threadIdx.x]) + (Cs[threadIdx.y][j] * Ds[j][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < p && col < r) {
        E[row * r + col] = temp ;
    }
}




// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */
	dim3 blockDim(32, 32);
	dim3 gridDim((r + blockDim.x - 1) / blockDim.x, (p + blockDim.y - 1) / blockDim.y);

	computeKernel<<<gridDim,blockDim>>>(p,q,r,d_matrixA,d_matrixB,d_matrixC,d_matrixD,d_matrixE);
	cudaDeviceSynchronize();
	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}


// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	
