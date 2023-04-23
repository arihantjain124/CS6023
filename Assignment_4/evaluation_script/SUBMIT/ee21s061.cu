#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <numeric>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

//*******************************************
__global__ void init_capacity(int *capacity_per_hour, int *capacity, int total_fac)
{

  unsigned int idx = threadIdx.x;
  unsigned int idy = threadIdx.y;
  unsigned int blx = blockIdx.x;
  unsigned int cap;
  unsigned int curr_fac = idy + (blx * blockDim.y);
  cap = capacity[curr_fac];
  if (curr_fac < total_fac)
  {
    unsigned int id = idx + curr_fac * blockDim.x;
    capacity_per_hour[id] = cap;
  }
  __syncthreads();
}

__global__ void allot_request(int *facility, int *capacity, int *req_id, int *req_cen, int *req_fac, int *req_start, int *req_slots, int offset, int R, int *tot_reqs, int *succ_reqs)
{

  unsigned int id = threadIdx.x;
  __shared__ unsigned int access_buffer[BLOCKSIZE];
  __shared__ unsigned int req_id_buffer[BLOCKSIZE];
  __shared__ unsigned int temp2_buffer[BLOCKSIZE];
  __shared__ long int temp_buffer[BLOCKSIZE + 1];
  __shared__ volatile int size;
  __shared__ unsigned int curr_offset;
  __shared__ unsigned int counter;
  counter = 0;
  size = 1;
  curr_offset = offset * BLOCKSIZE;

  __syncthreads();
  // creating unique ids for each facility based on center and facility id of current request
  unsigned int uid = req_cen[curr_offset + id] * 100 + req_fac[curr_offset + id];
  access_buffer[id] = uid;
  req_id_buffer[id] = curr_offset + id;
  // allocating 2 bufffers to store the unique id along with its request id.
  __syncthreads();

  // Sorting the access buffer to bundle up the conflicting resouce access and this sort is applied to request ids to allow 
  // the program to find non conflicting request access and process them parallely.

  // This sort is being done by 1 thread because of inconsistency i was facing with syncronization even after consulation with sir.
  if (threadIdx.x == 0)
  {
    int l1, l2, k, h1, h2, j, i;
    for (size = 1; size < R; size = size * 2)
    {
      l1 = 0;
      k = 0;
      while (l1 + size < R)
      {
        h1 = l1 + size - 1;
        l2 = h1 + 1;
        h2 = l2 + size - 1;
        if (h2 >= R)
          h2 = R - 1;
        i = l1;
        j = l2;
        while (i <= h1 && j <= h2)
        {
          if (access_buffer[i] <= access_buffer[j])
          {
            temp2_buffer[k] = req_id_buffer[i];
            temp_buffer[k++] = access_buffer[i++];
          }
          else
          {
            temp2_buffer[k] = req_id_buffer[j];
            temp_buffer[k++] = access_buffer[j++];
          }
        }

        while (i <= h1)
        {
          temp2_buffer[k] = req_id_buffer[i];
          temp_buffer[k++] = access_buffer[i++];
        }
        while (j <= h2)
        {
          temp2_buffer[k] = req_id_buffer[j];
          temp_buffer[k++] = access_buffer[j++];
        }
        l1 = h2 + 1;
      }
      for (i = l1; k < R; i++)
      {
        temp2_buffer[k] = req_id_buffer[i];
        temp_buffer[k++] = access_buffer[i];
      }

      for (i = 0; i < R; i++)
      {
        req_id_buffer[i] = temp2_buffer[i];
        access_buffer[i] = temp_buffer[i];
      }
    }
  }

  __syncthreads();

  temp_buffer[id + 1] = access_buffer[id];

  if (threadIdx.x == 0)
  {
    temp_buffer[0] = -1;
  }

  temp2_buffer[id] = 0;

  __syncthreads();

  bool flag = temp_buffer[id + 1] == temp_buffer[id];
  // This sets the flag for all the request which are the first request to the paticular access buffer. 
  // This flag is basically request ids locking the resouces before entering the critical section.
  if (!flag)
  {
    temp2_buffer[id] = 1;
    atomicAdd((unsigned *)&counter, 1);
  }
  // if(threadIdx.x == 0){
  //   printf("%d , %d\n" , offset,counter);
  // }
  __syncthreads();
  if (!flag)
  {
    unsigned int curr_req;
    for (int j = id;;)
    {
      // Processing request here..
      
      curr_req = req_id_buffer[j];
      int start_slot = req_start[curr_req] - 1;
      int end_slot = start_slot + req_slots[curr_req];
      // Picking up start and end slot to loop over
      bool pos = true;
      unsigned int base_index;
      unsigned cen = req_cen[curr_req];
      atomicAdd((unsigned *)&tot_reqs[cen], 1);
      if (cen == 0)
      {
        base_index = req_fac[curr_req] * 24;
      }
      else
      {
        int temp = cen - 1;
        base_index = (req_fac[curr_req] + facility[temp]) * 24;
      }
      // Checking if the capacity is available for the entire request time requirement.
      for (int i = start_slot; i < end_slot; i++)
      {
        if (capacity[base_index + i] == 0)
          pos = false;
      }
      // if pos is still true then request can be accomadated 
      // The next step basically says this request was succesful and before releasing the lock it reduces the capacity
      //  to reflect its occupation of this slot.
      if (pos == true)
      {
        for (int i = start_slot; i < end_slot; i++)
        {
          atomicAdd((int *)&capacity[base_index + i], -1);
        }
        atomicAdd((unsigned *)&succ_reqs[cen], 1);
      }

      j++;
      if (temp2_buffer[j] == 1 || j > R - 1)
      {
        // If the loop moves beyond the request ids or reach the point in buffer 
        // where the next request is already being processed by another thread then it breaks the execution.
        break;
      }
      __threadfence();
    }
  }

  __syncthreads();
}

//***********************************************

int main(int argc, char **argv)
{
  // variable declarations...
  int N, *centre, *facility, *capacity, *fac_ids, *succ_reqs, *tot_reqs;

  FILE *inputfilepointer;

  // File Opening for read
  char *inputfilename = argv[1];
  inputfilepointer = fopen(inputfilename, "r");

  if (inputfilepointer == NULL)
  {
    printf("input.txt file failed to open.");
    return 0;
  }

  fscanf(inputfilepointer, "%d", &N); // N is number of centres

  // Allocate memory on cpu
  centre = (int *)malloc(N * sizeof(int));           // Computer  centre numbers
  facility = (int *)malloc(N * sizeof(int));         // Number of facilities in each computer centre
  fac_ids = (int *)malloc(max_P * N * sizeof(int));  // Facility room numbers of each computer centre
  capacity = (int *)malloc(max_P * N * sizeof(int)); // stores capacities of each facility for every computer centre

  int success = 0;                            // total successful requests
  int fail = 0;                               // total failed requests
  tot_reqs = (int *)malloc(N * sizeof(int));  // total requests for each centre
  succ_reqs = (int *)malloc(N * sizeof(int)); // total successful requests for each centre

  // Input the computer centres data
  int k1 = 0, k2 = 0;
  for (int i = 0; i < N; i++)
  {
    fscanf(inputfilepointer, "%d", &centre[i]);
    fscanf(inputfilepointer, "%d", &facility[i]);

    for (int j = 0; j < facility[i]; j++)
    {
      fscanf(inputfilepointer, "%d", &fac_ids[k1]);
      k1++;
    }
    for (int j = 0; j < facility[i]; j++)
    {
      fscanf(inputfilepointer, "%d", &capacity[k2]);
      k2++;
    }
  }

  // variable declarations
  int *req_id, *req_cen, *req_fac, *req_start, *req_slots; // Number of slots requested for every request

  // Allocate memory on CPU
  int R;
  fscanf(inputfilepointer, "%d", &R);           // Total requests
  req_id = (int *)malloc((R) * sizeof(int));    // Request ids
  req_cen = (int *)malloc((R) * sizeof(int));   // Requested computer centre
  req_fac = (int *)malloc((R) * sizeof(int));   // Requested facility
  req_start = (int *)malloc((R) * sizeof(int)); // Start slot of every request
  req_slots = (int *)malloc((R) * sizeof(int)); // Number of slots requested for every request

  // Input the user request data
  for (int j = 0; j < R; j++)
  {
    fscanf(inputfilepointer, "%d", &req_id[j]);
    fscanf(inputfilepointer, "%d", &req_cen[j]);
    fscanf(inputfilepointer, "%d", &req_fac[j]);
    fscanf(inputfilepointer, "%d", &req_start[j]);
    fscanf(inputfilepointer, "%d", &req_slots[j]);
    tot_reqs[req_cen[j]] += 1;
  }

  //*********************************
  // Call the kernels here
  // Doing a prefix sum on facility number to find the total number of facility in given testcase
  int temp = 0;
  for (int i = 0; i < N; i++)
  {
    temp = temp + facility[i];
    facility[i] = temp;
  }
  // variable declarations...
  int *d_centre, *d_facility, *d_capacity, *d_fac_ids, *d_succ_reqs, *d_tot_reqs, *d_req_id, *d_req_cen, *d_req_fac, *d_req_start, *d_req_slots;
  int *capacity_per_hour;

  // Allocate memory on GPU
  cudaMalloc(&d_req_id, (R) * sizeof(int));
  cudaMalloc(&d_req_cen, (R) * sizeof(int));
  cudaMalloc(&d_req_fac, (R) * sizeof(int));
  cudaMalloc(&d_req_start, (R) * sizeof(int));
  cudaMalloc(&d_req_slots, (R) * sizeof(int));

  cudaMalloc(&d_centre, N * sizeof(int));
  cudaMalloc(&d_facility, N * sizeof(int));
  cudaMalloc(&d_capacity, max_P * N * sizeof(int));
  cudaMalloc(&d_fac_ids, max_P * N * sizeof(int));
  cudaMalloc(&capacity_per_hour, facility[N - 1] * 24 * sizeof(int));
  cudaMalloc(&d_succ_reqs, N * sizeof(int));
  cudaMalloc(&d_tot_reqs, N * sizeof(int));

  // Transferring Centre Details all at once
  cudaMemcpy(d_centre, centre, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_facility, facility, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_capacity, capacity, max_P * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_fac_ids, fac_ids, max_P * N * sizeof(int), cudaMemcpyHostToDevice);
  // Intializing Capacity array
  long int gridDimx = ceil(float(facility[N - 1]) / 40);
  long int blockDimy = 40;
  dim3 grid3(gridDimx, 1, 1);
  dim3 block3(24, blockDimy, 1);
  // intializing capacity array acc to the test case provided.
  init_capacity<<<grid3, block3>>>(capacity_per_hour, d_capacity, facility[N - 1]);
  cudaDeviceSynchronize();

  // Transferring Request in a batch of 1024
  unsigned int i = 0;
  long int max_iter = ceil(float(R) / BLOCKSIZE);
  // max_iter is the number of times we call kernel to process all request.
  // This means there is exactly one thread for each request and in one batch we pass 1024 request to the kernel
  // Transferring All Request
  cudaMemcpy(d_req_id, req_id, (R) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_req_cen, req_cen, (R) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_req_fac, req_fac, (R) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_req_start, req_start, (R) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_req_slots, req_slots, (R) * sizeof(int), cudaMemcpyHostToDevice);

  int remaining_request = R;
  int curr_request;

  for (i = 0; i < max_iter; i++)
  {

    if (remaining_request > 1024)
    {
      curr_request = BLOCKSIZE;
      remaining_request = remaining_request - BLOCKSIZE;
    }
    else
    {
      curr_request = remaining_request;
    }
    // allot request is launched for each batch of 1024 request which are then processed parallel
    allot_request<<<1, curr_request>>>(d_facility, capacity_per_hour, d_req_id, d_req_cen, d_req_fac, d_req_start, d_req_slots, i, curr_request, d_tot_reqs, d_succ_reqs);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(tot_reqs, d_tot_reqs, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(succ_reqs, d_succ_reqs, N * sizeof(int), cudaMemcpyDeviceToHost);

  int total = std::accumulate(tot_reqs, tot_reqs + N, 0);
  success = std::accumulate(succ_reqs, succ_reqs + N, 0);
  fail = total - success;

  //********************************

  // Output
  char *outputfilename = argv[2];
  FILE *outputfilepointer;
  outputfilepointer = fopen(outputfilename, "w");

  fprintf(outputfilepointer, "%d %d\n", success, fail);
  for (int j = 0; j < N; j++)
  {
    fprintf(outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j] - succ_reqs[j]);
  }
  fclose(inputfilepointer);
  fclose(outputfilepointer);
  cudaDeviceSynchronize();
  return 0;
}