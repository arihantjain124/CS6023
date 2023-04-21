#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <numeric>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

//*******************************************
__global__ void init_capacity(int *capacity_per_hour,int *capacity){

  int idx = threadIdx.x;
  int idy = threadIdx.y;
  int blx = blockIdx.x;
  int cap;
  cap = capacity[idy + (blx * gridDim.x)];
  long int id = idx + (idy * blockDim.x) + (blx * gridDim.x);
  capacity_per_hour[id] = cap;
}

__global__ void allot_request(int *facility,int *capacity,int *req_id,int *req_cen,int *req_fac,int *req_start,int *req_slots,int i,int R,int *tot_reqs,int *succ_reqs){

  unsigned int id = threadIdx.x;
  __shared__ unsigned int access_buffer[1024];
  __shared__ unsigned int req_id_buffer[1024];
  __shared__ unsigned int temp2_buffer[1024];
  __shared__ long int temp_buffer[1025];
  __shared__ volatile int size;
  size = 1;


  __syncthreads();
  unsigned int uid = req_cen[id] * 100 + req_fac[id];
  access_buffer[id] = uid;
  req_id_buffer[id] = id;
  


  __syncthreads();
  
  //Sort
  if (threadIdx.x == 0){
      int l1,l2,k,h1,h2,j;
        for(size=1; size < R; size=size*2)
        {
          l1=0;
          k=0;
          while( l1+size < R)
          {
            h1=l1+size-1;
            l2=h1+1;
            h2=l2+size-1;
            if( h2>=R ) 
              h2=R-1;
            i=l1;
            j=l2;
            while(i<=h1 && j<=h2 )
            {
              if( access_buffer[i] <= access_buffer[j] )
              {
                temp2_buffer[k] = req_id_buffer[i];
                temp_buffer[k++]=access_buffer[i++];
              }
              else
              { 
                temp2_buffer[k] = req_id_buffer[j];
                temp_buffer[k++]=access_buffer[j++];
              }
            }
            
            while(i<=h1)
            {
              temp2_buffer[k] = req_id_buffer[i];
              temp_buffer[k++]=access_buffer[i++];

            }
            while(j<=h2)
            {
              temp2_buffer[k] = req_id_buffer[j];
              temp_buffer[k++]=access_buffer[j++];
            }
            l1=h2+1; 
          }
          for(i=l1; k<R; i++) 
          {
            temp2_buffer[k] = req_id_buffer[i];
            temp_buffer[k++]=access_buffer[i];

          }

          for(i=0;i<R;i++)
          {
            req_id_buffer[i] = temp2_buffer[i];
            access_buffer[i] = temp_buffer[i];
          }
        }
  }

  __syncthreads();

  temp_buffer[id+1] = access_buffer[id];
  
  if(threadIdx.x == 0){
    temp_buffer[0] = -1;
  
  }

  temp2_buffer[id] = 0 ;

  __syncthreads();

    bool flag = temp_buffer[id+1]==temp_buffer[id];
  if(!flag)
  {
    temp2_buffer[id] = 1;
  }
  // __syncthreads();
  // if(threadIdx.x ==0)
  //   {
  //     for(int l =0;l<R;l++)
  //       printf("%d,%d,%d,%d\n\n",access_buffer[l],req_id_buffer[l],l,temp2_buffer[l]);
  //   }
    
  __syncthreads();
  if(!flag)
  {
    unsigned int curr_req;
   for(int j=id;;){
    curr_req = req_id_buffer[j];
    int start_slot = req_start[curr_req]-1;
    int end_slot = start_slot + req_slots[curr_req];

    bool pos = true;
    unsigned int base_index;
    unsigned cen = req_cen[curr_req];
    atomicAdd((unsigned *)&tot_reqs[cen],1);
    if(cen == 0){
        base_index = req_fac[curr_req] * 24;
    }
    else{
        int temp = cen-1;
        base_index = (req_fac[curr_req] + facility[temp]) * 24;
    }
    
    for(i=start_slot;i<end_slot;i++){
      if(capacity[base_index + i]==0)
        pos = false;
    }
    if(pos == true){
      for(i=start_slot;i<end_slot;i++){
        atomicAdd((int*)&capacity[base_index + i],-1);
        // capacity[base_index + i]-=1;
        }
      atomicAdd((unsigned *)&succ_reqs[cen],1);
      }
      
    // if(cen == 2 && req_fac[curr_req] == 1){
    //   printf("\ns=%d:e=%d:fac=%d:req=%d:sta=%d\n",start_slot,end_slot,req_fac[curr_req],curr_req,pos);
    //   // for(int g=0;g<24;g++){
    //   //   printf("%d ",g);
    //   // }
    //   // printf("\n");
    //   for(int g=0;g<24;g++){
    //     // if(g>10){
    //     //   printf(" ");
    //     // }
    //     printf("%d ",capacity[base_index + g]);
    //   }
    //   printf("\n");
    // }
    
    j++;
    if(temp2_buffer[j]==1 || j>R-1){
      // printf("break %d ,%d\n",j,id);
      break;
    }
    __threadfence();
    }
   }
  
  __syncthreads();
}

//***********************************************


int main(int argc,char **argv)
{
	// variable declarations...
    int N,*centre,*facility,*capacity,*fac_ids, *succ_reqs, *tot_reqs;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    centre=(int*)malloc(N * sizeof (int));  // Computer  centre numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer centre
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer centre
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer centre 
   

    int success=0;  // total successful requests
    int fail = 0;   // total failed requests
    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each centre
    succ_reqs = (int *)malloc(N*sizeof(int)); // total successful requests for each centre

    // Input the computer centres data
    int k1=0 , k2 = 0;
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[k1] );
        k1++;
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[k2]);
        k2++;     
      }
    }

    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots;   // Number of slots requested for every request
    
    // Allocate memory on CPU 
	int R;
	fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ( (R) * sizeof (int) );  // Request ids
    req_cen = (int *) malloc ( (R) * sizeof (int) );  // Requested computer centre
    req_fac = (int *) malloc ( (R) * sizeof (int) );  // Requested facility
    req_start = (int *) malloc ( (R) * sizeof (int) );  // Start slot of every request
    req_slots = (int *) malloc ( (R) * sizeof (int) );   // Number of slots requested for every request
    
    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;  
    }

    //*********************************
    // Call the kernels here
    int temp = 0;
    for(int i=0;i<N;i++)
    {
      temp = temp + facility[i];
      facility[i] = temp;
      
    }
    // variable declarations...
    int *d_centre,*d_facility,*d_capacity,*d_fac_ids,*d_succ_reqs,*d_tot_reqs,*d_req_id,*d_req_cen,*d_req_fac,*d_req_start,*d_req_slots;
    int *capacity_per_hour;

    // Allocate memory on GPU 
    cudaMalloc( &d_req_id   , (R) * sizeof (int) );
    cudaMalloc( &d_req_cen  , (R) * sizeof (int) );
    cudaMalloc( &d_req_fac  , (R) * sizeof (int) );
    cudaMalloc( &d_req_start, (R) * sizeof (int) );
    cudaMalloc( &d_req_slots, (R) * sizeof (int) );
    
    cudaMalloc( &d_centre    , N * sizeof (int)); 
    cudaMalloc( &d_facility  , N * sizeof (int)); 
    cudaMalloc( &d_capacity  , max_P * N  * sizeof (int));
    cudaMalloc( &d_fac_ids   , max_P * N  * sizeof (int));
    cudaMalloc( &capacity_per_hour  , facility[N-1] * N * 24 * sizeof (int));
    cudaMalloc( &d_succ_reqs , N*sizeof(int)); 
    cudaMalloc( &d_tot_reqs  , N*sizeof(int)); 

    // Transferring Centre Details all at once
    cudaMemcpy(d_centre   , centre   , N * sizeof (int)  , cudaMemcpyHostToDevice);
    cudaMemcpy(d_facility , facility , N * sizeof (int)  , cudaMemcpyHostToDevice);
    cudaMemcpy(d_capacity , capacity , max_P * N  * sizeof (int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fac_ids  , fac_ids  , max_P * N  * sizeof (int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_succ_reqs, succ_reqs, N*sizeof(int) , cudaMemcpyHostToDevice)
    // cudaMemcpy(d_tot_reqs , tot_reqs , N*sizeof(int) , cudaMemcpyHostToDevice)
    
    long int  gridDimx = ceil(float(N) / 40);
    long int  blockDimy = facility[N-1] % 40;
    dim3 grid3(gridDimx,1,1);
    dim3 block3(24,blockDimy,1);
    // printf("herehe %ld %ld ",gridDimx,blockDimy);
    init_capacity<<<grid3,block3>>>(capacity_per_hour,d_capacity);
    cudaDeviceSynchronize();

    // Transferring Request in a batch of 1024
    unsigned int i = 0;
    unsigned long int req_per_iter = BLOCKSIZE * (sizeof(int));
    long int max_iter =  ceil(float(R)/BLOCKSIZE);
    // printf("number of iteration required %ld \n",max_iter);

    if (max_iter>1){
      // printf("Byte Transfer per cycle %ld \n",req_per_iter);
      cudaMemcpy(d_req_id    , req_id     , req_per_iter, cudaMemcpyHostToDevice);
      cudaMemcpy(d_req_cen   , req_cen    , req_per_iter, cudaMemcpyHostToDevice);
      cudaMemcpy(d_req_fac   , req_fac    , req_per_iter, cudaMemcpyHostToDevice);
      cudaMemcpy(d_req_start , req_start  , req_per_iter, cudaMemcpyHostToDevice);
      cudaMemcpy(d_req_slots , req_slots  , req_per_iter, cudaMemcpyHostToDevice);
    }
    else{
      cudaMemcpy(d_req_id    , req_id     , (R) * sizeof (int) , cudaMemcpyHostToDevice);
      cudaMemcpy(d_req_cen   , req_cen    , (R) * sizeof (int) , cudaMemcpyHostToDevice);
      cudaMemcpy(d_req_fac   , req_fac    , (R) * sizeof (int) , cudaMemcpyHostToDevice);
      cudaMemcpy(d_req_start , req_start  , (R) * sizeof (int) , cudaMemcpyHostToDevice);
      cudaMemcpy(d_req_slots , req_slots  , (R) * sizeof (int) , cudaMemcpyHostToDevice);
    }
    
    // printf("%d,%d \n",sizeof(int),sizeof(unsigned int));

    for ( i = 1 ;i<max_iter; i++){

      cudaMemcpyAsync(d_req_id    + (i * req_per_iter), req_id    + (i * req_per_iter) , req_per_iter, cudaMemcpyHostToDevice);
      cudaMemcpyAsync(d_req_cen   + (i * req_per_iter), req_cen   + (i * req_per_iter) , req_per_iter, cudaMemcpyHostToDevice);
      cudaMemcpyAsync(d_req_fac   + (i * req_per_iter), req_fac   + (i * req_per_iter) , req_per_iter, cudaMemcpyHostToDevice);
      cudaMemcpyAsync(d_req_start + (i * req_per_iter), req_start + (i * req_per_iter) , req_per_iter, cudaMemcpyHostToDevice);
      cudaMemcpyAsync(d_req_slots + (i * req_per_iter), req_slots + (i * req_per_iter) , req_per_iter, cudaMemcpyHostToDevice);

      allot_request<<<1,BLOCKSIZE>>>(d_facility,capacity_per_hour,d_req_id,d_req_cen,d_req_fac,d_req_start,d_req_slots,i,R,d_tot_reqs,d_succ_reqs);
      
      cudaDeviceSynchronize();


    }

    allot_request<<<1,R>>>(d_facility,capacity_per_hour,d_req_id,d_req_cen,d_req_fac,d_req_start,d_req_slots,i,R,d_tot_reqs,d_succ_reqs);
    cudaDeviceSynchronize();
    // printf("Total Request %d\n",tot_reqs);
    cudaMemcpy(tot_reqs , d_tot_reqs , N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(succ_reqs, d_succ_reqs, N * sizeof(int), cudaMemcpyDeviceToHost);
    int total = std::accumulate(tot_reqs , tot_reqs + N , 0);
    success = std::accumulate(succ_reqs , succ_reqs + N , 0);
    fail = total - success;
    //********************************

    // Output
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    fprintf( outputfilepointer, "%d %d\n", success, fail);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j]-succ_reqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	return 0;
}