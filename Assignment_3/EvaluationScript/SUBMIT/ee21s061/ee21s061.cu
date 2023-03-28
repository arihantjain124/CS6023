/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;

__device__ unsigned int level = 0 ;
__device__ unsigned int node_counter = 0;
__device__ unsigned int last_thread_id;
__device__ unsigned int last_node_per_level = 0;
ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/


__global__ void nodes_per_level(int *csrList,int *offset,int *apr,int *aid,int V,int L, int *nodesper_level){
    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    for(int j=0;j<L;j++){
        // printf("%d:%d:%d\n",level,apr[id],id);
        if( ((apr[id] == 0 && level == 0) || (level > 0  && id>=node_counter && id<=last_node_per_level) ) && id<V  ){
            // printf("%d:%d:%d:%d\n",level,apr[id],id,node_counter);
            int start,end;
            start = offset[id];
            end = offset[id+1];
            for(int i =start;i<end;i++){
                int curr_edge = csrList[i];
                unsigned temp;
                if(aid[id]>=apr[id]){
                    temp = atomicAdd(&aid[curr_edge],1);
                }
                temp = atomicMax(&last_node_per_level,curr_edge);
                }
            unsigned temp = atomicAdd(&node_counter,1);
            }
        unsigned temp = atomicExch(&last_thread_id,id);
        __syncthreads();
        if(last_thread_id == id){
            unsigned temp = atomicExch(&nodesper_level[level+1],node_counter);
            // printf("%d:%d:%d:pl\n",nodesper_level[level+1],level);
            level+=1;
        }
        __syncthreads();
    }
}

__global__ void active_vertex_perlevel(int *csrList,int *offset,int *apr,int *aid,int V,int L, int *nodesper_level, int *activeVertex){

    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    for(int i=0;i<L;i++){
        while(i>level);
        if( id>=nodesper_level[i] && id<nodesper_level[i+1]  && id<V  ){
            if(aid[id]>=apr[id])
            {
                if(id>nodesper_level[i] && id<nodesper_level[i+1]-1 && aid[id-1]<apr[id-1] && aid[id+1]<apr[id+1]){
                    int start,end;
                    start = offset[id];
                    end = offset[id+1];
                    for(int i =start;i<end;i++){
                        int curr_edge = csrList[i];
                        unsigned temp = atomicAdd(&aid[curr_edge],-1);
                    }
                }
                else{
                    unsigned temp = atomicAdd(&activeVertex[i],1);
                }
            }
            
        }
        unsigned temp = atomicExch(&last_thread_id,id);
        __syncthreads();
        if(last_thread_id == id){
            // printf("%d:%d:av\n",activeVertex[i],i);
        }
        __syncthreads();
    }
}

    
    
    
    
/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    // Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    int *d_activeVertex;
	cudaMalloc(&d_activeVertex, L*sizeof(int));
    

    cudaMemset(d_aid, 0, V*sizeof(int));

/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/

int *d_nodesper_level;
cudaMalloc(&d_nodesper_level, (L+1)*sizeof(int));
cudaMemset(d_nodesper_level, 0, (L+1)*sizeof(int));

int num_threads = 10000;

if(V<10000)
{
    num_threads = V;
}

long int gridDimx = ceil(float(num_threads)/1024);
long int threadDimx = 1024;
printf("%ld",gridDimx);
nodes_per_level<<<gridDimx,threadDimx>>>(d_csrList,d_offset,d_apr,d_aid,V,L,d_nodesper_level);
// cudaDeviceSynchronize();
active_vertex_perlevel<<<gridDimx,threadDimx>>>(d_csrList,d_offset,d_apr,d_aid,V,L,d_nodesper_level,d_activeVertex);
cudaDeviceSynchronize();

 

    
   
    
    

     
cudaMemcpy(h_activeVertex, d_activeVertex, L*sizeof(int), cudaMemcpyDeviceToHost);

/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}
