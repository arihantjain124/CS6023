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

__device__ volatile unsigned level = 0 ;
__device__ volatile unsigned node_counter = 0;
__device__ volatile unsigned block_inc_pl = 0;
__device__ volatile unsigned block_inc_av = 0;
__device__ volatile unsigned last_node_per_level = 0;
__device__ volatile unsigned last_node_acrross_blocks_pl;
__device__ volatile unsigned last_node_acrross_blocks_av;
ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/


__global__ void nodes_per_level(int *csrList,int *offset,int *apr,int *aid,int V,int L, int *nodesper_level){
    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int level_id;
    for(int j=0;j<L;j++){
        level_id = nodesper_level[level] + id;
        if( ((apr[level_id] == 0 && level == 0) || (level > 0  && level_id>=node_counter && level_id<=last_node_per_level) ) && level_id<V  && (nodesper_level[level]!=0 || level==0) ){
            
            int start,end;
            start = offset[level_id];
            end = offset[level_id+1];
            for(int i =start;i<end;i++){
                int curr_edge = csrList[i];
                if(aid[level_id]>=apr[level_id]){
                    atomicAdd(&aid[curr_edge],1);
                }
                atomicMax((unsigned *)&last_node_per_level,curr_edge);
                }
            atomicAdd((unsigned *)&node_counter,1);
            }

        __syncthreads();
        if(threadIdx.x == 0){
            atomicExch((unsigned *)&last_node_acrross_blocks_pl,id);
            atomicInc((unsigned *)&block_inc_pl,gridDim.x + 1);
            
            while (block_inc_pl != gridDim.x);
            
            if(id == last_node_acrross_blocks_pl)
            {    
                atomicExch((unsigned *)&nodesper_level[level+1],node_counter);
                // if(level >10){
                // printf("%d:%d:%d pl\n",nodesper_level[level+1],level,last_node_per_level);
                // }
                atomicAdd((unsigned *)&level,1);
                atomicExch((unsigned *)&block_inc_pl,0);
            }
            while(block_inc_pl != 0);
            // if(nodesper_level[level] == 0 && level != 0){
            //     printf("%d:%d:%d:%d:%d\n",level,level_id,id,blockIdx.x,last_node_acrross_blocks_pl);
            
            // }
        }
        __syncthreads();
    }
}

__global__ void active_vertex_perlevel(int *csrList,int *offset,int *apr,int *aid,int V,int L, int *nodesper_level, int *activeVertex){

    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int level_id;
    for(int j=0;j<L;j++){
        level_id = nodesper_level[j] + id;
        while(j>level);
        if( level_id>=nodesper_level[j] && level_id<nodesper_level[j+1]  && level_id<V  ){
            if(aid[level_id]>=apr[level_id])
            {
                if(level_id>nodesper_level[j] && level_id<nodesper_level[j+1]-1 && aid[level_id-1]<apr[level_id-1] && aid[level_id+1]<apr[level_id+1]){
                    int start,end;
                    start = offset[level_id];
                    end = offset[level_id+1];
                    for(int i =start;i<end;i++){
                        int curr_edge = csrList[i];
                        atomicAdd(&aid[curr_edge],-1);
                    }
                }
                else{
                    atomicAdd(&activeVertex[j],1);
                }
            }
            
        }

        __syncthreads();
        if(threadIdx.x == 0){
            atomicExch((unsigned *)&last_node_acrross_blocks_av,id);
            atomicInc((unsigned *)&block_inc_av,gridDim.x + 1);
            
            while (block_inc_av != gridDim.x);
            
            if(id == last_node_acrross_blocks_av)
            {    
                // printf("%d:%d:av\n",activeVertex[j],j);
                atomicExch((unsigned *)&block_inc_av,0);
            }
            while(block_inc_av != 0);
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
printf("Memory Operations Over");
int num_threads = V;

if(V<10000)
{
    num_threads = 10000;
}

long int gridDimx = ceil(float(num_threads)/1024);
long int threadDimx = 1024;
printf("%ld",gridDimx);


nodes_per_level<<<gridDimx,threadDimx>>>(d_csrList,d_offset,d_apr,d_aid,V,L,d_nodesper_level);
cudaDeviceSynchronize();
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
