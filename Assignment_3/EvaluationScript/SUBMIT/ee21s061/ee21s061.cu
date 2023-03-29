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

// Here i have declared a few variable to maintain synchronization between multiple blocks as well as each level of the graph.
__device__ volatile unsigned level = 0 ; // Current level under process
__device__ volatile unsigned node_counter = 0; // Number of nodes in current level
__device__ volatile unsigned node_counter_prev = 0; // Number of nodes in prev level
__device__ volatile unsigned block_inc = 0; // counter to sync between multiple blocks
__device__ volatile unsigned last_node_per_level = 0; // last node id in the current level
__device__ volatile unsigned last_thread_blocks; // last thread to finish execution in the entire grid
ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/


__global__ void active_nodes_per_level(int *csrList,int *offset,int *apr,int *aid,int V,int L,int *activeVertex){



    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int level_id;
    unsigned int active = 0;
    
    for(int j=0;j<L;j++){ // Loop iterates the for each level in the graph
        
        level_id = node_counter_prev + id; // Maximum number of threads for any input is 10000 based on the threads required to process one level in parallel
        active = 0;
        if( ((apr[level_id] == 0 && level == 0) || (level > 0  && level_id>=node_counter && level_id<=last_node_per_level) ) && level_id<V ){
                // This condition ensures only valid id for each level are parsed

            if(aid[level_id]>=apr[level_id]){ // Rule 1
                if( level_id == node_counter ||  level_id == last_node_per_level || aid[level_id-1]>=apr[level_id-1] || aid[level_id+1]>=apr[level_id+1] ){ // Rule 2
                    atomicAdd(&activeVertex[j],1);
                    active = 1; // status of current node for each thread
                }
            }
            int start,end;
            start = offset[level_id];
            end = offset[level_id+1];
            // Parsing the CSR graph to find number of nodes in the next level
            for(int i =start;i<end;i++){
                int curr_edge = csrList[i];
                if(active){
                    atomicAdd(&aid[curr_edge],1); // Computing AID for next level 
                }
                atomicMax((unsigned *)&last_node_per_level,curr_edge); // maximum id of the node that is connected to any node in current level
                }
            atomicAdd((unsigned *)&node_counter,1); // number of nodes for this level
            }

        // The code below perform sync for all the threads
        __syncthreads();
        if(threadIdx.x == 0){
            // sync of each block
            atomicExch((unsigned *)&last_thread_blocks,id);
            atomicInc((unsigned *)&block_inc,gridDim.x + 1);
            
            while (block_inc != gridDim.x);
            
            if(id == last_thread_blocks)
            {    
                // last thread to execute the code sets the parameter for the next run of the loop
                // This part of the code runs only once for each level
                atomicExch((unsigned *)&node_counter_prev,node_counter);

                atomicAdd((unsigned *)&level,1);

                atomicExch((unsigned *)&block_inc,0);

            }
            while(block_inc != 0);
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

int num_threads = V;
// number of threads to launch is upper bound by number of nodes possible in a level which is mentioned to be 10000
if(V>10000)
{
    num_threads = 10000;
}

long int gridDimx = ceil(float(num_threads)/1024);
long int threadDimx = 1024;

// Kernel launch for the processing 
active_nodes_per_level<<<gridDimx,threadDimx>>>(d_csrList,d_offset,d_apr,d_aid,V,L,d_activeVertex);
cudaDeviceSynchronize();
// Transfer the computed data back to host
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
