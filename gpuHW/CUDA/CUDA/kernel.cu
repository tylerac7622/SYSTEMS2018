
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef ALIGNMENT
#define ALIGNMENT 64
#endif
#include <stdio.h>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;

#define O (2048 * 2048)
#define THREADS_PER_BLOCK 512
__global__ void add(int *a, int *b, int *c)
{
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
__global__ void dot(int*a, int*b, int*c)
{
	__shared__ int temp[THREADS_PER_BLOCK];
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	temp[threadIdx.x] = a[index] * b[index];

	__syncthreads();

	if (0 == threadIdx.x)
	{
		int sum = 0;
		for (int i = 0; i< THREADS_PER_BLOCK; i++)
			sum += temp[i];
		atomicAdd(c, sum);
	}
}

void random_ints(int* a, int n)
{
	int i;
	for (i = 0; i < n; ++i)
		a[i] = rand();
}

bool singleCore = true;

__global__ void fwStepK(int k, int devArray[], int Na, int N) 
{
	int col = blockIdx.x * blockDim.x + threadIdx.x; 
	/* This thread's matrix column */
	if (col >= N)
		return;
	int arrayIndex = Na * blockIdx.y + col; __shared__ int trkc;
	/* this row, kth column */
	// Improve by using the intermediate k, if we can
	if(threadIdx.x == 0)
		trkc = devArray[Na * blockIdx.y + k];
	__syncthreads();
	if(trkc == INT_MAX)
		/* infinity */
		return;
	int tckr = devArray[k*Na + col]; 
	/* this column, kth row */
	if(tckr == INT_MAX)    
		/* infinity */
		return;
	int betterMaybe = trkc + tckr;
	if(betterMaybe < devArray[arrayIndex])
		devArray[arrayIndex] =  betterMaybe;
}
map<string, int> nameToNum;
/* names of vertices */
map<string, map<string, int>> weightMap;
/* weights of edges */
int* readGraph(int& N, int& Na, char* argv)
{
	// Read the graph file from memory
	string vname1, vname2;
	ifstream graphFile;
	string dummyString;
	int thisWeight;
	/* weight of the edge just read from file */
	N = 0;
	/* number of vertices */
	graphFile.open(argv);
	//Read the graph into some maps
	graphFile >> vname1;
	while (vname1 != "--END--")
	{
		graphFile >> vname2; graphFile >> thisWeight;
		if (nameToNum.count(vname1) == 0)
		{
			nameToNum[vname1] = N;
			weightMap[vname1][vname1] = 0;
			N++;
		}
		if (nameToNum.count(vname2) == 0)
		{
			nameToNum[vname2] = N; weightMap[vname2][vname2] = 0;
			N++;
		}
		weightMap[vname1][vname2] = thisWeight;
		graphFile >> vname1;
	}
	graphFile.close(); // Nice and Tidy// "alignment" is what stored row sizes must be a multiple of30
	int alignment = ALIGNMENT;
	if (!singleCore)
	{
		Na = alignment*((N + alignment - 1) / alignment);
	}
	else
	{
		Na = N;
	}
	/* for the sizes of our arrays */
	printf("Alignment = %d\n", alignment);
	// Build the array
	int* a = (int*)malloc(N*Na * sizeof(int));
	for (int ii = 0; ii < N; ii++)
		for (int jj = 0; jj < N; jj++)
			a[ii * Na + jj] = INT_MAX;
	map<string, int>::iterator i;
	map<string, int>::iterator j;
	for (i = nameToNum.begin(); i != nameToNum.end(); ++i)
		for (j = nameToNum.begin(); j != nameToNum.end(); ++j)
		{
			if (weightMap[(*i).first].count((*j).first) != 0)
			{
				a[Na * (*i).second + (*j).second] = weightMap[(*i).first][(*j).first];
			}
		}
	return a;
}
void printArray(int Na, int* a)
{
	map<string, int>::iterator i, j;
	for (i = nameToNum.begin(); i != nameToNum.end(); ++i)
		if (i->second < 10)
			printf("\t%s", i->first.c_str());
	printf("\n"); j = nameToNum.begin();
	for (i = nameToNum.begin(); i != nameToNum.end(); ++i)
	{
		if (i->second < 10)
		{
			printf("%s\t", i->first.c_str());
			for (j = nameToNum.begin(); j != nameToNum.end(); ++j)
			{
				if (j->second < 10)
				{
					int dd = a[i->second * Na + j->second];
					if (dd != INT_MAX)
						printf("%d\t", dd);
					else
						printf("--\t");
				}
			}
		}
	}
}
int main(int argc, char* argv[])
{
	std::chrono::time_point<std::chrono::steady_clock> begin = std::chrono::high_resolution_clock::now();
	/*int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int size = O * sizeof(int);

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, sizeof(int));

	a = (int *)malloc(size); random_ints(a, O);
	b = (int *)malloc(size); random_ints(b, O);
	c = (int *)malloc(sizeof(int));

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	std::chrono::time_point<std::chrono::steady_clock> mid1 = std::chrono::high_resolution_clock::now();
	dot<<< O/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
	std::chrono::time_point<std::chrono::steady_clock> mid2 = std::chrono::high_resolution_clock::now();

	cudaMemcpy(c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

	std::chrono::time_point<std::chrono::steady_clock> mid3 = std::chrono::high_resolution_clock::now();
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);*/

	if (!singleCore)
	{
		int N = 0;
		int Na = 0;
		// Width of matrix to encourage coalescing
		int* graph = readGraph(N, Na, "graph.txt");
		// from fwHelpers.cpp
		printf("Kernel: Just read %s with %d vertices, Na = %d\n", argv[1], N, Na);
		// Copy the array to newly-allocated global memory
		int* devArray;
		cudaError_t err = cudaMalloc(&devArray, Na*N * sizeof(int));
		printf("Malloc device rules: %s\n", cudaGetErrorString(err));
		err = cudaMemcpy(devArray, graph, Na*N * sizeof(int), cudaMemcpyHostToDevice);
		printf("Pre-kernel copy memory onto device: %s\n", cudaGetErrorString(err));
		// Set up and run the kernels
		int threadsPerBlock = 256;
		dim3 blocksPerGrid((Na + threadsPerBlock - 1) / threadsPerBlock, N);
		// The kth run through this loop considers whether we might do better using// the kth vertex as an intermediate
		for (int k = 0; k < N; k++)
		{
			fwStepK <<< blocksPerGrid, threadsPerBlock >>> (k, devArray, Na, N);
			err = cudaThreadSynchronize();
			// Uncomment the following line when debugging the kernel
			// printf("Kernel: using %d as intermediate: error = %s\n", k, cudaGetErrorString(err));
			// Uncomment the following two lines to print intermediate results
			// err = cudaMemcpy(graph, devArray, Na*N*sizeof(int), cudaMemcpyDeviceToHost);
			// printArray(Na, graph);
		}
		err = cudaMemcpy(graph, devArray, Na*N * sizeof(int), cudaMemcpyDeviceToHost);
		printf("Post-kernel copy memory off of device: %s\n", cudaGetErrorString(err));
		printArray(Na, graph);
		free(graph);
		cudaFree(devArray);
	}
	else
	{
		int N = 0; 
		int Na = 0; 
		int* graph = readGraph(N, Na, "graph.txt");
		printf("Read %s with %d vertices, Na = %d\n", argv[1], N, Na); 
		printArray(N, graph); 
		for (int k = 0; k < N; k++)
			for (int i = 0; i < N; i++)
				if (graph[i*N + k] != INT_MAX)
					for (int j = 0; j < N; j++)
						if (graph[k*N + j] != INT_MAX)
							if (graph[i*N + k] + graph[k*N + j] < graph[i*N + j])
								graph[i*N + j] = graph[i*N + k] + graph[k*N + j]; 
		printArray(N, graph);
	}
	
	std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::high_resolution_clock::now(); 

	/*float percent1 = 1.0f * std::chrono::duration_cast<std::chrono::nanoseconds>(mid1 - begin).count() / std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	float percent2 = 1.0f * std::chrono::duration_cast<std::chrono::nanoseconds>(mid3 - mid2).count() / std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	std::cout << std::endl << percent1 << std::endl;
	std::cout << std::endl << percent2 << std::endl;*/
	std::cout << std::endl << std::endl << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "ns" << std::endl;

	int z;
	scanf("%d", &z);

    return 0;
}