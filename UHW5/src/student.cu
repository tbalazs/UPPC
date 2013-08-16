/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
	extern __shared__ unsigned int lhisto[];

	int tid = threadIdx.x;
	int addr = (threadIdx.x + blockIdx.x * blockDim.x) * 4;

	lhisto[tid] = 0;
	uint4* va =  (uint4*)(vals+addr);
	uint4 v = *va;
	__syncthreads();
	atomicAdd(&(lhisto[v.x]), 1);
	atomicAdd(&(lhisto[v.y]), 1);
	atomicAdd(&(lhisto[v.z]), 1);
	atomicAdd(&(lhisto[v.w]), 1);
	__syncthreads();
	atomicAdd(&(histo[tid]), lhisto[tid]);
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  unsigned int threadsPerBlocks = 1024;
  yourHisto<<<numElems / threadsPerBlocks / 4, threadsPerBlocks, sizeof(unsigned int)*numBins+1>>>(d_vals, d_histo, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
