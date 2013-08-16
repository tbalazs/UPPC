//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

#include <fstream>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__
void histogram(unsigned int* array, unsigned int* hist, const unsigned int bit, const unsigned int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= n) return;
	//if(array[tid] == 0) printf("tid %d\n", tid);
	int bin = ((int)array[tid] >> bit) & 0x1;
	atomicAdd(&(hist[bin]), 1);
}

__global__
void preda(unsigned int* array, unsigned int* pred_array, const unsigned int bit, const unsigned int n, const unsigned int val)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int pred = ((int)array[tid] >> bit) & 0x1;
	pred_array[tid] = (pred == val) ? 1 : 0;
}

__global__
void prefixSum(unsigned int* array, const unsigned int n)
{
	extern __shared__ unsigned int tmp[];
	int tid = threadIdx.x;
	tmp[tid] = (tid>0) ? array[tid-1] : 0;
	__syncthreads();
	for(int offset = 1; offset < n; offset *= 2)
	{
		unsigned int lv = tmp[tid];
		__syncthreads();
		if(tid + offset < n)
		{
			tmp[tid + offset] += lv;
		}
		__syncthreads();
	}
	array[tid] = tmp[tid];
}

__global__
void prefixSumBlock(unsigned int* array, unsigned int* max_array, const unsigned int n)
{
	extern __shared__ unsigned int tmp[];
	int tid = threadIdx.x;
	int toff = blockIdx.x * blockDim.x;
	unsigned int orig = array[tid + toff];
	tmp[tid] = (tid >0) ? array[tid + toff -1] : 0;
	__syncthreads();
	for(int offset = 1; offset < blockDim.x; offset *= 2)
	{
		unsigned int lv = tmp[tid];
		__syncthreads();
		if(tid + offset < blockDim.x)
		{
			tmp[tid + offset] += lv;
		}
		__syncthreads();
	}
	array[tid + toff] = tmp[tid];
	if(tid == blockDim.x - 1) max_array[blockIdx.x] = tmp[tid] + orig;
}

__global__
void prefixSumAdd(unsigned int* array, unsigned int* max_array)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = max_array[blockIdx.x];
	array[tid] += offset;
}

__global__
void reorder(unsigned int* in, unsigned int* out, unsigned int* inpos, unsigned int* outpos, unsigned int* hist, unsigned int* preda, const unsigned int bit, const unsigned int val)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int pred = ((int)in[tid] >> bit) & 0x1;

	if(pred == val)
	{
		int pos = hist[val] + preda[tid];
		out[pos] = in[tid];
		outpos[pos] = inpos[tid];
	}
}

__global__
void pada(unsigned int* in, unsigned int numElems, unsigned int val)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= numElems)
		in[tid] = val;
}

//#define SAVEI 1

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
	const unsigned int threadsPerBlock = 1024;
	const unsigned int n = exp2((float)((int)log2((float)numElems))+1);
	const unsigned int bins = 2;

#ifdef SAVEI
	std::ofstream outs;
	outs.open("sf.out", std::ofstream::out);
#endif

	unsigned int *d_in, *d_inp;
	unsigned int *d_out, *d_outp;
	cudaMalloc((void**)&d_in, sizeof(unsigned int)*n);
	cudaMemcpy(d_in, d_inputVals, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice);
	pada<<<n / threadsPerBlock, threadsPerBlock>>>(d_in, numElems, (unsigned int)(-1));

	//cudaMemcpy(d_in, h_in, sizeof(unsigned int)*n, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_inp, sizeof(unsigned int)*n);
	cudaMemcpy(d_inp, d_inputPos, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice);

	cudaMalloc((void**)&d_out, sizeof(unsigned int)*n);
	cudaMalloc((void**)&d_outp, sizeof(unsigned int)*n);

#ifdef SAVEI
	unsigned int *h_out = new unsigned int[n];
	unsigned int* h_hist = new unsigned int[bins];
	unsigned int* h_preda = new unsigned int[n];
	unsigned int* h_maxa = new unsigned int[n / threadsPerBlock];
#endif

	unsigned int *d_hist, *d_preda, *d_maxa;
	cudaMalloc((void**)&d_hist, sizeof(unsigned int) * bins);
	cudaMalloc((void**)&d_preda, sizeof(unsigned int) * n);
	cudaMalloc((void**)&d_maxa, sizeof(unsigned int) * n / threadsPerBlock);
	for(unsigned int bit = 0; bit < 32; ++bit)
	{
		cudaMemset(d_hist, 0, sizeof(unsigned int) * bins);
		histogram<<<n / threadsPerBlock, threadsPerBlock>>>(d_in, d_hist, bit, n);
#ifdef SAVEI
		cudaMemcpy(h_hist, d_hist, sizeof(unsigned int)* bins, cudaMemcpyDeviceToHost);
		outs << "Hist of bit " << bit << ": " << h_hist[0] << ", " << h_hist[1] << std::endl;
#endif
		prefixSum<<<1, bins, sizeof(unsigned int) * bins>>>(d_hist, bins);
#ifdef SAVEI
		cudaMemcpy(h_hist, d_hist, sizeof(unsigned int)* bins, cudaMemcpyDeviceToHost);
		outs << "PrefSum Hist of bit " << bit << ": " << h_hist[0] << ", " << h_hist[1] << std::endl;
#endif

		// pred val = 0

		preda<<<n / threadsPerBlock, threadsPerBlock>>>(d_in, d_preda, bit, n, 0);
#ifdef SAVEI
		cudaMemcpy(h_preda, d_preda, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
		outs << "Predicate array:          ";
		for(unsigned int i = 0; i < n; ++i)
		{
			outs << h_preda[i] << ", ";
		}
		outs << std::endl;
#endif
		prefixSumBlock<<<n / threadsPerBlock, threadsPerBlock, sizeof(unsigned int) * threadsPerBlock>>>(d_preda, d_maxa, n);
#ifdef SAVEI
		cudaMemcpy(h_maxa, d_maxa, sizeof(unsigned int) * n / threadsPerBlock, cudaMemcpyDeviceToHost);
		outs << "Max array: ";
		for(unsigned int i = 0; i < n /threadsPerBlock; ++i)
		{
			outs << h_maxa[i] << ", ";
		}
		outs << std::endl;
		cudaMemcpy(h_preda, d_preda, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
		outs << "Predicate array pref sum: ";
		for(unsigned int i = 0; i < n; ++i)
		{
			outs << h_preda[i] << ", ";
		}
		outs << std::endl;
		outs << n / threadsPerBlock << std::endl;
#endif
		prefixSum<<<1, n / threadsPerBlock, sizeof(unsigned int) * threadsPerBlock>>>(d_maxa, n / threadsPerBlock);
#ifdef SAVEI
		cudaMemcpy(h_maxa, d_maxa, sizeof(unsigned int) * n / threadsPerBlock, cudaMemcpyDeviceToHost);
		outs << "Max array pref sum: ";
		for(unsigned int i = 0; i < n /threadsPerBlock; ++i)
		{
			outs << h_maxa[i] << ", ";
		}
		outs << std::endl;
#endif
		prefixSumAdd<<<n / threadsPerBlock, threadsPerBlock>>>(d_preda, d_maxa);
#ifdef SAVEI
		cudaMemcpy(h_preda, d_preda, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
		outs << "Predicate array sum: ";
		for(unsigned int i = 0; i < n; ++i)
		{
			outs << h_preda[i] << ", ";
		}
		outs << std::endl;
#endif
		reorder<<<n / threadsPerBlock, threadsPerBlock>>>(d_in, d_out, d_inp, d_outp, d_hist, d_preda, bit, 0);
#ifdef SAVEI
		cudaMemcpy(h_out, d_out, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
		outs << "Reordered array along bit " << bit << " pred val: " << 0 << ": " ;
		for(unsigned int i = 0; i < n; ++i)
		{
			outs << h_out[i] << ", ";
		}
		outs << std::endl;
#endif

		cudaMemset(d_hist, 0, sizeof(unsigned int) * bins);
		histogram<<<n / threadsPerBlock, threadsPerBlock>>>(d_in, d_hist, bit, n);
#ifdef SAVEI
		cudaMemcpy(h_hist, d_hist, sizeof(unsigned int)* bins, cudaMemcpyDeviceToHost);
		outs << "Hist of bit " << bit << ": " << h_hist[0] << ", " << h_hist[1] << std::endl;
#endif
		prefixSum<<<1, bins, sizeof(unsigned int) * bins>>>(d_hist, bins);
#ifdef SAVEI
		cudaMemcpy(h_hist, d_hist, sizeof(unsigned int)* bins, cudaMemcpyDeviceToHost);
		outs << "PrefSum Hist of bit " << bit << ": " << h_hist[0] << ", " << h_hist[1] << std::endl;
#endif
		// pred val = 1
		preda<<<n / threadsPerBlock, threadsPerBlock>>>(d_in, d_preda, bit, n, 1);
#ifdef SAVEI
		cudaMemcpy(h_preda, d_preda, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
		outs << "Predicate array:          ";
		for(unsigned int i = 0; i < n; ++i)
		{
			outs << h_preda[i] << ", ";
		}
		outs << std::endl;
#endif
		prefixSumBlock<<<n / threadsPerBlock, threadsPerBlock, sizeof(unsigned int) * threadsPerBlock>>>(d_preda, d_maxa, n);
#ifdef SAVEI
		cudaMemcpy(h_maxa, d_maxa, sizeof(unsigned int) * n / threadsPerBlock, cudaMemcpyDeviceToHost);
		outs << "Max array: ";
		for(unsigned int i = 0; i < n /threadsPerBlock; ++i)
		{
			outs << h_maxa[i] << ", ";
		}
		outs << std::endl;
		cudaMemcpy(h_preda, d_preda, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
		outs << "Predicate array pref sum: ";
		for(unsigned int i = 0; i < n; ++i)
		{
			outs << h_preda[i] << ", ";
		}
		outs << std::endl;
#endif
		prefixSum<<<1, n / threadsPerBlock, sizeof(unsigned int) * threadsPerBlock>>>(d_maxa, n / threadsPerBlock);
#ifdef SAVEI
		cudaMemcpy(h_maxa, d_maxa, sizeof(unsigned int) * n / threadsPerBlock, cudaMemcpyDeviceToHost);
		outs << "Max array pref sum: ";
		for(unsigned int i = 0; i < n /threadsPerBlock; ++i)
		{
			outs << h_maxa[i] << ", ";
		}
		outs << std::endl;
#endif
		prefixSumAdd<<<n / threadsPerBlock, threadsPerBlock>>>(d_preda, d_maxa);
#ifdef SAVEI
		cudaMemcpy(h_preda, d_preda, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
		outs << "Predicate array sum: ";
		for(unsigned int i = 0; i < n; ++i)
		{
			outs << h_preda[i] << ", ";
		}
		outs << std::endl;
#endif
		reorder<<<n / threadsPerBlock, threadsPerBlock>>>(d_in, d_out, d_inp, d_outp, d_hist, d_preda, bit, 1);
#ifdef SAVEI
		cudaMemcpy(h_out, d_out, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
		outs << "Reordered array along bit " << bit << " pred val: " << 1 << ": " ;
		for(unsigned int i = 0; i < n; ++i)
		{
			outs << h_out[i] << ", ";
		}
		outs << std::endl;
#endif
		cudaMemcpy(d_in, d_out, sizeof(unsigned int) * n, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_inp, d_outp, sizeof(unsigned int) * n, cudaMemcpyDeviceToDevice);
	}

	cudaMemcpy(d_outputVals, d_out, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_outputPos, d_outp, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice);

#ifdef SAVEI
	outs.close();
	delete[] h_out;
	delete[] h_hist;
	delete[] h_preda;
	delete[] h_maxa;
#endif
	cudaFree(d_in);
	cudaFree(d_inp);
	cudaFree(d_out);
	cudaFree(d_outp);
	cudaFree(d_hist);
	cudaFree(d_preda);
	cudaFree(d_maxa);
}
