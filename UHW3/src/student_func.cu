/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

#include <iostream>

__global__
void reduce_min(const float* const d_logLuminance, float* d_reduceTmp)
{
	extern __shared__ float sdata[];
	int gid = threadIdx.x + blockDim.x * blockIdx.x;
	int lid = threadIdx.x;

	sdata[lid] = d_logLuminance[gid];
	__syncthreads();

	for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if(lid < s)
		{
			sdata[lid] = min(sdata[lid], sdata[lid + s]);
		}
		__syncthreads();
	}

	if(lid == 0)
	{
		d_reduceTmp[blockIdx.x] = sdata[0];
	}
}

__global__
void reduce_max(const float* const d_logLuminance, float* d_reduceTmp)
{
	extern __shared__ float sdata[];
	int gid = threadIdx.x + blockDim.x * blockIdx.x;
	int lid = threadIdx.x;

	sdata[lid] = d_logLuminance[gid];
	__syncthreads();

	for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if(lid < s)
		{
			sdata[lid] = max(sdata[lid], sdata[lid + s]);
		}
		__syncthreads();
	}

	if(lid == 0)
	{
		d_reduceTmp[blockIdx.x] = sdata[0];
	}
}

__global__
void histogram(const float * const d_logLuminance, unsigned int *d_hist, float minv, const float range, const int numBins)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float item = d_logLuminance[tid];
	int bin = (item - minv) / range * numBins;
	atomicAdd(&(d_hist[bin]), 1);
}

__global__
void exscan(unsigned int * d_hist, unsigned int * const d_cdf, const int numBins)
{
	extern __shared__ unsigned int tmp[];
	int tid = threadIdx.x;
	tmp[tid] = (tid>0) ? d_hist[tid-1] : 0;
	__syncthreads();
	for(int offset = 1; offset < numBins; offset *= 2)
	{
		unsigned int lv = tmp[tid];
		__syncthreads();
		if(tid + offset < numBins)
		{
			tmp[tid + offset] += lv;
		}
		__syncthreads();
	}
	d_cdf[tid] = tmp[tid];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
//	std::cout << "Elem: " << numRows * numCols << " bins: " << numBins << std::endl;
//
//	float* h_logLuminance = new float[numRows * numCols];
//	cudaMemcpy(h_logLuminance, d_logLuminance, sizeof(float) * numRows * numCols, cudaMemcpyDeviceToHost);
//	float vmin = h_logLuminance[0];
//	float vmax = h_logLuminance[0];
//	for(unsigned int i = 0; i < numRows * numCols; ++i)
//	{
//		vmin = min(vmin, h_logLuminance[i]);
//		vmax = max(vmax, h_logLuminance[i]);
//	}
//	std::cout << "hmin: " << vmin << " hmax: " << vmax << std::endl;

	float* d_reduceTmp, *d_min, *d_max;
	cudaMalloc((void**)&d_reduceTmp, sizeof(float) * numRows * numCols);
	cudaMalloc((void**)&d_min, sizeof(float));
	cudaMalloc((void**)&d_max, sizeof(float));

	int threadsPerBlock = 512;
	reduce_min<<<numRows * numCols / threadsPerBlock, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_logLuminance, d_reduceTmp);
	reduce_min<<<1, numRows * numCols / threadsPerBlock, sizeof(float) * numRows * numCols / threadsPerBlock>>>(d_reduceTmp, d_min);

	reduce_max<<<numRows * numCols / threadsPerBlock, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_logLuminance, d_reduceTmp);
	reduce_max<<<1, numRows * numCols / threadsPerBlock, sizeof(float) * numRows * numCols / threadsPerBlock>>>(d_reduceTmp, d_max);

	cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost);

	float range = max_logLum - min_logLum;

	unsigned int* d_hist, *h_hist;
	cudaMalloc((void**)&d_hist, sizeof(unsigned int) * numBins);
	cudaMemset(d_hist, 0, sizeof(unsigned int)*numBins);
	histogram<<<numRows * numCols / threadsPerBlock, threadsPerBlock>>>(d_logLuminance, d_hist, min_logLum, range, numBins);

//	h_hist = new unsigned int[numBins];
//	cudaMemcpy(h_hist, d_hist, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost);
//	for(unsigned int i = 0; i < numBins; ++i)
//	{
//		std::cout << h_hist[i] << ", ";
//	}
//	std::cout << std::endl;

	exscan<<<1, 1024, sizeof(unsigned int) * 1024>>>(d_hist, d_cdf, numBins);

//	cudaMemcpy(h_hist, d_cdf, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost);
//	for(unsigned int i = 0; i < numBins; ++i)
//	{
//		std::cout << h_hist[i] << ", ";
//	}
//	std::cout << std::endl;

}
