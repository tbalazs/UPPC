//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>

//const unsigned int tileSize = 16;

__global__
void generateMask(uchar4* d_sourceImg, char* d_mask, const unsigned int numRowsSource, const unsigned int numColsSource)
{
	if(threadIdx.x >= numColsSource ||
	   blockIdx.x >= numRowsSource) return;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	uchar4 pix = d_sourceImg[tid];
	if(pix.x + pix.y + pix.z < 3 * 255)
	{
		d_mask[tid] = 1;
	} else {
		d_mask[tid] = 0;
	}
}

__global__
void generateStructure(char* d_mask, char* d_interior, const unsigned int numRowsSource, const unsigned int numColsSource)
{
	if(threadIdx.x >= numColsSource ||
	   blockIdx.x >= numRowsSource) return;

	int x = threadIdx.x + 1;
	int y = blockIdx.x + 1;
	int tid = x + y * blockDim.x;

	if(d_mask[tid] == 1)
	{
		unsigned int mVal = d_mask[x - 1 + y * blockDim.x] +
							d_mask[x + 1 + y * blockDim.x] +
							d_mask[x + (y + 1) * blockDim.x] +
							d_mask[x + (y - 1) * blockDim.x];
		if(mVal == 4) d_interior[tid] = 2; // interior
		else d_interior[tid] = 1; // border
	}
}

__global__
void separateChannels(const uchar4 *d_sourceImg, float* d_rChannel, float* d_gChannel, float* d_bChannel, const unsigned int numRowsSource, const unsigned int numColsSource)
{
	if(threadIdx.x >= numColsSource ||
	   blockIdx.x >= numRowsSource) return;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	uchar4 imgVal = d_sourceImg[tid];
	d_rChannel[tid] = imgVal.x;
	d_gChannel[tid] = imgVal.y;
	d_bChannel[tid] = imgVal.z;
}

__global__
void copyByMask(char *d_mask, float* d_rChannel, float* d_gChannel, float* d_bChannel, uchar4* d_blendedImg, const unsigned int numRowsSource, const unsigned int numColsSource)
{
	if(threadIdx.x >= numColsSource ||
	   blockIdx.x >= numRowsSource) return;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(d_mask[tid] == 2)
	{
		d_blendedImg[tid].x = (unsigned char)d_rChannel[tid];
		d_blendedImg[tid].y = (unsigned char)d_gChannel[tid];
		d_blendedImg[tid].z = (unsigned char)d_bChannel[tid];
//		d_blendedImg[tid] = make_uchar4(200,200,200,200);
	}
//	if(d_mask[tid] == 1)
//	{
//		d_blendedImg[tid] = make_uchar4(100,100,100,100);
//	}
}

//	Follow these steps to implement one iteration:
//
//	   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
//	      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
//	             else if the neighbor in on the border then += DestinationImg[neighbor]
//
//	      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)
//
//	   2) Calculate the new pixel value:
//	      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
//	      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]
__global__
void iteration(char* d_interior, const uchar4* sourceImage, const uchar4* destImage,
								 float* d_rChannelIn, float* d_rChannelOut,
								 float*	d_gChannelIn, float* d_gChannelOut,
								 float* d_bChannelIn, float* d_bChannelOut,
								 const unsigned int numRowsSource, const unsigned int numColsSource)
{
	if(threadIdx.x >= numColsSource ||
	   blockIdx.x >= numRowsSource) return;

	int x = threadIdx.x;
	int y = blockIdx.x;
	int tid = x + y * blockDim.x;

	float3 sum1 = make_float3(0.0f, 0.0f, 0.0f);
	float3 sum2 = make_float3(0.0f, 0.0f, 0.0f);

	//	   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
	//	      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
	//	             else if the neighbor in on the border then += DestinationImg[neighbor]
	//
	//	      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)
	if(d_interior[tid] == 2)
	{
		int nid = x + 1 + y * blockDim.x;
		if(d_interior[nid] == 2)
		{
			sum1.x += d_rChannelIn[nid];
			sum1.y += d_gChannelIn[nid];
			sum1.z += d_bChannelIn[nid];
		} else if(d_interior[nid] == 1)
		{
			sum1.x += destImage[nid].x;
			sum1.y += destImage[nid].y;
			sum1.z += destImage[nid].z;
		}
		sum2.x += sourceImage[tid].x - sourceImage[nid].x;
		sum2.y += sourceImage[tid].y - sourceImage[nid].y;
		sum2.z += sourceImage[tid].z - sourceImage[nid].z;

		nid = x - 1 + y * blockDim.x;
		if(d_interior[nid] == 2)
		{
			sum1.x += d_rChannelIn[nid];
			sum1.y += d_gChannelIn[nid];
			sum1.z += d_bChannelIn[nid];
		} else if(d_interior[nid] == 1)
		{
			sum1.x += destImage[nid].x;
			sum1.y += destImage[nid].y;
			sum1.z += destImage[nid].z;
		}

		sum2.x += sourceImage[tid].x - sourceImage[nid].x;
		sum2.y += sourceImage[tid].y - sourceImage[nid].y;
		sum2.z += sourceImage[tid].z - sourceImage[nid].z;

		nid = x + (y + 1) * blockDim.x;
		if(d_interior[nid] == 2)
		{
			sum1.x += d_rChannelIn[nid];
			sum1.y += d_gChannelIn[nid];
			sum1.z += d_bChannelIn[nid];
		} else if(d_interior[nid] == 1)
		{
			sum1.x += destImage[nid].x;
			sum1.y += destImage[nid].y;
			sum1.z += destImage[nid].z;
		}
		sum2.x += sourceImage[tid].x - sourceImage[nid].x;
		sum2.y += sourceImage[tid].y - sourceImage[nid].y;
		sum2.z += sourceImage[tid].z - sourceImage[nid].z;

		nid = x + (y - 1) * blockDim.x;
		if(d_interior[nid] == 2)
		{
			sum1.x += d_rChannelIn[nid];
			sum1.y += d_gChannelIn[nid];
			sum1.z += d_bChannelIn[nid];
		} else if(d_interior[nid] == 1)
		{
			sum1.x += destImage[nid].x;
			sum1.y += destImage[nid].y;
			sum1.z += destImage[nid].z;
		}
		sum2.x += sourceImage[tid].x - sourceImage[nid].x;
		sum2.y += sourceImage[tid].y - sourceImage[nid].y;
		sum2.z += sourceImage[tid].z - sourceImage[nid].z;

		//	   2) Calculate the new pixel value:
		//	      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
		//	      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]
		float3 newVal = make_float3( (sum1.x + sum2.x) / 4.0f,
									 (sum1.y + sum2.y) / 4.0f,
									 (sum1.z + sum2.z) / 4.0f);
		d_rChannelOut[tid] = fmin(255.0f, fmax(0, newVal.x));
		d_gChannelOut[tid] = fmin(255.0f, fmax(0, newVal.y));
		d_bChannelOut[tid] = fmin(255.0f, fmax(0, newVal.z));
	}
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
	//std::cout << numColsSource << ", " << numRowsSource << std::endl;
	uchar4* d_sourceImage = NULL;
	cudaMalloc((void**)&d_sourceImage, sizeof(uchar4)*numRowsSource*numColsSource);
	cudaMemcpy(d_sourceImage, h_sourceImg, sizeof(uchar4)*numRowsSource*numColsSource, cudaMemcpyHostToDevice);

	uchar4* d_destImage = NULL;
	cudaMalloc((void**)&d_destImage, sizeof(uchar4)*numRowsSource*numColsSource);
	cudaMemcpy(d_destImage, h_destImg, sizeof(uchar4)*numRowsSource*numColsSource, cudaMemcpyHostToDevice);

	char *d_mask = NULL;
	cudaMalloc((void**)&d_mask, sizeof(char)*numRowsSource*numColsSource);
	cudaMemset(d_mask, 0, sizeof(char));
	generateMask<<<numRowsSource, numColsSource>>>(d_sourceImage, d_mask, numRowsSource, numColsSource);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	char *d_interior = NULL;
	cudaMalloc((void**)&d_interior, sizeof(char)*numRowsSource*numColsSource);
	cudaMemset(d_interior, 0, sizeof(char)*numRowsSource*numColsSource);
	generateStructure<<<numRowsSource, numColsSource>>>(d_mask, d_interior, numRowsSource, numColsSource);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	unsigned int input = 0;
	float* d_rChannel[2] = {NULL, NULL};
	cudaMalloc((void**)&(d_rChannel[0]), sizeof(float)*numRowsSource*numColsSource);
	cudaMalloc((void**)&(d_rChannel[1]), sizeof(float)*numRowsSource*numColsSource);
	float* d_gChannel[2] = {NULL, NULL};
	cudaMalloc((void**)&(d_gChannel[0]), sizeof(float)*numRowsSource*numColsSource);
	cudaMalloc((void**)&(d_gChannel[1]), sizeof(float)*numRowsSource*numColsSource);
	float* d_bChannel[2] = {NULL, NULL};
	cudaMalloc((void**)&(d_bChannel[0]), sizeof(float)*numRowsSource*numColsSource);
	cudaMalloc((void**)&(d_bChannel[1]), sizeof(float)*numRowsSource*numColsSource);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	separateChannels<<<numRowsSource, numColsSource>>>(d_sourceImage, d_rChannel[0], d_gChannel[0], d_bChannel[0], numRowsSource, numColsSource);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// Jacobi iteration
	for(unsigned int pass = 0; pass < 800; ++pass)
	{
		iteration<<<numRowsSource, numColsSource>>>(d_interior, d_sourceImage, d_destImage,
																d_rChannel[input], d_rChannel[(input+1)%2],
																d_gChannel[input], d_gChannel[(input+1)%2],
																d_bChannel[input], d_bChannel[(input+1)%2],
																numRowsSource, numColsSource);
		input = (input + 1) % 2;
	}

	uchar4* d_blendedImage = NULL;
	cudaMalloc((void**)&d_blendedImage, sizeof(uchar4)*numRowsSource * numColsSource);
	cudaMemcpy(d_blendedImage, h_destImg, sizeof(uchar4)*numRowsSource*numColsSource, cudaMemcpyHostToDevice);
	copyByMask<<<numRowsSource, numColsSource>>>(d_interior, d_rChannel[input], d_gChannel[input], d_bChannel[input], d_blendedImage, numRowsSource, numColsSource);
	//copyByMask<<<numRowsSource, numColsSource>>>(d_mask, d_rChannel[input], d_gChannel[input], d_bChannel[input], d_blendedImage, numRowsSource, numColsSource);
	cudaMemcpy(h_blendedImg, d_blendedImage, sizeof(uchar4)*numRowsSource*numColsSource, cudaMemcpyDeviceToHost);

	cudaFree(d_rChannel[0]);
	cudaFree(d_rChannel[1]);
	cudaFree(d_gChannel[0]);
	cudaFree(d_gChannel[1]);
	cudaFree(d_bChannel[0]);
	cudaFree(d_bChannel[1]);
	cudaFree(d_blendedImage);
	cudaFree(d_interior);
	cudaFree(d_mask);
	cudaFree(d_destImage);
	cudaFree(d_sourceImage);

  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */
}
