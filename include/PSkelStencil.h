//-------------------------------------------------------------------------------
// Copyright (c) 2015, Alyson D. Pereira <alyson.deives@outlook.com>,
//					   Rodrigo C. O. Rocha <rcor.cs@gmail.com>
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//-------------------------------------------------------------------------------

/**
 * \file PSkelStencil.h
 * This file contains the definition for the stencil skeleton.
*/
#ifndef PSKEL_STENCIL_H
#define PSKEL_STENCIL_H

//#ifdef PSKEL_TBB
//  #include <tbb/blocked_range.h>
//  #include <tbb/parallel_for.h>
//  #include <tbb/task_scheduler_init.h>
//#endif

#include "PSkelDefs.h"
#include "PSkelArray.h"
#include "PSkelMask.h"
#ifdef PSKEL_CUDA
  #include "PSkelStencilTiling.h"
#endif

namespace PSkel{

/*
template<typename T>
struct SharedMemory
{
    size_t width;
    size_t range;
    // Should never be instantiated.
    // We enforce this at compile time.
    __device__ T* GetPointer( void )
    {
        extern __device__ void error( void );
        error();
        return NULL;
    }
    
    __device__ T* get(size_t, size_t);
};

// specializations for types we use
template<>
struct SharedMemory<float>
{
	size_t width;
    size_t range;
    
    __device__ float* GetPointer(){
        extern __shared__ float sh_float[];
        // printf( "sh_float=%p\n", sh_float );
        return sh_float;
    }
    
    __device__ float* GetPointer(size_t blockWidth, size_t maskRange){
        width = blockWidth;
        range = maskRange;
        extern __shared__ float sh_float[];
        // printf( "sh_float=%p\n", sh_float );
        return sh_float;
    }
    
    __device__ float get(size_t h, size_t w){
		float* sh = this->GetPointer();
		printf("value: %f\n",sh[(h+range)*(width+2*range)+(w+range)]);
		return sh[(h+range)*(width+2*range)+(w+range)];
	}
	
};

//template<typename T>
//__device__ T getShared(SharedMemory<T> shared, size_t h, size_t w){
//	return shared[(h+shared.range)*(shared.width+2*shared.range)+(w+shared.range)];
//}
*/

//*******************************************************************************************
// Stencil Kernels that must be implemented by the users.
//*******************************************************************************************

/**
 * Function signature of the stencil kernel for processing 1-dimensional arrays.
 * This function must be implemented by the user.
 * \param[in] input 1-dimensional Array with the input data.
 * \param[in] output 1-dimensional Array that will hold the output data.
 * \param[in] mask 1-dimensional Mask used to scan through the input data,
 * accessing the neighbourhood for each element.
 * \param[in] args extra arguments that may be used by the stencil computations.
 * \param[in] i index for the current element to be processed.
 **/
template<typename T1, typename T2, class Args>
__parallel__ void stencilKernel(Array<T1> input, Array<T1> output, Mask<T2> mask, Args args, size_t i);

/**
 * Function signature of the stencil kernel for processing 2-dimensional arrays.
 * This function must be implemented by the user.
 * \param[in] input 1-dimensional Array with the input data.
 * \param[in] output 1-dimensional Array that will hold the output data.
 * \param[in] mask 1-dimensional Mask used to scan through the input data,
 * accessing the neighbourhood for each element.
 * \param[in] args extra arguments that may be used by the stencil computations.
 * \param[in] h height index for the current element to be processed.
 * \param[in] w width index for the current element to be processed.
 **/

//template<typename T1, class Args>
//__parallel__ void stencilKernel(T1 input[BLOCK_SIZE][BLOCK_SIZE], T1 output[BLOCK_SIZE][BLOCK_SIZE], Args args, size_t ty, size_t tx);

template<typename T1, typename T2, class Args>
__parallel__ void stencilKernel(const Array2D<T1>& input, const Array2D<T1>& output, const Mask2D<T2>& mask, const Args& args, size_t h, size_t w);


/**
 * Function signature of the stencil kernel for processing 3-dimensional arrays.
 * This function must be implemented by the user.
 * \param[in] input 1-dimensional Array with the input data.
 * \param[in] output 1-dimensional Array that will hold the output data.
 * \param[in] mask 1-dimensional Mask used to scan through the input data,
 * accessing the neighbourhood for each element.
 * \param[in] args extra arguments that may be used by the stencil computations.
 * \param[in] h height index for the current element to be processed.
 * \param[in] w width index for the current element to be processed.
 * \param[in] d depth index for the current element to be processed.
 **/
template<typename T1, typename T2, class Args>
__parallel__ void stencilKernel(Array3D<T1> input, Array3D<T1> output, Mask3D<T2> mask, Args args, size_t h, size_t w, size_t d);

//*******************************************************************************************
// Stencil Base
//*******************************************************************************************

/**
 * Class that implements the basic functionalities supported by the stencil skeletons.
 **/
template<class Array, class Mask, class Args=int>
class StencilBase{
private:
protected:
	Array input;
	Array output;
	Args args;
	Mask mask;

	virtual void runSeq(Array in, Array out,size_t width, size_t height, size_t maskRange) = 0;
	#ifdef PSKEL_TBB
	virtual void runTBB(Array in, Array out, size_t numThreads) = 0;
	#endif
	virtual inline __attribute__((always_inline)) void runOpenMP(Array in, Array out, size_t width, size_t height, size_t depth, size_t maskRange, size_t numThreads) = 0;
	#ifdef PSKEL_CUDA
	void runCUDA(Array,Array,size_t,size_t);
	void runIterativeTilingCUDA(Array in, Array out, StencilTiling<Array,Mask> tiling, size_t GPUBlockSizeX, size_t GPUBlockSizeY);
	#endif
public:
	
	/**
	 * Executes sequentially in CPU a single iteration of the stencil computation. 
	 **/
	void runSequential();

	/**
	 * Executes in CPU, using multithreads, a single iteration of the stencil computation. 
	 * \param[in] numThreads the number of threads used for processing the stencil kernel.
	 * if numThreads is 0, the number of threads is automatically chosen.
	 **/
	void runCPU(size_t numThreads=0);

	#ifdef PSKEL_CUDA
	/**
	 * Executes in GPU a single iteration of the stencil computation.
	 * This function does not handle data larger than the memory available in the GPU (see runAutoGPU.)
	 * \param[in] GPUBlockSize the block size used for the GPU processing the stencil kernel.
	 * if GPUBlockSize is 0, the block size is automatically chosen.
	 **/
	void runGPU(size_t GPUBlockSizeX=0, size_t GPUBlockSizeY=0);
	#endif

	#ifdef PSKEL_CUDA
	/**
	 * Executes in GPU a single iteration of the stencil computation, tiling the input data.
	 * This function is useful for processing data larger than the memory available in the GPU (see runAutoGPU.)
	 * \param[in] tilingWidth the width size for each (logical) tile of the input data.
	 * \param[in] tilingHeight the height size for each (logical) tile of the input data.
	 * \param[in] tilingDepth the depth size for each (logical) tile of the input data.
	 * \param[in] GPUBlockSize the block size used for the GPU processing the stencil kernel.
	 * if GPUBlockSize is 0, the block size is automatically chosen.
	 **/
	void runTilingGPU(size_t tilingWidth, size_t tilingHeight, size_t tilingDepth, size_t GPUBlockSizeX=0, size_t GPUBlockSizeY=0);
	#endif

	#ifdef PSKEL_CUDA
	/**
	 * Executes in GPU a single iteration of the stencil computation.
	 * If the data is larger than the memory available in the GPU, this function automatically
	 * selects a tiling execution of the stencil computation.
	 * \param[in] GPUBlockSize the block size used for the GPU processing the stencil kernel.
	 * if GPUBlockSize is 0, the block size is automatically chosen.
	 **/
	void runAutoGPU(size_t GPUBlockSize=0);
	#endif

	//void runHybrid(float GPUPartition, size_t GPUBlockSize, size_t numThreads);

	/**
	 * Executes sequentially in CPU multiple iterations of the stencil computation. 
	 * At each given iteration, except the first, the previous output is used as input.
	 * \param[in] iterations the number of iterations to be computed.
	 **/
	void runIterativeSequential(size_t iterations);

	/**
	 * Executes in CPU, using multithreads, multiple iterations of the stencil computation. 
	 * At each given iteration, except the first, the previous output is used as input.
	 * \param[in] iterations the number of iterations to be computed.
	 * \param[in] numThreads the number of threads used for processing the stencil kernel.
	 * if numThreads is 0, the number of threads is automatically chosen.
	 **/
	void runIterativeCPU(size_t iterations, size_t numThreads=0);

	#ifdef PSKEL_CUDA
	/**
	 * Executes in GPU multiple iterations of the stencil computation.
	 * At each given iteration, except the first, the previous output is used as input.
	 * This function does not handle data larger than the memory available in the GPU (see runIterativeAutoGPU.)
	 * \param[in] iterations the number of iterations to be computed.
	 * \param[in] GPUBlockSize the block size used for the GPU processing the stencil kernel.
	 * if GPUBlockSize is 0, the block size is automatically chosen.
	 **/
	void runIterativeGPU(size_t iterations, size_t GPUBlockSizeX, size_t GPUBlockSizeY);
	void runIterativeGPU(size_t iterations, size_t pyramidHeight, size_t GPUBlockSizeX, size_t GPUBlockSizeY);
	#endif

	#ifdef PSKEL_CUDA
	/**
	 * Executes in GPU and CPU multiple iterations of the stencil computation.
	 * The input data is fractioned in two tiles by the gpuFactor.
	 * The first fraction is processed by the GPU and the second by the CPU.
	 * At each given iteration, except the first, the previous output is used as input.
	 * \param[in] iterations the number of iterations to be computed.
	 * \param[in] gpuFactor fraction of input data processed by the GPU.
	 * \param[in] numThreads number of threads used by the CPU.
	 * \param[in] GPUBlockSizeX the block size (x dimension) used for the GPU processing the stencil kernel.
	 * \param[in] GPUBlockSizeY the block size (y dimension) used for the GPU processing the stencil kernel.
	 **/
	void runIterativePartition(size_t iterations, float gpuFactor, size_t numThreads=0, size_t GPUBlockSizeX=32, size_t GPUBlockSizeY=4);
	void runIterativePartition2(size_t iterations, float gpuFactor, size_t numThreads=0, size_t GPUBlockSizeX=32, size_t GPUBlockSizeY=4);
	#endif

	#ifdef PSKEL_CUDA
	/**
	 * Executes in GPU multiple iterations of the stencil computation, tiling the input data.
	 * At each given iteration, except the first, the previous output is used as input.
	 * This function is useful for processing data larger than the memory available in the GPU (see runIterativeAutoGPU.)
	 * \param[in] iterations the number of iterations to be computed.
	 * \param[in] tilingWidth the width size for each (logical) tile of the input data.
	 * \param[in] tilingHeight the height size for each (logical) tile of the input data.
	 * \param[in] tilingDepth the depth size for each (logical) tile of the input data.
	 * \param[in] innerIterations the number of iterations to be consecutively executed on GPU;
	 * the number of iterations executed consecutively on increases the amount of memory required.
	 * \param[in] GPUBlockSize the block size used for the GPU processing the stencil kernel.
	 * if GPUBlockSize is 0, the block size is automatically chosen.
	 **/
	void runIterativeTilingGPU(size_t iterations, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth, size_t innerIterations=1, size_t GPUBlockSizeX=0, size_t GPUBlockSizeY=0);
	#endif

	#ifdef PSKEL_CUDA
	/**
	 * Executes in GPU multiple iterations of the stencil computation.
	 * At each given iteration, except the first, the previous output is used as input.
	 * If the data is larger than the memory available in the GPU, this function automatically
	 * selects a tiling execution of the stencil computation,
	 * including the number of iterations to be consecutivelly executed on GPU.
	 * \param[in] iterations the number of iterations to be computed.
	 * \param[in] GPUBlockSize the block size used for the GPU processing the stencil kernel.
	 * if GPUBlockSize is 0, the block size is automatically chosen.
	 **/
	void runIterativeAutoGPU(size_t iterations, size_t GPUBlockSize=0);
	#endif

	//void runIterativeHybrid(size_t iterations, float GPUPartition, size_t GPUBlockSize, size_t numThreads);
	//
	//void operator()(const tbb::blocked_range<size_t> &r) const;
};
//*******************************************************************************************
// Stencil 3D
//*******************************************************************************************

template<class Array, class Mask, class Args>
class Stencil3D : public StencilBase<Array, Mask, Args>{
protected:
	void runSeq(Array in, Array out, size_t, size_t, size_t);
	void runOpenMP(Array in, Array out, size_t width, size_t height, size_t depth, size_t maskRange, size_t numThreads);
	#ifdef PSKEL_TBB
	void runTBB(Array in, Array out, size_t numThreads);
	#endif
public:
	Stencil3D();
	Stencil3D(Array _input, Array _output, Mask _mask, Args _args);
/*
	void operator()(const tbb::blocked_range<size_t> &r)const{
                size_t begin = r.begin();
                size_t end = r.end();
		size_t wbegin = this->mask.getMaskRange();
		size_t wend = this->input.getWidth()-wbegin;
                #pragma forceinline recursive
                #pragma ivdep
                for (size_t h = begin; h != end; ++h){
                	for (size_t w = wbegin; w < wend; ++w){
                       		stencilKernel(this->input,this->output,this->mask, this->args,h,w);
                       }
		}
        }
*/
};

//*******************************************************************************************
// Stencil 2D
//*******************************************************************************************

template<class Array, class Mask, class Args>
class Stencil2D : public StencilBase<Array, Mask, Args>{
protected:
	inline __attribute__((always_inline)) void runSeq(Array in, Array out, size_t width, size_t height, size_t maskRange);
	inline __attribute__((always_inline)) void runOpenMP(Array in, Array out, size_t width, size_t height, size_t depth, size_t maskRange, size_t numThreads);
	#ifdef PSKEL_TBB
	void runTBB(Array in, Array out, size_t numThreads);
	#endif
public:
	Stencil2D();
	Stencil2D(Array _input, Array _output, Mask _mask, Args _args);
	//~Stencil2D();
	//
/*
	void operator()(const tbb::blocked_range<size_t> &r)const{
                size_t begin = r.begin();
                size_t end = r.end();
		size_t wbegin = this->mask.getMaskRange();
		size_t wend = this->input.getWidth()-wbegin;
                #pragma forceinline recursive
                #pragma ivdep
                for (size_t h = begin; h != end; ++h){
                	for (size_t w = wbegin; w < wend; ++w){
                       		stencilKernel(this->input,this->output,this->mask, this->args,h,w);
                       }
		}
        }
*/                        
};

//*******************************************************************************************
// Stencil 1D
//*******************************************************************************************


template<class Array, class Mask, class Args>
class Stencil: public StencilBase<Array, Mask, Args>{
protected:
	void runSeq(Array in, Array out);
	inline __attribute__((always_inline)) void runOpenMP(Array in, Array out, size_t width, size_t height, size_t depth, size_t maskRange, size_t num_threads);
	#ifdef PSKEL_TBB
	void runTBB(Array in, Array out, size_t numThreads);
	#endif
public:
	Stencil();
	Stencil(Array _input, Array _output, Mask _mask, Args _args);

/*	void operator()(const tbb::blocked_range<size_t> &r)const{
                size_t begin = r.begin();
                size_t end = r.end();
		//size_t wbegin = this->mask.getMaskRange();
		//size_t wend = this->input.getWidth()-wbegin;
                #pragma forceinline recursive
                #pragma ivdep
                for (size_t w = begin; w != end; ++w){
                       		stencilKernel(this->input,this->output,this->mask, this->args,w);
		}
        }*/
};

}

#include "PSkelStencil.hpp"

#endif
