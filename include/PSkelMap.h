//-------------------------------------------------------------------------------
// Copyright (c) 2015, ICEI - PUC Minas
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

#ifndef PSKEL_MAP_H
#define PSKEL_MAP_H

#ifdef PSKEL_CUDA
  #include <cuda.h>
#endif

#ifdef PSKEL_TBB
  #include <tbb/blocked_range.h>
#endif

#include "PSkelDefs.h"
#include "PSkelArray.h"

namespace PSkel{

//*******************************************************************************************
// Kernels a serem implementados pelo usuário
//*******************************************************************************************

template<typename T, class Args>
__parallel__ void mapKernel(Array<T> input, Array<T> output, Args args, size_t i);

template<typename T, class Args>
__parallel__ void mapKernel(Array2D<T> input, Array2D<T> output, Args args, size_t h, size_t w);

template<typename T, class Args>
__parallel__ void mapKernel(Array3D<T> input, Array3D<T> output, Args args, size_t h, size_t w, size_t d);

//*******************************************************************************************
// Map Base
//*******************************************************************************************


template<class Arrays, class Args=int>
class MapBase{
private:
protected:
	Arrays input;
	Arrays output;
	Args args;
	
	virtual void runSeq(Arrays in, Arrays out) = 0;
	virtual void runOpenMP(Arrays in, Arrays out, size_t numThreads) = 0;
	#ifdef PSKEL_TBB
	virtual void runTBB(Arrays in, Arrays out, size_t numThreads) = 0;
	#endif
	#ifdef PSKEL_CUDA
	virtual void runCUDA(Arrays input, Arrays output, size_t blockSize) = 0;
	#endif
public:
	void runSequential();
	void runCPU(size_t numThreads=0);
	#ifdef PSKEL_CUDA
	void runGPU(size_t blockSize=0);
	#endif
	//void runHybrid(float GPUPartition, size_t GPUBlockSize, size_t numThreads);

	void runIterativeSequential(size_t iterations);
	void runIterativeCPU(size_t iterations, size_t numThreads=0);
	#ifdef PSKEL_CUDA
	void runIterativeGPU(size_t iterations, size_t blockSize=0);
	#endif
	//void runIterativeHybrid(size_t iterations, float GPUPartition, size_t GPUBlockSize, size_t numThreads);
};

//*******************************************************************************************
// Map 3D
//*******************************************************************************************


template<class Arrays, class Args>
class Map3D : public MapBase<Arrays, Args>{
protected:
	void runSeq(Arrays in, Arrays out);
	void runOpenMP(Arrays in, Arrays out, size_t numThreads);
	#ifdef PSKEL_TBB
	void runTBB(Arrays in, Arrays out, size_t numThreads);
	#endif
	#ifdef PSKEL_CUDA
	void runCUDA(Arrays in, Arrays out, size_t blockSize);
	#endif
public:
	Map3D();
	Map3D(Arrays input, Arrays output, Args args);
};

//*******************************************************************************************
// Stencil 2D
//*******************************************************************************************

template<class Arrays, class Args>
class Map2D : public MapBase<Arrays, Args>{
protected:
	void runSeq(Arrays in, Arrays out);
	void runOpenMP(Arrays in, Arrays out, size_t numThreads);
	#ifdef PSKEL_TBB
	void runTBB(Arrays in, Arrays out, size_t numThreads);
	#endif
	#ifdef PSKEL_CUDA
	void runCUDA(Arrays in, Arrays out, size_t blockSize);
	#endif
public:
	Map2D();
	Map2D(Arrays input, Arrays output, Args args);
};

//*******************************************************************************************
// Stencil 1D
//*******************************************************************************************


template<class Arrays, class Args>
class Map: public MapBase<Arrays, Args>{
protected:
	void runSeq(Arrays in, Arrays out);
	void runOpenMP(Arrays in, Arrays out, size_t numThreads);
	#ifdef PSKEL_TBB
	void runTBB(Arrays in, Arrays out, size_t numThreads);
	#endif
	#ifdef PSKEL_CUDA
	void runCUDA(Arrays in, Arrays out, size_t blockSize);
	#endif
public:
	Map();
	Map(Arrays input, Arrays output, Args args);
};

}//end namespace

#include "PSkelMap.hpp"

#endif
