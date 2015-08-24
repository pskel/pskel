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


#ifndef PSKEL_ARGS_H
#define PSKEL_ARGS_H

namespace PSkel{

//*******************************************************************************************
// ARGS
//*******************************************************************************************

template<typename T>
class Args{
public:
	T *hostArray;
	#ifdef PSKEL_CUDA
	T *deviceArray;
	#endif
	int width;
	
	Args();
	
	/*
	__stencil__ ~Args(){
		gpuErrchk( cudaFreeHost(hostArray));
	}
	*/
	
	Args(int _width);
	
	__device__ __host__ int getWidth() const;
	
	__device__ __host__ T & operator()(int x) const;
};

//*******************************************************************************************
// ARGS2D
//*******************************************************************************************

template<typename T>
class Args2D{
public:
	T *hostArray;
	#ifdef PSKEL_CUDA
	T *deviceArray;
	#endif
	int width, height;
	
	Args2D();
	
	/*
	__stencil__ ~Args(){
		gpuErrchk( cudaFreeHost(hostArray));
	}
	*/
	
	Args2D(int _width,int _height);
	
	__device__ __host__ int getWidth() const;
	
	__device__ __host__ int getHeight() const;
	
	__device__ __host__ T & operator()(int x,int y) const;
};

//*******************************************************************************************
// ARGS3D
//*******************************************************************************************

template<typename T>
class Args3D{
public:
	T *hostArray;
	#ifdef PSKEL_CUDA
	T *deviceArray;
	#endif
	int width,height,depth;
	
	Args3D();
	
	/*
	__stencil__ ~Args(){
		gpuErrchk( cudaFreeHost(hostArray));
	}
	*/
	
	Args3D(int _width, int _height, int _depth);
	
	__device__ __host__ int getWidth() const;
	
	__device__ __host__ int getHeight() const;
	
	__device__ __host__ int getDepth() const;	
	
	__device__ __host__ T & operator()(int x,int y,int z) const;
};

}//end namespace

#include "PSkelArgs.hpp"

#endif
