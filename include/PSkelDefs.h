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
 * \file PSkelDefs.h
 * This file contains basic macro definitions used throughout PSkel.
*/
#ifndef PSKEL_DEFS_H
#define PSKEL_DEFS_H

#include <stdio.h>

#define MIN(x,y) (x < y ? x : y)

#ifdef PSKEL_CUDA
 #include <cuda.h>
 #include <cuda_runtime_api.h>
 #define __parallel__ __device__ __host__ __attribute__((always_inline)) __forceinline__
 #define gpuErrchk(ans) { gpuAssert(((cudaError_t)ans), __FILE__, __LINE__,true); }
 inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false){
	if(code!=cudaSuccess){
		fprintf(stderr,(const char*)"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if(abort) exit(code);
	}
 }
#else
 #define __device__
 #define __host__
 #define __parallel__ inline

 #define __forceinline__ inline
#endif

#endif

#ifdef DEBUG
//	namespace PSkel{hr_timer_t PSkelTimer;}
#endif
