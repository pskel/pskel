//-------------------------------------------------------------------------------
// Copyright (c) 2015, Alyson D. Pereira <alyson.deives@outlook.com>,
//					   Rodrigo C. O. Rocha <rcor.cs@gmail.com>
//					   
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

/*! \mainpage PSkel: High-performance parallel skeletons
 *
 * \section intro_sec Introduction

PSkel is a high-performance framework for parallel skeletons.
Using a high-level abstraction for parallel skeletons, PSkel releases the
programmer from the responsibility of writing boiler-plate code for parallel
programming in heterogeneous architectures, e.g., explicit synchronization and
data exchanges between GPU memory and main memory.
Furthermore, the framework translates the abstractions described using its
application programming interface (API) into lowlevel C++ code compatible with
Intel TBB, OpenMP and NVIDIA CUDA.
PSkel's API is mainly based on a C++ template library that implements parallel
skeletons and provides useful constructs for developing parallel applications.
The framework provides an API for manipulating input and output data;
specifying stencil computations; encapsulating memory management, computations, and
runtime details.

 **/

/**
 * \file PSkel.h
 * This file contains all the includes required for using the PSkel framework.
*/
#ifndef PSKEL_H
#define PSKEL_H

#include <stdio.h>
#include <cstdio>
#include <stdlib.h>
#include <time.h>
#include <typeinfo>
#include <iostream>
#include <omp.h>

#ifdef PSKEL_CUDA
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif

#ifdef PSKEL_TBB
    #include <tbb/blocked_range.h>
    #include <tbb/parallel_for.h>
    #include <tbb/task_scheduler_init.h>
#endif

#ifdef DEBUG
    #include "hr_time.h"
#endif

#ifdef PSKEL_PAPI
	#include  "PSkelPAPI.h"
#endif

#ifdef PSKEL_GA
    #include <ga/ga.h>
    #include <ga/std_stream.h>
#endif

#include "PSkelDefs.h"
#include "PSkelArray.h"

#ifdef PSKEL_CUDA
    #include "PSkelArgs.h"
#endif

#include "PSkelMask.h"
#include "PSkelStencil.h"

#endif
