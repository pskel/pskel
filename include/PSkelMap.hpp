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

#ifndef PSKEL_MAP_HPP
#define PSKEL_MAP_HPP

#include <algorithm>
#include <cmath>

#ifdef PSKEL_TBB
  #include <tbb/blocked_range.h>
  #include <tbb/parallel_for.h>
  #include <tbb/task_scheduler_init.h>
#endif

namespace PSkel{

#ifdef PSKEL_CUDA
//********************************************************************************************
// Kernels CUDA. Chama o kernel implementado pelo usuario
//********************************************************************************************

template<typename T1, class Args>
__global__ void mapCU(Array<T1> input,Array<T1> output,Args args);

template<typename T1, class Args>
__global__ void mapCU2D(Array2D<T1> input,Array2D<T1> output,Args args);

template<typename T1, class Args>
__global__ void mapCU3D(Array3D<T1> input,Array3D<T1> output,Args args);


//********************************************************************************************
// Kernels CUDA. Chama o kernel implementado pelo usuario
//********************************************************************************************

template<typename T1, class Args>
__global__ void mapCU(Array<T1> input,Array<T1> output, Args args){
	size_t i = blockIdx.x*blockDim.x+threadIdx.x;
	
	if(i<input.getWidth()){
		mapKernel(input, output, args, i);
	}
}

template<typename T1, class Args>
__global__ void mapCU2D(Array2D<T1> input,Array2D<T1> output,Args args){
	size_t w = blockIdx.x*blockDim.x+threadIdx.x;
	size_t h = blockIdx.y*blockDim.y+threadIdx.y;
  
	if(w<input.getWidth() && h<input.getHeight()){
		mapKernel(input, output, args, h, w);
	}
}

template<typename T1, class Args>
__global__ void mapCU3D(Array3D<T1> input,Array3D<T1> output,Args args){
	size_t w = blockIdx.x*blockDim.x+threadIdx.x;
	size_t h = blockIdx.y*blockDim.y+threadIdx.y;
	size_t d = blockIdx.z*blockDim.z+threadIdx.z;
  
	if(w<input.getWidth() && h<input.getHeight() && d<input.getDepth()){
		mapKernel(input, output, args, h, w,d);
	}
}
#endif

//*******************************************************************************************
// Stencil Base
//*******************************************************************************************

template<class Arrays, class Args>
void MapBase<Arrays, Args>::runSequential(){
	this->runSeq(this->input, this->output);
}

template<class Arrays, class Args>
void MapBase<Arrays, Args>::runCPU(size_t numThreads){
	numThreads = (numThreads==0)?omp_get_num_procs():numThreads;
	#ifdef PSKEL_TBB
		this->runTBB(this->input, this->output, numThreads);
	#else
		this->runOpenMP(this->input, this->output, numThreads);
	#endif
}

#ifdef PSKEL_CUDA
template<class Arrays, class Args>
void MapBase<Arrays, Args>::runGPU(size_t blockSize){
	if(blockSize==0){
		int device;
		cudaGetDevice(&device);
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, device);
		blockSize = deviceProperties.warpSize;
	}
	input.deviceAlloc();
	output.deviceAlloc();
	input.copyToDevice();
	this->runCUDA(this->input, this->output, blockSize);
	output.copyToHost();
	input.deviceFree();
	output.deviceFree();
}
#endif

/*
template<class Arrays, class Args>
void MapBase<Arrays, Args>::runHybrid(float GPUPartition, size_t GPUBlockSize, size_t numThreads){
	if(GPUPartition==0.0){
		runCPU(numThreads);
	}else if(GPUPartition==1.0){
		runGPU(GPUBlockSize);
	}else{
		Arrays inputSliceGPU;
		Arrays outputSliceGPU;
		Arrays inputSliceCPU;
		Arrays outputSliceCPU;
		if(input.getHeight()>1){
			size_t GPUHeight = size_t(this->input.getHeight()*GPUPartition);
			inputSliceGPU.hostSlice(this->input, 0, 0, 0, this->input.getWidth(), GPUHeight, this->input.getDepth());
			outputSliceGPU.hostSlice(this->output, 0, 0, 0, this->output.getWidth(), GPUHeight, this->output.getDepth());
			inputSliceCPU.hostSlice(this->input, 0, GPUHeight, 0, this->input.getWidth(), this->input.getHeight()-GPUHeight, this->input.getDepth());
			outputSliceCPU.hostSlice(this->output, 0, GPUHeight, 0, this->output.getWidth(), this->output.getHeight()-GPUHeight, this->output.getDepth());
		}else{
			size_t GPUWidth= size_t(this->input.getWidth()*GPUPartition);
			inputSliceGPU.hostSlice(this->input, 0, 0, 0, GPUWidth, this->input.getHeight(), this->input.getDepth());
			outputSliceGPU.hostSlice(this->output, 0, 0, 0, GPUWidth, this->output.getHeight(), this->output.getDepth());
			inputSliceCPU.hostSlice(this->input, GPUWidth, 0, 0, this->input.getWidth()-GPUWidth, this->input.getHeight(), this->input.getDepth());
			outputSliceCPU.hostSlice(this->output, GPUWidth, 0, 0, this->output.getWidth()-GPUWidth, this->output.getHeight(), this->output.getDepth());
		}
		omp_set_num_threads(2);
	
		#pragma omp parallel sections
		{
			#pragma omp section
			{	
				inputSliceGPU.deviceAlloc();
				inputSliceGPU.copyToDevice();
				outputSliceGPU.deviceAlloc();
				this->runCUDA(inputSliceGPU, outputSliceGPU, GPUBlockSize);
				outputSliceGPU.copyToHost();
	
				inputSliceGPU.deviceFree();
				outputSliceGPU.deviceFree();
			}
			#pragma omp section
			{
				this->runTBB(inputSliceCPU, outputSliceCPU,numThreads);
			}
		}
	}
}
*/

template<class Arrays, class Args>
void MapBase<Arrays, Args>::runIterativeSequential(size_t iterations){
	Arrays inputCopy;
	inputCopy.hostClone(input);
	for(size_t it = 0; it<iterations; it++){
		if(it%2==0) this->runSeq(inputCopy, this->output);
		else this->runSeq(this->output, inputCopy);
	}
	if((iterations%2)==0) output.hostMemCopy(inputCopy);
	inputCopy.hostFree();
}

template<class Arrays, class Args>
void MapBase<Arrays, Args>::runIterativeCPU(size_t iterations, size_t numThreads){
	numThreads = (numThreads==0)?omp_get_num_procs():numThreads;
	Arrays inputCopy;
	inputCopy.hostClone(input);
	for(size_t it = 0; it<iterations; it++){
		if(it%2==0){
			#ifdef PSKEL_TBB
				this->runTBB(inputCopy, this->output, numThreads);
			#else
				this->runOpenMP(inputCopy, this->output, numThreads);
			#endif
		}else {
			#ifdef PSKEL_TBB
				this->runTBB(this->output, inputCopy, numThreads);
			#else
				this->runOpenMP(this->output, inputCopy, numThreads);
			#endif
		}
	}
	if((iterations%2)==0) output.hostMemCopy(inputCopy);
	inputCopy.hostFree();
}

#ifdef PSKEL_CUDA
template<class Arrays, class Args>
void MapBase<Arrays, Args>::runIterativeGPU(size_t iterations, size_t blockSize){
	if(blockSize==0){
		int device;
		cudaGetDevice(&device);
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, device);
		blockSize = deviceProperties.warpSize;
	}
	input.deviceAlloc();
	input.copyToDevice();
	output.deviceAlloc();
	for(size_t it = 0; it<iterations; it++){
		if((it%2)==0)
			this->runCUDA(this->input, this->output, blockSize);
		else this->runCUDA(this->output, this->input, blockSize);
	}
	if((iterations%2)==1)
		output.copyToHost();
	else output.copyFromDevice(input);
	input.deviceFree();
	output.deviceFree();
}
#endif

/*
template<class Arrays, class Args>
void MapBase<Arrays, Args>::runIterativeHybrid(size_t iterations, float GPUPartition, size_t GPUBlockSize, size_t numThreads){
	if(GPUPartition==0.0){
		runIterativeCPU(iterations, numThreads);
	}else if(GPUPartition==1.0){
		runIterativeGPU(iterations, GPUBlockSize);
	}else{
		Arrays inputSliceGPU;
		Arrays outputSliceGPU;
		Arrays inputSliceCPU;
		Arrays outputSliceCPU;
		if(input.getHeight()>1){
			size_t GPUHeight = size_t(this->input.getHeight()*GPUPartition);
			inputSliceGPU.hostSlice(this->input, 0, 0, 0, this->input.getWidth(), GPUHeight, this->input.getDepth());
			outputSliceGPU.hostSlice(this->output, 0, 0, 0, this->output.getWidth(), GPUHeight, this->output.getDepth());
			inputSliceCPU.hostSlice(this->input, 0, GPUHeight, 0, this->input.getWidth(), this->input.getHeight()-GPUHeight, this->input.getDepth());
			outputSliceCPU.hostSlice(this->output, 0, GPUHeight, 0, this->output.getWidth(), this->output.getHeight()-GPUHeight, this->output.getDepth());
		}else{
			size_t GPUWidth= size_t(this->input.getWidth()*GPUPartition);
			inputSliceGPU.hostSlice(this->input, 0, 0, 0, GPUWidth, this->input.getHeight(), this->input.getDepth());
			outputSliceGPU.hostSlice(this->output, 0, 0, 0, GPUWidth, this->output.getHeight(), this->output.getDepth());
			inputSliceCPU.hostSlice(this->input, GPUWidth, 0, 0, this->input.getWidth()-GPUWidth, this->input.getHeight(), this->input.getDepth());
			outputSliceCPU.hostSlice(this->output, GPUWidth, 0, 0, this->output.getWidth()-GPUWidth, this->output.getHeight(), this->output.getDepth());
		}
		
		omp_set_num_threads(2);
	
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				inputSliceGPU.deviceAlloc();
				inputSliceGPU.copyToDevice();
				outputSliceGPU.deviceAlloc();
				for(size_t it = 0; it<iterations; it++){
					if((it%2)==0)
						this->runCUDA(inputSliceGPU, outputSliceGPU, GPUBlockSize);
					else this->runCUDA(outputSliceGPU, inputSliceGPU, GPUBlockSize);
				}
				//outputSliceGPU.copyToHost();
	
				//outputSliceGPU.deviceFree();
			}
			#pragma omp section
			{
				Arrays inputCopy;
				inputCopy.hostClone(inputSliceCPU);
				for(size_t it = 0; it<iterations; it++){
					if(it%2==0) this->runTBB(inputCopy, outputSliceCPU, numThreads);
					else this->runTBB(outputSliceCPU, inputCopy, numThreads);
					//std::swap(input,output);
				}
				if((iterations%2)==0) outputSliceCPU.hostMemCopy(inputCopy);
				inputCopy.hostFree();
			}
		}
		if((iterations%2)==1)
			outputSliceGPU.copyToHost();
		else outputSliceGPU.copyFromDevice(inputSliceGPU);
		inputSliceGPU.deviceFree();
		outputSliceGPU.deviceFree();
	}
}
*/
//*******************************************************************************************
// Map 3D
//*******************************************************************************************

template<class Arrays, class Args>
Map3D<Arrays,Args>::Map3D(){}
	
template<class Arrays, class Args>
Map3D<Arrays,Args>::Map3D(Arrays input, Arrays output, Args args){
	this->input = input;
	this->output = output;
	this->args = args;
}

#ifdef PSKEL_CUDA
template<class Arrays, class Args>
void Map3D<Arrays,Args>::runCUDA(Arrays in, Arrays out, size_t blockSize){
	dim3 DimBlock(blockSize, blockSize, 1);
	dim3 DimGrid((in.getWidth() - 1)/blockSize + 1, ((in.getHeight()) - 1)/blockSize + 1, in.getDepth());
			
	mapCU3D<<<DimGrid, DimBlock>>>(in, out, this->args);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}
#endif

template<class Arrays, class Args>
void Map3D<Arrays,Args>::runSeq(Arrays in, Arrays out){
	for(int h = 0; h<in.getHeight(); ++h){
	for(int w = 0; w<in.getWidth(); ++w){
	for(int d = 0; d<in.getDepth(); ++d){
		mapKernel(in, out, this->args, h, w,d);
	}}}
}

template<class Arrays, class Args>
void Map3D<Arrays,Args>::runOpenMP(Arrays in, Arrays out, size_t numThreads){
	omp_set_num_threads(numThreads);
	#pragma omp parallel for
	for(int h = 0; h<in.getHeight(); ++h){
	for(int w = 0; w<in.getWidth(); ++w){
	for(int d = 0; d<in.getDepth(); ++d){
		mapKernel(in, out, this->args, h, w,d);
	}}}
}

#ifdef PSKEL_TBB
template<class Arrays, class Args>
struct TBBMap3D{
	Arrays input;
	Arrays output;
	Args args;
	TBBMap3D(Arrays input, Arrays output, Args args){
		this->input = input;
		this->output = output;
		this->args = args;
	}
	void operator()(tbb::blocked_range<int> r)const{
		for(int h = r.begin(); h!=r.end(); h++){
		for(int w = 0; w<this->input.getWidth(); ++w){
		for(int d = 0; d<this->input.getDepth(); ++d){
			mapKernel(this->input, this->output, this->args, h, w,d);
		}}}
	}
};

template<class Arrays, class Args>
void Map3D<Arrays, Args>::runTBB(Arrays in, Arrays out, size_t numThreads){
	TBBMap3D<Arrays, Args> tbbmap(in, out, this->args);
	tbb::task_scheduler_init init(numThreads);
	tbb::parallel_for(tbb::blocked_range<int>(0, in.getHeight()), tbbmap);
}
#endif

//*******************************************************************************************
// Map 2D
//*******************************************************************************************

template<class Arrays, class Args>
Map2D<Arrays,Args>::Map2D(){}

template<class Arrays, class Args>
Map2D<Arrays,Args>::Map2D(Arrays input, Arrays output, Args args){
	this->input = input;
	this->output = output;
	this->args = args;
}

#ifdef PSKEL_CUDA
template<class Arrays, class Args>
void Map2D<Arrays,Args>::runCUDA(Arrays in, Arrays out, size_t blockSize){
	dim3 DimBlock(blockSize, blockSize, 1);
	dim3 DimGrid((in.getWidth() - 1)/blockSize + 1, (in.getHeight() - 1)/blockSize + 1, 1);
			
	mapCU2D<<<DimGrid, DimBlock>>>(in, out, this->args);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );	
	//gpuErrchk( cudaGetLastError() );	
}
#endif

template<class Arrays, class Args>
void Map2D<Arrays,Args>::runSeq(Arrays in, Arrays out){
	for (int h = 0; h < in.getHeight(); h++){
	for (int w = 0; w < in.getWidth(); w++){
		mapKernel(in, out, this->args,h,w);
	}}
}

template<class Arrays, class Args>
void Map2D<Arrays,Args>::runOpenMP(Arrays in, Arrays out, size_t numThreads){
	omp_set_num_threads(numThreads);
	#pragma omp parallel for
	for (int h = 0; h < in.getHeight(); h++){
	for (int w = 0; w < in.getWidth(); w++){
		mapKernel(in, out, this->args,h,w);
	}}
}

#ifdef PSKEL_TBB
template<class Arrays, class Args>
struct TBBMap2D{
	Arrays input;
	Arrays output;
	Args args;
	TBBMap2D(Arrays input, Arrays output, Args args){
		this->input = input;
		this->output = output;
		this->args = args;
	}
	void operator()(tbb::blocked_range<int> r)const{
		for (int h = r.begin(); h != r.end(); h++){
		for (int w = 0; w < this->input.getWidth(); w++){
			mapKernel(this->input, this->output, this->args,h,w);
		}}
	}
};

template<class Arrays, class Args>
void Map2D<Arrays, Args>::runTBB(Arrays in, Arrays out, size_t numThreads){
	TBBMap2D<Arrays, Args> tbbmap(in, out, this->args);
	tbb::task_scheduler_init init(numThreads);
	tbb::parallel_for(tbb::blocked_range<int>(0, in.getHeight()), tbbmap);
}
#endif

//*******************************************************************************************
// Stencil 1D
//*******************************************************************************************


template<class Arrays, class Args>
Map<Arrays,Args>::Map(){}
	
template<class Arrays, class Args>
Map<Arrays,Args>::Map(Arrays input, Arrays output, Args args){
	this->input = input;
	this->output = output;
	this->args = args;
}

#ifdef PSKEL_CUDA
template<class Arrays, class Args>
void Map<Arrays,Args>::runCUDA(Arrays in, Arrays out, size_t blockSize){
	dim3 DimBlock(blockSize, 1, 1);
	dim3 DimGrid((in.getWidth() - 1)/blockSize + 1,1,1);
			
	mapCU<<<DimGrid, DimBlock>>>(in, out, this->args);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );		
}
#endif

template<class Arrays, class Args>
void Map<Arrays,Args>::runSeq(Arrays in, Arrays out){
	for (int i = 0; i < in.getWidth(); i++){
		mapKernel(in, out, this->args, i);
	}
}

template<class Arrays, class Args>
void Map<Arrays,Args>::runOpenMP(Arrays in, Arrays out, size_t numThreads){
	omp_set_num_threads(numThreads);
	#pragma omp parallel for
	for (int i = 0; i < in.getWidth(); i++){
		mapKernel(in, out, this->args, i);
	}
}

#ifdef PSKEL_TBB
template<class Arrays, class Args>
struct TBBMap{
	Arrays input;
	Arrays output;
	Args args;
	TBBMap(Arrays input, Arrays output, Args args){
		this->input = input;
		this->output = output;
		this->args = args;
	}
	void operator()(tbb::blocked_range<int> r)const{
		for (int i = r.begin(); i != r.end(); i++){
			mapKernel(this->input, this->output, this->args, i);
		}
	}
};

template<class Arrays, class Args>
void Map<Arrays, Args>::runTBB(Arrays in, Arrays out, size_t numThreads){
	TBBMap<Arrays, Args> tbbmap(in, out, this->args);
	tbb::task_scheduler_init init(numThreads);
	tbb::parallel_for(tbb::blocked_range<int>(0, in.getWidth()), tbbmap);
}
#endif

}//end namespace

#endif
