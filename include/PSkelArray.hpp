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

#ifndef PSKEL_ARRAY_HPP
#define PSKEL_ARRAY_HPP

#define VALUE_TO_STRING(x) #x
#define VALUE(x) VALUE_TO_STRING(x)
#define VAR_NAME_VALUE(var) #var "="  VALUE(var)

///#include <cstring>
//#include <omp.h>
#include<iostream>
#include <xmmintrin.h>
#ifdef __INTEL_COMPILER
#  define _MM_MALLOC_H_INCLUDED 1 /* disables gcc's <mm_malloc.h>, for Intel */
#endif

namespace PSkel{

template<typename T>
ArrayBase<T>::ArrayBase(size_t width, size_t height, size_t depth){
	this->width = width;
	this->height = height;
	this->depth = depth;
	this->realWidth = width;
	this->realHeight = height;
	this->realDepth = depth;
	this->widthOffset = 0;
	this->heightOffset = 0;
	this->depthOffset = 0;
	this->hostArray = 0;
	this->haloValue = (T) 0;
	this->haloValuePtr = &(this->haloValue);
	#ifdef PSKEL_CUDA
    	this->deviceArray = NULL;
	this->hostGPUArray = NULL;
	#endif
	
	if(size()>0) this->hostAlloc();	
}

#ifdef PSKEL_CUDA
template<typename T>
__forceinline__ void ArrayBase<T>::deviceAlloc(){
	#ifndef PSKEL_MANAGED
	if(this->deviceArray==NULL){
		gpuErrchk( cudaMalloc((void **) &deviceArray, size()*sizeof(T)) );
		//printf("Allocated %lld bytes in the GPU\n",size()*sizeof(T));
		//cudaMemset(this->deviceArray, 0, size()*sizeof(T));
	}
	#endif
}
#endif

#ifdef PSKEL_CUDA
template<typename T>
__forceinline__ void ArrayBase<T>::deviceFree(){
	#ifndef PSKEL_MANAGED
	if(this->deviceArray!=NULL){
		cudaFree(this->deviceArray);
		this->deviceArray = NULL;
	}
	#endif
}
#endif

template<typename T>
void ArrayBase<T>::hostAlloc(size_t width, size_t height, size_t depth){
	this->width = width;
	this->height = height;
	this->depth = depth;
	this->realWidth = width;
	this->realHeight = height;
	this->realDepth = depth;
	this->widthOffset = 0;
	this->heightOffset = 0;
	this->depthOffset = 0;
	this->hostArray = NULL;
	#ifdef PSKEL_CUDA
	this->deviceArray = NULL;
	this->hostGPUArray = NULL;
	#endif

	this->hostAlloc();
	
	#ifdef PSKEL_CUDA
        //gpuErrchk( cudaHostAlloc((void**)&hostGPUArray, size()*sizeof(T),cudaHostAllocWriteCombined) );
        //std::cout<<"Host Pinned memory allocated"<<std::endl;
	#endif
}

template<typename T>
void ArrayBase<T>::hostScalableAlloc(){
	//#ifdef PSKEL_TBB
	this->hostArray = (T*) scalable_malloc(size()*sizeof(T));
        std::cout<<"Host scalable memory allocated"<<std::endl;	
	//#endif
}

template<typename T>
void ArrayBase<T>::hostScalableAlloc(size_t _width, size_t _height, size_t _depth){
	this->width = _width;
	this->height = _height;
	this->depth = _depth;
	this->widthOffset = 0;
	this->heightOffset = 0;
	this->depthOffset = 0;
	this->realWidth = _width;
	this->realHeight = _height;
	this->realDepth = _depth;
	//Alloc scalable memory
	this->hostArray = NULL;
	#ifdef PSKEL_TBBALLOC
	this->hostScalableAlloc();
	#else
	std::cout<<"Warning! Allocating non-scalable memory"<<std::endl;
	this->hostAlloc();
	//this->hostScalableAlloc();	
	#endif
	//Copy clone memory
	//this->hostMemCopy(array);
}


//TODO When using CUDA, a page-locked memory is allocated. Verify the diference
//in performance between cudaMallocHost and cudaHostAlloc and its flags.
//It seems that Cloudsim CPU Performance is worst with cudaMallocHost.
template<typename T>
__forceinline__ void ArrayBase<T>::hostAlloc(){
	/*if(this->hostArray==NULL){
	#ifdef PSKEL_MANAGED
		cudaMallocManaged((void**)&hostArray,size()*sizeof(T));
	#else
	#ifdef PSKEL_CUDA
            //gpuErrchk( cudaMallocHost((void**)&hostArray, size()*sizeof(T)) );
	    gpuErrchk( cudaHostAlloc((void**)&hostArray, size()*sizeof(T),cudaHostAllocPortable) );
            //cudaMemset(this->hostArray, 0, size()*sizeof(T));
        #else
	#ifdef PSKEL_TBB
	    this->hostArray = (T*) scalable_malloc(size()*sizeof(T));	
    	#else
            //this->hostArray = (T*) calloc(size(), sizeof(T));
            this->hostArray = (T*) malloc(size()*sizeof(T));
    	#endif
	#endif
	#endif
	#ifdef DEBUG
		printf("Array allocated at address %p\n",(void*)&(this->hostArray));
	#endif
	}
	*/
	if(this->hostArray==NULL){
	#ifdef PSKEL_MANAGED
		cudaMallocManaged((void**)&hostArray,size()*sizeof(T));
	#else
	#ifdef PSKEL_CUDA
            //gpuErrchk( cudaMallocHost((void**)&hostArray, size()*sizeof(T)) );
	    //gpuErrchk( cudaHostAlloc((void**)&hostGPUArray, size()*sizeof(T),cudaHostAllocWriteCombined) );
	    //std::cout<<"Host Pinned memory allocated"<<std::endl;
            //cudaMemset(this->hostArray, 0, size()*sizeof(T));
        #endif
	#endif
	#ifdef PSKEL_TBBALLOC
	    //this->hostArray = (T*) scalable_malloc(size()*sizeof(T));	
	    //std::cout<<"Host scalable memory allocated"<<std::endl;
	    this->hostScalableAlloc();
    	#else
            //this->hostArray = (T*) scalable_malloc(size()*sizeof(T));	
	    //std::cout<<"Host scalable memory allocated"<<std::endl;
	    //this->hostArray = (T*) calloc(size(), sizeof(T));
            //this->hostArray = (T*) malloc(size()*sizeof(T));
 	    this->hostArray = (T*) _mm_malloc(size()*sizeof(T),16); /* aligned malloc */
	    std::cout << "Aligned memory allocated" << std::endl;
    	#endif
	#ifdef DEBUG
		printf("Array allocated at address %p\n",(void*)&(this->hostArray));
	#endif
	}
	else{
		std::cout << "Host Array Pointer already allocated" << std::endl;
	}

}

template<typename T>
void ArrayBase<T>::hostAllocPinned(){
	gpuErrchk( cudaHostAlloc((void**)&hostGPUArray, size()*sizeof(T),cudaHostAllocWriteCombined) );
        std::cout<<"Host Pinned memory allocated"<<std::endl;
}


template<typename T>
void ArrayBase<T>::hostAllocPinned(size_t _width, size_t _height, size_t _depth){
	this->width = _width;
        this->height = _height;
        this->depth = _depth;
        this->widthOffset = 0;
        this->heightOffset = 0;
        this->depthOffset = 0;
        this->realWidth = _width;
        this->realHeight = _height;
        this->realDepth = _depth;

	gpuErrchk( cudaHostAlloc((void**)&hostGPUArray, size()*sizeof(T),cudaHostAllocWriteCombined) );
        std::cout<<"Host Pinned memory allocated"<<std::endl;
}



template<typename T>
__forceinline__ void ArrayBase<T>::hostFree(){
/*	if(this->hostArray!=NULL){
	gpuErrchk( cudaHostAlloc((void**)&hostGPUArray, size()*sizeof(T),cudaHostAllocWriteCombined) );
            std::cout<<"Host Pinned memory allocated"<<std::endl;
#ifdef PSKEL_MANAGED
		cudaFree(this->hostArray);
	#else
	#ifdef PSKEL_CUDA
		gpuErrchk( cudaFreeHost(this->hostArray) );
	#else
	#ifdef PSKEL_TBB
		scalable_free(this->hostArray);
	#else
		free(this->hostArray);
	#endif	
	#endif
	#endif
	this->hostArray = NULL;
	}
*/
	if(this->hostArray!=NULL){
	#ifdef PSKEL_MANAGED
		cudaFree(this->hostArray);
	#endif
	#ifdef PSKEL_TBB
		scalable_free(this->hostArray);
	#else
		//free(this->hostArray);
		_mm_free(this->hostArray);	
		//scalable_free(this->hostArray);
	#endif	
	this->hostArray = NULL;
	}

	#ifdef PSKEL_CUDA
		if(this->hostGPUArray != NULL){
			gpuErrchk( cudaFreeHost(this->hostGPUArray) );
			this->hostGPUArray = NULL;
		}
	#endif
	
}

template<typename T>
inline void ArrayBase<T>::cudaHostMemCopy(){
	#ifdef PSKEL_CUDA
	//gpuErrchk ( cudaMemcpy(hostGPUArray, hostArray, size()*sizeof(T), cudaMemcpyHostToHost) );
	memcpy(this->hostGPUArray, this->hostArray, size()*sizeof(T));
	#endif
}

template<typename T>
__forceinline __host__ __device__ size_t ArrayBase<T>::getWidth() const{
	return width;
}
	
template<typename T>
__forceinline __host__ __device__ size_t ArrayBase<T>::getHeight() const{
	return height;
}

template<typename T>
__forceinline __host__ __device__ size_t ArrayBase<T>::getDepth() const{
	return depth;
}

template<typename T>
__forceinline __host__ __device__ size_t ArrayBase<T>::getWidthOffset() const{
        return widthOffset;
}

template<typename T>
__forceinline __host__ __device__ size_t ArrayBase<T>::getHeightOffset() const{
        return heightOffset;
}

template<typename T>
__forceinline __host__ __device__ size_t ArrayBase<T>::getRealHeight() const{
        return realHeight;
}

template<typename T>
__forceinline __host__ __device__ size_t ArrayBase<T>::getDepthOffset() const{
        return depthOffset;
}

	
template<typename T>
__forceinline__ size_t ArrayBase<T>::memSize() const{
	return size()*sizeof(T);
}

template<typename T>
__forceinline__ size_t ArrayBase<T>::size() const{
	return height*width*depth;
}

template<typename T>
__forceinline__ size_t ArrayBase<T>::typeSize() const{
	return sizeof(T);
}

template<typename T>
__forceinline__ size_t ArrayBase<T>::realSize() const{
	return realHeight*realWidth*realDepth;
}

#ifdef PSKEL_CUDA
template<typename T>
__forceinline__ __device__ T & ArrayBase<T>::deviceGet(size_t h, size_t w, size_t d) const {
	return this->deviceArray[(h*width+w)*depth+d];
}
#endif

template<typename T>
inline T & ArrayBase<T>::hostGet(size_t h, size_t w, size_t d) const {
	return this->hostArray[ ((h+heightOffset)*realWidth + (w+widthOffset))*realDepth + (d+depthOffset) ];
}

template<typename T> template<typename Arrays>
void ArrayBase<T>::hostSlice(Arrays array, size_t widthOffset, size_t heightOffset, size_t depthOffset, size_t width, size_t height, size_t depth){
	//maintain previous allocated area
	#ifdef PSKEL_CUDA
	if(this->deviceArray!=NULL){
		if(this->size()!=(width*height*depth)){
			std::cout<<"Host slice free GPU memory"<<std::endl;
			this->deviceFree();
			this->deviceArray = NULL;
		}
	}
	#endif
	//Copy dimensions
	this->width = width;
	this->height = height;
	this->depth = depth;
	this->widthOffset = array.widthOffset+widthOffset;
	this->heightOffset = array.heightOffset+heightOffset;
	this->depthOffset = array.depthOffset+depthOffset;
	this->realWidth = array.realWidth;
	this->realHeight = array.realHeight;
	this->realDepth = array.realDepth;
	this->hostArray = array.hostArray;
	this->hostGPUArray = array.hostGPUArray;

	#if DEBUG
		printf("Array of address %p sliced with offset (%d,%d,%d) starting at address %p\n",
		 (void*)&(array.hostArray),this->widthOffset,this->heightOffset,this->depthOffset,(void*)&(this->hostGet(0,0,0)));
	#endif
}

template<typename T>
template<typename Array>
void ArrayBase<T>::updateHalo(Array array, size_t height_offset, size_t halo_size, bool gpu_flag){
	//std::cout << "Update Halo Method\n";
	T* devicePtr = (T*)(this->deviceArray) + size_t(height_offset*this->width*this->depth);
	
	if(gpu_flag){
		/* Copy CPU Halo to GPU */
		//T* hostPtr = (T*)(this->hostArray) + size_t(height_offset*this->width*this->depth);
		T* devicePtr = (T*)(this->deviceArray) + size_t(height_offset*this->width*this->depth);
		cudaHostRegister(array.hostArray, halo_size*this->width*this->depth*sizeof(T), cudaHostRegisterPortable);
		gpuErrchk ( cudaMemcpyAsync(devicePtr, array.hostArray, halo_size*this->width*this->depth*sizeof(T),cudaMemcpyHostToDevice) );
		cudaHostUnregister(array.hostArray);
	}
	else{
		/* Copy GPU Halo to CPU*/
		T* devicePtr = (T*)(array.deviceArray) + size_t((height_offset-halo_size)*this->width*this->depth);
		gpuErrchk ( cudaMemcpyAsync(this->hostArray, devicePtr, halo_size*this->width*this->depth*sizeof(T),cudaMemcpyDeviceToHost) );
	}
	
}


//TODO: Alterar para retornar um Array ao invÃ©s de receber por parametro
template<typename T> template<typename Arrays>
void ArrayBase<T>::hostClone(Arrays array){
	//Copy dimensions
	this->width = array.width;
	this->height = array.height;
	this->depth = array.depth;
	this->widthOffset = 0;
	this->heightOffset = 0;
	this->depthOffset = 0;
	this->realWidth = array.width;
	this->realHeight = array.height;
	this->realDepth = array.depth;
	//Alloc clone memory
	this->hostArray = NULL;
	this->hostGPUArray = NULL;
	this->hostAlloc();
	//Copy clone memory
	this->hostMemCopy(array);
}
	
template<typename T> template<typename Arrays>
void ArrayBase<T>::hostMemCopy(Arrays array){
	#ifdef TIMER
		double start = omp_get_wtime();
	#endif
	if(array.size()==array.realSize() && this->size()==this->realSize()){
		memcpy(this->hostArray, array.hostArray, size()*sizeof(T));
	} else if(array.depth == array.realDepth && array.width == array.realWidth && this->depth == this->realDepth && this->width == this->realWidth){
		T *hostPtr = (T*)(this->hostArray) + size_t(heightOffset*realWidth*realDepth);
		memcpy(hostPtr, array.hostArray, size()*sizeof(T));
	} else if(array.realDepth == 1 && array.realHeight == 1 && this->realDepth == 1 && this->realHeight == 1){
		T *hostPtr = (T*)(this->hostArray) + size_t(widthOffset);
		memcpy(hostPtr, array.hostArray, size()*sizeof(T));	
	}else{
		std::cout<<"CPU hostMemCopy executing parallel"<<std::endl;
		#pragma omp parallel for
		for(size_t i = 0; i<height; ++i){
		for(size_t j = 0; j<width; ++j){
		for(size_t k = 0; k<depth; ++k){
                        this->hostGet(i,j,k)=array.hostGet(i,j,k);
		}}}
	}
	#ifdef TIMER
		double end = omp_get_wtime();
		std::cout<<"CPU_hostMemCopy: "<<end-start<<std::endl;
		//printf("Host copy from address %p to address %p took %f seconds\n",&(array.hostArray),&(this->hostArray),end-start);
	#endif
}

template<typename T> template<typename Arrays>
void ArrayBase<T>::hostPinnedMemCopy(Arrays array){
	#ifdef TIMER
		double start = omp_get_wtime();
	#endif
	if(array.size()==array.realSize() && this->size()==this->realSize()){
		memcpy(this->hostArray, array.hostGPUArray, size()*sizeof(T));
	} else if(array.depth == array.realDepth && array.width == array.realWidth && this->depth == this->realDepth && this->width == this->realWidth){
		T *hostPtr = (T*)(this->hostArray) + size_t(heightOffset*realWidth*realDepth);
		memcpy(hostPtr, array.hostGPUArray, size()*sizeof(T));
	} else if(array.realDepth == 1 && array.realHeight == 1 && this->realDepth == 1 && this->realHeight == 1){
		T *hostPtr = (T*)(this->hostArray) + size_t(widthOffset);
		memcpy(hostPtr, array.hostGPUArray, size()*sizeof(T));	
	}else{
		std::cout<<"CPU hostPinnedMemCopy executing parallel. TODO!"<<std::endl;
		/*
		#pragma omp parallel for
		for(size_t i = 0; i<height; ++i){
		for(size_t j = 0; j<width; ++j){
		for(size_t k = 0; k<depth; ++k){
                        this->hostGet(i,j,k)=array.hostGet(i,j,k);
		}}}
		*/
	}
	#ifdef TIMER
		double end = omp_get_wtime();
		std::cout<<"CPU_hostMemCopy: "<<end-start<<std::endl;
		//printf("Host copy from address %p to address %p took %f seconds\n",&(array.hostArray),&(this->hostArray),end-start);
	#endif
}


#ifdef PSKEL_CUDA
template<typename T>
inline void ArrayBase<T>::copyToDevice(){
	#ifndef PSKEL_MANAGED
	if(size()==realSize()){
		//std::cout<<"Copy type 1"<<std::endl;
		gpuErrchk ( cudaMemcpyAsync(deviceArray, hostArray, size()*sizeof(T), cudaMemcpyHostToDevice) );
	}else if(depth==realDepth && width==realWidth){
		std::cout<<"Copy type 2"<<std::endl;
		T *hostPtr = (T*)(hostArray) + size_t(heightOffset*realWidth*realDepth);
		gpuErrchk ( cudaMemcpyAsync(deviceArray, hostPtr, size()*sizeof(T),cudaMemcpyHostToDevice) );
	}else if(realDepth==1 && realHeight==1){
		std::cout<<"Copy type 3"<<std::endl;
		T *hostPtr = (T*)(hostArray) + size_t(widthOffset);
		gpuErrchk ( cudaMemcpyAsync(deviceArray, hostPtr, size()*sizeof(T),cudaMemcpyHostToDevice) );
	}else{ 
		//if "virtual" array is non-continuously allocated,
		//create a copy in pinned memory before transfering.
		std::cout<<"CPU copyToDevice executing in parallel"<<std::endl;
		T *copyPtr;
		gpuErrchk( cudaMallocHost((void**)&copyPtr, size()*sizeof(T)) );
		#pragma omp parallel for num_threads(12)
		for(size_t h = 0; h<height; ++h){
		for(size_t w = 0; w<width; ++w){
		for(size_t d = 0; d<depth; ++d){
                        copyPtr[(h*width+w)*depth+d] = this->hostGet(h,w,d);
		}}}
		gpuErrchk ( cudaMemcpyAsync(deviceArray, copyPtr, size()*sizeof(T), cudaMemcpyHostToDevice) );
		cudaFreeHost(copyPtr);
	}
	#endif
}
#endif

template<typename T>
inline void ArrayBase<T>::copyToDevicePinned(){
	gpuErrchk ( cudaMemcpyAsync(deviceArray, hostGPUArray, size()*sizeof(T), cudaMemcpyHostToDevice) );
}

template<typename T>
inline void ArrayBase<T>::copyFromDevicePinned(){
        gpuErrchk ( cudaMemcpyAsync(hostGPUArray,deviceArray, size()*sizeof(T), cudaMemcpyDeviceToHost) );
}

#ifdef PSKEL_CUDA
template<typename T> template<typename Arrays>
void ArrayBase<T>::copyFromDevice(Arrays array){
	#ifndef PSKEL_MANAGED
	if(array.size()==realSize()){
		gpuErrchk ( cudaMemcpy(hostArray, array.deviceArray, array.size()*sizeof(T),cudaMemcpyDeviceToHost) );
	}else if(array.depth==realDepth && array.width==realWidth){
		T *hostPtr = (T*)(hostArray) + size_t(heightOffset*realWidth*realDepth);
		gpuErrchk ( cudaMemcpy(hostPtr, array.deviceArray, array.size()*sizeof(T), cudaMemcpyDeviceToHost) );
	}else if(realDepth==1 && realHeight==1){
		T *hostPtr = (T*)(hostArray) + size_t(widthOffset);
		gpuErrchk ( cudaMemcpy(hostPtr, array.deviceArray, array.size()*sizeof(T), cudaMemcpyDeviceToHost) );
	}else{
		//if "virtual" array is non-continuously allocated,
		//create a copy in pinned memory before transfering.
		std::cout<<"GPU copyFromDevice executing in parallel"<<std::endl;
		T *copyPtr;
		gpuErrchk( cudaMallocHost((void**)&copyPtr, size()*sizeof(T)) );
		gpuErrchk ( cudaMemcpy(copyPtr, array.deviceArray, size()*sizeof(T), cudaMemcpyDeviceToHost) );
		#pragma omp parallel for
		for(size_t h = 0; h<height; ++h){
		for(size_t w = 0; w<width; ++w){
		for(size_t d = 0; d<depth; ++d){
                	this->hostGet(h,w,d) = copyPtr[(h*width+w)*depth+d];
		}}}
		cudaFreeHost(copyPtr);
	}
	#endif
}
#endif

#ifdef PSKEL_CUDA
template<typename T> 
template<typename Arrays>
void ArrayBase<T>::copyFromDevicePinned(Arrays array){
	#ifndef PSKEL_MANAGED
	if(array.size()==realSize()){
		std::cout<<"Copy from Device Pinned Type 1"<<std::endl;
		gpuErrchk ( cudaMemcpy(hostGPUArray, array.deviceArray, array.size()*sizeof(T),cudaMemcpyDeviceToHost) );
	}else if(array.depth==realDepth && array.width==realWidth){
		std::cout<<"Copy from Device Pinned Type 2"<<std::endl;
		T *hostPtr = (T*)(hostGPUArray) + size_t(heightOffset*realWidth*realDepth);
		gpuErrchk ( cudaMemcpy(hostPtr, array.deviceArray, array.size()*sizeof(T), cudaMemcpyDeviceToHost) );
	}else if(realDepth==1 && realHeight==1){
		std::cout<<"Copy from Device Pinned Type 3"<<std::endl;
		T *hostPtr = (T*)(hostGPUArray) + size_t(widthOffset);
		gpuErrchk ( cudaMemcpy(hostPtr, array.deviceArray, array.size()*sizeof(T), cudaMemcpyDeviceToHost) );
	}else{
		//if "virtual" array is non-continuously allocated,
		//create a copy in pinned memory before transfering.
		std::cout<<"Copy from Device Pinned Type 4"<<std::endl;
		std::cout<<"TODO!"<<std::endl;
		/*
 		T *copyPtr;
		gpuErrchk( cudaMallocHost((void**)&copyPtr, size()*sizeof(T)) );
		gpuErrchk ( cudaMemcpy(copyPtr, array.deviceArray, size()*sizeof(T), cudaMemcpyDeviceToHost) );
		#pragma omp parallel for
		for(size_t h = 0; h<height; ++h){
		for(size_t w = 0; w<width; ++w){
		for(size_t d = 0; d<depth; ++d){
                	this->hostGet(h,w,d) = copyPtr[(h*width+w)*depth+d];
		}}}
		cudaFreeHost(copyPtr);
		*/
	}
	#endif
}
#endif


#ifdef PSKEL_CUDA
template<typename T>
void ArrayBase<T>::copyToHost(){
	if(size()==realSize()){
		gpuErrchk ( cudaMemcpy(hostArray, deviceArray, size()*sizeof(T),cudaMemcpyDeviceToHost) );
	}else if(depth==realDepth && width==realWidth){
		T *hostPtr = (T*)(hostArray) + size_t(heightOffset*realWidth*realDepth);
		gpuErrchk ( cudaMemcpy(hostPtr, deviceArray, size()*sizeof(T), cudaMemcpyDeviceToHost) );
	}else if(realDepth==1 && realHeight==1){
		T *hostPtr = (T*)(hostArray) + size_t(widthOffset);
		gpuErrchk ( cudaMemcpy(hostPtr, deviceArray, size()*sizeof(T), cudaMemcpyDeviceToHost) );
	}else{
		//if "virtual" array is non-continuously allocated,
		//create a copy in pinned memory before transfering.
		std::cout<<"GPU copyToHost executing in parallel"<<std::endl;
		T *copyPtr;
		gpuErrchk( cudaMallocHost((void**)&copyPtr, size()*sizeof(T)) );
		gpuErrchk ( cudaMemcpy(copyPtr, deviceArray, size()*sizeof(T), cudaMemcpyDeviceToHost) );
		#pragma omp parallel for
		for(size_t h = 0; h<height; ++h){
		for(size_t w = 0; w<width; ++w){
		for(size_t d = 0; d<depth; ++d){
                	this->hostGet(h,w,d) = copyPtr[(h*width+w)*depth+d];
		}}}
		cudaFreeHost(copyPtr);
	}
}
#endif

template<typename T>
__forceinline__ __host__ __device__ ArrayBase<T>::operator bool() const {
	#ifdef __CUDA_ARCH__
	return(this->deviceArray!=NULL);
	#else
	return(this->hostArray!=NULL);
	#endif
}

//*******************************************************************************************
// Array 3D
//*******************************************************************************************

template<typename T>
Array3D<T>::Array3D() : ArrayBase<T>(0,0,0) {}
	
/*
//TODO O kernel cuda nÃ£o aceita structs com destrutores. Corrigir nas prÃ³ximas versÃµes
~Array3D(){
free(hostArray);
cudaFree(deviceArray);
}*/

template<typename T>
Array3D<T>::Array3D(size_t width, size_t height, size_t depth) : ArrayBase<T>(width,height,depth){}

template<typename T>
__forceinline__ __host__ __device__ T & Array3D<T>::operator()(size_t h,size_t w,size_t d) const {
	#ifdef __CUDA_ARCH__
		/* TODO deviceGet is not inlining!
		 return this->deviceGet(h,w,d);
		*/  
		return this->deviceArray[(h*width+w)*depth+d];
	#else
		return this->hostGet(h,w,d);
	#endif
}

//*******************************************************************************************
// Array 2D
//*******************************************************************************************

template<typename T>
Array2D<T>::Array2D() : ArrayBase<T>(0,0,0) {}

template<typename T>
Array2D<T>::Array2D(size_t width, size_t height) : ArrayBase<T>(width,height,1){}

template<typename T>
__forceinline __host__ __device__ T & Array2D<T>::operator()(size_t h, size_t w) const {
	#ifdef PSKEL_MANAGED
		//return this->hostGet(h,w,0);
		return this->hostArray[((h+this->heightOffset)*this->realWidth + (w+this->widthOffset))];
	#else
	#ifdef __CUDA_ARCH__
		/* TODO deviceGet is not inlining!
		return this->deviceGet(h,w,0); 
		*/
		return this->deviceArray[h*this->width+w];
	#else
		//return this->hostGet(h,w,0);
		//return ((h+this->heightOffset)<this->realHeight && (w+this->widthOffset)<this->realWidth)
		//	? this->hostArray[ ((h+this->heightOffset)*this->realWidth + (w+this->widthOffset))] : this->haloValuePtr[0];
		return this->hostArray[((h+this->heightOffset)*this->realWidth + (w+this->widthOffset))];
	
	#endif
	#endif
}

//*******************************************************************************************
// Array 1D
//*******************************************************************************************

template<typename T>
Array<T>::Array() : ArrayBase<T>(0,0,0){}

template<typename T>
Array<T>::Array(size_t size) : ArrayBase<T>(size,1,1){}

template<typename T>
__forceinline__ __host__ __device__ T & Array<T>::operator()(size_t w) const {
	#ifdef __CUDA_ARCH__
		/* TODO deviceGet is not inlining!
 		* return this->deviceGet(0,w,0);
 		*/
		return this->deviceArray[w];
	#else
		return this->hostGet(0,w,0);
	#endif
}

}//end namespace
#endif
