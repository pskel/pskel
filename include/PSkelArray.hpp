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

#include <cstring>
#include <omp.h>

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
	#ifdef PSKEL_CUDA
    	this->deviceArray = NULL;
	#endif
	
	if(size()>0) this->hostAlloc();
	
	
}

#ifdef PSKEL_CUDA
template<typename T>
void ArrayBase<T>::deviceAlloc(){
	#ifndef PSKEL_MANAGED
	if(this->deviceArray==NULL){
		gpuErrchk( cudaMalloc((void **) &deviceArray, size()*sizeof(T)) );
		//cudaMemset(this->deviceArray, 0, size()*sizeof(T));
	}
	#endif
}
#endif

#ifdef PSKEL_CUDA
template<typename T>
void ArrayBase<T>::deviceFree(){
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
	#endif

	this->hostAlloc();
}

//TODO When using CUDA, a page-locked memory is allocated. Verify the diference
//in performance between cudaMallocHost and cudaHostAlloc and its flags.
//It seems that Cloudsim CPU Performance is worst with cudaMallocHost.
template<typename T>
void ArrayBase<T>::hostAlloc(){
	if(this->hostArray==NULL){
	#ifdef PSKEL_MANAGED
		cudaMallocManaged((void**)&hostArray,size()*sizeof(T));
	#else
	#ifdef PSKEL_CUDA
            gpuErrchk( cudaMallocHost((void**)&hostArray, size()*sizeof(T)) );
            //cudaMemset(this->hostArray, 0, size()*sizeof(T));
        #else
            this->hostArray = (T*) calloc(size(), sizeof(T));
        #endif
	#endif
		//gpuErrchk( cudaMallocHost((void**)&hostArray, size()*sizeof(T)) );
		//memset(this->hostArray, 0, size()*sizeof(T));
	}
}
	
template<typename T>
void ArrayBase<T>::hostFree(){
	if(this->hostArray!=NULL){
	#ifdef PSKEL_MANAGED
		cudaFree(this->hostArray);
	#else
	#ifdef PSKEL_CUDA	
		gpuErrchk( cudaFreeHost(this->hostArray) );
	#else
		free(this->hostArray);
	#endif	
	#endif
	this->hostArray = NULL;
	}
}

template<typename T>
size_t ArrayBase<T>::getWidth() const{
	return width;
}
	
template<typename T>
size_t ArrayBase<T>::getHeight() const{
	return height;
}

template<typename T>
size_t ArrayBase<T>::getDepth() const{
	return depth;
}
	
template<typename T>
size_t ArrayBase<T>::memSize() const{
	return size()*sizeof(T);
}

template<typename T>
size_t ArrayBase<T>::size() const{
	return height*width*depth;
}

template<typename T>
size_t ArrayBase<T>::realSize() const{
	return realHeight*realWidth*realDepth;
}

#ifdef PSKEL_CUDA
template<typename T>
__device__ __forceinline__ T & ArrayBase<T>::deviceGet(size_t h, size_t w, size_t d) const {
	return this->deviceArray[(h*width+w)*depth+d];
}
#endif

template<typename T>
T & ArrayBase<T>::hostGet(size_t h, size_t w, size_t d) const {
	return this->hostArray[ ((h+heightOffset)*realWidth + (w+widthOffset))*realDepth + (d+depthOffset) ];
}

template<typename T> template<typename Arrays>
void ArrayBase<T>::hostSlice(Arrays array, size_t widthOffset, size_t heightOffset, size_t depthOffset, size_t width, size_t height, size_t depth){
	//maintain previous allocated area
	#ifdef PSKEL_CUDA
	if(this->deviceArray!=NULL){
		if(this->size()!=(width*height*depth)){
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
	this->hostAlloc();
	//Copy clone memory
	this->hostMemCopy(array);
}
	
template<typename T> template<typename Arrays>
void ArrayBase<T>::hostMemCopy(Arrays array){
	if(array.size()==array.realSize() && this->size()==this->realSize()){
		memcpy(this->hostArray, array.hostArray, size()*sizeof(T));
	}else{
		#pragma omp parallel for
		for(size_t i = 0; i<height; ++i){
		for(size_t j = 0; j<width; ++j){
		for(size_t k = 0; k<depth; ++k){
                        this->hostGet(i,j,k)=array.hostGet(i,j,k);
		}}}
	}
}

#ifdef PSKEL_CUDA
template<typename T>
void ArrayBase<T>::copyToDevice(){
	#ifndef PSKEL_MANAGED
	if(size()==realSize()){
		gpuErrchk ( cudaMemcpy(deviceArray, hostArray, size()*sizeof(T), cudaMemcpyHostToDevice) );
	}else if(depth==realDepth && width==realWidth){
		T *hostPtr = (T*)(hostArray) + size_t(heightOffset*realWidth*realDepth);
		gpuErrchk ( cudaMemcpy(deviceArray, hostPtr, size()*sizeof(T),cudaMemcpyHostToDevice) );
	}else if(realDepth==1 && realHeight==1){
		T *hostPtr = (T*)(hostArray) + size_t(widthOffset);
		gpuErrchk ( cudaMemcpy(deviceArray, hostPtr, size()*sizeof(T),cudaMemcpyHostToDevice) );
	}else{ 
		//if "virtual" array is non-continuously allocated,
		//create a copy in pinned memory before transfering.
		T *copyPtr;
		gpuErrchk( cudaMallocHost((void**)&copyPtr, size()*sizeof(T)) );
		#pragma omp parallel for
		for(size_t h = 0; h<height; ++h){
		for(size_t w = 0; w<width; ++w){
		for(size_t d = 0; d<depth; ++d){
                        copyPtr[(h*width+w)*depth+d] = this->hostGet(h,w,d);
		}}}
		gpuErrchk ( cudaMemcpy(deviceArray, copyPtr, size()*sizeof(T), cudaMemcpyHostToDevice) );
		cudaFreeHost(copyPtr);
	}
	#endif
}
#endif

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
ArrayBase<T>::operator bool() const {
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
T & Array3D<T>::operator()(size_t h,size_t w,size_t d) const {
	#ifdef __CUDA_ARCH__
		return this->deviceGet(h,w,d);
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
T & Array2D<T>::operator()(size_t h, size_t w) const {
	#ifdef PSKEL_MANAGED
		return this->hostGet(h,w,0);
	#else
	#ifdef __CUDA_ARCH__
		return this->deviceGet(h,w,0);
	#else
		return this->hostGet(h,w,0);
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
T & Array<T>::operator()(size_t w) const {
	#ifdef __CUDA_ARCH__
		return this->deviceGet(0,w,0);
	#else
		return this->hostGet(0,w,0);
	#endif
}

}//end namespace
#endif
