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

#ifndef PSKEL_MASK_HPP
#define PSKEL_MASK_HPP

#include <cstring>

#include "PSkelDefs.h"

namespace PSkel{
//*******************************************************************************************
// MASKBASE
//*******************************************************************************************
template<typename T>
MaskBase<T>::MaskBase(size_t size, size_t dimension, T haloVal, size_t range){
	this->size = size;
	this->dimension = dimension;
	this->range = range;
	this->haloValue = haloVal;
	this->hostMask = NULL;
	this->hostWeight = NULL;
	#ifdef PSKEL_CUDA
	this->deviceMask = NULL;
	this->deviceWeight = NULL;
	#endif
	if(size>0) this->hostAlloc();
}
	
template<typename T>
size_t MaskBase<T>::memSize() const{
	return (dimension*size*sizeof(int)+size*sizeof(T));
}

#ifdef PSKEL_CUDA
template<typename T>
void MaskBase<T>::deviceAlloc(){
	gpuErrchk( cudaMalloc((void **) &deviceMask, dimension * size * sizeof (int)) );
	gpuErrchk( cudaMalloc((void **) &deviceWeight, size * sizeof (T)) );
}
#endif

template<typename T>
void MaskBase<T>::hostAlloc(){
	if(this->hostMask==NULL && this->hostWeight==NULL){
		hostMask = (int*) malloc (dimension * size * sizeof (int));
		hostWeight = (T*) malloc (size * sizeof (T));
		//memset(hostWeight,1,size);
	}
}

template<typename T>
void MaskBase<T>::hostFree(){
	free(hostMask);
	free(hostWeight);
	hostMask = NULL;
	hostWeight = NULL;
}
	
#ifdef PSKEL_CUDA
template<typename T>
void MaskBase<T>::copyToDevice(){
	gpuErrchk ( cudaMemcpy(deviceMask, hostMask, dimension * size * sizeof(int),cudaMemcpyHostToDevice) );
	gpuErrchk ( cudaMemcpy(deviceWeight, hostWeight, size * sizeof(T),cudaMemcpyHostToDevice) );
}
#endif

#ifdef PSKEL_CUDA
template<typename T>
void MaskBase<T>::deviceFree(){
	//if(deviceMask!=NULL && deviceWeight!=NULL){
		cudaFree(deviceMask);
		cudaFree(deviceWeight);
		this->deviceMask = NULL;
		this->deviceWeight = NULL;
	//}
}
#endif

/*
template<typename T>
__host__ __device__ size_t MaskBase<T>::size() const{
	return this->size;
}
*/
template<typename T>
__forceinline__ __host__ __device__ T MaskBase<T>::getWeight(size_t n) {
	#ifdef __CUDA_ARCH__
		return deviceWeight[n];
	#else
		return hostWeight[n];
	#endif
}

// specializations for types we use
template<>
__device__ float* MaskBase<float>::GetSharedPointer(){
	extern __shared__ float sh_float[];
	// printf( "sh_float=%p\n", sh_float );
	return sh_float;
}

template<>
__device__ int* MaskBase<int>::GetSharedPointer(){
	extern __shared__ int sh_int[];
	// printf( "sh_float=%p\n", sh_float );
	return sh_int;
}


template<>
__device__ bool* MaskBase<bool>::GetSharedPointer(){
	extern __shared__ bool sh_bool[];
	// printf( "sh_float=%p\n", sh_float );
	return sh_bool;
}

//*******************************************************************************************
// MASK3D
//*******************************************************************************************

template<typename T>
Mask3D<T>::Mask3D(size_t size, T haloVal, size_t range) : MaskBase<T>(size,3, haloVal, range){}

template<typename T>
void Mask3D<T>::set(size_t n, int h, int w, int d, T weight){
	#ifdef __CUDA_ARCH__
		this->deviceMask[this->dimension*n] = h;
		this->deviceMask[this->dimension*n+1] = w;
		this->deviceMask[this->dimension*n+2] = d;
		this->deviceWeight[n] = weight;
	#else
		this->hostMask[this->dimension*n] = h;
		this->hostMask[this->dimension*n+1] = w;
		this->hostMask[this->dimension*n+2] = d;
		this->hostWeight[n] = weight;
	#endif
}
/*
template<typename T>
void Mask3D<T>::set(size_t n,int h,int w,int d){
	#ifdef __CUDA_ARCH__
		this->deviceMask[this->dimension*n] = h;
		this->deviceMask[this->dimension*n+1] = w;
		this->deviceMask[this->dimension*n+2] = d;
		this->deviceWeight[n] = 1;
	#else	
		this->hostMask[this->dimension*n] = h;
		this->hostMask[this->dimension*n+1] = w;
		this->hostMask[this->dimension*n+2] = d;
		this->hostWeight[n] = 1;
	#endif
}
*/
	
template<typename T> template<typename V>
T Mask3D<T>::get(size_t n, Array3D<V> array, size_t h, size_t w, size_t d){
	#ifdef __CUDA_ARCH__
		h += this->deviceMask[this->dimension*n];
		w += this->deviceMask[this->dimension*n+1];
		d += this->deviceMask[this->dimension*n+2];
		return (w<array.getWidth() && h<array.getHeight() && d<array.getDepth())?array(h,w,d):this->haloValue;
	#else
		h += this->hostMask[this->dimension*n];
		w += this->hostMask[this->dimension*n+1];
		d += this->hostMask[this->dimension*n+2];
		return (w<array.getWidth() && h<array.getHeight() && d<array.getDepth())?array(h,w,d):this->haloValue;
	#endif
}

template<typename T>
size_t Mask3D<T>::getRange(){
	if(this->range>0) return this->range;
	int *ptr;
	#ifdef __CUDA_ARCH__
		ptr = this->deviceMask;
	#else
		ptr = this->hostMask;
	#endif
	
	size_t max_h = abs(ptr[0]);
	size_t max_w = abs(ptr[1]);
	size_t max_d = abs(ptr[2]);
	//size_t range = std::max(std::max(max_h, max_w), max_d);
	this->range = ((max_h>max_w)?max_h:max_w);
	this->range = ((this->range>max_d)?this->range:max_d);
	for(size_t i=1; i < this->size; i++){
		max_h = abs(ptr[this->dimension*i]);
		max_w = abs(ptr[this->dimension*i+1]);
		max_d = abs(ptr[this->dimension*i+2]);
		//size_t value = std::max(std::max(max_h, max_w), max_d);
		size_t value = ((max_h>max_w)?max_h:max_w);
		value = ((value>max_d)?value:max_d);
		if(value > this->range){
			this->range = value;
		}
	}
	return this->range;
}
/*
template<typename T> template<typename Arrays>
int Mask3D<T>::setMaskRadius(Arrays array){
	int max_h = abs(this->hostMask[0]);
	int max_w = abs(this->hostMask[1]);
	int max_d = abs(this->hostMask[2]);
	//this->maskRadius = (max_h*array.getWidth() + max_w)*array.getDepth() + max_d;
	this->maskRadius = std::max(std::max(max_h, max_w), max_d);
	for(int i=1; i < this->size; i++){
		max_h = abs(this->hostMask[this->dimension*i]);
		max_w = abs(this->hostMask[this->dimension*i+1]);
		max_d = abs(this->hostMask[this->dimension*i+2]);
		int value = std::max(std::max(max_h, max_w), max_d);
		//int value = (max_h*array.getWidth() + max_w)*array.getDepth() + max_d;
		if(value > this->maskRadius){
			this->maskRadius = value;
		}
	}
	return this->maskRadius;
}
*/
//*******************************************************************************************
// MASK2D
//*******************************************************************************************

template<typename T>
Mask2D<T>::Mask2D(size_t size, T haloVal, size_t range) : MaskBase<T>(size,2,haloVal, range) {}

template<typename T>
Mask2D<T>::Mask2D(size_t size, int array[][2]): MaskBase<T>(size, 2, 0,0){
	for(size_t i=0;i<size;i++){
		this->set(i,array[i][0],array[i][1],(T)1);
	}
}
	
	
template<typename T>
void Mask2D<T>::set(size_t n, int h, int w, T weight){
	#ifdef __CUDA_ARCH__
		this->deviceMask[this->dimension*n] = h;
		this->deviceMask[this->dimension*n+1] = w;
		this->deviceWeight[n] = weight;
	#else
		this->hostMask[this->dimension*n] = h;
		this->hostMask[this->dimension*n+1] = w;
		this->hostWeight[n] = weight;
	#endif
}
/*
template<typename T>
void Mask2D<T>::set(size_t n, int h,int w){
	#ifdef __CUDA_ARCH__
		this->deviceMask[this->dimension*n] = h;
		this->deviceMask[this->dimension*n+1] = w;
		this->deviceWeight[n] = 1;
	#else	
		this->hostMask[this->dimension*n] = h;
		this->hostMask[this->dimension*n+1] = w;
		this->hostWeight[n] = 1;
	#endif
}
*/	
template<typename T> template<typename V>
__forceinline__ __host__ __device__ T Mask2D<T>::get(size_t n, Array2D<V> array, size_t h, size_t w){
	#ifdef __CUDA_ARCH__
		h += this->deviceMask[this->dimension*n];
		w += this->deviceMask[this->dimension*n+1];
		return (w<array.getWidth() && h<array.getHeight())?array(h,w):this->haloValue;
	#else
		h += this->hostMask[this->dimension*n];
		w += this->hostMask[this->dimension*n+1];
		return (w<array.getWidth() && h<array.getHeight())?array(h,w):this->haloValue;
	#endif
}
	
template<typename T>
size_t Mask2D<T>::getRange(){
	if(this->range>0) return this->range;
	int *ptr;
	#ifdef __CUDA_ARCH__
		ptr = this->deviceMask;
	#else
		ptr = this->hostMask;
	#endif
	//size_t range;
	//if(this->maskRadius == 0){
		size_t max_h = abs(ptr[0]);
		size_t max_w = abs(ptr[1]);
		//range = std::max(max_h, max_w);
		this->range = ((max_h>max_w)?max_h:max_w);
	
		for(size_t i=1; i < this->size; i++){
			max_h = abs(ptr[this->dimension*i]);
			max_w = abs(ptr[this->dimension*i+1]);
			//size_t value = std::max(max_h, max_w);
			size_t value = ((max_h>max_w)?max_h:max_w);
			if(value > this->range){
				this->range = value;
			}
		}
	//}
	return this->range;
}
/*
template<typename T> template<typename Arrays>
int Mask2D<T>::setMaskRadius(Arrays array){
	if(this->maskRadius == 0){
		int max_h = abs(this->hostMask[0]);
		int max_w = abs(this->hostMask[1]);
		this->maskRadius = std::max(max_h, max_w);
		//this->maskRadius = max_h*array.getWidth() + max_w;
	
		for(int i=1; i < this->size; i++){
			max_h = abs(this->hostMask[this->dimension*i]);
			max_w = abs(this->hostMask[this->dimension*i+1]);
			int value = std::max(max_h, max_w);
			//int value = max_h*array.getWidth() + max_w;
			if(value > this->maskRadius){
				this->maskRadius = value;
			}
		}
	}
	return this->maskRadius;
}
*/
//*******************************************************************************************
// MASK
//*******************************************************************************************

template<typename T>
Mask<T>::Mask(size_t size, T haloVal, size_t range) : MaskBase<T>(size,1,haloVal,range) {}
/*
template<typename T>
void Mask<T>::set(size_t n,int x, )){
	#ifdef __CUDA_ARCH__
		this->deviceMask[n] = x;
		this->deviceWeight[n] = 1;
	#else	
		this->hostMask[n] = x;
		this->hostWeight[n] = 1;
	#endif
}
*/	
template<typename T>
void Mask<T>::set(size_t n, int i, T weight){
	#ifdef __CUDA_ARCH__
		this->deviceMask[n] = i;
		this->deviceWeight[n] = weight;
	#else
		this->hostMask[n] = i;
		this->hostWeight[n] = weight;
	#endif
}
	
template<typename T> template<typename V>
T Mask<T>::get(size_t n, Array<V> array, size_t i){
	#ifdef __CUDA_ARCH__
		i += this->deviceMask[n];
		return (i<array.getWidth())?array(i):this->haloValue;
	#else
		i += this->hostMask[n];
		return (i<array.getWidth())?array(i):this->haloValue;
	#endif
}
	
template<typename T>
size_t Mask<T>::getRange(){
	if(this->range>0) return this->range;
	int *ptr;
	#ifdef __CUDA_ARCH__
		ptr = this->deviceMask;
	#else
		ptr = this->hostMask;
	#endif
	this->range = abs(ptr[0]);
	for(size_t i=1; i < this->size; i++){
		size_t value = abs(ptr[i]);
		if(value > this->range){
			this->range = value;
		}
	}
	return this->range;
}
/*
template<typename T> template<typename Arrays>
int Mask<T>::setMaskRadius(Arrays array){
	this->maskRadius = abs(this->hostMask[0]);
	for(int i=1; i < this->size; i++){
		int value = abs(this->hostMask[i]);
		if(value > this->maskRadius){
			this->maskRadius = value;
		}
	}
	return this->maskRadius;
}
*/
}//end namespace

#endif
