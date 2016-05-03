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
#include <cstring>
#include <iostream>

#ifndef MPPA_MASTER
#include <omp.h>
#endif

// Maximum size of the cluster
#define KB 1024
#define MB 1024 * KB
// maximum size of a message
#define MAX_CLUSTER_SIZE 1 * MB + MB / 2

//Change how to read and write in the array for later use for a bunch of clusters
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
	this->deviceArray = 0;
	#endif
	#ifdef PSKEL_MPPA
	this->mppaArray = 0;
	this->write_portal = 0;
	this->read_portal = 0;
	this->aux_write_portal = 0;
	this->aux_read_portal = 0;
	this->aux = 0;//(int *) malloc(sizeof(size_t*) * 13);
	if(size()>0) this->mppaAlloc();
	#else
	if(size()>0) this->hostAlloc();
	#endif
}

#ifdef PSKEL_CUDA
template<typename T>
void ArrayBase<T>::deviceAlloc(){
	if(this->deviceArray==NULL){
		gpuErrchk( cudaMalloc((void **) &deviceArray, size()*sizeof(T)) );
		cudaMemset(this->deviceArray, 0, size()*sizeof(T));
	}
}
#endif

#ifdef PSKEL_CUDA
template<typename T>
void ArrayBase<T>::deviceFree(){
	//if(this->deviceArray!=NULL){
		cudaFree(this->deviceArray);
		this->deviceArray = NULL;
	//}
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

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::mppaAlloc(){
	if(this->mppaArray==NULL){
		this->mppaArray = (T*) calloc(this->size(), sizeof(T));
		assert(mppaArray!= NULL);
		#ifdef DEBUG
			#ifdef MPPA_MASTER
				std::cout<<"MASTER: Allocating "<<this->size()*sizeof(T)<<" bytes"<<std::endl;
			#else
				std::cout<<"SLAVE: Allocating "<<this->size()*sizeof(T)<<" bytes"<<std::endl;
			#endif
		#endif
	}
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::mppaAlloc(size_t width, size_t height, size_t depth){
	this->width = width;
	this->height = height;
	this->depth = depth;
	this->realWidth = width;
	this->realHeight = height;
	this->realDepth = depth;
	this->widthOffset = 0;
	this->heightOffset = 0;
	this->depthOffset = 0;

	this->mppaArray = NULL;
	this->mppaAlloc();
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::mppaFree(){
	//if(this->mppaArray!=NULL){
	free(this->mppaArray);
	//cudaFreeHost(this->hostArray);
	this->mppaArray = NULL;
	//}
	#ifdef DEBUG
		#ifdef MPPA_MASTER
			std::cout<<"MASTER: Deallocating "<<this->size()*sizeof(T)<<" bytes"<<std::endl;
		#else
			std::cout<<"SLAVE: Deallocating "<<this->size()*sizeof(T)<<" bytes"<<std::endl;
		#endif
	#endif
}	
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::auxFree(){
	//if(this->mppaArray!=NULL){
	free(this->aux);
	//cudaFreeHost(this->hostArray);
	this->aux = NULL;
	//}
}	
#endif

template<typename T>
void ArrayBase<T>::hostAlloc(){
	if(this->hostArray==NULL){
		this->hostArray = (T*) calloc(size(), sizeof(T));
		//gpuErrchk( cudaMallocHost((void**)&hostArray, size()*sizeof(T)) );
		//memset(this->hostArray, 0, size()*sizeof(T));
	}
}
	
template<typename T>
void ArrayBase<T>::hostFree(){
	//if(this->hostArray!=NULL){
	free(this->hostArray);
	//cudaFreeHost(this->hostArray);
	this->hostArray = NULL;
	//}
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

#ifdef PSKEL_MPPA
template<typename T>
T& ArrayBase<T>::mppaGet(size_t h, size_t w, size_t d) const {
	return this->mppaArray[ ((h+heightOffset)*realWidth + (w+widthOffset))*realDepth + (d+depthOffset) ];
}
#endif

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
	
	this->width = width;
	this->height = height;
	this->depth = depth;
	this->widthOffset = array.widthOffset+widthOffset;
	this->heightOffset = array.heightOffset+heightOffset;
	this->depthOffset = array.depthOffset+depthOffset;
	this->realWidth = array.realWidth;
	this->realHeight = array.realHeight;
	this->realDepth = array.realDepth;
	#ifdef MPPA_MASTER
	
	#endif
	#ifndef PSKEL_MPPA
	this->hostArray = array.hostArray;
	#else
	this->mppaArray = array.mppaArray;
	#endif
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
#ifndef MPPA_MASTER
template<typename T> template<typename Arrays>
void ArrayBase<T>::mppaClone(Arrays array){
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
	this->mppaArray = NULL;
	this->mppaAlloc();
	//Copy clone memory
	//#ifdef MPPA_SLAVE
	this->mppaMemCopy(array);
	//#endif
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::setTrigger(int trigger){
    mppa_aiocb_set_trigger(&read_portal->aiocb, trigger);
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
int* ArrayBase<T>::getAux(){
	return this->aux;
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::auxAlloc(){
	this->aux = (int *) malloc(sizeof(size_t*) * 13);	
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::setAux(int heightOffset, int it, int subIterations, size_t coreWidthOffset, size_t coreHeightOffset, size_t coreDepthOffset, size_t coreWidth, size_t coreHeight, size_t coreDepth, int outterIterations, size_t height, size_t width, size_t depth){
	this->aux[0] = size_t(heightOffset*realWidth*realDepth);

	this->aux[1] = it;
	this->aux[2] = subIterations;
	this->aux[3] = coreWidthOffset;
	this->aux[4] = coreHeightOffset;
	this->aux[5] = coreDepthOffset;
	this->aux[6] = coreWidth;
	this->aux[7] = coreHeight;
	this->aux[8] = coreDepth;
	this->aux[9] = outterIterations;
	this->aux[10] = height;
	this->aux[11] = width;
	this->aux[12] = depth;


}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::copyToAux(){
	mppa_async_write_portal(this->aux_write_portal, this->aux, (sizeof(int) * 13), 0);
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::copyFromAux(){
	mppa_async_read_wait_portal(this->aux_read_portal);
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::copyTo(size_t heightOffset, int offset){
	T *mppaSlicePtr = (T*)(this->mppaArray) + size_t(heightOffset*realWidth*realDepth);
	mppa_async_write_portal(this->write_portal, mppaSlicePtr, this->memSize(), offset);
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::copyFrom(){
	mppa_async_read_wait_portal(this->read_portal);

}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::portalReadAlloc(int trigger, int nb_cluster){
	#ifdef MPPA_MASTER
		char pathMaster[256];
	    sprintf(pathMaster, "/mppa/portal/%d:3", 128);
		this->read_portal = mppa_create_read_portal(pathMaster, this->mppaArray, this->memSize(), trigger, NULL);
	#endif
	#ifdef MPPA_SLAVE
	char pathSlave[25];
	char path[25];
	sprintf(pathSlave, "/mppa/portal/%d:%d", nb_cluster, 4 + nb_cluster);
    
    this->read_portal = mppa_create_read_portal(pathSlave, this->mppaArray, this->memSize(), trigger, NULL);
	#endif
}
#endif


#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::portalAuxWriteAlloc(int nb_cluster){
	char path[256];
	#ifdef MPPA_MASTER
		sprintf(path, "/mppa/portal/%d:%d", nb_cluster, (21 + nb_cluster));
    	this->aux_write_portal = mppa_create_write_portal(path, NULL, 0, nb_cluster);
	#endif
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::portalAuxReadAlloc(int trigger, int nb_cluster){
	char path[25];
	#ifdef MPPA_SLAVE
		sprintf(path, "/mppa/portal/%d:%d", nb_cluster, (21 + nb_cluster));
    	this->aux_read_portal = mppa_create_read_portal(path, this->aux, (sizeof(int) * 13), trigger, NULL);
	#endif
}
#endif


#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::portalWriteAlloc(int nb_cluster){
	char path[256];
	#ifdef MPPA_MASTER
		sprintf(path, "/mppa/portal/%d:%d", nb_cluster, 4 + nb_cluster);
    	this->write_portal = mppa_create_write_portal(path, NULL, 0, nb_cluster);
	#endif
    #ifdef MPPA_SLAVE
		sprintf(path, "/mppa/portal/%d:3", 128);
    	this->write_portal = mppa_create_write_portal(path, NULL, 0, 128);
    #endif
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::waitRead(){
	mppa_async_read_wait_portal(this->read_portal);
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::waitWrite(){
	mppa_async_write_wait_portal(this->write_portal);
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::waitAuxWrite(){
	mppa_async_write_wait_portal(this->aux_write_portal);
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::closeReadPortal(){
	mppa_close_portal(this->read_portal);
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::closeAuxReadPortal(){
	mppa_close_portal(this->aux_read_portal);
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::closeAuxWritePortal(){
	mppa_close_portal(this->aux_write_portal);
}
#endif

#ifdef PSKEL_MPPA
template<typename T>
void ArrayBase<T>::closeWritePortal(){
	mppa_close_portal(this->write_portal);
}
#endif

#ifndef MPPA_MASTER
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
#endif

template<typename T> template<typename Arrays>
void ArrayBase<T>::mppaMasterCopy(Arrays array){
	if(array.size()==array.realSize() && this->size()==this->realSize()){
		memcpy(this->mppaArray, array.mppaArray, size()*sizeof(T));
	}
}

#ifndef MPPA_MASTER
template<typename T> template<typename Arrays>
void ArrayBase<T>::mppaMemCopy(Arrays array){
	if(array.size()==array.realSize() && this->size()==this->realSize()){
		memcpy(this->mppaArray, array.mppaArray, size()*sizeof(T));
	}else{
		#pragma omp parallel for
		for(size_t i = 0; i<height; ++i){
		for(size_t j = 0; j<width; ++j){
		for(size_t k = 0; k<depth; ++k){
                        	this->mppaGet(i,j,k)=array.mppaGet(i,j,k);
		}}}
	}
}
#endif

#ifdef PSKEL_CUDA
template<typename T>
void ArrayBase<T>::copyToDevice(){
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
}
#endif

#ifdef PSKEL_CUDA
template<typename T> template<typename Arrays>
void ArrayBase<T>::copyFromDevice(Arrays array){
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
	#endif
	#ifdef PSKEL_MPPA
		return this->mppaGet(h,w,d);
	#else
		return this->hostGet(h,w,d);
	#endif
}

//*******************************************************************************************
// Array 2D
//*******************************************************************************************

template<typename T>
Array2D<T>::Array2D() : ArrayBase<T>(0,0,0) {}

//template<typename T>
// Array2D<T>::~Array2D(){
// 	this->mppaFree();
// }

template<typename T>
Array2D<T>::Array2D(size_t width, size_t height) : ArrayBase<T>(width,height,1){}

template<typename T>
T & Array2D<T>::operator()(size_t h, size_t w) const {
	#ifdef __CUDA_ARCH__
		return this->deviceGet(h,w,0);
	#endif
	#ifdef PSKEL_MPPA
		return this->mppaGet(h,w,0);
	#else
		return this->hostGet(h,w,0);
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
	#endif
	#ifdef PSKEL_MPPA
		return this->mppaGet(0,w,0);
	#else
		return this->hostGet(0,w,0);
	#endif
}

}//end namespace
#endif
