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

#ifndef PSKEL_ARGS_HPP
#define PSKEL_ARGS_HPP

namespace PSkel{

//*******************************************************************************************
// ARGS
//*******************************************************************************************

template<typename T>
Args<T>::Args(){};
	
	/*
	__stencil__ ~Args(){
		gpuErrchk( cudaFreeHost(hostArray));
	}
	*/
	
template<typename T>
Args<T>::Args(int _width){
	width = _width;
	gpuErrchk( cudaDeviceReset() );
	gpuErrchk( cudaSetDeviceFlags(cudaDeviceMapHost) );
	gpuErrchk( cudaHostAlloc((void **) &hostArray, width*sizeof(T), cudaHostAllocWriteCombined | cudaHostAllocMapped) );
	gpuErrchk( cudaHostGetDevicePointer(&deviceArray, hostArray, 0) );
}
	
template<typename T>
int Args<T>::getWidth() const{
	return width;
}
	
template<typename T>
T & Args<T>::operator()(int x) const {
	#ifdef __CUDA_ARCH__
		return deviceArray[x];
	#else
		return hostArray[x];
	#endif
}	

//*******************************************************************************************
// ARGS2D
//*******************************************************************************************

template<typename T>
Args2D<T>::Args2D(){};
	
	/*
	__stencil__ ~Args(){
		gpuErrchk( cudaFreeHost(hostArray));
	}
	*/
	
template<typename T>
Args2D<T>::Args2D(int _width,int _height){
	width = _width;
	height = _height;
	gpuErrchk( cudaHostAlloc((void **) &hostArray, width*height*sizeof(T), cudaHostAllocWriteCombined | cudaHostAllocMapped) );
	gpuErrchk( cudaHostGetDevicePointer(&deviceArray, hostArray, 0) );
}
	
template<typename T>
int Args2D<T>::getWidth() const{
	return width;
}
	
template<typename T>
int Args2D<T>::getHeight() const{
	return height;
}
	
template<typename T>
T & Args2D<T>::operator()(int x,int y) const {
	#ifdef __CUDA_ARCH__
		return deviceArray[y*width+x];
	#else
		return hostArray[y*width+x];
	#endif
}	

//*******************************************************************************************
// ARGS3D
//*******************************************************************************************

template<typename T>
Args3D<T>::Args3D(){};
	
	/*
	__stencil__ ~Args(){
		gpuErrchk( cudaFreeHost(hostArray));
	}
	*/
	
template<typename T>
Args3D<T>::Args3D(int _width, int _height, int _depth){
	width = _width;
	height = _height;
	depth = _depth;
	gpuErrchk( cudaHostAlloc((void **) &hostArray, width*height*depth*sizeof(T), cudaHostAllocWriteCombined | cudaHostAllocMapped) );
	gpuErrchk( cudaHostGetDevicePointer(&deviceArray, hostArray, 0) );
}
	
template<typename T>
int Args3D<T>::getWidth() const{
	return width;
}
	
template<typename T>
int Args3D<T>::getHeight() const{
	return height;
}
	
template<typename T>
int Args3D<T>::getDepth() const{	
	return depth;
}
	
template<typename T>
T & Args3D<T>::operator()(int x,int y,int z) const {
	#ifdef __CUDA_ARCH__
		return deviceArray[(z*height + y)*width + x];
	#else
		return hostArray[(z*height + y)*width + x];
	#endif
}

}//end namespace

#endif
