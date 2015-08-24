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

/**
 * \file PSkelArray.h
 * This file contains the definition for the arrays data structures in PSkel.
*/
#ifndef PSKEL_ARRAY_H
#define PSKEL_ARRAY_H

#ifdef PSKEL_CUDA
#include <cuda.h>
#endif

#include "PSkelDefs.h"

namespace PSkel{

/**
 * Class that implements the basic data structure used by the parallel
 * skeletons (such as stencil and map.) PSkel::ArrayBase is a 3D array that is also
 * interfaced via 1D and 2D arrays.  The PSkel::ArrayBase data structure that
 * is extended by PSkel::Array, PSkel::Array2D, and PSkel::Array3D.
 **/
template<typename T>
class ArrayBase{
private:
	//variables that hold the real boundaries (total allocated data.)
	size_t realWidth, realHeight, realDepth;
	//variables that hold the "virtual" array boundaries (important for sliced arrays.)
	size_t width, height,depth;
	//offsets for the sliced array.
	size_t widthOffset, heightOffset, depthOffset;
	//host and device (GPU memory) pointers
	T *hostArray;
	#ifdef PSKEL_CUDA
	T *deviceArray;
	#endif
protected:
	#ifdef PSKEL_CUDA
	/**
         * Access a specific element of the array allocated in the device memory.
         * This function is accessible only during the execution in the device environment.
         * \param[in] h height offset for the element being accessed.
         * \param[in] w width offset for the element being accessed.
         * \param[in] d depth offset for the element being accessed.
         * \return the reference of the element specified via parameters.
         **/
	__device__ __forceinline__ T & deviceGet(size_t h,size_t w,size_t d) const ;
	#endif

	/**
         * Access a specific element of the array allocated in the host memory.
         * This function is accessible only during the execution in the host environment.
         * \param[in] h height offset for the element being accessed.
         * \param[in] w width offset for the element being accessed.
         * \param[in] d depth offset for the element being accessed.
         * \return the reference of the element specified via parameters.
         **/
	__host__ __forceinline__ T & hostGet(size_t h,size_t w,size_t d) const ;

	/**
         * The ArrayBase constructor creates and allocates the specified array
         * in the host memory.
         * \param[in] width Width for the 3D array being created.
         * \param[in] height Height for the 3D array being created.
         * \param[in] depth Depth for the 3D array being created.
         **/
	ArrayBase(size_t width, size_t height, size_t depth);
public:	
	#ifdef PSKEL_CUDA
	/**
	 * Allocates the "virtual" array in device memory.
	 **/
	void deviceAlloc();
	#endif

	#ifdef PSKEL_CUDA
	/**
	 * Frees the allocated device memory.
	 **/
	void deviceFree();
	#endif

	void hostAlloc(size_t width, size_t height, size_t depth);

	/**
	 * Allocates the array in host (main) memory.
	 **/
	void hostAlloc();

	/**
	 * Frees the allocated host (main) memory.
	 **/
	void hostFree();
	
	/**
	 * Get the width size of the "virtual" array.
	 * \return the "virtual" width of the array data structure.
	 **/
	__device__ __host__ size_t getWidth() const;
	
	/**
	 * Get the height size of the "virtual" array.
	 * \return the "virtual" height of the array data structure.
	 **/
	__device__ __host__ size_t getHeight() const;
	
	/**
	 * Get the depth size of the "virtual" array.
	 * \return the "virtual" depth of the array data structure.
	 **/
	__device__ __host__ size_t getDepth() const;
	
	/**
	 * Get the size, in bytes, of the allocated memory for the "virtual" array.
	 * \return the total of bytes allocated in memory for the "virtual" array.
	 **/
	__device__ __host__ size_t memSize() const;

	/**
	 * Get the size of the "virtual" array, i.e. the number of elements 
	 * \return the size of the "virtual" array.
	 **/
	__device__ __host__ size_t size() const;

	/**
	 * Get the size of the real allocated array, i.e. the number of elements 
	 * \return the size of the real allocated array.
	 **/
	__device__ __host__ size_t realSize() const;

	/**
	 * Creates a sliced reference, in the host (main) memory, of the array given as argument.
	 * The slice points to the same memory space as the sliced array.
	 * \param[in] array original array that will be sliced.
	 * \param[in] widthOffset the width offset for the sliced region, relative to the array given as argument.
	 * \param[in] heightOffset the height offset for the sliced region, relative to the array given as argument.
	 * \param[in] depthOffset the depth offset for the sliced region, relative to the array given as argument.
	 * \param[in] width the width of the slice.
	 * \param[in] height the height of the slice.
	 * \param[in] depth the depth of the slice.
	 **/
	template<typename Arrays>
	void hostSlice(Arrays array, size_t widthOffset, size_t heightOffset, size_t depthOffset, size_t width, size_t height, size_t depth);

	/**
	 * Creates a clone, in the host (main) memory, of the array given as argument.
	 * The clone is a copy of the array in a different memory space.
	 * \param[in] array original array that will be cloned.
	 **/
	template<typename Arrays>
	void hostClone(Arrays array);
	
	/**
	 * Copies the data, in the host (main) memory, from the array given as argument.
	 * \param[in] array original array that will be copied.
	 **/
	template<typename Arrays>
	void hostMemCopy(Arrays array);

	#ifdef PSKEL_CUDA
	/**
	 * The array is copied from the host allocated memory to the device allocated memory.
	 * The data is efficiently transferred from host to device.
	 * Both the host and device memory must be allocated before the data is transferred.
	 **/
	void copyToDevice();
	#endif

	#ifdef PSKEL_CUDA
	/**
	 * The array given as argument is copied from the device allocated memory to the host allocated memory of this array.
	 * The data is efficiently transferred from device to host.
	 * \param[in] array the source array that holds the data that will be copied from device to the host memory of this array.
	 **/
	template<typename Arrays>
	void copyFromDevice(Arrays array);
	#endif

	#ifdef PSKEL_CUDA
	/**
	 * The array is copied from the device allocated memory to the host allocated memory.
	 * The data is efficiently transferred from device to host.
	 * Both the host and device memory must be allocated before the data is transferred.
	 **/
	void copyToHost();
	#endif
	
	/**
	 * Verifies if there is memory allocated for the array data structure.
	 * This function can be called both from device and host environment,
	 * and the respective memory space is verified.
	 * \return true if there is a valid memory spaced allocated for the array; false otherwise.
	 **/
	__device__ __host__ operator bool() const ;
};

//*******************************************************************************************
// Array 3D
//*******************************************************************************************

template <typename T>
class Array3D: public ArrayBase<T>{
public:
	/**
         * The Array3D default constructor creates an empty array withour allocating memory space.
         **/
	Array3D();
	
	/*
	//TODO O kernel cuda não aceita structs com destrutores. Corrigir nas próximas versões
	~Array3D(){
		free(hostArray);
		cudaFree(deviceArray);
	}*/
	
	/**
         * The Array3D constructor creates and allocates the specified 3-dimensional array
         * in the host memory.
         * \param[in] width width for the 3D array being created.
         * \param[in] height height for the 3D array being created.
         * \param[in] depth depth for the 3D array being created.
         **/
	Array3D(size_t width, size_t height, size_t depth);
	
	/**
         * Access a specific element of the array allocated in the memory space
         * relative to the execution environment, i.e. either in the host or device memory.
         * \param[in] h height offset for the element being accessed.
         * \param[in] w width offset for the element being accessed.
         * \param[in] d depth offset for the element being accessed.
         * \return the reference of the element specified via parameters.
         **/
	__attribute__((always_inline)) __forceinline__ __device__ __host__ T & operator()(size_t h,size_t w,size_t d) const ;
	
};

//*******************************************************************************************
// Array 2D
//*******************************************************************************************

template<typename T>
class Array2D: public ArrayBase<T>{
public:	
	/**
         * The Array2D default constructor creates an empty array withour allocating memory space.
         **/
	Array2D();
	
	/**
         * The Array2D constructor creates and allocates the specified 2-dimensional array
         * in the host memory.
         * \param[in] width width for the 2D array being created.
         * \param[in] height height for the 2D array being created.
         **/
	Array2D(size_t width, size_t height);
	
	/**
         * Access a specific element of the array allocated in the memory space
         * relative to the execution environment, i.e. either in the host or device memory.
         * \param[in] h height offset for the element being accessed.
         * \param[in] w width offset for the element being accessed.
         * \return the reference of the element specified via parameters.
         **/
	__attribute__((always_inline)) __forceinline__ __device__ __host__ T & operator()(size_t h, size_t w) const ;
	
};

//*******************************************************************************************
// Array 1D
//*******************************************************************************************

template<typename T>
class Array: public ArrayBase<T>{
public:
	/**
         * The Array default constructor creates an empty array withour allocating memory space.
         **/
	Array();
	
	/**
         * The Array constructor creates and allocates the specified 1-dimensional array
         * in the host memory.
         * \param[in] size size for the 1D array being created.
         **/
	Array(size_t size);

	/**
         * Access a specific element of the array allocated in the memory space
         * relative to the execution environment, i.e. either in the host or device memory.
         * \param[in] w offset for the element being accessed.
         * \return the reference of the element specified via parameters.
         **/
	__attribute__((always_inline)) __forceinline__ __device__ __host__ T & operator()(size_t w) const;	
};

}//end namespace

#include "PSkelArray.hpp"

#endif
