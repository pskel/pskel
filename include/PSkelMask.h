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
 * \file PSkelMask.h
 * This file contains the definition for the mask data structures, which is used by the stencil skeleton.
*/
#ifndef PSKEL_MASK_H
#define PSKEL_MASK_H

#include "PSkelArray.h"

namespace PSkel{

/**
 * MaskBase is the basic class that implements the mask data structure used by the stencil skeleton
 * in order to select the neighborhood for each element of a given input array.
 **/
template<typename T>
class MaskBase{
protected:
	int *hostMask;
	#ifdef PSKEL_CUDA
	int *deviceMask;
	#endif
	//size_t maskRadius;
	T *hostWeight;
	#ifdef PSKEL_CUDA
	T *deviceWeight;
	#endif
	T haloValue;
	
	/**
         * The MaskBase constructor creates and allocates the specified mask
         * in the host memory.
         * \param[in] size the size of the mask.
         * \param[in] dimension the dimension of the mask.
         * \param[in] haloVal the value used when the array is accessed out of bounds.
         * \param[in] range the range of the mask; if range is 0,
         * it is calculated as the maximum absolute value on the mask.
         **/
	MaskBase(size_t size=0, size_t dimension=0, T haloVal=T(0), size_t range=0);
	//template<size_t rows, size_t cols>
	//MaskBase(size_t size, int (&array)[][]);
public:
	size_t size, dimension;
	size_t range;

	#ifdef PSKEL_CUDA
	/**
	 * Allocates the mask in device memory, including both the indexes and the weights.
	 **/
	void deviceAlloc();
	void copyToDevice();
	/**
	 * Frees the allocated device memory.
	 **/
	void deviceFree();
	#endif
	
	/**
	 * Allocates the mask in host (main) memory, including both the indexes and the weights.
	 **/
	void hostAlloc();

	/**
	 * Frees the allocated host (main) memory.
	 **/
	void hostFree();

	/**
	 * Get the size, in bytes, of the allocated memory for storing the mask.
	 * \return the total of bytes allocated in memory for storing the mask.
	 **/
	size_t memSize() const;

	//__device__ __host__ size_t size() const;
	
	/**
	 * Get the weight of a given element in the mask.
	 * \param[in] n index of the element in the mask. 
	 * \return the weight of the specified element.
	 **/
	__device__ __host__ T & getWeight(size_t n) const;
	
	#ifdef PSKEL_CUDA	
	__device__ T* GetSharedPointer( void ){
        	extern __device__ void error( void );
        	error();
        	return NULL;	
    	}
	#endif
};


//*******************************************************************************************
// MASK3D
//*******************************************************************************************

template<typename T>
class Mask3D : public MaskBase<T>{
public:
	/**
         * The Mask3D constructor creates and allocates the specified 3-dimensional mask
         * in the host memory.
         * \param[in] size the size of the 3D mask.
         * \param[in] haloVal the value used when the array is accessed out of bounds.
         * \param[in] range the range of the mask; if range is 0,
         * it is calculated as the maximum absolute value on the mask.
         **/
	Mask3D(size_t size=0, T haloVal=T(0), size_t range=0);

	/**
         * Set the mask information for accessing the n-th neighbor for a given element.
         * \param[in] n the index of the neighbor.
         * \param[in] h the height index translation needed for acessing the n-th neighbor.
         * \param[in] w the width index translation needed for acessing the n-th neighbor.
         * \param[in] d the depth index translation needed for acessing the n-th neighbor.
         * \param[in] weight the weight defined for the n-th neighbor.
         **/
	__device__ __host__ void set(size_t n, int h, int w, int d, T weight=T(0));
	
	//__device__ __host__ void set(size_t n,int x,int y,int z);
	
	/**
         * Get the n-th neighbor from the specified input array.
         * \param[in] n the index of the neighbor.
         * \param[in] array the input 3D array.
         * \param[in] h the height index for the central element.
         * \param[in] w the width index for the central element.
         * \param[in] d the depth index for the central element.
         * \return the n-th neighbor of the given central element, from the input array.
         **/
	template<typename V>
	__forceinline __device__ __host__ T get(size_t n, Array3D<V> array, size_t h, size_t w, size_t d);
	
	__device__ __host__ size_t getRange();
	/*
        template<typename Arrays>
	int setMaskRadius(Arrays array);
	*/
};

//*******************************************************************************************
// MASK2D
//*******************************************************************************************

template<typename T>
class Mask2D : public MaskBase<T>{
public:	
	/**
         * The Mask2D constructor creates and allocates the specified 2-dimensional mask
         * in the host memory.
         * \param[in] size the size of the 2D mask.
         * \param[in] haloVal the value used when the array is accessed out of bounds.
         * \param[in] range the range of the mask; if range is 0,
         * it is calculated as the maximum absolute value on the mask.
         **/
	Mask2D(size_t size=0, T haloVal=T(0), size_t range=0);
	
	Mask2D(size_t size, int array[][2]);
	
	/**
         * Set the mask information for accessing the n-th neighbor for a given element.
         * \param[in] n the index of the neighbor.
         * \param[in] h the height index translation needed for acessing the n-th neighbor.
         * \param[in] w the width index translation needed for acessing the n-th neighbor.
         * \param[in] weight the weight defined for the n-th neighbor.
         **/
	__device__ __host__ void set(size_t n, int h, int w, T weight=T(0));
	
	//__device__ __host__ void set(size_t n,int x,int y);
	
	/**
         * Get the n-th neighbor from the specified input array.
         * \param[in] n the index of the neighbor.
         * \param[in] array the input 2D array.
         * \param[in] h the height index for the central element.
         * \param[in] w the width index for the central element.
         * \return the n-th neighbor of the given central element, from the input array.
         **/
	template<typename V>
	__forceinline __device__ __host__ T & get(size_t n, Array2D<V> array, size_t h, size_t w) const;
	
	__device__ __host__ size_t getRange();
	/*
        template<typename Arrays>
	int setMaskRadius(Arrays array);
	*/
};

//*******************************************************************************************
// MASK
//*******************************************************************************************

template<typename T>
class Mask : public MaskBase<T>{
public:
	/**
         * The Mask constructor creates and allocates the specified 1-dimensional mask
         * in the host memory.
         * \param[in] size the size of the 1D mask.
         * \param[in] haloVal the value used when the array is accessed out of bounds.
         * \param[in] range the range of the mask; if range is 0,
         * it is calculated as the maximum absolute value on the mask.
         **/
	Mask(size_t size=0, T haloVal=T(0), size_t range=0);
		
	//__device__ __host__ void set(size_t n,int x);
	
	/**
         * Set the mask information for accessing the n-th neighbor for a given element.
         * \param[in] n the index of the neighbor.
         * \param[in] i the index translation needed for acessing the n-th neighbor.
         * \param[in] weight the weight defined for the n-th neighbor.
         **/
	__device__ __host__ void set(size_t n, int i, T weight=T(0));
	
	/**
         * Get the n-th neighbor from the specified input array.
         * \param[in] n the index of the neighbor.
         * \param[in] array the input 1D array.
         * \param[in] i the index for the central element.
         * \return the n-th neighbor of the given central element, from the input array.
         **/
	template<typename V>
	__forceinline__ __device__ __host__ T get(size_t n, Array<V> array, size_t i);
	
	__device__ __host__ size_t getRange();
	/*
        template<typename Arrays>
	int setMaskRadius(Arrays array);
	*/
};

}//end namespace

#include "PSkelMask.hpp"

#endif
