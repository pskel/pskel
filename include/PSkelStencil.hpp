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

#ifndef PSKEL_STENCIL_HPP
#define PSKEL_STENCIL_HPP

#include <cmath>
#include <algorithm>
#include <iostream>

#include <iostream>

using namespace std;

namespace PSkel{

#ifdef PSKEL_CUDA
//********************************************************************************************
// Kernels CUDA. Chama o kernel implementado pelo usuario
//********************************************************************************************

//template<typename T1, typename T2, class Args>
//__global__ void stencilTilingCU(Array<T1> input,Array<T1> output,Mask<T2> mask,Args args, size_t widthOffset, size_t heightOffset, size_t depthOffset, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth);

//template<typename T1, typename T2, class Args>
//__global__ void stencilTilingCU(Array2D<T1> input,Array2D<T1> output,Mask2D<T2> mask,Args args, size_t widthOffset, size_t heightOffset, size_t depthOffset, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth);

//template<typename T1, typename T2, class Args>
//__global__ void stencilTilingCU(Array3D<T1> input,Array3D<T1> output,Mask3D<T2> mask,Args args, size_t widthOffset, size_t heightOffset, size_t depthOffset, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth);


//********************************************************************************************
// Kernels CUDA. Chama o kernel implementado pelo usuario
//********************************************************************************************
/*template<typename T>
struct SharedMemory
{
    // Should never be instantiated.
    // We enforce this at compile time.
    __device__ T* GetPointer( void )
    {
        extern __device__ void error( void );
        error();
        return NULL;
    }
};

// specializations for types we use
template<>
struct SharedMemory<float>
{
    size_t width;
    size_t range;
    extern __shared__ float sh_float[];
    __device__ float* GetPointer(size_t blockWidth, size_t maskRange){
        width = BlockWidth;
        range = maskRange;
        // printf( "sh_float=%p\n", sh_float );
        return sh_float;
    }
    
    __device__ float get(size_t h, size_t w){
		return sh_float[(h+range)*(width+2*range)+(w+range)];
	}
};
*/
/*
__device__ size_t ToGlobalRow( int gidRow, int lszRow, int lidRow ){
    return gidRow * lszRow + lidRow;
}

__device__ size_t ToGlobalCol( int gidCol, int lszCol, int lidCol ){
    return gidCol * lszCol + lidCol;
}

__device__ int ToFlatIdx( int row, int col, int rowWidth ){
    // assumes input coordinates and dimensions are logical (without halo)
    // and a halo width of 1
    return (row+1)*(rowWidth + 2) + (col+1);
}
*/

/*
template<typename T1, typename T2, class Args>
__global__ void StencilKernel(Array2D<T> input, Array2D<T> output, Args, args, int alignment, int nStripItems){
    // determine our location in the coordinate system
    // see the comment in operator() at the definition of the dimGrid
    // and dimBlock dim3s to understand why .x == row and .y == column.
    int gidRow = blockIdx.x;
    int gidCol = blockIdx.y;
    // int gszRow = gridDim.x;
    int gszCol = gridDim.y;
    int lidRow = threadIdx.x;
    int lidCol = threadIdx.y;
    int lszRow = nStripItems;
    int lszCol = blockDim.y;

    // determine our logical global data coordinates (without halo)
    int gRow = ToGlobalRow( gidRow, lszRow, lidRow );
    int gCol = ToGlobalCol( gidCol, lszCol, lidCol );

    // determine pitch of rows (without halo)
    int nCols = gszCol * lszCol + 2;     // assume halo is there for computing padding
    int nPaddedCols = nCols + (((nCols % alignment) == 0) ? 0 : (alignment - (nCols % alignment)));
    int gRowWidth = nPaddedCols - 2;    // remove the halo

    // Copy my global data item to a shared local buffer.
    // That local buffer is passed to us.
    // We assume it is large enough to hold all the data computed by
    // our thread block, plus a halo of width 1.
    SharedMemory<T> shobj;
    T* sh = shobj.GetPointer();
    int lRowWidth = lszCol;
    for( int i = 0; i < (lszRow + 2); i++ )
    {
        int lidx = ToFlatIdx( lidRow - 1 + i, lidCol, lRowWidth );
        int gidx = ToFlatIdx( gRow - 1 + i, gCol, gRowWidth );
        sh[lidx] = data[gidx];
    }

    // Copy the "left" and "right" halo rows into our local memory buffer.
    // Only two threads are involved (first column and last column).
    if( lidCol == 0 ){
        for( int i = 0; i < (lszRow + 2); i++ ){
            int lidx = ToFlatIdx(lidRow - 1 + i, lidCol - 1, lRowWidth );
            int gidx = ToFlatIdx(gRow - 1 + i, gCol - 1, gRowWidth );
            sh[lidx] = data[gidx];
        }
    } else if( lidCol == (lszCol - 1) ){
        for( int i = 0; i < (lszRow + 2); i++ ) {
            int lidx = ToFlatIdx(lidRow - 1 + i, lidCol + 1, lRowWidth );
            int gidx = ToFlatIdx(gRow - 1 + i, gCol + 1, gRowWidth );
            sh[lidx] = data[gidx];
        }
    }

    // let all those loads finish
    __syncthreads();

    // do my part of the smoothing operation
    for( int i = 0; i < lszRow; i++ ) {
        int cidx  = ToFlatIdx( lidRow     + i, lidCol    , lRowWidth );
        int nidx  = ToFlatIdx( lidRow - 1 + i, lidCol    , lRowWidth );
        int sidx  = ToFlatIdx( lidRow + 1 + i, lidCol    , lRowWidth );
        int eidx  = ToFlatIdx( lidRow     + i, lidCol + 1, lRowWidth );
        int widx  = ToFlatIdx( lidRow     + i, lidCol - 1, lRowWidth );
        int neidx = ToFlatIdx( lidRow - 1 + i, lidCol + 1, lRowWidth );
        int seidx = ToFlatIdx( lidRow + 1 + i, lidCol + 1, lRowWidth );
        int nwidx = ToFlatIdx( lidRow - 1 + i, lidCol - 1, lRowWidth );
        int swidx = ToFlatIdx( lidRow + 1 + i, lidCol - 1, lRowWidth );

        T centerValue = sh[cidx];
        T cardinalValueSum = sh[nidx] + sh[sidx] + sh[eidx] + sh[widx];
        T diagonalValueSum = sh[neidx] + sh[seidx] + sh[nwidx] + sh[swidx];

        newData[ToFlatIdx(gRow + i, gCol, gRowWidth)] = wCenter * centerValue +
                wCardinal * cardinalValueSum +
                wDiagonal * diagonalValueSum;
    }
}*/


template<typename T1, typename T2, class Args>
__global__ void stencilTilingCU(Array<T1> input,Array<T1> output,Mask<T2> mask,Args args, size_t widthOffset, size_t heightOffset, size_t depthOffset, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth){
	size_t i = blockIdx.x*blockDim.x+threadIdx.x;
	#ifdef PSKEL_SHARED_MASK
	extern __shared__ int shared[];
 	if(threadIdx.x<(mask.size*mask.dimension))
		shared[threadIdx.x] = mask.deviceMask[threadIdx.x];
	__syncthreads();
	mask.deviceMask = shared;
	#endif
	if(i>=widthOffset && i<(widthOffset+tilingWidth)){
		stencilKernel(input, output, mask, args, i);
	}
}

/* Shared memory kernel development */
#ifdef PSKEL_SHARED
#define TIME_TILE_SIZE 2

extern __shared__ float sh_input[];

template<typename T1, typename T2, class Args>
__global__ void stencilTilingCU(Array2D<T1> input,Array2D<T1> output,Mask2D<T2> mask,Args args, size_t maskRange, size_t timeTileSize, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth){
  // Determine our start position
    int offsetI = blockIdx.y * (blockDim.y-2*(timeTileSize-1)) + threadIdx.y;
    offsetI -= timeTileSize-1;
    int offsetJ = blockIdx.x * (blockDim.x-2*(timeTileSize-1)) + threadIdx.x;
    offsetJ -= timeTileSize-1;
    
    #ifdef PSKEL_DEBUG
    printf("STEP 1 - offset %d %d\n",offsetI,offsetJ);
    #endif
    //__shared__ T1 sh_input[(BLOCK_SIZE + 2*(TIME_TILE_SIZE-1))*(BLOCK_SIZE + 2*(TIME_TILE_SIZE-1))];
  
    sh_input[threadIdx.y*blockDim.y+threadIdx.x] = ((offsetI >= 0) && (offsetI <= (tilingHeight-1)) &&
    (offsetJ >= 0) && (offsetJ <= (tilingWidth-1))) ? input(offsetI,offsetJ) : sh_input[threadIdx.y*blockDim.y+threadIdx.x];
    
    #ifdef PSKEL_DEBUG
    printf("STEP 2 - sh_intput[%d] = %f\n",threadIdx.y*blockDim.y+threadIdx.x,sh_input[threadIdx.y*blockDim.y+threadIdx.x]);
    #endif
    __syncthreads();
  
    for(int t = 0; t < timeTileSize-1; ++t) {
		//stencilComputation
		//printf("Computing it %d\n",t);
		/*T1 l = ((offsetI-1 >= 0) && (offsetI-1 <= (blockDim.y-1)) &&
                 (offsetJ >= 0) && (offsetJ <= (blockDim.x-1)))
                 ? sh_input[(offsetI-1)*blockDim.y+offsetJ] : 0.0f;
                 
		T1 r = ((offsetI+1 >= 0) && (offsetI+1 <= (blockDim.y-1)) &&
                 (offsetJ >= 0) && (offsetJ <= (blockDim.x-1)))
                 ? sh_input[(offsetI+1)*blockDim.y+offsetJ] : 0.0f;
                 
		T1 t = ((offsetI >= 0) && (offsetI <= (blockDim.y-1)) &&
                 (offsetJ-1 >= 0) && (offsetJ-1 <= (blockDim.x-1)))
                 ? sh_input[offsetI*blockDim.y+offsetJ-1] : 0.0f;
                 
		T1 b = ((offsetI >= 0) && (offsetI <= (blockDim.y-1)) &&
                 (offsetJ+1 >= 0) && (offsetJ+1 <= (blockDim.x-1)))
                 ? sh_input[offsetI*blockDim.y+offsetJ+1] : 0.0f;
		*/
		
		T1 l = (threadIdx.y > 0) ? sh_input[(threadIdx.y-1)*blockDim.y+threadIdx.x] : 0.0f;
		T1 r = (threadIdx.y < blockDim.y-1) ? sh_input[(threadIdx.y+1)*blockDim.y+threadIdx.x] : 0.0f;
		T1 t = (threadIdx.x > 0) ? sh_input[threadIdx.y*blockDim.y+threadIdx.x-1] : 0.0f;
		T1 b = (threadIdx.x < blockDim.x-1) ? sh_input[threadIdx.y*blockDim.y+threadIdx.x+1] : 0.0f;
		#ifdef JACOBI_KERNEL
		T1 val = 0.25f * (l+r+t+b-args.h);
		#else
		#ifdef CLOUDSIM_KERNEL
		T1 c = sh_input[threadIdx.y*blockDim.y+threadIdx.x];
		float xwind = ((offsetI >= 0) && (offsetI <= (tilingHeight-1)) &&
    			       (offsetJ >= 0) && (offsetJ <= (tilingWidth-1))) ? args.wind_x(offsetI,offsetJ) : 0.0f;
        	float ywind =  ((offsetI >= 0) && (offsetI <= (tilingHeight-1)) &&
                               (offsetJ >= 0) && (offsetJ <= (tilingWidth-1))) ? args.wind_y(offsetI,offsetJ) : 0.0f;
        	int xfactor = (xwind>0)?1:-1;
        	int yfactor = (ywind>0)?1:-1;
	
        	T1 sum =  4*c - (l+r+b+t);
         
        	float temperaturaNeighborX = (threadIdx.x > 0 && threadIdx.x < blockDim.x-1) ? sh_input[threadIdx.y*blockDim.y+threadIdx.x+xfactor] : 0.0f;
       		float temperaturaNeighborY = (threadIdx.y > 0 && threadIdx.y < blockDim.y-1) ? sh_input[(threadIdx.y+yfactor)*blockDim.y+threadIdx.x] : 0.0f;
        
        	float componenteVentoY = yfactor * ywind;
     		float componenteVentoX = xfactor * xwind;
        
        	float temp_wind = (-componenteVentoX * ((c - temperaturaNeighborX)*10.0f)) -
                          ( componenteVentoY * ((c - temperaturaNeighborY)*10.0f));
                          
        	float temperatura_conducao = -0.0243f*(sum * 0.25f) * args.deltaT;
        	float result = c + temperatura_conducao;
        	T1 val = result + temp_wind * args.deltaT;
		#else 
		#ifdef GOL_KERNEL
		
		#endif
		#endif
		#endif
		
		#ifdef PSKEL_DEBUG
		printf("STEP 3 - val: %f\n",val);
		#endif
		/*T1 val = 0.25f * (sh_input[(threadIdx.y+1)*blockDim.y+(threadIdx.x)] + 
						  sh_input[(threadIdx.y-1)*blockDim.y+(threadIdx.x)] + 
						  sh_input[(threadIdx.y)*blockDim.y+(threadIdx.x+1)] + 
						  sh_input[(threadIdx.y)*blockDim.y+(threadIdx.x-1)] + 
						  - args.h)
		*/
		__syncthreads();
		
		sh_input[threadIdx.y*blockDim.y+threadIdx.x] =
        		((offsetI >= 0) && (offsetI <= (tilingHeight-1)) &&
        		 (offsetJ >= 0) && (offsetJ <= (tilingWidth-1))) ? val : sh_input[threadIdx.y*blockDim.y+threadIdx.x];

		#ifdef PSKEL_DEBUG
		printf("STEP 4 - sh_intput[%d] = %f\n",threadIdx.y*blockDim.y+threadIdx.x,sh_input[threadIdx.y*blockDim.y+threadIdx.x]);
        #endif
        // Sync before re-reading shared
        __syncthreads();
	}
	
	if(threadIdx.x >= (timeTileSize-1) &&
     threadIdx.x <= (blockDim.x-1-(timeTileSize-1)) &&
     threadIdx.y >= (TIME_TILE_SIZE-1) &&
     threadIdx.y <= (blockDim.y-1-(timeTileSize-1))) {
    output(offsetI,offsetJ) = sh_input[threadIdx.y*blockDim.y+threadIdx.x];
    #ifdef PSKEL_DEBUG
    printf("STEP 5 - output(%d,%d) = %f\n",output(offsetI,offsetJ));
    #endif
	}
}
/* This is not better than naive pskel
template<typename T1, typename T2, class Args>
__global__ void stencilTilingCU(Array2D<T1> input,Array2D<T1> output,Mask2D<T2> mask,Args args, size_t maskRange, size_t iteration, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth, size_t border_cols, size_t border_rows){
	//size_t w = blockIdx.x*blockDim.x+threadIdx.x;
	//size_t h = blockIdx.y*blockDim.y+threadIdx.y;
	__shared__ T1 sh_input[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ T1 sh_output[BLOCK_SIZE][BLOCK_SIZE]; //temporary output
	
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;
	
	 // calculate the small block size
	int small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
	int small_block_cols = BLOCK_SIZE-iteration*2;//EXPAND_RATE

	// calculate the boundary for the block according to 
	// the boundary of its small block
	int blkY = small_block_rows*by-border_rows;
	int blkX = small_block_cols*bx-border_cols;
	int blkYmax = blkY+BLOCK_SIZE-1;
	int blkXmax = blkX+BLOCK_SIZE-1;
	
	// calculate the global thread coordination
	int yidx = blkY+ty;
	int xidx = blkX+tx;

	// load data if it is within the valid input range
	int loadYidx=yidx;
	int loadXidx=xidx;
	//int index = loadYidx*tilingWidth + loadXidx;
	
	if(IN_RANGE(loadYidx, 0, tilingHeight-1) && IN_RANGE(loadXidx, 0, tilingWidth-1)){
		sh_input[ty][tx] = input(loadXidx,loadYidx);  // Load the temperature data from global memory to shared memory
		//power_on_cuda[ty][tx] = power[index];// Load the power data from global memory to shared memory
	}
	__syncthreads();
	
	// effective range within this block that falls within 
	// the valid range of the input data
	// used to rule out computation outside the boundary.
	int validYmin = (blkY < 0) ? -blkY : 0;
	int validYmax = (blkYmax > tilingHeight-1) ? BLOCK_SIZE-1-(blkYmax-tilingHeight+1) : BLOCK_SIZE-1;
	int validXmin = (blkX < 0) ? -blkX : 0;
	int validXmax = (blkXmax > tilingWidth-1) ? BLOCK_SIZE-1-(blkXmax-tilingWidth+1) : BLOCK_SIZE-1;
	
	/ Offset, need to check how this will be implemented. Get function maybe?
	//int N = ty-1;
	//int S = ty+1;
	//int W = tx-1;
	//int E = tx+1;
	
	//N = (N < validYmin) ? validYmin : N;
	//S = (S > validYmax) ? validYmax : S;
	//W = (W < validXmin) ? validXmin : W;
	//E = (E > validXmax) ? validXmax : E;
	
	bool computed;
	for (int i=0; i<iteration ; i++){ 
		computed = false;
		if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
			IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
			IN_RANGE(tx, validXmin, validXmax) && \
			IN_RANGE(ty, validYmin, validYmax) ) {
			computed = true;
			//stencilKernel(sh_input, sh_output, args, ty, tx);
			//stencilKernel(sh_input, sh_output, args, ty, tx);
			sh_output[ty][tx] = 0.25f * (sh_input[ty][tx-1] + sh_input[ty][tx+1] + sh_input[ty-1][tx] + sh_input[ty+1][tx] - args.h);
		}
		__syncthreads();
		if(i==iteration-1)
			break;
		if(computed)	 //Assign the computation range
			sh_input[ty][tx]= sh_output[ty][tx];
		__syncthreads();
	}

	// update the global memory
	// after the last iteration, only threads coordinated within the 
	// small block perform the calculation and switch on ``computed''
	if (computed){
	  output(loadXidx,loadYidx) = sh_output[ty][tx];		
	}
}
*/
#else
template<typename T1, typename T2, class Args>
__global__ void stencilTilingCU(Array2D<T1> input,Array2D<T1> output,Mask2D<T2> mask,Args args, size_t maskRange, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth){
	size_t w = blockIdx.x*blockDim.x+threadIdx.x;
	size_t h = blockIdx.y*blockDim.y+threadIdx.y;
	if(w>=maskRange && w<(tilingWidth-maskRange) && h>=maskRange && h<(tilingHeight-maskRange) ){
		#ifdef PSKEL_SHARED
		//extern __shared__ int shared[];
		//if(threadIdx.x<(mask.size*mask.dimension))
		//	shared[threadIdx.x] = mask.deviceMask[threadIdx.x];
		//__syncthreads();
		//mask.deviceMask = shared;
		//#endif
		//SharedMemory<T1> shobj;
		//T1* shared = mask.GetPointer(blockDim.x,maskRange);
		T1* shared = mask.GetSharedPointer();
		size_t index;
		/*
		if(threadIdx.x < maskRange){
			shared[threadIdx.x*blockDim.x+threadIdx.y] = input(h-maskRange,w);
		}
		if(threadIdx.y < maskRange){
			shared[threadIdx.x*blockDim.x+threadIdx.y] = input(h,w-maskRange);
		}
		if(threadIdx.x > blockDim.x-maskRange){
			shared[threadIdx.x*blockDim.x+threadIdx.y] = input(h+maskRange,w);
		}
		if(threadIdx.y > blockDim.y-maskRange){
			shared[threadIdx.x*blockDim.x+threadIdx.y] = input(h,w+maskRange);
		}
		*/
		//Copy the left and right halo rows into shared shared memory
		if(threadIdx.x == 0){ //first col of the block
			for(size_t i=0;i<maskRange;i++){
				index = (threadIdx.y+i)*(blockDim.x+2*maskRange)+(threadIdx.x);
				shared[index] = input(h,w-1);
				//printf("blx %d bly %d tx %d ty %d: index: %d\n",blockDim.x,blockDim.y,threadIdx.x,threadIdx.y,index);
			}
		}
		else if(threadIdx.x == blockDim.x-1){ //last col of the block
			for(size_t i=1;i<=maskRange;i++){
				index = (threadIdx.y+i)*(blockDim.x+2*maskRange)+(threadIdx.x+maskRange+i);
				//printf("blx %d bly %d tx %d ty %d: index: %d\n",blockDim.x,blockDim.y,threadIdx.x,threadIdx.y,index);
				shared[index] = input(h,w+1);
			}
		}
		if(threadIdx.y == 0){ //first row of the block
			for(size_t i=0;i<maskRange;i++){
				index = (threadIdx.y)*(blockDim.x+2*maskRange)+(threadIdx.x+i);
				//printf("blx %d bly %d tx %d ty %d: index: %d\n",blockDim.x,blockDim.y,threadIdx.x,threadIdx.y,index);
				shared[index] = input(h-1,w);
			}
		}
		if(threadIdx.y == blockDim.y-1){//last row of the block
			for(size_t i=1;i<=maskRange;i++){
				index = (threadIdx.y+maskRange+i)*(blockDim.x+2*maskRange)+(threadIdx.x+i);
				//printf("blx %d bly %d tx %d ty %d: index: %d\n",blockDim.x,blockDim.y,threadIdx.x,threadIdx.y,index);
				shared[index] = input(h+1,w);
			}
		}		
		
		index = (threadIdx.y+maskRange)*(blockDim.x+2*maskRange)+(threadIdx.x+maskRange);
		//printf("%d position\n",index);
		shared[index] = input(h,w);
		__syncthreads();
		
		/*
		if(threadIdx.x==0 && threadIdx.y==0){
			for(size_t j = 0;j<(blockDim.y+2*maskRange);j++){
				for(size_t i = 0;i<(blockDim.x+2*maskRange);i++){
					index = j*(blockDim.x+2*maskRange)+i;
					printf("id (%d,%d) %d value %f \n",index, shared[index]);
				}
			}
		}*/
			
		stencilKernel(input, output, shared, args, h, w, threadIdx.x, threadIdx.y);
		#else
		stencilKernel(input, output, mask, args, h, w);
		#endif
	}
}
#endif

/*
template<typename T1, typename T2, class Args>
__global__ void stencilTilingCU(Array2D<T1> input,Array2D<T1> output,Mask2D<T2> mask,Args args, size_t maskRange, size_t widthOffset, size_t heightOffset, size_t depthOffset, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth){
	size_t w = blockIdx.x*blockDim.x+threadIdx.x;
	size_t h = blockIdx.y*blockDim.y+threadIdx.y;
	
	#ifdef PSKEL_SHARED_MASK
	T1* shared = mask.GetSharedPointer();
	if(threadIdx.x < maskRange){
		shared[(threadIdx.x-maskRange)*blockDim.x+threadIdx.y] = input(h-maskRange,w);
	}
	if(threadIdx.y < maskRange){
		shared[threadIdx.x*blockDim.x+(threadIdx.y-maskRange)] = input(h,w-maskRange);
	}
	if(threadIdx.x > blockDim.x-maskRange){
		shared[(threadIdx.x+maskRange)*blockDim.x+threadIdx.y] = input(h+maskRange,w);
	}
	if(threadIdx.y > blockDim.y-maskRange){
		shared[threadIdx.x*blockDim.x+(threadIdx.y+maskRange)] = input(h,w+maskRange);
	}
	shared[threadIdx.x*blockDim.x+threadIdx.y] = input(h,w);
	__syncthreads();
	#endif
     	//Ignores all borders except the lower one
	if(w>=(widthOffset+maskRange) && w<(widthOffset+tilingWidth-maskRange) && h>=(heightOffset+maskRange) && h<(heightOffset+tilingHeight) ){
		stencilKernel(input, output, mask, args, h, w);
	}
}
*/
template<typename T1, typename T2, class Args>
__global__ void stencilTilingCU(Array3D<T1> input,Array3D<T1> output,Mask3D<T2> mask,Args args, size_t widthOffset, size_t heightOffset, size_t depthOffset, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth){
	size_t w = blockIdx.x*blockDim.x+threadIdx.x;
	size_t h = blockIdx.y*blockDim.y+threadIdx.y;
	size_t d = blockIdx.z*blockDim.z+threadIdx.z;
	#ifdef PSKEL_SHARED_MASK
	extern __shared__ int shared[];
  	if(threadIdx.x<(mask.size*mask.dimension))
		shared[threadIdx.x] = mask.deviceMask[threadIdx.x];
	__syncthreads();
	mask.deviceMask = shared;
	#endif
  
	if(w>=widthOffset && w<(widthOffset+tilingWidth) && h>=heightOffset && h<(heightOffset+tilingHeight) && d>=depthOffset && d<(depthOffset+tilingDepth) ){
		stencilKernel(input, output, mask, args, h, w, d);
	}
}
#endif

//*******************************************************************************************
// Stencil Base
//*******************************************************************************************

template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runSequential(){
	this->runSeq(this->input, this->output);
}

template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runCPU(size_t numThreads){
	numThreads = (numThreads==0)?omp_get_num_procs():numThreads;
	#ifdef PSKEL_TBB
		this->runTBB(this->input, this->output, numThreads);
	#else
		this->runOpenMP(this->input, this->output, numThreads);
	#endif
}

#ifdef PSKEL_CUDA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runGPU(size_t GPUBlockSizeX, size_t GPUBlockSizeY){
	if(GPUBlockSizeX==0){
		int device;
		cudaGetDevice(&device);
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, device);
		//GPUBlockSize = deviceProperties.maxThreadsPerBlock/2;
		GPUBlockSizeX = GPUBlockSizeY = deviceProperties.warpSize;
		//int minGridSize, blockSize;
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, stencilTilingCU, 0, in.size());
		//GPUBlockSize = blockSize;
		//cout << "GPUBlockSize: "<<GPUBlockSize<<endl;
		//int maxActiveBlocks;
  		//cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, stencilTilingCU, GPUBlockSize, 0);
		//float occupancy = (maxActiveBlocks * GPUBlockSize / deviceProperties.warpSize) / 
                //    (float)(deviceProperties.maxThreadsPerMultiProcessor / 
                //            deviceProperties.warpSize);
		//printf("Launched blocks of size %d. Theoretical occupancy: %f\n", GPUBlockSize, occupancy);
	}
	input.deviceAlloc();
	output.deviceAlloc();
	mask.deviceAlloc();
	mask.copyToDevice();
	input.copyToDevice();
	//this->setGPUInputData();
	this->runCUDA(this->input, this->output, GPUBlockSizeX, GPUBlockSizeY);
	//this->getGPUOutputData();
	output.copyToHost();
	input.deviceFree();
	output.deviceFree();
	mask.deviceFree();
}
#endif

#ifdef PSKEL_CUDA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runTilingGPU(size_t tilingWidth, size_t tilingHeight, size_t tilingDepth, size_t GPUBlockSizeX, size_t GPUBlockSizeY){
	if(GPUBlockSizeX==0){
		int device;
		cudaGetDevice(&device);
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, device);
		//GPUBlockSize = deviceProperties.maxThreadsPerBlock/2;
		GPUBlockSizeX = GPUBlockSizeY = deviceProperties.warpSize;
		//int minGridSize, blockSize;
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, stencilTilingCU, 0, in.size());
		//GPUBlockSize = blockSize;
		//cout << "GPUBlockSize: "<<GPUBlockSize<<endl;
		//int maxActiveBlocks;
  		//cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, stencilTilingCU, GPUBlockSize, 0);
		//float occupancy = (maxActiveBlocks * GPUBlockSize / deviceProperties.warpSize) / 
                //    (float)(deviceProperties.maxThreadsPerMultiProcessor / 
                //            deviceProperties.warpSize);
		//printf("Launched blocks of size %d. Theoretical occupancy: %f\n", GPUBlockSize, occupancy);
	}
	size_t wTiling = ceil(float(this->input.getWidth())/float(tilingWidth));
	size_t hTiling = ceil(float(this->input.getHeight())/float(tilingHeight));
	size_t dTiling = ceil(float(this->input.getDepth())/float(tilingDepth));
	mask.deviceAlloc();
	mask.copyToDevice();
	//setGPUMask();
	StencilTiling<Array, Mask> tiling(input, output, mask);
	Array inputTile;
	Array outputTile;
	Array tmp;
	for(size_t ht=0; ht<hTiling; ht++){
	 for(size_t wt=0; wt<wTiling; wt++){
	  for(size_t dt=0; dt<dTiling; dt++){
		size_t heightOffset = ht*tilingHeight;
		size_t widthOffset = wt*tilingWidth;
		size_t depthOffset = dt*tilingDepth;
		//CUDA input memory copy
		tiling.tile(1, widthOffset, heightOffset, depthOffset, tilingWidth, tilingHeight, tilingDepth);
		inputTile.hostSlice(tiling.input, tiling.widthOffset, tiling.heightOffset, tiling.depthOffset, tiling.width, tiling.height, tiling.depth);
		outputTile.hostSlice(tiling.output, tiling.widthOffset, tiling.heightOffset, tiling.depthOffset, tiling.width, tiling.height, tiling.depth);
		inputTile.deviceAlloc();
		outputTile.deviceAlloc();
		inputTile.copyToDevice();
		tmp.hostAlloc(tiling.width, tiling.height, tiling.depth);
		//this->setGPUInputDataIterative(inputCopy, output, innerIterations, widthOffset, heightOffset, depthOffset, tilingWidth, tilingHeight, tilingDepth);
		//CUDA kernel execution
		this->runIterativeTilingCUDA(inputTile, outputTile, tiling, GPUBlockSizeX, GPUBlockSizeY);
		tmp.copyFromDevice(outputTile);
		Array coreTmp;
		Array coreOutput;
		coreTmp.hostSlice(tmp, tiling.coreWidthOffset, tiling.coreHeightOffset, tiling.coreDepthOffset, tiling.coreWidth, tiling.coreHeight, tiling.coreDepth);
		coreOutput.hostSlice(outputTile, tiling.coreWidthOffset, tiling.coreHeightOffset, tiling.coreDepthOffset, tiling.coreWidth, tiling.coreHeight, tiling.coreDepth);
		coreOutput.hostMemCopy(coreTmp);
		tmp.hostFree();
	}}}
	inputTile.deviceFree();
	outputTile.deviceFree();
	mask.deviceFree();
	cudaDeviceSynchronize();
}
#endif

#ifdef PSKEL_CUDA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runAutoGPU(size_t GPUBlockSize){
	size_t gpuMemFree, gpuMemTotal;
	//gpuErrchk( cudaDeviceSynchronize() );
	cudaMemGetInfo(&gpuMemFree, &gpuMemTotal);
	if((this->input.memSize()+this->output.memSize()+this->mask.memSize())<(0.998*gpuMemFree)){
		runGPU(GPUBlockSize);
	}else{
		size_t typeSize = this->input.memSize()/this->input.size();
		float div = float(this->input.memSize()+this->output.memSize())/((gpuMemFree-this->mask.memSize())*0.97);
		if(this->input.getHeight()==1){
			size_t width = floor(float(this->input.getWidth())/div);
			width = (width>0)?width:1;
			while( (((this->input.getHeight()*this->input.getDepth()+this->output.getHeight()*this->output.getDepth())*(2*this->mask.getRange() + width))*typeSize + this->mask.memSize()) > gpuMemFree*0.998 ){
				width+=2;
			}
			while( (((this->input.getHeight()*this->input.getDepth()+this->output.getHeight()*this->output.getDepth())*(2*this->mask.getRange() + width))*typeSize + this->mask.memSize()) > gpuMemFree*0.998 ){
				width--;
			}
			runTilingGPU(width, this->input.getHeight(), this->input.getDepth(), GPUBlockSize);
		}else{
			size_t height = floor(float(this->input.getHeight())/div);
			height = (height>0)?height:1;
			while( (((this->input.getWidth()*this->input.getDepth()+this->output.getWidth()*this->output.getDepth())*(2*this->mask.getRange() + height))*typeSize + this->mask.memSize()) < gpuMemFree*0.998 ){
				height+=2;
			}
			while( (((this->input.getWidth()*this->input.getDepth()+this->output.getWidth()*this->output.getDepth())*(2*this->mask.getRange() + height))*typeSize + this->mask.memSize()) > gpuMemFree*0.998 ){
				height--;
			}
			runTilingGPU(this->input.getWidth(), height, this->input.getDepth(), GPUBlockSize);
		}
	}
}
#endif

template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runIterativeSequential(size_t iterations){
	Array inputCopy;
	inputCopy.hostClone(input);
	for(size_t it = 0; it<iterations; it++){
		if(it%2==0) this->runSeq(inputCopy, this->output);
		else this->runSeq(this->output, inputCopy);
	}
	if((iterations%2)==0) output.hostMemCopy(inputCopy);
	inputCopy.hostFree();
}

template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runIterativeCPU(size_t iterations, size_t numThreads){
	numThreads = (numThreads==0)?omp_get_num_procs():numThreads;
	//cout << "numThreads: " << numThreads << endl;
	Array inputCopy;
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
/*Shared memory development*/
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runIterativeGPU(size_t iterations, size_t pyramidHeight,size_t GPUBlockSizeX, size_t GPUBlockSizeY){
	if(GPUBlockSizeX==0){
		int device;
		cudaGetDevice(&device);
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, device);
		//GPUBlockSize = deviceProperties.maxThreadsPerBlock/2;
		GPUBlockSizeX = deviceProperties.warpSize;
		//int minGridSize, blockSize;
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, stencilTilingCU, 0, in.size());
		//GPUBlockSize = blockSize;
		//cout << "GPUBlockSize: "<<GPUBlockSize<<endl;
		//int maxActiveBlocks;
  		//cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, stencilTilingCU, GPUBlockSize, 0);
		//float occupancy = (maxActiveBlocks * GPUBlockSize / deviceProperties.warpSize) / 
                //    (float)(deviceProperties.maxThreadsPerMultiProcessor / 
                //            deviceProperties.warpSize);
		//printf("Launched blocks of size %d. Theoretical occupancy: %f\n", GPUBlockSize, occupancy);
	}
	input.deviceAlloc();
	input.copyToDevice();
	mask.deviceAlloc();
	mask.copyToDevice();
	output.deviceAlloc();
	size_t maskRange = mask.getRange();

	/* Hotspot method, bad performance */
	/*
	#define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline
    size_t borderCols = (pyramidHeight)*EXPAND_RATE/2;
    size_t borderRows = (pyramidHeight)*EXPAND_RATE/2;
    size_t smallBlockCol = BLOCK_SIZE-(pyramidHeight)*EXPAND_RATE;
    size_t smallBlockRow = BLOCK_SIZE-(pyramidHeight)*EXPAND_RATE;
    size_t blockCols = input.getWidth()  /smallBlockCol+((input.getWidth() % smallBlockCol==0)?0:1);
    size_t blockRows = input.getHeight() /smallBlockRow+((input.getHeight()% smallBlockRow==0)?0:1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    dim3 dimGrid(blockCols, blockRows);

    printf("pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n",\
           pyramidHeight, input.getWidth(), input.getHeight(), borderCols, borderRows, blockCols, blockRows, smallBlockCol, smallBlockRow);  

	for(size_t t = 0; t<iterations; t+=pyramidHeight){
		size_t it = MIN(pyramidHeight, iterations-t);
		if((it%2)==0)
			stencilTilingCU<<<dimGrid, dimBlock>>>(this->input, this->output, this->mask,this->args,maskRange,it,input.getWidth(),input.getHeight(),input.getDepth(),borderCols,borderRows);
		else stencilTilingCU<<<dimGrid, dimBlock>>>(this->output, this->input, this->mask,this->args,maskRange,it,input.getWidth(),input.getHeight(),input.getDepth(),borderCols,borderRows);
 
	*/
	
	/*Howelinsk method */
	dim3 dimGrid(input.getWidth() / GPUBlockSizeX, input.getHeight() / GPUBlockSizeY);
	dim3 dimBlock(GPUBlockSizeX + 2*(pyramidHeight-1), GPUBlockSizeY + 2*(pyramidHeight-1));
	const int sharedMemSize = dimBlock.x * dimBlock.y * sizeof(float); //need to get this size from somewhere
	
	
	for(size_t t = 0; t<iterations; t+=pyramidHeight){
		//size_t it = MIN(pyramidHeight, iterations-t);
		if((t%2)==0)
			stencilTilingCU<<<dimGrid, dimBlock, sharedMemSize>>>(this->input, this->output, this->mask, this->args,maskRange,pyramidHeight,input.getWidth(),input.getHeight(),input.getDepth());
		else stencilTilingCU<<<dimGrid, dimBlock, sharedMemSize>>>(this->output, this->input, this->mask, this->args,maskRange,pyramidHeight,input.getWidth(),input.getHeight(),input.getDepth());
	}
	if((iterations%2)==1)
		output.copyToHost();
	else output.copyFromDevice(input);
	input.deviceFree();
	mask.deviceFree();
	output.deviceFree();
}
#endif


#ifdef PSKEL_CUDA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runIterativeGPU(size_t iterations, size_t GPUBlockSizeX, size_t GPUBlockSizeY){
	if(GPUBlockSizeX==0){
		int device;
		cudaGetDevice(&device);
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, device);
		//GPUBlockSize = deviceProperties.maxThreadsPerBlock/2;
		GPUBlockSizeX = deviceProperties.warpSize;
		//int minGridSize, blockSize;
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, stencilTilingCU, 0, in.size());
		//GPUBlockSize = blockSize;
		//cout << "GPUBlockSize: "<<GPUBlockSize<<endl;
		//int maxActiveBlocks;
  		//cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, stencilTilingCU, GPUBlockSize, 0);
		//float occupancy = (maxActiveBlocks * GPUBlockSize / deviceProperties.warpSize) / 
                //    (float)(deviceProperties.maxThreadsPerMultiProcessor / 
                //            deviceProperties.warpSize);
		//printf("Launched blocks of size %d. Theoretical occupancy: %f\n", GPUBlockSize, occupancy);
	}
	input.deviceAlloc();
	input.copyToDevice();
	mask.deviceAlloc();
	mask.copyToDevice();
	output.deviceAlloc();
	//output.copyToDevice();
	//this->setGPUInputData();
	for(size_t it = 0; it<iterations; it++){
		if((it%2)==0)
			this->runCUDA(this->input, this->output, GPUBlockSizeX, GPUBlockSizeY);
		else this->runCUDA(this->output, this->input, GPUBlockSizeX, GPUBlockSizeY);
	}
	if((iterations%2)==1)
		output.copyToHost();
	else output.copyFromDevice(input);
	input.deviceFree();
	mask.deviceFree();
	output.deviceFree();
	//this->getGPUOutputData();
}
#endif

template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runIterativePartition(size_t iterations, float gpuFactor, size_t numThreads, size_t GPUBlockSizeX, size_t GPUBlockSizeY){
	numThreads = (numThreads==0)?omp_get_num_procs():numThreads;
	if(GPUBlockSizeX==0){
		int device;
		cudaGetDevice(&device);
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, device);
		//GPUBlockSize = deviceProperties.maxThreadsPerBlock/2;
		GPUBlockSizeX = GPUBlockSizeY = deviceProperties.warpSize;
		//int minGridSize, blockSize;
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, stencilTilingCU, 0, in.size());
		//GPUBlockSize = blockSize;
		//cout << "GPUBlockSize: "<<GPUBlockSize<<endl;
		//int maxActiveBlocks;
  		//cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, stencilTilingCU, GPUBlockSize, 0);
		//float occupancy = (maxActiveBlocks * GPUBlockSize / deviceProperties.warpSize) / 
                //    (float)(deviceProperties.maxThreadsPerMultiProcessor / 
                //            deviceProperties.warpSize);
		//printf("Launched blocks of size %d. Theoretical occupancy: %f\n", GPUBlockSize, occupancy);
	}
	mask.deviceAlloc();
	mask.copyToDevice();

	StencilTiling<Array, Mask> gpuTiling(this->input, this->output, this->mask);
	StencilTiling<Array, Mask> cpuTiling(this->input, this->output, this->mask);
	Array inputGPU;
	Array outputGPU;
	Array tmp;
	Array inputCPU;
	Array outputCPU;
	if(this->input.getHeight()==1){
		
	}
	else{
		size_t gpuHeight = ceil(this->input.getHeight()*gpuFactor);
		size_t cpuHeight = this->input.getHeight()-gpuHeight;
		if(cpuHeight==0) 
			this->runIterativeGPU(iterations, GPUBlockSizeX,GPUBlockSizeY);
		else if(gpuHeight==0) 
			this->runIterativeCPU(iterations, numThreads);
		else{
			#pragma omp parallel sections
			{
				#pragma omp section         
				{//begin GPU
					//printf("%f Running GPU iterations\n",omp_get_wtime());
					gpuTiling.tile(iterations, 0,0,0, this->input.getWidth(), gpuHeight, this->input.getDepth());
					
					inputGPU.hostSlice(gpuTiling.input, gpuTiling.widthOffset, gpuTiling.heightOffset, gpuTiling.depthOffset, gpuTiling.width, gpuTiling.height, gpuTiling.depth);
					
					outputGPU.hostSlice(gpuTiling.output, gpuTiling.widthOffset, gpuTiling.heightOffset, gpuTiling.depthOffset, gpuTiling.width, gpuTiling.height, gpuTiling.depth);

					inputGPU.deviceAlloc();
					inputGPU.copyToDevice();
					
					outputGPU.deviceAlloc();
					tmp.hostAlloc(gpuTiling.width, gpuTiling.height, gpuTiling.depth);
					
					//CUDA kernel execution
					this->runIterativeTilingCUDA(inputGPU, outputGPU, gpuTiling, GPUBlockSizeX, GPUBlockSizeY);
					//printf("%f Finished GPU iterations\n",omp_get_wtime());
				}//end GPU section
				
				#pragma omp section
				{//begin CPU
					//printf("%f Running CPU iterations\n",omp_get_wtime());
					cpuTiling.tile(iterations, 0, gpuHeight, 0, this->input.getWidth(), cpuHeight, this->input.getDepth());
					
					inputCPU.hostSlice(cpuTiling.input, cpuTiling.widthOffset, cpuTiling.heightOffset, cpuTiling.depthOffset, cpuTiling.width, cpuTiling.height, cpuTiling.depth);
					
					outputCPU.hostSlice(cpuTiling.output, cpuTiling.widthOffset, cpuTiling.heightOffset, cpuTiling.depthOffset, cpuTiling.width, cpuTiling.height, cpuTiling.depth);
					//inputCPU.hostSlice(this->input, 0, gpuHeight, 0, this->input.getWidth(), cpuHeight, this->input.getDepth());
					//outputCPU.hostSlice(this->output, 0, gpuHeight, 0, this->input.getWidth(), cpuHeight, this->input.getDepth());

					Array inputCopy;
					inputCopy.hostClone(inputCPU);
					for(size_t it = 0; it<iterations; it++){
						if(it%2==0){
							#ifdef PSKEL_TBB
								this->runTBB(inputCopy, outputCPU, numThreads);
							#else
								this->runOpenMP(inputCopy, outputCPU, numThreads);
							#endif
						}else {
							#ifdef PSKEL_TBB
								this->runTBB(outputCPU, inputCopy, numThreads);
							#else
								this->runOpenMP(outputCPU, inputCopy, numThreads);
							#endif
						}
					}//end for
					if((iterations%2)==0) outputCPU.hostMemCopy(inputCopy);
					inputCopy.hostFree();
					//printf("%f Finished CPU iterations\n",omp_get_wtime());
				}//end CPU section
			}//end parallel omp sections
			if(iterations%2==0)
				tmp.copyFromDevice(inputGPU);
			else
				tmp.copyFromDevice(outputGPU);	
			Array coreTmp;
			Array coreOutput;
			coreTmp.hostSlice(tmp, gpuTiling.coreWidthOffset, gpuTiling.coreHeightOffset, gpuTiling.coreDepthOffset, gpuTiling.coreWidth, gpuTiling.coreHeight, gpuTiling.coreDepth);
			coreOutput.hostSlice(outputGPU, gpuTiling.coreWidthOffset, gpuTiling.coreHeightOffset, gpuTiling.coreDepthOffset, gpuTiling.coreWidth, gpuTiling.coreHeight, gpuTiling.coreDepth);
			coreOutput.hostMemCopy(coreTmp);
			tmp.hostFree();
		}//end if partitioned
	}//end if input.getHeight()
	inputGPU.deviceFree();
	outputGPU.deviceFree();
	mask.deviceFree();
	//cudaDeviceSynchronize();
}

#ifdef PSKEL_CUDA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runIterativeTilingGPU(size_t iterations, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth, size_t innerIterations, size_t GPUBlockSizeX, size_t GPUBlockSizeY){
	if(GPUBlockSizeX==0){
		int device;
		cudaGetDevice(&device);
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, device);
		//GPUBlockSize = deviceProperties.maxThreadsPerBlock/2;
		GPUBlockSizeX = GPUBlockSizeY = deviceProperties.warpSize;
		//int minGridSize, blockSize;
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, stencilTilingCU, 0, in.size());
		//GPUBlockSize = blockSize;
		//cout << "GPUBlockSize: "<<GPUBlockSize<<endl;
		//int maxActiveBlocks;
  		//cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, stencilTilingCU, GPUBlockSize, 0);
		//float occupancy = (maxActiveBlocks * GPUBlockSize / deviceProperties.warpSize) / 
                //    (float)(deviceProperties.maxThreadsPerMultiProcessor / 
                //            deviceProperties.warpSize);
		//printf("Launched blocks of size %d. Theoretical occupancy: %f\n", GPUBlockSize, occupancy);
	}
	Array inputCopy;
	inputCopy.hostClone(this->input);
	size_t wTiling = ceil(float(this->input.getWidth())/float(tilingWidth));
	size_t hTiling = ceil(float(this->input.getHeight())/float(tilingHeight));
	size_t dTiling = ceil(float(this->input.getDepth())/float(tilingDepth));
	mask.deviceAlloc();
	mask.copyToDevice();
	//setGPUMask();
	StencilTiling<Array, Mask> tiling(inputCopy, this->output, this->mask);
	Array inputTile;
	Array outputTile;
	Array tmp;
	size_t outterIterations = ceil(float(iterations)/innerIterations);
	for(size_t it = 0; it<outterIterations; it++){
		size_t subIterations = innerIterations;
		if(((it+1)*innerIterations)>iterations){
			subIterations = iterations-(it*innerIterations);
		}
		//cout << "Iteration: " << it << endl;
		//cout << "#SubIterations: " << subIterations << endl;
		for(size_t ht=0; ht<hTiling; ht++){
		 for(size_t wt=0; wt<wTiling; wt++){
		  for(size_t dt=0; dt<dTiling; dt++){
			size_t heightOffset = ht*tilingHeight;
			size_t widthOffset = wt*tilingWidth;
			size_t depthOffset = dt*tilingDepth;

			//CUDA input memory copy
			tiling.tile(subIterations, widthOffset, heightOffset, depthOffset, tilingWidth, tilingHeight, tilingDepth);
			inputTile.hostSlice(tiling.input, tiling.widthOffset, tiling.heightOffset, tiling.depthOffset, tiling.width, tiling.height, tiling.depth);
			outputTile.hostSlice(tiling.output, tiling.widthOffset, tiling.heightOffset, tiling.depthOffset, tiling.width, tiling.height, tiling.depth);
			inputTile.deviceAlloc();
			outputTile.deviceAlloc();
			tmp.hostAlloc(tiling.width, tiling.height, tiling.depth);
			//this->setGPUInputDataIterative(inputCopy, output, innerIterations, widthOffset, heightOffset, depthOffset, tilingWidth, tilingHeight, tilingDepth);
			if(it%2==0){
				inputTile.copyToDevice();
				//CUDA kernel execution
				this->runIterativeTilingCUDA(inputTile, outputTile, tiling, GPUBlockSizeX, GPUBlockSizeY);
				if(subIterations%2==0){
					tmp.copyFromDevice(inputTile);
				}else{
					tmp.copyFromDevice(outputTile);
				}
				Array coreTmp;
				Array coreOutput;
				coreTmp.hostSlice(tmp, tiling.coreWidthOffset, tiling.coreHeightOffset, tiling.coreDepthOffset, tiling.coreWidth, tiling.coreHeight, tiling.coreDepth);
				coreOutput.hostSlice(outputTile, tiling.coreWidthOffset, tiling.coreHeightOffset, tiling.coreDepthOffset, tiling.coreWidth, tiling.coreHeight, tiling.coreDepth);
				coreOutput.hostMemCopy(coreTmp);
				//this->copyTilingOutput(output, innerIterations, widthOffset, heightOffset, depthOffset, tilingWidth, tilingHeight, tilingDepth);
				tmp.hostFree();
			}else{
				outputTile.copyToDevice();
				//CUDA kernel execution
				this->runIterativeTilingCUDA(outputTile, inputTile, tiling, GPUBlockSizeX, GPUBlockSizeY);
				if(subIterations%2==0){
					tmp.copyFromDevice(outputTile);
				}else{
					tmp.copyFromDevice(inputTile);
				}
				Array coreTmp;
				Array coreInput;
				//cout << "[Computing iteration: " << it << "]" << endl;
				coreTmp.hostSlice(tmp, tiling.coreWidthOffset, tiling.coreHeightOffset, tiling.coreDepthOffset, tiling.coreWidth, tiling.coreHeight, tiling.coreDepth);
				coreInput.hostSlice(inputTile, tiling.coreWidthOffset, tiling.coreHeightOffset, tiling.coreDepthOffset, tiling.coreWidth, tiling.coreHeight, tiling.coreDepth);
				coreInput.hostMemCopy(coreTmp);
				tmp.hostFree();
			}
		}}}
	}
	inputTile.deviceFree();
	outputTile.deviceFree();
	mask.deviceFree();
	cudaDeviceSynchronize();

	if((outterIterations%2)==0) tiling.output.hostMemCopy(tiling.input);
	inputCopy.hostFree();
}
#endif

#ifdef PSKEL_CUDA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runCUDA(Array in, Array out, size_t GPUBlockSizeX, size_t GPUBlockSizeY){
	dim3 DimBlock(GPUBlockSizeX, GPUBlockSizeY, 1);
	dim3 DimGrid((in.getWidth() - 1)/GPUBlockSizeX + 1, (in.getHeight() - 1)/GPUBlockSizeY + 1, in.getDepth());
    size_t maskRange = mask.getRange();

	#ifdef PSKEL_SHARED
		size_t sharedSize = sizeof(float)*(GPUBlockSizeX+2*maskRange)*(GPUBlockSizeY+2*maskRange);
        stencilTilingCU<<<DimGrid, DimBlock,sharedSize>>>(in, out, this->mask, this->args, maskRange, in.getWidth(),in.getHeight(),in.getDepth());
	#else
        //stencilTilingCU<<<DimGrid, DimBlock>>>(in, out, this->mask, this->args, 0,0,0,in.getWidth(),in.getHeight(),in.getDepth());
        stencilTilingCU<<<DimGrid, DimBlock>>>(in, out, this->mask, this->args,maskRange,in.getWidth(),in.getHeight(),in.getDepth());
	#endif
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}
#endif

#ifdef PSKEL_CUDA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runIterativeTilingCUDA(Array in, Array out, StencilTiling<Array, Mask> tiling, size_t GPUBlockSizeX, size_t GPUBlockSizeY){
	dim3 DimBlock(GPUBlockSizeX,GPUBlockSizeY, 1);
	dim3 DimGrid((in.getWidth() - 1)/GPUBlockSizeX + 1, (in.getHeight() - 1)/GPUBlockSizeY + 1, 1);
	//dim3 DimBlock(GPUBlockSize,GPUBlockSize, 1);
	//dim3 DimGrid((in.getWidth() - 1)/GPUBlockSize + 1, (in.getHeight() - 1)/GPUBlockSize + 1, 1);
	size_t maskRange = this->mask.getRange();
	for(size_t it=0; it<tiling.iterations; it++){
		//cout << "[Computing iteration: " << it << "]" << endl;
		//cout << "mask range: " <<maskRange << endl;
		//cout << "mask margin: " <<(maskRange*(tiling.iterations-(it+1))) << endl;
		size_t margin = (maskRange*(tiling.iterations-(it+1)));
		size_t widthOffset = 0;
		size_t extra = 0;
		if(tiling.coreWidthOffset>margin){
			widthOffset = tiling.coreWidthOffset-margin;
		}else extra = margin-widthOffset;
		//cout << "width extra: " << extra << endl;
		size_t width = tiling.coreWidth+margin*2 - extra;
		if((widthOffset+width)>=tiling.width){
			width = tiling.width-widthOffset;
		}
		size_t heightOffset = 0;
		extra = 0;
		if(tiling.coreHeightOffset>margin){
			heightOffset = tiling.coreHeightOffset-margin;
		}else extra = margin-heightOffset;
		//cout << "height extra: " << extra << endl;
		size_t height = tiling.coreHeight+margin*2-extra;
		if((heightOffset+height)>=tiling.height){
			height = tiling.height-heightOffset;
		}
		size_t depthOffset = 0;
		extra = 0;
		if(tiling.coreDepthOffset>margin){
			depthOffset = tiling.coreDepthOffset-margin;
		}else extra = margin-depthOffset;
		//cout << "depth extra: " << extra << endl;
		size_t depth = tiling.coreDepth+margin*2-extra;
		if((depthOffset+depth)>=tiling.depth){
			depth = tiling.depth-depthOffset;
		}
		
		//cout << "width-offset: " <<widthOffset << endl;
		//cout << "height-offset: " <<heightOffset << endl;
		//cout << "depth-offset: " <<depthOffset << endl;
		
		//cout << "width: " <<width << endl;
		//cout << "height: " <<height << endl;
		//cout << "depth: " <<depth << endl;
		if(it%2==0){
			#ifdef PSKEL_SHARED_MASK
			stencilTilingCU<<<DimGrid, DimBlock, (this->mask.size*this->mask.dimension)>>>(in, out, this->mask, this->args, widthOffset, heightOffset, depthOffset, width, height, depth);
			#else
			stencilTilingCU<<<DimGrid, DimBlock>>>(in, out, this->mask, this->args, maskRange, widthOffset, heightOffset, depthOffset, width, height, depth);
			#endif
		}else{
			#ifdef PSKEL_SHARED_MASK
			stencilTilingCU<<<DimGrid, DimBlock, (this->mask.size*this->mask.dimension)>>>(out, in, this->mask, this->args, widthOffset, heightOffset, depthOffset, width, height, depth);
			#else
			stencilTilingCU<<<DimGrid, DimBlock>>>(out, in, this->mask, this->args, maskRange, widthOffset, heightOffset, depthOffset, width, height, depth);
			#endif
		}
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
	}
}
#endif

#ifdef PSKEL_GA
struct TilingGPUGeneticEvaluationFunction{
    size_t iterations;
    size_t height;
    size_t width;
    size_t depth;
    size_t range;
    size_t typeSize;
    size_t memFree;
    size_t popsize;
    size_t ngen;
    size_t dw;
    size_t dt;
    size_t dh;
    float score;
};
TilingGPUGeneticEvaluationFunction tilingGPUEvaluator;

float objective2D(GAGenome &c){
	GABin2DecGenome &genome = (GABin2DecGenome &)c;
	
	float h = genome.phenotype(0);
	float it = genome.phenotype(1);
	size_t tileHeight = ((tilingGPUEvaluator.height<=(2*it*tilingGPUEvaluator.range + h))?tilingGPUEvaluator.height:(2*it*tilingGPUEvaluator.range + h));
 
	if(2*(tilingGPUEvaluator.width*tilingGPUEvaluator.depth*tileHeight*tilingGPUEvaluator.typeSize) > tilingGPUEvaluator.memFree)return 0;
	else {
		float val = h/tileHeight;
		return val*((it*h)/(tilingGPUEvaluator.height*tilingGPUEvaluator.iterations));
	}
}

void solve2D(unsigned int seed){
	int popsize = tilingGPUEvaluator.popsize;
	int ngen = tilingGPUEvaluator.ngen;
	float pmut = 0.01;
	float pcross = 0.6;
	
	float div = (2.0*(tilingGPUEvaluator.width*tilingGPUEvaluator.depth*tilingGPUEvaluator.height*tilingGPUEvaluator.typeSize))/(tilingGPUEvaluator.memFree*1.1);
	size_t maxHeight = ceil(float(tilingGPUEvaluator.height)/div);
	//Create a phenotype for two variables.  The number of bits you can use to
	//represent any number is limited by the type of computer you are using.  In
	//this case, we use 16 bits to represent a floating point number whose value
	//can range from -5 to 5, inclusive.  The bounds on x1 and x2 can be applied
	//here and/or in the objective function.
	GABin2DecPhenotype map;
	map.add(16, 1, maxHeight); //min/max boundaries, inclusive
	map.add(16, 1, tilingGPUEvaluator.iterations);

	//Create the template genome using the phenotype map we just made.
	GABin2DecGenome genome(map, objective2D);

	//Now create the GA using the genome and run it.  We'll use sigma truncation
	//scaling so that we can handle negative objective scores.
	GASimpleGA ga(genome);
	GASigmaTruncationScaling scaling;
	ga.populationSize(popsize);
	ga.nGenerations(ngen);
	ga.pMutation(pmut);
	ga.pCrossover(pcross);
	ga.scaling(scaling);
	ga.scoreFrequency(0);
	ga.flushFrequency(0); //stop flushing the record of the score of given generations
	//ga.scoreFilename(0); //stop recording the score of given generations
	ga.evolve(seed);

	//Obtains the best individual from the best population evolved
	genome = ga.statistics().bestIndividual();

	//cout << "the ga found an optimum at the point (";
	//cout << genome.phenotype(0) << ", " << genome.phenotype(1) << ")\n\n";
	//cout << "best of generation data are in '" << ga.scoreFilename() << "'\n";
	tilingGPUEvaluator.dw = tilingGPUEvaluator.width;
	tilingGPUEvaluator.dh = genome.phenotype(0);//height;
	tilingGPUEvaluator.dt = genome.phenotype(1);//subIterations;
	tilingGPUEvaluator.score = objective2D(genome);
}

float objective3D(GAGenome &c){
	GABin2DecGenome &genome = (GABin2DecGenome &)c;
	
	float w = genome.phenotype(0);
	float h = genome.phenotype(1);
	float t = genome.phenotype(2);
	float tileWidth = ((tilingGPUEvaluator.width<=(2*t*tilingGPUEvaluator.range + w))?tilingGPUEvaluator.width:(2*t*tilingGPUEvaluator.range + w));
	float tileHeight = ((tilingGPUEvaluator.height<=(2*t*tilingGPUEvaluator.range + h))?tilingGPUEvaluator.height:(2*t*tilingGPUEvaluator.range + h));
 
	if(2*(tileWidth*tileHeight*tilingGPUEvaluator.depth*tilingGPUEvaluator.typeSize) > tilingGPUEvaluator.memFree) return 0;
	else {
		float val = (w*h)/(tileWidth*tileHeight);
		return val*((w*h*t)/(tilingGPUEvaluator.width*tilingGPUEvaluator.height*tilingGPUEvaluator.iterations));
	}
}

void solve3D(unsigned int seed){
	int popsize = tilingGPUEvaluator.popsize;
	int ngen = tilingGPUEvaluator.ngen;
	float pmut = 0.01;
	float pcross = 0.6;
	
	//float div = (2.0*(tilingGPUEvaluator.width*tilingGPUEvaluator.depth*tilingGPUEvaluator.height*tilingGPUEvaluator.typeSize))/(tilingGPUEvaluator.memFree*1.1);
	//size_t maxHeight = ceil(float(tilingGPUEvaluator.height)/div);
	//Create a phenotype for two variables.  The number of bits you can use to
	//represent any number is limited by the type of computer you are using.  In
	//this case, we use 16 bits to represent a floating point number whose value
	//can range from -5 to 5, inclusive.  The bounds on x1 and x2 can be applied
	//here and/or in the objective function.
	GABin2DecPhenotype map;
	//map.add(16, 1, maxHeight); //min/max boundaries, inclusive
	map.add(16, 1, tilingGPUEvaluator.width);
	map.add(16, 1, tilingGPUEvaluator.height);
	map.add(16, 1, tilingGPUEvaluator.iterations);

	//Create the template genome using the phenotype map we just made.
	GABin2DecGenome genome(map, objective3D);

	//Now create the GA using the genome and run it.  We'll use sigma truncation
	//scaling so that we can handle negative objective scores.
	GASimpleGA ga(genome);
	GASigmaTruncationScaling scaling;
	ga.populationSize(popsize);
	ga.nGenerations(ngen);
	ga.pMutation(pmut);
	ga.pCrossover(pcross);
	ga.scaling(scaling);
	ga.scoreFrequency(0);
	ga.flushFrequency(0); //stop flushing the record of the score of given generations
	//ga.scoreFilename(0); //stop recording the score of given generations
	ga.evolve(seed);

	//Obtains the best individual from the best population evolved
	genome = ga.statistics().bestIndividual();

	//cout << "the ga found an optimum at the point (";
	//cout << genome.phenotype(0) << ", " << genome.phenotype(1) << ")\n\n";
	//cout << "best of generation data are in '" << ga.scoreFilename() << "'\n";
	tilingGPUEvaluator.dw = genome.phenotype(0);//width;
	tilingGPUEvaluator.dh = genome.phenotype(1);//height;
	tilingGPUEvaluator.dt = genome.phenotype(2);//subIterations;
	tilingGPUEvaluator.score = objective3D(genome);
}

template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runIterativeAutoGPU(size_t iterations, size_t GPUBlockSize){
	size_t gpuMemFree, gpuMemTotal;
	//gpuErrchk( cudaDeviceSynchronize() );
	cudaMemGetInfo(&gpuMemFree, &gpuMemTotal);
	if((this->input.memSize()+this->output.memSize()+this->mask.memSize())<(0.999*gpuMemFree)){
		runIterativeGPU(iterations, GPUBlockSize);
	}else if(this->input.getHeight()==1){
		//solving for a 'transposed matrix'
		tilingGPUEvaluator.typeSize = this->input.memSize()/this->input.size();
		tilingGPUEvaluator.iterations = iterations;
		tilingGPUEvaluator.width = this->input.getDepth(); //'transposed matrix'
		tilingGPUEvaluator.height = this->input.getWidth(); //'transposed matrix'
		tilingGPUEvaluator.depth = 1;
		tilingGPUEvaluator.range = this->mask.getRange();
		tilingGPUEvaluator.memFree = (gpuMemFree-this->mask.memSize())*0.999;//gpuMemFree*0.998;

		tilingGPUEvaluator.popsize = 100;
		tilingGPUEvaluator.ngen = 2500;

  		unsigned int seed = time(NULL);
		solve2D(seed);

		size_t subIterations = tilingGPUEvaluator.dt;
		size_t width = tilingGPUEvaluator.dh;
		//cout << "GPU Mem Total: "<< gpuMemTotal <<endl;
		//cout << "GPU Mem Free: "<< gpuMemFree <<endl;
		//cout << "sub iterations: "<< subIterations <<endl;
		//cout << "tiling height: "<<height<<endl;
		runIterativeTilingGPU(iterations, width, 1, this->input.getDepth(), subIterations, GPUBlockSize);
		
	}else {
		size_t typeSize = this->input.memSize()/this->input.size();
		tilingGPUEvaluator.typeSize = typeSize;
		tilingGPUEvaluator.iterations = iterations;
		tilingGPUEvaluator.width = this->input.getWidth();
		tilingGPUEvaluator.height = this->input.getHeight();
		tilingGPUEvaluator.depth = this->input.getDepth();
		tilingGPUEvaluator.range = this->mask.getRange();
		tilingGPUEvaluator.memFree = (gpuMemFree-this->mask.memSize())*0.999;//gpuMemFree*0.998;
		if( (2*(1+2*this->mask.getRange())*(this->input.getWidth()*this->input.getDepth())*typeSize+this->mask.memSize()) > (0.98*gpuMemFree) ){
			tilingGPUEvaluator.popsize = 100;
			tilingGPUEvaluator.ngen = 2500;
	  		unsigned int seed = time(NULL);
			solve3D(seed);

			size_t width = tilingGPUEvaluator.dw;
			size_t height = tilingGPUEvaluator.dh;
			size_t subIterations = tilingGPUEvaluator.dt;
			//cout << "GPU Mem Total: "<< gpuMemTotal <<endl;
			//cout << "GPU Mem Free: "<< gpuMemFree <<endl;
			//cout << "sub iterations: "<< subIterations <<endl;
			//cout << "tiling height: "<<height<<endl;
			runIterativeTilingGPU(iterations, width, height, this->input.getDepth(), subIterations, GPUBlockSize);
		}else{
			tilingGPUEvaluator.popsize = 100;
			tilingGPUEvaluator.ngen = 2500;
	  		unsigned int seed = time(NULL);
			solve2D(seed);

			size_t subIterations = tilingGPUEvaluator.dt;
			size_t height = tilingGPUEvaluator.dh;
			//cout << "GPU Mem Total: "<< gpuMemTotal <<endl;
			//cout << "GPU Mem Free: "<< gpuMemFree <<endl;
			//cout << "sub iterations: "<< subIterations <<endl;
			//cout << "tiling height: "<<height<<endl;
			runIterativeTilingGPU(iterations, this->input.getWidth(), height, this->input.getDepth(), subIterations, GPUBlockSize);
		}
	}
}
#endif

//*******************************************************************************************
// Stencil 3D
//*******************************************************************************************


template<class Array, class Mask, class Args>
Stencil3D<Array,Mask,Args>::Stencil3D(){}
	
template<class Array, class Mask, class Args>
Stencil3D<Array,Mask,Args>::Stencil3D(Array _input, Array _output, Mask _mask, Args _args){
	this->input = _input;
	this->output = _output;
	this->args = _args;
	this->mask = _mask;
}

template<class Array, class Mask, class Args>
void Stencil3D<Array,Mask,Args>::runSeq(Array in, Array out){
	for (int h = 0; h < in.getHeight(); h++){
	for (int w = 0; w < in.getWidth(); w++){
	for (int d = 0; d < in.getDepth(); d++){
		stencilKernel(in,out,this->mask,this->args,h,w,d);
	}}}
}

template<class Array, class Mask, class Args>
void Stencil3D<Array,Mask,Args>::runOpenMP(Array in, Array out, size_t numThreads){
	omp_set_num_threads(numThreads);
	#pragma omp parallel for
	for (int h = 0; h < in.getHeight(); h++){
	for (int w = 0; w < in.getWidth(); w++){
	for (int d = 0; d < in.getDepth(); d++){
		stencilKernel(in,out,this->mask,this->args,h,w,d);
	}}}
}

#ifdef PSKEL_TBB
template<class Array, class Mask, class Args>
struct TBBStencil3D{
	Array input;
	Array output;
	Mask mask;
	Args args;
	TBBStencil3D(Array input, Array output, Mask mask, Args args){
		this->input = input;
		this->output = output;
		this->mask = mask;
		this->args = args;
	}
	void operator()(tbb::blocked_range<int> r)const{
		for (int h = r.begin(); h != r.end(); h++){
		for (int w = 0; w < this->input.getWidth(); w++){
		for (int d = 0; d < this->input.getDepth(); d++){
			stencilKernel(this->input,this->output,this->mask,this->args,h,w,d);
		}}}
	}
};

template<class Array, class Mask, class Args>
void Stencil3D<Array,Mask,Args>::runTBB(Array in, Array out, size_t numThreads){
	TBBStencil3D<Array, Mask, Args> tbbstencil(in, out, this->mask, this->args);
	tbb::task_scheduler_init init(numThreads);
	tbb::parallel_for(tbb::blocked_range<int>(0, in.getHeight()), tbbstencil);
}
#endif

//*******************************************************************************************
// Stencil 2D
//*******************************************************************************************

template<class Array, class Mask, class Args>
Stencil2D<Array,Mask,Args>::Stencil2D(){}

template<class Array, class Mask, class Args>
Stencil2D<Array,Mask,Args>::Stencil2D(Array _input, Array _output, Mask _mask, Args _args){
	this->input = _input;
	this->output = _output;
	this->args = _args;
	this->mask = _mask;
}
/*
template<class Array, class Mask, class Args>
Stencil2D<Array,Mask,Args>::~Stencil2D(){
	this->cudaMemFree();
	this->cpuMemFree();
}
*/

template<class Array, class Mask, class Args>
void Stencil2D<Array,Mask,Args>::runSeq(Array in, Array out){
	size_t height = in.getHeight();
	size_t width = in.getWidth();
	for (size_t h = 0; h < height; h++){
	for (size_t w = 0; w < width; w++){
		stencilKernel(in,out,this->mask, this->args,h,w);
	}}
}

template<class Array, class Mask, class Args>
void Stencil2D<Array,Mask,Args>::runOpenMP(Array in, Array out, size_t numThreads){
	omp_set_num_threads(numThreads);
	size_t height = in.getHeight();
	size_t width = in.getWidth();
    	size_t maskRange = this->mask.getRange();
	#ifdef CACHE_BLOCK
	const size_t TH = 16;
	const size_t TW = 16;
	#pragma omp parallel for
	for(size_t hh = maskRange; hh < height-maskRange;hh+=TH){
	for(size_t ww = maskRange; ww < width-maskRange; ww+=TW){
		for(size_t h = hh; h < MIN(hh+TH,height-maskRange);h++){
		for(size_t w = ww; w < MIN(ww+TW,width-maskRange);w++){
			stencilKernel(in,out,this->mask, this->args,h,w);
		}}
	}}	
	#else
	#pragma omp parallel for
	for (size_t h = maskRange; h < height-maskRange; h++){
	for (size_t w = maskRange; w < width-maskRange; w++){
		stencilKernel(in,out,this->mask, this->args,h,w);
	}}
	#endif
}

#ifdef PSKEL_TBB
template<class Array, class Mask, class Args>
struct TBBStencil2D{
	Array input;
	Array output;
	Mask mask;
	Args args;
	size_t maskRange;
	size_t width;
	TBBStencil2D(Array input, Array output, Mask mask, Args args){
		this->input = input;
		this->output = output;
		this->mask = mask;
		this->args = args;
		this->maskRange = mask.getRange();
		this->width = input.getWidth();
	}
	void operator()(tbb::blocked_range<int> r)const{
		for (int h = r.begin(); h != r.end(); h++){
		for (int w = maskRange; w < this->width-maskRange; w++){
			stencilKernel(this->input,this->output,this->mask, this->args,h,w);
		}}
	}
};

template<class Array, class Mask, class Args>
void Stencil2D<Array,Mask,Args>::runTBB(Array in, Array out, size_t numThreads){
	TBBStencil2D<Array, Mask, Args> tbbstencil(in, out, this->mask, this->args);
	tbb::task_scheduler_init init(numThreads);
	size_t maskRange = this->mask.getRange();
	tbb::parallel_for(tbb::blocked_range<int>(maskRange, in.getHeight()-maskRange), tbbstencil);
}
#endif

//*******************************************************************************************
// Stencil 1D
//*******************************************************************************************


template<class Array, class Mask, class Args>
Stencil<Array,Mask,Args>::Stencil(){}
	
template<class Array, class Mask, class Args>
Stencil<Array,Mask,Args>::Stencil(Array _input, Array _output, Mask _mask, Args _args){
	this->input = _input;
	this->output = _output;
	this->args = _args;
	this->mask = _mask;
}

template<class Array, class Mask, class Args>
void Stencil<Array,Mask,Args>::runSeq(Array in, Array out){
	for (int i = 0; i < in.getWidth(); i++){
		stencilKernel(in,out,this->mask, this->args,i);
	}
}

template<class Array, class Mask, class Args>
void Stencil<Array,Mask,Args>::runOpenMP(Array in, Array out, size_t numThreads){
	omp_set_num_threads(numThreads);
	#pragma omp parallel for
	for (int i = 0; i < in.getWidth(); i++){
		stencilKernel(in,out,this->mask, this->args,i);
	}
}

#ifdef PSKEL_TBB
template<class Array, class Mask, class Args>
struct TBBStencil{
	Array input;
	Array output;
	Mask mask;
	Args args;
	TBBStencil(Array input, Array output, Mask mask, Args args){
		this->input = input;
		this->output = output;
		this->mask = mask;
		this->args = args;
	}
	void operator()(tbb::blocked_range<int> r)const{
		for (int i = r.begin(); i != r.end(); i++){
			stencilKernel(this->input,this->output,this->mask, this->args,i);
		}
	}
};

template<class Array, class Mask, class Args>
void Stencil<Array,Mask,Args>::runTBB(Array in, Array out, size_t numThreads){
	TBBStencil<Array, Mask, Args> tbbstencil(in, out, this->mask, this->args);
	tbb::task_scheduler_init init(numThreads);
	tbb::parallel_for(tbb::blocked_range<int>(0, in.getWidth()), tbbstencil);
}
#endif

}//end namespace


#endif
