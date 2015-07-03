#include <omp.h>
#include <fstream>
#include <string>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <cassert>

#include <omp.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>

#include <gtest/gtest.h>

#include "../include/PSkelArray.h"
#include "../include/PSkelMask.h"
#include "../include/PSkelStencil.h"

#include "conf.h"

using namespace std;
using namespace PSkel;

namespace PSkel{
	__parallel__ void stencilKernel(Array<int> input,Array<int> output,Mask<int> mask,int null,size_t i){
		int sum=input(i);
		for(int z=0;z<mask.size;z++){
			sum += mask.get(z,input,i);
		}
		output(i) = sum;
	}

	__parallel__ void stencilKernel(Array2D<int> input,Array2D<int> output,Mask2D<int> mask,int null, size_t h, size_t w){
		int sum=input(h,w);
		for(int z=0;z<mask.size;z++){
			sum += mask.get(z,input,h,w);
		}
		output(h,w) = sum;
	}
}

TEST(Stencil, GPU){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runGPU(GPUBlockSize);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i){
		int val = inputGrid(i);
		if(i>0) val += inputGrid(i-1);
		if(i<(inputGrid.size()-1)) val += inputGrid(i+1);
		ASSERT_EQ(outputGrid(i), val);
	}

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, GPUAlignedTiling){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runTilingGPU(TESTS_ALIGNED_TILING_WIDTH, 1, 1, GPUBlockSize);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i){
		int val = inputGrid(i);
		if(i>0) val += inputGrid(i-1);
		if(i<(inputGrid.size()-1)) val += inputGrid(i+1);
		ASSERT_EQ(outputGrid(i), val);
	}

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, GPUUnalignedTiling){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runTilingGPU(TESTS_UNALIGNED_TILING_WIDTH, 1, 1, GPUBlockSize);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i){
		int val = inputGrid(i);
		if(i>0) val += inputGrid(i-1);
		if(i<(inputGrid.size()-1)) val += inputGrid(i+1);
		ASSERT_EQ(outputGrid(i), val);
	}

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, GPUEvenIterations){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runIterativeGPU(iterations, GPUBlockSize);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i){
		int val = inputGrid(i)*iterations;
		if(i>0) val += inputGrid(i-1)*iterations;
		if(i<(inputGrid.size()-1)) val += inputGrid(i+1)*iterations;
		ASSERT_GT(outputGrid(i), val);
	}

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, GPUOddIterations){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int iterations = TESTS_ODD_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runIterativeGPU(iterations, GPUBlockSize);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i){
		int val = inputGrid(i)*iterations;
		if(i>0) val += inputGrid(i-1)*iterations;
		if(i<(inputGrid.size()-1)) val += inputGrid(i+1)*iterations;
		ASSERT_GT(outputGrid(i), val);
	}

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, GPUSlice){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	int offset = n/2 - n/4;

	Array<int> inputSlice;
	Array<int> outputSlice;
	inputSlice.hostSlice(inputGrid, offset,0,0,n/2,1,1);
	outputSlice.hostSlice(outputGrid, offset,0,0,n/2,1,1);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputSlice, outputSlice, mask, 0);
	stencil.runGPU(GPUBlockSize);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputSlice.size(); ++i){
		int val = inputSlice(i);
		if(i>0) val += inputSlice(i-1);
		if(i<(inputSlice.size()-1)) val += inputSlice(i+1);
		ASSERT_EQ(outputSlice(i), val);
	}
	for(int i = 0; i<outputGrid.size(); ++i)
		if( i>=offset && i<(offset+n/2) ){
			int val = inputGrid(i);
			if(i>offset) val += inputGrid(i-1);
			if(i<(offset+n/2-1)) val += inputGrid(i+1);
			ASSERT_EQ(outputGrid(i), val);
		}else ASSERT_EQ(outputGrid(i), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, GPUSliceEvenIterations){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	int offset = n/2 - n/4;

	Array<int> inputSlice;
	Array<int> outputSlice;
	inputSlice.hostSlice(inputGrid, offset,0,0,n/2,1,1);
	outputSlice.hostSlice(outputGrid, offset,0,0,n/2,1,1);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputSlice, outputSlice, mask, 0);
	stencil.runIterativeGPU(iterations,GPUBlockSize);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputSlice.size(); ++i){
		int val = inputSlice(i);
		if(i>0) val += inputSlice(i-1);
		if(i<(inputSlice.size()-1)) val += inputSlice(i+1);
		ASSERT_GT(outputSlice(i), val*iterations);
	}
	for(int i = 0; i<outputGrid.size(); ++i)
		if( i>=offset && i<(offset+n/2) ){
			int val = inputGrid(i);
			if(i>offset) val += inputGrid(i-1);
			if(i<(offset+n/2-1)) val += inputGrid(i+1);
			ASSERT_GT(outputGrid(i), val*iterations);
		}else ASSERT_EQ(outputGrid(i), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, GPUSliceOddIterations){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int iterations = TESTS_ODD_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	int offset = n/2 - n/4;

	Array<int> inputSlice;
	Array<int> outputSlice;
	inputSlice.hostSlice(inputGrid, offset,0,0,n/2,1,1);
	outputSlice.hostSlice(outputGrid, offset,0,0,n/2,1,1);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputSlice, outputSlice, mask, 0);
	stencil.runIterativeGPU(iterations,GPUBlockSize);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputSlice.size(); ++i){
		int val = inputSlice(i);
		if(i>0) val += inputSlice(i-1);
		if(i<(inputSlice.size()-1)) val += inputSlice(i+1);
		ASSERT_GT(outputSlice(i), val*iterations);
	}
	for(int i = 0; i<outputGrid.size(); ++i)
		if( i>=offset && i<(offset+n/2) ){
			int val = inputGrid(i);
			if(i>offset) val += inputGrid(i-1);
			if(i<(offset+n/2-1)) val += inputGrid(i+1);
			ASSERT_GT(outputGrid(i), val*iterations);
		}else ASSERT_EQ(outputGrid(i), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, TBB){
	int n = TESTS_WIDTH;
	int TBBNumThreads = TESTS_TBBNumThreads;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runCPU(TBBNumThreads);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i){
		int val = inputGrid(i);
		if(i>0) val += inputGrid(i-1);
		if(i<(inputGrid.size()-1)) val += inputGrid(i+1);
		ASSERT_EQ(outputGrid(i), val);
	}

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, TBBEvenIterations){
	int n = TESTS_WIDTH;
	int TBBNumThreads = TESTS_TBBNumThreads;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runIterativeCPU(iterations, TBBNumThreads);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i){
		int val = inputGrid(i)*iterations;
		if(i>0) val += inputGrid(i-1)*iterations;
		if(i<(inputGrid.size()-1)) val += inputGrid(i+1)*iterations;
		ASSERT_GT(outputGrid(i), val);
	}

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, TBBOddIterations){
	int n = TESTS_WIDTH;
	int TBBNumThreads = TESTS_TBBNumThreads;
	int iterations = TESTS_ODD_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runIterativeCPU(iterations, TBBNumThreads);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i){
		int val = inputGrid(i)*iterations;
		if(i>0) val += inputGrid(i-1)*iterations;
		if(i<(inputGrid.size()-1)) val += inputGrid(i+1)*iterations;
		ASSERT_GT(outputGrid(i), val);
	}

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, TBBSlice){
	int n = TESTS_WIDTH;
	int TBBNumThreads = TESTS_TBBNumThreads;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	int offset = n/2 - n/4;

	Array<int> inputSlice;
	Array<int> outputSlice;
	inputSlice.hostSlice(inputGrid, offset,0,0,n/2,1,1);
	outputSlice.hostSlice(outputGrid, offset,0,0,n/2,1,1);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputSlice, outputSlice, mask, 0);
	stencil.runCPU(TBBNumThreads);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputSlice.size(); ++i){
		int val = inputSlice(i);
		if(i>0) val += inputSlice(i-1);
		if(i<(inputSlice.size()-1)) val += inputSlice(i+1);
		ASSERT_EQ(outputSlice(i), val);
	}
	for(int i = 0; i<outputGrid.size(); ++i)
		if( i>=offset && i<(offset+n/2) ){
			int val = inputGrid(i);
			if(i>offset) val += inputGrid(i-1);
			if(i<(offset+n/2-1)) val += inputGrid(i+1);
			ASSERT_EQ(outputGrid(i), val);
		}else ASSERT_EQ(outputGrid(i), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, TBBSliceEvenIterations){
	int n = TESTS_WIDTH;
	int TBBNumThreads = TESTS_TBBNumThreads;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	int offset = n/2 - n/4;

	Array<int> inputSlice;
	Array<int> outputSlice;
	inputSlice.hostSlice(inputGrid, offset,0,0,n/2,1,1);
	outputSlice.hostSlice(outputGrid, offset,0,0,n/2,1,1);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputSlice, outputSlice, mask, 0);
	stencil.runIterativeCPU(iterations,TBBNumThreads);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputSlice.size(); ++i){
		int val = inputSlice(i);
		if(i>0) val += inputSlice(i-1);
		if(i<(inputSlice.size()-1)) val += inputSlice(i+1);
		ASSERT_GT(outputSlice(i), val*iterations);
	}
	for(int i = 0; i<outputGrid.size(); ++i)
		if( i>=offset && i<(offset+n/2) ){
			int val = inputGrid(i);
			if(i>offset) val += inputGrid(i-1);
			if(i<(offset+n/2-1)) val += inputGrid(i+1);
			ASSERT_GT(outputGrid(i), val*iterations);
		}else ASSERT_EQ(outputGrid(i), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, TBBSliceOddIterations){
	int n = TESTS_WIDTH;
	int TBBNumThreads = TESTS_TBBNumThreads;
	int iterations = TESTS_ODD_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	int offset = n/2 - n/4;

	Array<int> inputSlice;
	Array<int> outputSlice;
	inputSlice.hostSlice(inputGrid, offset,0,0,n/2,1,1);
	outputSlice.hostSlice(outputGrid, offset,0,0,n/2,1,1);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputSlice, outputSlice, mask, 0);
	stencil.runIterativeCPU(iterations,TBBNumThreads);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputSlice.size(); ++i){
		int val = inputSlice(i);
		if(i>0) val += inputSlice(i-1);
		if(i<(inputSlice.size()-1)) val += inputSlice(i+1);
		ASSERT_GT(outputSlice(i), val*iterations);
	}
	for(int i = 0; i<outputGrid.size(); ++i)
		if( i>=offset && i<(offset+n/2) ){
			int val = inputGrid(i);
			if(i>offset) val += inputGrid(i-1);
			if(i<(offset+n/2-1)) val += inputGrid(i+1);
			ASSERT_GT(outputGrid(i), val*iterations);
		}else ASSERT_EQ(outputGrid(i), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, Seq){
	int n = TESTS_WIDTH;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runSequential();
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i){
		int val = inputGrid(i);
		if(i>0) val += inputGrid(i-1);
		if(i<(inputGrid.size()-1)) val += inputGrid(i+1);
		ASSERT_EQ(outputGrid(i), val);
	}

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, SeqEvenIterations){
	int n = TESTS_WIDTH;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runIterativeSequential(iterations);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i){
		int val = inputGrid(i)*iterations;
		if(i>0) val += inputGrid(i-1)*iterations;
		if(i<(inputGrid.size()-1)) val += inputGrid(i+1)*iterations;
		ASSERT_GT(outputGrid(i), val);
	}

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, SeqOddIterations){
	int n = TESTS_WIDTH;
	int iterations = TESTS_ODD_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runIterativeSequential(iterations);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i){
		int val = inputGrid(i)*iterations;
		if(i>0) val += inputGrid(i-1)*iterations;
		if(i<(inputGrid.size()-1)) val += inputGrid(i+1)*iterations;
		ASSERT_GT(outputGrid(i), val);
	}

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, SeqSlice){
	int n = TESTS_WIDTH;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	int offset = n/2 - n/4;

	Array<int> inputSlice;
	Array<int> outputSlice;
	inputSlice.hostSlice(inputGrid, offset,0,0,n/2,1,1);
	outputSlice.hostSlice(outputGrid, offset,0,0,n/2,1,1);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputSlice, outputSlice, mask, 0);
	stencil.runSequential();
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputSlice.size(); ++i){
		int val = inputSlice(i);
		if(i>0) val += inputSlice(i-1);
		if(i<(inputSlice.size()-1)) val += inputSlice(i+1);
		ASSERT_EQ(outputSlice(i), val);
	}
	for(int i = 0; i<outputGrid.size(); ++i)
		if( i>=offset && i<(offset+n/2) ){
			int val = inputGrid(i);
			if(i>offset) val += inputGrid(i-1);
			if(i<(offset+n/2-1)) val += inputGrid(i+1);
			ASSERT_EQ(outputGrid(i), val);
		}else ASSERT_EQ(outputGrid(i), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, SeqSliceEvenIterations){
	int n = TESTS_WIDTH;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	int offset = n/2 - n/4;

	Array<int> inputSlice;
	Array<int> outputSlice;
	inputSlice.hostSlice(inputGrid, offset,0,0,n/2,1,1);
	outputSlice.hostSlice(outputGrid, offset,0,0,n/2,1,1);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputSlice, outputSlice, mask, 0);
	stencil.runIterativeSequential(iterations);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputSlice.size(); ++i){
		int val = inputSlice(i);
		if(i>0) val += inputSlice(i-1);
		if(i<(inputSlice.size()-1)) val += inputSlice(i+1);
		ASSERT_GT(outputSlice(i), val*iterations);
	}
	for(int i = 0; i<outputGrid.size(); ++i)
		if( i>=offset && i<(offset+n/2) ){
			int val = inputGrid(i);
			if(i>offset) val += inputGrid(i-1);
			if(i<(offset+n/2-1)) val += inputGrid(i+1);
			ASSERT_GT(outputGrid(i), val*iterations);
		}else ASSERT_EQ(outputGrid(i), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil, SeqSliceOddIterations){
	int n = TESTS_WIDTH;
	int iterations = TESTS_ODD_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(i, inputGrid(i) = i);
	
	int offset = n/2 - n/4;

	Array<int> inputSlice;
	Array<int> outputSlice;
	inputSlice.hostSlice(inputGrid, offset,0,0,n/2,1,1);
	outputSlice.hostSlice(outputGrid, offset,0,0,n/2,1,1);
	
	Mask<int> mask(2);
	mask.set(0,-1);
	mask.set(1,1);

	Stencil<Array<int>, Mask<int>, int> stencil(inputSlice, outputSlice, mask, 0);
	stencil.runIterativeSequential(iterations);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		ASSERT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputSlice.size(); ++i){
		int val = inputSlice(i);
		if(i>0) val += inputSlice(i-1);
		if(i<(inputSlice.size()-1)) val += inputSlice(i+1);
		ASSERT_GT(outputSlice(i), val*iterations);
	}
	for(int i = 0; i<outputGrid.size(); ++i)
		if( i>=offset && i<(offset+n/2) ){
			int val = inputGrid(i);
			if(i>offset) val += inputGrid(i-1);
			if(i<(offset+n/2-1)) val += inputGrid(i+1);
			ASSERT_GT(outputGrid(i), val*iterations);
		}else ASSERT_EQ(outputGrid(i), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}


///
// Stencil2D Tests
//

TEST(Stencil2D, GPU){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int GPUBlockSize = TESTS_GPUBlockSize;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		ASSERT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Mask2D<int> mask(4);
	mask.set(0,-1, 0);
	mask.set(1,1, 0);
	mask.set(2,0,-1);
	mask.set(3,0,1);

	Stencil2D<Array2D<int>, Mask2D<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runGPU(GPUBlockSize);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		ASSERT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w){
		int val = inputGrid(h,w);
		if(h>0) val += inputGrid(h-1,w);
		if(h<(inputGrid.getHeight()-1)) val += inputGrid(h+1,w);
		if(w>0) val += inputGrid(h,w-1);
		if(w<(inputGrid.getWidth()-1)) val += inputGrid(h,w+1);
		ASSERT_EQ(outputGrid(h,w), val);
	}
	inputGrid.hostFree();
	outputGrid.hostFree();
}


TEST(Stencil2D, GPUAlignedTiling){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int GPUBlockSize = TESTS_GPUBlockSize;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		ASSERT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Mask2D<int> mask(4);
	mask.set(0,-1, 0);
	mask.set(1,1, 0);
	mask.set(2,0,-1);
	mask.set(3,0,1);

	Stencil2D<Array2D<int>, Mask2D<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runTilingGPU(width, TESTS_ALIGNED_TILING_HEIGHT, 1, GPUBlockSize);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		ASSERT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w){
		int val = inputGrid(h,w);
		if(h>0) val += inputGrid(h-1,w);
		if(h<(inputGrid.getHeight()-1)) val += inputGrid(h+1,w);
		if(w>0) val += inputGrid(h,w-1);
		if(w<(inputGrid.getWidth()-1)) val += inputGrid(h,w+1);
		ASSERT_EQ(outputGrid(h,w), val);
	}
	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil2D, GPUUnalignedTiling){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int GPUBlockSize = TESTS_GPUBlockSize;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		ASSERT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Mask2D<int> mask(4);
	mask.set(0,-1, 0);
	mask.set(1,1, 0);
	mask.set(2,0,-1);
	mask.set(3,0,1);

	Stencil2D<Array2D<int>, Mask2D<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runTilingGPU(width, TESTS_UNALIGNED_TILING_HEIGHT, 1, GPUBlockSize);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		ASSERT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w){
		int val = inputGrid(h,w);
		if(h>0) val += inputGrid(h-1,w);
		if(h<(inputGrid.getHeight()-1)) val += inputGrid(h+1,w);
		if(w>0) val += inputGrid(h,w-1);
		if(w<(inputGrid.getWidth()-1)) val += inputGrid(h,w+1);
		ASSERT_EQ(outputGrid(h,w), val);
	}
	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil2D, GPUAlignedSquaredTiling){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int GPUBlockSize = TESTS_GPUBlockSize;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		ASSERT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Mask2D<int> mask(4);
	mask.set(0,-1, 0);
	mask.set(1,1, 0);
	mask.set(2,0,-1);
	mask.set(3,0,1);

	Stencil2D<Array2D<int>, Mask2D<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runTilingGPU(TESTS_ALIGNED_TILING_WIDTH, TESTS_ALIGNED_TILING_HEIGHT, 1, GPUBlockSize);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		ASSERT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w){
		int val = inputGrid(h,w);
		if(h>0) val += inputGrid(h-1,w);
		if(h<(inputGrid.getHeight()-1)) val += inputGrid(h+1,w);
		if(w>0) val += inputGrid(h,w-1);
		if(w<(inputGrid.getWidth()-1)) val += inputGrid(h,w+1);
		ASSERT_EQ(outputGrid(h,w), val);
	}
	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil2D, GPUUnalignedSquaredTiling){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int GPUBlockSize = TESTS_GPUBlockSize;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		ASSERT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Mask2D<int> mask(4);
	mask.set(0,-1, 0);
	mask.set(1,1, 0);
	mask.set(2,0,-1);
	mask.set(3,0,1);

	Stencil2D<Array2D<int>, Mask2D<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runTilingGPU(TESTS_UNALIGNED_TILING_WIDTH, TESTS_UNALIGNED_TILING_HEIGHT, 1, GPUBlockSize);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		ASSERT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w){
		int val = inputGrid(h,w);
		if(h>0) val += inputGrid(h-1,w);
		if(h<(inputGrid.getHeight()-1)) val += inputGrid(h+1,w);
		if(w>0) val += inputGrid(h,w-1);
		if(w<(inputGrid.getWidth()-1)) val += inputGrid(h,w+1);
		ASSERT_EQ(outputGrid(h,w), val);
	}
	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil2D, TBB){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int TBBNumThreads = TESTS_TBBNumThreads;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		ASSERT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Mask2D<int> mask(4);
	mask.set(0,-1, 0);
	mask.set(1,1, 0);
	mask.set(2,0,-1);
	mask.set(3,0,1);

	Stencil2D<Array2D<int>, Mask2D<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runCPU(TBBNumThreads);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		ASSERT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w){
		int val = inputGrid(h,w);
		if(h>0) val += inputGrid(h-1,w);
		if(h<(inputGrid.getHeight()-1)) val += inputGrid(h+1,w);
		if(w>0) val += inputGrid(h,w-1);
		if(w<(inputGrid.getWidth()-1)) val += inputGrid(h,w+1);
		ASSERT_EQ(outputGrid(h,w), val);
	}
	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil2D, Seq){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		ASSERT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Mask2D<int> mask(4);
	mask.set(0,-1, 0);
	mask.set(1,1, 0);
	mask.set(2,0,-1);
	mask.set(3,0,1);

	Stencil2D<Array2D<int>, Mask2D<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runSequential();
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		ASSERT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w){
		int val = inputGrid(h,w);
		if(h>0) val += inputGrid(h-1,w);
		if(h<(inputGrid.getHeight()-1)) val += inputGrid(h+1,w);
		if(w>0) val += inputGrid(h,w-1);
		if(w<(inputGrid.getWidth()-1)) val += inputGrid(h,w+1);
		ASSERT_EQ(outputGrid(h,w), val);
	}
	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil2D, SeqIterative){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		ASSERT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Mask2D<int> mask(4);
	mask.set(0,-1, 0);
	mask.set(1,1, 0);
	mask.set(2,0,-1);
	mask.set(3,0,1);

	Stencil2D<Array2D<int>, Mask2D<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runIterativeSequential(2);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		ASSERT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w){
		int val = inputGrid(h,w);
		if(h>0) val += inputGrid(h-1,w);
		if(h<(inputGrid.getHeight()-1)) val += inputGrid(h+1,w);
		if(w>0) val += inputGrid(h,w-1);
		if(w<(inputGrid.getWidth()-1)) val += inputGrid(h,w+1);
		ASSERT_GT(outputGrid(h,w), val*2);
	}
	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Stencil2D, GPUIterative){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int GPUBlockSize = TESTS_GPUBlockSize;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	Array2D<int> baselineGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		ASSERT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Mask2D<int> mask(4);
	mask.set(0,-1, 0);
	mask.set(1,1, 0);
	mask.set(2,0,-1);
	mask.set(3,0,1);

	Stencil2D<Array2D<int>, Mask2D<int>, int> baselineStencil(inputGrid, baselineGrid, mask, 0);
	baselineStencil.runIterativeSequential(2);

	Stencil2D<Array2D<int>, Mask2D<int>, int> stencil(inputGrid, outputGrid, mask, 0);
	stencil.runIterativeTilingGPU(2, 2, width, TESTS_ALIGNED_TILING_HEIGHT, 1, GPUBlockSize);

	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		ASSERT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		ASSERT_EQ(baselineGrid(h,w), outputGrid(h,w));
	
	/*
	ofstream out("input.txt");
	for(int h = 0; h<outputGrid.getHeight(); ++h){
		for(int w = 0; w<outputGrid.getWidth(); ++w){
			out << inputGrid(h,w) << " ";
		}
		out << endl;
	}
	out.close();
	out.open("baseline.txt");
	for(int h = 0; h<outputGrid.getHeight(); ++h){
		for(int w = 0; w<outputGrid.getWidth(); ++w){
			out << baselineGrid(h,w) << " ";
		}
		out << endl;
	}
	out.close();
	out.open("output.txt");
	for(int h = 0; h<outputGrid.getHeight(); ++h){
		for(int w = 0; w<outputGrid.getWidth(); ++w){
			out << outputGrid(h,w) << " ";
		}
		out << endl;
	}
	out.close();
	out.open("err.txt");
	for(int h = 0; h<outputGrid.getHeight(); ++h){
		for(int w = 0; w<outputGrid.getWidth(); ++w){
			out << (outputGrid(h,w)-baselineGrid(h,w)) << " ";
		}
		out << endl;
	}
	out.close();
	*/
	inputGrid.hostFree();
	outputGrid.hostFree();
	baselineGrid.hostFree();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
