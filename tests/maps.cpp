#include <cmath>
#include <omp.h>

#ifdef PSKEL_CUDA
 #include <cuda.h>
 #include <cuda_runtime_api.h>
#endif

#include <gtest/gtest.h>

#include "../include/PSkelMap.h"
#include "../include/PSkelArray.h"

#include "conf.h"

using namespace std;
using namespace PSkel;

namespace PSkel {
	__parallel__ void mapKernel(Array<int> input, Array<int> output, int null, size_t i){
		output(i) = input(i)*2;
	}

	__parallel__ void mapKernel(Array2D<int> input, Array2D<int> output, int null, size_t h, size_t w){
		output(h,w) = input(h,w)*2;
	}
}


#ifdef PSKEL_CUDA
TEST(Map, GPU){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	Map<Array<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runGPU(GPUBlockSize);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i)
		EXPECT_EQ(outputGrid(i), 2*inputGrid(i));

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

#ifdef PSKEL_CUDA
TEST(Map, GPUEvenIterations){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	Map<Array<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runIterativeGPU(iterations,GPUBlockSize);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i)
		EXPECT_EQ(outputGrid(i), pow(2,iterations)*inputGrid(i));

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

#ifdef PSKEL_CUDA
TEST(Map, GPUOddIterations){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int iterations = TESTS_ODD_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	Map<Array<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runIterativeGPU(iterations,GPUBlockSize);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i)
		EXPECT_EQ(outputGrid(i), pow(2,iterations)*inputGrid(i));

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

#ifdef PSKEL_CUDA
TEST(Map, GPUSlice){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	int offset = n/2 - n/4;

	Array<int> inputSlice;
	Array<int> outputSlice;
	inputSlice.hostSlice(inputGrid, offset,0,0,n/2,1,1);
	outputSlice.hostSlice(outputGrid, offset,0,0,n/2,1,1);
	
	Map<Array<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runGPU(GPUBlockSize);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputSlice.size(); ++i)
		EXPECT_EQ(outputSlice(i), 2*inputSlice(i));
	for(int i = 0; i<outputGrid.size(); ++i)
		if( i>=offset && i<(offset+n/2) )
			EXPECT_EQ(outputGrid(i), 2*inputGrid(i));
		else EXPECT_EQ(outputGrid(i), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

#ifdef PSKEL_CUDA
TEST(Map, GPUSliceEvenIterations){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	int offset = n/2 - n/4;

	Array<int> inputSlice;
	Array<int> outputSlice;
	inputSlice.hostSlice(inputGrid, offset,0,0,n/2,1,1);
	outputSlice.hostSlice(outputGrid, offset,0,0,n/2,1,1);
	
	Map<Array<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runIterativeGPU(iterations,GPUBlockSize);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputSlice.size(); ++i)
		EXPECT_EQ(outputSlice(i), pow(2,iterations)*inputSlice(i));
	for(int i = 0; i<outputGrid.size(); ++i)
		if( i>=offset && i<(offset+n/2) )
			EXPECT_EQ(outputGrid(i), pow(2,iterations)*inputGrid(i));
		else EXPECT_EQ(outputGrid(i), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

#ifdef PSKEL_CUDA
TEST(Map, GPUSliceOddIterations){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int iterations = TESTS_ODD_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	int offset = n/2 - n/4;

	Array<int> inputSlice;
	Array<int> outputSlice;
	inputSlice.hostSlice(inputGrid, offset,0,0,n/2,1,1);
	outputSlice.hostSlice(outputGrid, offset,0,0,n/2,1,1);
	
	Map<Array<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runIterativeGPU(iterations,GPUBlockSize);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputSlice.size(); ++i)
		EXPECT_EQ(outputSlice(i), pow(2,iterations)*inputSlice(i));
	for(int i = 0; i<outputGrid.size(); ++i)
		if( i>=offset && i<(offset+n/2) )
			EXPECT_EQ(outputGrid(i), pow(2,iterations)*inputGrid(i));
		else EXPECT_EQ(outputGrid(i), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

TEST(Map, Seq){
	int n = TESTS_WIDTH;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	Map<Array<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runSequential();
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i)
		EXPECT_EQ(outputGrid(i), 2*inputGrid(i));

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map, SeqEvenIterations){
	int n = TESTS_WIDTH;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	Map<Array<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runIterativeSequential(iterations);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i)
		EXPECT_EQ(outputGrid(i), pow(2,iterations)*inputGrid(i));

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map, SeqOddIterations){
	int n = TESTS_WIDTH;
	int iterations = TESTS_ODD_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	Map<Array<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runIterativeSequential(iterations);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i)
		EXPECT_EQ(outputGrid(i), pow(2,iterations)*inputGrid(i));

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map, SeqSlice){
	int n = TESTS_WIDTH;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	int offset = n/2 - n/4;

	Array<int> inputSlice;
	Array<int> outputSlice;
	inputSlice.hostSlice(inputGrid, offset,0,0,n/2,1,1);
	outputSlice.hostSlice(outputGrid, offset,0,0,n/2,1,1);
	
	Map<Array<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runSequential();
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputSlice.size(); ++i)
		EXPECT_EQ(outputSlice(i), 2*inputSlice(i));
	for(int i = 0; i<outputGrid.size(); ++i)
		if( i>=offset && i<(offset+n/2) )
			EXPECT_EQ(outputGrid(i), 2*inputGrid(i));
		else EXPECT_EQ(outputGrid(i), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map, SeqSliceEvenIterations){
	int n = TESTS_WIDTH;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	int offset = n/2 - n/4;

	Array<int> inputSlice;
	Array<int> outputSlice;
	inputSlice.hostSlice(inputGrid, offset,0,0,n/2,1,1);
	outputSlice.hostSlice(outputGrid, offset,0,0,n/2,1,1);
	
	Map<Array<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runIterativeSequential(iterations);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputSlice.size(); ++i)
		EXPECT_EQ(outputSlice(i), pow(2,iterations)*inputSlice(i));
	for(int i = 0; i<outputGrid.size(); ++i)
		if( i>=offset && i<(offset+n/2) )
			EXPECT_EQ(outputGrid(i), pow(2,iterations)*inputGrid(i));
		else EXPECT_EQ(outputGrid(i), 0);
	
	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map, SeqSliceOddIterations){
	int n = TESTS_WIDTH;
	int iterations = TESTS_ODD_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	int offset = n/2 - n/4;

	Array<int> inputSlice;
	Array<int> outputSlice;
	inputSlice.hostSlice(inputGrid, offset,0,0,n/2,1,1);
	outputSlice.hostSlice(outputGrid, offset,0,0,n/2,1,1);
	
	Map<Array<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runIterativeSequential(iterations);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputSlice.size(); ++i)
		EXPECT_EQ(outputSlice(i), pow(2,iterations)*inputSlice(i));
	for(int i = 0; i<outputGrid.size(); ++i)
		if( i>=offset && i<(offset+n/2) )
			EXPECT_EQ(outputGrid(i), pow(2,iterations)*inputGrid(i));
		else EXPECT_EQ(outputGrid(i), 0);
	
	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map, TBB){
	int n = TESTS_WIDTH;
	int TBBNumThreads = TESTS_TBBNumThreads;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	Map<Array<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runCPU(TBBNumThreads);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i)
		EXPECT_EQ(outputGrid(i), 2*inputGrid(i));

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map, TBBEvenIterations){
	int n = TESTS_WIDTH;
	int TBBNumThreads = TESTS_TBBNumThreads;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	Map<Array<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runIterativeCPU(iterations,TBBNumThreads);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i)
		EXPECT_EQ(outputGrid(i), pow(2,iterations)*inputGrid(i));

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map, TBBOddIterations){
	int n = TESTS_WIDTH;
	int TBBNumThreads = TESTS_TBBNumThreads;
	int iterations = TESTS_ODD_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	Map<Array<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runIterativeCPU(iterations,TBBNumThreads);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i)
		EXPECT_EQ(outputGrid(i), pow(2,iterations)*inputGrid(i));

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map, TBBSlice){
	int n = TESTS_WIDTH;
	int TBBNumThreads = TESTS_TBBNumThreads;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	int offset = n/2 - n/4;

	Array<int> inputSlice;
	Array<int> outputSlice;
	inputSlice.hostSlice(inputGrid, offset,0,0,n/2,1,1);
	outputSlice.hostSlice(outputGrid, offset,0,0,n/2,1,1);
	
	Map<Array<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runCPU(TBBNumThreads);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputSlice.size(); ++i)
		EXPECT_EQ(outputSlice(i), 2*inputSlice(i));
	for(int i = 0; i<outputGrid.size(); ++i)
		if( i>=offset && i<(offset+n/2) )
			EXPECT_EQ(outputGrid(i), 2*inputGrid(i));
		else EXPECT_EQ(outputGrid(i), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

/*
#ifdef PSKEL_CUDA
TEST(Map, Hybrid){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int TBBNumThreads = TESTS_TBBNumThreads;
	float GPUPartition = TESTS_GPUPartition;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	Map<Array<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runHybrid(GPUPartition,GPUBlockSize, TBBNumThreads);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i)
		EXPECT_EQ(outputGrid(i), 2*inputGrid(i));

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

#ifdef PSKEL_CUDA
TEST(Map, HybridZeroGPUPortion){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int TBBNumThreads = TESTS_TBBNumThreads;
	float GPUPartition = 0.0;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	Map<Array<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runHybrid(GPUPartition,GPUBlockSize, TBBNumThreads);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i)
		EXPECT_EQ(outputGrid(i), 2*inputGrid(i));

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

#ifdef PSKEL_CUDA
TEST(Map, HybridFullGPUPortion){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int TBBNumThreads = TESTS_TBBNumThreads;
	float GPUPartition = 1.0;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	Map<Array<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runHybrid(GPUPartition,GPUBlockSize, TBBNumThreads);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i)
		EXPECT_EQ(outputGrid(i), 2*inputGrid(i));

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

#ifdef PSKEL_CUDA
TEST(Map, HybridEvenIterations){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int TBBNumThreads = TESTS_TBBNumThreads;
	float GPUPartition = TESTS_GPUPartition;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	Map<Array<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runIterativeHybrid(iterations, GPUPartition,GPUBlockSize, TBBNumThreads);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i)
		EXPECT_EQ(outputGrid(i), pow(2,iterations)*inputGrid(i));

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

#ifdef PSKEL_CUDA
TEST(Map, HybridOddIterations){
	int n = TESTS_WIDTH;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int TBBNumThreads = TESTS_TBBNumThreads;
	float GPUPartition = TESTS_GPUPartition;
	int iterations = TESTS_ODD_ITERATIONS;

	Array<int> inputGrid(n);
	Array<int> outputGrid(n);
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(i, inputGrid(i) = i);
	
	Map<Array<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runIterativeHybrid(iterations, GPUPartition,GPUBlockSize, TBBNumThreads);
	
	for(int i = 0; i<inputGrid.size(); ++i)
		EXPECT_EQ(inputGrid(i), i);
	for(int i = 0; i<outputGrid.size(); ++i)
		EXPECT_EQ(outputGrid(i), pow(2,iterations)*inputGrid(i));

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif
*/

///
// Map2D Tests
//

#ifdef PSKEL_CUDA
TEST(Map2D, GPU){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int GPUBlockSize = TESTS_GPUBlockSize;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Map2D<Array2D<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runGPU(GPUBlockSize);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(outputGrid(h,w), 2*inputGrid(h,w));

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

#ifdef PSKEL_CUDA
TEST(Map2D, GPUEvenIterations){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Map2D<Array2D<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runIterativeGPU(iterations,GPUBlockSize);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(outputGrid(h,w), pow(2,iterations)*inputGrid(h,w));

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

#ifdef PSKEL_CUDA
TEST(Map2D, GPUOddIterations){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int iterations = TESTS_ODD_ITERATIONS;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Map2D<Array2D<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runIterativeGPU(iterations,GPUBlockSize);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(outputGrid(h,w), pow(2,iterations)*inputGrid(h,w));

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

#ifdef PSKEL_CUDA
TEST(Map2D, GPUSlice){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int GPUBlockSize = TESTS_GPUBlockSize;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	int widthOffset = width/2 - width/4;
	int heightOffset = height/2-height/4;

	Array2D<int> inputSlice;
	Array2D<int> outputSlice;
	inputSlice.hostSlice(inputGrid, widthOffset,heightOffset,0,width/2,height/2,1);
	outputSlice.hostSlice(outputGrid, widthOffset,heightOffset,0,width/2,height/2,1);
	
	Map2D<Array2D<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runGPU(GPUBlockSize);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputSlice.getHeight(); ++h)
	for(int w = 0; w<outputSlice.getWidth(); ++w)
		EXPECT_EQ(outputSlice(h,w), 2*inputSlice(h,w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		if( h>=heightOffset && h<(heightOffset+height/2) && w>=widthOffset && w<(widthOffset+width/2))
			EXPECT_EQ(outputGrid(h,w), 2*inputGrid(h,w));
		else EXPECT_EQ(outputGrid(h,w), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

#ifdef PSKEL_CUDA
TEST(Map2D, GPUHorizontalSlice){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int GPUBlockSize = TESTS_GPUBlockSize;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	int widthOffset = 0;
	int heightOffset = height/2-height/4;

	Array2D<int> inputSlice;
	Array2D<int> outputSlice;
	inputSlice.hostSlice(inputGrid, widthOffset,heightOffset,0,width,height/2,1);
	outputSlice.hostSlice(outputGrid, widthOffset,heightOffset,0,width,height/2,1);
	
	Map2D<Array2D<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runGPU(GPUBlockSize);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputSlice.getHeight(); ++h)
	for(int w = 0; w<outputSlice.getWidth(); ++w)
		EXPECT_EQ(outputSlice(h,w), 2*inputSlice(h,w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		if( h>=heightOffset && h<(heightOffset+height/2) )
			EXPECT_EQ(outputGrid(h,w), 2*inputGrid(h,w));
		else EXPECT_EQ(outputGrid(h,w), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

#ifdef PSKEL_CUDA
TEST(Map2D, GPUVerticalSlice){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int GPUBlockSize = TESTS_GPUBlockSize;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	int widthOffset = width/2 - width/4;
	int heightOffset = 0;

	Array2D<int> inputSlice;
	Array2D<int> outputSlice;
	inputSlice.hostSlice(inputGrid, widthOffset,heightOffset,0,width/2,height,1);
	outputSlice.hostSlice(outputGrid, widthOffset,heightOffset,0,width/2,height,1);
	
	Map2D<Array2D<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runGPU(GPUBlockSize);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputSlice.getHeight(); ++h)
	for(int w = 0; w<outputSlice.getWidth(); ++w)
		EXPECT_EQ(outputSlice(h,w), 2*inputSlice(h,w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		if( w>=widthOffset && w<(widthOffset+width/2))
			EXPECT_EQ(outputGrid(h,w), 2*inputGrid(h,w));
		else EXPECT_EQ(outputGrid(h,w), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

TEST(Map2D, Seq){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Map2D<Array2D<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runSequential();
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(outputGrid(h,w), 2*inputGrid(h,w));

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map2D, SeqEvenIterations){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Map2D<Array2D<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runIterativeSequential(iterations);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(outputGrid(h,w), pow(2,iterations)*inputGrid(h,w));

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map2D, SeqOddIterations){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int iterations = TESTS_ODD_ITERATIONS;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Map2D<Array2D<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runIterativeSequential(iterations);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(outputGrid(h,w), pow(2,iterations)*inputGrid(h,w));

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map2D, SeqSlice){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	int widthOffset = width/2 - width/4;
	int heightOffset = height/2-height/4;

	Array2D<int> inputSlice;
	Array2D<int> outputSlice;
	inputSlice.hostSlice(inputGrid, widthOffset,heightOffset,0,width/2,height/2,1);
	outputSlice.hostSlice(outputGrid, widthOffset,heightOffset,0,width/2,height/2,1);
	
	Map2D<Array2D<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runSequential();
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputSlice.getHeight(); ++h)
	for(int w = 0; w<outputSlice.getWidth(); ++w)
		EXPECT_EQ(outputSlice(h,w), 2*inputSlice(h,w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		if( h>=heightOffset && h<(heightOffset+height/2) && w>=widthOffset && w<(widthOffset+width/2))
			EXPECT_EQ(outputGrid(h,w), 2*inputGrid(h,w));
		else EXPECT_EQ(outputGrid(h,w), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map2D, SeqSliceEvenIterations){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	int widthOffset = width/2 - width/4;
	int heightOffset = height/2-height/4;

	Array2D<int> inputSlice;
	Array2D<int> outputSlice;
	inputSlice.hostSlice(inputGrid, widthOffset,heightOffset,0,width/2,height/2,1);
	outputSlice.hostSlice(outputGrid, widthOffset,heightOffset,0,width/2,height/2,1);
	
	Map2D<Array2D<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runIterativeSequential(iterations);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputSlice.getHeight(); ++h)
	for(int w = 0; w<outputSlice.getWidth(); ++w)
		EXPECT_EQ(outputSlice(h,w), pow(2,iterations)*inputSlice(h,w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		if( h>=heightOffset && h<(heightOffset+height/2) && w>=widthOffset && w<(widthOffset+width/2))
			EXPECT_EQ(outputGrid(h,w), pow(2,iterations)*inputGrid(h,w));
		else EXPECT_EQ(outputGrid(h,w), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map2D, SeqSliceOddIterations){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int iterations = TESTS_ODD_ITERATIONS;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	int widthOffset = width/2 - width/4;
	int heightOffset = height/2-height/4;

	Array2D<int> inputSlice;
	Array2D<int> outputSlice;
	inputSlice.hostSlice(inputGrid, widthOffset,heightOffset,0,width/2,height/2,1);
	outputSlice.hostSlice(outputGrid, widthOffset,heightOffset,0,width/2,height/2,1);
	
	Map2D<Array2D<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runIterativeSequential(iterations);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputSlice.getHeight(); ++h)
	for(int w = 0; w<outputSlice.getWidth(); ++w)
		EXPECT_EQ(outputSlice(h,w), pow(2,iterations)*inputSlice(h,w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		if( h>=heightOffset && h<(heightOffset+height/2) && w>=widthOffset && w<(widthOffset+width/2))
			EXPECT_EQ(outputGrid(h,w), pow(2,iterations)*inputGrid(h,w));
		else EXPECT_EQ(outputGrid(h,w), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map2D, SeqHorizontalSlice){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	int widthOffset = 0;
	int heightOffset = height/2-height/4;

	Array2D<int> inputSlice;
	Array2D<int> outputSlice;
	inputSlice.hostSlice(inputGrid, widthOffset,heightOffset,0,width,height/2,1);
	outputSlice.hostSlice(outputGrid, widthOffset,heightOffset,0,width,height/2,1);
	
	Map2D<Array2D<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runSequential();
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputSlice.getHeight(); ++h)
	for(int w = 0; w<outputSlice.getWidth(); ++w)
		EXPECT_EQ(outputSlice(h,w), 2*inputSlice(h,w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		if( h>=heightOffset && h<(heightOffset+height/2) )
			EXPECT_EQ(outputGrid(h,w), 2*inputGrid(h,w));
		else EXPECT_EQ(outputGrid(h,w), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map2D, SeqVerticalSlice){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	int widthOffset = width/2 - width/4;
	int heightOffset = 0;

	Array2D<int> inputSlice;
	Array2D<int> outputSlice;
	inputSlice.hostSlice(inputGrid, widthOffset,heightOffset,0,width/2,height,1);
	outputSlice.hostSlice(outputGrid, widthOffset,heightOffset,0,width/2,height,1);
	
	Map2D<Array2D<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runSequential();
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputSlice.getHeight(); ++h)
	for(int w = 0; w<outputSlice.getWidth(); ++w)
		EXPECT_EQ(outputSlice(h,w), 2*inputSlice(h,w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		if( w>=widthOffset && w<(widthOffset+width/2))
			EXPECT_EQ(outputGrid(h,w), 2*inputGrid(h,w));
		else EXPECT_EQ(outputGrid(h,w), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map2D, TBB){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int TBBNumThreads = TESTS_TBBNumThreads;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Map2D<Array2D<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runCPU(TBBNumThreads);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(outputGrid(h,w), 2*inputGrid(h,w));

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map2D, TBBEvenIterations){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int TBBNumThreads = TESTS_TBBNumThreads;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Map2D<Array2D<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runIterativeCPU(iterations,TBBNumThreads);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(outputGrid(h,w), pow(2,iterations)*inputGrid(h,w));

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map2D, TBBOddIterations){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int TBBNumThreads = TESTS_TBBNumThreads;
	int iterations = TESTS_ODD_ITERATIONS;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Map2D<Array2D<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runIterativeCPU(iterations,TBBNumThreads);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(outputGrid(h,w), pow(2,iterations)*inputGrid(h,w));

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map2D, TBBSlice){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int TBBNumThreads = TESTS_TBBNumThreads;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	int widthOffset = width/2 - width/4;
	int heightOffset = height/2-height/4;

	Array2D<int> inputSlice;
	Array2D<int> outputSlice;
	inputSlice.hostSlice(inputGrid, widthOffset,heightOffset,0,width/2,height/2,1);
	outputSlice.hostSlice(outputGrid, widthOffset,heightOffset,0,width/2,height/2,1);
	
	Map2D<Array2D<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runCPU(TBBNumThreads);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputSlice.getHeight(); ++h)
	for(int w = 0; w<outputSlice.getWidth(); ++w)
		EXPECT_EQ(outputSlice(h,w), 2*inputSlice(h,w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		if( h>=heightOffset && h<(heightOffset+height/2) && w>=widthOffset && w<(widthOffset+width/2))
			EXPECT_EQ(outputGrid(h,w), 2*inputGrid(h,w));
		else EXPECT_EQ(outputGrid(h,w), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map2D, TBBHorizontalSlice){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int TBBNumThreads = TESTS_TBBNumThreads;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	int widthOffset = 0;
	int heightOffset = height/2-height/4;

	Array2D<int> inputSlice;
	Array2D<int> outputSlice;
	inputSlice.hostSlice(inputGrid, widthOffset,heightOffset,0,width,height/2,1);
	outputSlice.hostSlice(outputGrid, widthOffset,heightOffset,0,width,height/2,1);
	
	Map2D<Array2D<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runCPU(TBBNumThreads);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputSlice.getHeight(); ++h)
	for(int w = 0; w<outputSlice.getWidth(); ++w)
		EXPECT_EQ(outputSlice(h,w), 2*inputSlice(h,w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		if( h>=heightOffset && h<(heightOffset+height/2) )
			EXPECT_EQ(outputGrid(h,w), 2*inputGrid(h,w));
		else EXPECT_EQ(outputGrid(h,w), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

TEST(Map2D, TBBVerticalSlice){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int TBBNumThreads = TESTS_TBBNumThreads;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	int widthOffset = width/2 - width/4;
	int heightOffset = 0;

	Array2D<int> inputSlice;
	Array2D<int> outputSlice;
	inputSlice.hostSlice(inputGrid, widthOffset,heightOffset,0,width/2,height,1);
	outputSlice.hostSlice(outputGrid, widthOffset,heightOffset,0,width/2,height,1);
	
	Map2D<Array2D<int>, int> twiceSlice(inputSlice, outputSlice, 0);
	twiceSlice.runCPU(TBBNumThreads);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputSlice.getHeight(); ++h)
	for(int w = 0; w<outputSlice.getWidth(); ++w)
		EXPECT_EQ(outputSlice(h,w), 2*inputSlice(h,w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		if( w>=widthOffset && w<(widthOffset+width/2))
			EXPECT_EQ(outputGrid(h,w), 2*inputGrid(h,w));
		else EXPECT_EQ(outputGrid(h,w), 0);

	inputGrid.hostFree();
	outputGrid.hostFree();
}

/*
#ifdef PSKEL_CUDA
TEST(Map2D,  Hybrid){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int TBBNumThreads = TESTS_TBBNumThreads;
	float GPUPartition = TESTS_GPUPartition;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Map2D<Array2D<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runHybrid(GPUPartition,GPUBlockSize, TBBNumThreads);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(outputGrid(h,w), 2*inputGrid(h,w));

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

#ifdef PSKEL_CUDA
TEST(Map2D, HybridEvenIterations){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int TBBNumThreads = TESTS_TBBNumThreads;
	float GPUPartition = TESTS_GPUPartition;
	int iterations = TESTS_EVEN_ITERATIONS;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Map2D<Array2D<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runIterativeHybrid(iterations, GPUPartition,GPUBlockSize, TBBNumThreads);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(outputGrid(h,w), pow(2,iterations)*inputGrid(h,w));

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif

#ifdef PSKEL_CUDA
TEST(Map2D, HybridOddIterations){
	int width = TESTS_WIDTH;
	int height = TESTS_HEIGHT;
	int GPUBlockSize = TESTS_GPUBlockSize;
	int TBBNumThreads = TESTS_TBBNumThreads;
	float GPUPartition = TESTS_GPUPartition;
	int iterations = TESTS_ODD_ITERATIONS;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);
	for(int h = 0; h<inputGrid.getHeight(); ++h)
	for(int w = 0; w<inputGrid.getWidth(); ++w)
		EXPECT_EQ(h*inputGrid.getWidth()+w, inputGrid(h,w) = h*inputGrid.getWidth()+w);
	
	Map2D<Array2D<int>, int> twice(inputGrid, outputGrid, 0);
	twice.runIterativeHybrid(iterations, GPUPartition,GPUBlockSize, TBBNumThreads);
	
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(inputGrid(h,w), (h*inputGrid.getWidth()+w));
	for(int h = 0; h<outputGrid.getHeight(); ++h)
	for(int w = 0; w<outputGrid.getWidth(); ++w)
		EXPECT_EQ(outputGrid(h,w), pow(2,iterations)*inputGrid(h,w));

	inputGrid.hostFree();
	outputGrid.hostFree();
}
#endif
*/
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
