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
#include "../include/PSkelStencilTiling.h"

#include "conf.h"

using namespace std;
using namespace PSkel;

TEST(StencilTile2D, Basic){
	int width = 200;
	int height = 300;

	Array2D<int> inputGrid(width, height);
	Array2D<int> outputGrid(width, height);

	Mask2D<int> mask(4);
	mask.set(0,-1, 0);
	mask.set(1,1, 0);
	mask.set(2,0,-1);
	mask.set(3,0,1);
	StencilTiling< Array2D<int>, Mask2D<int> > tiling(inputGrid, outputGrid, mask);

	tiling.tile(2, 0, 0, 0, width, height/3, 1);
	ASSERT_EQ(tiling.widthOffset, 0);
	ASSERT_EQ(tiling.heightOffset, 0);
	ASSERT_EQ(tiling.depthOffset, 0);
	ASSERT_EQ(tiling.width, 200);
	ASSERT_EQ(tiling.height, 102);
	ASSERT_EQ(tiling.depth, 1);
	ASSERT_EQ(tiling.coreWidthOffset, 0);
	ASSERT_EQ(tiling.coreHeightOffset, 0);
	ASSERT_EQ(tiling.coreDepthOffset, 0);
	ASSERT_EQ(tiling.coreWidth, 200);
	ASSERT_EQ(tiling.coreHeight, 100);
	ASSERT_EQ(tiling.coreDepth, 1);

	tiling.tile(2, 0, (height/3), 0, width, height/3, 1);
	ASSERT_EQ(tiling.widthOffset, 0);
	ASSERT_EQ(tiling.heightOffset, 98);
	ASSERT_EQ(tiling.depthOffset, 0);
	ASSERT_EQ(tiling.width, 200);
	ASSERT_EQ(tiling.height, 104);
	ASSERT_EQ(tiling.depth, 1);
	ASSERT_EQ(tiling.coreWidthOffset, 0);
	ASSERT_EQ(tiling.coreHeightOffset, 2);
	ASSERT_EQ(tiling.coreDepthOffset, 0);
	ASSERT_EQ(tiling.coreWidth, 200);
	ASSERT_EQ(tiling.coreHeight, 100);
	ASSERT_EQ(tiling.coreDepth, 1);
	
	tiling.tile(2, 0, 2*(height/3), 0, width, height/3, 1);
	ASSERT_EQ(tiling.widthOffset, 0);
	ASSERT_EQ(tiling.heightOffset, 198);
	ASSERT_EQ(tiling.depthOffset, 0);
	ASSERT_EQ(tiling.width, 200);
	ASSERT_EQ(tiling.height, 102);
	ASSERT_EQ(tiling.depth, 1);
	ASSERT_EQ(tiling.coreWidthOffset, 0);
	ASSERT_EQ(tiling.coreHeightOffset, 2);
	ASSERT_EQ(tiling.coreDepthOffset, 0);
	ASSERT_EQ(tiling.coreWidth, 200);
	ASSERT_EQ(tiling.coreHeight, 100);
	ASSERT_EQ(tiling.coreDepth, 1);

	inputGrid.hostFree();
	outputGrid.hostFree();
	mask.hostFree();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
