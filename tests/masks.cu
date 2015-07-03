#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "../include/PSkelArray.h"
#include "../include/PSkelMask.h"

using namespace std;
using namespace PSkel;

TEST(Mask2D, Basic){
	Mask2D<int> mask(8);
	mask.set(0,-1,-1);	mask.set(1,-1,0);	mask.set(2,-1,1);
	mask.set(3,0,-1);				mask.set(4,0,1);
	mask.set(5,1,-1);	mask.set(6,1,0);	mask.set(7,1,1);
	Array2D<int> arr(3, 3);
	arr(0,0) = 0; arr(0,1) = 1; arr(0,2) = 2;
	arr(1,0) = 3; arr(1,1) = 0; arr(1,2) = 4;
	arr(2,0) = 5; arr(2,1) = 6; arr(2,2) = 7;
	
	for(int i=0;i<mask.size;i++){
		EXPECT_EQ(mask.get(i, arr, 1, 1), i);
	}
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
