#ifdef PSKEL_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif
#include <gtest/gtest.h>

#include "../include/PSkelArray.h"

#include "conf.h"

using namespace PSkel;

TEST(Array, Constructor){
	int n = TESTS_WIDTH;
	Array<int> arr(n);

	EXPECT_EQ(n, arr.getWidth());
	EXPECT_EQ(1, arr.getHeight());
	EXPECT_EQ(1, arr.getDepth());
	EXPECT_EQ(n, arr.size());
	EXPECT_EQ(n, arr.realSize());
	EXPECT_TRUE(arr);
	arr.hostFree();
	EXPECT_FALSE(arr);
}

TEST(Array, Assignment){
	int n = TESTS_WIDTH;
	Array<int> arr(n);
	
	for(int i=0;i<arr.getWidth();i++){
		EXPECT_EQ(i, arr(i)=i);
	}
	for(int i=0;i<arr.getWidth();i++){
		EXPECT_EQ(i, arr(i));
	}
	arr.hostFree();
}

#ifdef PSKEL_CUDA
TEST(Array, HostDeviceCopy){
	int n = TESTS_WIDTH;
	Array<int> arr(n);
	
	for(int i=0;i<arr.getWidth();i++){
		EXPECT_EQ(i, arr(i)=i);
	}
	arr.deviceAlloc();
	arr.copyToDevice();
	for(int i=0;i<arr.getWidth();i++){
		EXPECT_EQ(0, arr(i)=0);
	}
	for(int i=0;i<arr.getWidth();i++){
		EXPECT_EQ(0, arr(i));
	}
	arr.copyToHost();
	for(int i=0;i<arr.getWidth();i++){
		EXPECT_EQ(i, arr(i));
	}
	arr.deviceFree();
	arr.hostFree();
}
#endif

TEST(Array, Clone){
	int n = TESTS_WIDTH;
	Array<int> arr(n);
	
	for(int i=0;i<arr.getWidth();i++){
		EXPECT_EQ(i, arr(i)=i);
	}
	Array<int> arrClone;
	arrClone.hostClone(arr);
	EXPECT_EQ(arrClone.getWidth(), arr.getWidth());
	EXPECT_EQ(arrClone.getHeight(), arr.getHeight());
	EXPECT_EQ(arrClone.getDepth(), arr.getDepth());
	EXPECT_EQ(arrClone.size(), arr.size());
	EXPECT_EQ(arrClone.realSize(), arr.size());
	for(int i=0;i<arrClone.getWidth();i++){
		EXPECT_EQ(arrClone(i), arr(i));
	}
	for(int i=0;i<arrClone.getWidth();i++){
		arrClone(i) = arrClone(i)*2;
	}
	for(int i=0;i<arrClone.getWidth();i++){
		EXPECT_EQ(arrClone(i), (2*arr(i)));
	}
	for(int i=0;i<arr.getWidth();i++){
		EXPECT_EQ(i, arr(i));
	}
	arrClone.hostFree();
	arr.hostFree();
}

TEST(Array, Slice){
	int n = TESTS_WIDTH;
	Array<int> arr(n);
	
	for(int i=0;i<arr.getWidth();i++){
		EXPECT_EQ(i, arr(i)=i);
	}

	Array<int> arrSlice;
	arrSlice.hostSlice(arr,n/2-n/4,0,0,n/2,1,1);
	EXPECT_EQ(arrSlice.getWidth(), n/2);
	EXPECT_EQ(arrSlice.getHeight(), 1);
	EXPECT_EQ(arrSlice.getDepth(), 1);
	EXPECT_EQ(arrSlice.size(), n/2);
	EXPECT_EQ(arrSlice.realSize(), n);
	for(int i=0;i<arrSlice.getWidth();i++){
		EXPECT_EQ(arrSlice(i), arr((n/2-n/4)+i));
	}
	for(int i=0;i<arrSlice.getWidth();i++){
		arrSlice(i) = arrSlice(i)*2;
	}
	for(int i=0;i<arrSlice.getWidth();i++){
		EXPECT_EQ(arrSlice(i), arr((n/2-n/4)+i));
	}
	for(int i=0;i<arr.getWidth();i++){
		if(i>=(n/2-n/4) && i<((n/2-n/4)+n/2))
			EXPECT_EQ(arr(i), 2*i);
		else EXPECT_EQ(arr(i), i);
	}
	
	#ifdef PSKEL_CUDA
	arrSlice.deviceAlloc();
	arrSlice.copyToDevice();
	#endif
	for(int i=0;i<arrSlice.getWidth();i++){
		EXPECT_EQ(0, arrSlice(i)=0);
	}
	for(int i=0;i<arrSlice.getWidth();i++){
		EXPECT_EQ(0, arrSlice(i));
	}
	for(int i=0;i<arr.getWidth();i++){
		if(i>=(n/2-n/4) && i<((n/2-n/4)+n/2))
			EXPECT_EQ(arr(i), 0);
		else EXPECT_EQ(arr(i), i);
	}

	#ifdef PSKEL_CUDA
	arrSlice.copyToHost();
	for(int i=0;i<arr.getWidth();i++){
		if(i>=(n/2-n/4) && i<((n/2-n/4)+n/2))
			EXPECT_EQ(arr(i), 2*i);
		else EXPECT_EQ(arr(i), i);
	}

	for(int i=0;i<arrSlice.getWidth();i++){
		EXPECT_EQ(arrSlice(i), arr((n/2-n/4)+i));
	}
	arrSlice.deviceFree();
	for(int i=0;i<arr.getWidth();i++){
		if(i>=(n/2-n/4) && i<((n/2-n/4)+n/2))
			EXPECT_EQ(arr(i), 2*i);
		else EXPECT_EQ(arr(i), i);
	}
	#endif
	arr.hostFree();
}

TEST(Array2D, Constructor){
	size_t width = TESTS_WIDTH;
	size_t height = TESTS_HEIGHT;
	Array2D<int> arr(width, height);

	EXPECT_EQ(width, arr.getWidth());
	EXPECT_EQ(height, arr.getHeight());
	EXPECT_EQ(1, arr.getDepth());
	EXPECT_EQ(width*height, arr.size());
	EXPECT_EQ(width*height, arr.realSize());
	EXPECT_TRUE(arr);
	arr.hostFree();
	EXPECT_FALSE(arr);
}

TEST(Array2D, Assignment){
	size_t width = TESTS_WIDTH;
	size_t height = TESTS_HEIGHT;
	Array2D<int> arr(width, height);
	
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		EXPECT_EQ(h*arr.getWidth()+w, arr(h,w)=h*arr.getWidth()+w);
	}}
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		EXPECT_EQ(h*arr.getWidth()+w, arr(h,w));
	}}
	arr.hostFree();
}

#ifdef PSKEL_CUDA
TEST(Array2D, HostDeviceCopy){
	size_t width = TESTS_WIDTH;
	size_t height = TESTS_HEIGHT;
	Array2D<int> arr(width, height);
	
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		EXPECT_EQ(h*arr.getWidth()+w, arr(h,w)=h*arr.getWidth()+w);
	}}
	arr.deviceAlloc();
	arr.copyToDevice();
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		EXPECT_EQ(0, arr(h,w)=0);
	}}
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		EXPECT_EQ(0, arr(h,w));
	}}
	arr.copyToHost();
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		EXPECT_EQ(h*arr.getWidth()+w, arr(h,w));
	}}
	arr.deviceFree();
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		EXPECT_EQ(h*arr.getWidth()+w, arr(h,w));
	}}
	arr.hostFree();
}
#endif

TEST(Array2D, Clone){
	size_t width = TESTS_WIDTH;
	size_t height = TESTS_HEIGHT;
	Array2D<int> arr(width, height);
	
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		EXPECT_EQ(h*arr.getWidth()+w, arr(h,w)=h*arr.getWidth()+w);
	}}

	Array2D<int> arrClone;
	arrClone.hostClone(arr);
	EXPECT_EQ(arrClone.getWidth(), arr.getWidth());
	EXPECT_EQ(arrClone.getHeight(), arr.getHeight());
	EXPECT_EQ(arrClone.getDepth(), arr.getDepth());
	EXPECT_EQ(arrClone.size(), arr.size());
	EXPECT_EQ(arrClone.realSize(), arr.size());
	for(int h=0;h<arrClone.getHeight();h++){
	for(int w=0;w<arrClone.getWidth();w++){
		EXPECT_EQ(arrClone(h,w), arr(h,w));
	}}
	for(int h=0;h<arrClone.getHeight();h++){
	for(int w=0;w<arrClone.getWidth();w++){
		arrClone(h,w) = arrClone(h,w)*2;
	}}
	for(int h=0;h<arrClone.getHeight();h++){
	for(int w=0;w<arrClone.getWidth();w++){
		EXPECT_EQ(arrClone(h,w), 2*arr(h,w));
	}}
	arrClone.hostFree();
	arr.hostFree();
}

TEST(Array2D, Slice){
	size_t width = TESTS_WIDTH;
	size_t height = TESTS_HEIGHT;
	Array2D<int> arr(width, height);
	
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		EXPECT_EQ(h*arr.getWidth()+w, arr(h,w)=h*arr.getWidth()+w);
	}}

	Array2D<int> arrSlice;
	arrSlice.hostSlice(arr,width/2-width/4,height/2-height/4,0,width/2,height/2,1);
	EXPECT_EQ(arrSlice.getWidth(), width/2);
	EXPECT_EQ(arrSlice.getHeight(), height/2);
	EXPECT_EQ(arrSlice.getDepth(), 1);
	EXPECT_EQ(arrSlice.size(), ((width/2)*(height/2)));
	EXPECT_EQ(arrSlice.realSize(), arr.realSize());
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		EXPECT_EQ(arrSlice(h,w), arr(height/2-height/4+h, width/2-width/4+w));
	}}
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		arrSlice(h,w) = arrSlice(h,w)*2;
	}}
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		EXPECT_EQ(arrSlice(h,w), arr(height/2-height/4+h, width/2-width/4+w));
	}}
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		if(h>=(height/2-height/4) && h<((height/2-height/4)+height/2) && w>=(width/2-width/4) && w<((width/2-width/4)+width/2)){
			EXPECT_EQ(arr(h,w), 2*(h*arr.getWidth()+w));
		}else EXPECT_EQ(arr(h,w), (h*arr.getWidth()+w));
	}}

	#ifdef PSKEL_CUDA
	arrSlice.deviceAlloc();
	arrSlice.copyToDevice();
	#endif
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		EXPECT_EQ(0, arrSlice(h,w)=0);
	}}
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		EXPECT_EQ(arrSlice(h,w), 0);
	}}
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		if(h>=(height/2-height/4) && h<((height/2-height/4)+height/2) && w>=(width/2-width/4) && w<((width/2-width/4)+width/2)){
			EXPECT_EQ(arr(h,w), 0);
		}else EXPECT_EQ(arr(h,w), (h*arr.getWidth()+w));
	}}
	
	#ifdef PSKEL_CUDA
	arrSlice.copyToHost();
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		if(h>=(height/2-height/4) && h<((height/2-height/4)+height/2) && w>=(width/2-width/4) && w<((width/2-width/4)+width/2)){
			EXPECT_EQ(arr(h,w), 2*(h*arr.getWidth()+w));
		}else EXPECT_EQ(arr(h,w), (h*arr.getWidth()+w));
	}}
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		EXPECT_EQ(arrSlice(h,w), arr(height/2-height/4+h, width/2-width/4+w));
	}}
	arrSlice.deviceFree();
	#endif
	arr.hostFree();
}

TEST(Array2D, HorizontalSlice){
	size_t width = TESTS_WIDTH;
	size_t height = TESTS_HEIGHT;
	Array2D<int> arr(width, height);
	
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		EXPECT_EQ(h*arr.getWidth()+w, arr(h,w)=h*arr.getWidth()+w);
	}}

	Array2D<int> arrSlice;
	arrSlice.hostSlice(arr,0,height/2-height/4,0,width,height/2,1);
	EXPECT_EQ(arrSlice.getWidth(), width);
	EXPECT_EQ(arrSlice.getHeight(), height/2);
	EXPECT_EQ(arrSlice.getDepth(), 1);
	EXPECT_EQ(arrSlice.size(), (width*(height/2)));
	EXPECT_EQ(arrSlice.realSize(), arr.realSize());
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		EXPECT_EQ(arrSlice(h,w), arr(height/2-height/4+h, w));
	}}
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		arrSlice(h,w) = arrSlice(h,w)*2;
	}}
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		EXPECT_EQ(arrSlice(h,w), arr(height/2-height/4+h, w));
	}}
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		if(h>=(height/2-height/4) && h<((height/2-height/4)+height/2)){
			EXPECT_EQ(arr(h,w), 2*(h*arr.getWidth()+w));
		}else EXPECT_EQ(arr(h,w), (h*arr.getWidth()+w));
	}}
	#ifdef PSKEL_CUDA
	arrSlice.deviceAlloc();
	arrSlice.copyToDevice();
	#endif
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		EXPECT_EQ(0, arrSlice(h,w)=0);
	}}
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		EXPECT_EQ(arrSlice(h,w), 0);
	}}
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		if(h>=(height/2-height/4) && h<((height/2-height/4)+height/2)){
			EXPECT_EQ(arr(h,w), 0);
		}else EXPECT_EQ(arr(h,w), (h*arr.getWidth()+w));
	}}
	#ifdef PSKEL_CUDA
	arrSlice.copyToHost();
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		if(h>=(height/2-height/4) && h<((height/2-height/4)+height/2)){
			EXPECT_EQ(arr(h,w), 2*(h*arr.getWidth()+w));
		}else EXPECT_EQ(arr(h,w), (h*arr.getWidth()+w));
	}}
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		EXPECT_EQ(arrSlice(h,w), arr(height/2-height/4+h, w));
	}}
	arrSlice.deviceFree();
	#endif
	arr.hostFree();
}

TEST(Array2D, VerticalSlice){
	size_t width = TESTS_WIDTH;
	size_t height = TESTS_HEIGHT;
	Array2D<int> arr(width, height);
	
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		EXPECT_EQ(h*arr.getWidth()+w, arr(h,w)=h*arr.getWidth()+w);
	}}

	Array2D<int> arrSlice;
	arrSlice.hostSlice(arr,width/2-width/4,0,0,width/2,height,1);
	EXPECT_EQ(arrSlice.getWidth(), width/2);
	EXPECT_EQ(arrSlice.getHeight(), height);
	EXPECT_EQ(arrSlice.getDepth(), 1);
	EXPECT_EQ(arrSlice.size(), (width*(height/2)));
	EXPECT_EQ(arrSlice.realSize(), arr.realSize());
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		EXPECT_EQ(arrSlice(h,w), arr(h, width/2-width/4+w));
	}}
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		arrSlice(h,w) = arrSlice(h,w)*2;
	}}
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		EXPECT_EQ(arrSlice(h,w), arr(h, width/2-width/4+w));
	}}
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		if(w>=(width/2-width/4) && w<((width/2-width/4)+width/2)){
			EXPECT_EQ(arr(h,w), 2*(h*arr.getWidth()+w));
		}else EXPECT_EQ(arr(h,w), (h*arr.getWidth()+w));
	}}
	#ifdef PSKEL_CUDA
	arrSlice.deviceAlloc();
	arrSlice.copyToDevice();
	#endif
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		EXPECT_EQ(0, arrSlice(h,w)=0);
	}}
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		EXPECT_EQ(arrSlice(h,w), 0);
	}}
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		if(w>=(width/2-width/4) && w<((width/2-width/4)+width/2)){
			EXPECT_EQ(arr(h,w), 0);
		}else EXPECT_EQ(arr(h,w), (h*arr.getWidth()+w));
	}}
	#ifdef PSKEL_CUDA
	arrSlice.copyToHost();
	for(int h=0;h<arr.getHeight();h++){
	for(int w=0;w<arr.getWidth();w++){
		if(w>=(width/2-width/4) && w<((width/2-width/4)+width/2)){
			EXPECT_EQ(arr(h,w), 2*(h*arr.getWidth()+w));
		}else EXPECT_EQ(arr(h,w), (h*arr.getWidth()+w));
	}}
	for(int h=0;h<arrSlice.getHeight();h++){
	for(int w=0;w<arrSlice.getWidth();w++){
		EXPECT_EQ(arrSlice(h,w), arr(h, width/2-width/4+w));
	}}
	arrSlice.deviceFree();
	#endif
	arr.hostFree();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
