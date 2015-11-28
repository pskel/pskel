#include <stdio.h>
//#include <omp.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

//#define PSKEL_SHARED_MASK
#include "../PSkel/include/PSkel.h"

//#include "../utils/hr_time.h"

using namespace std;
using namespace PSkel;

struct Arguments
{
	int externCircle;
	int internCircle;
	float power;
};

namespace PSkel{
	__parallel__ void stencilKernel(Array2D<int> input,Array2D<int> output,Mask2D<int> mask, Arguments arg, size_t h, size_t w){
		int numberA = 0;
		int numberI = 0;
		for (int z = 0; z < mask.size; z++) {
			if(z < arg.internCircle) {
				numberA += mask.get(z, input, h, w);
				printf("A: %d\n", numberA);

			} else {
				numberI += mask.get(z, input, h, w);
				printf("I: %d\n", numberI);
			}
		}
		float totalPowerI = numberI*(arg.power);// The power of Inhibitors
		printf("Power of I: %f\n", totalPowerI);
		if(numberA - totalPowerI < 0) {
			output(h,w) = 0; //without color and inhibitor
			printf("Zero\n");
		} else if(numberA - totalPowerI > 0) {
			output(h,w) = 1;//with color and active
			printf("One\n");
		} else {
			output(h,w) = input(h,w);//doesn't change
			printf("K\n");
		}
	}
}
int CalcSize(int level) {
	if (level == 1) {
		return 3;
	}
	if (level >= 1) {
		return CalcSize(level-1) + 2;
	}
	return 0;
}
int main(int argc, char **argv){
	int width,height,iterations,GPUBlockSize, numCPUThreads, mode, tileHeight, tileIterations, level;
	double power;
	//if (argc != 11){
	//	printf ("Usage: fur WIDTH HEIGHT ITERATIONS GPUBLOCKS CPUTHREADS MODE TILEHEIGHT TILEITERATIONS MASKLEVEL POWER\n");
	//	exit (-1);
	//}

	width = 5;//atoi (argv[1]);
	height = 5;//atoi (argv[2]);
	iterations = 1;//atoi (argv[3]);
    GPUBlockSize = 0;//atoi(argv[4]);
	numCPUThreads = 2;//atoi(argv[5]);
    mode = 0;//atoi(argv[6]);
	tileHeight = 1;//atoi(argv[7]);
	tileIterations = 1;//atoi(argv[8]);
	level = 1;//atoi(argv[9]);

	//power
	power = 2;//atof(argv[10]);

	//create the mask with the level.
	int externCircle;
	int internCircle;
	internCircle = pow(CalcSize(level), 2) - 1;
	externCircle = pow(CalcSize(2*level), 2) - 1 - internCircle;
	int size = internCircle + externCircle;


	Mask2D<int> mask(size);
	
	int count = 0;
	for (int x = (level-2*level); x <= level; x++) {
		for (int y = (level-2*level); y <= level; y++) {
			if (x != 0 || y != 0) {
					mask.set(count, x, y);
					count++;
			}
		}
	}

	for (int x = (2*level-4*level); x <= 2*level; x++) {
		for (int y = (2*level-4*level); y <= 2*level; y++) {
			if (x != 0 || y != 0) {
				if (!(x <= level && x >= -1*level && y <= level && y >= -1*level)) {
					mask.set(count, x, y);
					count++;
				}
			}
		}
	}

	Array2D<int> inputGrid(width,height);
	Array2D<int> outputGrid(width,height);
	
	Arguments arg;
	arg.power = power;
	arg.internCircle = internCircle;
	arg.externCircle = externCircle;

	srand(123456789);
	for(int h=0;h<height;h++) {
		for(int w=0;w<width;w++) {
			inputGrid(h,w) = rand()%2;
			printf("In position %d, %d we have %d\n", h, w, inputGrid(h,w));
		}
	}

	Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(inputGrid, outputGrid, mask, arg);

//	hr_timer_t timer;
	switch(mode){
		case 0:
//			hrt_start(&timer);
			stencil.runIterativeSequential(iterations);
//			hrt_stop(&timer);
			break;
		default:
//			hrt_start(&timer);
			stencil.runIterativeCPU(iterations,numCPUThreads);
//			hrt_stop(&timer);
			break;

		#ifdef PSKEL_CUDA
		case 1:
//			hrt_start(&timer);
			stencil.runIterativeGPU(iterations, GPUBlockSize);	
//			hrt_stop(&timer);
			break;
		case 2:
//			hrt_start(&timer);
			stencil.runIterativeTilingGPU(iterations, width, tileHeight, 1, tileIterations, GPUBlockSize);	
//			hrt_stop(&timer);
			break;
		case 3:
//			hrt_start(&timer);
			stencil.runIterativeAutoGPU(iterations, GPUBlockSize);	
//			hrt_stop(&timer);
			break;
		#endif
	}

//	cout << hrt_elapsed_time(&timer) << endl;
	inputGrid.hostFree();
	outputGrid.hostFree();
	return 0;
}
