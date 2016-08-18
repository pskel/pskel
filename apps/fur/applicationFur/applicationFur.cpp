#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

//#define PSKEL_SHARED_MASK
#include "../include/PSkel.h"

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
				//printf("A: %d\n", numberA);

			} else {
				numberI += mask.get(z, input, h, w);
				//printf("I: %d\n", numberI);
			}
		}
		float totalPowerI = numberI*(arg.power);// The power of Inhibitors
		//printf("Power of I: %f\n", totalPowerI);
		if(numberA - totalPowerI < 0) {
			output(h,w) = 0; //without color and inhibitor
			//printf("Zero\n");
		} else if(numberA - totalPowerI > 0) {
			output(h,w) = 1;//with color and active
			//printf("One\n");
		} else {
			output(h,w) = input(h,w);//doesn't change
			//printf("K\n");
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
	if (argc != 11){
		printf ("Usage: fur WIDTH HEIGHT ITERATIONS GPUBLOCKS CPUTHREADS MODE TILEHEIGHT TILEITERATIONS MASKLEVEL POWER\n");
		exit (-1);
	}

	width = atoi (argv[1]);
	//printf("width:%d\n", width);
	height = atoi (argv[2]);
	iterations = atoi (argv[3]);
    GPUBlockSize = atoi(argv[4]);
	numCPUThreads = atoi(argv[5]);
    mode = atoi(argv[6]);
	tileHeight = atoi(argv[7]);
	tileIterations = atoi(argv[8]);
	level = atoi(argv[9]);

	//power
	power = atof(argv[10]);

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
	// ofstream myfile ("inputGrid.txt");
	for(int h=0;h<height;h++) {
		for(int w=0;w<width;w++) {
			inputGrid(h,w) = rand()%2;
			//inputGrid(h,w) = count;
			//count++;
		  	//if (myfile.is_open())
		  	//	if(inputGrid(h,w) == 0) {
			//    	myfile << "0\n";
			    	//myfile.close();
			//	} else {
					//myfile << "1\n";
			//	}
		  	//else cout << "Unable to open file";
			//printf("inputGrid(%d,%d) = %d;\n", h, w, inputGrid(h,w));
		}
	}
// inputGrid(0,0) = 1;
// inputGrid(0,1) = 0;
// inputGrid(0,2) = 0;
// inputGrid(0,3) = 0;
// inputGrid(0,4) = 0;
// inputGrid(0,5) = 1;
// inputGrid(0,6) = 1;
// inputGrid(0,7) = 0;
// inputGrid(0,8) = 0;
// inputGrid(0,9) = 1;
// inputGrid(0,10) = 1;
// inputGrid(0,11) = 0;
// inputGrid(0,12) = 1;
// inputGrid(0,13) = 0;
// inputGrid(0,14) = 1;
// inputGrid(0,15) = 0;
// inputGrid(1,0) = 0;
// inputGrid(1,1) = 0;
// inputGrid(1,2) = 0;
// inputGrid(1,3) = 0;
// inputGrid(1,4) = 1;
// inputGrid(1,5) = 0;
// inputGrid(1,6) = 1;
// inputGrid(1,7) = 1;
// inputGrid(1,8) = 0;
// inputGrid(1,9) = 0;
// inputGrid(1,10) = 0;
// inputGrid(1,11) = 0;
// inputGrid(1,12) = 1;
// inputGrid(1,13) = 1;
// inputGrid(1,14) = 1;
// inputGrid(1,15) = 1;
// inputGrid(2,0) = 1;
// inputGrid(2,1) = 1;
// inputGrid(2,2) = 0;
// inputGrid(2,3) = 0;
// inputGrid(2,4) = 0;
// inputGrid(2,5) = 1;
// inputGrid(2,6) = 1;
// inputGrid(2,7) = 1;
// inputGrid(2,8) = 1;
// inputGrid(2,9) = 1;
// inputGrid(2,10) = 0;
// inputGrid(2,11) = 0;
// inputGrid(2,12) = 1;
// inputGrid(2,13) = 0;
// inputGrid(2,14) = 1;
// inputGrid(2,15) = 1;
// inputGrid(3,0) = 0;
// inputGrid(3,1) = 0;
// inputGrid(3,2) = 0;
// inputGrid(3,3) = 0;
// inputGrid(3,4) = 0;
// inputGrid(3,5) = 1;
// inputGrid(3,6) = 1;
// inputGrid(3,7) = 0;
// inputGrid(3,8) = 1;
// inputGrid(3,9) = 1;
// inputGrid(3,10) = 1;
// inputGrid(3,11) = 1;
// inputGrid(3,12) = 0;
// inputGrid(3,13) = 0;
// inputGrid(3,14) = 1;
// inputGrid(3,15) = 1;
// inputGrid(4,0) = 1;
// inputGrid(4,1) = 0;
// inputGrid(4,2) = 1;
// inputGrid(4,3) = 1;
// inputGrid(4,4) = 0;
// inputGrid(4,5) = 1;
// inputGrid(4,6) = 0;
// inputGrid(4,7) = 0;
// inputGrid(4,8) = 0;
// inputGrid(4,9) = 0;
// inputGrid(4,10) = 1;
// inputGrid(4,11) = 1;
// inputGrid(4,12) = 1;
// inputGrid(4,13) = 1;
// inputGrid(4,14) = 0;
// inputGrid(4,15) = 0;
// inputGrid(5,0) = 0;
// inputGrid(5,1) = 0;
// inputGrid(5,2) = 1;
// inputGrid(5,3) = 1;
// inputGrid(5,4) = 0;
// inputGrid(5,5) = 0;
// inputGrid(5,6) = 0;
// inputGrid(5,7) = 1;
// inputGrid(5,8) = 0;
// inputGrid(5,9) = 1;
// inputGrid(5,10) = 0;
// inputGrid(5,11) = 0;
// inputGrid(5,12) = 1;
// inputGrid(5,13) = 1;
// inputGrid(5,14) = 1;
// inputGrid(5,15) = 0;
// inputGrid(6,0) = 1;
// inputGrid(6,1) = 1;
// inputGrid(6,2) = 0;
// inputGrid(6,3) = 1;
// inputGrid(6,4) = 1;
// inputGrid(6,5) = 1;
// inputGrid(6,6) = 0;
// inputGrid(6,7) = 1;
// inputGrid(6,8) = 1;
// inputGrid(6,9) = 0;
// inputGrid(6,10) = 1;
// inputGrid(6,11) = 1;
// inputGrid(6,12) = 0;
// inputGrid(6,13) = 0;
// inputGrid(6,14) = 1;
// inputGrid(6,15) = 1;
// inputGrid(7,0) = 1;
// inputGrid(7,1) = 0;
// inputGrid(7,2) = 1;
// inputGrid(7,3) = 1;
// inputGrid(7,4) = 1;
// inputGrid(7,5) = 0;
// inputGrid(7,6) = 1;
// inputGrid(7,7) = 1;
// inputGrid(7,8) = 1;
// inputGrid(7,9) = 0;
// inputGrid(7,10) = 0;
// inputGrid(7,11) = 0;
// inputGrid(7,12) = 0;
// inputGrid(7,13) = 0;
// inputGrid(7,14) = 0;
// inputGrid(7,15) = 1;
// inputGrid(8,0) = 1;
// inputGrid(8,1) = 0;
// inputGrid(8,2) = 1;
// inputGrid(8,3) = 1;
// inputGrid(8,4) = 0;
// inputGrid(8,5) = 0;
// inputGrid(8,6) = 1;
// inputGrid(8,7) = 0;
// inputGrid(8,8) = 0;
// inputGrid(8,9) = 1;
// inputGrid(8,10) = 0;
// inputGrid(8,11) = 0;
// inputGrid(8,12) = 0;
// inputGrid(8,13) = 0;
// inputGrid(8,14) = 1;
// inputGrid(8,15) = 0;
// inputGrid(9,0) = 1;
// inputGrid(9,1) = 1;
// inputGrid(9,2) = 0;
// inputGrid(9,3) = 0;
// inputGrid(9,4) = 0;
// inputGrid(9,5) = 1;
// inputGrid(9,6) = 0;
// inputGrid(9,7) = 0;
// inputGrid(9,8) = 1;
// inputGrid(9,9) = 1;
// inputGrid(9,10) = 0;
// inputGrid(9,11) = 1;
// inputGrid(9,12) = 0;
// inputGrid(9,13) = 0;
// inputGrid(9,14) = 1;
// inputGrid(9,15) = 1;
// inputGrid(10,0) = 0;
// inputGrid(10,1) = 0;
// inputGrid(10,2) = 1;
// inputGrid(10,3) = 0;
// inputGrid(10,4) = 0;
// inputGrid(10,5) = 0;
// inputGrid(10,6) = 0;
// inputGrid(10,7) = 1;
// inputGrid(10,8) = 1;
// inputGrid(10,9) = 0;
// inputGrid(10,10) = 0;
// inputGrid(10,11) = 0;
// inputGrid(10,12) = 1;
// inputGrid(10,13) = 1;
// inputGrid(10,14) = 1;
// inputGrid(10,15) = 1;
// inputGrid(11,0) = 1;
// inputGrid(11,1) = 1;
// inputGrid(11,2) = 1;
// inputGrid(11,3) = 1;
// inputGrid(11,4) = 1;
// inputGrid(11,5) = 1;
// inputGrid(11,6) = 0;
// inputGrid(11,7) = 0;
// inputGrid(11,8) = 0;
// inputGrid(11,9) = 1;
// inputGrid(11,10) = 1;
// inputGrid(11,11) = 1;
// inputGrid(11,12) = 1;
// inputGrid(11,13) = 0;
// inputGrid(11,14) = 0;
// inputGrid(11,15) = 1;
// inputGrid(12,0) = 0;
// inputGrid(12,1) = 1;
// inputGrid(12,2) = 1;
// inputGrid(12,3) = 1;
// inputGrid(12,4) = 1;
// inputGrid(12,5) = 0;
// inputGrid(12,6) = 1;
// inputGrid(12,7) = 1;
// inputGrid(12,8) = 0;
// inputGrid(12,9) = 1;
// inputGrid(12,10) = 0;
// inputGrid(12,11) = 1;
// inputGrid(12,12) = 1;
// inputGrid(12,13) = 1;
// inputGrid(12,14) = 1;
// inputGrid(12,15) = 0;
// inputGrid(13,0) = 1;
// inputGrid(13,1) = 0;
// inputGrid(13,2) = 1;
// inputGrid(13,3) = 0;
// inputGrid(13,4) = 0;
// inputGrid(13,5) = 0;
// inputGrid(13,6) = 0;
// inputGrid(13,7) = 1;
// inputGrid(13,8) = 0;
// inputGrid(13,9) = 0;
// inputGrid(13,10) = 0;
// inputGrid(13,11) = 1;
// inputGrid(13,12) = 0;
// inputGrid(13,13) = 0;
// inputGrid(13,14) = 0;
// inputGrid(13,15) = 1;
// inputGrid(14,0) = 1;
// inputGrid(14,1) = 1;
// inputGrid(14,2) = 1;
// inputGrid(14,3) = 1;
// inputGrid(14,4) = 0;
// inputGrid(14,5) = 0;
// inputGrid(14,6) = 0;
// inputGrid(14,7) = 1;
// inputGrid(14,8) = 1;
// inputGrid(14,9) = 1;
// inputGrid(14,10) = 0;
// inputGrid(14,11) = 1;
// inputGrid(14,12) = 1;
// inputGrid(14,13) = 0;
// inputGrid(14,14) = 1;
// inputGrid(14,15) = 1;
// inputGrid(15,0) = 0;
// inputGrid(15,1) = 1;
// inputGrid(15,2) = 1;
// inputGrid(15,3) = 1;
// inputGrid(15,4) = 0;
// inputGrid(15,5) = 1;
// inputGrid(15,6) = 1;
// inputGrid(15,7) = 0;
// inputGrid(15,8) = 0;
// inputGrid(15,9) = 1;
// inputGrid(15,10) = 0;
// inputGrid(15,11) = 1;
// inputGrid(15,12) = 1;
// inputGrid(15,13) = 1;
// inputGrid(15,14) = 0;
// inputGrid(15,15) = 0;
//	MYFILE.Close();
	//int cc = 0;
	// for(int h=0;h<height;h++) {
	// 	for(int w=0;w<width;w++) {
	// 		inputGrid(h,w) = rand()%2;
	// 		//inputGrid(h,w) = cc;
	// 		//cc++;
	// 	}
	// }
	// inputGrid(0,0) = 1;
	// inputGrid(0,1) = 1;
	// inputGrid(0,2) = 1;
	// inputGrid(0,3) = 1;
	// inputGrid(1,0) = 0;
	// inputGrid(1,1) = 0;
	// inputGrid(1,2) = 0;
	// inputGrid(1,3) = 1;
	// inputGrid(2,0) = 1;
	// inputGrid(2,1) = 0;
	// inputGrid(2,2) = 1;
	// inputGrid(2,3) = 0;
	// inputGrid(3,0) = 1;
	// inputGrid(3,1) = 0;
	// inputGrid(3,2) = 0;
	// inputGrid(3,3) = 1;
	// inputGrid(4,0) = 1;
	// inputGrid(4,1) = 1;
	// inputGrid(4,2) = 0;
	// inputGrid(4,3) = 0;
	// inputGrid(5,0) = 1;
	// inputGrid(5,1) = 1;
	// inputGrid(5,2) = 1;
	// inputGrid(5,3) = 0;
	// inputGrid(6,0) = 1;
	// inputGrid(6,1) = 1;
	// inputGrid(6,2) = 0;
	// inputGrid(6,3) = 0;
	// inputGrid(7,0) = 1;
	// inputGrid(7,1) = 0;
	// inputGrid(7,2) = 1;
	// inputGrid(7,3) = 1;
	// inputGrid(8,0) = 1;
	// inputGrid(8,1) = 1;
	// inputGrid(8,2) = 1;
	// inputGrid(8,3) = 0;
	// inputGrid(9,0) = 0;
	// inputGrid(9,1) = 0;
	// inputGrid(9,2) = 1;
	// inputGrid(9,3) = 1;
	// inputGrid(10,0) = 0;
	// inputGrid(10,1) = 1;
	// inputGrid(10,2) = 0;
	// inputGrid(10,3) = 0;
	// inputGrid(11,0) = 0;
	// inputGrid(11,1) = 1;
	// inputGrid(11,2) = 0;
	// inputGrid(11,3) = 0;
	// inputGrid(12,0) = 1;
	// inputGrid(12,1) = 0;
	// inputGrid(12,2) = 0;
	// inputGrid(12,3) = 1;
	// inputGrid(13,0) = 0;
	// inputGrid(13,1) = 0;
	// inputGrid(13,2) = 0;
	// inputGrid(13,3) = 1;
	// inputGrid(14,0) = 0;
	// inputGrid(14,1) = 0;
	// inputGrid(14,2) = 1;
	// inputGrid(14,3) = 0;
	// inputGrid(15,0) = 0;
	// inputGrid(15,1) = 1;
	// inputGrid(15,2) = 1;
	// inputGrid(15,3) = 1;



	Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(inputGrid, outputGrid, mask, arg);

	//hr_timer_t timer;
	switch(mode){
		case 0:
			//hrt_start(&timer);
			stencil.runSequential();
			for(int i = 0; i < outputGrid.getHeight(); i++){
				for(int j = 0; j < outputGrid.getWidth(); j++){
					printf("FinalOutput(%d,%d):%d\n", i, j, outputGrid(i, j));
				}
			}
			//hrt_stop(&timer);
			break;
        case 1:
            //hrt_start(&timer);
            stencil.runIterativeCPU(iterations,numCPUThreads);
            //for(int i = 0; i < outputGrid.getHeight();i++) {
             //   for(int j = 0; j < outputGrid.getWidth();j++) {
            //        printf("FinalOutput(%d,%d):%d\n",i,j,outputGrid(i,j));
            //    }
            //}
            //hrt_stop(&timer);
            break;

		default:
			//hrt_start(&timer);
			stencil.runCPU(numCPUThreads);
			//hrt_stop(&timer);
			break;

		#ifdef PSKEL_CUDA
		case 1:
			//hrt_start(&timer);
			stencil.runIterativeGPU(iterations, GPUBlockSize);	
			//hrt_stop(&timer);
			break;
		case 2:
			//hrt_start(&timer);
			stencil.runIterativeTilingGPU(iterations, width, tileHeight, 1, tileIterations, GPUBlockSize);	
			//hrt_stop(&timer);
			break;
		case 3:
			//hrt_start(&timer);
			stencil.runIterativeAutoGPU(iterations, GPUBlockSize);	
			//hrt_stop(&timer);
			break;
		#endif
	}
   /// for() {
     //   for() {
     //       printf("(%d)")
    //    }
    //}
	//cout << hrt_elapsed_time(&timer) << endl;
	inputGrid.hostFree();
	outputGrid.hostFree();
	return 0;
}
