#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

//#define PSKEL_SHARED_MASK
#include "../../include/PSkel.h"

#include "../util/hr_time.h"

using namespace std;
using namespace PSkel;

namespace PSkel{
	__parallel__ void stencilKernel(Array2D<float> input,Array2D<float> output,Mask2D<float> mask,float factor, size_t h, size_t w){
		
		output(h,w) = 0.25*(mask.get(0,input,h,w) + mask.get(1,input,h,w) + 
				mask.get(2,input,h,w) + mask.get(3,input,h,w) - 4*factor*factor );
		
	}
}

int main(int argc, char **argv){
	int width,height,iterations,GPUBlockSize,numCPUThreads,mode,tileHeight,tileIterations;
	if (argc != 9){
		printf ("Wrong number of parameters.\n", argv[0]);
		printf ("Usage: gol WIDTH HEIGHT ITERATIONS MODE GPUBLOCKS CPUTHREADS TILEHEIGHT TILEITERATIONS\n");
		exit (-1);
	}

	width = atoi (argv[1]);
	height = atoi (argv[2]);
	iterations = atoi (argv[3]);
	mode = atoi(argv[4]);
	GPUBlockSize = atoi(argv[5]);
	numCPUThreads = atoi(argv[6]);
	tileHeight = atoi(argv[7]);
	tileIterations = atoi(argv[8]);
	
	Array2D<float> inputGrid(width,height);
	Array2D<float> outputGrid(width,height);

	Mask2D<float> mask(4);
	
	mask.set(0,1,0,0);
	mask.set(1,-1,0,0);
	mask.set(2,0,1,0);
	mask.set(3,0,-1,0);
	
	float factor = 1.f/(float)width;

	//omp_set_num_threads(numCPUThreads);

	/* initialize the first timesteps */
	#pragma omp parallel for
    	for(size_t h = 0; h < inputGrid.getHeight(); h++){		
		for(size_t w = 0; w < inputGrid.getWidth(); w++){
			inputGrid(h,w) = 1.0 + w*0.1 + h*0.01;
		}
	}	
	
	Stencil2D<Array2D<float>,Mask2D<float>,float> stencil(inputGrid, outputGrid, mask, factor);

	hr_timer_t timer;
	switch(mode){
	case 0:
		hrt_start(&timer);
		stencil.runIterativeSequential(iterations);
		hrt_stop(&timer);
		break;
	case 1:
		hrt_start(&timer);
		stencil.runIterativeCPU(iterations,numCPUThreads);
		hrt_stop(&timer);
		break;
	case 2:
		hrt_start(&timer);
		stencil.runIterativeGPU(iterations, GPUBlockSize);	
		hrt_stop(&timer);
		break;
	case 3:
		hrt_start(&timer);
		stencil.runIterativeTilingGPU(iterations, width, tileHeight, 1, tileIterations, GPUBlockSize);	
		hrt_stop(&timer);
		break;
	case 4:
		hrt_start(&timer);
		stencil.runIterativeAutoGPU(iterations, GPUBlockSize);	
		hrt_stop(&timer);
		break;
	}
	cout << hrt_elapsed_time(&timer);
	inputGrid.hostFree();
	outputGrid.hostFree();
	return 0;
}
