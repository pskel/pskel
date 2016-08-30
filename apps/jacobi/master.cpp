#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <sched.h>
#include <unistd.h>
#include <mppaipc.h>

#define PSKEL_MPPA
#define MPPA_MASTER
// #define DEBUG
//#define BUG_TEST
// #define PRINT_OUT
#define TIME_EXEC
#define TIME_SEND
#define ARGC_SLAVE 4
#include "../../include/PSkel.h"

using namespace std;
using namespace PSkel;

int main(int argc, char **argv){ 
	int width, height, tilingHeight, tilingWidth, iterations, innerIterations, pid, nb_clusters, nb_threads; //stencil size
	if(argc != 8){
		printf ("Wrong number of parameters.\n");
		printf("Usage: WIDTH HEIGHT TILING_HEIGHT TILING_WIDTH ITERATIONS INNER_ITERATIONS NUMBER_CLUSTERS NUMBER_THREADS\n");
		mppa_exit(-1);
	}
	
	//Stencil configuration
	width = atoi(argv[1]);
	height = atoi(argv[2]);
  	tilingHeight = atoi(argv[3]);
  	tilingWidth = atoi(argv[4]);
	iterations = atoi(argv[5]);
	innerIterations = atoi(argv[6]);
	nb_clusters = atoi(argv[7]);
	nb_threads = atoi(argv[8]);
	
	//Mask configuration	
	
	Array2D<float> inputGrid(width,height);
	Array2D<float> outputGrid(width,height);

	float factor = 1.f/(float)width;

	Mask2D<float> mask(4);
	
	mask.set(0,1,0,0);
	mask.set(1,-1,0,0);
	mask.set(2,0,1,0);
	mask.set(3,0,-1,0);


	for(size_t h=0;h<inputGrid.getHeight();h++) {
		for(size_t w=0;w<inputGrid.getWidth();w++) {
			//inputGrid(h,w) = 1.0 + w*0.1 + h*0.01;
			inputGrid(h,w) = h*inputGrid.getWidth() + w;
		    //printf("inputGrid(%d,%d) = %f;\n", h, w, inputGrid(h,w));
		}
	}


	//Instantiate Stencil 2D
	Stencil2D<Array2D<float>, Mask2D<float>, float> stencil(inputGrid, outputGrid, mask, factor);
	
	//Schedule computation to slaves
	stencil.scheduleMPPA("slave", nb_clusters, nb_threads, tilingHeight, tilingWidth, iterations, innerIterations);
	
	mppa_exit(0);
}
