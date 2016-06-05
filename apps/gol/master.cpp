#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <sched.h>
#include <unistd.h>
#include <mppaipc.h>

#define PSKEL_MPPA
#define MPPA_MASTER
#define ARGC_SLAVE 4
#include "../../include/PSkel.h"

using namespace std;
using namespace PSkel;


struct Arguments
{
	int dummy;
};

int main(int argc, char **argv){ 
	int width, height, tilingHeight, iterations, innerIterations, pid, nb_clusters, nb_threads; //stencil size
	if(argc == 8){
		printf("Usage: WIDTH, HEIGHT, TILINGHEIGHT, ITERATIONS, INNERITERATIONS, NUMBER CLUSTERS, NUMBER THREADS\n");
		mppa_exit(0);
	}
	width = atoi(argv[1]);
	height= atoi(argv[2]);
  	tilingHeight = atoi(argv[3]);
	iterations = atoi(argv[4]);
	innerIterations = atoi(argv[5]);
	nb_clusters = atoi(argv[6]);
	nb_threads = atoi(argv[7]);
	//Mask configuration
	Mask2D<int> mask(8);
	mask.set(0,-1,-1);	mask.set(1,-1,0);	mask.set(2,-1,1);
	mask.set(3,0,-1);				mask.set(4,0,1);
	mask.set(5,1,-1);	mask.set(6,1,0);	mask.set(7,1,1);
	
	//Arguments arg;
	Array2D<int> inputGrid(width,height);
	Array2D<int> outputGrid(width,height);
	
	srand(123456789);
	for(int h=0;h<height;h++) {
		for(int w=0;w<width;w++) {
			inputGrid(h,w) = rand()%2;
			//printf("inputGrid(%d,%d) = %d;\n", h, w, inputGrid(h,w));
		}
	}


	//Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(inputGrid, outputGrid, mask, arg);
	Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(inputGrid, outputGrid, mask);
	stencil.scheduleMPPA("slave", nb_clusters, nb_threads, tilingHeight, iterations, innerIterations);

	mppa_exit(0);
}
