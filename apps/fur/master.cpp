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
	int externCircle;
	int internCircle;
	float power;
};



int main(int argc, char **argv){ 
	int width, height, tilingHeight, iterations, pid, nb_clusters, nb_threads; //stencil size
	if(argc < 7 || argc > 7){
		printf("Usage: WIDTH, HEIGHT, TILINGHEIGHT, ITERATIONS, NUMBER CLUSTERS, NUMBER THREADS\n");
	}
	width = atoi(argv[1]);
	height= atoi(argv[2]);
  	tilingHeight = atoi(argv[3]);
	iterations = atoi(argv[4]);
	nb_clusters = atoi(argv[5]);
	nb_threads = atoi(argv[6]);

	//Mask configuration
	// Mask2D<int> mask;
	// Arguments arg;

	Array2D<int> inputGrid(width,height);
	Array2D<int> outputGrid(width,height);

	srand(123456789);
	for(int h=0;h<height;h++) {
		for(int w=0;w<width;w++) {
			inputGrid(h,w) = rand()%2;
		}
	}

	//Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(inputGrid, outputGrid, mask, arg);
	Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(inputGrid, outputGrid);
	stencil.scheduleMPPA("slave", nb_clusters, nb_threads, tilingHeight, iterations);

	mppa_exit(0);
}
