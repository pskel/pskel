#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <sched.h>
#include <unistd.h>
#include <mppaipc.h>
//#include <mppa/osconfig.h>
/**Problema com o omp.h**/

//#include "../../include/mppaStencil.h"
//#include "../../include/PSkelMask.h"
//#include "../../include/PSkelDefs.h"
//#include "../../include/mppaArray.h"
#define PSKEL_MPPA
#define MPPA_MASTER
#define ARGC_SLAVE 4
//#include "../../include/interface_mppa.h"
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
	int width, height, tilingHeight,iterations, pid, nb_clusters, nb_threads; //stencil size
	if(argc < 7 || argc > 7){
		printf("Usage: WIDTH, HEIGHT, TILINGHEIGHT, ITERATIONS, NUMBER CLUSTERS, NUMBER THREADS\n");
	}
	width = atoi(argv[1]);//4;
	height= atoi(argv[2]);//4;
  	tilingHeight = atoi(argv[3]);
	iterations = atoi(argv[4]);//1;
	nb_clusters = atoi(argv[5]);//2;
	nb_threads = atoi(argv[6]);//16;

	//Mask configuration
	Mask2D<int> mask;
	Arguments arg;

	Array2D<int> inputGrid(width,height);
	Array2D<int> outputGrid(width,height);

	srand(123456789);
	for(int h=0;h<height;h++) {
		for(int w=0;w<width;w++) {
			inputGrid(h,w) = rand()%2;
		}
	}

	Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(inputGrid, outputGrid, mask, arg);
	stencil.scheduleMPPA("slave", nb_clusters, nb_threads, tilingHeight);

	inputGrid.mppaFree();
	outputGrid.mppaFree();
	mppa_exit(0);
}
