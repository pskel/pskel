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


int CalcSize(int level){
	if (level == 1) {
    		return 3;
  	}
  	if (level >= 1) {
    		return CalcSize(level-1) + 2;
  	}
	return 0;
}



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
	int level = 1;
  	int power = 2;
  	int internCircle = pow(CalcSize(level), 2) - 1;
  	int externCircle = pow(CalcSize(2*level), 2) - 1 - internCircle;
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
	//Arguments arg;

	Array2D<int> inputGrid(width,height);
	Array2D<int> outputGrid(width,height);

	srand(123456789);
	// for(int h=0;h<height;h++) {
	// 	for(int w=0;w<width;w++) {
	// 		inputGrid(h,w) = rand()%2;
	// 	}
	// }
	inputGrid(0,0) = 0;
	inputGrid(0,1) = 1;
	inputGrid(0,2) = 1;
	inputGrid(0,3) = 1;
	inputGrid(1,0) = 0;
	inputGrid(1,1) = 0;
	inputGrid(1,2) = 1;
	inputGrid(1,3) = 0;
	inputGrid(2,0) = 1;
	inputGrid(2,1) = 0;
	inputGrid(2,2) = 1;
	inputGrid(2,3) = 1;
	inputGrid(3,0) = 0;
	inputGrid(3,1) = 0;
	inputGrid(3,2) = 0;
	inputGrid(3,3) = 1;
	//Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(inputGrid, outputGrid, mask, arg);
	Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(inputGrid, outputGrid, mask);
	stencil.scheduleMPPA("slave", nb_clusters, nb_threads, tilingHeight, iterations);

	mppa_exit(0);
}
