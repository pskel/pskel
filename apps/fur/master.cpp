#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <sched.h>
#include <unistd.h>
#include <mppaipc.h>
#include <iostream>
#include <fstream>
#include <string>

#define PSKEL_MPPA
#define MPPA_MASTER
#define ARGC_SLAVE 4
// #define DEBUG
// #define BUG_TEST
// #define PRINT_OUT
// #define TIME_EXEC
// #define TIME_SEND
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
	int width, height, tilingHeight, tilingWidth, iterations, innerIterations, pid, nb_clusters, nb_threads; //stencil size
	if(argc != 9){
		printf ("Wrong number of parameters.\n");
		printf("Usage: WIDTH HEIGHT TILING_HEIGHT TILING_WIDTH ITERATIONS INNER_ITERATIONS NUMBER_CLUSTERS NUMBER_THREADS\n");
		mppa_exit(-1);
	}
	// for(int i = 0; i < argc; i++) {
	// 	cout << "Indice: "<< i << "valor: " << atoi(argv[i]) << endl;
	// }
	//Stencil configuration
	width = atoi(argv[1]);
	printf("width: %d\n", width);
	height = atoi(argv[2]);
	printf("height: %d\n", height);

  	tilingHeight = atoi(argv[3]);
	printf("tilingHeight: %d\n", tilingHeight);

  	tilingWidth = atoi(argv[4]);
	printf("tilingWidth: %d\n", tilingWidth);

	iterations = atoi(argv[5]);
	printf("iterations: %d\n", iterations);

	innerIterations = atoi(argv[6]);
	printf("innerIterations: %d\n", innerIterations);

	nb_clusters = atoi(argv[7]);
	printf("nb_clusters: %d\n", nb_clusters);

	nb_threads = atoi(argv[8]);
	printf("nb_threads: %d\n", nb_threads);


	//Mask configuration
	int level = 1;
  	int power = 2;
  	int count = 0;
  	int internCircle = pow(CalcSize(level), 2) - 1;
  	int externCircle = pow(CalcSize(2*level), 2) - 1 - internCircle;
  	int size = internCircle + externCircle;


	Array2D<int> inputGrid(width,height);
	Array2D<int> outputGrid(width,height);
  	Mask2D<int> mask(size);

	for (int x = (level-2*level); x <= level; x++) {
		for (int y = (level-2*level); y <= level; y++) {
			if (x != 0 || y != 0) {
				mask.set(count, x, y);
				//cout<<count<<": ("<<x<<","<<y<<")"<<endl;
				count++;
			}
		}
  	}

  	for (int x = (2*level-4*level); x <= 2*level; x++) {
		for (int y = (2*level-4*level); y <= 2*level; y++) {
			if (x != 0 || y != 0) {
				if (!(x <= level && x >= -1*level && y <= level && y >= -1*level)) {
					mask.set(count, x, y);
					//cout<<count<<": ("<<x<<","<<y<<")"<<endl;
					count++;
				}
			}
		}
  	}

	//Arguments arg;

	count = 0;
	srand(1234);
	for(int h=0;h<height;h++) {
	    for(int w=0;w<width;w++) {
	    	inputGrid(h,w) = rand()%2;
	        #ifdef DEBUG
//                printf("In position %d, %d we have %d\n", h, w, inputGrid(h,w));
                // printf("inputGrid(%d,%d) = %d;\n", h, w, inputGrid(h,w));
            #endif
        }
    }



	//Print input
	/*cout.precision(12);
	cout<<"INPUT"<<endl;
	for(int i=0; i<width;i+=25){
		for(int j=0; j<height;j+=25){
			cout<<"("<<i      <<","<<j       <<") = "<<inputGrid(i,j)<<"\t\t";
			cout<<"("<<width-1-i<<","<<height-1-j<<") = "<<inputGrid(width-1-i,height-1-j)<<endl;
		}
	}
	cout<<endl;
	*/
	//Instantiate Stencil 2D
	Arguments arg;

	Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(inputGrid, outputGrid, mask, arg);
	struct timeval start=mppa_master_get_time();

	//Schedule computation to slaves

	stencil.scheduleMPPA("slave", nb_clusters, nb_threads, tilingHeight, tilingWidth, iterations, innerIterations);

	struct timeval end=mppa_master_get_time();
	cout<<"Master Time: " << mppa_master_diff_time(start,end) << endl;
	//Print results
	/*cout<<"OUTPUT"<<endl;
	for(int i=0;i<width;i+=25){
		for(int j=0; j<height;j+=25){
			cout<<"("<<i    <<","<<j       <<") = "<<outputGrid(i,j)<<"\t\t";
			cout<<"("<<width-1-i<<","<<height-1-j<<") = "<<outputGrid(width-1-i,height-1-j)<<endl;
		}
	}
	*/

	stencil.~Stencil2D();
	mppa_exit(0);
}
