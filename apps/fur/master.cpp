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
#define DEBUG
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
                //printf("inputGrid(%d,%d) = %d;\n", h, w, inputGrid(h,w));
            #endif
        }
    }



//inputGrid(1,0) = 0;
//inputGrid(0,1) = 1;
//inputGrid(0,2) = 0;
//inputGrid(0,3) = 1;
//inputGrid(1,0) = 0;
//inputGrid(1,1) = 0;
//inputGrid(1,2) = 0;
//inputGrid(1,3) = 0;
//inputGrid(2,0) = 1;
//inputGrid(2,1) = 0;
//inputGrid(2,2) = 1;
//inputGrid(2,3) = 0;
//inputGrid(3,0) = 1;
//inputGrid(3,1) = 1;
//inputGrid(3,2) = 1;
//inputGrid(3,3) = 0;

// inputGrid(0,0) = 0;
// inputGrid(0,1) = 1;
// inputGrid(1,0) = 0;
// inputGrid(1,1) = 1;

//inputGrid(0,0) = 0;
//inputGrid(0,1) = 1;
//inputGrid(0,2) = 0;
//inputGrid(0,3) = 1;
//inputGrid(0,4) = 0;
//inputGrid(0,5) = 0;
//inputGrid(0,6) = 0;
//inputGrid(0,7) = 0;
//inputGrid(1,0) = 1;
//inputGrid(1,1) = 0;
//inputGrid(1,2) = 1;
//inputGrid(1,3) = 0;
//inputGrid(1,4) = 1;
//inputGrid(1,5) = 1;
//inputGrid(1,6) = 1;
//inputGrid(1,7) = 0;
//inputGrid(2,0) = 1;
//inputGrid(2,1) = 0;
//inputGrid(2,2) = 1;
//inputGrid(2,3) = 0;
//inputGrid(2,4) = 0;
//inputGrid(2,5) = 1;
//inputGrid(2,6) = 1;
//inputGrid(2,7) = 1;
//inputGrid(3,0) = 1;
//inputGrid(3,1) = 0;
//inputGrid(3,2) = 1;
//inputGrid(3,3) = 1;
//inputGrid(3,4) = 0;
//inputGrid(3,5) = 1;
//inputGrid(3,6) = 1;
//inputGrid(3,7) = 1;
//inputGrid(4,0) = 0;
//inputGrid(4,1) = 0;
//inputGrid(4,2) = 0;
//inputGrid(4,3) = 0;
//inputGrid(4,4) = 0;
//inputGrid(4,5) = 0;
//inputGrid(4,6) = 0;
//inputGrid(4,7) = 1;
//inputGrid(5,0) = 1;
//inputGrid(5,1) = 1;
//inputGrid(5,2) = 1;
//inputGrid(5,3) = 0;
//inputGrid(5,4) = 0;
//inputGrid(5,5) = 0;
//inputGrid(5,6) = 0;
//inputGrid(5,7) = 0;
//inputGrid(6,0) = 0;
//inputGrid(6,1) = 0;
//inputGrid(6,2) = 0;
//inputGrid(6,3) = 0;
//inputGrid(6,4) = 1;
//inputGrid(6,5) = 0;
//inputGrid(6,6) = 1;
//inputGrid(6,7) = 0;
//inputGrid(7,0) = 0;
//inputGrid(7,1) = 0;
//inputGrid(7,2) = 0;
//inputGrid(7,3) = 1;
//inputGrid(7,4) = 0;
//inputGrid(7,5) = 1;
//inputGrid(7,6) = 0;
//inputGrid(7,7) = 0;


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
	Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(inputGrid, outputGrid, mask);
	
	//Schedule computation to slaves
	stencil.scheduleMPPA("slave", nb_clusters, nb_threads, tilingHeight, tilingWidth, iterations, innerIterations);
	
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
