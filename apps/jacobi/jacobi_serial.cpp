#include <fstream>
#include <string>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <stdlib.h>

using namespace std;

void stencilKernel(float *input, float *output, int width, int height, int T_MAX,float alpha, float beta){
	for (int t = 0; t < T_MAX; t++){
		for (int y = 1; y < height - 1; y++){
    		for (int x = 1; x < width - 1; x++){
                output[y*width+x] = alpha * input[y*width + x] +
									beta * (input[(y+1)*width + x] + input[(y-1)*width + x] +
											input[y*width + (x+1)] + input[y*width + (x-1)])
									- 4 * beta * beta;
    		}
    	}   
    	
    	//swap(output,input)
    	for (int y = 1; y < height - 1; y++){
    		for (int x = 1; x < width - 1; x++){
				input[y*width+x] = output[y*width+x];
			}
		}
	}
	//swap(output,input);
	//if(T_MAX%2==0)
	//   memcpy(input,output,width*height*sizeof(int));
}

int main(int argc, char **argv){
	int width;
	int height;
	int T_MAX;

	float *inputGrid;
	float *outputGrid;
	float alpha,beta;

	if (argc != 4){
		printf ("Wrong number of parameters.\n");
		printf ("Usage: gol WIDTH HEIGHT ITERATIONS\n");
		exit (-1);
	}

	width = atoi (argv[1]);
	height = atoi (argv[2]);
	T_MAX = atoi (argv[3]);

	alpha = 0.25/(float) width;
    beta = 1.0/(float) height;

	inputGrid = (float*) malloc(width*height*sizeof(float));
	outputGrid = (float*) malloc(width*height*sizeof(float));

	for(int j=0;j<height;j++) {
		for(int i=0;i<width;i++) {
			inputGrid[j*width + i] = 1. + i*0.1 + j*0.01;
		}
	}
  
	#pragma pskel stencil dim2d(width,height) inout(inputGrid, outputGrid) iterations(T_MAX) device(gpu)
	stencilKernel(inputGrid, outputGrid,width,height,T_MAX,alpha,beta);
  
	return 0;
}
