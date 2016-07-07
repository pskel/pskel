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

void stencilKernel(int *input, int *output, int width, int height, int T_MAX){
	for(int t=0;t<T_MAX;t++){
		for(int j=0;j<height;j++){
			for(int i=0;i<width;i++){
				int neighbors=0;
				for(int y=-1;y<=1;y++){
					for(int x=-1;x<=1;x++){
						if( (i+x)>=0 && (i+x)<width && (j+y)>=0 && (j+y)<height ){
							neighbors += input[(j+y)*width + (i+x)];
						}
					}
				}
				if(neighbors == 3 || (input[j*width + i]==1 && neighbors == 2)){
					output[j*width + i] = 1;
				}
				else{
					output[j*width + i] = 0;
				}
			}
		}
		swap(output,input);
	}
   swap(output,input);
	if(T_MAX%2==0)
	   memcpy(input,output,width*height*sizeof(int));
}

int main(int argc, char **argv){
	int x_max;
	int y_max;
	int T_MAX;

	int *inputGrid;
	int *outputGrid;

	if (argc != 4){
		printf ("Wrong number of parameters.\n");
		printf ("Usage: gol WIDTH HEIGHT ITERATIONS\n");
		exit (-1);
	}

	x_max = atoi (argv[1]);
	y_max = atoi (argv[2]);
	T_MAX = atoi (argv[3]);

	inputGrid = (int*) malloc(x_max*y_max*sizeof(int));
	outputGrid = (int*) malloc(x_max*y_max*sizeof(int));

	srand(123456789);
	for(int j=0;j<y_max;j++) {
		for(int i=0;i<x_max;i++) {
			inputGrid[j*x_max + i] = rand()%2;
		}
	}
  
	#pragma pskel stencil dim2d(x_max, y_max) \
	    inout(inputGrid, outputGrid) \
	    iterations(T_MAX) device(cpu)
	stencilKernel(inputGrid, outputGrid,x_max,y_max,T_MAX);
  
	return 0;
}
