#include <fstream>
#include <string>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>  
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <stdlib.h>

#include "hr_time.h"
#include <omp.h>
#include <openacc.h>

using namespace std;

inline void stencilKernel(float* __restrict__ input, float* __restrict__ output, int width, int height, int T_MAX,float alpha, float beta){
	#pragma acc enter data copyin(input[0:width*height]) create(output[0:width*height])
	{
	for (int t = 0; t < T_MAX; t++){
		#pragma acc kernels //loop gang worker vector_length(32) num_workers(32)
		{
		#pragma acc loop independent //gang(32) vector(16) 
		for (int y = 0; y < height ; y++){
			#pragma acc loop independent //gang(16) vector(32) 
			for (int x = 0; x < width ; x++){
				output[y*width+x] = 0.25f * (input[(y+1)*width + x] + input[(y-1)*width + x] +
						             input[y*width + (x+1)] + input[y*width + (x-1)] - beta);
			}
		}  
		
		//swap data
		if(T_MAX > 1 && t<T_MAX-1){
			#pragma acc loop independent //gang(32) vector(16) 
			for (int y = 0; y < height ; y++){
				#pragma acc loop independent //independent //gang(16) vector(32)
				for (int x = 0; x < width ; x++){
					input[y*width+x] = output[y*width+x];
				}
			}
		}
		}
	}
	}
	#pragma acc exit data
}

int main(int argc, char **argv){
	int width;
	int height;
	int T_MAX;
	int verbose;

	float *inputGrid;
	float *outputGrid;
	float alpha,beta;

	if (argc != 5){
		printf ("Wrong number of parameters.\n");
		printf ("Usage: jacobi WIDTH HEIGHT ITERATIONS VERBOSE\n");
		exit (-1);
	}

	width = atoi (argv[1]);
	height = atoi (argv[2]);
	T_MAX = atoi (argv[3]);
	verbose = atoi (argv[4]);

	alpha = 0.25/(float) width;
    	beta = 4.0f/(float) (height*height);

	inputGrid = (float*) malloc(width*height*sizeof(float));
	outputGrid = (float*) malloc(width*height*sizeof(float));

	#pragma omp parallel for
	for(int j=0;j<height;j++) {
		for(int i=0;i<width;i++) {
			inputGrid[j*width + i] = 1. + i*0.1 + j*0.01;
			outputGrid[j*width + i] = 0.0f;
		}
	}
  		
	hr_timer_t timer;
	hrt_start(&timer);
	//#pragma pskel stencil dim2d(width,height) inout(inputGrid, outputGrid) iterations(T_MAX) device(cpu)
	stencilKernel(inputGrid, outputGrid,width,height,T_MAX,alpha,beta);
	
	hrt_stop(&timer);
	
	cout << "Exec_time\t" << hrt_elapsed_time(&timer) << endl;
	
	if(verbose){		
		cout<<setprecision(6);
		cout<<fixed;
		cout<<"INPUT"<<endl;
		for(int i=0; i<width/10;i+=10){
			cout<<"("<<i<<","<<i<<") = "<<inputGrid[i*width+i]<<"\t\t("<<width-i<<","<<height-i<<") = "<<inputGrid[(height-i)*width+(width-i)]<<endl;
		}
		cout<<endl;
		
		cout<<"OUTPUT"<<endl;
		//for(int i=0; i<width/10;i+=10){
		//	cout<<"("<<i<<","<<i<<") = "<<outputGrid[i*width+i]<<"\t\t("<<width-i<<","<<height-i<<") = "<<outputGrid[(height-i)*width+(width-i)]<<endl;
		//}
		//cout<<endl;
		
		for(size_t h = 0; h < height; h++){		
			for(size_t w = 0; w < width; w++){
				cout<<outputGrid[h*width + w]<<"\t\t";
			}
			cout<<endl;
		}
	}
  
	return 0;
}
