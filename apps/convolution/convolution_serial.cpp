#define PSKEL_TBB
#define PSKEL_CUDA

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "PSkel.h"
#include "hr_time.h"

#define MASK_RADIUS 2
#define MASK_WIDTH  5

using namespace std;
using namespace PSkel;

//*******************************************************************************************
// CONVOLUTION
//*******************************************************************************************

namespace PSkel{
	__parallel__ void stencilKernel(Array2D<float> input, Array2D<float> output, Mask2D<float> mask, int null, size_t i,size_t j){
		//float accum = 0.0f;
		/*for(int n=0;n<mask.size;n++){
			accum += mask.get(n,input,i,j) * mask.getWeight(n);
		}
		output(i,j)= accum;
		*/
		output(i,j) = mask.get(0,input,i,j) * mask.getWeight(0) +
					  mask.get(1,input,i,j) * mask.getWeight(1) +
					  mask.get(2,input,i,j) * mask.getWeight(2) +
					  mask.get(3,input,i,j) * mask.getWeight(3) +
					  mask.get(4,input,i,j) * mask.getWeight(4) +
					  mask.get(5,input,i,j) * mask.getWeight(5) +
					  mask.get(6,input,i,j) * mask.getWeight(6) +
					  mask.get(7,input,i,j) * mask.getWeight(7) +
					  mask.get(8,input,i,j) * mask.getWeight(8) +
					  mask.get(9,input,i,j) * mask.getWeight(9) +
					  mask.get(10,input,i,j) * mask.getWeight(10) +
					  mask.get(11,input,i,j) * mask.getWeight(11) +
					  mask.get(12,input,i,j) * mask.getWeight(12) +
					  mask.get(13,input,i,j) * mask.getWeight(13) +
					  mask.get(14,input,i,j) * mask.getWeight(14) +
					  mask.get(15,input,i,j) * mask.getWeight(15) +
					  mask.get(16,input,i,j) * mask.getWeight(16) +
					  mask.get(17,input,i,j) * mask.getWeight(17) +
					  mask.get(18,input,i,j) * mask.getWeight(18) +
					  mask.get(19,input,i,j) * mask.getWeight(19) +
					  mask.get(20,input,i,j) * mask.getWeight(20) +
					  mask.get(21,input,i,j) * mask.getWeight(21) +
					  mask.get(22,input,i,j) * mask.getWeight(22) +
					  mask.get(23,input,i,j) * mask.getWeight(23) +
					  mask.get(24,input,i,j) * mask.getWeight(24); 
	}
}//end namespace

//*******************************************************************************************
// MAIN
//*******************************************************************************************

int main(int argc, char **argv){	
		
	Mask2D<float> mask(25);
	float GPUTime;
	int GPUBlockSize, numCPUThreads,x_max,y_max;
	
	if (argc != 8){
		printf ("Wrong number of parameters.\n");
		//printf ("Usage: convolution INPUT_IMAGE ITERATIONS GPUTIME GPUBLOCKS CPUTHREADS OUTPUT_WRITE_FLAG\n");
		printf ("Usage: convolution WIDTH HEIGHT ITERATIONS GPUTIME GPUBLOCKS CPUTHREADS OUTPUT_WRITE_FLAG\n");
		printf ("You entered: ");
		for(int i=0; i< argc;i++){
			printf("%s ",argv[i]);
		}
		printf("\n");
		exit (-1);
	}
	
	x_max = atoi(argv[1]);
	y_max = atoi(argv[2]);
	int T_MAX = atoi(argv[3]);
	GPUTime = atof(argv[4]);
	GPUBlockSize = atoi(argv[5]);
	numCPUThreads = atoi(argv[6]);
	int writeToFile = atoi(argv[7]);
	
	Array2D<float> inputGrid(x_max, y_max);
	Array2D<float> outputGrid(x_max, y_max);	

	mask.set(0,-2,2,0.0);	mask.set(1,-1,2,0.0);	mask.set(2,0,2,0.0);	mask.set(3,1,2,0.0);	mask.set(4,2,2,0.0);
	mask.set(5,-2,1,0.0);	mask.set(6,-1,1,0.0);	mask.set(7,0,1,0.1);	mask.set(8,1,1,0.0);	mask.set(9,2,1,0.0);
	mask.set(10,-2,0,0.0);	mask.set(11,-1,0,0.1);	mask.set(12,0,0,0.2);	mask.set(13,1,0,0.1);	mask.set(14,2,0,0.0);
	mask.set(15,-2,-1,0.0);	mask.set(16,-1,-1,0.0);	mask.set(17,0,-1,0.1);	mask.set(18,1,-1,0.0);	mask.set(19,2,-1,0.0);
	mask.set(20,-2,-2,0.0);	mask.set(21,-1,-2,0.0);	mask.set(22,0,-2,0.0);	mask.set(23,1,-2,0.0);	mask.set(24,2,-2,0.0);
	
	#pragma omp parallel
	{
		srand(1234 ^ omp_get_thread_num());
		#pragma omp for
		for (int x = 0; x < x_max; x++){
			for (int y = 0; y < y_max; y++){		
				inputGrid(x,y) = ((float)(rand() % 255))/255;
			}
		}
	}
	
	Stencil2D<Array2D<float>, Mask2D<float>, int> stencil(inputGrid, outputGrid, mask, 0);
	hr_timer_t timer;
	
	
	#ifdef PSKEL_PAPI
		if(GPUTime < 1.0)
			PSkelPAPI::init(PSkelPAPI::CPU);
	#endif
	
	hrt_start(&timer);
	
	if(GPUTime == 0.0){
		stencil.runIterativeCPU(T_MAX, numCPUThreads);
	}
	else if(GPUTime == 1.0){
		stencil.runIterativeGPU(T_MAX, GPUBlockSize);
	}
	else{
		stencil.runIterativePartition(T_MAX, GPUTime, numCPUThreads,GPUBlockSize);
	}
	
	hrt_stop(&timer);

	#ifdef PSKEL_PAPI
		if(GPUTime < 1.0){
			PSkelPAPI::print_profile_values(PSkelPAPI::CPU);
			PSkelPAPI::shutdown();
		}
	#endif
	
	cout << "Exec_time\t" << hrt_elapsed_time(&timer) << endl;

	if(writeToFile == 1){
		cout.precision(12);
		cout<<"INPUT"<<endl;
		for(int i=10; i<y_max;i+=10){
			cout<<"("<<i<<","<<i<<") = "<<inputGrid(i,i)<<"\t\t("<<x_max-i<<","<<y_max-i<<") = "<<inputGrid(x_max-i,y_max-i)<<endl;
		}
		cout<<endl;
		
		cout<<"OUTPUT"<<endl;
		for(int i=10; i<y_max;i+=10){
			cout<<"("<<i<<","<<i<<") = "<<outputGrid(i,i)<<"\t\t("<<x_max-i<<","<<y_max-i<<") = "<<outputGrid(x_max-i,y_max-i)<<endl;
		}
		cout<<endl;
	}
	return 0;
}


