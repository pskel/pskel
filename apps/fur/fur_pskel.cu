//#define PSKEL_LOGMODE 1

#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <math.h>

//#define PSKEL_SHARED_MASK
#define PSKEL_OMP
#define PSKEL_CUDA
//#define PSKEL_PAPI
//#define PSKEL_PAPI_DEBUG

#include "PSkel.h"
#include "hr_time.h"
//#include "wb.h"

using namespace std;
using namespace PSkel;

struct Arguments{
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

namespace PSkel{
	
	
__parallel__ void stencilKernel(Array2D<int> input,Array2D<int> output,Mask2D<int> mask,Arguments arg, size_t h, size_t w){
		int numberA = 0;
		int numberI = 0;
		for (int z = 0; z < mask.size; z++) {
			if(z < arg.internCircle) {
				numberA += mask.get(z, input, h, w);
			} else {
				numberI += mask.get(z, input, h, w);
			}
		}
		float totalPowerI = numberI*(arg.power);// The power of Inhibitors
		
		if(numberA - totalPowerI < 0) {
			output(h,w) = 0; //without color and inhibitor
		} else if(numberA - totalPowerI > 0) {
			output(h,w) = 1; //with color and active
		} else {
			output(h,w) = input(h,w); //doesn't change
		}
	}
}

int main(int argc, char **argv){
	int x_max, y_max, T_MAX, GPUBlockSizeX, GPUBlockSizeY, numCPUThreads;
	float GPUTime;

	if (argc != 9){
		printf ("Wrong number of parameters.\n");
		printf ("Usage: fur WIDTH HEIGHT ITERATIONS GPUPERCENT GPUBLOCKS_X GPUBLOCKS_Y CPUTHREADS OUTPUT_WRITE_FLAG\n");
		exit (-1);
	}

	x_max = atoi (argv[1]);
	y_max = atoi (argv[2]);
	T_MAX=atoi(argv[3]);
	GPUTime = atof(argv[4]);
	GPUBlockSizeX = atoi(argv[5]);
	GPUBlockSizeY = atoi(argv[6]);
	numCPUThreads = atoi(argv[7]);
	int writeToFile = atoi(argv[8]);

	//Mask configuration
	int level = 1;
  	int power = 2;
  	int count = 0;
  	int internCircle = pow(CalcSize(level), 2) - 1;
  	int externCircle = pow(CalcSize(2*level), 2) - 1 - internCircle;
  	int size = internCircle + externCircle;
	
	
	Array2D<int> inputGrid(x_max, y_max);
	Array2D<int> outputGrid(x_max, y_max);
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
	
	Arguments args;
	args.power = power;
	args.internCircle = internCircle;
	args.externCircle = externCircle;

		
	omp_set_num_threads(numCPUThreads);

	/* initialize the first timesteps */
	#pragma omp parallel for
    	for(size_t h = 0; h < inputGrid.getHeight(); h++){		
		for(size_t w = 0; w < inputGrid.getWidth(); w++){
			inputGrid(h,w) = rand()%2;
		}
	}	
	
	hr_timer_t timer;
	hrt_start(&timer);
    
	Stencil2D<Array2D<int>, Mask2D<int>, Arguments> fur(inputGrid, outputGrid, mask, args);
	
	#ifdef PSKEL_PAPI
		if(GPUTime < 1.0)
			PSkelPAPI::init(PSkelPAPI::RAPL);
	#endif
	
	if(GPUTime == 0.0){
		//jacobi.runIterativeCPU(T_MAX, numCPUThreads);
		//#ifdef PSKEL_PAPI
		//	for(unsigned int i=0;i<NUM_GROUPS_CPU;i++){
				//cout << "Running iteration " << i << endl;
		//		fur.runIterativeCPU(T_MAX, numCPUThreads, i);	
		//	}
		//#else
			//cout<<"Running Iterative CPU"<<endl;
			#ifdef PSKEL_PAPI
				PSkelPAPI::papi_start(PSkelPAPI::RAPL,0);
			#endif	
				fur.runIterativeCPU(T_MAX, numCPUThreads);	

			#ifdef PSKEL_PAPI
				PSkelPAPI::papi_stop(PSkelPAPI::RAPL,0);
			#endif
		//#endif
	}
	else if(GPUTime == 1.0){
		fur.runIterativeGPU(T_MAX, GPUBlockSizeX, GPUBlockSizeY);
	}
	else{
		//jacobi.runIterativePartition(T_MAX, GPUTime, numCPUThreads,GPUBlockSize);
		/*
        #ifdef PSKEL_PAPI
			for(unsigned int i=0;i<NUM_GROUPS_CPU;i++){
				jacobi.runIterativePartition(T_MAX, GPUTime, numCPUThreads,GPUBlockSizeX,i);
			}
		#else
			jacobi.runIterativePartition(T_MAX, GPUTime, numCPUThreads,GPUBlockSizeX);
		#endif
        */
	}
	
	
	//wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
	hrt_stop(&timer);

	#ifdef PSKEL_PAPI
		if(GPUTime < 1.0){
			PSkelPAPI::print_profile_values(PSkelPAPI::RAPL);
			PSkelPAPI::shutdown();
		}
	#endif
	
	cout << "Exec_time\t" << hrt_elapsed_time(&timer) << endl;

	if(writeToFile == 1){
		/*stringstream outputFile;
		outputFile << "output_" <<x_max << "_" << y_max << "_" << T_MAX << "_" << GPUTime << "_" << GPUBlockSize <<"_" << numCPUThreads << ".txt";
		string out = outputFile.str();
		
		ofstream ofs(out.c_str(), std::ofstream::out);
		
		ofs.precision(6);
		
		for (size_t h = 1; h < outputGrid.getHeight()-1; h++){		
			for (size_t w = 1; w < outputGrid.getWidth()-1; w++){
				ofs<<outputGrid(h,w)<<" ";
			}
			ofs<<endl;
		}*/		
		
		cout<<setprecision(6);
		cout<<fixed;
		cout<<"INPUT"<<endl;
		for(int i=0; i<y_max/10;i+=10){
			cout<<"("<<i<<","<<i<<") = "<<inputGrid(i,i)<<"\t\t("<<x_max-i<<","<<y_max-i<<") = "<<inputGrid(x_max-i,y_max-i)<<endl;
		}
		cout<<endl;
		
		cout<<"OUTPUT"<<endl;
		//for(int i=0; i<y_max/10;i+=10){
		//	cout<<"("<<i<<","<<i<<") = "<<outputGrid(i,i)<<"\t\t("<<x_max-i<<","<<y_max-i<<") = "<<outputGrid(x_max-i,y_max-i)<<endl;
		//}
		//cout<<endl;
		
		for(size_t h = 0; h < outputGrid.getHeight(); h++){		
			for(size_t w = 0; w < outputGrid.getWidth(); w++){
				cout<<outputGrid(h,w)<<"\t\t";
			}
			cout<<endl;
		}
	}
	return 0;
}
