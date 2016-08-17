//#define PSKEL_LOGMODE 1

#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>

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
	float h;
};

namespace PSkel{
	
	
__parallel__ void stencilKernel(Array2D<float> input,Array2D<float> output,Mask2D<float> mask,Arguments args, size_t i, size_t j){
	output(i,j) = 0.25f * ( mask.get(0, input, i, j) + mask.get(1, input, i, j) +  
				mask.get(2, input, i, j) + mask.get(3, input, i, j) - args.h );
						  ////mask.get(2, input, i, j) + mask.get(3, input, i, j) - args.h );
						  
	//output(i,j) = 0.2 * (input(i,j) + mask.get(0, input, i, j) + mask.get(1, input, i, j) + 
	//					 mask.get(2, input, i, j) + mask.get(3, input, i, j));
						 
	//output(i,j) = 0.25f * ( input(i-1,j) + (input(i,j-1) + input(i,j+1)) +
        //                    input(i+1,j) - args.h);
		
	}

}

int main(int argc, char **argv){
	int x_max, y_max, T_MAX, GPUBlockSizeX, GPUBlockSizeY, numCPUThreads;
	float GPUTime;

	if (argc != 9){
		printf ("Wrong number of parameters.\n");
		printf ("Usage: jacobi WIDTH HEIGHT ITERATIONS GPUPERCENT GPUBLOCKS_X GPUBLOCKS_Y CPUTHREADS OUTPUT_WRITE_FLAG\n");
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
	
	Array2D<float> inputGrid(x_max, y_max);
	Array2D<float> outputGrid(x_max, y_max);
	Mask2D<float> mask(4);
	
	mask.set(0,0,-1,0);
	mask.set(1,0,1,0);
	mask.set(2,1,0,0);
	mask.set(3,-1,0,0);
	
	Arguments args;
	//args.h = 1.f / (float) x_max;
	args.h = 4.f / (float) (x_max*x_max);
		
	omp_set_num_threads(numCPUThreads);

	/* initialize the first timesteps */
	#pragma omp parallel for
    	for(size_t h = 0; h < inputGrid.getHeight(); h++){		
		for(size_t w = 0; w < inputGrid.getWidth(); w++){
			inputGrid(h,w) = 1.0 + w*0.1 + h*0.01;
			outputGrid(h,w) = 0.0f;
		}
	}	
	
	hr_timer_t timer;
	hrt_start(&timer);
    
	//wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
	Stencil2D<Array2D<float>, Mask2D<float>, Arguments> jacobi(inputGrid, outputGrid, mask, args);
	
	//Runtime< Stencil2D<Array2D<float>, Mask2D<float>, Arguments> > stencilComponent(&jacobi);
	/*
	hrt_start(&timer);
	//stencil.runIterativeCPU(iterations, numCPUThreads);
	//stencil.runIterativeAutoGPU(T_MAX,GPUBlockSize);
	stencil.runIterativeGPU(T_MAX,GPUBlockSize);
	hrt_stop(&timer);
	cout << hrt_elapsed_time(&timer) << endl;
	*/
	
	#ifdef PSKEL_PAPI
		if(GPUTime < 1.0)
			PSkelPAPI::init(PSkelPAPI::CPU);
	#endif
	
	//stencil.runIterativePartition(T_MAX, 1.0-CPUTime, numCPUThreads, GPUBlockSize);
	//stencil.runIterativeAutoHybrid(T_MAX, CPUTime, numCPUThreads, GPUBlockSize);	
	
	//jacobi.runSequential();
	//jacobi.runIterativeCPU(T_MAX, numCPUThreads);
	
	
	if(GPUTime == 0.0){
		//jacobi.runIterativeCPU(T_MAX, numCPUThreads);
		#ifdef PSKEL_PAPI
			for(unsigned int i=0;i<NUM_GROUPS_CPU;i++){
				//cout << "Running iteration " << i << endl;
				jacobi.runIterativeCPU(T_MAX, numCPUThreads, i);	
			}
		#else
			//cout<<"Running Iterative CPU"<<endl;
			jacobi.runIterativeCPU(T_MAX, numCPUThreads);	
		#endif
	}
	else if(GPUTime == 1.0){
		jacobi.runIterativeGPU(T_MAX, GPUBlockSizeX, GPUBlockSizeY);
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
			PSkelPAPI::print_profile_values(PSkelPAPI::CPU);
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
