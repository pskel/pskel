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

namespace PSkel{
	
	
__parallel__ void stencilKernel(Array2D<bool> input, Array2D<bool> output,
                  Mask2D<bool> mask, short args, size_t i, size_t j){
    //int neighbors =  input(i-1,j-1) + input(i-1,j) + input(i-1,j+1)  +
    //                 input(i+1,j-1) + input(i+1,j) + input(i+1,j+1)  + 
    //                 input(i,j-1)   + input(i,j+1) ; 

    //int neighbors = mask.get(0,input,i,j) + mask.get(1,input,i,j) + mask.get(2,input,i,j) +
    //	              mask.get(3,input,i,j) + mask.get(4,input,i,j) + mask.get(5,input,i,j) +
    //		      mask.get(6,input,i,j) + mask.get(7,input,i,j);

                      
    
    int neighbors = 0;
    int height=input.getHeight();
    int width=input.getWidth();
    
    if ( (j == 0) && (i == 0) ) { //	Corner 1	
        neighbors = input(i+1,j) + input(i,j+1) + input (i+1,j+1);
    }	//	Corner 2	
    else if ((j == 0) && (i == width-1)) {
        neighbors = input(i-1,j) + input(i,j+1) + input(i-1,j+1);
    }	//	Corner 3	
    else if ((j == height-1) && (i == width-1)) {
        neighbors = input(i-1,j) + input(i,j-1) + input(i-1,j-1);
    }	//	Corner 4	
    else if ((j == height-1) && (i == 0)) {
        neighbors = input(i,j-1) + input(i+1,j) + input(i+1,j-1);
    }	//	Edge 1	
    else if (j == 0) {
        neighbors = input(i-1,j) + input(i+1,j) + input(i-1,j+1) + input(i,j+1) + input(i+1,j+1);
    }	//	Edge 2	
    else if (i == width-1) {
        neighbors = input(i,j-1) +  input(i-1,j-1) + input(i-1,j) +  input(i-1,j+1) + input(i,j+1);
    }	//Edge 3	
    else if (j == height-1) {
        neighbors = input(i-1,j-1) + input(i,j-1) + input(i+1,j-1) + input(i-1,j) + input(i+1,j);
    }	//Edge 4
    else if (i == 0) {
        neighbors = input(i,j-1) + input(i+1,j-1) + input(i+1,j) + input(i,j+1) + input(i+1,j+1);
    }	//Inside the grid
    else {
        neighbors =  input(i-1,j-1) + input(i-1,j) + input(i-1,j+1)  +
                     input(i+1,j-1) + input(i+1,j) + input(i+1,j+1)  + 
                     input(i,j-1)   + input(i,j+1) ;
    }
                  
    output(i,j) = (neighbors == 3 || (input(i,j) == 1 && neighbors == 2))?1:0;
        
    }
}

int main(int argc, char **argv){
	int width, height, T_MAX, GPUBlockSizeX, GPUBlockSizeY, numCPUThreads,verbose;
	float GPUTime;

	if (argc != 9){
		printf ("Wrong number of parameters.\n");
		printf ("Usage: gol WIDTH HEIGHT ITERATIONS GPUPERCENT GPUBLOCKS_X GPUBLOCKS_Y CPUTHREADS VERBOSE\n");
		exit (-1);
	}

	width = atoi (argv[1]);
	height = atoi (argv[2]);
	T_MAX= atoi(argv[3]);
	GPUTime = atof(argv[4]);
	GPUBlockSizeX = atoi(argv[5]);
	GPUBlockSizeY = atoi(argv[6]);
	numCPUThreads = atoi(argv[7]);
	verbose = atoi(argv[8]);
	
	Array2D<bool> inputGrid(width, height);
	Array2D<bool> outputGrid(width, height);
	Mask2D<bool> mask(8);
	
	mask.set(0,-1,-1);	mask.set(1,-1,0);	mask.set(2,-1,1);
	mask.set(3,0,-1);						mask.set(4,0,1);
	mask.set(5,1,-1);	mask.set(6,1,0);	mask.set(7,1,1);
		
	omp_set_num_threads(numCPUThreads);

    srand(1234);
    for(int h = 1; h < height-1; h++){		
        for(int w = 1; w < width-1; w++){
            inputGrid(h,w) = (rand()%2);            
            //outputGrid(i,j) =  inputGrid(i,j);
		}
	}	
	
	hr_timer_t timer;
	hrt_start(&timer);
	//wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
	Stencil2D<Array2D<bool>, Mask2D<bool>, short> stencil(inputGrid, outputGrid, mask, 0);
	
	#ifdef PSKEL_PAPI
		if(GPUTime < 1.0)
			PSkelPAPI::init(PSkelPAPI::CPU);
	#endif	
	
	if(GPUTime == 0.0){
		//jacobi.runIterativeCPU(T_MAX, numCPUThreads);
		#ifdef PSKEL_PAPI
			for(unsigned int i=0;i<NUM_GROUPS_CPU;i++){
				//cout << "Running iteration " << i << endl;
				stencil.runIterativeCPU(T_MAX, numCPUThreads, i);	
			}
		#else
			//cout<<"Running Iterative CPU"<<endl;
			stencil.runIterativeCPU(T_MAX, numCPUThreads);	
		#endif
	}
	else if(GPUTime == 1.0){
		stencil.runIterativeGPU(T_MAX, GPUBlockSizeX, GPUBlockSizeY);
	}
	else{
		//jacobi.runIterativePartition(T_MAX, GPUTime, numCPUThreads,GPUBlockSize);
		/*
        #ifdef PSKEL_PAPI
			for(unsigned int i=0;i<NUM_GROUPS_CPU;i++){
				stencil.runIterativePartition(T_MAX, GPUTime, numCPUThreads,GPUBlockSizeX,i);
			}
		#else
			//stencil.runIterativePartition(T_MAX, GPUTime, numCPUThreads,GPUBlockSizeX);
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
    
    if(verbose){		
		//cout<<setprecision(6);
		//cout<<fixed;
		cout<<"INPUT"<<endl;
		/*for(int i=0; i<width;i+=10){
            
			cout<<"("<<i<<","<<i<<") = "<<inputGrid(i,i)<<"\t\t(";
            cout<<width-i<<","<<height-i<<") = "<<inputGrid(height-i,width-i)<<endl;
		}
		cout<<endl;
        */
        for(int h = 0; h < height; h++){		
			for(int w = 0; w < width; w++){
				cout<<inputGrid(h,w);
			}
			cout<<endl;
		}
		
		cout<<"OUTPUT"<<endl;
		//for(int i=0; i<width/10;i+=10){
		//	cout<<"("<<i<<","<<i<<") = "<<outputGrid[i*width+i]<<"\t\t("<<width-i<<","<<height-i<<") = "<<outputGrid[(height-i)*width+(width-i)]<<endl;
		//}
		//cout<<endl;
		
		for(int h = 0; h < height; ++h){		
			for(int w = 0; w < width; ++w){
				cout<<outputGrid(h,w);
			}
			cout<<endl;
		}
	}
    
    cout << "Exec_time\t" << hrt_elapsed_time(&timer) << endl;
    
	return 0;
}
