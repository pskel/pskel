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
//#include <openacc.h>

using namespace std;

void stencilKernel(int *input, int *output, int width, int height, int T_MAX){
    #pragma acc data copyin(input[0:width*height]) copy(output[0:width*height])
    {
	for(int t=0;t<T_MAX;t++){
        #pragma acc parallel loop independent
		for(int j=0;j<height;j++){
            #pragma acc loop
			for(int i=0;i<width;i++){
                
                //int neighbors = input[(j)*width + (i+1)] + input[(j)*width + (i-1)] +
                //                input[(j+1)*width + (i)] + input[(j-1)*width + (i)] +
                //                input[(j+1)*width + (i+1)] + input[(j-1)*width + (i-1)] +
                //                input[(j+1)*width + (i-1)] + input[(j-1)*width + (i+1)];
                
                /*int neighbors = 0;
				for(int y=-1;y<=1;y++){
					for(int x=-1;x<=1;x++){
						if( x!=0 || y!=0 ){
							neighbors += input[(j+y)*width + (i+x)];
						}
					}
				}
                */
				int neighbors = 0;
				if ( (j == 0) && (i == 0) ) { //	Corner 1	
					neighbors = input[(j)*width + (i+1)] + input[(j+1)*width + (i)] + input[(j+1)*width + (i+1)];
								//input(i+1,j) + input(i,j+1) + input (i+1,j+1);
				}	//	Corner 2	
				else if ((j == 0) && (i == width-1)) {
					neighbors = input[(j)*width + (i-1)] + input[(j+1)*width + (i)] + input[(j+1)*width + (i-1)];
								//input(i-1,j) + input(i,j+1) + input(i-1,j+1);
				}	//	Corner 3	
				else if ((j == height-1) && (i == width-1)) {
					neighbors =	input[(j)*width + (i-1)] + input[(j-1)*width + (i)] + input[(j-1)*width + (i-1)];
								//input(i-1,j) + input(i,j-1) + input(i-1,j-1);
				}	//	Corner 4	
				else if ((j == height-1) && (i == 0)) {
					neighbors =	input[(j-1)*width + (i)] + input[(j)*width + (i+1)] + input[(j-1)*width + (i+1)];
								//input(i,j-1) + input(i+1,j) + input(i+1,j-1);
				}	//	Edge 1	
				else if (j == 0) {
					neighbors =	input[(j)*width + (i-1)] + input[(j)*width + (i+1)] + input[(j+1)*width + (i-1)] +
								input[(j+1)*width + (i)] + input[(j+1)*width + (i+1)];
								//input(i-1,j) + input(i+1,j) + input(i-1,j+1) + input(i,j+1) + input(i+1,j+1);
				}	//	Edge 2	
				else if (i == width-1) {
					neighbors =	input[(j-1)*width + (i)] + input[(j-1)*width + (i-1)] + input[(j)*width + (i-1)] +
								input[(j+1)*width + (i-1)] + input[(j+1)*width + (i)];
								//input(i,j-1) +  input(i-1,j-1) + input(i-1,j) +  input(i-1,j+1) + input(i,j+1);
				}	//Edge 3	
				else if (j == height-1) {
					neighbors =	input[(j-1)*width + (i-1)] + input[(j-1)*width + (i)] + input[(j-1)*width + (i+1)] +
								input[(j)*width + (i-1)] + input[(j)*width + (i+1)];
								//input(i-1,j-1) + input(i,j-1) + input(i+1,j-1) + input(i-1,j) + input(i+1,j);
				}	//Edge 4
				else if (i == 0) {
					neighbors =	input[(j-1)*width + (i)] + input[(j-1)*width + (i+1)] + input[(j)*width + (i+1)] +
								input[(j+1)*width + (i)] + input[(j+1)*width + (i+1)];
								//input(i,j-1) + input(i+1,j-1) + input(i+1,j) + input(i,j+1) + input(i+1,j+1);
				}	//Inside the grid
				else {
					neighbors = input[(j-1)*width + (i-1)] + input[(j-1)*width + (i)] + input[(j-1)*width + (i+1)] +
								input[(j)*width + (i-1)] + input[(j)*width + (i+1)] +
								input[(j+1)*width + (i-1)] + input[(j+1)*width + (i)] + input[(j+1)*width + (i+1)];
								//input(i-1,j-1) + input(i-1,j) + input(i-1,j+1) + input(i+1,j-1) + input(i+1,j) +
								//input(i+1,j+1) + input(i,j-1)   + input(i,j+1) ;
				}

                
                output[j*width + i] = (neighbors == 3 || (input[j*width + i] == 1 && neighbors == 2))?1:0;
				
                /*if(neighbors == 3 || (input[j*width + i] == 1 && neighbors == 2)){
					output[j*width + i] = 1;
				}
				else{
					output[j*width + i] = 0;
				}
                */
			}
		}
        
        //swap
        if(t>1 && t<T_MAX - 1){
			#pragma acc parallel loop independent
			for(int j=0;j<height;j++){
				#pragma acc loop independent
				for(int i=0;i<width;i++){
					input[j*width + i] = output[j*width + i];
				}
			}
		}
	}
    }
}

int main(int argc, char **argv){
	int width;
	int height;
	int T_MAX;
    int verbose;

	int *inputGrid;
	int *outputGrid;

	if (argc != 5){
		printf ("Wrong number of parameters.\n");
		printf ("Usage: gol WIDTH HEIGHT ITERATIONS VERBOSE\n");
		exit (-1);
	}

	width = atoi (argv[1]);
	height = atoi (argv[2]);
	T_MAX = atoi (argv[3]);
    verbose = atoi (argv[4]);

	inputGrid = (int*) calloc(width*height,sizeof(int));
	outputGrid = (int*) calloc(width*height,sizeof(int));

	srand(123456789);
	//#pragma omp parallel for
	for(int j=1;j<height-1;j++) {
		for(int i=1;i<width-1;i++) {
			inputGrid[j*width + i] = rand()%2;
		}
	}
  
    hr_timer_t timer;
	hrt_start(&timer);
	//#pragma pskel stencil dim2d(width, height) inout(inputGrid, outputGrid) iterations(T_MAX) device(cpu)
	stencilKernel(inputGrid, outputGrid,width,height,T_MAX);
    
    
	hrt_stop(&timer);
	
	if(verbose){		
		//cout<<setprecision(6);
		//cout<<fixed;
		cout<<"INPUT"<<endl;
		for(int i=0; i<width;i+=10){
            
			cout<<"("<<i<<","<<i<<") = "<<inputGrid[i*width+i]<<"\t\t(";
            cout<<width-i<<","<<height-i<<") = "<<inputGrid[(height-i)*width+(width-i)]<<endl;
		}
		cout<<endl;
		
		cout<<"OUTPUT"<<endl;
		//for(int i=0; i<width/10;i+=10){
		//	cout<<"("<<i<<","<<i<<") = "<<outputGrid[i*width+i]<<"\t\t("<<width-i<<","<<height-i<<") = "<<outputGrid[(height-i)*width+(width-i)]<<endl;
		//}
		//cout<<endl;
		
		for(int h = 0; h < height; ++h){		
			for(int w = 0; w < width; ++w){
				cout<<outputGrid[h*width + w];
			}
			cout<<endl;
		}
	}
    
    cout << "Exec_time\t" << hrt_elapsed_time(&timer) << endl;
  
	return 0;
}
