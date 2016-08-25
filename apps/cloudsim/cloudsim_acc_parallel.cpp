#define PSKEL_OMP 1
#define PSKEL_CUDA 1

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <algorithm>
#include <string.h>

#include <omp.h>
#include <openacc.h>
#include "hr_time.h"

using namespace std;

#define WIND_X_BASE	15
#define WIND_Y_BASE	12
#define DISTURB		0.1f
#define CELL_LENGTH	0.1f
#define K           	0.0243f
#define DELTAPO       	0.5f
#define TAM_VETOR_FILENAME  200

void stencilKernel(float* __restrict__ input,float* __restrict__ output, int width, int height, int T_MAX,float* __restrict__ wind_x,float* __restrict__ wind_y,float deltaT){
	#pragma acc data copyin(input[0:width*height],wind_x[0:width*height],wind_y[0:width*height]) copyout(output[0:width*height])
    	{
	for(int t=0;t<T_MAX;t++){
		#pragma acc parallel loop
		for(int j=0;j<height;j++){
			#pragma acc loop
			for(int i=0;i<width;i++){
				int numNeighbor = 0;
				float sum = 0.0f;
				float inValue = input[j*width+i];
                float temp_wind = 0.0f;
				
                /*	Corner 1	*/
                if ( (j == 0) && (i == 0) ) {
                    sum = (inValue - input[j*width+(i+1)]) +
                          (inValue - input[(j+1)*width+i]);
                    numNeighbor = 2;
                }	/*	Corner 2	*/
                else if ((j == 0) && (i == width-1)) {
                    sum = (inValue - input[j*width+(i-1)]) +
                          (inValue - input[(j+1)*width+i]);
                    numNeighbor = 2;
                }	/*	Corner 3	*/
                else if ((j == height-1) && (i == width-1)) {
                    sum = (inValue - input[j*width+(i-1)]) +
                          (inValue - input[(j-1)*width+i]);
                    numNeighbor = 2;
                }	/*	Corner 4	*/
                else if ((j == height-1) && (i == 0)) {
                    sum = (inValue - input[j*width+(i+1)]) +
                          (inValue - input[(j-1)*width+i]);
                    numNeighbor = 2;
                }	/*	Edge 1	*/
                else if (j == 0) {
                    sum = (inValue - input[j*width+(i-1)]) +
                          (inValue - input[j*width+(i+1)]) +
                          (inValue - input[(j+1)*width+i]);
                    numNeighbor = 3;
                }	/*	Edge 2	*/
                else if (i == width-1) {
                    sum = (inValue - input[j*width+(i-1)]) +
                          (inValue - input[(j-1)*width+i]) +
                          (inValue - input[(j+1)*width+i]);
                    numNeighbor = 3;
                }	/*	Edge 3	*/
                else if (j == height-1) {
                    sum = (inValue - input[j*width+(i-1)]) +
                          (inValue - input[j*width+(i+1)]) +
                          (inValue - input[(j-1)*width+i]);
                    numNeighbor = 3;
                }	/*	Edge 4	*/
                else if (i == 0) {
                    sum = (inValue - input[(j-1)*width+i]) +
                          (inValue - input[j*width+(i+1)]) +
                          (inValue - input[(j+1)*width+i]);
                    numNeighbor = 3;
                }	/*	Inside the cloud  */
                else {
                    sum = (inValue - input[(j-1)*width+i]) +
                          (inValue - input[j*width+(i-1)]) +
                          (inValue - input[j*width+(i+1)]) +
                          (inValue - input[(j+1)*width+i]);
                    numNeighbor = 4;
                    
                    float xwind = wind_x[j*width+i];
                    float ywind = wind_y[j*width+i];
                    int xfactor = (xwind>0)?1:-1;
                    int yfactor = (ywind>0)?1:-1;

                    float temperaturaNeighborX = input[(j+xfactor) * width + i];
                    float componenteVentoX = xfactor * xwind;
                    float temperaturaNeighborY = input[j*width + (i+yfactor)];
                    float componenteVentoY = yfactor * ywind;
				
                    temp_wind = (-componenteVentoX * ((inValue - temperaturaNeighborX)/CELL_LENGTH)) -
                                ( componenteVentoY * ((inValue - temperaturaNeighborY)/CELL_LENGTH));
                    
                }
				float temperatura_conducao = -K*(sum / numNeighbor) * deltaT;
				float result = inValue + temperatura_conducao;
				output[j*width+i] = result + temp_wind * deltaT;
			}
		}
		
		if(t>1 && t<T_MAX-1){
		//swap(output,input);	
			#pragma acc parallel loop
			for(int j=0;j<height;j++){
				#pragma acc loop
				for(int i=0;i<width;i++){
					input[j*width+i] = output[j*width+i];
				}
			}	
		}
	}
    }
	//swap(output,input);
	//if(T_MAX%2==0)
	//   memcpy(input,output,width*height*sizeof(float));
}

/* Convert Celsius to Kelvin */
float Convert_Celsius_To_Kelvin(float number_celsius)
{
	float number_kelvin;
	number_kelvin = number_celsius + 273.15f;
	return number_kelvin;
}

/* Convert Pressure(hPa) to Pressure(mmHg) */
float Convert_hPa_To_mmHg(float number_hpa)
{
	float number_mmHg;
	number_mmHg = number_hpa * 0.750062f;

	return number_mmHg;
}

/* Convert Pressure Millibars to mmHg */
float Convert_milibars_To_mmHg(float number_milibars)
{
	float number_mmHg;
	number_mmHg = number_milibars * 0.750062f;

	return number_mmHg;
}

/* Calculate RPV */
float CalculateRPV(float temperature_Kelvin, float pressure_mmHg)
{
	float realPressureVapor; //e
	float PsychrometricConstant = 6.7f * powf(10,-4); //A
	float PsychrometricDepression = 1.2f; //(t - tu) in ºC
	float esu = pow(10, ((-2937.4f / temperature_Kelvin) - 4.9283f * log10(temperature_Kelvin) + 23.5470f)); //10 ^ (-2937,4 / t - 4,9283 log t + 23,5470)
	realPressureVapor = Convert_milibars_To_mmHg(esu) - (PsychrometricConstant * pressure_mmHg * PsychrometricDepression);

	return realPressureVapor;
}

/* Calculate Dew Point */
float CalculateDewPoint(float temperature_Kelvin, float pressure_mmHg)
{
	float dewPoint; //TD
	float realPressureVapor = CalculateRPV(temperature_Kelvin, pressure_mmHg); //e
	dewPoint = (186.4905f - 237.3f * log10(realPressureVapor)) / (log10(realPressureVapor) -8.2859f);

	return dewPoint;
}

int main(int argc, char **argv){
	int linha, coluna, i, j, T_MAX, raio_nuvem, menu_option;
	//int GPUBlockSize, numCPUThreads;
	float temperaturaAtmosferica, pressaoAtmosferica, pontoOrvalho, limInfPO, limSupPO, deltaT;
	//float GPUTime;
	float *inputGrid, *outputGrid, *wind_x, *wind_y;
	
	if (argc != 5){
		printf ("Wrong number of parameters.\n");
		//printf ("Usage: cloudsim Numero_Iteraoes Linha Coluna Raio_Nuvem Temperatura_Atmosferica Altura_Nuvem Pressao_Atmosferica Delta_T GPUTIME GPUBLOCKS CPUTHREADS Menu_Option Write_Step\n");
		printf ("Usage: cloudsim WIDTH HEIGHT ITERATIONS OUTPUT_WRITE_FLAG\n");
		exit (-1);
	}
	//20 -3 5.0 700.0 0.001 1.0 32 12 0 10
	
	coluna = atoi(argv[1]);
	linha = atoi(argv[2]);
	T_MAX = atoi(argv[3]);
	//GPUTime = atof(argv[4]);
	//GPUBlockSize = atoi(argv[5]);
	//numCPUThreads = atoi(argv[6]);
	menu_option = atoi(argv[4]);
	
	raio_nuvem = 20; 				//atoi(argv[4]);
	temperaturaAtmosferica = -3.0f; 	//atof(argv[5]);
	//alturaNuvem = 5.0; 				//atof(argv[6]);
	pressaoAtmosferica =  700.0f;		//atof(argv[7]);
	deltaT = 0.01f;					//atof(argv[8]);
	
	pontoOrvalho = CalculateDewPoint(Convert_Celsius_To_Kelvin(temperaturaAtmosferica), Convert_hPa_To_mmHg(pressaoAtmosferica));
	limInfPO = pontoOrvalho - DELTAPO;
	limSupPO = pontoOrvalho + DELTAPO;
		
	inputGrid = (float*) malloc(coluna*linha*sizeof(float));
	outputGrid = (float*) malloc(coluna*linha*sizeof(float));
	wind_x = (float*) malloc(coluna*linha*sizeof(float));	
	wind_y = (float*) malloc(coluna*linha*sizeof(float));		
	
	//omp_set_num_threads(numCPUThreads);

	/* Inicialização da matriz de entrada com a temperatura ambiente */
	#pragma omp parallel for private (i,j)   	
	for (i = 0; i < linha; i++){		
		for (j = 0; j < coluna; j++){
			inputGrid[i*coluna+j] = temperaturaAtmosferica;
			outputGrid[i*coluna+j] = temperaturaAtmosferica;
		}
	}
		
	/* Inicialização dos ventos Latitudinal(Wind_X) e Longitudinal(Wind_Y) */
    	//cout<<"Initializing wind"<<endl;
	for( i = 0; i < linha; i++ ){
		for(j = 0; j < coluna; j++ ){			
			wind_x[i*coluna+j] = (WIND_X_BASE - DISTURB) + (float)rand()/RAND_MAX * 2 * DISTURB;
			wind_y[i*coluna+j] = (WIND_Y_BASE - DISTURB) + (float)rand()/RAND_MAX * 2 * DISTURB;		
		}
	}

	/* Inicialização de uma nuvem no centro da matriz de entrada */
    	//cout<<"Generating initial cloud in center of inputGrid"<<endl;
	int y, x0 = linha/2, y0 = coluna/2;
	srand(1);
	for(i = x0 - raio_nuvem; i < x0 + raio_nuvem; i++){
		 // Equação da circunferencia: (x0 - x)² + (y0 - y)² = r²
		y = (int)((floor(sqrt(pow((float)raio_nuvem, 2.0) - pow(((float)x0 - (float)i), 2)) - y0) * -1));
		for(int j = y0 + (y0 - y); j >= y; j--){
			float value = limInfPO + (float)rand()/RAND_MAX * (limSupPO - limInfPO);
			inputGrid[i*coluna+j] = value;
			outputGrid[i*coluna+j] = value;
		}
	}
	
    	//cout<<"Starting simulation..."<<endl;
	hr_timer_t timer;
	hrt_start(&timer);
	
	//#pragma pskel stencil dim2d(coluna,linha) inout(inputGrid, outputGrid) iterations(T_MAX) device(cpu)
	stencilKernel(inputGrid, outputGrid, coluna, linha, T_MAX, wind_x, wind_y, deltaT);
	
	hrt_stop(&timer);
	cout << "Exec_time\t" << hrt_elapsed_time(&timer) << endl;
	
	if(menu_option == 1){		
		cout.precision(12);
		cout<<"INPUT"<<endl;
		for(int i=10; i<coluna;i+=10){
			cout<<"("<<i<<","<<i<<") = "<<inputGrid[i*coluna+i]<<"\t\t("<<coluna-i<<","<<linha-i<<") = "<<inputGrid[(coluna-i)*coluna+linha-i]<<endl;
		}
		cout<<endl;
		
		cout<<"OUTPUT"<<endl;
		for(int i=10; i<coluna;i+=10){
			cout<<"("<<i<<","<<i<<") = "<<outputGrid[i*coluna+i]<<"\t\t("<<coluna-i<<","<<linha-i<<") = "<<outputGrid[(coluna-i)*coluna+(linha-i)]<<endl;
		}
		cout<<endl;
	}
	return 0;
}
