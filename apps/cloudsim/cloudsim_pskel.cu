#define PSKEL_OMP 1
#define PSKEL_CUDA 1

#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <algorithm>

#include "../../pskel/include/PSkel.h"
#include "../../pskel/include/hr_time.h"

using namespace std;
using namespace PSkel;

#define WIND_X_BASE	15
#define WIND_Y_BASE	12
#define DISTURB		0.1f
#define CELL_LENGTH	0.1f
#define K           	0.0243f
#define DELTAPO       	0.5f
#define TAM_VETOR_FILENAME  200

struct Cloud{	
	//Args2D<float> wind_x, wind_y;
	Array2D<float> wind_x;
	Array2D<float> wind_y;
	float deltaT;
	
	Cloud(){};
	
	Cloud(int linha, int coluna){		
		//new (&wind_x) Args2D<float>(linha, coluna);
		//new (&wind_y) Args2D<float>(linha, coluna);
		new (&wind_x) Array2D<float>(linha, coluna);
		new (&wind_y) Array2D<float>(linha, coluna);
	}
};

namespace PSkel{	
	__parallel__ void stencilKernel(Array2D<float> input,Array2D<float> output,Mask2D<float> mask,Cloud cloud,size_t i, size_t j){
		int numNeighbor = 0;
		float sum = 0.0f;
		float inValue = input(i,j);
        float temp_wind = 0.0f;
        int height=input.getHeight();
        int width=input.getWidth();
        
         if ( (j == 0) && (i == 0) ) {
                    sum = (inValue - input(i+1,j) ) +
                          (inValue - input(i,j+1) );
                    numNeighbor = 2;
                }	/*	Corner 2	*/
                else if ((j == 0) && (i == width-1)) {
                    sum = (inValue - input(i-1,j) ) +
                          (inValue - input(i,j+1) );
                    numNeighbor = 2;
                }	/*	Corner 3	*/
                else if ((j == height-1) && (i == width-1)) {
                    sum = (inValue - input(i-1,j) ) +
                          (inValue - input(i,j-1) );
                    numNeighbor = 2;
                }	/*	Corner 4	*/
                else if ((j == height-1) && (i == 0)) {
                    sum = (inValue - input(i,j-1) ) +
                          (inValue - input(i+1,j) );
                    numNeighbor = 2;
                }	/*	Edge 1	*/
                else if (j == 0) {
                    sum = (inValue - input(i-1,j) ) +
                          (inValue - input(i+1,j) ) +
                          (inValue - input(i,j+1) );
                    numNeighbor = 3;
                }	/*	Edge 2	*/
                else if (i == width-1) {
                    sum = (inValue - input(i-1,j) ) +
                          (inValue - input(i,j-1) ) +
                          (inValue - input(i,j+1) );
                    numNeighbor = 3;
                }	/*	Edge 3	*/
                else if (j == height-1) {
                    sum = (inValue - input(i-1,j) ) +
                          (inValue - input(i,j-1) ) +
                          (inValue - input(i+1,j) );
                    numNeighbor = 3;
                }	/*	Edge 4	*/
                else if (i == 0) {
                    sum = (inValue - input(i,j-1) ) +
                          (inValue - input(i,j+1) ) +
                          (inValue - input(i+1,j) );
                    numNeighbor = 3;
                }	/*	Inside the cloud  */
                else {
                    sum = (inValue - input(i-1,j) ) +
                          (inValue - input(i,j-1) ) +
                          (inValue - input(i,j+1) ) +
                          (inValue - input(i+1,j) );
                    numNeighbor = 4;
                    
                    float xwind = 1;//cloud.wind_x(i,j);
                    float ywind = 1;//cloud.wind_y(i,j);
                    int xfactor = (xwind>0)?1:-1;
                    int yfactor = (ywind>0)?1:-1;

                    float temperaturaNeighborX = input(i,(j+xfactor));
                    float componenteVentoX = xfactor * xwind;
                    float temperaturaNeighborY = input((i+yfactor),j);
                    float componenteVentoY = yfactor * ywind;
				
                    temp_wind = (-componenteVentoX * ((inValue - temperaturaNeighborX)/CELL_LENGTH)) -
                                ( componenteVentoY * ((inValue - temperaturaNeighborY)/CELL_LENGTH));
                    
                }
				float temperatura_conducao = -K*(sum / numNeighbor) * cloud.deltaT;
				float result = inValue + temperatura_conducao;
				output(i,j) = result + temp_wind * cloud.deltaT;

		/*
        for( int m = 0; m < mask.size ; m++ ){
			float temperatura_vizinho = mask.get(m,input,i,j);
			int factor = (temperatura_vizinho==0)?0:1;
			sum += factor*(inValue - temperatura_vizinho);
			numNeighbor += factor;
		}
		
        		
		float temperatura_conducao = -K*(sum / numNeighbor)*cloud.deltaT;
		
		float result = inValue + temperatura_conducao;
		
		float xwind = cloud.wind_x(i,j);
		float ywind = cloud.wind_y(i,j);
		int xfactor = (xwind>0)?3:1;
		int yfactor = (ywind>0)?2:0;

		float temperaturaNeighborX = mask.get(xfactor,input,i,j);
		float componenteVentoX = (xfactor-2)*xwind;
		float temperaturaNeighborY = mask.get(yfactor,input,i,j);
		float componenteVentoY = (yfactor-1)*ywind;
		
		float temp_wind = (-componenteVentoX * ((inValue - temperaturaNeighborX)/CELL_LENGTH)) -(componenteVentoY * ((inValue - temperaturaNeighborY)/CELL_LENGTH));
		
		output(i,j) = result + ((numNeighbor==4)?(temp_wind*cloud.deltaT):0.0f);
        */
	}	
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
	int linha, coluna, i, j, numero_iteracoes, raio_nuvem, menu_option, GPUBlockSizeX, GPUBlockSizeY, numCPUThreads;
	float temperaturaAtmosferica, pressaoAtmosferica, pontoOrvalho, limInfPO, limSupPO, deltaT, GPUTime;
	//float alturaNuvem;
    //int write_step;
	if (argc != 9){
		printf ("Wrong number of parameters.\n");
		//printf ("Usage: cloudsim Numero_Iteraoes Linha Coluna Raio_Nuvem Temperatura_Atmosferica Altura_Nuvem Pressao_Atmosferica Delta_T GPUTIME GPUBLOCKS CPUTHREADS Menu_Option Write_Step\n");
		printf ("Usage: cloudsim WIDTH HEIGHT ITERATIONS GPUTIME GPUBLOCK_X GPU_BLOCK_Y CPUTHREADS OUTPUT_WRITE_FLAG\n");
		exit (-1);
	}
	//20 -3 5.0 700.0 0.001 1.0 32 12 0 10
	
	coluna = atoi(argv[1]);
	linha = atoi(argv[2]);
	numero_iteracoes = atoi(argv[3]);
	GPUTime = atof(argv[4]);
	GPUBlockSizeX = atoi(argv[5]);
    	GPUBlockSizeY = atoi(argv[6]);
	numCPUThreads = atoi(argv[7]);
	menu_option = atoi(argv[8]);
	
	raio_nuvem = 20; 				//atoi(argv[4]);
	temperaturaAtmosferica = -3.0f; 	//atof(argv[5]);
	//alturaNuvem = 5.0; 				//atof(argv[6]);
	pressaoAtmosferica =  700.0f;		//atof(argv[7]);
	deltaT = 0.01f;					//atof(argv[8]);
	
	//numThreads = numCPUThreads;
	//write_step = 10;				//atoi(argv[13]);
	
	//global_write_step = write_step;
	pontoOrvalho = CalculateDewPoint(Convert_Celsius_To_Kelvin(temperaturaAtmosferica), Convert_hPa_To_mmHg(pressaoAtmosferica));
	limInfPO = pontoOrvalho - DELTAPO;
	limSupPO = pontoOrvalho + DELTAPO;
	//char maindir[30];
	//char dirname[TAM_VETOR_FILENAME];
	//char dirMatrix_temp[TAM_VETOR_FILENAME];
	//char dirMatrix_stat[TAM_VETOR_FILENAME];
	//char dirMatrix_windX[TAM_VETOR_FILENAME];
	//char dirMatrix_windY[TAM_VETOR_FILENAME];
	//float start_time = 0;
	//float end_time = 0;
		
	Array2D<float> inputGrid(coluna, linha);
	Array2D<float> outputGrid(coluna, linha);
	Mask2D<float> mask(4);
	
	mask.set(0,0,1);
	mask.set(1,1,0);
	mask.set(2,0,-1);
	mask.set(3,-1,0);
	
	Cloud cloud(linha,coluna);
	cloud.deltaT = deltaT;
	
	omp_set_num_threads(numCPUThreads);

	/* Inicialização da matriz de entrada com a temperatura ambiente */
	//#pragma omp parallel for private (i,j)
	for (i = 0; i < linha; i++){		
		for (j = 0; j < coluna; j++){
			inputGrid(i,j) = temperaturaAtmosferica;
			//outputGrid(i,j) = temperaturaAtmosferica;
		}
	}	
	/* Inicialização dos ventos Latitudinal(Wind_X) e Longitudinal(Wind_Y) */
	for( i = 0; i < linha; i++ ){
		for(j = 0; j < coluna; j++ ){			
			cloud.wind_x(i,j) = (WIND_X_BASE - DISTURB) + (float)rand()/RAND_MAX * 2 * DISTURB;
			cloud.wind_y(i,j) = (WIND_Y_BASE - DISTURB) + (float)rand()/RAND_MAX * 2 * DISTURB;		
		}
	}
	
	//Forcing copy
	//cloud.wind_x.deviceAlloc();
	//cloud.wind_x.copyToDevice();
	//cloud.wind_y.deviceAlloc();
	//cloud.wind_y.copyToDevice();	
					
	/* Inicialização de uma nuvem no centro da matriz de entrada */
	int y, x0 = linha/2, y0 = coluna/2;
	srand(1);
	for(i = x0 - raio_nuvem; i < x0 + raio_nuvem; i++){
		 // Equação da circunferencia: (x0 - x)² + (y0 - y)² = r²
		y = (int)((floor(sqrt(pow((float)raio_nuvem, 2.0) - pow(((float)x0 - (float)i), 2)) - y0) * -1));
		for(int j = y0 + (y0 - y); j >= y; j--){
			inputGrid(i,j) = limInfPO + (float)rand()/RAND_MAX * (limSupPO - limInfPO);
			//outputGrid(i,j) = limInfPO + (float)rand()/RAND_MAX * (limSupPO - limInfPO);
		}
	}
	
	Stencil2D<Array2D<float>, Mask2D<float>, Cloud> stencilCloud(inputGrid, outputGrid, mask, cloud);
	
	hr_timer_t timer;
	hrt_start(&timer);
	
	if(GPUTime == 0.0){
		//cout<<"Running Iterative CPU"<<endl;
		if(numCPUThreads == 1)
			stencilCloud.runSequential();
		else
			stencilCloud.runIterativeCPU(numero_iteracoes, numCPUThreads);
	}
	else if(GPUTime == 1.0){
		//Forcing copy
		cloud.wind_x.deviceAlloc();
		cloud.wind_x.copyToDevice();
		cloud.wind_y.deviceAlloc();
		cloud.wind_y.copyToDevice();	
		stencilCloud.runIterativeGPU(numero_iteracoes, GPUBlockSizeX, GPUBlockSizeY);
	}
	else{
		//stencilCloud.runIterativePartition(numero_iteracoes, GPUTime, numCPUThreads,GPUBlockSize);
	}
	
	hrt_stop(&timer);
	cout << "Exec_time\t" << hrt_elapsed_time(&timer) << endl;
	
	if(menu_option == 1){		
		cout.precision(12);
		cout<<"INPUT"<<endl;
		for(int i=10; i<coluna;i+=10){
			cout<<"("<<i<<","<<i<<") = "<<inputGrid(i,i)<<"\t\t("<<coluna-i<<","<<linha-i<<") = "<<inputGrid(coluna-i,linha-i)<<endl;
		}
		cout<<endl;
		
		cout<<"OUTPUT"<<endl;
		for(int i=10; i<coluna;i+=10){
			cout<<"("<<i<<","<<i<<") = "<<outputGrid(i,i)<<"\t\t("<<coluna-i<<","<<linha-i<<") = "<<outputGrid(coluna-i,linha-i)<<endl;
		}
		cout<<endl;
	}
	return 0;
}

