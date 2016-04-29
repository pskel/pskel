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

#include "../../include/PSkel.h"

#include "../../include/hr_time.h"

using namespace std;
using namespace PSkel;

#define WIND_X_BASE	15
#define WIND_Y_BASE	12
#define DISTURB		0.1f
#define CELL_LENGTH	0.1f
#define K           	0.0243f
#define DELTAPO       	0.5f
#define TAM_VETOR_FILENAME  200

int numThreads;
int count_write_step;
int global_write_step;
int global_count_write_step;
char dirname[TAM_VETOR_FILENAME];
char *typeSim;

struct Cloud{	
	Args2D<float> wind_x, wind_y;
	float deltaT;
	
	Cloud(){};
	
	Cloud(int linha, int coluna){		
		new (&wind_x) Args2D<float>(linha, coluna);
		new (&wind_y) Args2D<float>(linha, coluna);
	}
};

//namespace PSkel{
	//void callback(Array2D<float> grid, float GPUTime,int iteration){
		//if(iteration == global_count_write_step){
			//cout<<"inside"<<endl;						
			//FILE *fileaverage, *fileadesviopadrao;
			//char filenameaverage[TAM_VETOR_FILENAME];
			//char filenamedesviopadrao[TAM_VETOR_FILENAME];
			//double sum = 0;
			//double average = 0;
			//double desviopadrao = 0;
			//double variancia = 0;

						
			//if(numThreads > 1){		
				//sprintf(filenameaverage,"%s//%s-average-temperature-simulation-%d-thread.txt",dirname, typeSim,numThreads);
				//sprintf(filenamedesviopadrao,"%s//%s-desviopadrao-temperature-simulation-%d-thread.txt",dirname, typeSim,numThreads);
			//}
			//else{		
				//sprintf(filenameaverage,"%s//%s-average-temperature-simulation.txt",dirname, typeSim);
				//sprintf(filenamedesviopadrao,"%s//%s-desviopadrao-temperature-simulation.txt",dirname, typeSim);
			//}
			//fileaverage = fopen(filenameaverage, "a");
			//fileadesviopadrao = fopen(filenamedesviopadrao, "a");
			
			//if(GPUTime > 0.0){
				////grid.copyToHost(GPUTime);
				//grid.copyToHost(); //TODO need to slice the grid based on GPUTime partition
			//}
			
							 
			//#pragma omp parallel for reduction (+:sum)
			//for(int i = 0; i < grid.getHeight(); i++)
			//{		
				//for(int j = 0; j < grid.getWidth(); j++)
				//{			
					//sum+=grid(i,j);
				//}		
			 //}
			 //average = sum/(grid.getHeight()*grid.getWidth());			 		 			
			 
			 //#pragma omp parallel for reduction (+:variancia)
			//for(int i = 0; i < grid.getHeight(); i++)
			//{		
				//for(int j = 0; j < grid.getWidth(); j++)
				//{			
					//variancia+=pow((grid(i,j)-average),2);
				//}
			 //}	 			 			 			 
			 //desviopadrao = sqrt(variancia/(grid.getHeight()*grid.getWidth()-1));

 			 //fprintf(fileaverage,"%d\t%lf\n",iteration, average);
			 //fprintf(fileadesviopadrao,"%d\t%f\n",iteration, desviopadrao);
				 
			 //fclose(fileaverage);
			 //fclose(fileadesviopadrao);
			 
			 //global_count_write_step += global_write_step;
		//}
	//}
//}

namespace PSkel{
/*
	__parallel__ void stencilKernel(Array2D<float> input,Array2D<float> output,Mask2D<float> mask,Cloud cloud,size_t i, size_t j){
		int numNeighbor = 0;

		float sum = 0;
		float temperatura_conducao = 0;
		for( int m = 0; m < 4 ; m++ )
		{
			float temperatura_vizinho = mask.get(m,input,i,j);
			if(temperatura_vizinho != 0){
				sum += input(i,j) - temperatura_vizinho;
				numNeighbor++;
			}				
		}			
		temperatura_conducao = -K*(sum / numNeighbor)*cloud.deltaT;

		output(i,j) = input(i,j) + temperatura_conducao;


		// Implementation the vertical wind
		if(numNeighbor == 4)
		{
			float componenteVentoX = 0;
			float componenteVentoY = 0;
			float temperaturaNeighborX = 0;
			float temperaturaNeighborY = 0;				

			if(cloud.wind_x(i,j) > 0)
			{
				temperaturaNeighborX = mask.get(3,input,i,j);
				componenteVentoX     = cloud.wind_x(i,j);
			}
			else
			{
				temperaturaNeighborX = mask.get(1,input,i,j);
				componenteVentoX     = -1*cloud.wind_x(i,j);
			}

			if(cloud.wind_y(i,j) > 0)
			{
				temperaturaNeighborY = mask.get(2,input,i,j);
				componenteVentoY     = cloud.wind_y(i,j);
			}
			else
			{
				temperaturaNeighborY = mask.get(0,input,i,j);
				componenteVentoY     = -1*cloud.wind_y(i,j);
			}

			float temp_wind = (-componenteVentoX * ((input(i,j) - temperaturaNeighborX)/CELL_LENGTH)) -(componenteVentoY * ((input(i,j) - temperaturaNeighborY)/CELL_LENGTH));
			output(i,j) = output(i,j) + (temp_wind * cloud.deltaT);
		}
	}
*/

	__parallel__ void stencilKernel(Array2D<float> input,Array2D<float> output,Mask2D<float> mask,Cloud cloud,size_t i, size_t j){
		int numNeighbor = 0;
		float sum = 0.0f;
		float inValue = input(i,j);

		#pragma unroll
		for( int m = 0; m < 4 ; m++ )
		{
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

/*Calculate temperature grid standard deviation */
void StandardDeviation(Array2D<float> grid, int linha, int coluna, int iteracao, char *dirname, int numThreads, char *typeSim, float average)
{
	FILE *file;
	char filename[TAM_VETOR_FILENAME];
	double desviopadrao = 0;
	double variancia = 0;	

	if(numThreads > 1)		
		sprintf(filename,"%s//%s-desviopadrao-temperature-simulation-%d-thread.txt",dirname, typeSim,numThreads);
	else		
		sprintf(filename,"%s//%s-desviopadrao-temperature-simulation.txt",dirname, typeSim);
	file = fopen(filename, "a");	
	
	#pragma omp parallel for reduction (+:variancia)
	for(int i = 0; i < linha; i++)
	{
		for(int j = 0; j < coluna; j++)
		{			
			variancia+=pow((grid(i,j)-average),2);
		}		
	 }	
	 desviopadrao = sqrt(variancia/(linha*coluna-1));
	 fprintf(file,"%d\t%lf\n",iteracao, desviopadrao);
	 fclose(file);
}

/*Calculate temperature grid average */
void CalculateAverage(Array2D<float> grid, int linha, int coluna, int iteracao, char *dirname, int numThreads, char *typeSim)
{
	FILE *file;
	char filename[TAM_VETOR_FILENAME];
	double sum = 0;
	double average = 0;	

	if(numThreads > 1)		
		sprintf(filename,"%s//%s-average-temperature-simulation-%d-thread.txt",dirname, typeSim,numThreads);		
	else		
		sprintf(filename,"%s//%s-average-temperature-simulation.txt",dirname, typeSim);
	file = fopen(filename, "a");	
	
	#pragma omp parallel for reduction (+:sum)
	for(int i = 0; i < linha; i++)
	{		
		for(int j = 0; j < coluna; j++)
		{			
			sum+=grid(i,j);
		}		
	 }	
	 average = sum/(linha*coluna);
	 //fprintf(file,"%d\t%lf\n",iteracao, average);
	 fprintf(file,"%lf\n", average);
	 printf("Media:%d\t%lf\n",iteracao, average);
	 fclose(file);
	 
	// StandardDeviation(grid, linha, coluna, iteracao, dirname, numThreads, typeSim, average);
	 
}

/* Calculate statistics of temperature grid */
void CalculateStatistics(Array2D<float> grid, int linha, int coluna, int iteracao, char *dirname, int numThreads, char *typeSim)
{
	CalculateAverage(grid, linha, coluna, iteracao, dirname, numThreads, typeSim);	
}


/* Write grid temperature in text file */
void WriteGridTemp(Array2D<float> grid, int linha, int coluna, int iteracao, int numThreads, char *dirname, char *typeSim)
{
	FILE *file;
	char filename[TAM_VETOR_FILENAME];
	
	if(numThreads > 1)
		sprintf(filename,"%s//%s-temp_%d-thread_iteration#_%d.txt",dirname, typeSim, numThreads, iteracao);
	else
		sprintf(filename,"%s//%s-temp_iteration#_%d.txt",dirname, typeSim, iteracao);
		
	file = fopen(filename, "w");		

	fprintf(file, "Iteração: %d\n", iteracao);
	for(int i = 0; i < linha; i++)
	{
		for(int j = 0; j < coluna; j++)
		{
			fprintf(file, "%.4f  ", grid(i,j));
		}
		fprintf(file, "\n");
	 }		
	fclose(file);
}

/* Write time simulation */
void WriteTimeSimulation(float time, int numThreads, char *dirname, char *typeSim)
{
	FILE *file;
	char filename[TAM_VETOR_FILENAME];
	
	if(numThreads > 1)
		sprintf(filename,"%s//%stime-sim_%d-thread.txt",dirname, typeSim, numThreads);
	else
		sprintf(filename,"%s//%stime-sim.txt",dirname, typeSim);

	file = fopen(filename,"r");
	if (file==NULL)
	{
		file = fopen(filename, "w");
		//fprintf(file,"Time %s-simulation", typeSim);
		fprintf(file,"\nUpdate Time: %f segundos", time);
	}
	else
	{
		file = fopen(filename, "a");
		fprintf(file,"\nUpdate Time: %f segundos", time);
	}
	
	fclose(file);
}

/* Write Simulation info all parameters */
void WriteSimulationInfo(int numero_iteracoes, int linha, int coluna, int raio_nuvem, float temperaturaAtmosferica, float alturaNuvem, float pressaoAtmosferica, float deltaT, float pontoOrvalho, int menu_option, int write_step, int numThreads, float GPUTime, char *dirname, char *typeSim)
{	
	FILE *file;
	char filename[TAM_VETOR_FILENAME];
	sprintf(filename,"%s//%s-simulationinfo.txt",dirname, typeSim);
	
	file = fopen(filename,"r");
	if (file==NULL)
	{		
		file = fopen(filename, "w");
		fprintf(file,"***Experimento %s***", typeSim);
		fprintf(file,"\nData_%s",__DATE__);
		if(numThreads > 1){
		fprintf(file,"\nNúmero de Threads:%d", numThreads);
		fprintf(file,"\nProporção GPU:%.1f", GPUTime*100);
		fprintf(file,"\nProporção CPU:%.1f", (1.0-GPUTime)*100);
		}
		fprintf(file,"\nTemperatura Atmosférica:%.1f", temperaturaAtmosferica);
		fprintf(file,"\nAltura da Nuvem:%.1f", alturaNuvem);
		fprintf(file,"\nPonto de Orvalho:%f", pontoOrvalho);
		fprintf(file,"\nPressao:%.1f", pressaoAtmosferica);
		fprintf(file,"\nCondutividade térmica:%f", K);
		fprintf(file,"\nDeltaT:%f", deltaT);		
		fprintf(file,"\nNúmero de Iterações:%d", numero_iteracoes);
		fprintf(file,"\nTamanho da Grid:%dX%d", linha, coluna);
		fprintf(file,"\nRaio da nuvem:%d", raio_nuvem);
		fprintf(file,"\nNúmero de Processadores do Computador:%d", omp_get_num_procs());
		fprintf(file,"\nDelta Ponto de Orvalho:%f", DELTAPO);
		fprintf(file,"\nLimite Inferior Ponto de Orvalho:%lf", (pontoOrvalho - DELTAPO));
		fprintf(file,"\nLimite Superior Ponto de Orvalho:%lf", (pontoOrvalho + DELTAPO));
		fprintf(file,"\nMenu Option:%d", menu_option);
		fprintf(file,"\nWrite Step:%d", write_step);
		
		fclose(file);
	}
	else
	{
		char filename_old[TAM_VETOR_FILENAME];
		string line;
		int posicao;
		char buffer [33];

		sprintf(filename_old,"%s//file_temp.txt",dirname);
		ofstream outFile(filename_old, ios::out);
		ifstream fileread(filename);

	    	while(!fileread.eof())
		{
			getline(fileread, line);
			posicao = line.find("Threads:");
			if (posicao!= string::npos)
			{
				string line_temp = line.substr(posicao+1, line.size());
				sprintf (buffer, "%d", numThreads);
				posicao = line_temp.find(buffer);
				if (posicao!= string::npos)
				{
					outFile << line << '\n';
				}
				else
				{
					sprintf (buffer, ",%d", numThreads);
					line.append(buffer);
					outFile << line << '\n';
				}
			}
			else
				outFile << line << '\n';
		}
		remove(filename);
		rename(filename_old, filename);
		outFile.close();
		fileread.close();
	}	
}

/* Write grid wind in text file */
void WriteGridWind(Cloud cloud, int linha, int coluna, char *dirname_windx, char *dirname_windy)
{
	FILE *file_windx, *file_windy;
	char filename_windx[TAM_VETOR_FILENAME];
	char filename_windy[TAM_VETOR_FILENAME];

	sprintf(filename_windx,"%s//windX.txt",dirname_windx);
	sprintf(filename_windy,"%s//windY.txt",dirname_windy);
	file_windx = fopen(filename_windx,"r");
	file_windy = fopen(filename_windy, "r");

        if (file_windx == NULL && file_windy == NULL)
	{
		file_windx = fopen(filename_windx, "w");
        	file_windy = fopen(filename_windy, "w");
		for(int i = 0; i < linha; i++)
		{
			for(int j = 0; j < coluna; j++)
			{
			fprintf(file_windx, "%.4f  ", cloud.wind_x(i,j));
			fprintf(file_windy, "%.4f  ", cloud.wind_y(i,j));
			}
		fprintf(file_windx, "\n");
		fprintf(file_windy, "\n");
	 	}
	}
	fclose(file_windx);
        fclose(file_windy);
}


int main(int argc, char **argv){
	int linha, coluna, i, j, numero_iteracoes, raio_nuvem, menu_option, write_step, GPUBlockSize, numCPUThreads;
	float temperaturaAtmosferica, pressaoAtmosferica, pontoOrvalho, limInfPO, limSupPO, deltaT, GPUTime;
	//float alturaNuvem;
	if (argc != 8){
		printf ("Wrong number of parameters.\n");
		//printf ("Usage: cloudsim Numero_Iteraoes Linha Coluna Raio_Nuvem Temperatura_Atmosferica Altura_Nuvem Pressao_Atmosferica Delta_T GPUTIME GPUBLOCKS CPUTHREADS Menu_Option Write_Step\n");
		printf ("Usage: cloudsim WIDTH HEIGHT ITERATIONS GPUTIME GPUBLOCKS CPUTHREADS OUTPUT_WRITE_FLAG\n");
		exit (-1);
	}
	//20 -3 5.0 700.0 0.001 1.0 32 12 0 10
	
	coluna = atoi(argv[1]);
	linha = atoi(argv[2]);
	numero_iteracoes = atoi(argv[3]);
	GPUTime = atof(argv[4]);
	GPUBlockSize = atoi(argv[5]);
	numCPUThreads = atoi(argv[6]);
	menu_option = atoi(argv[7]);
	
	raio_nuvem = 20; 				//atoi(argv[4]);
	temperaturaAtmosferica = -3.0f; 	//atof(argv[5]);
	//alturaNuvem = 5.0; 				//atof(argv[6]);
	pressaoAtmosferica =  700.0f;		//atof(argv[7]);
	deltaT = 0.01f;					//atof(argv[8]);
	
	numThreads = numCPUThreads;
	write_step = 10;				//atoi(argv[13]);
	
	global_write_step = write_step;
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
	
	/*
	if(numCPUThreads > 1)
		typeSim = "parallel";
	else
		typeSim = "sequential";
	*/	
	omp_set_num_threads(numCPUThreads);

	/* Inicialização da matriz de entrada com a temperatura ambiente */
	#pragma omp parallel for private (i,j)
	for (i = 0; i < linha; i++){		
		for (j = 0; j < coluna; j++){
			inputGrid(i,j) = temperaturaAtmosferica;
			//outputGrid(i,j) = temperaturaAtmosferica;
		}
	}	
	/* Inicialização dos ventos Latitudinal(Wind_X) e Longitudinal(Wind_Y) */
	for( i = 0; i < linha; i++ )
	{
		for(j = 0; j < coluna; j++ )
		{			
			cloud.wind_x(i,j) = (WIND_X_BASE - DISTURB) + (float)rand()/RAND_MAX * 2 * DISTURB;
			cloud.wind_y(i,j) = (WIND_Y_BASE - DISTURB) + (float)rand()/RAND_MAX * 2 * DISTURB;		
		}
	}

	/* Inicialização de uma nuvem no centro da matriz de entrada */
	int y, x0 = linha/2, y0 = coluna/2;
	srand(1);
	for(i = x0 - raio_nuvem; i < x0 + raio_nuvem; i++)
	{
		 // Equação da circunferencia: (x0 - x)² + (y0 - y)² = r²
		y = (int)((floor(sqrt(pow((float)raio_nuvem, 2.0) - pow(((float)x0 - (float)i), 2)) - y0) * -1));

		for(int j = y0 + (y0 - y); j >= y; j--)
		{
			inputGrid(i,j) = limInfPO + (float)rand()/RAND_MAX * (limSupPO - limInfPO);
			//outputGrid(i,j) = limInfPO + (float)rand()/RAND_MAX * (limSupPO - limInfPO);
		}
	}
	
	Stencil2D<Array2D<float>, Mask2D<float>, Cloud> stencilCloud(inputGrid, outputGrid, mask, cloud);
	
	#ifdef PSKEL_PAPI
		if(GPUTime < 1.0)
			PSkelPAPI::init(PSkelPAPI::CPU);
	#endif
	
	hr_timer_t timer;
	hrt_start(&timer);
	
	if(GPUTime == 0.0){
		//stencilCloud.runIterativeCPU(numero_iteracoes, numCPUThreads);
		#ifdef PSKEL_PAPI
			for(unsigned int i=0;i<NUM_GROUPS_CPU;i++){
				//cout << "Running iteration " << i << endl;
				stencilCloud.runIterativeCPU(numero_iteracoes, numCPUThreads, i);	
			}
		#else
			//cout<<"Running Iterative CPU"<<endl;
			if(numCPUThreads == 1)
				stencilCloud.runSequential();
			else
				stencilCloud.runIterativeCPU(numero_iteracoes, numCPUThreads);	
		#endif
	}
	else if(GPUTime == 1.0){
		stencilCloud.runIterativeGPU(numero_iteracoes, GPUBlockSize);
	}
	else{
		//stencilCloud.runIterativePartition(numero_iteracoes, GPUTime, numCPUThreads,GPUBlockSize);
		#ifdef PSKEL_PAPI
			for(unsigned int i=0;i<NUM_GROUPS_CPU;i++){
				stencilCloud.runIterativePartition(numero_iteracoes, GPUTime, numCPUThreads,GPUBlockSize,i);
			}
		#else
			stencilCloud.runIterativePartition(numero_iteracoes, GPUTime, numCPUThreads,GPUBlockSize);
		#endif
	}
	
	
	hrt_stop(&timer);
	cout << "Exec_time\t" << hrt_elapsed_time(&timer) << endl;
	
	#ifdef PSKEL_PAPI
		cudaDeviceReset();
		if(GPUTime < 1.0){
			PSkelPAPI::print_profile_values(PSkelPAPI::CPU);
			PSkelPAPI::shutdown();
		}
	#endif
	
	if(menu_option == 1){
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
	
	
	//Runtime< Stencil2D<Array2D<float>, Mask2D<float>, Cloud> > runStencil(&stencilCloud);
	
	/*
	printf("\n\nExecutando a Simulacao...\n");
	printf("Grid: %dx%d\nIterações:%d\nDelta:%f\nProporção de CPU:%.0f%%\nProporção de GPU:%.0f%%\n\n", linha,coluna,numero_iteracoes,deltaT,(1.0-GPUTime)*100,GPUTime*100);
	*/
	
	/* Criar diretório para escrever arquivos da simulação  */
	/*
	mkdir("resultados", S_IRWXU|S_IRGRP|S_IXGRP);
	mkdir("resultados//pskel", S_IRWXU|S_IRGRP|S_IXGRP);
	if (GPUTime == 1.0)
	   sprintf(maindir, "resultados//pskel//gpu");
	else 
	    if (GPUTime == 0.0 && numCPUThreads > 1)
	       sprintf(maindir, "resultados//pskel//tbb");
//	else callback
	else
            if (GPUTime == 0.0 && numCPUThreads == 1)
               sprintf(maindir, "resultados//pskel//seq");
	else
	   sprintf(maindir, "resultados//pskel//hibrido-%0.f%%gpu-%0.f%%cpu", GPUTime*100, (1.0-GPUTime)*100);
	mkdir(maindir, S_IRWXU|S_IRGRP|S_IXGRP);
	
	sprintf(dirname,"%s//MO-%d_Experimento_matriz-%dx%d-%d-iteracoes_delta_t-%f",maindir,menu_option, linha, coluna, numero_iteracoes, deltaT);
	mkdir(dirname, S_IRWXU|S_IRGRP|S_IXGRP);
	
	switch(menu_option)
	{
	case 0:
		start_time = omp_get_wtime();
		//runStencil.runIterator(numero_iteracoes,GPUTime, GPUBlockSize, numCPUThreads);	
		//stencilCloud.runIterativeGPU(numero_iteracoes, GPUBlockSize);
		//stencilCloud.runIterativeTilingGPU(numero_iteracoes, 1, coluna,  (linha/2), 1, GPUBlockSize);
		//stencilCloud.runIterativeSequential(numero_iteracoes);

		hr_timer_t timer;
		hrt_start(&timer);
		//stencilCloud.runIterativeTilingGPU(numero_iteracoes, coluna,  13000, 1, 1, GPUBlockSize);
		stencilCloud.runIterativeGeneticGPU(numero_iteracoes, GPUBlockSize);
		//stencilCloud.runIterativeAutoGPU(numero_iteracoes, GPUBlockSize);

		//stencilCloud.runIterativeCPU(numero_iteracoes, numCPUThreads);
		hrt_stop(&timer);

		end_time = omp_get_wtime();
	
		cout << hrt_elapsed_time(&timer) << endl;
		break;
	
	case 1:
		// Criar diretório para escrever as matrizes de temperatura
        	sprintf(dirMatrix_temp,"%s//matrizes_temp",dirname);
        	mkdir(dirMatrix_temp, S_IRWXU|S_IRGRP|S_IXGRP);
        	sprintf(dirMatrix_stat,"%s//matrizes_statistics",dirname);
			mkdir(dirMatrix_stat, S_IRWXU|S_IRGRP|S_IXGRP);
        	
		*//*start_time = omp_get_wtime();
		
		runStencil.runIterator(numero_iteracoes,GPUTime, GPUBlockSize, numCPUThreads);//executa iterativamente o kernel da aplicação
		
		end_time = omp_get_wtime();
		// Escrever matrizes de temperatura em arquivo texto 
                //WriteGridTemp(inputGrid, linha, coluna, 0, numCPUThreads, dirMatrix_temp, typeSim); //Escreve a Matriz de Entrada(inputGrid)
                //WriteGridTemp(outputGrid, linha, coluna, numero_iteracoes, numCPUThreads, dirMatrix_temp, typeSim); //Escreve a Matriz de Saída(outputGrid)*/
		/*
		global_count_write_step = 0;
		//int i=0;
		start_time = omp_get_wtime();
		printf("iterations: %d; GPU partition: %f; blockSize: %d; threads: %d;\n", numero_iteracoes, GPUTime, GPUBlockSize, numCPUThreads);
		//void runIteratorTilingGPU(int iterations, int tilingWidth, int tilingHeight, int tilingDepth, int _BlockSize){
		//runStencil.runIteratorTilingGPU(numero_iteracoes, coluna,  (linha/2), 1, GPUBlockSize);
		//void runIteratorHybrid(int iterations, float _GPUTime, int _blockSize, int _TBBNumThreads);
		//runStencil.runIteratorHybrid(numero_iteracoes, 0.7, GPUBlockSize, numCPUThreads);
		//runStencil.runIterator(numero_iteracoes,GPUTime, GPUBlockSize, numCPUThreads);

		//stencilCloud.runIterativeCPU(numero_iteracoes, numCPUThreads);
                //WriteGridTemp(inputGrid, linha, coluna, i, numCPUThreads, dirMatrix_temp, typeSim); //Escreve a Matriz de Entrada(inputGrid)
		size_t gpuMemFree, gpuMemTotal;
               	for(i=0;i<numero_iteracoes-1;i++){
                       	
			stencilCloud.runCPU(numCPUThreads);
			//runStencil.run(GPUTime, GPUBlockSize, numCPUThreads);			 
			//runStencil.runTilingGPU(500, 100, 1, GPUBlockSize);			 
			//void runTilingGPU(int tilingWidth, int tilingHeight, int tilingDepth, int _BlockSize);
			cudaMemGetInfo(&gpuMemFree, &gpuMemTotal);
                        if( i == count_write_step)
			{
				//WriteGridTemp(outputGrid, linha, coluna, i+1, numCPUThreads, dirMatrix_temp, typeSim); //Escreve a Matriz de Saída(outputGrid)
				//printf("GPU Tiling\n");
				CalculateStatistics(outputGrid, linha, coluna, count_write_step, dirMatrix_stat, numCPUThreads, typeSim); //Cálculos Estatíticos (média e desvio padrão) na matriz de temperatura
				count_write_step += write_step;
				//printf("GPU Free memory: %d bytes\n", gpuMemFree);
			}
		
			//runStencil.run(GPUTime, GPUBlockSize, numCPUThreads);

			 /* Escrever matrizes de temperatura em arquivo texto  */
                	//WriteGridTemp(outputGrid, linha, coluna, i+1, numCPUThreads, dirMatrix_temp, typeSim); //Escreve a Matriz de Saída(outputGrid)
			//cudaDeviceSynchronize();
			//TODO swap(stencilCloud.input, stencilCloud.output);
			/* 
                       	if(i%2 == 0){
				
                               new (&stencilCloud) Stencil2D<Array2D<float>, Mask2D<float>, Cloud>(outputGrid, inputGrid, mask, cloud);
                               new (&runStencil) Runtime< Stencil2D<Array2D<float>, Mask2D<float>, Cloud> >(&stencilCloud);
                       
                       	}
                       	else{
                               new (&stencilCloud) Stencil2D<Array2D<float>, Mask2D<float>, Cloud>(inputGrid, outputGrid, mask, cloud);
                               new (&runStencil) Runtime< Stencil2D<Array2D<float>, Mask2D<float>, Cloud> >(&stencilCloud);                                
                       	}*/
          /*   	}
               	
               	//runStencil.run(GPUTime, GPUBlockSize, numCPUThreads);		
                WriteGridTemp(outputGrid, linha, coluna, numero_iteracoes, numCPUThreads, dirMatrix_temp, typeSim); //Escreve a Matriz de Entrada(inputGrid)
		CalculateStatistics(outputGrid, linha, coluna, count_write_step, dirMatrix_stat, numCPUThreads, typeSim); /*Cálculos Estatíticos (média e desvio padrão) na matriz de temperatura */
                
		//end_time = omp_get_wtime();
		
		/* Criar diretório para escrever as matrizes dos ventos Latitudinal(Wind_X) e Longitudinal(Wind_Y) */
        	//sprintf(dirMatrix_windX,"%s//matriz_windX",dirname);
	        //sprintf(dirMatrix_windY,"%s//matriz_windY",dirname);
        	//mkdir(dirMatrix_windX, S_IRWXU|S_IRGRP|S_IXGRP);
        	//mkdir(dirMatrix_windY, S_IRWXU|S_IRGRP|S_IXGRP);

        	/* Escrever as matrizes dos ventos Latitudinal(wind_X) e Longitudinal(wind_Y) */
        	//WriteGridWind(cloud, linha, coluna, dirMatrix_windX, dirMatrix_windY);
		//break;
	//}
	
	/*cout.precision(6);
	
	cout<<"Tempo Processamento: "<<end_time-start_time<<endl;
	
	//Escrever o tempo de simulação em arquivo texto
	WriteTimeSimulation(end_time - start_time, numCPUThreads, dirname, typeSim);
		
	//Escrever informações da simulação
	WriteSimulationInfo(numero_iteracoes, linha, coluna, raio_nuvem, temperaturaAtmosferica, alturaNuvem, pressaoAtmosferica, deltaT, pontoOrvalho, menu_option, write_step, numCPUThreads, GPUTime, dirname, typeSim);

			
	printf("\n\n**FIM DO EXPERIMENTO %dx%d_%d-iterações_delta_t-%f**\n\n", linha, coluna, numero_iteracoes, deltaT);
	*/
	return 0;
}

