#define PSKEL_OMP

#include "include/PSkel.h"
using namespace PSkel;

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

#include "../../pskel/include/hr_time.h"

using namespace std;

#define WIND_X_BASE	15
#define WIND_Y_BASE	12
#define DISTURB		0.1f
#define CELL_LENGTH	0.1f
#define K           	0.0243f
#define DELTAPO       	0.5f
#define TAM_VETOR_FILENAME  200

namespace PSkel{
__parallel__ void stencilKernel(Array2D<float> input,Array2D<float> output,Mask2D<float> mask, size_t args,size_t j, size_t i)
{
#define  width input.getWidth()
#define height input.getHeight()
int numNeighbor = 0;
float sum = 0.0f;
float inValue = input(i,j);
float temperatura_vizinho = 0.0f;
int factor = 0;
for (int y = -1; y < 1; y++)
{
  for (int x = -1; x < 1; x++)
  {
    if ((x != 0) && (y != 0))
    {
      temperatura_vizinho = input(j + y,i + x);
      factor = temperatura_vizinho == 0 ? 0 : 1;
      sum += factor * (inValue - temperatura_vizinho);
      numNeighbor += factor;
    }

  }

}

float temperatura_conducao = ((-K) * (sum / numNeighbor)) * deltaT;
float result = inValue + temperatura_conducao;
float xwind = wind_x[(j * width) + i];
float ywind = wind_y[(j * width) + i];
int xfactor = xwind > 0 ? 1 : -1;
int yfactor = ywind > 0 ? 1 : -1;
float temperaturaNeighborX = input(j + xfactor,i);
float componenteVentoX = xfactor * xwind;
float temperaturaNeighborY = input(j,i + yfactor);
float componenteVentoY = yfactor * ywind;
float temp_wind = ((-componenteVentoX) * ((inValue - temperaturaNeighborX) / CELL_LENGTH)) - (componenteVentoY * ((inValue - temperaturaNeighborY) / CELL_LENGTH));
output(j,i) = result + (numNeighbor == 4 ? temp_wind * deltaT : 0.0f);




#undef  width
#undef height
}
}



float Convert_Celsius_To_Kelvin(float number_celsius)
{
	float number_kelvin;
	number_kelvin = number_celsius + 273.15f;
	return number_kelvin;
}


float Convert_hPa_To_mmHg(float number_hpa)
{
	float number_mmHg;
	number_mmHg = number_hpa * 0.750062f;

	return number_mmHg;
}


float Convert_milibars_To_mmHg(float number_milibars)
{
	float number_mmHg;
	number_mmHg = number_milibars * 0.750062f;

	return number_mmHg;
}


float CalculateRPV(float temperature_Kelvin, float pressure_mmHg)
{
	float realPressureVapor; 
	float PsychrometricConstant = 6.7f * powf(10,-4); 
	float PsychrometricDepression = 1.2f; 
	float esu = pow(10, ((-2937.4f / temperature_Kelvin) - 4.9283f * log10(temperature_Kelvin) + 23.5470f)); 
	realPressureVapor = Convert_milibars_To_mmHg(esu) - (PsychrometricConstant * pressure_mmHg * PsychrometricDepression);

	return realPressureVapor;
}


float CalculateDewPoint(float temperature_Kelvin, float pressure_mmHg)
{
	float dewPoint; 
	float realPressureVapor = CalculateRPV(temperature_Kelvin, pressure_mmHg); 
	dewPoint = (186.4905f - 237.3f * log10(realPressureVapor)) / (log10(realPressureVapor) -8.2859f);

	return dewPoint;
}
int main(int argc, char **argv)
{
  int linha;
  int coluna;
  int i;
  int j;
  int T_MAX;
  int raio_nuvem;
  int menu_option;
  int write_step;
  int GPUBlockSize;
  int numCPUThreads;
  float temperaturaAtmosferica;
  float pressaoAtmosferica;
  float pontoOrvalho;
  float limInfPO;
  float limSupPO;
  float deltaT;
  float GPUTime;
  ;
  ;
  float *wind_x;
  float *wind_y;
  if (argc != 8)
  {
    printf("Wrong number of parameters.\n");
    printf("Usage: cloudsim WIDTH HEIGHT ITERATIONS GPUTIME GPUBLOCKS CPUTHREADS OUTPUT_WRITE_FLAG\n");
    exit(-1);
  }

  coluna = atoi(argv[1]);
  linha = atoi(argv[2]);
  T_MAX = atoi(argv[3]);
  GPUTime = atof(argv[4]);
  GPUBlockSize = atoi(argv[5]);
  numCPUThreads = atoi(argv[6]);
  menu_option = atoi(argv[7]);
  raio_nuvem = 20;
  temperaturaAtmosferica = -3.0f;
  pressaoAtmosferica = 700.0f;
  deltaT = 0.01f;
  pontoOrvalho = CalculateDewPoint(Convert_Celsius_To_Kelvin(temperaturaAtmosferica), Convert_hPa_To_mmHg(pressaoAtmosferica));
  limInfPO = pontoOrvalho - DELTAPO;
  limSupPO = pontoOrvalho + DELTAPO;
  Array2D<float> inputGrid(coluna,linha);
  Array2D<float> outputGrid(coluna,linha);
  wind_x = (float *) malloc((coluna * linha) * (sizeof(float)));
  wind_y = (float *) malloc((coluna * linha) * (sizeof(float)));
  for (i = 0; i < linha; i++)
  {
    for (j = 0; j < coluna; j++)
    {
      inputGrid(j,i) = temperaturaAtmosferica;
      outputGrid(j,i) = temperaturaAtmosferica;
    }

  }

  for (i = 0; i < linha; i++)
  {
    for (j = 0; j < coluna; j++)
    {
      wind_x[(j * coluna) + i] = (WIND_X_BASE - DISTURB) + (((((float) rand()) / RAND_MAX) * 2) * DISTURB);
      wind_y[(j * coluna) + i] = (WIND_Y_BASE - DISTURB) + (((((float) rand()) / RAND_MAX) * 2) * DISTURB);
    }

  }

  int y;
  int x0 = linha / 2;
  int y0 = coluna / 2;
  srand(1);
  for (i = x0 - raio_nuvem; i < (x0 + raio_nuvem); i++)
  {
    y = (int) (floor(sqrt(pow((float) raio_nuvem, 2.0) - pow(((float) x0) - ((float) i), 2)) - y0) * (-1));
    for (int j = y0 + (y0 - y); j >= y; j--)
    {
      float value = limInfPO + ((((float) rand()) / RAND_MAX) * (limSupPO - limInfPO));
      inputGrid(j,i) = value;
      outputGrid(j,i) = value;
    }

  }

  Mask2D<float> _pskelcc_stencil_5334_5417_mask;
  Stencil2D<Array2D<float>, Mask2D<float>, size_t> _pskelcc_stencil_5334_5417(inputGrid, outputGrid, _pskelcc_stencil_5334_5417_mask, 0);
  _pskelcc_stencil_5334_5417.runIterativeCPU(T_MAX, 0);
  if (menu_option == 1)
  {
    cout.precision(12);
    (cout << "INPUT") << endl;
    for (int i = 10; i < coluna; i += 10)
    {
      ((((((((((((cout << "(") << i) << ",") << i) << ") = ") << inputGrid(i,i)) << "\t\t(") << (coluna - i)) << ",") << (linha - i)) << ") = ") << inputGrid((coluna*(coluna - i) - i + linha -((coluna*(coluna - i) - i + linha)%( coluna)))/coluna,((coluna*(coluna - i) - i + linha)%( coluna)))) << endl;
    }

    cout << endl;
    (cout << "OUTPUT") << endl;
    for (int i = 10; i < coluna; i += 10)
    {
      ((((((((((((cout << "(") << i) << ",") << i) << ") = ") << outputGrid(i,i)) << "\t\t(") << (coluna - i)) << ",") << (linha - i)) << ") = ") << outputGrid((coluna*(coluna - i) - i + linha -((coluna*(coluna - i) - i + linha)%( coluna)))/coluna,((coluna*(coluna - i) - i + linha)%( coluna)))) << endl;
    }

    cout << endl;
  }

  return 0;
}


