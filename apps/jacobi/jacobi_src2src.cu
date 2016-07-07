#define PSKEL_OMP

#include "PSkel.h"
using namespace PSkel;

#include <fstream>
#include <string>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <stdlib.h>

using namespace std;

namespace PSkel{
__parallel__ void stencilKernel(Array2D<float> input,Array2D<float> output,Mask2D<float> mask, size_t args,size_t y, size_t x)
{
#define  width input.getWidth()
#define height input.getHeight()
output(x,y) = ((alpha * input(y,x)) + (beta * (((input(y + 1,x) + input(y - 1,x)) + input(y,x + 1)) + input(y,x - 1)))) - ((4 * beta) * beta);




#undef  width
#undef height
}
}

int main(int argc, char **argv)
{
  int width;
  int height;
  int T_MAX;
  ;
  ;
  float alpha;
  float beta;
  if (argc != 4)
  {
    printf("Wrong number of parameters.\n");
    printf("Usage: gol WIDTH HEIGHT ITERATIONS\n");
    exit(-1);
  }

  width = atoi(argv[1]);
  height = atoi(argv[2]);
  T_MAX = atoi(argv[3]);
  alpha = 0.25 / ((float) width);
  beta = 1.0 / ((float) height);
  Array2D<float> inputGrid(width,height);
  Array2D<float> outputGrid(width,height);
  for (int j = 0; j < height; j++)
  {
    for (int i = 0; i < width; i++)
    {
      inputGrid(j,i) = (1. + (i * 0.1)) + (j * 0.01);
    }

  }

  Mask2D<float> _pskelcc_stencil_1511_1578_mask;
  Stencil2D<Array2D<float>, Mask2D<float>, size_t> _pskelcc_stencil_1511_1578(inputGrid, outputGrid, _pskelcc_stencil_1511_1578_mask, 0);
  _pskelcc_stencil_1511_1578.runIterativeCPU(T_MAX, 0);
  return 0;
}


