#define PSKEL_OMP

#include "include/PSkel.h"
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
__parallel__ void stencilKernel(Array2D<int> input,Array2D<int> output,Mask2D<int> mask, size_t args,size_t j, size_t i)
{
#define  width input.getWidth()
#define height input.getHeight()
int neighbors = 0;
for (int y = -1; y <= 1; y++)
{
  for (int x = -1; x <= 1; x++)
  {
    if (((((i + x) >= 0) && ((i + x) < width)) && ((j + y) >= 0)) && ((j + y) < height))
    {
      neighbors += input(j + y,i + x);
    }

  }

}

if ((neighbors == 3) || ((input(j,i) == 1) && (neighbors == 2)))
{
  output(j,i) = 1;
}
else
{
  output(j,i) = 0;
}





#undef  width
#undef height
}
}

int main(int argc, char **argv)
{
  int x_max;
  int y_max;
  int T_MAX;
  ;
  ;
  if (argc != 4)
  {
    printf("Wrong number of parameters.\n");
    printf("Usage: gol WIDTH HEIGHT ITERATIONS\n");
    exit(-1);
  }

  x_max = atoi(argv[1]);
  y_max = atoi(argv[2]);
  T_MAX = atoi(argv[3]);
  Array2D<int> inputGrid(x_max,y_max);
  Array2D<int> outputGrid(x_max,y_max);
  srand(123456789);
  for (int j = 0; j < y_max; j++)
  {
    for (int i = 0; i < x_max; i++)
    {
      inputGrid(j,i) = rand() % 2;
    }

  }

  Mask2D<int> _pskelcc_stencil_1498_1553_mask;
  Stencil2D<Array2D<int>, Mask2D<int>, size_t> _pskelcc_stencil_1498_1553(inputGrid, outputGrid, _pskelcc_stencil_1498_1553_mask, 0);
  _pskelcc_stencil_1498_1553.runIterativeCPU(T_MAX, 0);
  return 0;
}


