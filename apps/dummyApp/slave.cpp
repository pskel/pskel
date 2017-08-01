#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define PSKEL_MPPA
#define MPPA_SLAVE
// #define DEBUG
// #define PRINT_OUT
// #define TIME_EXEC
// #define TIME_SEND
#define BARRIER_SYNC_MASTER "/mppa/sync/128:1"
#define BARRIER_SYNC_SLAVE "/mppa/sync/[0..15]:2"

#include "../../include/PSkel.h"

using namespace std;
using namespace PSkel;

struct Arguments
{
  int externCircle;
  int internCircle;
  int level;
  float power;
};

namespace PSkel{
__parallel__ void stencilKernel(Array2D<int> input,Array2D<int> output,Mask2D<int> mask, Arguments arg, size_t h, size_t w){
       output=input;
       int c;
       c++;
  }
}


int main(int argc,char **argv) {

  /**************Mask for test porpuses****************/
  Mask2D<int> mask(1);
  mask.set(1,0,0);
  /*************************************************/

  /*********************Arg************************/
  Arguments arg;
  /***********************************************/

  int nb_tiles = atoi(argv[0]);
  int width = atoi(argv[1]);
  int height = atoi(argv[2]);
  int cluster_id = atoi(argv[3]);
  int nb_threads = atoi(argv[4]);
  int iterations = atoi(argv[5]);
  int outteriterations = atoi(argv[6]);
  int itMod = atoi(argv[7]);

  Array2D<int> partInput(width, height);
  Array2D<int> output(width, height);
  Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(partInput, output, mask, arg);
  // if(iterations == 0)  {
  stencil.runMPPA(cluster_id, nb_threads, nb_tiles, outteriterations, itMod);
  // } else {
  //      stencil.runIterativeMPPA(cluster_id, nb_threads, nb_tiles, iterations);
  //}
  //stencil.~Stencil2D();
  stencil.~Stencil2D();

  printf("Exiting slave: %d", cluster_id);
  mppa_exit(0);
}
