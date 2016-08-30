#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define PSKEL_MPPA
#define MPPA_SLAVE
// #define DEBUG
//#define BUG_TEST
// #define PRINT_OUT
#define TIME_EXEC
#define TIME_SEND
#define BARRIER_SYNC_MASTER "/mppa/sync/128:1"
#define BARRIER_SYNC_SLAVE "/mppa/sync/[0..15]:2"
//#include "common.h"
//#include "../../include/interface_mppa.h"
#include "../../include/PSkel.h"

using namespace std;
using namespace PSkel;


struct Arguments
{
  int dummy;
};


namespace PSkel{
  __parallel__ void stencilKernel(Array2D<int> input,Array2D<int> output,Mask2D<int> mask, Arguments arg, size_t h, size_t w){
      int neighbors=0;
      for(int z=0;z<mask.size;z++){
        neighbors += mask.get(z,input,h,w);
      } 
      output(h,w) = ((neighbors==3 || (input(h,w)==1 && neighbors==2))?1:0);
  }
}

int main(int argc,char **argv) {

   /**************Mask for test porpuses****************/
  Arguments arg;
  arg.dummy = 1;
  Mask2D<int> mask(8);
  mask.set(0,-1,-1);  mask.set(1,-1,0); mask.set(2,-1,1);
  mask.set(3,0,-1);       mask.set(4,0,1);
  mask.set(5,1,-1); mask.set(6,1,0);  mask.set(7,1,1);
  /*************************************************/

  // width = atoi (argv[1]);
  // height = atoi (argv[2]);
  // iterations = atoi (argv[3]);
  // mode = atoi(argv[4]);
  // GPUBlockSize = atoi(argv[5]);
  // numCPUThreads = atoi(argv[6]);
  // tileHeight = atoi(argv[7]);
  // tileIterations = atoi(argv[8]);
  
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

}
