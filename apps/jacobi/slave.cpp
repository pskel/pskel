#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define PSKEL_MPPA
#define MPPA_SLAVE
#define DEBUG
#define BARRIER_SYNC_MASTER "/mppa/sync/128:1"
#define BARRIER_SYNC_SLAVE "/mppa/sync/[0..15]:2"
//#include "common.h"
//#include "../../include/interface_mppa.h"
#include "../../include/PSkel.h"

using namespace std;
using namespace PSkel;


namespace PSkel{
  __parallel__ void stencilKernel(Array2D<float> input,Array2D<float> output,Mask2D<float> mask, float factor, size_t h, size_t w){
      //printf("MaskGet(0): %f\n", mask.get(0, input, h, w));
      //printf("MaskGet(1): %f\n", mask.get(1, input, h, w));
      //printf("MaskGet(2): %f\n", mask.get(2, input, h, w));
      //printf("MaskGet(3): %f\n", mask.get(3, input, h, w));
      //printf("factor: %f\n", factor);
      output(h,w) = 0.25*(mask.get(0,input,h,w) + mask.get(1,input,h,w) + 
        mask.get(2,input,h,w) + mask.get(3,input,h,w) - 4*factor*factor );
      //printf("OutputValor: %f\n", output(h,w));
    
  }
}

int main(int argc,char **argv) {

   /**************Mask for test porpuses****************/


  Mask2D<float> mask(4);
  
  mask.set(0,1,0,0);
  mask.set(1,-1,0,0);
  mask.set(2,0,1,0);
  mask.set(3,0,-1,0);

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
  int realHeight = atoi(argv[8]);
  int realWidth = atoi(argv[9]);

  float factor = 1.f/(float)realWidth;

  Array2D<float> partInput(width, height);
  Array2D<float> output(width, height);
  Stencil2D<Array2D<float>, Mask2D<float>, float> stencil(partInput, output, mask, factor);
  // if(iterations == 0)  {
  stencil.runMPPA(cluster_id, nb_threads, nb_tiles, outteriterations, itMod);
  // } else {
  //      stencil.runIterativeMPPA(cluster_id, nb_threads, nb_tiles, iterations);
  //}


}
