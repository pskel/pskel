#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define PSKEL_MPPA
#define MPPA_SLAVE
#define BARRIER_SYNC_MASTER "/mppa/sync/128:1"
#define BARRIER_SYNC_SLAVE "/mppa/sync/[0..15]:2"
//#include "common.h"
//#include "../../include/interface_mppa.h"
#include "../../include/PSkel.h"

using namespace std;
using namespace PSkel;

struct Arguments
{
  int externCircle;
  int internCircle;
  float power;
};

namespace PSkel{
  __parallel__ void stencilKernel(Array2D<int> input,Array2D<int> output,Mask2D<int> mask, Arguments arg, size_t h, size_t w){
    int numberA = 0;
    int numberI = 0;
    for (int z = 0; z < mask.size; z++) {
      if(z < arg.internCircle) {
        numberA += mask.get(z, input, h, w);

      } else {
        numberI += mask.get(z, input, h, w);
      }
    }
    float totalPowerI = numberI*(arg.power);// The power of Inhibitors
    if(numberA - totalPowerI < 0) {
      output(h,w) = 0; //without color and inhibitor
    } else if(numberA - totalPowerI > 0) {
      output(h,w) = 1;//with color and active
    } else {
      output(h,w) = input(h,w);//doesn't change
    }
  }
}


int CalcSize(int level){
  if (level == 1) {
    return 3;
  }
  if (level >= 1) {
    return CalcSize(level-1) + 2;
  }
  return 0;
}


int main(int argc,char **argv) {

  int nb_clusters = atoi(argv[0]);
  int nb_threads  = atoi(argv[1]);
  int cluster_id  = atoi(argv[2]);

   /**************Mask for test porpuses****************/
   int level = 1;
   int power = 2;
   int internCircle = pow(CalcSize(level), 2) - 1;
   int externCircle = pow(CalcSize(2*level), 2) - 1 - internCircle;
   int size = internCircle + externCircle;
   Mask2D<int> mask(size);
   int count = 0;
   for (int x = (level-2*level); x <= level; x++) {
     for (int y = (level-2*level); y <= level; y++) {
       if (x != 0 || y != 0) {
           mask.set(count, x, y);
           count++;
       }
     }
   }
 
   for (int x = (2*level-4*level); x <= 2*level; x++) {
     for (int y = (2*level-4*level); y <= 2*level; y++) {
       if (x != 0 || y != 0) {
         if (!(x <= level && x >= -1*level && y <= level && y >= -1*level)) {
           mask.set(count, x, y);
           count++;
         }
       }
     }
   }
   /*************************************************/

   /*********************Arg************************/
   Arguments arg;
   arg.power = power;
   arg.internCircle = internCircle;
   arg.externCircle = externCircle;
   /***********************************************/

   Array2D<int> input(4,4);
   Array2D<int> output(4,4);

   barrier_t *global_barrier = mppa_create_slave_barrier (BARRIER_SYNC_MASTER, BARRIER_SYNC_SLAVE);
   mppa_barrier_wait(global_barrier);

   input.portalReadAlloc(1);

   output.portalWriteAlloc(0);
   input.copyFrom();


   /**Emmaunel: Arg também precisa de um portal? */
   Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(input, output, mask, arg);
   stencil.runMPPA(16);
   //output.copyTo();
   
   output.copyTo();
   mppa_barrier_wait(global_barrier);

  /** Alyson: embutir isso no final método run? **/
  mppa_close_barrier(global_barrier);
  input.closePortals();
  
  mppa_exit(0);
  
  return 0;
}
