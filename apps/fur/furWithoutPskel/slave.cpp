#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include "hr_time.h"

#define PSKEL_MPPA
#define MPPA_SLAVE
#define DEBUG
// #define PRINT_OUT
// #define TIME_EXEC
// #define TIME_SEND
#define BARRIER_SYNC_MASTER "/mppa/sync/128:1"
#define BARRIER_SYNC_SLAVE "/mppa/sync/[0..15]:2"

// #include "../../../include/PSkel.h"

using namespace std;
// using namespace PSkel;

struct Arguments
{
  int externCircle;
  int internCircle;
  float power;
};

// namespace PSkel{
void stencilKernel(int* input,int* output,int* mask, int size_mask, Arguments arg, size_t h, size_t w){
    int numberA = 0;
    int numberI = 0;
    int auxH = h;
    int auxW = w;
    for (int z = 0; z < size_mask; z++) {
      if(z < arg.internCircle) {
        //numberA += mask.get(z, input, h, w);
        // auxH += mask[z];
        // auxW += mask[z+1];
        // numberA += input[(h+w)];

      } else {
        // numberI += mask.get(z, input, h, w);
        // auxH += mask[z];
        // auxW += mask[z+1];
        // numberI += input[auxH+auxW];
        //printf("I: %d\n", numberI);
      }
    }
    // h += this->hostMask[this->dimension*n];
    // w += this->hostMask[this->dimension*n+1];
    // return (w<array.getWidth() && h<array.getHeight())?array(h,w):this->haloValue;
    //printf("A: %d\n", numberA);
    float totalPowerI = numberI*(arg.power);// The power of Inhibitors
    //printf("Power of I: %f\n", totalPowerI);
    if(numberA - totalPowerI < 0) {
		output[h+w] = 0; //without color and inhibitor
    }
    else if(numberA - totalPowerI > 0) {
		output[h+w] = 1;//with color and active
    }
    else {
		output[h+w] = input[h+w];//doesn't change
    }
  // int c = 0;
  // c++;
}
// }


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
  #ifdef DEBUG
    cout << "Slave Begin!" << endl;
  #endif
  /**************Mask for test porpuses****************/
  int level = 1;
  int power = 2;
  int internCircle = pow(CalcSize(level), 2) - 1;
  int externCircle = pow(CalcSize(2*level), 2) - 1 - internCircle;
  int size = internCircle + externCircle;
  //Mask2D<int> mask(size);
  int mask[size];
  int count = 0;
  for (int x = (level-2*level); x <= level; x++) {
    for (int y = (level-2*level); y <= level; y++) {
      if (x != 0 || y != 0) {
          //mask.set(count, x, y);
          mask[(x+y)] = count;
          count++;
      }
    }
  }
 
  for (int x = (2*level-4*level); x <= 2*level; x++) {
    for (int y = (2*level-4*level); y <= 2*level; y++) {
      if (x != 0 || y != 0) {
        if (!(x <= level && x >= -1*level && y <= level && y >= -1*level)) {
          //mask.set(count, x, y);
          mask[(x+y)] = count;
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

  int nb_tiles = atoi(argv[0]);
  int width = atoi(argv[1]);
  int height = atoi(argv[2]);
  int cluster_id = atoi(argv[3]);
  int nb_threads = atoi(argv[4]);
  int iterations = atoi(argv[5]);
  int outteriterations = atoi(argv[6]);
  int itMod = atoi(argv[7]);
  

  int subIterations = 1;
  int h = 128;
  int w = 128;
  int d = 1;

  // tmp.mppaAlloc(w,h,d);
  int* tmp = (int*) calloc(h*w, sizeof(int));;
  // inputTmp.mppaAlloc(w,h,d);
  int* inputTmp = (int*) calloc(h*w, sizeof(int));;
  // outputTmp.mppaAlloc(w,h,d);
  int* outputTmp = (int*) calloc(h*w, sizeof(int));;
  #ifdef DEBUG
    cout << "Arrays allocated" << endl;
  #endif


  // this->runIterativeMPPA(inputTmp, outputTmp, subIterations, nb_threads);

  omp_set_num_threads(16);
  #pragma omp parallel for
  for (int h = 0; h < 128; h++){
  for (int w = 0; w < 128; w++){
    stencilKernel(inputTmp,outputTmp, mask, size, arg, h, w);
  }}

  // Array fTmp;

  // tmp.mppaFree();
  // tmp.auxFree();

  // inputTmp.mppaFree();
  // inputTmp.auxFree();

  // outputTmp.mppaFree();
  // outputTmp.auxFree();

  // mppa_exit(0);
}
