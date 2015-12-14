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
  // Global data
  /** Alyson: necessario?**/
  //char path[25]; //Alyson: embutir?
  //int i;
  //char *comm_buffer = (char *) malloc(MAX_BUFFER_SIZE);
  //assert(comm_buffer != NULL);
  
  /** Alyson: necessario?**/
  /** Emmanuel: Nao é necessario inicializar**/
  //  for(i = 0; i < MAX_BUFFER_SIZE; i++)
  //  comm_buffer[i] = 0;
  
  // Set initial parameters
  /** Alyson: o mais indicado seria tirar do argv mesmo? **/
  //--int nb_clusters = atoi(argv[0]);
  //--int nb_threads  = atoi(argv[1]);
  //--int cluster_id  = atoi(argv[2]);
  // Initialize global barrier
  /** Alyson: aonde colocar esta barreira? Variavel global na classe Stencil? **/
  //--barrier_t *global_barrier = mppa_create_slave_barrier (BARRIER_SYNC_MASTER, BARRIER_SYNC_SLAVE);
  
  /** Alyson: pensei em definir arrays vazios para cada variavel e 
   * fazer a leitura dos valores do master na inicialização/construtor da classe Stencil
   * Ex: Array2D<float> input;
   * 	 Array2D<float> output;
   *     Mask2D<float> mask;
   *	 Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(inputGrid, outputGrid, mask, arg);
   * Verificar se existem macros de compilacao para diferenciar codigo master de slave
   */
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
   #ifdef MPPA_SLAVE
   printf("Sou o Escravo\n");
   #else
   printf("Sou o Mestre\n");
   #endif
   Array2D<int> input(512,512);
   Array2D<int> output(512,512);
   //barrier_t *global_barrier = mppa_create_slave_barrier (BARRIER_SYNC_MASTER, BARRIER_SYNC_SLAVE);
   //mppa_barrier_wait(global_barrier);
   //input.mppaAlloc();
   //output.mppaAlloc();
   printf("Arrived at slave!\n");
   input.portalReadAlloc(1);
   //output.portalWriteAlloc(0);
   input.copyFrom();
   //input.waitRead();
   //mask.portalReadAlloc(1);
   //mask.copyFrom();
   //mask.waitRead();


   /**Emmaunel: Arg também precisa de um portal? */
   Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(input, output, mask, arg);
   stencil.runMPPA(16);
   printf("Hello!");
   //output.copyTo();
  // Initialize communication portals
  //--sprintf(path, "/mppa/portal/%d:3", 128 + (cluster_id % 4));
  //--portal_t *write_portal = mppa_create_write_portal(path, comm_buffer, MAX_BUFFER_SIZE, 128 + (cluster_id % 4));
  
  // Initialize communication portal to receive messages from IO-node
  //--sprintf(path, "/mppa/portal/%d:%d", cluster_id, 4 + cluster_id);
  //--portal_t *read_portal = mppa_create_read_portal(path, comm_buffer, MAX_BUFFER_SIZE, 1, NULL);

  //--mppa_barrier_wait(global_barrier);
  
  
  //--LOG("Slave %d started\n", cluster_id);
  
  /** Alyson: pensei em aqui fazer apenas a chamada do run da classe Stencil
   * e o metodo run se encarrega da leitura e chamar o metodo stencilKernel.
   * Vide runCPU em PSkelStencil.hpp
   * stencil.runMPPA() com parametros a definir;
  */
  /*
  int nb_exec;
  for (nb_exec = 1; nb_exec <= NB_EXEC; nb_exec++) {
    // ----------- MASTER -> SLAVE ---------------
    for (i = 1; i <= MAX_BUFFER_SIZE; i *= 2) {
      mppa_barrier_wait(global_barrier);
      
      // Block until receive the asynchronous write and prepare for next asynchronous writes		
      mppa_async_read_wait_portal(read_portal);
    }
    
    // ----------- SLAVE -> MASTER ---------------
    for (i = 1; i <= MAX_BUFFER_SIZE; i *= 2) {
      mppa_barrier_wait(global_barrier);
      
      // post asynchronous write
      mppa_async_write_portal(write_portal, comm_buffer, i, cluster_id * MAX_BUFFER_SIZE);
      
      // wait for the end of the transfer
      mppa_async_write_wait_portal(write_portal);
    }
  }
  */
  
  /** Alyson: embutir isso no final método run? **/
  //mppa_close_barrier(global_barrier);
  input.closePortals();
  //mppa_close_portal(write_portal);
  //mppa_close_portal(read_portal);
  
  //LOG("Slave %d finished\n", cluster_id);
  
  mppa_exit(0);
  
  return 0;
}
