#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <sched.h>
#include <unistd.h>
#include <mppaipc.h>
//#include <mppa/osconfig.h>
/**Problema com o omp.h**/

//#include "../../include/mppaStencil.h"
//#include "../../include/PSkelMask.h"
//#include "../../include/PSkelDefs.h"
//#include "../../include/mppaArray.h"
#define PSKEL_MPPA
#define MPPA_MASTER
#define ARGC_SLAVE 4
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


int CalcSize(int level){
	if (level == 1) {
		return 3;
	}
	if (level >= 1) {
		return CalcSize(level-1) + 2;
	}
	return 0;
}

void spawn_slaves(const char slave_bin_name[], int nb_clusters, int nb_threads) 
{
 	  int i;
	  int cluster_id;
	  int pid;
	  // Prepare arguments to send to slaves
	  char **argv_slave = (char**) malloc(sizeof (char*) * ARGC_SLAVE);
	  for (i = 0; i < ARGC_SLAVE - 1; i++)
	    argv_slave[i] = (char*) malloc (sizeof (char) * 10);
	  
	  sprintf(argv_slave[0], "%d", nb_clusters);
	  sprintf(argv_slave[1], "%d", nb_threads);
	  argv_slave[3] = NULL;
	  
	  // Spawn slave processes
	  for (cluster_id = 0; cluster_id < nb_clusters; cluster_id++) {
	    sprintf(argv_slave[2], "%d", cluster_id);
	  	printf("Hello!\n");
	    pid = mppa_spawn(cluster_id, NULL, "slave", (const char **)argv_slave, NULL);
	    assert(pid >= 0);
	  }
  
  // Free arguments
  for (i = 0; i < ARGC_SLAVE; i++)
    free(argv_slave[i]);
  free(argv_slave);
}

int main(int argc, char **argv){ 
	int width,height,iterations; //stencil size
	int umCPUThreads;
	int internCircle, externCircle, level,size; //mask
	double power;
	int nb_clusters=1, nb_threads=1;
	int pid;
  
	power = 2;
	width = 512; height=512; iterations=1;
  
	internCircle = 2;//pow(CalcSize(level), 2) - 1;
	externCircle = 2;//pow(CalcSize(2*level), 2) - 1 - internCircle;
	size = internCircle + externCircle;
	
	//Mask configuration
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
	Array2D<int> inputGrid(width,height);
	printf("Criou Array Input\n");
	Array2D<int> outputGrid(width,height);
	printf("Criou Array Input\n");

	Arguments arg;
	arg.power = power;
	arg.internCircle = internCircle;
	arg.externCircle = externCircle;

	srand(123456789);
	for(int h=0;h<height;h++) {
		for(int w=0;w<width;w++) {
			inputGrid(h,w) = rand()%2;
			//printf("In position %d, %d we have %d\n", h, w, inputGrid(h,w));
		}
	}

	printf("Stencil\n");
	Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(inputGrid, outputGrid, mask, arg);
	printf("EndStencil\n");
	
	/** Alyson: talves mudar para stencil.scheduleMPPA() para o codigo
	* do master e stencil.runMPPA() para o codigo do slave?
	* Emmanuel: Faz sentido, fica melhor organizado.
	*/
	
	printf("Begin inputGrid\n");
	//spawn_slaves("slave", nb_clusters, nb_threads);
	stencil.scheduleMPPA("slave", nb_clusters, nb_threads);

	//for (pid = 0; pid < nb_clusters; pid++) {
    // 	mppa_waitpid(pid, NULL, 0);
	//}
	printf("End inputGrid\n");
	//spawn_slaves("slave", nb_clusters, nb_threads);
	/** Alyson: necessario? 
	**/
	inputGrid.mppaFree();
	outputGrid.mppaFree();
}
