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


int main(int argc, char **argv){ 
	int width,height,iterations; //stencil size
	int umCPUThreads;
	int internCircle, externCircle, level,size; //mask
	double power;
	int nb_clusters=1, nb_threads=1;
	int pid;
  	
  	level = 1;
	power = 2;
	width = 4; height= 4; iterations=1;
  
	internCircle = pow(CalcSize(level), 2) - 1;
	externCircle = pow(CalcSize(2*level), 2) - 1 - internCircle;
	size = internCircle + externCircle;
	
	//Mask configuration
	Mask2D<int> mask(size);

	Array2D<int> inputGrid(width,height);
	Array2D<int> outputGrid(width,height);

	Arguments arg;
	arg.power = power;
	arg.internCircle = internCircle;
	arg.externCircle = externCircle;

	srand(123456789);
	for(int h=0;h<height;h++) {
		for(int w=0;w<width;w++) {
			inputGrid(h,w) = rand()%2;
		}
	}

	Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(inputGrid, outputGrid, mask, arg);

     /** Alyson: talves mudar para stencil.scheduleMPPA() para o codigo
	* do master e stencil.runMPPA() para o codigo do slave?
	* Emmanuel: Faz sentido, fica melhor organizado.
	*/
	
	stencil.scheduleMPPA("slave", nb_clusters, nb_threads);

	for(int h=0;h<height;h++) {
		for(int w=0;w<width;w++) {
			printf("outputGridApplication(%d,%d):%d\n",h,w, outputGrid(h,w));
		}
	}
	inputGrid.mppaFree();
	outputGrid.mppaFree();
}
