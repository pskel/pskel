#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sched.h>
#include <unistd.h>

#include "../include/PSkel.h"

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
	int nb_clusters=16, nb_threads=1;
  
	power = 2;
	width = 512; height=512; iterations=1;
  
	internCircle = pow(CalcSize(level), 2) - 1;
	externCircle = pow(CalcSize(2*level), 2) - 1 - internCircle;
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
	Array2D<int> outputGrid(width,height);
	
	Arguments arg;
	arg.power = power;
	arg.internCircle = internCircle;
	arg.externCircle = externCircle;

	srand(123456789);
	for(int h=0;h<height;h++) {
		for(int w=0;w<width;w++) {
			inputGrid(h,w) = rand()%2;
			printf("In position %d, %d we have %d\n", h, w, inputGrid(h,w));
		}
	}

	Stencil2D<Array2D<int>, Mask2D<int>, Arguments> stencil(inputGrid, outputGrid, mask, arg);
	
	/** Alyson: talves mudar para stencil.scheduleMPPA() para o codigo
	* do master e stencil.runMPPA() para o codigo do slave?
	*/
	stencil.runMPPA("slave",nb_clusters,nb_slaves);
	
	/** Alyson: necessario? **/
	inputGrid.hostFree();
	outputGrid.hostFree();
}
