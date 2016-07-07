#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
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
#define STR_SIZE	256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5
//#define OPEN
//#define NUM_THREAD 4

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

//int num_omp_threads;

/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations 
 * by one time step
 */
void stencilKernel(float *temp, float *result, int grid_rows, int grid_cols, int sim_time,
				   float *power, float Cap, float Rx, float Ry, float Rz, float step){
			
	for(int i = 0; i < sim_time; i++){
		for (int r = 0; r < grid_rows; r++) {
			for (int c = 0; c < grid_cols; c++) {
				float delta;	
				////*	Corner 1	*/
				//if ( (r == 0) && (c == 0) ) {
					//delta = (step / Cap) * (power[0] +
							//(temp[1] - temp[0]) / Rx +
							//(temp[grid_cols] - temp[0]) / Ry +
							//(amb_temp - temp[0]) / Rz);
				//}	/*	Corner 2	*/
				//else if ((r == 0) && (c == grid_cols-1)) {
					//delta = (step / Cap) * (power[c] +
							//(temp[c-1] - temp[c]) / Rx +
							//(temp[c+grid_cols] - temp[c]) / Ry +
							//(amb_temp - temp[c]) / Rz);
				//}	/*	Corner 3	*/
				//else if ((r == grid_rows-1) && (c == grid_cols-1)) {
					//delta = (step / Cap) * (power[r*grid_cols+c] + 
							//(temp[r*grid_cols+c-1] - temp[r*grid_cols+c]) / Rx + 
							//(temp[(r-1)*grid_cols+c] - temp[r*grid_cols+c]) / Ry + 
							//(amb_temp - temp[r*grid_cols+c]) / Rz);					
				//}	/*	Corner 4	*/
				//else if ((r == grid_rows-1) && (c == 0)) {
					//delta = (step / Cap) * (power[r*grid_cols] + 
							//(temp[r*grid_cols+1] - temp[r*grid_cols]) / Rx + 
							//(temp[(r-1)*grid_cols] - temp[r*grid_cols]) / Ry + 
							//(amb_temp - temp[r*grid_cols]) / Rz);
				//}	/*	Edge 1	*/
				//else if (r == 0) {
					//delta = (step / Cap) * (power[c] + 
							//(temp[c+1] + temp[c-1] - 2.0*temp[c]) / Rx + 
							//(temp[grid_cols+c] - temp[c]) / Ry + 
							//(amb_temp - temp[c]) / Rz);
				//}	/*	Edge 2	*/
				//else if (c == grid_cols-1) {
					//delta = (step / Cap) * (power[r*grid_cols+c] + 
							//(temp[(r+1)*grid_cols+c] + temp[(r-1)*grid_cols+c] - 2.0*temp[r*grid_cols+c]) / Ry + 
							//(temp[r*grid_cols+c-1] - temp[r*grid_cols+c]) / Rx + 
							//(amb_temp - temp[r*grid_cols+c]) / Rz);
				//}	/*	Edge 3	*/
				//else if (r == grid_rows-1) {
					//delta = (step / Cap) * (power[r*grid_cols+c] + 
							//(temp[r*grid_cols+c+1] + temp[r*grid_cols+c-1] - 2.0*temp[r*grid_cols+c]) / Rx + 
							//(temp[(r-1)*grid_cols+c] - temp[r*grid_cols+c]) / Ry + 
							//(amb_temp - temp[r*grid_cols+c]) / Rz);
				//}	/*	Edge 4	*/
				//else if (c == 0) {
					//delta = (step / Cap) * (power[r*grid_cols] + 
							//(temp[(r+1)*grid_cols] + temp[(r-1)*grid_cols] - 2.0*temp[r*grid_cols]) / Ry + 
							//(temp[r*grid_cols+1] - temp[r*grid_cols]) / Rx + 
							//(amb_temp - temp[r*grid_cols]) / Rz);
				//}	/*	Inside the chip	*/
				//else {
					//delta = (step / Cap) * (power[r*grid_cols+c] + 
							//(temp[(r+1)*grid_cols+c] + temp[(r-1)*grid_cols+c] - 2.0*temp[r*grid_cols+c]) / Ry + 
							//(temp[r*grid_cols+c+1] + temp[r*grid_cols+c-1] - 2.0*temp[r*grid_cols+c]) / Rx + 
							//(amb_temp - temp[r*grid_cols+c]) / Rz);
				//}
				/*	Update Temperatures	*/
				result[r*grid_cols+c] = temp[r*grid_cols+c] + delta;
			}
		}
		swap(result,temp);
	}
	swap(result,temp);
	if(sim_time%2==0)
	   memcpy(temp,result,grid_rows*grid_cols*sizeof(float));
}

/* Transient solver driver routine: simply converts the heat 
 * transfer differential equations to difference equations 
 * and solves the difference equations by iterating
 */
void compute_tran_temp(float *result, int sim_time, float *temp, float *power, int row, int col) 
{
	#ifdef VERBOSE
	int i = 0;
	#endif

	float grid_height = chip_height / row;
	float grid_width = chip_width / col;

	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);

	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float step = PRECISION / max_slope;

	#ifdef VERBOSE
	fprintf(stdout, "total iterations: %d s\tstep size: %g s\n", sim_time, step);
	fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);
	#endif

	#ifdef VERBOSE
	//fprintf(stdout, "iteration %d\n", i++);
	#endif

	#pragma pskel stencil dim2d(row,col) \
		inout(temp,result) \
		iterations(sim_time) device(cpu)
	stencilKernel(temp, result, row, col, sim_time, power, Cap, Rx, Ry, Rz, step);	

	#ifdef VERBOSE
	//fprintf(stdout, "iteration %d\n", i++);
	#endif
}

void fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);
	exit(1);
}

void read_input(float *vect, int grid_rows, int grid_cols, char *file)
{
  	int i, index;
	FILE *fp;
	char str[STR_SIZE];
	float val;

	fp = fopen (file, "r");
	if (!fp)
		fatal ("file could not be opened for reading");

	for (i=0; i < grid_rows * grid_cols; i++) {
		fgets(str, STR_SIZE, fp);
		if (feof(fp))
			fatal("not enough lines in file");
		if ((sscanf(str, "%lf", &val) != 1) )
			fatal("invalid file format");
		vect[i] = val;
	}

	fclose(fp);	
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of threads><temp_file> <power_file>\n", argv[0]);
	fprintf(stderr, "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
	fprintf(stderr, "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
	fprintf(stderr, "\t<sim_time>   - number of iterations\n");
	fprintf(stderr, "\t<no. of threads>   - number of threads\n");
	fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
	fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
	exit(1);
}

int main(int argc, char **argv)
{
	int grid_rows;
	int grid_cols;
	int sim_time; 
	float *temp;
	float *power;
	float *result;
	char *tfile, *pfile;
	int num_omp_threads;
	
	/* check validity of inputs	*/
	if (argc != 7)
		usage(argc, argv);
	if ((grid_rows = atoi(argv[1])) <= 0 ||
		(grid_cols = atoi(argv[2])) <= 0 ||
		(sim_time = atoi(argv[3])) <= 0 || 
		(num_omp_threads = atoi(argv[4])) <= 0
		)
		usage(argc, argv);

	/* allocate memory for the temperature and power arrays	*/
	temp = (float *) calloc (grid_rows * grid_cols, sizeof(float));
	power = (float *) calloc (grid_rows * grid_cols, sizeof(float));
	result = (float *) calloc (grid_rows * grid_cols, sizeof(float));
	if(!temp || !power)
		fatal("unable to allocate memory");

	/* read initial temperatures and input power	*/
	tfile = argv[5];
	pfile = argv[6];
	read_input(temp, grid_rows, grid_cols, tfile);
	read_input(power, grid_rows, grid_cols, pfile);


	float grid_height = chip_height / row;
	float grid_width = chip_width / col;

	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);

	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float step = PRECISION / max_slope;
	
	printf("Start computing the transient temperature\n");
	
	#pragma pskel stencil dim2d(grid_rows,grid_cols) \
		inout(temp,result) \
		iterations(sim_time) device(cpu)
	stencilKernel(temp, result, grid_rows, grid_cols, sim_time, power, Cap, Rx, Ry, Rz, step);	

	//#pragma pskel stencil dim2d(grid_rows,grid_cols) \
		//inout(temp,result) \
		//iterations(sim_time) device(cpu)
	//stencilKernel(temp, result, grid_rows,grid_cols, sim_time);	

	//compute_tran_temp(result,sim_time, temp, power, grid_rows, grid_cols);
	
	//printf("Ending simulation\n");
	/* output results	*/
//#ifdef VERBOSE
	//fprintf(stdout, "Final Temperatures:\n");
//#endif

//#ifdef OUTPUT
	//for(i=0; i < grid_rows * grid_cols; i++)
	//fprintf(stdout, "%d\t%g\n", i, temp[i]);
//#endif
	/* cleanup	*/
	free(temp);
	free(power);
	free(result);

	return 0;
}
