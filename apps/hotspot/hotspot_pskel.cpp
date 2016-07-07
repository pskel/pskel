#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PSKEL_OMP

#include "../../../pskel/include/PSkel.h"

// Returns the current system time in microseconds 
long long get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

using namespace std;
using namespace PSkel;
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
double t_chip = 0.0005;
double chip_height = 0.016;
double chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
double amb_temp = 80.0;

int num_omp_threads;

struct Arguments{	
	Array2D<double> power;
	double Cap,step,Rx,Ry,Rz,amb_temp;

	Arguments(){};
	
	Arguments(int grid_rows, int grid_cols){		
		new (&power) Array2D<double>(grid_rows, grid_cols);
	}
};

namespace PSkel{
/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations 
 * by one time step
 */
	__parallel__ void stencilKernel(Array2D<double> temp, Array2D<double> result,
				Mask2D<double> mask, Arguments args, size_t c, size_t r){
		double delta = 0.0;
		
		int row = temp.getHeight();
		int col = temp.getWidth();		
		
		/*	Corner 1	*/
		if ( (r == 0) && (c == 0) ) {
			delta = (args.step / args.Cap) * (args.power(0,0) +
					(temp(0,1) - temp(0,0)) / args.Rx +
					(temp(1,0) - temp(0,0)) / args.Ry +
					(args.amb_temp - temp(0,0)) / args.Rz);
		}	/*	Corner 2	*/
		else if ((r == 0) && (c == col-1)) {
			delta = (args.step / args.Cap) * (args.power(0,c) +
					(temp(0,c-1) - temp(0,c)) / args.Rx +
					(temp(1,c) - temp(0,c)) / args.Ry +
					(args.amb_temp - temp(0,c)) / args.Rz);
		}	/*	Corner 3	*/
		else if ((r == row-1) && (c == col-1)) {
			delta = (args.step / args.Cap) * (args.power(r,c) + 
					(temp(r,c-1) - temp(r,c)) / args.Rx + 
					(temp(r-1,c) - temp(r,c)) / args.Ry + 
					(args.amb_temp - temp(r,c)) / args.Rz);					
		}	/*	Corner 4	*/
		else if ((r == row-1) && (c == 0)) {
			delta = (args.step / args.Cap) * (args.power(r,0) + 
					(temp(r,1)   - temp(r,0)) / args.Rx + 
					(temp(r-1,0) - temp(r,0)) / args.Ry + 
					(args.amb_temp - temp(r,0)) / args.Rz);
		}	/*	Edge 1	*/
		else if (r == 0) {
			delta = (args.step / args.Cap) * (args.power(0,c) + 
					(temp(0,c+1) + temp(0,c-1) - 2.0*temp(0,c)) / args.Rx + 
					(temp(1,c) - temp(0,c)) / args.Ry + 
					(args.amb_temp - temp(0,c)) / args.Rz);
		}	/*	Edge 2	*/
		else if (c == col-1) {
			delta = (args.step / args.Cap) * (args.power(r,c) + 
					(temp(r,c-1) - temp(r,c)) / args.Rx +
					(temp(r+1,c) + temp(r-1,c) - 2.0*temp(r,c)) / args.Ry + 
					(args.amb_temp - temp(r,c)) / args.Rz);
		}	/*	Edge 3	*/
		else if (r == row-1) {
			delta = (args.step / args.Cap) * (args.power(r,c) + 
					(temp(r,c+1) + temp(r,c-1) - 2.0*temp(r,c)) / args.Rx + 
					(temp(r-1,c) - temp(r,c)) / args.Ry + 
					(args.amb_temp - temp(r,c)) / args.Rz);
		}	/*	Edge 4	*/
		else if (c == 0) {
			delta = (args.step / args.Cap) * (args.power(r,0) + 
					(temp(r,1) - temp(r,0)) / args.Rx +
					(temp(r+1,0) + temp(r-1,0) - 2.0*temp(r,0)) / args.Ry + 
					(args.amb_temp - temp(r,0)) / args.Rz);
		}	/*	Inside the chip	*/
		else {
			delta = (args.step / args.Cap) * (args.power(r,c) + 
					(temp(r,c+1) + temp(r,c-1) - 2.0*temp(r,c)) / args.Rx +
					(temp(r+1,c) + temp(r-1,c) - 2.0*temp(r,c)) / args.Ry + 
					(args.amb_temp - temp(r,c)) / args.Rz);
		}
		
		/*	Update Temperatures	*/
		result(r,c) = temp(r,c) + delta;
	}
}

void fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);
	exit(1);
}

void writeoutput(Array2D<double> vect, int grid_rows, int grid_cols, char *file) {

    int i,j, index=0;
    FILE *fp;
    char str[STR_SIZE];

    if( (fp = fopen(file, "w" )) == 0 )
        printf( "The file was not opened\n" );


    for (i=0; i < grid_rows; i++) 
        for (j=0; j < grid_cols; j++)
        {

            sprintf(str, "%d\t%g\n", index, vect(i,j));
            fputs(str,fp);
            index++;
        }

    fclose(fp);	
}

void read_input(Array2D<double> vect, int grid_rows, int grid_cols, char *file)
{
  	int i, j,index;
	FILE *fp;
	char str[STR_SIZE];
	double val;

	fp = fopen (file, "r");
	if (!fp)
		fatal ("file could not be opened for reading");

	for (i=0; i < grid_rows; i++) {
		for (j=0; j < grid_cols; j++) {
			fgets(str, STR_SIZE, fp);
			if (feof(fp))
				fatal("not enough lines in file");
			if ((sscanf(str, "%lf", &val) != 1) )
				fatal("invalid file format");
			vect(i,j) = val;
		}
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
	int grid_rows, grid_cols, sim_time, i,j;
	//double *temp, *power, *result;
	char *tfile, *pfile, *ofile;
	
	/* check validity of inputs	*/
	if (argc != 8)
		usage(argc, argv);
	if ((grid_rows = atoi(argv[1])) <= 0 ||
		(grid_cols = atoi(argv[2])) <= 0 ||
		(sim_time = atoi(argv[3])) <= 0 || 
		(num_omp_threads = atoi(argv[4])) <= 0
		)
		usage(argc, argv);

	/* allocate memory for the temperature and power arrays	*/
	Array2D<double> temp(grid_rows,grid_cols);
	Array2D<double> result(grid_rows,grid_cols);
	Mask2D<double> mask(4);
	mask.set(0,0,1);
	mask.set(1,0,-1);
	mask.set(2,1,0);
	mask.set(3,-1,0);
	
	Arguments args(grid_rows,grid_cols);

	double grid_height = chip_height / grid_rows;
	double grid_width = chip_width / grid_cols;
	double max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);

	args.amb_temp = amb_temp;
	args.Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	args.Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	args.Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	args.Rz = t_chip / (K_SI * grid_height * grid_width);
	args.step = PRECISION / max_slope;
	
	
	//temp = (double *) calloc (grid_rows * grid_cols, sizeof(double));
	//power = (double *) calloc (grid_rows * grid_cols, sizeof(double));
	//result = (double *) calloc (grid_rows * grid_cols, sizeof(double));
	//if(!temp || !power)
	//	fatal("unable to allocate memory");

	/* read initial temperatures and input power	*/
	tfile = argv[5];
	pfile = argv[6];
	ofile = argv[7];
	read_input(temp, grid_rows, grid_cols, tfile);
	read_input(args.power, grid_rows, grid_cols, pfile);

	Stencil2D<Array2D<double>, Mask2D<double>, Arguments> stencil(temp,result,mask,args);

	printf("Start computing the transient temperature\n");

	long long start_time = get_time();
	
	stencil.runIterativeCPU(sim_time, num_omp_threads);
	//compute_tran_temp(result,sim_time, temp, power, grid_rows, grid_cols);
	
	long long end_time = get_time();

    printf("Ending simulation\n");
    printf("Total time: %.3f seconds\n", ((float) (end_time - start_time)) / (1000*1000));

    writeoutput((1&sim_time) ? result : temp, grid_rows, grid_cols, ofile);
	
	/* output results	*/
#ifdef VERBOSE
	fprintf(stdout, "Final Temperatures:\n");
#endif

#ifdef OUTPUT
	for(i=0; i < grid_rows; i++)
		for(j=0; j < grid_cols; j++)
			//fprintf(stdout, "%d,%d\t%g\n", i, j, result(i,j));
#endif
	/* cleanup	*/
	//free(temp);
	//free(power);

	return 0;
}
