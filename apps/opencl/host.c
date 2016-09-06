/*--INCLUDE--*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <CL/opencl.h>
#include <CL/cl_kalray.h>
/*--INCLUDE--*/

/*--DEVICE_CODE--*/
const char *kernelSource =                                 "\n" \
"__kernel void vecAdd( __constant double *a,                \n" \
"                __constant double *b,                      \n" \
"                __global double *c,                        \n" \
"                const unsigned int n)                      \n" \
"{                                                          \n" \
"     unsigned int id_gx = get_global_id(0);                   \n" \
"     unsigned int id_gy = get_global_id(1);	            \n"\
"     unsigned int tx = get_local_id(0);		    \n"\
"     unsigned int ty = get_local_id(1);		    \n"\
"     unsigned int bx = get_group_id(0);		    \n"\
"     unsigned int by = get_group_id(1);		    \n"\
"     unsigned int index = get_global_size(0)*(bx+tx)+(by+ty);\n"\
"     if (index > n)	                 		    \n" \
"         return;                                           \n" \
"                                                           \n" \
"     c[index] = a[index] + b[index];                       \n" \
"}                                                          \n" \
                                                           "\n" ;
/*--DEVICE_CODE--*/

/*--PROLOGUE--*/
int main( int argc, char* argv[] )
{
	unsigned int vec_num;

	size_t bytes;

	double *vec_in_a;
	double *vec_in_b;
	double *vec_out_c;

	cl_mem buf_in_a;
	cl_mem buf_in_b;
	cl_mem buf_out_c;

	cl_platform_id mppa_platform;
	cl_device_id device_id;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_int err;
	cl_uint page_size;
	
	size_t global_size[2], local_size[2];
/*--PROLOGUE--*/

/*--PLATFORM--*/
	err = clGetPlatformIDs(1, &mppa_platform, NULL);
	assert(!err);

	err = clGetDeviceIDs(mppa_platform, CL_DEVICE_TYPE_ACCELERATOR,
					1, &device_id, NULL);
	assert(!err);

	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	assert(context);
/*--PLATFORM--*/

/*--PAGE_SIZE--*/
	err = clGetMPPAInfo(device_id, CL_MPPA_PAGE_SIZE_KALRAY, sizeof(cl_uint), &page_size, NULL);
	assert(!err);
/*--PAGE_SIZE--*/

/*--VEC_INIT--*/
	vec_num = 16 * page_size / sizeof(double);
	bytes = vec_num * sizeof(double);

	vec_in_a = (double*)malloc(bytes);
	vec_in_b = (double*)malloc(bytes);
	vec_out_c = (double*)malloc(bytes);
	unsigned int i;

	for( i = 0; i < vec_num; i++ )
	{
		vec_in_a[i] = i * 1.1; 
		vec_in_b[i] = -0.1 * i;
	}
	memset(vec_out_c, 0, bytes);
/*--VEC_INIT--*/

/*--QUEUE--*/
	queue = clCreateCommandQueue(context, device_id,
			CL_QUEUE_PROFILING_ENABLE, &err);
	assert(queue);
/*--QUEUE--*/

/*--PROGRAM--*/
	program = clCreateProgramWithSource(context, 1,
				(const char **) & kernelSource, NULL, &err);
	assert(program);

	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
/*--PROGRAM--*/

/*--KERNEL--*/
	kernel = clCreateKernel(program, "vecAdd", &err);
	assert(kernel);
/*--KERNEL--*/

/*--BUFFER--*/
	buf_in_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	assert(buf_in_a);
	buf_in_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	assert(buf_in_b);
	buf_out_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
	assert(buf_out_c);

	err = clEnqueueWriteBuffer(queue, buf_in_a, CL_TRUE, 0,
					bytes, vec_in_a, 0, NULL, NULL);
	assert(!err);
	err = clEnqueueWriteBuffer(queue, buf_in_b, CL_TRUE, 0,
					bytes, vec_in_b, 0, NULL, NULL);
	assert(!err);
/*--BUFFER--*/

/*--ARGS--*/
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in_a);
	assert(!err);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_in_b);
	assert(!err);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_out_c);
	assert(!err);
	err = clSetKernelArg(kernel, 3, sizeof(unsigned int), &vec_num);
	assert(!err);
/*--ARGS--*/

/*--NDRANGE--*/
	local_size[0] = 4;
	local_size[1] = 4;
	global_size[0] = 128;
	global_size[1] = 128;
	printf("vec_num %d\n",vec_num);
	err = clEnqueueNDRangeKernel(queue, kernel, 2,
			NULL, global_size, local_size,
			0, NULL, NULL);
	assert(!err);
/*--NDRANGE--*/

/*--FINISH--*/
	clFinish(queue);

	clEnqueueReadBuffer(queue, buf_out_c, CL_TRUE, 0,
			bytes, vec_out_c, 0, NULL, NULL);

	for(i = 0; i < vec_num; i++) {
		double expected = vec_in_a[i] + vec_in_b[i];
		if (fabs(vec_out_c[i] - expected) > DBL_EPSILON) {
			printf("Check failed at offset %d, %lf instead of %lf\n", i, vec_out_c[i], vec_in_a[i] + vec_in_b[i]);
			//exit(1);
		}
		else{
			printf("%lf\t",vec_out_c[i]);
		}
	}
/*--FINISH--*/

/*--CLEANUP--*/
	clReleaseMemObject(buf_in_a);
	clReleaseMemObject(buf_in_b);
	clReleaseMemObject(buf_out_c);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	free(vec_in_a);
	free(vec_in_b);
	free(vec_out_c);

	return 0;
}
/*--CLEANUP--*/
