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

#define TIME

/*--DEVICE_CODE--*/
const char *kernelSource =                                 "\n" \
"__kernel void vecAdd( __constant double *a,                \n" \
"                __constant double *b,                      \n" \
"                __global double *c,                        \n" \
"                const unsigned int n)                      \n" \
"{                                                          \n" \
"     const unsigned int width = get_global_size(0);              \n"\
"     const unsigned int height = get_global_size(1);             \n"\
"     const unsigned int xOut = get_global_id(0);                 \n"\
"     const unsigned int yOut = get_global_id(1);	            \n"\
"     const unsigned int index = yOut*width+xOut;                 \n"\
"     double sum = 0.0;                                      \n"\
"     for (int r = 0; r < 3; r++){                          \n"\
"        const int idxFtmp = r * 3;                         \n"\
"        const int yIn = yOut + r;                          \n"\
"        const int idxIntmp = yIn * width + xOut;           \n"\
"        for (int c = 0; c < 3; c++){                       \n"\
"            const int idxF  = idxFtmp  + c;                \n"\
"            const int idxIn = idxIntmp + c;                \n"\
"            if(idxIn<n) sum += b[idxF]*a[idxIn];           \n"\
"        }                                                  \n"\
"     }                                                     \n"\
"     c[index] = sum;                                       \n"\
"}                                                          \n";
 
/*--DEVICE_CODE--*/

/*--PROLOGUE--*/
int main( int argc, char* argv[] )
{
	unsigned int vec_num, vec_size;

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
	cl_uint page_size, page_count_cu, page_count_pe;

	#ifdef TIME
	cl_event event;
	cl_ulong time_start=0, time_end=0;
	double total_time=0;
	#endif
	
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
	err = clGetMPPAInfo(device_id, CL_MPPA_PAGE_PER_CU_KALRAY, sizeof(cl_uint), &page_count_cu, NULL);
	assert(!err);
	
	err = clGetMPPAInfo(device_id, CL_MPPA_PAGE_PER_PE_KALRAY, sizeof(cl_uint), &page_count_pe, NULL);
	assert(!err);

/*--VEC_INIT--*/
	vec_size = 2* page_size / sizeof(double);
	vec_num = vec_size * vec_size;
	bytes = vec_num* sizeof(double);
	printf("pagesize: %d page per computer unit: %d page per pe: %d vec_size: %d vec_num: %d\n",page_size,page_count_cu, page_count_pe, vec_size,vec_num);

	vec_in_a = (double*)malloc(bytes);
	vec_in_b = (double*)malloc(9*sizeof(double));
	vec_out_c = (double*)malloc(bytes);
	unsigned int i,j,r,c;

	for( i = 0; i < vec_num; i++ )
	{
		vec_in_a[i] = 100; 
		//vec_in_b[i] = i;
	}

	for(i=0;i<9;i++){
		vec_in_b[i] = 0.25;
	}
	memset(vec_out_c,0.0, bytes);
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
	buf_in_b = clCreateBuffer(context, CL_MEM_READ_ONLY, 9*sizeof(double), NULL, NULL);
	assert(buf_in_b);
	buf_out_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
	assert(buf_out_c);

	err = clEnqueueWriteBuffer(queue, buf_in_a, CL_TRUE, 0,
					bytes, vec_in_a, 0, NULL, NULL);
	assert(!err);
	err = clEnqueueWriteBuffer(queue, buf_in_b, CL_TRUE, 0,
					9*sizeof(double), vec_in_b, 0, NULL, NULL);
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
	local_size[0] = 16;
	local_size[1] = 1;
	global_size[0] = vec_size;
	global_size[1] = vec_size;
	printf("vec_size %d\n",vec_size);
	#ifdef TIME 
	err = clEnqueueNDRangeKernel(queue, kernel, 2,NULL, global_size, local_size,0, NULL, &event);
	#else
	err = clEnqueueNDRangeKernel(queue, kernel, 2,NULL, global_size, local_size,0, NULL, NULL);
	#endif	
	assert(!err);
/*--NDRANGE--*/

/*--FINISH--*/
	#ifdef TIME
	err = clWaitForEvents(1,&event);
	assert(!err);
	#endif
	clFinish(queue);
	
	err = clEnqueueReadBuffer(queue, buf_out_c, CL_TRUE, 0,bytes, vec_out_c, 0, NULL, NULL);
	assert(!err);

	#ifdef TIME
	err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	assert(!err);
	err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	assert(!err);
	total_time = time_end-time_start;
	printf("Exec_time\t%.3lf\n",total_time*1.0e-9);
	#endif

	for(j=0;j<vec_size;j++){
		for(i=0;i<vec_size;i++){
			double sum = 0.0;
		    	for (r = 0; r < 3; r++){
				const int idxFtmp = r * 3;
        			const int yIn = j + r;
			        const int idxIntmp = yIn * vec_size + i;
				for (c = 0; c < 3; c++){
			            const int idxF  = idxFtmp  + c;
			            const int idxIn = idxIntmp + c;
            			    if(idxIn<vec_num) sum += vec_in_b[idxF]*vec_in_a[idxIn];

        			}
			}			
			if (fabs(vec_out_c[j*vec_size+i] - sum) > DBL_EPSILON) {
                        	printf("Check failed at offset [%d][%d], %lf instead of %lf\n",j,i, vec_out_c[j*vec_size+i], sum);
			}
			//else{printf("[%d][%d]=%lf\n",j,i, vec_out_c[j*vec_size+i]);}
    		}		
	}


	/*for(i = 0; i < vec_num; i++) {
		float expected = vec_in_a[i] + vec_in_b[i];
		if (fabs(vec_out_c[i] - expected) > DBL_EPSILON) {
			printf("Check failed at offset %d, %lf instead of %lf\n", i, vec_out_c[i], vec_in_a[i] + vec_in_b[i]);
			//exit(1);
		}
		else{
			printf("%lf\t",vec_out_c[i]);
		}
	}*/
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
