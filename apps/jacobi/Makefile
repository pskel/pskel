CC 		= g++
CUDAC 		= nvcc
ACC		= /opt/pgi/linux86-64/2016/bin/pgc++
PSKEL_SRC	= python ../pskelccv2.py 
OPTIMUS         = optirun #remover este valor se não estiver em um sistema OPTIMUS (notebook)
CUDA_LIB_PATH   = /usr/local/cuda-5.5/lib64
CUDAFLAGS    	= -arch=sm_21 -O3 -m64 -lineinfo -Xcompiler -ftree-vectorize -Xcompiler -march=native -Xcompiler -mtune=native -Xcompiler -O3 
ACCFLAGS	= -O3 -ta=tesla,managed -tp=sandybridge -Minfo=accel
CXXFLAGS	= -O3 -w
GENCODE		= -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\"
OMP_TBB         = -fopenmp -lgomp -ltbb
GA_LIB_PATH 	= /home/alyson/galib247/ga
GA_SRC_PATH 	= /home/alyson/galib247/
PAPI_LIB_PATH 	= /home/alyson/papi-5.4.1/build/lib
PAPI_SRC_PATH 	= /home/alyson/papi-5.4.1/build/include
HR_TIME		= ../../pskel/include
LIBS		= -Xcompiler -fopenmp -lgomp -I${HR_TIME} -I${GA_SRC_PATH} -L${GA_LIB_PATH} -lga -lm  
ACC_LIBS	= -I${HR_TIME} -lrt -mp 
SRC_NAME        = jacobi
BIN_DIR         = ../../bin/

all: acc_parallel acc_kernels 

serial: ${SRC_NAME}_serial.cpp
	${CUDAC} ${CUDAFLAGS} -o ${BIN_DIR}${SRC_NAME}_serial ${SRC_NAME}_serial.cpp -Xcompiler ${OMP_TBB}

acc_parallel: ${SRC_NAME}_acc_parallel.cpp
	${ACC} ${ACCFLAGS} -o  ${BIN_DIR}${SRC_NAME}_acc_parallel ${SRC_NAME}_acc_parallel.cpp ${ACC_LIBS}
	
acc_kernels: ${SRC_NAME}_acc_kernels.cpp
	${ACC} ${ACCFLAGS} -o  ${BIN_DIR}${SRC_NAME}_acc_kernels ${SRC_NAME}_acc_kernels.cpp ${ACC_LIBS}

pskel: ${SRC_NAME}_pskel.cu
	${CUDAC} ${CUDAFLAGS} -o ${BIN_DIR}${SRC_NAME}_pskel ${SRC_NAME}_pskel.cu ${LIBS}
	
src: ${SRC_NAME}_serial.cpp
	${PSKEL_SRC} ${SRC_NAME}_serial.cpp ${SRC_NAME}_src2src.cu
	${CUDAC} ${CUDAFLAGS} -ccbin=/usr/bin/g++ -o ${BIN_DIR}${SRC_NAME}_src2src ${SRC_NAME}_src2src.cu ${LIBS}
	
papi: ${SRC_NAME}.cu
	${CUDAC} ${CUDAFLAGS} -o ${BIN_DIR}${SRC_NAME} ${SRC_NAME}.cu ${LIBS} -DPSKEL_PAPI -DTIME
	
papi_debug:${SRC_NAME}.cu
	${CUDAC} ${CUDAFLAGS} -o ${BIN_DIR}${SRC_NAME} ${SRC_NAME}.cu ${LIBS} -DPSKEL_PAPI -DPSKEL_PAPI_DEBUG

papi_quadro: ${SRC_NAME}.cu
	${CUDAC} ${CUDAFLAGS} -o ${BIN_DIR}${SRC_NAME} ${SRC_NAME}.cu ${LIBS} -DPSKEL_PAPI -DQUADRO
	
papi_debug_quadro: ${SRC_NAME}.cu
	${CUDAC} ${CUDAFLAGS} -o ${BIN_DIR}${SRC_NAME} ${SRC_NAME}.cu ${LIBS} -DPSKEL_PAPI -DPSKEL_PAPI_DEBUG -DQUADRO	

cupti:  ${SRC_NAME}.cu
	${CUDAC} ${CUDAFLAGS} -o ${BIN_DIR}${SRC_NAME} ${SRC_NAME}.cu ${LIBS} -DCUPTI

time:  ${SRC_NAME}.cu
	${CUDAC} ${CUDAFLAGS} -o ${BIN_DIR}${SRC_NAME} ${SRC_NAME}.cu ${LIBS} -DTIME
	
pskelold: ${SRC_NAME}.cu
	${CUDAC} ${CUDAFLAGS} ${GENCODE} -o ${SRC_NAME}.o -c ${SRC_NAME}.cu
	${CUDAC} ${CUDAFLAGS} -o ${BIN_DIR}${SRC_NAME} ${SRC_NAME}.o ${LIBS} -Ilibcutil

testseq:
	time -p ${BIN_DIR}${SEQ_SRC_NAME} 480 480 1
	
testpskel:
	time -p ${BIN_DIR}${SRC_NAME} 480 480 1 0.0 32 8 0

testoptimus:
	time -p ${OPTIMUS} ${BIN_DIR}${SRC_NAME} 480 480 1 1.0 32 8 0
	
clean:
	 rm -f *.o ${BIN_DIR}${SRC_NAME}_pskel ${BIN_DIR}${SEQ_SRC_NAME}
