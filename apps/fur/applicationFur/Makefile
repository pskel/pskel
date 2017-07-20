CC = g++
CUDAC = nvcc
OPTIMUS = optirun #remover este valor se não estiver em um sistema OPTIMUS (notebook)
CUDA_LIB_PATH = /usr/local/cuda-5.5/lib64
CUDAFLAGS = --compiler-bindir /usr/bin/gcc-4.8 -O3 -Xcompiler -ftree-vectorize -Xcompiler -msse4 #-I/usr/local/cuda-5.5/extras/CUPTI/include -ccbin g++  -Xcompiler -Wall
CXXFLAGS = -O3 -w
GENCONDE = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\"
OMP_TBB = -fopenmp -lgomp -ltbb
GCCLIB = -lgomp -fopenmp -lpapi
LIBS = -lgomp -ltbb -lcuda -lpapi -I../../lib/ -L../../lib/ga -lga -lm #-papi -L /usr/local/cuda-5.5/extras/CUPTI/lib -L /usr/local/cuda-5.5/extras/CUPTI/lib64 -lcupti
SRC_NAME = applicationFur
BIN_DIR = ./

all: clean pskel

pskel:  ${SRC_NAME}.cu
	${CUDAC} ${CUDAFLAGS} -o ${BIN_DIR}${SRC_NAME} ${SRC_NAME}.cu ${LIBS} -DPSKEL_CUDA

pskelcc: ${SRC_NAME}.cpp
	${CC} ${CXXFLAGS} -c ${BIN_DIR}${SRC_NAME}.cpp
	g++ -o fur ${SRC_NAME}.o ${GCCLIB}


clean:
	 rm -f *.o ${BIN_DIR}${SRC_NAME}
