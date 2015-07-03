CC 				= g++
CUDAC 			= nvcc
OPTIMUS         = optirun #remover este valor se não estiver em um sistema OPTIMUS (notebook)
CUDA_LIB_PATH   = /usr/local/cuda-5.5/lib64
CUDAFLAGS = --compiler-bindir /usr/bin/gcc-4.4.7 -O3 -Xcompiler -ftree-vectorize -Xcompiler -msse4 #-I/usr/local/cuda-5.5/extras/CUPTI/include -ccbin g++  -Xcompiler -Wall
CXXFLAGS		= -O3 -w
GENCONDE		= -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\"
OMP_TBB         = -fopenmp -lgomp -ltbb
LIBS			= -Xcompiler -fopenmp -lgomp -ltbb -lpapi -lcuda -I../../lib/ -L../../lib/ga -lga -lm -lgtest #-L /usr/local/cuda-5.5/extras/CUPTI/lib -L /usr/local/cuda-5.5/extras/CUPTI/lib64 -lcupti
SRC_NAME        = gol
SEQ_SRC_NAME    = gol_seq
BIN_DIR         = ../../bin/

all: clean build test clean
	 rm -f *.o arrays masks maps stencils stenciltiles

build: arrays.cu masks.cu maps.cu
	${CUDAC} ${CUDAFLAGS} -o arrays arrays.cu ${LIBS}
	${CUDAC} ${CUDAFLAGS} -o masks masks.cu ${LIBS}
	${CUDAC} ${CUDAFLAGS} -o maps maps.cu ${LIBS}
	${CUDAC} ${CUDAFLAGS} -o stenciltiles stenciltiles.cu ${LIBS}
	${CUDAC} ${CUDAFLAGS} -o stencils stencils.cu ${LIBS}

test: arrays maps
	./arrays
	./masks
	./maps
	./stenciltiles
	./stencils
	
clean:
	 rm -f *.o arrays masks maps stencils stenciltiles
