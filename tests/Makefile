CC = g++
CXXFLAGS = -O3 -w
LIBS = -fopenmp -lgomp -lm -lgtest

CUDAC = nvcc
CUDAFLAGS = -O3 -Xcompiler -ftree-vectorize -Xcompiler -msse4
CUDALIBS = -Xcompiler -fopenmp -lgomp -ltbb -lpapi -lcuda -lm -lgtest -lga
GENCODE = 
#GENCODE = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\"

all: clean build test clean
	 rm -f *.o arrays masks maps stencils stenciltiles

build: arrays.cpp masks.cpp maps.cpp stenciltiles.cpp stencils.cpp
	${CC} ${CXXFLAGS} -o arrays arrays.cpp ${LIBS}
	${CC} ${CXXFLAGS} -o masks masks.cpp ${LIBS}
	${CC} ${CXXFLAGS} -o maps maps.cpp ${LIBS}
	${CC} ${CXXFLAGS} -o stenciltiles stenciltiles.cpp ${LIBS}
	${CC} ${CXXFLAGS} -o stencils stencils.cpp ${LIBS}

gpu: arrays.cpp masks.cpp maps.cpp stenciltiles.cpp stencils.cpp
	cp arrays.cpp arrays.cu
	cp masks.cpp masks.cu
	cp maps.cpp maps.cu
	cp stenciltiles.cpp stenciltiles.cu
	cp stencils.cpp stencils.cu
	${CUDAC} ${CUDAFLAGS} -o arrays arrays.cu ${CUDALIBS} -DPSKEL_CUDA ${GENCODE}
	${CUDAC} ${CUDAFLAGS} -o masks masks.cu ${CUDALIBS} -DPSKEL_CUDA ${GENCODE}
	${CUDAC} ${CUDAFLAGS} -o maps maps.cu ${CUDALIBS} -DPSKEL_CUDA ${GENCODE}
	${CUDAC} ${CUDAFLAGS} -o stenciltiles stenciltiles.cu ${CUDALIBS} -DPSKEL_CUDA ${GENCODE}
	${CUDAC} ${CUDAFLAGS} -o stencils stencils.cu ${CUDALIBS} -DPSKEL_CUDA ${GENCODE}
	rm arrays.cu masks.cu maps.cu stenciltiles.cu stencils.cu

test: arrays maps
	./arrays
	./masks
	./maps
	./stenciltiles
	./stencils
	
clean:
	 rm -f *.o arrays masks maps stencils stenciltiles
