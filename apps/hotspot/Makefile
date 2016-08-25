# C compiler
CC = g++
CC_FLAGS = -g -fopenmp -O3

src:
	python ../pskelccv2.py hotspot.cpp hotspot_pskel.cpp

all:
	$(CC) $(CC_FLAGS) hotspot.cpp -o hotspot 
	
run:
	./hotspot 512 512 100 8 temp_512 power_512

clean:
	rm -f hotspot
