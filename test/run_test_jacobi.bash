#!/bin/bash
CPU_EVENTS="" #"perf stat -e cpu-cycles -e instructions -e branches -e branch-misses -e cache-references -e cache-misses -e LLC-loads -e LLC-load-misses -e cpu-clock"
GPU_EVENTS="nvprof --metrics l1_cache_global_hit_rate,sm_efficiency,achieved_occupancy,ipc,inst_per_warp,inst_executed,warp_execution_efficiency,gld_efficiency,gst_efficiency"
OUTPUT_FLAG=0
OPTIMUS=""
OUTPUT_DIR=""
EXEC="jacobi"
TEST_DIR="./quadro"
BIN_ACC_KERNEL="../bin/jacobi_acc_kernels"
BIN_ACC_PARALLEL="../bin/jacobi_acc_parallel"
ITERATIONS=50

######TESTES DE TEMPO###########
make acc_kernels -C ../apps/${EXEC}
make acc_parallel -C ../apps/${EXEC}
OUTPUT_DIR="${TEST_DIR}/${EXEC}"
mkdir ${OUTPUT_DIR}

#teste GPU a 100%
VERBOSE=0
for INPUT_SIZE in 512 1024 2048 4096 8192
do
	for ITERATION in {1..3..1}
	do
		echo $"Running with INPUT_SIZE = ${INPUT_SIZE}"
		echo "ITERATION #${ITERATION}"
        
		${BIN_ACC_KERNELS} ${INPUT_SIZE} ${INPUT_SIZE} ${ITERATIONS} ${VERBOSE} 2>> ${OUTPUT_DIR}/${EXEC}_kernels_${INPUT_SIZE}_${ITERATIONS}.txt
		sleep 1
        
        ${BIN_ACC_PARALLEL} ${INPUT_SIZE} ${INPUT_SIZE} ${ITERATIONS} ${VERBOSE} 2>> ${OUTPUT_DIR}/${EXEC}_parallel_${INPUT_SIZE}_${ITERATIONS}.txt
		sleep 1
	done
done

#teste com uma variacao de gpu de 0.1 a 0.9 e max threads
#CPU_THREADS=8
#for BLOCK_SIZE in 8 16 32
#do
	#for GPU_PERCENT in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
	#do
		#for ITERATION in {1..10..1}
		#do
			#echo $"Running with GPU_PERCENT=${GPU_PERCENT} BLOCK_SIZE=${BLOCK_SIZE} CPU_THREADS=${CPU_THREADS} ITERATION #${ITERATION}"
			
			#echo "ITERATION #${ITERATION}" >> ${OUTPUT_DIR}/${EXEC}_${IMG1_SIZE}_${ITERATIONS}_${GPU_PERCENT}_${BLOCK_SIZE}_${CPU_THREADS}.txt
			#time -p ${OPTIMUS} ${CPU_EVENTS} ${GPU_EVENTS} ${BIN} ${IMG1_DIR} ${ITERATIONS} $GPU_PERCENT $BLOCK_SIZE $CPU_THREADS ${OUTPUT_FLAG} 2>> ${OUTPUT_DIR}/${EXEC}_${IMG1_SIZE}_${ITERATIONS}_${GPU_PERCENT}_${BLOCK_SIZE}_${CPU_THREADS}.txt
			#sleep 1
			
			#echo "ITERATION #${ITERATION}" >> ${OUTPUT_DIR}/${EXEC}_${IMG2_SIZE}_${ITERATIONS}_${GPU_PERCENT}_${BLOCK_SIZE}_${CPU_THREADS}.txt
			#time -p ${OPTIMUS} ${CPU_EVENTS} ${GPU_EVENTS} ${BIN} ${IMG2_DIR} ${ITERATIONS} $GPU_PERCENT $BLOCK_SIZE $CPU_THREADS ${OUTPUT_FLAG} 2>> ${OUTPUT_DIR}/${EXEC}_${IMG2_SIZE}_${ITERATIONS}_${GPU_PERCENT}_${BLOCK_SIZE}_${CPU_THREADS}.txt
			#sleep 1

			#echo "ITERATION #${ITERATION}" >> ${OUTPUT_DIR}/${EXEC}_${IMG3_SIZE}_${ITERATIONS}_${GPU_PERCENT}_${BLOCK_SIZE}_${CPU_THREADS}.txt
			#time -p ${OPTIMUS} ${CPU_EVENTS} ${GPU_EVENTS} ${BIN} ${IMG3_DIR} ${ITERATIONS} $GPU_PERCENT $BLOCK_SIZE $CPU_THREADS ${OUTPUT_FLAG} 2>> ${OUTPUT_DIR}/${EXEC}_${IMG3_SIZE}_${ITERATIONS}_${GPU_PERCENT}_${BLOCK_SIZE}_${CPU_THREADS}.txt
			#sleep 1
		#done
	#done
#done

#teste CPU sequencial e max threads
#GPU_EVENTS=""
#GPU_PERCENT=0
#BLOCK_SIZE=0
#for CPU_THREADS in 1 8
#do
#	for ITERATION in {1..10..1}
#	do
#		echo $"Running with GPU_PERCENT=${GPU_PERCENT} BLOCK_SIZE=${BLOCK_SIZE} CPU_THREADS=${CPU_THREADS} ITERATION #${ITERATION}"
#		
#		echo "ITERATION #${ITERATION}" >> ${OUTPUT_DIR}/${EXEC}_${IMG1_SIZE}_${ITERATIONS}_${GPU_PERCENT}_${BLOCK_SIZE}_${CPU_THREADS}.txt
#		time -p ${OPTIMUS} ${CPU_EVENTS} ${GPU_EVENTS} ${BIN} ${IMG1_DIR} ${ITERATIONS} $GPU_PERCENT $BLOCK_SIZE $CPU_THREADS ${OUTPUT_FLAG} 2>> ${OUTPUT_DIR}/${EXEC}_${IMG1_SIZE}_${ITERATIONS}_${GPU_PERCENT}_${BLOCK_SIZE}_${CPU_THREADS}.txt
#		sleep 1
#		
#		echo "ITERATION #${ITERATION}" >> ${OUTPUT_DIR}/${EXEC}_${IMG2_SIZE}_${ITERATIONS}_${GPU_PERCENT}_${BLOCK_SIZE}_${CPU_THREADS}.txt
#		time -p ${OPTIMUS} ${CPU_EVENTS} ${GPU_EVENTS} ${BIN} ${IMG2_DIR} ${ITERATIONS} $GPU_PERCENT $BLOCK_SIZE $CPU_THREADS ${OUTPUT_FLAG} 2>> ${OUTPUT_DIR}/${EXEC}_${IMG2_SIZE}_${ITERATIONS}_${GPU_PERCENT}_${BLOCK_SIZE}_${CPU_THREADS}.txt
#		sleep 1
#
#		echo "ITERATION #${ITERATION}" >> ${OUTPUT_DIR}/${EXEC}_${IMG3_SIZE}_${ITERATIONS}_${GPU_PERCENT}_${BLOCK_SIZE}_${CPU_THREADS}.txt
#		time -p ${OPTIMUS} ${CPU_EVENTS} ${GPU_EVENTS} ${BIN} ${IMG3_DIR} ${ITERATIONS} $GPU_PERCENT $BLOCK_SIZE $CPU_THREADS ${OUTPUT_FLAG} 2>> ${OUTPUT_DIR}/${EXEC}_${IMG3_SIZE}_${ITERATIONS}_${GPU_PERCENT}_${BLOCK_SIZE}_${CPU_THREADS}.txt
#		sleep 1
#	done
#done
