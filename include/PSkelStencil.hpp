//-------------------------------------------------------------------------------
// Copyright (c) 2015, Alyson D. Pereira <alyson.deives@outlook.com>,
//					   Rodrigo C. O. Rocha <rcor.cs@gmail.com>
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//-------------------------------------------------------------------------------

#ifndef PSKEL_STENCIL_HPP
#define PSKEL_STENCIL_HPP
//#define BARRIER_SYNC_MASTER "/mppa/sync/128:1"
//#define BARRIER_SYNC_SLAVE "/mppa/sync/[0..15]:2"
#include <cmath>
#include <algorithm>
#include <iostream>

#include <iostream>
#include <unistd.h>

using namespace std;
#ifdef PSKEL_CUDA
  #include <ga/ga.h>
  #include <ga/std_stream.h>
#endif

#define ARGC_SLAVE 9

namespace PSkel{

#ifdef PSKEL_CUDA
//********************************************************************************************
// Kernels CUDA. Chama o kernel implementado pelo usuario
//********************************************************************************************

template<typename T1, typename T2, class Args>
__global__ void stencilTilingCU(Array<T1> input,Array<T1> output,Mask<T2> mask,Args args, size_t widthOffset, size_t heightOffset, size_t depthOffset, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth);

template<typename T1, typename T2, class Args>
__global__ void stencilTilingCU(Array2D<T1> input,Array2D<T1> output,Mask2D<T2> mask,Args args, size_t widthOffset, size_t heightOffset, size_t depthOffset, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth);

template<typename T1, typename T2, class Args>
__global__ void stencilTilingCU(Array3D<T1> input,Array3D<T1> output,Mask3D<T2> mask,Args args, size_t widthOffset, size_t heightOffset, size_t depthOffset, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth);


//********************************************************************************************
// Kernels CUDA. Chama o kernel implementado pelo usuario
//********************************************************************************************

template<typename T1, typename T2, class Args>
__global__ void stencilTilingCU(Array<T1> input,Array<T1> output,Mask<T2> mask,Args args, size_t widthOffset, size_t heightOffset, size_t depthOffset, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth){
	size_t i = blockIdx.x*blockDim.x+threadIdx.x;
	#ifdef PSKEL_SHARED_MASK
	extern __shared__ int shared[];
 	if(threadIdx.x<(mask.size*mask.dimension))
		shared[threadIdx.x] = mask.deviceMask[threadIdx.x];
	__syncthreads();
	mask.deviceMask = shared;
	#endif
	if(i>=widthOffset && i<(widthOffset+tilingWidth)){
		stencilKernel(input, output, mask, args, i);
	}
}

template<typename T1, typename T2, class Args>
__global__ void stencilTilingCU(Array2D<T1> input,Array2D<T1> output,Mask2D<T2> mask,Args args, size_t widthOffset, size_t heightOffset, size_t depthOffset, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth){
	size_t w = blockIdx.x*blockDim.x+threadIdx.x;
	size_t h = blockIdx.y*blockDim.y+threadIdx.y;
	#ifdef PSKEL_SHARED_MASK
	extern __shared__ int shared[];
  	if(threadIdx.x<(mask.size*mask.dimension))
		shared[threadIdx.x] = mask.deviceMask[threadIdx.x];
	__syncthreads();
	mask.deviceMask = shared;
	#endif
	if(w>=widthOffset && w<(widthOffset+tilingWidth) && h>=heightOffset && h<(heightOffset+tilingHeight) ){
		stencilKernel(input, output, mask, args, h, w);
	}
}

template<typename T1, typename T2, class Args>
__global__ void stencilTilingCU(Array3D<T1> input,Array3D<T1> output,Mask3D<T2> mask,Args args, size_t widthOffset, size_t heightOffset, size_t depthOffset, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth){
	size_t w = blockIdx.x*blockDim.x+threadIdx.x;
	size_t h = blockIdx.y*blockDim.y+threadIdx.y;
	size_t d = blockIdx.z*blockDim.z+threadIdx.z;
	#ifdef PSKEL_SHARED_MASK
	extern __shared__ int shared[];
  	if(threadIdx.x<(mask.size*mask.dimension))
		shared[threadIdx.x] = mask.deviceMask[threadIdx.x];
	__syncthreads();
	mask.deviceMask = shared;
	#endif
  
	if(w>=widthOffset && w<(widthOffset+tilingWidth) && h>=heightOffset && h<(heightOffset+tilingHeight) && d>=depthOffset && d<(depthOffset+tilingDepth) ){
		stencilKernel(input, output, mask, args, h, w, d);
	}
}
#endif

//*******************************************************************************************
// Stencil Base
//*******************************************************************************************
#ifndef MPPA_MASTER
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runSequential(){
	this->runSeq(this->input, this->output);
}
#endif

//*******************************************************************************************
// MPPA
//*******************************************************************************************

#ifdef PSKEL_MPPA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::spawn_slaves(const char slave_bin_name[], size_t tilingHeight, int nb_clusters, int nb_threads, int iterations, int innerIterations){
	// Prepare arguments to send to slaves
	int i;
	int cluster_id;
	int pid;
	size_t hTiling = ceil(float(this->input.getHeight())/float(tilingHeight));
	int tiles = hTiling/nb_clusters;
	int itMod = hTiling % nb_clusters;
	int tilesSlave;
	int r;
	int outterIterations = ceil(float(iterations)/innerIterations);


	#ifdef DEBUG
	cout<<"MASTER: width="<<this->input.getWidth()<<" height="<<this->input.getWidth();
	cout<<" tilingHeight="<<tilingHeight <<" iterations="<<iterations;
	cout<<" inner_iterations="<<innerIterations<<" nbclusters="<<nb_clusters<<" nbthreads="<<nb_threads<<endl;
	cout<<"MASTER: tiles="<<tiles<<" itMod="<<itMod<<" outterIterations="<<outterIterations<<endl;
	#endif

	char **argv_slave = (char**) malloc(sizeof (char*) * ARGC_SLAVE);
	for (i = 0; i < ARGC_SLAVE - 1; i++)
	  argv_slave[i] = (char*) malloc (sizeof (char) * 10);
	 
	sprintf(argv_slave[1], "%d", this->input.getWidth());
	sprintf(argv_slave[2], "%d", tilingHeight);
	sprintf(argv_slave[4], "%d", nb_threads);
	sprintf(argv_slave[5], "%d", iterations);
	sprintf(argv_slave[6], "%d", outterIterations);
	sprintf(argv_slave[7], "%d", itMod);

	argv_slave[8] = NULL;

	  
	// Spawn slave processes
	for (cluster_id = 0; cluster_id < nb_clusters && cluster_id < hTiling; cluster_id++) {
		r = (cluster_id < itMod)?1:0;
		tilesSlave = tiles + r;

		sprintf(argv_slave[0], "%d", tilesSlave);
		sprintf(argv_slave[3], "%d", cluster_id);
	    pid = mppa_spawn(cluster_id, NULL, slave_bin_name, (const char **)argv_slave, NULL);
	    assert(pid >= 0);
	}
	for (i = 0; i < ARGC_SLAVE; i++)
		free(argv_slave[i]);
	free(argv_slave);
}
#endif


#ifdef PSKEL_MPPA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::mppaSlice(size_t tilingHeight, int nb_clusters, int iterations, int innerIterations) {
	size_t hTiling = ceil(float(this->input.getHeight())/float(tilingHeight));
	int tiles = hTiling/nb_clusters;
	int itMod = hTiling % nb_clusters;

	this->output.portalReadAlloc(nb_clusters, 0);
	size_t outterIterations;
	size_t heightOffset;
	
	StencilTiling<Array, Mask>tiling(this->input, this->output, this->mask);

	barrier_t *barrierNbClusters;
	outterIterations = ceil(float(iterations)/innerIterations);
	Array inputCopy(this->input.getWidth(),this->input.getHeight());

	for(size_t it = 0; it<outterIterations; it++){
		size_t subIterations = innerIterations;
		if(((it+1)*innerIterations)>iterations){
			subIterations = iterations-(it*innerIterations);
		}
		
		if(tiles == 0) {
			Array slice[hTiling];
			Array outputNumberHt[hTiling];
			////////////////////////////////Number of clusters are higher//////////////
			#ifdef DEBUG
				cout<<"MASTER ["<<it<<"]: number of clusters are higher than number of tilings"<<endl;
			#endif

			#ifdef DEBUG
				cout<<"MASTER["<<it<<"]: Barrier CREATED"<<endl;
			#endif
			barrierNbClusters = mppa_create_master_barrier(BARRIER_SYNC_MASTER, BARRIER_SYNC_SLAVE, hTiling);

			#ifdef DEBUG
				cout<<"MASTER["<<it<<"]: Barrier 1"<<endl;
			#endif
			//mppa_barrier_wait(barrierNbClusters);

			for (int i = 0; i < hTiling; i++) {
				outputNumberHt[i].portalAuxWriteAlloc(i);
			}

			#ifdef DEBUG
				cout<<"MASTER["<<it<<"]: Barrier 2"<<endl;
			#endif
			mppa_barrier_wait(barrierNbClusters);

			for (int ht = 0; ht < hTiling; ht++) {
				heightOffset = ht*tilingHeight;
				tiling.tile(subIterations, 0, heightOffset, 0, this->input.getWidth(), tilingHeight, 1);
				outputNumberHt[ht].hostSlice(tiling.input, tiling.widthOffset, tiling.heightOffset, tiling.depthOffset, tiling.width, tiling.height, tiling.depth);
				outputNumberHt[ht].setAux(heightOffset, it, subIterations, tiling.coreWidthOffset, tiling.coreHeightOffset, tiling.coreDepthOffset, tiling.coreWidth, tiling.coreHeight, tiling.coreDepth, outterIterations, tiling.height, tiling.width, tiling.depth);
				outputNumberHt[ht].copyToAux();
				outputNumberHt[ht].waitAuxWrite();
			}
			
			for (int i = 0; i < hTiling; i++) {
				outputNumberHt[i].closeAuxWritePortal();
			}
			#ifdef DEBUG
				cout<<"MASTER["<<it<<"]: Barrier 3"<<endl;
			#endif
			//mppa_barrier_wait(barrierNbClusters);

			for (int i = 0; i < hTiling; i++) {
				slice[i].portalWriteAlloc(i);
			}
			
			mppa_barrier_wait(barrierNbClusters);
			for(int ht = 0; ht < hTiling; ht++) {
				heightOffset = ht*tilingHeight;
				tiling.tile(subIterations, 0, heightOffset, 0, this->input.getWidth(), tilingHeight, 1);
	   			slice[ht].hostSlice(tiling.input, tiling.widthOffset, tiling.heightOffset, tiling.depthOffset, tiling.width, tiling.height, tiling.depth);
	   			slice[ht].copyTo(tiling.heightOffset, 0);
	   			slice[ht].waitWrite();
			}
			
			this->output.setTrigger(hTiling);
			this->output.copyFrom();
			for (int i = 0; i < hTiling; i++) {
				slice[i].closeWritePortal();
			}
			#ifdef DEBUG
				cout<<"MASTER["<<it<<"]: Barrier 4"<<endl;
			#endif
			mppa_close_barrier(barrierNbClusters);
		} else {
			///////////////////////////hTiling is higher///////////////////////////////
			#ifdef DEBUG
				cout<<"MASTER ["<<it<<"]: number of tiles are higher than number of clusters"<<endl;
			#endif
			int counter = 0;

			Array cluster[nb_clusters];

			#ifdef DEBUG
				cout<<"MASTER["<<it<<"]: Barrier CREATED"<<endl;
			#endif
			barrierNbClusters = mppa_create_master_barrier(BARRIER_SYNC_MASTER, BARRIER_SYNC_SLAVE, nb_clusters);

			Array outputNumberNb[nb_clusters];
			Array outputNumberMod[itMod];

			for(int i = 0; i < tiles; i++) {
				#ifdef DEBUG
					cout<<"MASTER["<<it<<":"<<i<<"]: Barrier 1"<<endl;
				#endif
				//mppa_barrier_wait(barrierNbClusters);
				
				for (int i = 0; i < nb_clusters; i++) {
					outputNumberNb[i].portalAuxWriteAlloc(i);
				}

				#ifdef DEBUG
					cout<<"MASTER["<<it<<":"<<i<<"]: Barrier 2"<<endl;
				#endif
				mppa_barrier_wait(barrierNbClusters);

				for (int j = 0; j < nb_clusters; j++) {
					heightOffset = (j+counter)*tilingHeight;
					tiling.tile(subIterations, 0, heightOffset, 0, this->input.getWidth(), tilingHeight, 1);
					outputNumberNb[j].hostSlice(tiling.input, tiling.widthOffset, tiling.heightOffset, tiling.depthOffset, tiling.width, tiling.height, tiling.depth);
					outputNumberNb[j].setAux(heightOffset, it, subIterations, tiling.coreWidthOffset, tiling.coreHeightOffset, tiling.coreDepthOffset, tiling.coreWidth, tiling.coreHeight, tiling.coreDepth, outterIterations, tiling.height, tiling.width, tiling.depth);
					outputNumberNb[j].copyToAux();
					outputNumberNb[j].waitAuxWrite();

				}

				for (int i = 0; i < nb_clusters; i++) {
					outputNumberNb[i].closeAuxWritePortal();
				}

				#ifdef DEBUG
					cout<<"MASTER["<<it<<":"<<i<<"]: Barrier 3"<<endl;
				#endif
				//mppa_barrier_wait(barrierNbClusters);

				for (int i = 0; i < nb_clusters; i++) {
					cluster[i].portalWriteAlloc(i);
				}
				#ifdef DEBUG
					cout<<"MASTER["<<it<<":"<<i<<"]: Barrier 4"<<endl;
				#endif
				mppa_barrier_wait(barrierNbClusters);
				
				for (int j = 0; j < nb_clusters; j++) {
					heightOffset = (j+counter)*tilingHeight;
					
					tiling.tile(subIterations, 0, heightOffset, 0, this->input.getWidth(), tilingHeight, 1);
					
					cluster[j].hostSlice(tiling.input, tiling.widthOffset, tiling.heightOffset, tiling.depthOffset, tiling.width, tiling.height, tiling.depth);
					cluster[j].copyTo(tiling.heightOffset, 0);//j+counter
					cluster[j].waitWrite();

				}

				this->output.copyFrom();
				#ifdef DEBUG
					cout<<"MASTER["<<it<<":"<<i<<"]: Received processed data from clusters"<<endl;
				#endif

				for (int i = 0; i < nb_clusters; i++) {
					cluster[i].closeWritePortal();
				}

		   		counter += nb_clusters;
		   	}

			#ifdef DEBUG
				cout<<"MASTER["<<it<<"]: Barrier 5"<<endl;
			#endif
			//mppa_barrier_wait(barrierNbClusters);

			for(int j = 0; j < itMod; j++) {
				outputNumberMod[j].portalAuxWriteAlloc(j);
			}
			
			#ifdef DEBUG
				cout<<"MASTER["<<it<<"]: Barrier 6"<<endl;
			#endif
			mppa_barrier_wait(barrierNbClusters);

			for (int j = 0; j < itMod; j++) {
				heightOffset = (j+counter)*tilingHeight;
				tiling.tile(subIterations, 0, heightOffset, 0, this->input.getWidth(), tilingHeight, 1);
				outputNumberMod[j].hostSlice(tiling.input, tiling.widthOffset, tiling.heightOffset, tiling.depthOffset, tiling.width, tiling.height, tiling.depth);
				outputNumberMod[j].setAux(heightOffset, it, subIterations, tiling.coreWidthOffset, tiling.coreHeightOffset, tiling.coreDepthOffset, tiling.coreWidth, tiling.coreHeight, tiling.coreDepth, outterIterations, tiling.height, tiling.width, tiling.depth);
				outputNumberMod[j].copyToAux();
				outputNumberMod[j].waitAuxWrite();
			}
			
			for(int i = 0; i < itMod; i++) {
				outputNumberMod[i].closeAuxWritePortal();
			}

			#ifdef DEBUG
				cout<<"MASTER["<<it<<"]: Barrier 7"<<endl;
			#endif
			//mppa_barrier_wait(barrierNbClusters);
			
			for(int j = 0; j < itMod; j++) {
				cluster[j].portalWriteAlloc(j);
			}

			#ifdef DEBUG
				cout<<"MASTER["<<it<<"]: Barrier 8"<<endl;
			#endif
			mppa_barrier_wait(barrierNbClusters);

			for (int j = 0; j < itMod; j++) {
				heightOffset = (j+counter)*tilingHeight;
				tiling.tile(subIterations, 0, heightOffset, 0, this->input.getWidth(), tilingHeight, 1);
				cluster[j].hostSlice(tiling.input, tiling.widthOffset, tiling.heightOffset, tiling.depthOffset, tiling.width, tiling.height, tiling.depth);
				cluster[j].copyTo(tiling.heightOffset, 0);
				cluster[j].waitWrite();
			}
			this->output.setTrigger(itMod);

			
			this->output.copyFrom();
			#ifdef DEBUG
					cout<<"MASTER["<<it<<"]: Received processed data of remaining tiles from clusters"<<endl;
			#endif
			
			for (int i = 0; i < itMod; i++) {
				cluster[i].closeWritePortal();
			}
			for(int i = 0; i < itMod; i++) {
				outputNumberMod[i].auxFree();
			}

			#ifdef DEBUG
				cout<<"MASTER["<<it<<"]: Barrier CLOSED"<<endl;
			#endif
			mppa_close_barrier(barrierNbClusters);
		}
		
		inputCopy.mppaMasterCopy(this->input);
		this->input.mppaMasterCopy(this->output);
		this->output.mppaMasterCopy(inputCopy);
	}
	
	inputCopy.mppaFree();
	this->output.closeReadPortal();
	this->output.mppaMasterCopy(this->input);
	
	#ifdef DEBUG
	
 	for(int h=0;h<this->output.getHeight();h++) {
		for(int w=0;w<this->output.getWidth();w++) {
			printf("FinalOutput(%d,%d):%d\n",h,w, output(h,w));
		}
	}
	
	#endif
}
#endif




#ifdef PSKEL_MPPA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::waitSlaves(int nb_clusters, int tilingHeight) {
	size_t hTiling = ceil(float(this->input.getHeight())/float(tilingHeight));
	int pid;
	if(hTiling < nb_clusters)
		nb_clusters = hTiling;
	for (pid = 0; pid < nb_clusters; pid++) {
    		mppa_waitpid(pid, NULL, 0);
	}
}
#endif


#ifdef PSKEL_MPPA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::scheduleMPPA(const char slave_bin_name[], int nb_clusters, int nb_threads, size_t tilingHeight, int iterations, int innerIterations){
	this->spawn_slaves(slave_bin_name, tilingHeight, nb_clusters, nb_threads, iterations, innerIterations);
	this->mppaSlice(tilingHeight, nb_clusters, iterations, innerIterations);
	this->waitSlaves(nb_clusters, tilingHeight);

}
#endif


#ifdef PSKEL_MPPA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runMPPA(int cluster_id, int nb_threads, int nb_tiles, int outterIterations, int itMod){
	Array finalArr;
	Array coreTmp;
	Array tmp;
	Array inputTmp;
	Array outputTmp;
	Array auxPortal;
	int *aux;

	
	#ifdef DEBUG
	cout<<"SLAVE["<<cluster_id<<"]: opened finalArr portal in write mode"<<endl;
	#endif	

	for(int j = 0; j < outterIterations; j++) {
		barrier_t *global_barrier = mppa_create_slave_barrier(BARRIER_SYNC_MASTER, BARRIER_SYNC_SLAVE);
		for(int i = 0; i < nb_tiles; i++) {
			//mppa_barrier_wait(global_barrier);
			//auxPortal.auxAlloc();
			
			if(i == 0) {
				auxPortal.portalAuxReadAlloc(1, cluster_id);
				finalArr.portalWriteAlloc(0);
				#ifdef DEBUG
				cout<<"SLAVE["<<cluster_id<<"]: opened auxPortal in read mode for iteration #"<<j<<endl;
				#endif
			}
			mppa_barrier_wait(global_barrier);

	
			auxPortal.copyFromAux();

			#ifdef DEBUG
			cout<<"SLAVE["<<cluster_id<<"]: copied data from auxPortal"<<endl;
			#endif
			
			aux = auxPortal.getAux();

			int val = aux[0];
			int it = aux[1];
			int subIterations = aux[2];
			int coreWidthOffset = aux[3];
			int coreHeightOffset = aux[4];
			int coreDepthOffset = aux[5];
			int coreWidth = aux[6];
			int coreHeight = aux[7];
			int coreDepth = aux[8];
			int h = aux[10];
			int w = aux[11];
			int d = aux[12];


			tmp.mppaAlloc(w,h,d);
			inputTmp.mppaAlloc(w,h,d);
			outputTmp.mppaAlloc(w,h,d);

			inputTmp.portalReadAlloc(1, cluster_id);
			mppa_barrier_wait(global_barrier);

			#ifdef DEBUG
			cout<<"SLAVE["<<cluster_id<<"]: opened inputTmp portal in read mode for tile #"<<i<<" of iteration #"<<j<<endl;
			#endif
			



			inputTmp.copyFrom();


			#ifdef DEBUG
			cout<<"SLAVE["<<cluster_id<<"]: is processing tile #"<<i<<" of iteration #"<<j<<" with "<<subIterations<<" subIterations"<<endl;
			#endif

			this->runIterativeMPPA(inputTmp, outputTmp, subIterations, nb_threads);
			
			#ifdef DEBUG
			cout<<"SLAVE["<<cluster_id<<"]: finished processing tile #"<<i<<" of iteration #"<<j<<endl;
			#endif

			if (subIterations%2==0) {
				tmp.mppaMemCopy(inputTmp);
			} else {
				tmp.mppaMemCopy(outputTmp);
			}


			finalArr.hostSlice(tmp, coreWidthOffset, coreHeightOffset, coreDepthOffset, coreWidth, coreHeight, coreDepth);

			finalArr.copyTo(coreHeightOffset, sizeof(int)*val);

			finalArr.waitWrite();
	

			tmp.mppaFree();
			tmp.auxFree();
			#ifdef DEBUG
			cout<<"SLAVE["<<cluster_id<<"]: copied tile #"<<i<<" data of iteration #"<<j<<" to master"<<endl;
			#endif
			

			inputTmp.mppaFree();
			inputTmp.auxFree();

			outputTmp.mppaFree();
			outputTmp.auxFree();

			//auxPortal.mppaFree();

			inputTmp.closeReadPortal();
			#ifdef DEBUG
			cout<<"SLAVE["<<cluster_id<<"]: closed read portal of inputTmp for tile #"<<i<<" of iteration #"<<j<<endl;
			#endif
			if (i == (nb_tiles-1)) {
				auxPortal.closeAuxReadPortal();
				finalArr.closeWritePortal();
				#ifdef DEBUG
				cout<<"SLAVE["<<cluster_id<<"]: closed read portal of auxPortal for iteration #"<<j<<endl;
				#endif
						
			}
		}
		if(cluster_id >= itMod) {

			#ifdef DEBUG
			cout<<"SLAVE["<<cluster_id<<"]: is insided InOut Barrier for iteration "<<j<<endl;
			#endif
			mppa_barrier_wait(global_barrier);
			mppa_barrier_wait(global_barrier);
		}
		//finalArr.mppaFree();
		//finalArr.auxFree();
		mppa_close_barrier(global_barrier);
	}
	
	#ifdef DEBUG
	cout<<"SLAVE["<<cluster_id<<"]: closed write portal for finalArr"<<endl;
	#endif
	
}
#endif

#ifdef PSKEL_MPPA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runIterativeMPPA(Array in, Array out, int iterations, int numThreads){
	for(int i = 0; i < iterations; i++) {
		if(i%2==0) {
			this->runOpenMP(in, out, numThreads);
		} else {
			this->runOpenMP(out, in, numThreads);

		}
	}

}
#endif
//*******************************************************************************************
//*******************************************************************************************
#ifndef MPPA_MASTER
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runCPU(size_t numThreads){
	numThreads = (numThreads==0)?omp_get_num_procs():numThreads;
	#ifdef PSKEL_TBB
		this->runTBB(this->input, this->output, numThreads);
	#else
		this->runOpenMP(this->input, this->output, numThreads);
	#endif
}
#endif

#ifdef PSKEL_CUDA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runGPU(size_t GPUBlockSize){
	if(GPUBlockSize==0){
		int device;
		cudaGetDevice(&device);
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, device);
		//GPUBlockSize = deviceProperties.maxThreadsPerBlock/2;
		GPUBlockSize = deviceProperties.warpSize;
		//int minGridSize, blockSize;
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, stencilTilingCU, 0, in.size());
		//GPUBlockSize = blockSize;
		//cout << "GPUBlockSize: "<<GPUBlockSize<<endl;
		//int maxActiveBlocks;
  		//cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, stencilTilingCU, GPUBlockSize, 0);
		//float occupancy = (maxActiveBlocks * GPUBlockSize / deviceProperties.warpSize) / 
                //    (float)(deviceProperties.maxThreadsPerMultiProcessor / 
                //            deviceProperties.warpSize);
		//printf("Launched blocks of size %d. Theoretical occupancy: %f\n", GPUBlockSize, occupancy);
	}
	input.deviceAlloc();
	output.deviceAlloc();
	mask.deviceAlloc();
	mask.copyToDevice();
	input.copyToDevice();
	//this->setGPUInputData();
	this->runCUDA(this->input, this->output, GPUBlockSize);
	//this->getGPUOutputData();
	output.copyToHost();
	input.deviceFree();
	output.deviceFree();
	mask.deviceFree();
}
#endif

#ifdef PSKEL_CUDA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runTilingGPU(size_t tilingWidth, size_t tilingHeight, size_t tilingDepth, size_t GPUBlockSize){
	if(GPUBlockSize==0){
		int device;
		cudaGetDevice(&device);
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, device);
		//GPUBlockSize = deviceProperties.maxThreadsPerBlock/2;
		GPUBlockSize = deviceProperties.warpSize;
		//int minGridSize, blockSize;
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, stencilTilingCU, 0, in.size());
		//GPUBlockSize = blockSize;
		//cout << "GPUBlockSize: "<<GPUBlockSize<<endl;
		//int maxActiveBlocks;
  		//cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, stencilTilingCU, GPUBlockSize, 0);
		//float occupancy = (maxActiveBlocks * GPUBlockSize / deviceProperties.warpSize) / 
                //    (float)(deviceProperties.maxThreadsPerMultiProcessor / 
                //            deviceProperties.warpSize);
		//printf("Launched blocks of size %d. Theoretical occupancy: %f\n", GPUBlockSize, occupancy);
	}
	size_t wTiling = ceil(float(this->input.getWidth())/float(tilingWidth));
	size_t hTiling = ceil(float(this->input.getHeight())/float(tilingHeight));
	size_t dTiling = ceil(float(this->input.getDepth())/float(tilingDepth));
	mask.deviceAlloc();
	mask.copyToDevice();
	//setGPUMask();
	StencilTiling<Array, Mask> tiling(input, output, mask);
	Array inputTile;
	Array outputTile;
	Array tmp;
	for(size_t ht=0; ht<hTiling; ht++){
	 for(size_t wt=0; wt<wTiling; wt++){
	  for(size_t dt=0; dt<dTiling; dt++){
		size_t heightOffset = ht*tilingHeight;
		size_t widthOffset = wt*tilingWidth;
		size_t depthOffset = dt*tilingDepth;
		//CUDA input memory copy
		tiling.tile(1, widthOffset, heightOffset, depthOffset, tilingWidth, tilingHeight, tilingDepth);
		inputTile.hostSlice(tiling.input, tiling.widthOffset, tiling.heightOffset, tiling.depthOffset, tiling.width, tiling.height, tiling.depth);
		outputTile.hostSlice(tiling.output, tiling.widthOffset, tiling.heightOffset, tiling.depthOffset, tiling.width, tiling.height, tiling.depth);
		inputTile.deviceAlloc();
		outputTile.deviceAlloc();
		inputTile.copyToDevice();
		tmp.hostAlloc(tiling.width, tiling.height, tiling.depth);
		//this->setGPUInputDataIterative(inputCopy, output, innerIterations, widthOffset, heightOffset, depthOffset, tilingWidth, tilingHeight, tilingDepth);
		//CUDA kernel execution
		this->runIterativeTilingCUDA(inputTile, outputTile, tiling, GPUBlockSize);
		tmp.copyFromDevice(outputTile);
		Array coreTmp;
		Array coreOutput;
		coreTmp.hostSlice(tmp, tiling.coreWidthOffset, tiling.coreHeightOffset, tiling.coreDepthOffset, tiling.coreWidth, tiling.coreHeight, tiling.coreDepth);
		coreOutput.hostSlice(outputTile, tiling.coreWidthOffset, tiling.coreHeightOffset, tiling.coreDepthOffset, tiling.coreWidth, tiling.coreHeight, tiling.coreDepth);
		coreOutput.hostMemCopy(coreTmp);
		tmp.hostFree();
	}}}
	inputTile.deviceFree();
	outputTile.deviceFree();
	mask.deviceFree();
	cudaDeviceSynchronize();
}
#endif

#ifdef PSKEL_CUDA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runAutoGPU(size_t GPUBlockSize){
	size_t gpuMemFree, gpuMemTotal;
	//gpuErrchk( cudaDeviceSynchronize() );
	cudaMemGetInfo(&gpuMemFree, &gpuMemTotal);
	if((this->input.memSize()+this->output.memSize()+this->mask.memSize())<(0.998*gpuMemFree)){
		runGPU(GPUBlockSize);
	}else{
		size_t typeSize = this->input.memSize()/this->input.size();
		float div = float(this->input.memSize()+this->output.memSize())/((gpuMemFree-this->mask.memSize())*0.97);
		if(this->input.getHeight()==1){
			size_t width = floor(float(this->input.getWidth())/div);
			width = (width>0)?width:1;
			while( (((this->input.getHeight()*this->input.getDepth()+this->output.getHeight()*this->output.getDepth())*(2*this->mask.getRange() + width))*typeSize + this->mask.memSize()) > gpuMemFree*0.998 ){
				width+=2;
			}
			while( (((this->input.getHeight()*this->input.getDepth()+this->output.getHeight()*this->output.getDepth())*(2*this->mask.getRange() + width))*typeSize + this->mask.memSize()) > gpuMemFree*0.998 ){
				width--;
			}
			runTilingGPU(width, this->input.getHeight(), this->input.getDepth(), GPUBlockSize);
		}else{
			size_t height = floor(float(this->input.getHeight())/div);
			height = (height>0)?height:1;
			while( (((this->input.getWidth()*this->input.getDepth()+this->output.getWidth()*this->output.getDepth())*(2*this->mask.getRange() + height))*typeSize + this->mask.memSize()) < gpuMemFree*0.998 ){
				height+=2;
			}
			while( (((this->input.getWidth()*this->input.getDepth()+this->output.getWidth()*this->output.getDepth())*(2*this->mask.getRange() + height))*typeSize + this->mask.memSize()) > gpuMemFree*0.998 ){
				height--;
			}
			runTilingGPU(this->input.getWidth(), height, this->input.getDepth(), GPUBlockSize);
		}
	}
}
#endif

#ifndef MPPA_MASTER
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runIterativeSequential(size_t iterations){
	Array inputCopy;
	inputCopy.hostClone(input);
	for(size_t it = 0; it<iterations; it++){
		if(it%2==0) this->runSeq(inputCopy, this->output);
		else this->runSeq(this->output, inputCopy);
	}
	if((iterations%2)==0) output.hostMemCopy(inputCopy);
	inputCopy.hostFree();
}
#endif

#ifndef MPPA_MASTER
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runIterativeCPU(size_t iterations, size_t numThreads){
	numThreads = (numThreads==0)?omp_get_num_procs():numThreads;
	//cout << "numThreads: " << numThreads << endl;
	Array inputCopy;
	inputCopy.hostClone(input);
	for(size_t it = 0; it<iterations; it++){
		if(it%2==0){
			#ifdef PSKEL_TBB
				this->runTBB(inputCopy, this->output, numThreads);
			#else
				this->runOpenMP(inputCopy, this->output, numThreads);
			#endif
		}else {
			#ifdef PSKEL_TBB
				this->runTBB(this->output, inputCopy, numThreads);
			#else
				this->runOpenMP(this->output, inputCopy, numThreads);
			#endif
		}
	}
	if((iterations%2)==0) output.hostMemCopy(inputCopy);
	inputCopy.hostFree();
}
#endif

#ifdef PSKEL_CUDA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runIterativeGPU(size_t iterations, size_t GPUBlockSize){
	if(GPUBlockSize==0){
		int device;
		cudaGetDevice(&device);
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, device);
		//GPUBlockSize = deviceProperties.maxThreadsPerBlock/2;
		GPUBlockSize = deviceProperties.warpSize;
		//int minGridSize, blockSize;
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, stencilTilingCU, 0, in.size());
		//GPUBlockSize = blockSize;
		//cout << "GPUBlockSize: "<<GPUBlockSize<<endl;
		//int maxActiveBlocks;
  		//cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, stencilTilingCU, GPUBlockSize, 0);
		//float occupancy = (maxActiveBlocks * GPUBlockSize / deviceProperties.warpSize) / 
                //    (float)(deviceProperties.maxThreadsPerMultiProcessor / 
                //            deviceProperties.warpSize);
		//printf("Launched blocks of size %d. Theoretical occupancy: %f\n", GPUBlockSize, occupancy);
	}
	input.deviceAlloc();
	input.copyToDevice();
	mask.deviceAlloc();
	mask.copyToDevice();
	output.deviceAlloc();
	//output.copyToDevice();
	//this->setGPUInputData();
	for(size_t it = 0; it<iterations; it++){
		if((it%2)==0)
			this->runCUDA(this->input, this->output, GPUBlockSize);
		else this->runCUDA(this->output, this->input, GPUBlockSize);
	}
	if((iterations%2)==1)
		output.copyToHost();
	else output.copyFromDevice(input);
	input.deviceFree();
	mask.deviceFree();
	output.deviceFree();
	//this->getGPUOutputData();
}
#endif

#ifdef PSKEL_CUDA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runIterativeTilingGPU(size_t iterations, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth, size_t innerIterations, size_t GPUBlockSize){
	if(GPUBlockSize==0){
		int device;
		cudaGetDevice(&device);
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, device);
		//GPUBlockSize = deviceProperties.maxThreadsPerBlock/2;
		GPUBlockSize = deviceProperties.warpSize;
		//int minGridSize, blockSize;
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, stencilTilingCU, 0, in.size());
		//GPUBlockSize = blockSize;
		//cout << "GPUBlockSize: "<<GPUBlockSize<<endl;
		//int maxActiveBlocks;
  		//cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, stencilTilingCU, GPUBlockSize, 0);
		//float occupancy = (maxActiveBlocks * GPUBlockSize / deviceProperties.warpSize) / 
                //    (float)(deviceProperties.maxThreadsPerMultiProcessor / 
                //            deviceProperties.warpSize);
		//printf("Launched blocks of size %d. Theoretical occupancy: %f\n", GPUBlockSize, occupancy);
	}
	Array inputCopy;
	inputCopy.hostClone(this->input);
	size_t wTiling = ceil(float(this->input.getWidth())/float(tilingWidth));
	size_t hTiling = ceil(float(this->input.getHeight())/float(tilingHeight));
	size_t dTiling = ceil(float(this->input.getDepth())/float(tilingDepth));
	mask.deviceAlloc();
	mask.copyToDevice();
	//setGPUMask();
	StencilTiling<Array, Mask> tiling(inputCopy, this->output, this->mask);
	Array inputTile;
	Array outputTile;
	Array tmp;
	size_t outterIterations = ceil(float(iterations)/innerIterations);
	for(size_t it = 0; it<outterIterations; it++){
		size_t subIterations = innerIterations;
		if(((it+1)*innerIterations)>iterations){
			subIterations = iterations-(it*innerIterations);
		}
		//cout << "Iteration: " << it << endl;
		//cout << "#SubIterations: " << subIterations << endl;
		for(size_t ht=0; ht<hTiling; ht++){
		 for(size_t wt=0; wt<wTiling; wt++){
		  for(size_t dt=0; dt<dTiling; dt++){
			size_t heightOffset = ht*tilingHeight;
			size_t widthOffset = wt*tilingWidth;
			size_t depthOffset = dt*tilingDepth;

			//CUDA input memory copy
			tiling.tile(subIterations, widthOffset, heightOffset, depthOffset, tilingWidth, tilingHeight, tilingDepth);
			inputTile.hostSlice(tiling.input, tiling.widthOffset, tiling.heightOffset, tiling.depthOffset, tiling.width, tiling.height, tiling.depth);
			outputTile.hostSlice(tiling.output, tiling.widthOffset, tiling.heightOffset, tiling.depthOffset, tiling.width, tiling.height, tiling.depth);
			inputTile.deviceAlloc();
			outputTile.deviceAlloc();
			tmp.hostAlloc(tiling.width, tiling.height, tiling.depth);
			//this->setGPUInputDataIterative(inputCopy, output, innerIterations, widthOffset, heightOffset, depthOffset, tilingWidth, tilingHeight, tilingDepth);
			if(it%2==0){
				inputTile.copyToDevice();
				//CUDA kernel execution
				this->runIterativeTilingCUDA(inputTile, outputTile, tiling, GPUBlockSize);
				if(subIterations%2==0){
					tmp.copyFromDevice(inputTile);
				}else{
					tmp.copyFromDevice(outputTile);
				}
				Array coreTmp;
				Array coreOutput;
				coreTmp.hostSlice(tmp, tiling.coreWidthOffset, tiling.coreHeightOffset, tiling.coreDepthOffset, tiling.coreWidth, tiling.coreHeight, tiling.coreDepth);
				coreOutput.hostSlice(outputTile, tiling.coreWidthOffset, tiling.coreHeightOffset, tiling.coreDepthOffset, tiling.coreWidth, tiling.coreHeight, tiling.coreDepth);
				coreOutput.hostMemCopy(coreTmp);
				//this->copyTilingOutput(output, innerIterations, widthOffset, heightOffset, depthOffset, tilingWidth, tilingHeight, tilingDepth);
				tmp.hostFree();
			}else{
				outputTile.copyToDevice();
				//CUDA kernel execution
				this->runIterativeTilingCUDA(outputTile, inputTile, tiling, GPUBlockSize);
				if(subIterations%2==0){
					tmp.copyFromDevice(outputTile);
				}else{
					tmp.copyFromDevice(inputTile);
				}
				Array coreTmp;
				Array coreInput;
				//cout << "[Computing iteration: " << it << "]" << endl;
				coreTmp.hostSlice(tmp, tiling.coreWidthOffset, tiling.coreHeightOffset, tiling.coreDepthOffset, tiling.coreWidth, tiling.coreHeight, tiling.coreDepth);
				coreInput.hostSlice(inputTile, tiling.coreWidthOffset, tiling.coreHeightOffset, tiling.coreDepthOffset, tiling.coreWidth, tiling.coreHeight, tiling.coreDepth);
				coreInput.hostMemCopy(coreTmp);
				tmp.hostFree();
			}
		}}}
	}
	inputTile.deviceFree();
	outputTile.deviceFree();
	mask.deviceFree();
	cudaDeviceSynchronize();

	if((outterIterations%2)==0) tiling.output.hostMemCopy(tiling.input);
	inputCopy.hostFree();
}
#endif

#ifdef PSKEL_CUDA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runCUDA(Array in, Array out, int GPUBlockSize){
	dim3 DimBlock(GPUBlockSize, GPUBlockSize, 1);
	dim3 DimGrid((in.getWidth() - 1)/GPUBlockSize + 1, (in.getHeight() - 1)/GPUBlockSize + 1, in.getDepth());

	#ifdef PSKEL_SHARED_MASK
	stencilTilingCU<<<DimGrid, DimBlock, (this->mask.size*this->mask.dimension)>>>(in, out, this->mask, this->args, 0,0,0,in.getWidth(),in.getHeight(),in.getDepth());
	#else
	stencilTilingCU<<<DimGrid, DimBlock>>>(in, out, this->mask, this->args, 0,0,0,in.getWidth(),in.getHeight(),in.getDepth());
	#endif
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}
#endif

#ifdef PSKEL_CUDA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runIterativeTilingCUDA(Array in, Array out, StencilTiling<Array, Mask> tiling, size_t GPUBlockSize){
	dim3 DimBlock(GPUBlockSize,GPUBlockSize, 1);
	dim3 DimGrid((in.getWidth() - 1)/GPUBlockSize + 1, (in.getHeight() - 1)/GPUBlockSize + 1, 1);
	//dim3 DimBlock(GPUBlockSize,GPUBlockSize, 1);
	//dim3 DimGrid((in.getWidth() - 1)/GPUBlockSize + 1, (in.getHeight() - 1)/GPUBlockSize + 1, 1);
	size_t maskRange = this->mask.getRange();
	for(size_t it=0; it<tiling.iterations; it++){
		//cout << "[Computing iteration: " << it << "]" << endl;
		//cout << "mask range: " <<maskRange << endl;
		//cout << "mask margin: " <<(maskRange*(tiling.iterations-(it+1))) << endl;
		size_t margin = (maskRange*(tiling.iterations-(it+1)));
		size_t widthOffset = 0;
		size_t extra = 0;
		if(tiling.coreWidthOffset>margin){
			widthOffset = tiling.coreWidthOffset-margin;
		}else extra = margin-widthOffset;
		//cout << "width extra: " << extra << endl;
		size_t width = tiling.coreWidth+margin*2 - extra;
		if((widthOffset+width)>=tiling.width){
			width = tiling.width-widthOffset;
		}
		size_t heightOffset = 0;
		extra = 0;
		if(tiling.coreHeightOffset>margin){
			heightOffset = tiling.coreHeightOffset-margin;
		}else extra = margin-heightOffset;
		//cout << "height extra: " << extra << endl;
		size_t height = tiling.coreHeight+margin*2-extra;
		if((heightOffset+height)>=tiling.height){
			height = tiling.height-heightOffset;
		}
		size_t depthOffset = 0;
		extra = 0;
		if(tiling.coreDepthOffset>margin){
			depthOffset = tiling.coreDepthOffset-margin;
		}else extra = margin-depthOffset;
		//cout << "depth extra: " << extra << endl;
		size_t depth = tiling.coreDepth+margin*2-extra;
		if((depthOffset+depth)>=tiling.depth){
			depth = tiling.depth-depthOffset;
		}
		
		//cout << "width-offset: " <<widthOffset << endl;
		//cout << "height-offset: " <<heightOffset << endl;
		//cout << "depth-offset: " <<depthOffset << endl;
		
		//cout << "width: " <<width << endl;
		//cout << "height: " <<height << endl;
		//cout << "depth: " <<depth << endl;
		if(it%2==0){
			#ifdef PSKEL_SHARED_MASK
			stencilTilingCU<<<DimGrid, DimBlock, (this->mask.size*this->mask.dimension)>>>(in, out, this->mask, this->args, widthOffset, heightOffset, depthOffset, width, height, depth);
			#else
			stencilTilingCU<<<DimGrid, DimBlock>>>(in, out, this->mask, this->args, widthOffset, heightOffset, depthOffset, width, height, depth);
			#endif
		}else{
			#ifdef PSKEL_SHARED_MASK
			stencilTilingCU<<<DimGrid, DimBlock, (this->mask.size*this->mask.dimension)>>>(out, in, this->mask, this->args, widthOffset, heightOffset, depthOffset, width, height, depth);
			#else
			stencilTilingCU<<<DimGrid, DimBlock>>>(out, in, this->mask, this->args, widthOffset, heightOffset, depthOffset, width, height, depth);
			#endif
		}
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
	}
}
#endif

#ifdef PSKEL_CUDA
struct TilingGPUGeneticEvaluationFunction{
    size_t iterations;
    size_t height;
    size_t width;
    size_t depth;
    size_t range;
    size_t typeSize;
    size_t memFree;
    size_t popsize;
    size_t ngen;
    size_t dw;
    size_t dt;
    size_t dh;
    float score;
};
TilingGPUGeneticEvaluationFunction tilingGPUEvaluator;

float objective2D(GAGenome &c){
	GABin2DecGenome &genome = (GABin2DecGenome &)c;
	
	float h = genome.phenotype(0);
	float it = genome.phenotype(1);
	size_t tileHeight = ((tilingGPUEvaluator.height<=(2*it*tilingGPUEvaluator.range + h))?tilingGPUEvaluator.height:(2*it*tilingGPUEvaluator.range + h));
 
	if(2*(tilingGPUEvaluator.width*tilingGPUEvaluator.depth*tileHeight*tilingGPUEvaluator.typeSize) > tilingGPUEvaluator.memFree)return 0;
	else {
		float val = h/tileHeight;
		return val*((it*h)/(tilingGPUEvaluator.height*tilingGPUEvaluator.iterations));
	}
}

void solve2D(unsigned int seed){
	int popsize = tilingGPUEvaluator.popsize;
	int ngen = tilingGPUEvaluator.ngen;
	float pmut = 0.01;
	float pcross = 0.6;
	
	float div = (2.0*(tilingGPUEvaluator.width*tilingGPUEvaluator.depth*tilingGPUEvaluator.height*tilingGPUEvaluator.typeSize))/(tilingGPUEvaluator.memFree*1.1);
	size_t maxHeight = ceil(float(tilingGPUEvaluator.height)/div);
	//Create a phenotype for two variables.  The number of bits you can use to
	//represent any number is limited by the type of computer you are using.  In
	//this case, we use 16 bits to represent a floating point number whose value
	//can range from -5 to 5, inclusive.  The bounds on x1 and x2 can be applied
	//here and/or in the objective function.
	GABin2DecPhenotype map;
	map.add(16, 1, maxHeight); //min/max boundaries, inclusive
	map.add(16, 1, tilingGPUEvaluator.iterations);

	//Create the template genome using the phenotype map we just made.
	GABin2DecGenome genome(map, objective2D);

	//Now create the GA using the genome and run it.  We'll use sigma truncation
	//scaling so that we can handle negative objective scores.
	GASimpleGA ga(genome);
	GASigmaTruncationScaling scaling;
	ga.populationSize(popsize);
	ga.nGenerations(ngen);
	ga.pMutation(pmut);
	ga.pCrossover(pcross);
	ga.scaling(scaling);
	ga.scoreFrequency(0);
	ga.flushFrequency(0); //stop flushing the record of the score of given generations
	//ga.scoreFilename(0); //stop recording the score of given generations
	ga.evolve(seed);

	//Obtains the best individual from the best population evolved
	genome = ga.statistics().bestIndividual();

	//cout << "the ga found an optimum at the point (";
	//cout << genome.phenotype(0) << ", " << genome.phenotype(1) << ")\n\n";
	//cout << "best of generation data are in '" << ga.scoreFilename() << "'\n";
	tilingGPUEvaluator.dw = tilingGPUEvaluator.width;
	tilingGPUEvaluator.dh = genome.phenotype(0);//height;
	tilingGPUEvaluator.dt = genome.phenotype(1);//subIterations;
	tilingGPUEvaluator.score = objective2D(genome);
}

float objective3D(GAGenome &c){
	GABin2DecGenome &genome = (GABin2DecGenome &)c;
	
	float w = genome.phenotype(0);
	float h = genome.phenotype(1);
	float t = genome.phenotype(2);
	float tileWidth = ((tilingGPUEvaluator.width<=(2*t*tilingGPUEvaluator.range + w))?tilingGPUEvaluator.width:(2*t*tilingGPUEvaluator.range + w));
	float tileHeight = ((tilingGPUEvaluator.height<=(2*t*tilingGPUEvaluator.range + h))?tilingGPUEvaluator.height:(2*t*tilingGPUEvaluator.range + h));
 
	if(2*(tileWidth*tileHeight*tilingGPUEvaluator.depth*tilingGPUEvaluator.typeSize) > tilingGPUEvaluator.memFree) return 0;
	else {
		float val = (w*h)/(tileWidth*tileHeight);
		return val*((w*h*t)/(tilingGPUEvaluator.width*tilingGPUEvaluator.height*tilingGPUEvaluator.iterations));
	}
}

void solve3D(unsigned int seed){
	int popsize = tilingGPUEvaluator.popsize;
	int ngen = tilingGPUEvaluator.ngen;
	float pmut = 0.01;
	float pcross = 0.6;
	
	//float div = (2.0*(tilingGPUEvaluator.width*tilingGPUEvaluator.depth*tilingGPUEvaluator.height*tilingGPUEvaluator.typeSize))/(tilingGPUEvaluator.memFree*1.1);
	//size_t maxHeight = ceil(float(tilingGPUEvaluator.height)/div);
	//Create a phenotype for two variables.  The number of bits you can use to
	//represent any number is limited by the type of computer you are using.  In
	//this case, we use 16 bits to represent a floating point number whose value
	//can range from -5 to 5, inclusive.  The bounds on x1 and x2 can be applied
	//here and/or in the objective function.
	GABin2DecPhenotype map;
	//map.add(16, 1, maxHeight); //min/max boundaries, inclusive
	map.add(16, 1, tilingGPUEvaluator.width);
	map.add(16, 1, tilingGPUEvaluator.height);
	map.add(16, 1, tilingGPUEvaluator.iterations);

	//Create the template genome using the phenotype map we just made.
	GABin2DecGenome genome(map, objective3D);

	//Now create the GA using the genome and run it.  We'll use sigma truncation
	//scaling so that we can handle negative objective scores.
	GASimpleGA ga(genome);
	GASigmaTruncationScaling scaling;
	ga.populationSize(popsize);
	ga.nGenerations(ngen);
	ga.pMutation(pmut);
	ga.pCrossover(pcross);
	ga.scaling(scaling);
	ga.scoreFrequency(0);
	ga.flushFrequency(0); //stop flushing the record of the score of given generations
	//ga.scoreFilename(0); //stop recording the score of given generations
	ga.evolve(seed);

	//Obtains the best individual from the best population evolved
	genome = ga.statistics().bestIndividual();

	//cout << "the ga found an optimum at the point (";
	//cout << genome.phenotype(0) << ", " << genome.phenotype(1) << ")\n\n";
	//cout << "best of generation data are in '" << ga.scoreFilename() << "'\n";
	tilingGPUEvaluator.dw = genome.phenotype(0);//width;
	tilingGPUEvaluator.dh = genome.phenotype(1);//height;
	tilingGPUEvaluator.dt = genome.phenotype(2);//subIterations;
	tilingGPUEvaluator.score = objective3D(genome);
}

template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runIterativeAutoGPU(size_t iterations, size_t GPUBlockSize){
	size_t gpuMemFree, gpuMemTotal;
	//gpuErrchk( cudaDeviceSynchronize() );
	cudaMemGetInfo(&gpuMemFree, &gpuMemTotal);
	if((this->input.memSize()+this->output.memSize()+this->mask.memSize())<(0.999*gpuMemFree)){
		runIterativeGPU(iterations, GPUBlockSize);
	}else if(this->input.getHeight()==1){
		//solving for a 'transposed matrix'
		tilingGPUEvaluator.typeSize = this->input.memSize()/this->input.size();
		tilingGPUEvaluator.iterations = iterations;
		tilingGPUEvaluator.width = this->input.getDepth(); //'transposed matrix'
		tilingGPUEvaluator.height = this->input.getWidth(); //'transposed matrix'
		tilingGPUEvaluator.depth = 1;
		tilingGPUEvaluator.range = this->mask.getRange();
		tilingGPUEvaluator.memFree = (gpuMemFree-this->mask.memSize())*0.999;//gpuMemFree*0.998;

		tilingGPUEvaluator.popsize = 100;
		tilingGPUEvaluator.ngen = 2500;

  		unsigned int seed = time(NULL);
		solve2D(seed);

		size_t subIterations = tilingGPUEvaluator.dt;
		size_t width = tilingGPUEvaluator.dh;
		//cout << "GPU Mem Total: "<< gpuMemTotal <<endl;
		//cout << "GPU Mem Free: "<< gpuMemFree <<endl;
		//cout << "sub iterations: "<< subIterations <<endl;
		//cout << "tiling height: "<<height<<endl;
		runIterativeTilingGPU(iterations, width, 1, this->input.getDepth(), subIterations, GPUBlockSize);
		
	}else {
		size_t typeSize = this->input.memSize()/this->input.size();
		tilingGPUEvaluator.typeSize = typeSize;
		tilingGPUEvaluator.iterations = iterations;
		tilingGPUEvaluator.width = this->input.getWidth();
		tilingGPUEvaluator.height = this->input.getHeight();
		tilingGPUEvaluator.depth = this->input.getDepth();
		tilingGPUEvaluator.range = this->mask.getRange();
		tilingGPUEvaluator.memFree = (gpuMemFree-this->mask.memSize())*0.999;//gpuMemFree*0.998;
		if( (2*(1+2*this->mask.getRange())*(this->input.getWidth()*this->input.getDepth())*typeSize+this->mask.memSize()) > (0.98*gpuMemFree) ){
			tilingGPUEvaluator.popsize = 100;
			tilingGPUEvaluator.ngen = 2500;
	  		unsigned int seed = time(NULL);
			solve3D(seed);

			size_t width = tilingGPUEvaluator.dw;
			size_t height = tilingGPUEvaluator.dh;
			size_t subIterations = tilingGPUEvaluator.dt;
			//cout << "GPU Mem Total: "<< gpuMemTotal <<endl;
			//cout << "GPU Mem Free: "<< gpuMemFree <<endl;
			//cout << "sub iterations: "<< subIterations <<endl;
			//cout << "tiling height: "<<height<<endl;
			runIterativeTilingGPU(iterations, width, height, this->input.getDepth(), subIterations, GPUBlockSize);
		}else{
			tilingGPUEvaluator.popsize = 100;
			tilingGPUEvaluator.ngen = 2500;
	  		unsigned int seed = time(NULL);
			solve2D(seed);

			size_t subIterations = tilingGPUEvaluator.dt;
			size_t height = tilingGPUEvaluator.dh;
			//cout << "GPU Mem Total: "<< gpuMemTotal <<endl;
			//cout << "GPU Mem Free: "<< gpuMemFree <<endl;
			//cout << "sub iterations: "<< subIterations <<endl;
			//cout << "tiling height: "<<height<<endl;
			runIterativeTilingGPU(iterations, this->input.getWidth(), height, this->input.getDepth(), subIterations, GPUBlockSize);
		}
	}
}
#endif

//*******************************************************************************************
// Stencil 3D
//*******************************************************************************************


template<class Array, class Mask, class Args>
Stencil3D<Array,Mask,Args>::Stencil3D(){}
	
template<class Array, class Mask, class Args>
Stencil3D<Array,Mask,Args>::Stencil3D(Array _input, Array _output, Mask _mask, Args _args){
	this->input = _input;
	this->output = _output;
	this->args = _args;
	this->mask = _mask;
}

#ifndef MPPA_MASTER
template<class Array, class Mask, class Args>
void Stencil3D<Array,Mask,Args>::runSeq(Array in, Array out){
	for (int h = 0; h < in.getHeight(); h++){
	for (int w = 0; w < in.getWidth(); w++){
	for (int d = 0; d < in.getDepth(); d++){
		stencilKernel(in,out,this->mask,this->args,h,w,d);
	}}}
}
#endif

#ifndef MPPA_MASTER
template<class Array, class Mask, class Args>
void Stencil3D<Array,Mask,Args>::runOpenMP(Array in, Array out, size_t numThreads){
	omp_set_num_threads(numThreads);
	#pragma omp parallel for
	for (int h = 0; h < in.getHeight(); h++){
	for (int w = 0; w < in.getWidth(); w++){
	for (int d = 0; d < in.getDepth(); d++){
		stencilKernel(in,out,this->mask,this->args,h,w,d);
	}}}
}
#endif

#ifdef PSKEL_TBB
template<class Array, class Mask, class Args>
struct TBBStencil3D{
	Array input;
	Array output;
	Mask mask;
	Args args;
	TBBStencil3D(Array input, Array output, Mask mask, Args args){
		this->input = input;
		this->output = output;
		this->mask = mask;
		this->args = args;
	}
	void operator()(tbb::blocked_range<int> r)const{
		for (int h = r.begin(); h != r.end(); h++){
		for (int w = 0; w < this->input.getWidth(); w++){
		for (int d = 0; d < this->input.getDepth(); d++){
			stencilKernel(this->input,this->output,this->mask,this->args,h,w,d);
		}}}
	}
};

template<class Array, class Mask, class Args>
void Stencil3D<Array,Mask,Args>::runTBB(Array in, Array out, size_t numThreads){
	TBBStencil3D<Array, Mask, Args> tbbstencil(in, out, this->mask, this->args);
	tbb::task_scheduler_init init(numThreads);
	tbb::parallel_for(tbb::blocked_range<int>(0, in.getHeight()), tbbstencil);
}
#endif

//*******************************************************************************************
// Stencil 2D
//*******************************************************************************************

template<class Array, class Mask, class Args>
Stencil2D<Array,Mask,Args>::Stencil2D(){}

template<class Array, class Mask, class Args>
Stencil2D<Array,Mask,Args>::Stencil2D(Array _input, Array _output, Mask _mask, Args _args){
	this->input = _input;
	this->output = _output;
	this->args = _args;
	this->mask = _mask;
}

template<class Array, class Mask, class Args>
Stencil2D<Array,Mask,Args>::Stencil2D(Array _input, Array _output, Mask _mask){
	this->input = _input;
	this->output = _output;
	this->mask = _mask;
}


template<class Array, class Mask, class Args>
Stencil2D<Array,Mask,Args>::~Stencil2D(){
	#ifdef PSKEL_CUDA
		this->input.deviceFree();
		this->output.deviceFree();
	#endif
	#ifdef PSKEL_MPPA
		this->input.mppaFree();
		this->output.mppaFree();
	#endif
	#ifdef PSKEL_CPU
		this->input.hostFree();
		this->output.hostFree();
	#endif
}

#ifndef MPPA_MASTER
template<class Array, class Mask, class Args>
void Stencil2D<Array,Mask,Args>::runSeq(Array in, Array out){
	for (int h = 0; h < in.getHeight(); h++){
	for (int w = 0; w < in.getWidth(); w++){
		stencilKernel(in,out,this->mask, this->args,h,w);
	}}
}
#endif

#ifndef MPPA_MASTER
template<class Array, class Mask, class Args>
void Stencil2D<Array,Mask,Args>::runOpenMP(Array in, Array out, size_t numThreads){
	omp_set_num_threads(numThreads);
	#pragma omp parallel for
	for (int h = 0; h < in.getHeight(); h++){
	for (int w = 0; w < in.getWidth(); w++){
		stencilKernel(in,out,this->mask, this->args,h,w);
	}}
}
#endif

#ifdef PSKEL_TBB
template<class Array, class Mask, class Args>
struct TBBStencil2D{
	Array input;
	Array output;
	Mask mask;
	Args args;
	TBBStencil2D(Array input, Array output, Mask mask, Args args){
		this->input = input;
		this->output = output;
		this->mask = mask;
		this->args = args;
	}
	void operator()(tbb::blocked_range<int> r)const{
		for (int h = r.begin(); h != r.end(); h++){
		for (int w = 0; w < this->input.getWidth(); w++){
			stencilKernel(this->input,this->output,this->mask, this->args,h,w);
		}}
	}
};

template<class Array, class Mask, class Args>
void Stencil2D<Array,Mask,Args>::runTBB(Array in, Array out, size_t numThreads){
	TBBStencil2D<Array, Mask, Args> tbbstencil(in, out, this->mask, this->args);
	tbb::task_scheduler_init init(numThreads);
	tbb::parallel_for(tbb::blocked_range<int>(0, in.getHeight()), tbbstencil);
}
#endif

//*******************************************************************************************
// Stencil 1D
//*******************************************************************************************


template<class Array, class Mask, class Args>
Stencil<Array,Mask,Args>::Stencil(){}
	
template<class Array, class Mask, class Args>
Stencil<Array,Mask,Args>::Stencil(Array _input, Array _output, Mask _mask, Args _args){
	this->input = _input;
	this->output = _output;
	this->args = _args;
	this->mask = _mask;
}

#ifndef MPPA_MASTER
template<class Array, class Mask, class Args>
void Stencil<Array,Mask,Args>::runSeq(Array in, Array out){
	for (int i = 0; i < in.getWidth(); i++){
		stencilKernel(in,out,this->mask, this->args,i);
	}
}
#endif

#ifndef MPPA_MASTER
template<class Array, class Mask, class Args>
void Stencil<Array,Mask,Args>::runOpenMP(Array in, Array out, size_t numThreads){
	omp_set_num_threads(numThreads);
	#pragma omp parallel for
	for (int i = 0; i < in.getWidth(); i++){
		stencilKernel(in,out,this->mask, this->args,i);
	}
}
#endif

#ifdef PSKEL_TBB
template<class Array, class Mask, class Args>
struct TBBStencil{
	Array input;
	Array output;
	Mask mask;
	Args args;
	TBBStencil(Array input, Array output, Mask mask, Args args){
		this->input = input;
		this->output = output;
		this->mask = mask;
		this->args = args;
	}
	void operator()(tbb::blocked_range<int> r)const{
		for (int i = r.begin(); i != r.end(); i++){
			stencilKernel(this->input,this->output,this->mask, this->args,i);
		}
	}
};

template<class Array, class Mask, class Args>
void Stencil<Array,Mask,Args>::runTBB(Array in, Array out, size_t numThreads){
	TBBStencil<Array, Mask, Args> tbbstencil(in, out, this->mask, this->args);
	tbb::task_scheduler_init init(numThreads);
	tbb::parallel_for(tbb::blocked_range<int>(0, in.getWidth()), tbbstencil);
}
#endif

}//end namespace


#endif
