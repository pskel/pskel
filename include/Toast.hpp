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

#ifndef TOAST_HPP
#define TOAST_HPP

using namespace std;

namespace PSkel{
	
#ifdef PSKEL_GA
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

#ifdef PSKEL_CUDA
template<class Array, class Mask, class Args>
void StencilBase<Array, Mask,Args>::runIterativeTilingGPU(size_t iterations, size_t tilingWidth, size_t tilingHeight, size_t tilingDepth, size_t innerIterations, size_t GPUBlockSizeX, size_t GPUBlockSizeY){
	if(GPUBlockSizeX==0){
		int device;
		cudaGetDevice(&device);
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, device);
		//GPUBlockSize = deviceProperties.maxThreadsPerBlock/2;
		GPUBlockSizeX = GPUBlockSizeY = deviceProperties.warpSize;
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
		//cout << "Iteration: " << it << end
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
				this->runIterativeTilingCUDA(inputTile, outputTile, tiling, GPUBlockSizeX, GPUBlockSizeY);
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
				this->runIterativeTilingCUDA(outputTile, inputTile, tiling, GPUBlockSizeX, GPUBlockSizeY);
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
	
#endif
}//end namespace

#endif
