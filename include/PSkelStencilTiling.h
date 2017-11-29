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

#ifndef PSKEL_STENCIL_TILING_H
#define PSKEL_STENCIL_TILING_H

namespace PSkel{

/**
 * Class that performs the necessary calculations -- regarding the halo region, tile size, etc. --  for tiling stencil computations.
 **/
template<class Array, class Mask>
class StencilTiling {
public:
	Mask mask;
	Array input;
	Array output;
	
	size_t iterations;

	//global offsets that define the tiling
	//necessary for the given iterations,
	//i.e., including the halo region (neighbourhood margin)
	size_t widthOffset;
	size_t heightOffset;
	size_t depthOffset;
	size_t width;
	size_t height;
	size_t depth;

	//core offsets represent the inner margin
	//that consist of the halo region (neighbourhood) for 
	//the given iterations
	size_t coreWidthOffset;
	size_t coreHeightOffset;
	size_t coreDepthOffset;
	size_t coreWidth;
	size_t coreHeight;
	size_t coreDepth;

	StencilTiling(Array in, Array out, Mask mask){
		this->input = in;
		this->output = out;
		this->mask = mask;
	}

	/**
	 * Updates the stencil tiling information for the specified number of iterations and tile size.
	 * \param[in] iterations number of iterations consecutively executed on the device (GPU).
	 * \param[in] widthOffset width offset for the logical tile region, considering the input and output arrays.
	 * \param[in] heightOffset height offset for the logical tile region, considering the input and output arrays.
	 * \param[in] depthOffset depth offset for the logical tile region, considering the input and output arrays.
	 * \param[in] width width of for the logical tile region.
	 * \param[in] height height of the logical tile region.
	 * \param[in] depth depth of the logical tile region.
	 **/
	void tile(size_t iterations, size_t widthOffset, size_t heightOffset, size_t depthOffset, size_t width, size_t height, size_t depth){
		/*std::cout<<"-----------------------"<<std::endl;
		std::cout<<"STENCIL TILING original:"<<std::endl;
		std::cout<<"width: "<<width <<std::endl;
		std::cout<<"height: "<<height<<std::endl;
		std::cout<<"depth: "<<depth<<std::endl;
		std::cout<<"-----------------------"<<std::endl;
		*/
		this->iterations = iterations;
		//check for unaligned tiling
		if((widthOffset+width)>this->input.getWidth())
			width = this->input.getWidth()-widthOffset;
		if((heightOffset+height)>this->input.getHeight())
			height = this->input.getHeight()-heightOffset;
		if((depthOffset+depth)>this->input.getDepth())
			depth = this->input.getDepth()-depthOffset;

		this->coreWidthOffset = widthOffset; //temporary value of core width offset
		this->coreHeightOffset = heightOffset; //temporary value of core height offset
		this->coreDepthOffset = depthOffset; //temporary value of core depth offset
		this->coreWidth = width; //set the width for the logical tile region
		this->coreHeight = height; //set the height for the logical tile region
		this->coreDepth = depth; //set the depth for the logical tile region
		
		size_t maskRange = mask.getRange()*iterations;

		int widthExtra = 0;
		int heightExtra = 0;
		int depthExtra = 0;

		if(widthOffset>=maskRange){
			widthOffset=widthOffset-maskRange;
		}else{
			widthOffset = 0;
			widthExtra = maskRange-widthOffset;
		}
		width = width+(maskRange*2)-widthExtra;
		if((widthOffset+width)>this->input.getWidth())
			width = this->input.getWidth()-widthOffset;

		if(heightOffset>=maskRange){
			heightOffset=heightOffset-maskRange;
		}else{
			heightOffset = 0;
			heightExtra = maskRange-heightOffset;
		}
		height = height+(maskRange*2)-heightExtra;
		if((heightOffset+height)>this->input.getHeight())
			height = this->input.getHeight()-heightOffset;

		if(depthOffset>=maskRange){
			depthOffset=depthOffset-maskRange;
		}else{
			depthOffset = 0;
			depthExtra = maskRange-depthOffset;
		}
		depth = depth+(maskRange*2)-depthExtra;
		if((depthOffset+depth)>this->input.getDepth())
			depth = this->input.getDepth()-depthOffset;
		
		this->coreWidthOffset -= widthOffset; //final value of core width offset
		this->coreHeightOffset -= heightOffset; //final value of core height offset
		this->coreDepthOffset -= depthOffset; //final value of core depth offset
		
		this->widthOffset = widthOffset;
		this->heightOffset = heightOffset;
		this->depthOffset = depthOffset;
		this->width = width;
		this->height = height;
		this->depth = depth;
	
		/*
		std::cout<<"-----------------------"<<std::endl;
		std::cout<<"STENCIL TILING after:"<<std::endl;
		std::cout<<"width: "<<width <<std::endl;
		std::cout<<"height: "<<height<<std::endl;
		std::cout<<"depth: "<<depth<<std::endl;
		std::cout<<"widthOffset: "<<widthOffset <<std::endl;
		std::cout<<"heightOffset: "<<heightOffset<<std::endl;
		std::cout<<"depthOffset: "<<depthOffset<<std::endl;
		std::cout<<"coreWidthOffset: "<<this->coreWidthOffset<<std::endl;
		std::cout<<"coreHeightOffset: "<<this->coreHeightOffset<<std::endl;
		std::cout<<"coreDepthOffset: "<<this->coreDepthOffset<<std::endl;
		std::cout<<"-----------------------"<<std::endl;
		*/
	}	
	
};

}

#endif
