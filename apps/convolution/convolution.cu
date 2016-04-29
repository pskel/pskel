#define PSKEL_TBB
#define PSKEL_CUDA

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "../../include/PSkel.h"
#include "../../include/hr_time.h"
#define MASK_RADIUS 2
#define MASK_WIDTH  5

using namespace std;
using namespace PSkel;

struct Coeff{
	int null;
};

//*******************************************************************************************
// WB_IMAGE
//*******************************************************************************************
struct wbImage_t
{
    int  _imageWidth;
    int  _imageHeight;
    int  _imageChannels;
	PSkel::Array2D<float> _red;
	PSkel::Array2D<float> _green;
	PSkel::Array2D<float> _blue;

    wbImage_t(int imageWidth = 0, int imageHeight = 0, int imageChannels = 0) :_imageWidth(imageWidth), _imageHeight(imageHeight), _imageChannels(imageChannels)
    {
		new (&_red) PSkel::Array2D<float>(_imageWidth,_imageHeight);
		new (&_green) PSkel::Array2D<float>(_imageWidth,_imageHeight);
		new (&_blue) PSkel::Array2D<float>(_imageWidth,_imageHeight);
    }
};

int wbImage_getWidth(const wbImage_t& image){
    return image._imageWidth;
}

int wbImage_getHeight(const wbImage_t& image){
    return image._imageHeight;
}

int wbImage_getChannels(const wbImage_t& image){
    return image._imageChannels;
}

wbImage_t wbImport(char* inputFile){
    wbImage_t image;
    image._imageChannels = 3;

    std::ifstream fileInput;
    fileInput.open(inputFile, std::ios::binary);
    if (fileInput.is_open()) {
        char magic[2];
        fileInput.read(magic, 2);
        if (magic[0] != 'P' || magic[1] !='6') {
            std::cout << "expected 'P6' but got " << magic[0] << magic[1] << std::endl;
            exit(1);
        }
        char tmp = fileInput.peek();
        while (isspace(tmp)) {
            fileInput.read(&tmp, 1);
            tmp = fileInput.peek();
        }
        // filter image comments
        if (tmp == '#') {
            fileInput.read(&tmp, 1);
            tmp = fileInput.peek();
            while (tmp != '\n') {
                fileInput.read(&tmp, 1);
                tmp = fileInput.peek();
            }
        }
        // get rid of whitespaces
        while (isspace(tmp)) {
            fileInput.read(&tmp, 1);
            tmp = fileInput.peek();
        }

        //read dimensions (TODO add error checking)
        char widthStr[64], heightStr[64], numColorsStr[64], *p;
        p = widthStr;
        if(isdigit(tmp)) {
            while(isdigit(*p = fileInput.get())) {
                p++;
            }
            *p = '\0';
            image._imageWidth = atoi(widthStr);
            //std::cout << "Width: " << image._imageWidth << std::endl;
            p = heightStr;
            while(isdigit(*p = fileInput.get())) {
                p++;
            }
            *p = '\0';
            image._imageHeight = atoi(heightStr);
            //std::cout << "Height: " << image._imageHeight << std::endl;
            p = numColorsStr;
            while(isdigit(*p = fileInput.get())) {
                p++;
            }
            *p = '\0';
            int numColors = atoi(numColorsStr);
            //std::cout << "Num colors: " << numColors << std::endl;
            if (numColors != 255) {
                std::cout << "the number of colors should be 255, but got " << numColors << std::endl;
                exit(1);
            }
        } else  {
            std::cout << "error - cannot read dimensions" << std::endl;
        }

        int dataSize = image._imageWidth*image._imageHeight*image._imageChannels;
        unsigned char* data = new unsigned char[dataSize];
        fileInput.read((char*)data, dataSize);
        //float* floatData = new float[dataSize];
        
		new (&(image._red)) PSkel::Array2D<float>(image._imageWidth,image._imageHeight);
		new (&(image._green)) PSkel::Array2D<float>(image._imageWidth,image._imageHeight);
		new (&(image._blue)) PSkel::Array2D<float>(image._imageWidth,image._imageHeight);
		
		for (int y = 0; y < image._imageHeight; y++){
			for (int x = 0; x < image._imageWidth; x++){
					image._red(x,y) = 	1.0*data[(y*image._imageWidth + x)*3 + 0]/255.0f;
					image._green(x,y) = 1.0*data[(y*image._imageWidth + x)*3 + 1]/255.0f;
					image._blue(x,y) = 	1.0*data[(y*image._imageWidth + x)*3 + 2]/255.0f;
					
					/*if(x==1000 && y==1000){
						cout<<(int)data[(y*image._imageWidth + x)*3 + 0]<<" ";
						cout<<(int)data[(y*image._imageWidth + x)*3 + 1]<<" ";
						cout<<(int)data[(y*image._imageWidth + x)*3 + 2]<<endl;
					}*/
			}
		}
        fileInput.close();
    } else  {
         std::cout << "cannot open file " << inputFile;
         exit(1);
    }
    return image;
}

wbImage_t wbImage_new(int imageWidth, int imageHeight, int imageChannels)
{
    wbImage_t image(imageWidth, imageHeight, imageChannels);
    return image;
}

void wbImage_save(wbImage_t& image, char* outputfile) {
    std::ofstream outputFile(outputfile, std::ios::binary);
    char buffer[64];
    std::string magic = "P6\n";
    outputFile.write(magic.c_str(), magic.size());
    std::string comment  =  "# image generated by applying convolution\n";
    outputFile.write(comment.c_str(), comment.size());
    //write dimensions
    sprintf(buffer,"%d", image._imageWidth);
    outputFile.write(buffer, strlen(buffer));
    buffer[0] = ' ';
    outputFile.write(buffer, 1);
    sprintf(buffer,"%d", image._imageHeight);
    outputFile.write(buffer, strlen(buffer));
    buffer[0] = '\n';
    outputFile.write(buffer, 1);
    std::string colors = "255\n";
    outputFile.write(colors.c_str(), colors.size());

    int dataSize = image._imageWidth*image._imageHeight*image._imageChannels;
    unsigned char* rgbData = new unsigned char[dataSize];

	
	for (int y = 0; y < image._imageHeight; y++){		
		for (int x = 0; x < image._imageWidth; x++){
			rgbData[(y*image._imageWidth + x)*3 + 0] = ceil(image._red(x,y) * 255);
			rgbData[(y*image._imageWidth + x)*3 + 1] = ceil(image._green(x,y) * 255);
			rgbData[(y*image._imageWidth + x)*3 + 2] = ceil(image._blue(x,y) * 255);
		}	
	}

	outputFile.write((char*)rgbData, dataSize);
    delete[] rgbData;
	outputFile.close();
}

//*******************************************************************************************
// CONVOLUTION
//*******************************************************************************************

namespace PSkel{
	__parallel__ void stencilKernel(Array2D<float> input, Array2D<float> output, Mask2D<float> mask, int null, size_t i,size_t j){
		//float accum = 0.0f;
		/*for(int n=0;n<mask.size;n++){
			accum += mask.get(n,input,i,j) * mask.getWeight(n);
		}
		output(i,j)= accum;
		*/
		output(i,j) = mask.get(0,input,i,j) * mask.getWeight(0) +
					  mask.get(1,input,i,j) * mask.getWeight(1) +
					  mask.get(2,input,i,j) * mask.getWeight(2) +
					  mask.get(3,input,i,j) * mask.getWeight(3) +
					  mask.get(4,input,i,j) * mask.getWeight(4) +
					  mask.get(5,input,i,j) * mask.getWeight(5) +
					  mask.get(6,input,i,j) * mask.getWeight(6) +
					  mask.get(7,input,i,j) * mask.getWeight(7) +
					  mask.get(8,input,i,j) * mask.getWeight(8) +
					  mask.get(9,input,i,j) * mask.getWeight(9) +
					  mask.get(10,input,i,j) * mask.getWeight(10) +
					  mask.get(11,input,i,j) * mask.getWeight(11) +
					  mask.get(12,input,i,j) * mask.getWeight(12) +
					  mask.get(13,input,i,j) * mask.getWeight(13) +
					  mask.get(14,input,i,j) * mask.getWeight(14) +
					  mask.get(15,input,i,j) * mask.getWeight(15) +
					  mask.get(16,input,i,j) * mask.getWeight(16) +
					  mask.get(17,input,i,j) * mask.getWeight(17) +
					  mask.get(18,input,i,j) * mask.getWeight(18) +
					  mask.get(19,input,i,j) * mask.getWeight(19) +
					  mask.get(20,input,i,j) * mask.getWeight(20) +
					  mask.get(21,input,i,j) * mask.getWeight(21) +
					  mask.get(22,input,i,j) * mask.getWeight(22) +
					  mask.get(23,input,i,j) * mask.getWeight(23) +
					  mask.get(24,input,i,j) * mask.getWeight(24); 
	}
}//end namespace

//*******************************************************************************************
// MAIN
//*******************************************************************************************

int main(int argc, char **argv){	
	//wbImage_t inputImage;
	//wbImage_t outputImage;	
	
	Mask2D<float> mask(25);
	float GPUTime;
	int GPUBlockSize, numCPUThreads,x_max,y_max;
	
	if (argc != 8){
		printf ("Wrong number of parameters.\n");
		//printf ("Usage: convolution INPUT_IMAGE ITERATIONS GPUTIME GPUBLOCKS CPUTHREADS OUTPUT_WRITE_FLAG\n");
		printf ("Usage: convolution WIDTH HEIGHT ITERATIONS GPUTIME GPUBLOCKS CPUTHREADS OUTPUT_WRITE_FLAG\n");
		printf ("You entered: ");
		for(int i=0; i< argc;i++){
			printf("%s ",argv[i]);
		}
		printf("\n");
		exit (-1);
	}
	
	//inputImage = wbImport(argv[1]);
	//int x_max = wbImage_getWidth(inputImage);
	//int y_max = wbImage_getHeight(inputImage);
	x_max = atoi(argv[1]);
	y_max = atoi(argv[2]);
	int T_MAX = atoi(argv[3]);
	GPUTime = atof(argv[4]);
	GPUBlockSize = atoi(argv[5]);
	numCPUThreads = atoi(argv[6]);
	int writeToFile = atoi(argv[7]);
	
	Array2D<float> inputGrid(x_max, y_max);
	Array2D<float> outputGrid(x_max, y_max);	
	
	//outputImage = wbImage_new(wbImage_getWidth(inputImage), wbImage_getHeight(inputImage), wbImage_getChannels(inputImage));

	mask.set(0,-2,2,0.0);	mask.set(1,-1,2,0.0);	mask.set(2,0,2,0.0);	mask.set(3,1,2,0.0);	mask.set(4,2,2,0.0);
	mask.set(5,-2,1,0.0);	mask.set(6,-1,1,0.0);	mask.set(7,0,1,0.1);	mask.set(8,1,1,0.0);	mask.set(9,2,1,0.0);
	mask.set(10,-2,0,0.0);	mask.set(11,-1,0,0.1);	mask.set(12,0,0,0.2);	mask.set(13,1,0,0.1);	mask.set(14,2,0,0.0);
	mask.set(15,-2,-1,0.0);	mask.set(16,-1,-1,0.0);	mask.set(17,0,-1,0.1);	mask.set(18,1,-1,0.0);	mask.set(19,2,-1,0.0);
	mask.set(20,-2,-2,0.0);	mask.set(21,-1,-2,0.0);	mask.set(22,0,-2,0.0);	mask.set(23,1,-2,0.0);	mask.set(24,2,-2,0.0);
	
	#pragma omp parallel
	{
		srand(1234 ^ omp_get_thread_num());
		#pragma omp for
		for (int x = 0; x < x_max; x++){
			for (int y = 0; y < y_max; y++){		
				inputGrid(x,y) = ((float)(rand() % 255))/255;
			}
		}
	}
	
	Stencil2D<Array2D<float>, Mask2D<float>, int> stencil(inputGrid, outputGrid, mask, 0);
	hr_timer_t timer;
	
	/*
	if(GPUTime == 0.0){
		stencil.runIterativeCPU(T_MAX, numCPUThreads);
	}
	else if(GPUTime == 1.0){
		stencil.runIterativeGPU(T_MAX, GPUBlockSize);
	}
	else{
		stencil.runIterativePartition(T_MAX, GPUTime, numCPUThreads,GPUBlockSize);
	}
	*/
	
	#ifdef PSKEL_PAPI
		if(GPUTime < 1.0)
			PSkelPAPI::init(PSkelPAPI::CPU);
	#endif
	hrt_start(&timer);
	//stencil.runIterativePartition(T_MAX, 1.0-CPUTime, numCPUThreads, GPUBlockSize);
	//stencil.runIterativeAutoHybrid(T_MAX, CPUTime, numCPUThreads, GPUBlockSize);	
	
	//stencil.runSequential();
	
	if(GPUTime == 0.0){
		//jacobi.runIterativeCPU(T_MAX, numCPUThreads);
		#ifdef PSKEL_PAPI
			for(unsigned int i=0;i<NUM_GROUPS_CPU;i++){
				//cout << "Running iteration " << i << endl;
				stencil.runIterativeCPU(T_MAX, numCPUThreads, i);	
			}
		#else
			//cout<<"Running Iterative CPU"<<endl;
			stencil.runIterativeCPU(T_MAX, numCPUThreads);	
		#endif
	}
	else if(GPUTime == 1.0){
		stencil.runIterativeGPU(T_MAX, GPUBlockSize);
	}
	else{
		//jacobi.runIterativePartition(T_MAX, GPUTime, numCPUThreads,GPUBlockSize);
		#ifdef PSKEL_PAPI
			for(unsigned int i=0;i<NUM_GROUPS_CPU;i++){
				stencil.runIterativePartition(T_MAX, GPUTime, numCPUThreads,GPUBlockSize,i);
			}
		#else
			stencil.runIterativePartition(T_MAX, GPUTime, numCPUThreads,GPUBlockSize);
		#endif
	}
	
	hrt_stop(&timer);

	#ifdef PSKEL_PAPI
		if(GPUTime < 1.0){
			PSkelPAPI::print_profile_values(PSkelPAPI::CPU);
			PSkelPAPI::shutdown();
		}
	#endif
	
	cout << "Exec_time\t" << hrt_elapsed_time(&timer) << endl;
	
	//PRIMEIRA ITERACAO
	/*
	//cout<<"Canal R"<<endl;
	Stencil2D<Array2D<float>, Mask2D<float>, int > red(inputImage._red, outputImage._red, mask, 0);
	//Runtime<Stencil2D<Array2D<float>, Mask2D<float>, int > > stencilR(&red);
	//stencilR.runIterator(T_MAX,GPUTime, GPUBlockSize, numCPUThreads);
	red.runIterativeGPU(T_MAX,GPUBlockSize);
	
	//cout<<"Canal G"<<endl;
	Stencil2D<Array2D<float>, Mask2D<float>, int > green(inputImage._green, outputImage._green, mask, 0);
	//Runtime<Stencil2D<Array2D<float>, Mask2D<float>, int > > stencilG(&green);
	//stencilG.runIterator(T_MAX,GPUTime, GPUBlockSize, numCPUThreads);
	green.runIterativeGPU(T_MAX,GPUBlockSize);
	
	//cout<<"Canal B"<<endl;
	Stencil2D<Array2D<float>, Mask2D<float>, int > blue(inputImage._blue, outputImage._blue, mask, 0);
	//Runtime<Stencil2D<Array2D<float>, Mask2D<float>, int > > stencilB(&blue);
	//stencilB.runIterator(T_MAX,GPUTime, GPUBlockSize, numCPUThreads);
	blue.runIterativeGPU(T_MAX,GPUBlockSize);
	*/

	if(writeToFile == 1){
		/*
		stringstream inputFile;
		inputFile << "input_" <<x_max << "_" << y_max << "_" << T_MAX << "_" << GPUTime << "_" << GPUBlockSize <<"_" << numCPUThreads << ".pgm";
		string in = inputFile.str();
	
		//wbImage_save(outputImage, (char*) out.c_str());
		ofstream ifs(in.c_str(), std::ofstream::out);
		ifs<<"P2"<<endl;
		ifs<<x_max<<" "<<y_max<<endl;
		ifs<<"255"<<endl;
		for(int x=0;x<x_max;x++) {
			for(int y=0;y<y_max;y++){
				ifs<<ceil(inputGrid(x,y)*255.0)<<" ";
			}
			ifs<<endl;
		}
		
		stringstream outputFile;
		outputFile << "output_" <<x_max << "_" << y_max << "_" << T_MAX << "_" << GPUTime << "_" << GPUBlockSize <<"_" << numCPUThreads << ".pgm";
		string out = outputFile.str();
	
		//wbImage_save(outputImage, (char*) out.c_str());
		ofstream ofs(out.c_str(), std::ofstream::out);
		ofs<<"P2"<<endl;
		ofs<<x_max<<" "<<y_max<<endl;
		ofs<<"255"<<endl;
		for(int x=0;x<x_max;x++) {
			for(int y=0;y<y_max;y++){
				ofs<<ceil(outputGrid(x,y)*255.0)<<" ";
			}
			ofs<<endl;
		}
		*/
		cout.precision(12);
		cout<<"INPUT"<<endl;
		for(int i=10; i<y_max;i+=10){
			cout<<"("<<i<<","<<i<<") = "<<inputGrid(i,i)<<"\t\t("<<x_max-i<<","<<y_max-i<<") = "<<inputGrid(x_max-i,y_max-i)<<endl;
		}
		cout<<endl;
		
		cout<<"OUTPUT"<<endl;
		for(int i=10; i<y_max;i+=10){
			cout<<"("<<i<<","<<i<<") = "<<outputGrid(i,i)<<"\t\t("<<x_max-i<<","<<y_max-i<<") = "<<outputGrid(x_max-i,y_max-i)<<endl;
		}
		cout<<endl;
	}
	return 0;
}


