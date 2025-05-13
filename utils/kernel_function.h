#pragma once
#include "common_include.h"
#include "utils.h"

#define CHECK(x)  cuda_check(x, __FILE__, __LINE__)
#define BLOCK_SIZE 256

void cuda_check(cudaError_t state, std::string file, int line);

void resizeDevice(unsigned char* src, int src_width, int src_height, float* dst, int dstWidth, int dstHeight, float paddingValue, 
                    utils::AffineMat matrix);
                    
void bgr2rgbDevice(float* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight);

void normDevice(float* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight, utils::InitParameter norm_param);

void hwc2chwDevice(float* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight);

