#pragma once
#include "common_include.h"
#include "utils.h"

#define CHECK(x)  cuda_check(x, __FILE__, __LINE__)
#define BLOCK_SIZE 16

void cuda_check(cudaError_t state, std::string file, int line);

//note: resize rgb with padding
// void resizeDevice(float* src, int src_width, int src_height, float* dst, int dstWidth, int dstHeight, float paddingValue, 
//                     utils::AffineMat matrix);

//overload:resize rgb with padding, but src's type is uin8
void resizeDevice(unsigned char* src, int src_width, int src_height, float* dst, int dstWidth, int dstHeight, float paddingValue, 
                    utils::AffineMat matrix);

// overload: resize rgb/gray without padding
// void resizeDevice(float* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight, utils::ColorMode mode, utils::AffineMat matrix);

void bgr2rgbDevice(float* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight);

void normDevice(float* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight, utils::InitParameter norm_param);

void hwc2chwDevice(float* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight);

void cudaWarpAffine(unsigned char* src, const int src_cols, const int src_rows, float* dst, const int dst_cols, const int dst_rows, 
						const utils::AffineMat matrix, const float3 paddingValue, const float3 alpha, const float3 beta);
// void decodeDevice(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcLength, float* dst, int dstWidth, int dstHeight);

// nms fast
// void nmsDeviceV1(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea);

// nms sort
// void nmsDeviceV2(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea, int* idx, float* conf);

// void copyWithPaddingDevice(const int& batchSize, float* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight, float paddingValue, int padTop, int padLeft);