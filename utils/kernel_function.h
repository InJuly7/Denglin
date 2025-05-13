#pragma once
#include "common_include.h"
#include "utils.h"

#define CHECK(x)  cuda_check(x, __FILE__, __LINE__)
#define BLOCK_SIZE 16

void cuda_check(cudaError_t state, std::string file, int line);

void cudaWarpAffine(unsigned char* src, const int src_cols, const int src_rows, float* dst, const int dst_cols, const int dst_rows, 
						const utils::AffineMat matrix, const float3 paddingValue, const float3 alpha, const float3 beta);
