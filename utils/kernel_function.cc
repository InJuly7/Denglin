#include "kernel_function.h"
#include<math.h>


void cuda_check(cudaError_t state, std::string file, int line)
{
    if(cudaSuccess != state) {
        std::cout << "CUDA Error code num is:" << state;
        std::cout << "CUDA Error:" << cudaGetErrorString(state);
        std::cout << "Error location:" << file << ": " << line;
        abort();
    }
}


inline __device__ __host__ int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ float3 uchar3_to_float3(uchar3 v) {
    return make_float3(v.x, v.y, v.z);
}

__device__ bool in_bounds(int x, int y, int cols, int rows) {
    return (x >= 0 && x < cols && y >= 0 && y < rows);
}

__device__ float3 operator*(float value0, float3 value1) {
    float3 result;
    result.x = value0 * value1.x;
    result.y = value0 * value1.y;
    result.z = value0 * value1.z;

    return result;
}

__device__ float3 operator+(float value0, float3 value1) {
    float3 result;
    result.x = value0 + value1.x;
    result.y = value0 + value1.y;
    result.z = value0 + value1.z;

    return result;
}

__device__ void operator+=(float3& result, float3& value) {
    result.x += value.x;
    result.y += value.y;
    result.z += value.z;
}

__device__ float3 operator+=(float3& value0, const float3& value1) {
    value0.x += value1.x;
    value0.y += value1.y;
    value0.z += value1.z;
    return value0;
}

__device__ void affine_project_device_kernel(const utils::AffineMat* matrix, int x, int y, float* proj_x, float* proj_y)
{
	*proj_x = matrix->v0 * x + matrix->v1 * y + matrix->v2;
	*proj_y = matrix->v3 * x + matrix->v4 * y + matrix->v5;
}

__device__ void warp_affine_bilinear(unsigned char* src, const int src_cols, const int src_rows, float* dst, const int dst_cols, 
										const int dst_rows, const utils::AffineMat matrix, const float3 paddingValue, const float3 alpha, 
										const float3 beta, int element_x, int element_y) {
    if (element_x >= dst_cols || element_y >= dst_rows) { return; }
	float2 src_xy = make_float2(0.0f, 0.0f);
	affine_project_device_kernel(&matrix, element_x, element_y, &src_xy.x, &src_xy.y);

    int src_x0 = __float2int_rd(src_xy.x);
    int src_y0 = __float2int_rd(src_xy.y);
    int src_x1 = src_x0 + 1;
    int src_y1 = src_y0 + 1;

    float wx0 = src_x1 - src_xy.x; // hx
    float wx1 = src_xy.x - src_x0; // lx
    float wy0 = src_y1 - src_xy.y; // hy
    float wy1 = src_xy.y - src_y0; // ly

    float3 src_value0, src_value1, value0;
    bool   flag0 = in_bounds(src_x0, src_y0, src_cols, src_rows);
    bool   flag1 = in_bounds(src_x1, src_y0, src_cols, src_rows);
    bool   flag2 = in_bounds(src_x0, src_y1, src_cols, src_rows);
    bool   flag3 = in_bounds(src_x1, src_y1, src_cols, src_rows);


    uchar3* input = (uchar3*)(src + src_y0 * src_cols * 3);
    src_value0 = flag0 ? uchar3_to_float3(input[src_x0]) : paddingValue;
    src_value1 = flag1 ? uchar3_to_float3(input[src_x1]) : paddingValue;
    value0 = wx0 * wy0 * src_value0; // hx * hy = w1
    value0 += wx1 * wy0 * src_value1; // lx * hy = w2
    

    input = (uchar3*)(src + src_y1 * src_cols * 3);
    src_value0 = flag2 ? uchar3_to_float3(input[src_x0]) : paddingValue;
    src_value1 = flag3 ? uchar3_to_float3(input[src_x1]) : paddingValue;
    value0 += wx0 * wy1 * src_value0; // hx * ly = w3
    value0 += wx1 * wy1 * src_value1; // lx * ly = w4
    value0 = 0.5f + value0;
    float3 sum;
    sum.x = __float2int_rd(value0.x);
    sum.y = __float2int_rd(value0.y);
    sum.z = __float2int_rd(value0.z);
    
    float temp = sum.x;
    sum.x      = sum.z;
    sum.z      = temp;
    

    float* output                   = (float*)dst + element_y * dst_cols + element_x;
    output[0]                       = sum.x * alpha.x + beta.x;
    output[dst_cols * dst_rows]     = sum.y * alpha.y + beta.y;
    output[2 * dst_cols * dst_rows] = sum.z * alpha.z + beta.z;
}

__global__ void gpuBilinearWarpAffine(unsigned char* src, const int src_cols, const int src_rows, float* dst, const int dst_cols, 
										const int dst_rows, const utils::AffineMat matrix, const float3 paddingValue, const float3 alpha, 
										const float3 beta) {
    int element_x = blockDim.x * blockIdx.x + threadIdx.x;
    int element_y = blockDim.y * blockIdx.y + threadIdx.y;

    warp_affine_bilinear(src, src_cols, src_rows, dst, dst_cols, dst_rows, matrix, paddingValue, alpha, beta, element_x, element_y);
}

void cudaWarpAffine(unsigned char* src, const int src_cols, const int src_rows, float* dst, const int dst_cols, const int dst_rows, 
						const utils::AffineMat matrix, const float3 paddingValue, const float3 alpha, const float3 beta) {
    // launch kernel
    const dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 gridDim(iDivUp(dst_cols, blockDim.x), iDivUp(dst_rows, blockDim.y));
    gpuBilinearWarpAffine<<<gridDim, blockDim, 0, nullptr>>>(src, src_cols, src_rows, dst, dst_cols, dst_rows, matrix, paddingValue, alpha, 
																beta);
}
