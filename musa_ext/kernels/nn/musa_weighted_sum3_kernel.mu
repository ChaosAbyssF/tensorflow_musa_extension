#include <math.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <stdint.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"

#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

__device__ __forceinline__ float LoadFloat(const float* p) { return *p; }
__device__ __forceinline__ void StoreFloat(float* p, float v) { *p = v; }

__device__ __forceinline__ float LoadFloat(const Eigen::half* p) {
  const __half* h_ptr = reinterpret_cast<const __half*>(p);
  return __half2float(*h_ptr);
}

__device__ __forceinline__ void StoreFloat(Eigen::half* p, float v) {
  __half h = __float2half(v);
  *reinterpret_cast<__half*>(p) = h;
}

__device__ __forceinline__ float LoadFloat(const bfloat16* p) {
  float res = 0.0f;
  uint16_t* b_ptr = (uint16_t*)p;
  uint32_t* f_ptr = (uint32_t*)&res;
  *f_ptr = (static_cast<uint32_t>(*b_ptr)) << 16;
  return res;
}

__device__ __forceinline__ void StoreFloat(bfloat16* p, float v) {
  uint32_t* f_ptr = (uint32_t*)&v;
  uint16_t b_val = (*f_ptr) >> 16;
  *reinterpret_cast<uint16_t*>(p) = b_val;
}

template <typename T>
__global__ void WeighedSum3Kernel(const T* a, const T* b, const T* c,
                                  const T* alpha, const T* beta, const T* gamma,
                                  T* output, int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float a_val = LoadFloat(&a[idx]);
    float b_val = LoadFloat(&b[idx]);
    float c_val = LoadFloat(&c[idx]);
    float alpha_val = LoadFloat(alpha);
    float beta_val = LoadFloat(beta);
    float gamma_val = LoadFloat(gamma);
    StoreFloat(&output[idx],
               alpha_val * a_val + beta_val * b_val + gamma_val * c_val);
  }
}

template <>
__global__ void WeighedSum3Kernel<double>(const double* a, const double* b,
                                          const double* c, const double* alpha,
                                          const double* beta,
                                          const double* gamma, double* output,
                                          int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = (*alpha) * a[idx] + (*beta) * b[idx] + (*gamma) * c[idx];
  }
}

template <typename T>
void LaunchWeightedSum3Kernel(const T* a, const T* b, const T* c,
                              const T* alpha, const T* beta, const T* gamma,
                              T* output, int num_elements,
                              musaStream_t stream) {
  const int block_size = 256;
  const int grid_size = (num_elements + block_size - 1) / block_size;
  WeighedSum3Kernel<<<grid_size, block_size, 0, stream>>>(
      a, b, c, alpha, beta, gamma, output, num_elements);
}

#define REGISTER_WEIGHTED_SUM3_KERNEL(TYPE)                                \
  template void LaunchWeightedSum3Kernel<TYPE>(                            \
      const TYPE* a, const TYPE* b, const TYPE* c, const TYPE* alpha,      \
      const TYPE* beta, const TYPE* gamma, TYPE* output, int num_elements, \
      musaStream_t stream);

REGISTER_WEIGHTED_SUM3_KERNEL(float);
REGISTER_WEIGHTED_SUM3_KERNEL(double);
REGISTER_WEIGHTED_SUM3_KERNEL(Eigen::half);
REGISTER_WEIGHTED_SUM3_KERNEL(bfloat16);

}  // namespace musa
}  // namespace tensorflow
