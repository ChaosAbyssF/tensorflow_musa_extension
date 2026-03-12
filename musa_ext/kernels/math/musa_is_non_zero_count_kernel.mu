#include <math.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <cub/cub.cuh>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

__device__ __forceinline__ float LoadFloat(const float* p) { return *p; }

__device__ __forceinline__ float LoadFloat(const Eigen::half* p) {
  const __half* h_ptr = reinterpret_cast<const __half*>(p);
  return __half2float(*h_ptr);
}

__device__ __forceinline__ float LoadFloat(const bfloat16* p) {
  float res = 0.0f;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(p);
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&res);
  *f_ptr = (static_cast<uint32_t>(*b_ptr)) << 16;
  return res;
}

template <typename T>
__device__ __forceinline__ bool IsNonZeroValue(const T& v) {
  return v != static_cast<T>(0);
}

template <typename TIndex>
__device__ __forceinline__ TIndex AtomicAddOne(TIndex* addr);

template <>
__device__ __forceinline__ int32 AtomicAddOne<int32>(int32* addr) {
  return atomicAdd(addr, 1);
}

template <>
__device__ __forceinline__ int64 AtomicAddOne<int64>(int64* addr) {
  return static_cast<int64>(
      atomicAdd(reinterpret_cast<unsigned long long*>(addr), 1ULL));
}

template <>
__device__ __forceinline__ bool IsNonZeroValue<float>(const float& v) {
  return v != 0.0f;
}

template <>
__device__ __forceinline__ bool IsNonZeroValue<double>(const double& v) {
  return v != 0.0;
}

template <>
__device__ __forceinline__ bool IsNonZeroValue<Eigen::half>(
    const Eigen::half& v) {
  float fv = LoadFloat(&v);
  return fv != 0.0f;
}

template <>
__device__ __forceinline__ bool IsNonZeroValue<bfloat16>(const bfloat16& v) {
  float fv = LoadFloat(&v);
  return fv != 0.0f;
}

template <typename T, typename TIndex>
struct NonZeroToCountOp {
  __device__ __forceinline__ TIndex operator()(const T& v) const {
    return IsNonZeroValue<T>(v) ? static_cast<TIndex>(1) : static_cast<TIndex>(0);
  }
};

template <typename T, typename TIndex>
__global__ void IsNonZeroCountKernelFallback(const T* input, TIndex* output,
                                             int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && IsNonZeroValue<T>(input[idx])) {
    AtomicAddOne<TIndex>(output);
  }
}

template <typename T, typename TIndex>
size_t GetIsNonZeroCountWorkspaceSize(const T* input, TIndex* output, int n,
                                      musaStream_t stream) {
  if (n <= 0) return 0;

  cub::TransformInputIterator<TIndex, NonZeroToCountOp<T, TIndex>, const T*>
      transformed_input(input, NonZeroToCountOp<T, TIndex>());

  size_t temp_bytes = 0;
  musaError_t st = cub::DeviceReduce::Sum(nullptr, temp_bytes,
                                          transformed_input, output, n,
                                          stream);
  return st == musaSuccess ? temp_bytes : 0;
}

template <typename T, typename TIndex>
void LaunchIsNonZeroCount(const T* input, TIndex* output, int n,
                          void* temp_storage, size_t temp_storage_bytes,
                          musaStream_t stream) {
  musaMemsetAsync(output, 0, sizeof(TIndex), stream);
  if (n <= 0) return;

  cub::TransformInputIterator<TIndex, NonZeroToCountOp<T, TIndex>, const T*>
      transformed_input(input, NonZeroToCountOp<T, TIndex>());

  if (temp_storage != nullptr && temp_storage_bytes > 0) {
    musaError_t st = cub::DeviceReduce::Sum(
        temp_storage, temp_storage_bytes, transformed_input, output, n,
        stream);
    if (st == musaSuccess) {
      return;
    }
  }

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  IsNonZeroCountKernelFallback<T, TIndex>
      <<<blocks, threads, 0, stream>>>(input, output, n);
}

#define REGISTER_IS_NON_ZERO_COUNT(T, TIndex)    \
  template size_t GetIsNonZeroCountWorkspaceSize<T, TIndex>( \
      const T* input, TIndex* output, int n, musaStream_t stream); \
  \
  template void LaunchIsNonZeroCount<T, TIndex>( \
      const T* input, TIndex* output, int n, void* temp_storage, \
      size_t temp_storage_bytes, musaStream_t stream)

REGISTER_IS_NON_ZERO_COUNT(float, int32);
REGISTER_IS_NON_ZERO_COUNT(double, int32);
REGISTER_IS_NON_ZERO_COUNT(Eigen::half, int32);
REGISTER_IS_NON_ZERO_COUNT(bfloat16, int32);
REGISTER_IS_NON_ZERO_COUNT(int32, int32);
REGISTER_IS_NON_ZERO_COUNT(int64, int32);
REGISTER_IS_NON_ZERO_COUNT(bool, int32);
REGISTER_IS_NON_ZERO_COUNT(int8, int32);
REGISTER_IS_NON_ZERO_COUNT(int16, int32);
REGISTER_IS_NON_ZERO_COUNT(uint8, int32);
REGISTER_IS_NON_ZERO_COUNT(uint16, int32);

REGISTER_IS_NON_ZERO_COUNT(float, int64);
REGISTER_IS_NON_ZERO_COUNT(double, int64);
REGISTER_IS_NON_ZERO_COUNT(Eigen::half, int64);
REGISTER_IS_NON_ZERO_COUNT(bfloat16, int64);
REGISTER_IS_NON_ZERO_COUNT(int32, int64);
REGISTER_IS_NON_ZERO_COUNT(int64, int64);
REGISTER_IS_NON_ZERO_COUNT(bool, int64);
REGISTER_IS_NON_ZERO_COUNT(int8, int64);
REGISTER_IS_NON_ZERO_COUNT(int16, int64);
REGISTER_IS_NON_ZERO_COUNT(uint8, int64);
REGISTER_IS_NON_ZERO_COUNT(uint16, int64);
}  // namespace musa
}  // namespace tensorflow