#include <musa_fp16.h>
#include <musa_runtime.h>

#include <cstdint>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace musa {

// -------- Select indices of true values kernel --------

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
__global__ void MusaMarkFlaggedKernel(const T* __restrict__ d_flags,
                                      TIndex* d_marks, int num_items) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_items) {
    d_marks[idx] = IsNonZeroValue<T>(d_flags[idx]) ? 1 : 0;
  }
}

// Wrapper to launch Mark kernel separately since muDNN needs to be in .h/.cc
template <typename T, typename TIndex>
void LaunchMusaMarkFlaggedKernel(const T* input, TIndex* d_marks, int num_items,
                                 musaStream_t stream) {
  if (num_items <= 0) return;
  const int threads = 256;
  const int blocks = (num_items + threads - 1) / threads;
  MusaMarkFlaggedKernel<T, TIndex>
      <<<blocks, threads, 0, stream>>>(input, d_marks, num_items);
}

#define INSTANTIATE_SELECT_FLAGGED(T, TINDEX)                            \
  template void LaunchMusaMarkFlaggedKernel<T, TINDEX>(                  \
      const T* input, TINDEX* d_marks, int num_items, musaStream_t stream)

#define INSTANTIATE_SELECT_FLAGGED_ALL(T) \
  INSTANTIATE_SELECT_FLAGGED(T, int32_t); \
  INSTANTIATE_SELECT_FLAGGED(T, int64_t)

INSTANTIATE_SELECT_FLAGGED_ALL(bool);
INSTANTIATE_SELECT_FLAGGED_ALL(float);
INSTANTIATE_SELECT_FLAGGED_ALL(double);
INSTANTIATE_SELECT_FLAGGED_ALL(int8);
INSTANTIATE_SELECT_FLAGGED_ALL(uint8);
INSTANTIATE_SELECT_FLAGGED_ALL(int16);
INSTANTIATE_SELECT_FLAGGED_ALL(uint16);
INSTANTIATE_SELECT_FLAGGED_ALL(int32);
INSTANTIATE_SELECT_FLAGGED_ALL(int64);
INSTANTIATE_SELECT_FLAGGED_ALL(bfloat16);
#undef INSTANTIATE_SELECT_FLAGGED
#undef INSTANTIATE_SELECT_FLAGGED_ALL

// -------- Propagate selected indices into NDIM output kernel --------

template <int NDIM, typename TIndex>
struct StridesPack {
  TIndex v[NDIM];
};

template <int NDIM, typename TIndex>
__global__ void ScatterAndPropagateWhereIndicesKernel(
    const TIndex* __restrict__ d_marks, const TIndex* __restrict__ d_scanned,
    TIndex* __restrict__ output, const StridesPack<NDIM, TIndex> strides,
    int num_items) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_items && d_marks[idx] == 1) {
    // d_scanned is inclusive sum from muDNN CumSum, so (sum - 1) is the
    // zero-based index
    TIndex pos = d_scanned[idx] - 1;
    TIndex index_value = static_cast<TIndex>(idx);
#pragma unroll
    for (int c = 0; c < NDIM; ++c) {
      const TIndex stride = strides.v[c];
      *(output + NDIM * pos + c) = index_value / stride;
      index_value %= stride;
    }
  }
}

template <int NDIM, typename TIndex>
void LaunchScatterAndPropagateWhereIndicesKernel(
    const TIndex* d_marks, const TIndex* d_scanned, TIndex* output,
    const TIndex* strides_host, int num_items, musaStream_t stream) {
  if (num_items <= 0) {
    return;
  }

  StridesPack<NDIM, TIndex> pack;
#pragma unroll
  for (int i = 0; i < NDIM; ++i) {
    pack.v[i] = strides_host[i];
  }

  const int threads = 256;
  const int blocks = (num_items + threads - 1) / threads;
  ScatterAndPropagateWhereIndicesKernel<NDIM, TIndex>
      <<<blocks, threads, 0, stream>>>(d_marks, d_scanned, output, pack,
                                       num_items);
}

#define INSTANTIATE_PROPAGATE(NDIM, TINDEX)                            \
  template void LaunchScatterAndPropagateWhereIndicesKernel<NDIM, TINDEX>( \
      const TINDEX* d_marks, const TINDEX* d_scanned, TINDEX* output,      \
      const TINDEX* strides_host, int num_items, musaStream_t stream)

INSTANTIATE_PROPAGATE(1, int32);
INSTANTIATE_PROPAGATE(2, int32);
INSTANTIATE_PROPAGATE(3, int32);
INSTANTIATE_PROPAGATE(4, int32);
INSTANTIATE_PROPAGATE(5, int32);
INSTANTIATE_PROPAGATE(6, int32);
INSTANTIATE_PROPAGATE(7, int32);
INSTANTIATE_PROPAGATE(8, int32);

INSTANTIATE_PROPAGATE(1, int64);
INSTANTIATE_PROPAGATE(2, int64);
INSTANTIATE_PROPAGATE(3, int64);
INSTANTIATE_PROPAGATE(4, int64);
INSTANTIATE_PROPAGATE(5, int64);
INSTANTIATE_PROPAGATE(6, int64);
INSTANTIATE_PROPAGATE(7, int64);
INSTANTIATE_PROPAGATE(8, int64);

#undef INSTANTIATE_PROPAGATE

}  // namespace musa
}  // namespace tensorflow