#include <musa_fp16.h>
#include <musa_runtime.h>

#include <cstdint>

#include <cub/cub.cuh>

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
struct NonZeroFlagOp {
  __device__ __forceinline__ bool operator()(const T& v) const {
    return IsNonZeroValue<T>(v);
  }
};

template <typename T, typename TIndex>
__global__ void SelectTrueIndicesFallbackKernel(const T* __restrict__ input,
                                                TIndex* __restrict__ out,
                                                TIndex* __restrict__ counter,
                                                int num_items) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_items && IsNonZeroValue<T>(input[idx])) {
    TIndex pos = AtomicAddOne<TIndex>(counter);
    out[pos] = static_cast<TIndex>(idx);
  }
}

template <typename T, typename TIndex>
size_t GetSelectTrueIndicesWorkspaceSize(const T* input, TIndex* output_indices,
                                         TIndex* num_selected, int num_items,
                                         musaStream_t stream) {
  if (num_items <= 0) return 0;

  cub::CountingInputIterator<TIndex> counting_iter(0);
  cub::TransformInputIterator<bool, NonZeroFlagOp<T, TIndex>, const T*>
      flag_iter(input, NonZeroFlagOp<T, TIndex>());

  size_t temp_bytes = 0;
  musaError_t st = cub::DeviceSelect::Flagged(
      nullptr, temp_bytes, counting_iter, flag_iter, output_indices,
      num_selected, num_items, stream);
  return st == musaSuccess ? temp_bytes : 0;
}

template <typename T, typename TIndex>
void LaunchSelectTrueIndicesKernel(const T* input, TIndex* output_indices,
                                   TIndex* num_selected, int num_items,
                                   void* temp_storage,
                                   size_t temp_storage_bytes,
                                   musaStream_t stream) {
  if (num_items <= 0) return;

  cub::CountingInputIterator<TIndex> counting_iter(0);
  cub::TransformInputIterator<bool, NonZeroFlagOp<T, TIndex>, const T*>
      flag_iter(input, NonZeroFlagOp<T, TIndex>());

  if (temp_storage != nullptr && temp_storage_bytes > 0) {
    musaError_t st = cub::DeviceSelect::Flagged(
        temp_storage, temp_storage_bytes, counting_iter, flag_iter,
        output_indices, num_selected, num_items, stream);
    if (st == musaSuccess) {
      return;
    }
  }

  // Counter is only required for the fallback path.
  musaMemsetAsync(num_selected, 0, sizeof(TIndex), stream);

  const int threads = 256;
  const int blocks = (num_items + threads - 1) / threads;
  SelectTrueIndicesFallbackKernel<T, TIndex>
      <<<blocks, threads, 0, stream>>>(input, output_indices, num_selected,
                                       num_items);
}

#define INSTANTIATE_SELECT_FLAGGED(T, TINDEX)                            \
  template size_t GetSelectTrueIndicesWorkspaceSize<T, TINDEX>(           \
      const T* input, TINDEX* output_indices, TINDEX* num_selected,       \
      int num_items, musaStream_t stream);                                \
                                                                          \
  template void LaunchSelectTrueIndicesKernel<T, TINDEX>(                \
      const T* input, TINDEX* output_indices, TINDEX* num_selected,      \
      int num_items, void* temp_storage, size_t temp_storage_bytes,      \
      musaStream_t stream)

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
__global__ void PropagateWhereIndicesFromFlatKernel(
    const TIndex* __restrict__ flat_indices, TIndex* __restrict__ output,
    const StridesPack<NDIM, TIndex> strides, int num_selected) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < num_selected) {
    TIndex index_value = flat_indices[pos];
#pragma unroll
    for (int c = 0; c < NDIM; ++c) {
      const TIndex stride = strides.v[c];
      *(output + NDIM * pos + c) = index_value / stride;
      index_value %= stride;
    }
  }
}

template <int NDIM, typename TIndex>
void LaunchPropagateWhereIndicesFromFlatKernel(const TIndex* flat_indices,
                                               TIndex* output,
                                               const TIndex* strides_host,
                                               int num_selected,
                                               musaStream_t stream) {
  if (num_selected <= 0) {
    return;
  }

  StridesPack<NDIM, TIndex> pack;
#pragma unroll
  for (int i = 0; i < NDIM; ++i) {
    pack.v[i] = strides_host[i];
  }

  const int threads = 256;
  const int blocks = (num_selected + threads - 1) / threads;
  PropagateWhereIndicesFromFlatKernel<NDIM, TIndex>
      <<<blocks, threads, 0, stream>>>(flat_indices, output, pack,
                                       num_selected);
}

#define INSTANTIATE_PROPAGATE(NDIM, TINDEX)                            \
  template void LaunchPropagateWhereIndicesFromFlatKernel<NDIM, TINDEX>(    \
      const TINDEX* flat_indices, TINDEX* output, const TINDEX* strides_host, \
      int num_selected, musaStream_t stream)

#define INSTANTIATE_PROPAGATE_ALL(NDIM) \
  INSTANTIATE_PROPAGATE(NDIM, int32); \
  INSTANTIATE_PROPAGATE(NDIM, int64)

INSTANTIATE_PROPAGATE_ALL(1);
INSTANTIATE_PROPAGATE_ALL(2);
INSTANTIATE_PROPAGATE_ALL(3);
INSTANTIATE_PROPAGATE_ALL(4);
INSTANTIATE_PROPAGATE_ALL(5);
INSTANTIATE_PROPAGATE_ALL(6);
INSTANTIATE_PROPAGATE_ALL(7);
INSTANTIATE_PROPAGATE_ALL(8);

#undef INSTANTIATE_PROPAGATE

}  // namespace musa
}  // namespace tensorflow