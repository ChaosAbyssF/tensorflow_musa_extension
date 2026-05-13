#include <musa_runtime.h>
#include <musa_bf16.h>
#include <musa_fp16.h>

#include <stdint.h>
#include <string.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

namespace {

constexpr int kThreadsPerBlock = 256;

static inline int64_t CeilDiv(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

static inline bool IsAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

static inline bool IsAligned4(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0x3) == 0;
}

// Add a 32-bit register holding two packed bf16 values to another such
// register and return the packed sum. Promotion to fp32 + RNE round back to
// bf16 is mathematically what muDNN would do; doing it inline here lets us
// keep the entire pipeline in registers between the 16-byte ::uint4 load and
// store, which is bandwidth-optimal for the most common residual-add shape.
__device__ __forceinline__ uint32_t add_bf16_pair_packed(uint32_t a,
                                                          uint32_t b) {
  __mt_bfloat162 ap, bp;
  memcpy(&ap, &a, sizeof(ap));
  memcpy(&bp, &b, sizeof(bp));
  const float lo = __low2float(ap) + __low2float(bp);
  const float hi = __high2float(ap) + __high2float(bp);
  const __mt_bfloat162 sum = __floats2bfloat162_rn(lo, hi);
  uint32_t result;
  memcpy(&result, &sum, sizeof(result));
  return result;
}

// Same packed-pair helper for fp16. __hadd2 is hardware-vectorized add on
// __half2 and is bit-for-bit identical to two separate __half adds. We use
// it directly here (rather than via fp32 promotion) because for fp16 the
// hardware operation is a single-issue instruction; falling back through
// fp32 would lose that optimization.
__device__ __forceinline__ uint32_t add_half_pair_packed(uint32_t a,
                                                          uint32_t b) {
  __half2 ap, bp;
  memcpy(&ap, &a, sizeof(ap));
  memcpy(&bp, &b, sizeof(bp));
  const __half2 sum = __hadd2(ap, bp);
  uint32_t result;
  memcpy(&result, &sum, sizeof(result));
  return result;
}

}  // namespace

extern "C" {

__global__ void AddContiguousKernelFloat(const float* lhs, const float* rhs,
                                         float* output, int64_t size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = lhs[idx] + rhs[idx];
  }
}

__global__ void AddContiguousKernelFloat4(const float4* lhs, const float4* rhs,
                                          float4* output, int64_t vec_size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < vec_size) {
    const float4 l = lhs[idx];
    const float4 r = rhs[idx];
    float4 out;
    out.x = l.x + r.x;
    out.y = l.y + r.y;
    out.z = l.z + r.z;
    out.w = l.w + r.w;
    output[idx] = out;
  }
}

__global__ void AddScalarKernelFloat(const float* dense, const float* scalar,
                                     float* output, int64_t size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = dense[idx] + scalar[0];
  }
}

__global__ void AddScalarKernelFloat4(const float4* dense, const float* scalar,
                                      float4* output, int64_t vec_size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < vec_size) {
    const float scalar_value = scalar[0];
    const float4 d = dense[idx];
    float4 out;
    out.x = d.x + scalar_value;
    out.y = d.y + scalar_value;
    out.z = d.z + scalar_value;
    out.w = d.w + scalar_value;
    output[idx] = out;
  }
}

__global__ void AddTailVectorKernelFloat(const float* dense,
                                         const float* tail_vector,
                                         float* output, int64_t size,
                                         int64_t width) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const int64_t col = idx % width;
    output[idx] = dense[idx] + tail_vector[col];
  }
}

void LaunchMusaAddContiguousFloat(const float* lhs, const float* rhs,
                                  float* output, int64_t size,
                                  musaStream_t stream) {
  if (size <= 0) {
    return;
  }

  if (size >= 4 && IsAligned16(lhs) && IsAligned16(rhs) && IsAligned16(output)) {
    const int64_t vec_size = size / 4;
    const int64_t vec_blocks = CeilDiv(vec_size, kThreadsPerBlock);
    AddContiguousKernelFloat4<<<vec_blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float4*>(lhs),
        reinterpret_cast<const float4*>(rhs),
        reinterpret_cast<float4*>(output), vec_size);

    const int64_t tail = size - vec_size * 4;
    if (tail > 0) {
      const int64_t tail_blocks = CeilDiv(tail, kThreadsPerBlock);
      AddContiguousKernelFloat<<<tail_blocks, kThreadsPerBlock, 0, stream>>>(
          lhs + vec_size * 4, rhs + vec_size * 4, output + vec_size * 4, tail);
    }
    return;
  }

  const int64_t blocks = CeilDiv(size, kThreadsPerBlock);
  AddContiguousKernelFloat<<<blocks, kThreadsPerBlock, 0, stream>>>(
      lhs, rhs, output, size);
}

void LaunchMusaAddScalarFloat(const float* dense, const float* scalar,
                              float* output, int64_t size,
                              musaStream_t stream) {
  if (size <= 0) {
    return;
  }

  if (size >= 4 && IsAligned16(dense) && IsAligned16(output)) {
    const int64_t vec_size = size / 4;
    const int64_t vec_blocks = CeilDiv(vec_size, kThreadsPerBlock);
    AddScalarKernelFloat4<<<vec_blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float4*>(dense), scalar,
        reinterpret_cast<float4*>(output), vec_size);

    const int64_t tail = size - vec_size * 4;
    if (tail > 0) {
      const int64_t tail_blocks = CeilDiv(tail, kThreadsPerBlock);
      AddScalarKernelFloat<<<tail_blocks, kThreadsPerBlock, 0, stream>>>(
          dense + vec_size * 4, scalar, output + vec_size * 4, tail);
    }
    return;
  }

  const int64_t blocks = CeilDiv(size, kThreadsPerBlock);
  AddScalarKernelFloat<<<blocks, kThreadsPerBlock, 0, stream>>>(
      dense, scalar, output, size);
}

void LaunchMusaAddTailVectorFloat(const float* dense, const float* tail_vector,
                                  float* output, int64_t size, int64_t width,
                                  musaStream_t stream) {
  if (size <= 0 || width <= 0 || size % width != 0) {
    return;
  }

  const int64_t blocks = CeilDiv(size, kThreadsPerBlock);
  AddTailVectorKernelFloat<<<blocks, kThreadsPerBlock, 0, stream>>>(
      dense, tail_vector, output, size, width);
}

// ===========================================================================
// bf16 / fp16 fast paths.
//
// AddV2 is one of the hottest ops in transformer / mixer training (every
// residual connection issues one). The default code path falls through to
// muDNN Binary which has descriptor-setup overhead and does not vectorize
// for bf16. These plain-C launchers mirror the fp32 path's structure and
// dispatch a ::uint4-packed vec8 kernel when both inputs are 16-byte aligned
// (the case TF allocators produce by default), giving 8x the per-thread
// throughput of the scalar fallback.
// ===========================================================================

__global__ void AddContiguousKernelBFloat16(const bfloat16* __restrict__ lhs,
                                             const bfloat16* __restrict__ rhs,
                                             bfloat16* __restrict__ output,
                                             int64_t size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const __mt_bfloat16 l =
        *reinterpret_cast<const __mt_bfloat16*>(&lhs[idx]);
    const __mt_bfloat16 r =
        *reinterpret_cast<const __mt_bfloat16*>(&rhs[idx]);
    const float a = __bfloat162float(l);
    const float b = __bfloat162float(r);
    const __mt_bfloat16 sum = __float2bfloat16(a + b);
    *reinterpret_cast<__mt_bfloat16*>(&output[idx]) = sum;
  }
}

// 8 bf16 per thread (16-byte ::uint4 load), 4 packed-pair adds per thread.
__global__ void AddContiguousKernelBFloat16Vec8(const ::uint4* __restrict__ lhs,
                                                 const ::uint4* __restrict__ rhs,
                                                 ::uint4* __restrict__ output,
                                                 int64_t vec_size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < vec_size) {
    const ::uint4 l = lhs[idx];
    const ::uint4 r = rhs[idx];
    ::uint4 out;
    out.x = add_bf16_pair_packed(l.x, r.x);
    out.y = add_bf16_pair_packed(l.y, r.y);
    out.z = add_bf16_pair_packed(l.z, r.z);
    out.w = add_bf16_pair_packed(l.w, r.w);
    output[idx] = out;
  }
}

__global__ void AddScalarKernelBFloat16(const bfloat16* __restrict__ dense,
                                         const bfloat16* __restrict__ scalar,
                                         bfloat16* __restrict__ output,
                                         int64_t size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const __mt_bfloat16 s =
        *reinterpret_cast<const __mt_bfloat16*>(&scalar[0]);
    const float sf = __bfloat162float(s);
    const __mt_bfloat16 d =
        *reinterpret_cast<const __mt_bfloat16*>(&dense[idx]);
    const float df = __bfloat162float(d);
    const __mt_bfloat16 sum = __float2bfloat16(df + sf);
    *reinterpret_cast<__mt_bfloat16*>(&output[idx]) = sum;
  }
}

__global__ void AddTailVectorKernelBFloat16(
    const bfloat16* __restrict__ dense, const bfloat16* __restrict__ tail_vector,
    bfloat16* __restrict__ output, int64_t size, int64_t width) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const int64_t col = idx % width;
    const __mt_bfloat16 d =
        *reinterpret_cast<const __mt_bfloat16*>(&dense[idx]);
    const __mt_bfloat16 t =
        *reinterpret_cast<const __mt_bfloat16*>(&tail_vector[col]);
    const float sum_f = __bfloat162float(d) + __bfloat162float(t);
    const __mt_bfloat16 sum = __float2bfloat16(sum_f);
    *reinterpret_cast<__mt_bfloat16*>(&output[idx]) = sum;
  }
}

void LaunchMusaAddContiguousBFloat16(const void* lhs, const void* rhs,
                                      void* output, int64_t size,
                                      musaStream_t stream) {
  if (size <= 0) return;
  if (size >= 8 && IsAligned16(lhs) && IsAligned16(rhs) && IsAligned16(output)) {
    const int64_t vec_size = size / 8;
    const int64_t vec_blocks = CeilDiv(vec_size, kThreadsPerBlock);
    AddContiguousKernelBFloat16Vec8<<<vec_blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const ::uint4*>(lhs),
        reinterpret_cast<const ::uint4*>(rhs),
        reinterpret_cast<::uint4*>(output), vec_size);

    const int64_t tail = size - vec_size * 8;
    if (tail > 0) {
      const auto* l = reinterpret_cast<const bfloat16*>(lhs) + vec_size * 8;
      const auto* r = reinterpret_cast<const bfloat16*>(rhs) + vec_size * 8;
      auto* o = reinterpret_cast<bfloat16*>(output) + vec_size * 8;
      const int64_t tail_blocks = CeilDiv(tail, kThreadsPerBlock);
      AddContiguousKernelBFloat16<<<tail_blocks, kThreadsPerBlock, 0, stream>>>(
          l, r, o, tail);
    }
    return;
  }
  const int64_t blocks = CeilDiv(size, kThreadsPerBlock);
  AddContiguousKernelBFloat16<<<blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const bfloat16*>(lhs),
      reinterpret_cast<const bfloat16*>(rhs),
      reinterpret_cast<bfloat16*>(output), size);
}

void LaunchMusaAddScalarBFloat16(const void* dense, const void* scalar,
                                  void* output, int64_t size,
                                  musaStream_t stream) {
  if (size <= 0) return;
  const int64_t blocks = CeilDiv(size, kThreadsPerBlock);
  AddScalarKernelBFloat16<<<blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const bfloat16*>(dense),
      reinterpret_cast<const bfloat16*>(scalar),
      reinterpret_cast<bfloat16*>(output), size);
}

void LaunchMusaAddTailVectorBFloat16(const void* dense, const void* tail_vector,
                                      void* output, int64_t size, int64_t width,
                                      musaStream_t stream) {
  if (size <= 0 || width <= 0 || size % width != 0) return;
  const int64_t blocks = CeilDiv(size, kThreadsPerBlock);
  AddTailVectorKernelBFloat16<<<blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const bfloat16*>(dense),
      reinterpret_cast<const bfloat16*>(tail_vector),
      reinterpret_cast<bfloat16*>(output), size, width);
}

// ---- fp16 mirror ----

__global__ void AddContiguousKernelHalf(const Eigen::half* __restrict__ lhs,
                                         const Eigen::half* __restrict__ rhs,
                                         Eigen::half* __restrict__ output,
                                         int64_t size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const __half l = *reinterpret_cast<const __half*>(&lhs[idx]);
    const __half r = *reinterpret_cast<const __half*>(&rhs[idx]);
    const __half sum = __hadd(l, r);
    *reinterpret_cast<__half*>(&output[idx]) = sum;
  }
}

__global__ void AddContiguousKernelHalfVec8(const ::uint4* __restrict__ lhs,
                                             const ::uint4* __restrict__ rhs,
                                             ::uint4* __restrict__ output,
                                             int64_t vec_size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < vec_size) {
    const ::uint4 l = lhs[idx];
    const ::uint4 r = rhs[idx];
    ::uint4 out;
    out.x = add_half_pair_packed(l.x, r.x);
    out.y = add_half_pair_packed(l.y, r.y);
    out.z = add_half_pair_packed(l.z, r.z);
    out.w = add_half_pair_packed(l.w, r.w);
    output[idx] = out;
  }
}

__global__ void AddScalarKernelHalf(const Eigen::half* __restrict__ dense,
                                     const Eigen::half* __restrict__ scalar,
                                     Eigen::half* __restrict__ output,
                                     int64_t size) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const __half s = *reinterpret_cast<const __half*>(&scalar[0]);
    const __half d = *reinterpret_cast<const __half*>(&dense[idx]);
    const __half sum = __hadd(d, s);
    *reinterpret_cast<__half*>(&output[idx]) = sum;
  }
}

__global__ void AddTailVectorKernelHalf(
    const Eigen::half* __restrict__ dense,
    const Eigen::half* __restrict__ tail_vector,
    Eigen::half* __restrict__ output, int64_t size, int64_t width) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const int64_t col = idx % width;
    const __half d = *reinterpret_cast<const __half*>(&dense[idx]);
    const __half t = *reinterpret_cast<const __half*>(&tail_vector[col]);
    *reinterpret_cast<__half*>(&output[idx]) = __hadd(d, t);
  }
}

void LaunchMusaAddContiguousHalf(const void* lhs, const void* rhs, void* output,
                                  int64_t size, musaStream_t stream) {
  if (size <= 0) return;
  if (size >= 8 && IsAligned16(lhs) && IsAligned16(rhs) && IsAligned16(output)) {
    const int64_t vec_size = size / 8;
    const int64_t vec_blocks = CeilDiv(vec_size, kThreadsPerBlock);
    AddContiguousKernelHalfVec8<<<vec_blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const ::uint4*>(lhs),
        reinterpret_cast<const ::uint4*>(rhs),
        reinterpret_cast<::uint4*>(output), vec_size);

    const int64_t tail = size - vec_size * 8;
    if (tail > 0) {
      const auto* l = reinterpret_cast<const Eigen::half*>(lhs) + vec_size * 8;
      const auto* r = reinterpret_cast<const Eigen::half*>(rhs) + vec_size * 8;
      auto* o = reinterpret_cast<Eigen::half*>(output) + vec_size * 8;
      const int64_t tail_blocks = CeilDiv(tail, kThreadsPerBlock);
      AddContiguousKernelHalf<<<tail_blocks, kThreadsPerBlock, 0, stream>>>(
          l, r, o, tail);
    }
    return;
  }
  const int64_t blocks = CeilDiv(size, kThreadsPerBlock);
  AddContiguousKernelHalf<<<blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const Eigen::half*>(lhs),
      reinterpret_cast<const Eigen::half*>(rhs),
      reinterpret_cast<Eigen::half*>(output), size);
}

void LaunchMusaAddScalarHalf(const void* dense, const void* scalar,
                              void* output, int64_t size, musaStream_t stream) {
  if (size <= 0) return;
  const int64_t blocks = CeilDiv(size, kThreadsPerBlock);
  AddScalarKernelHalf<<<blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const Eigen::half*>(dense),
      reinterpret_cast<const Eigen::half*>(scalar),
      reinterpret_cast<Eigen::half*>(output), size);
}

void LaunchMusaAddTailVectorHalf(const void* dense, const void* tail_vector,
                                  void* output, int64_t size, int64_t width,
                                  musaStream_t stream) {
  if (size <= 0 || width <= 0 || size % width != 0) return;
  const int64_t blocks = CeilDiv(size, kThreadsPerBlock);
  AddTailVectorKernelHalf<<<blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const Eigen::half*>(dense),
      reinterpret_cast<const Eigen::half*>(tail_vector),
      reinterpret_cast<Eigen::half*>(output), size, width);
}

}  // extern "C"

}  // namespace musa
}  // namespace tensorflow
