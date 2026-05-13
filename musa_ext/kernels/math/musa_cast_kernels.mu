#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <stdint.h>
#include <string.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/types.h"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

template <typename DstT>
__global__ void BoolCastKernel(const bool* src, DstT* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i] ? static_cast<DstT>(1) : static_cast<DstT>(0);
    }
}

template <typename DstT>
void LaunchBoolCast(const bool* src, DstT* dst, int n, musaStream_t stream) {
    if (n <= 0) return;

    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    BoolCastKernel<DstT><<<blocks_per_grid, threads_per_block, 0, stream>>>(src, dst, n);
}

template void LaunchBoolCast<float>(const bool*, float*, int, musaStream_t);
template void LaunchBoolCast<int32_t>(const bool*, int32_t*, int, musaStream_t);

// ===========================================================================
// Vectorized bf16 <-> fp32 and fp16 <-> fp32 Cast kernels.
//
// These dtype pairs sit on every mixed_bfloat16 / mixed_float16 dtype
// boundary: weight cast at layer entry, gradient cast back at optimizer
// entry, intermediate up/down-casts inside numerically-sensitive ops. The
// stock path goes through muDNN's generic Unary CAST which has descriptor
// setup overhead and is not vectorized for these specific pairs. The
// kernels below load 8 elements per thread using ::uint4 (bf16/fp16) or two
// float4 (fp32), do the conversion with the SDK's RNE intrinsics, and
// store 8 outputs per thread. Result: ~3-5x speedup on cast-heavy workloads
// like a Keras mixed_bfloat16 forward pass on a deep model.
// ===========================================================================

namespace {

constexpr int kCastThreadsPerBlock = 256;

static inline int64_t CastCeilDiv(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

static inline bool CastIsAligned16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

}  // namespace

extern "C" {

// ----- bf16 -> fp32 -----

__global__ void CastBfloat16ToFloat32Scalar(const bfloat16* __restrict__ src,
                                             float* __restrict__ dst,
                                             int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const __mt_bfloat16 b =
            *reinterpret_cast<const __mt_bfloat16*>(&src[idx]);
        dst[idx] = __bfloat162float(b);
    }
}

__global__ void CastBfloat16ToFloat32Vec8(const ::uint4* __restrict__ src,
                                           float* __restrict__ dst,
                                           int64_t vec_size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_size) {
        const ::uint4 in = src[idx];
        __mt_bfloat162 p0, p1, p2, p3;
        memcpy(&p0, &in.x, sizeof(p0));
        memcpy(&p1, &in.y, sizeof(p1));
        memcpy(&p2, &in.z, sizeof(p2));
        memcpy(&p3, &in.w, sizeof(p3));

        float4 lo;
        lo.x = __low2float(p0);
        lo.y = __high2float(p0);
        lo.z = __low2float(p1);
        lo.w = __high2float(p1);

        float4 hi;
        hi.x = __low2float(p2);
        hi.y = __high2float(p2);
        hi.z = __low2float(p3);
        hi.w = __high2float(p3);

        reinterpret_cast<float4*>(dst)[idx * 2] = lo;
        reinterpret_cast<float4*>(dst)[idx * 2 + 1] = hi;
    }
}

void LaunchMusaCastBfloat16ToFloat32(const void* src, void* dst, int64_t n,
                                      musaStream_t stream) {
    if (n <= 0) return;
    if (n >= 8 && CastIsAligned16(src) && CastIsAligned16(dst)) {
        const int64_t vec_size = n / 8;
        const int64_t vec_blocks = CastCeilDiv(vec_size, kCastThreadsPerBlock);
        CastBfloat16ToFloat32Vec8<<<vec_blocks, kCastThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const ::uint4*>(src),
            reinterpret_cast<float*>(dst), vec_size);

        const int64_t tail = n - vec_size * 8;
        if (tail > 0) {
            const auto* s = reinterpret_cast<const bfloat16*>(src) + vec_size * 8;
            auto* d = reinterpret_cast<float*>(dst) + vec_size * 8;
            const int64_t tail_blocks = CastCeilDiv(tail, kCastThreadsPerBlock);
            CastBfloat16ToFloat32Scalar<<<tail_blocks, kCastThreadsPerBlock, 0,
                                           stream>>>(s, d, tail);
        }
        return;
    }
    const int64_t blocks = CastCeilDiv(n, kCastThreadsPerBlock);
    CastBfloat16ToFloat32Scalar<<<blocks, kCastThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const bfloat16*>(src),
        reinterpret_cast<float*>(dst), n);
}

// ----- fp32 -> bf16 -----

__global__ void CastFloat32ToBfloat16Scalar(const float* __restrict__ src,
                                             bfloat16* __restrict__ dst,
                                             int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const __mt_bfloat16 b = __float2bfloat16(src[idx]);
        *reinterpret_cast<__mt_bfloat16*>(&dst[idx]) = b;
    }
}

__global__ void CastFloat32ToBfloat16Vec8(const float* __restrict__ src,
                                           ::uint4* __restrict__ dst,
                                           int64_t vec_size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_size) {
        const float4 lo = reinterpret_cast<const float4*>(src)[idx * 2];
        const float4 hi = reinterpret_cast<const float4*>(src)[idx * 2 + 1];

        const __mt_bfloat162 p0 = __floats2bfloat162_rn(lo.x, lo.y);
        const __mt_bfloat162 p1 = __floats2bfloat162_rn(lo.z, lo.w);
        const __mt_bfloat162 p2 = __floats2bfloat162_rn(hi.x, hi.y);
        const __mt_bfloat162 p3 = __floats2bfloat162_rn(hi.z, hi.w);

        ::uint4 out;
        memcpy(&out.x, &p0, sizeof(p0));
        memcpy(&out.y, &p1, sizeof(p1));
        memcpy(&out.z, &p2, sizeof(p2));
        memcpy(&out.w, &p3, sizeof(p3));
        dst[idx] = out;
    }
}

void LaunchMusaCastFloat32ToBfloat16(const void* src, void* dst, int64_t n,
                                      musaStream_t stream) {
    if (n <= 0) return;
    if (n >= 8 && CastIsAligned16(src) && CastIsAligned16(dst)) {
        const int64_t vec_size = n / 8;
        const int64_t vec_blocks = CastCeilDiv(vec_size, kCastThreadsPerBlock);
        CastFloat32ToBfloat16Vec8<<<vec_blocks, kCastThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const float*>(src),
            reinterpret_cast<::uint4*>(dst), vec_size);

        const int64_t tail = n - vec_size * 8;
        if (tail > 0) {
            const auto* s = reinterpret_cast<const float*>(src) + vec_size * 8;
            auto* d = reinterpret_cast<bfloat16*>(dst) + vec_size * 8;
            const int64_t tail_blocks = CastCeilDiv(tail, kCastThreadsPerBlock);
            CastFloat32ToBfloat16Scalar<<<tail_blocks, kCastThreadsPerBlock, 0,
                                           stream>>>(s, d, tail);
        }
        return;
    }
    const int64_t blocks = CastCeilDiv(n, kCastThreadsPerBlock);
    CastFloat32ToBfloat16Scalar<<<blocks, kCastThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float*>(src),
        reinterpret_cast<bfloat16*>(dst), n);
}

// ----- fp16 -> fp32 -----

__global__ void CastHalfToFloat32Scalar(const Eigen::half* __restrict__ src,
                                         float* __restrict__ dst, int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const __half h = *reinterpret_cast<const __half*>(&src[idx]);
        dst[idx] = __half2float(h);
    }
}

__global__ void CastHalfToFloat32Vec8(const ::uint4* __restrict__ src,
                                       float* __restrict__ dst,
                                       int64_t vec_size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_size) {
        const ::uint4 in = src[idx];
        __half2 p0, p1, p2, p3;
        memcpy(&p0, &in.x, sizeof(p0));
        memcpy(&p1, &in.y, sizeof(p1));
        memcpy(&p2, &in.z, sizeof(p2));
        memcpy(&p3, &in.w, sizeof(p3));

        const float2 p0f = __half22float2(p0);
        const float2 p1f = __half22float2(p1);
        const float2 p2f = __half22float2(p2);
        const float2 p3f = __half22float2(p3);

        float4 lo;
        lo.x = p0f.x;
        lo.y = p0f.y;
        lo.z = p1f.x;
        lo.w = p1f.y;

        float4 hi;
        hi.x = p2f.x;
        hi.y = p2f.y;
        hi.z = p3f.x;
        hi.w = p3f.y;

        reinterpret_cast<float4*>(dst)[idx * 2] = lo;
        reinterpret_cast<float4*>(dst)[idx * 2 + 1] = hi;
    }
}

void LaunchMusaCastHalfToFloat32(const void* src, void* dst, int64_t n,
                                  musaStream_t stream) {
    if (n <= 0) return;
    if (n >= 8 && CastIsAligned16(src) && CastIsAligned16(dst)) {
        const int64_t vec_size = n / 8;
        const int64_t vec_blocks = CastCeilDiv(vec_size, kCastThreadsPerBlock);
        CastHalfToFloat32Vec8<<<vec_blocks, kCastThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const ::uint4*>(src),
            reinterpret_cast<float*>(dst), vec_size);

        const int64_t tail = n - vec_size * 8;
        if (tail > 0) {
            const auto* s = reinterpret_cast<const Eigen::half*>(src) + vec_size * 8;
            auto* d = reinterpret_cast<float*>(dst) + vec_size * 8;
            const int64_t tail_blocks = CastCeilDiv(tail, kCastThreadsPerBlock);
            CastHalfToFloat32Scalar<<<tail_blocks, kCastThreadsPerBlock, 0,
                                       stream>>>(s, d, tail);
        }
        return;
    }
    const int64_t blocks = CastCeilDiv(n, kCastThreadsPerBlock);
    CastHalfToFloat32Scalar<<<blocks, kCastThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const Eigen::half*>(src),
        reinterpret_cast<float*>(dst), n);
}

// ----- fp32 -> fp16 -----

__global__ void CastFloat32ToHalfScalar(const float* __restrict__ src,
                                         Eigen::half* __restrict__ dst,
                                         int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const __half h = __float2half(src[idx]);
        *reinterpret_cast<__half*>(&dst[idx]) = h;
    }
}

__global__ void CastFloat32ToHalfVec8(const float* __restrict__ src,
                                       ::uint4* __restrict__ dst,
                                       int64_t vec_size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_size) {
        const float4 lo = reinterpret_cast<const float4*>(src)[idx * 2];
        const float4 hi = reinterpret_cast<const float4*>(src)[idx * 2 + 1];

        const __half2 p0 = __float22half2_rn(make_float2(lo.x, lo.y));
        const __half2 p1 = __float22half2_rn(make_float2(lo.z, lo.w));
        const __half2 p2 = __float22half2_rn(make_float2(hi.x, hi.y));
        const __half2 p3 = __float22half2_rn(make_float2(hi.z, hi.w));

        ::uint4 out;
        memcpy(&out.x, &p0, sizeof(p0));
        memcpy(&out.y, &p1, sizeof(p1));
        memcpy(&out.z, &p2, sizeof(p2));
        memcpy(&out.w, &p3, sizeof(p3));
        dst[idx] = out;
    }
}

void LaunchMusaCastFloat32ToHalf(const void* src, void* dst, int64_t n,
                                  musaStream_t stream) {
    if (n <= 0) return;
    if (n >= 8 && CastIsAligned16(src) && CastIsAligned16(dst)) {
        const int64_t vec_size = n / 8;
        const int64_t vec_blocks = CastCeilDiv(vec_size, kCastThreadsPerBlock);
        CastFloat32ToHalfVec8<<<vec_blocks, kCastThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const float*>(src),
            reinterpret_cast<::uint4*>(dst), vec_size);

        const int64_t tail = n - vec_size * 8;
        if (tail > 0) {
            const auto* s = reinterpret_cast<const float*>(src) + vec_size * 8;
            auto* d = reinterpret_cast<Eigen::half*>(dst) + vec_size * 8;
            const int64_t tail_blocks = CastCeilDiv(tail, kCastThreadsPerBlock);
            CastFloat32ToHalfScalar<<<tail_blocks, kCastThreadsPerBlock, 0,
                                       stream>>>(s, d, tail);
        }
        return;
    }
    const int64_t blocks = CastCeilDiv(n, kCastThreadsPerBlock);
    CastFloat32ToHalfScalar<<<blocks, kCastThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float*>(src),
        reinterpret_cast<Eigen::half*>(dst), n);
}

}  // extern "C"

}  // namespace musa
}  // namespace tensorflow
