#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#pragma GCC diagnostic pop

namespace tensorflow {
namespace musa {

// ========================
// Device-side helper function
// ========================

// 模板函数：计算 range 值
template <typename T>
__device__ __forceinline__ T ComputeRangeValue(T start, T delta, int64_t idx) {
  return start + static_cast<T>(idx) * delta;
}

// ========================
// Range Kernel
// ========================
template <typename T>
__global__ void RangeKernel(const T start, const T delta, const int64_t size, T* out) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = ComputeRangeValue(start, delta, idx);  // ← 现在可以了
  }
}

// ========================
// Launcher
// ========================
template <typename T>
void MusaRangeKernelLauncher(const T start, const T delta, const int64_t size, void* out, musaStream_t stream) {
  if (size == 0) return;

  constexpr int block_size = 256;
  const int grid_size = static_cast<int>((size + block_size - 1) / block_size);

  RangeKernel<T><<<grid_size, block_size, 0, stream>>>(
      start, delta, size, static_cast<T*>(out));
}

// ========================
// Explicit instantiations
// ========================
template void MusaRangeKernelLauncher<float>(const float, const float, const int64_t, void*, musaStream_t);
template void MusaRangeKernelLauncher<double>(const double, const double, const int64_t, void*, musaStream_t);
template void MusaRangeKernelLauncher<int>(const int, const int, const int64_t, void*, musaStream_t);          // int32
template void MusaRangeKernelLauncher<long long>(const long long, const long long, const int64_t, void*, musaStream_t); // int64
//template void MusaRangeKernelLauncher<Eigen::half>(const Eigen::half, const Eigen::half, const int64_t, void*, musaStream_t);
//template void MusaRangeKernelLauncher<bfloat16>(const bfloat16, const bfloat16, const int64_t, void*, musaStream_t);

}  // namespace musa
}  // namespace tensorflow

