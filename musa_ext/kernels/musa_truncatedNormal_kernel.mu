#include <musa_runtime.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/stream_executor/stream.h"

#if defined(__MUSACC__)
#define __CUDACC__
#define __MUSA_DEFINED_CUDACC__
#endif
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#if defined(__MUSA_DEFINED_CUDACC__)
#undef __CUDACC__
#undef __MUSA_DEFINED_CUDACC__
#endif

namespace tensorflow {
namespace musa {

using random::PhiloxRandom;
using random::TruncatedNormalDistribution;

// 每个线程处理 kGroupSize 个元素（TruncatedNormalDistribution 每次生成 4 个 float）
template <typename T, int kBlockSize = 256>
__global__ void __launch_bounds__(kBlockSize)
PhiloxTruncatedNormalKernel(
    const uint64_t num_elements,
    const PhiloxRandom base_gen,
    TruncatedNormalDistribution<PhiloxRandom, T> dist,
    T* __restrict__ data) {
  using TruncatedDist = TruncatedNormalDistribution<PhiloxRandom, T>;
  constexpr int kGroupSize = TruncatedDist::kResultElementCount;

  const uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t thread_count = gridDim.x * blockDim.x;
  uint64_t group_index = thread_id;

  while (group_index * kGroupSize < num_elements) {
    PhiloxRandom gen = base_gen;
    gen.Skip(group_index);

    auto samples = dist(&gen);

    for (int i = 0; i < kGroupSize; ++i) {
      const uint64_t index = group_index * kGroupSize + i;
      if (index < num_elements) {
        data[index] = samples[i];
      }
    }
    group_index += thread_count;
  }
}

template <typename T>
void LaunchPhiloxTruncatedNormal(
    musaStream_t stream,
    T* data,
    uint64_t num_elements,
    const PhiloxRandom& philox) {
  using TruncatedDist = TruncatedNormalDistribution<PhiloxRandom, T>;
  TruncatedDist dist;
  constexpr int kBlockSize = 256;
  constexpr int kGroupSize = TruncatedDist::kResultElementCount;
  const uint64_t num_groups = (num_elements + kGroupSize - 1) / kGroupSize;
  const int num_blocks = (num_groups + kBlockSize - 1) / kBlockSize;

  PhiloxTruncatedNormalKernel<T><<<num_blocks, kBlockSize, 0, stream>>>(
      num_elements, philox, dist, data);
}

// 显式实例化
template void LaunchPhiloxTruncatedNormal<float>(
    musaStream_t, float*, uint64_t, const tensorflow::random::PhiloxRandom&);
// template void LaunchPhiloxTruncatedNormal<double>(
//     musaStream_t, double*, uint64_t, const tensorflow::random::PhiloxRandom&);
// template void LaunchPhiloxTruncatedNormal<Eigen::half>(
//     musaStream_t, Eigen::half*, uint64_t, const tensorflow::random::PhiloxRandom&);

}  // namespace musa
}  // namespace tensorflow
