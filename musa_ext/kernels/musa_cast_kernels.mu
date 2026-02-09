#include <musa_runtime.h>
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace musa {

// 1. 定义核函数
template <typename DstT>
__global__ void BoolCastKernel(const bool* src, DstT* dst, int n) {
    // 手动计算全局索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i] ? static_cast<DstT>(1) : static_cast<DstT>(0);
    }
}

// 2. 定义启动函数
template <typename DstT>
void LaunchBoolCast(const bool* src, DstT* dst, int n, musaStream_t stream) {
    if (n <= 0) return;

    // 手动配置：每个 Block 256 个线程
    int threads_per_block = 256;
    // 计算需要的 Block 数量，确保覆盖所有数据
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // 启动 MUSA Kernel
    BoolCastKernel<DstT><<<blocks_per_grid, threads_per_block, 0, stream>>>(src, dst, n);
}

// 显式实例化支持的类型
template void LaunchBoolCast<float>(const bool*, float*, int, musaStream_t);
template void LaunchBoolCast<int32_t>(const bool*, int32_t*, int, musaStream_t);

} // namespace musa
} // namespace tensorflow
