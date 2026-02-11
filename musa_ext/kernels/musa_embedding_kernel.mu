#include <musa_runtime.h>
#include <cstdint>

// 必须进入 tensorflow 命名空间，才能和 .cc 文件里的调用签名完全匹配
namespace tensorflow {

    // ==========================================
    // 1. EmbeddingLookup 核函数
    // ==========================================
    template <typename T, typename Tidx>
    __global__ void EmbeddingLookupKernel(
        const T* params,
        const Tidx* ids,
        T* output,
        int vocab_size,
        int embedding_dim,
        int total_count
    ) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index < total_count) {
            int id_idx = index / embedding_dim;
            int dim_idx = index % embedding_dim;

            Tidx lookup_id = ids[id_idx];

            // 边界检查
            if (lookup_id >= 0 && lookup_id < vocab_size) {
                int src_idx = lookup_id * embedding_dim + dim_idx;
                output[index] = params[src_idx];
            }
            else {
                // 越界则填充 0 (static_cast 确保类型安全)
                output[index] = static_cast<T>(0);
            }
        }
    }

    // ==========================================
    // 2. Launcher 函数 (供 .cc 文件调用)
    // ==========================================
    template <typename T, typename Tidx>
    void LaunchEmbeddingLookup(
        const T* params,
        const Tidx* ids,
        T* output,
        int vocab_size,
        int embedding_dim,
        int total_count,
        musaStream_t stream
    ) {
        if (total_count <= 0) return;

        int block_size = 256;
        int grid_size = (total_count + block_size - 1) / block_size;

        // 注意：<<< >>> 中间严禁空格
        EmbeddingLookupKernel<T, Tidx> <<<grid_size, block_size, 0, stream >>> (
            params, ids, output, vocab_size, embedding_dim, total_count
            );
    }

    // ==========================================
    // 3. 显式实例化 (Explicit Instantiation)
    // ==========================================
    // 只有在这里显式写出来，编译器才会生成对应的二进制符号。
    // 我们定义一个宏来简化代码，确保每种数据类型 T 都自动支持所有索引类型 Tidx。

#define INSTANTIATE_LAUNCH(T, Tidx) \
    template void LaunchEmbeddingLookup<T, Tidx>( \
        const T*, const Tidx*, T*, int, int, int, musaStream_t);

    // 定义一个宏：为一个数据类型 T 注册所有可能的索引类型 (int32, long, long long)
#define REGISTER_ALL_INDICES(T) \
    INSTANTIATE_LAUNCH(T, int32_t);   /* int32 索引 */ \
    INSTANTIATE_LAUNCH(T, long);      /* int64 索引 (Linux long) */ \
    INSTANTIATE_LAUNCH(T, long long); /* int64 索引 (Linux long long) - 解决 undefined symbol */

    // -----------------------------------------------------
    // A. 浮点数类型 (Float, Double)
    // -----------------------------------------------------
    REGISTER_ALL_INDICES(float);
    REGISTER_ALL_INDICES(double);

    // -----------------------------------------------------
    // B. 整数类型 (Int32, Int64)
    // -----------------------------------------------------
    // 这就是你之前报错 "Could not find device... T=DT_INT32" 缺失的部分！
    REGISTER_ALL_INDICES(int32_t);
    REGISTER_ALL_INDICES(int64_t);

    // -----------------------------------------------------
    // C. (可选) 半精度浮点 (Half)
    // -----------------------------------------------------
    // 如果你的 MUSA 环境支持 Eigen::half 或 __half，可以打开下面的注释
    // #include <Eigen/Core> 
    // REGISTER_ALL_INDICES(Eigen::half);

} // namespace tensorflow

