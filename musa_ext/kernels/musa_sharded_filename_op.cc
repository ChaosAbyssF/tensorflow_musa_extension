#include <cstdio> // 用于 snprintf
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace musa {

class MusaShardedFilenameOp : public OpKernel {
 public:
  explicit MusaShardedFilenameOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // 1. 获取输入
    // 按照官方定义，这三个输入通常都是标量 (Scalar)
    const Tensor& basename_t = ctx->input(0);
    const Tensor& shard_t = ctx->input(1);
    const Tensor& num_shards_t = ctx->input(2);

    // 2. 校验输入是否为标量
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(basename_t.shape()),
                errors::InvalidArgument("basename must be a scalar"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(shard_t.shape()),
                errors::InvalidArgument("shard must be a scalar"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(num_shards_t.shape()),
                errors::InvalidArgument("num_shards must be a scalar"));

    // 3. 提取数值 (在 CPU 内存中读取)
    const tstring& basename = basename_t.scalar<tstring>()();
    int32 shard = shard_t.scalar<int32>()();
    int32 num_shards = num_shards_t.scalar<int32>()();

    // 4. 逻辑校验
    OP_REQUIRES(ctx, shard >= 0, 
                errors::InvalidArgument("shard must be >= 0"));
    OP_REQUIRES(ctx, num_shards > 0, 
                errors::InvalidArgument("num_shards must be > 0"));
    OP_REQUIRES(ctx, shard < num_shards, 
                errors::InvalidArgument("shard must be < num_shards"));

    // 5. 分配输出
    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, basename_t.shape(), &output_t));

    // 6. 格式化字符串
    // TensorFlow 标准格式是: %s-%05d-of-%05d
    // 例如: train-00001-of-00010
    char buffer[128];
    int n = snprintf(buffer, sizeof(buffer), "-%05d-of-%05d", shard, num_shards);
    
    // 检查 buffer 是否溢出 (虽然对于 int32 来说不太可能，但为了严谨)
    OP_REQUIRES(ctx, n >= 0 && n < sizeof(buffer),
                errors::Internal("Formatting filename failed."));

    // 拼接并赋值给输出
    output_t->scalar<tstring>()() = basename + string(buffer);
  }
};

// 7. 注册 Kernel
// 关键：全部使用 .HostMemory，因为这纯粹是字符串操作
#define REGISTER_MUSA_KERNEL()                      \
  REGISTER_KERNEL_BUILDER(Name("ShardedFilename")   \
                              .Device("MUSA")       \
                              .HostMemory("basename") \
                              .HostMemory("shard")    \
                              .HostMemory("num_shards") \
                              .HostMemory("filename"), \
                          MusaShardedFilenameOp);

REGISTER_MUSA_KERNEL();

}  // namespace musa
}  // namespace tensorflow
