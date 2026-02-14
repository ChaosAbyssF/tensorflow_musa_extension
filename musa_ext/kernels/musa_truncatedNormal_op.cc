#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/guarded_philox_random.h"
#include <random>
#include <limits>

#include "utils_op.h"
#include "mu/device/musa_memcpy.h"

namespace tensorflow {
namespace musa {

// 前向声明 MUSA Kernel 启动函数
template <typename T>
void LaunchPhiloxTruncatedNormal(musaStream_t stream, T* output, 
                                     uint64_t num_elements,
                                     const random::PhiloxRandom& philox);

template <typename T>
class MusaTruncatedNormalOp : public MusaOpKernel {
 public:
  explicit MusaTruncatedNormalOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed2", &seed2_));
    
    // TruncatedNormal 默认截断范围为 [-2σ, +2σ]
    // 这里硬编码标准差为1,均值为0
  }

  void Compute(OpKernelContext* ctx) override {
    // 解析输入 shape
    const Tensor& shape_tensor = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &shape));

    // 分配输出张量
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    
    int64_t total_elements = shape.num_elements();
    if (total_elements == 0) return;

    // 初始化 Philox 随机数生成器
    GuardedPhiloxRandom generator;
    generator.Init(seed_, seed2_);
    
    // 预分配随机数样本
    // TruncatedNormal 需要更多样本(因为拒绝采样),预留 4x 空间
    const int64_t samples_needed = 
        (total_elements + 3) / 4 * 8;  // 4x oversampling
    auto philox = generator.ReserveSamples32(samples_needed);

    // 获取 MUSA Stream
    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    // 启动 MUSA Kernel
    LaunchPhiloxTruncatedNormal<T>(
        stream, 
        output->flat<T>().data(), 
        total_elements, 
        philox
    );
  }

 private:
  tensorflow::int64 seed_;
  tensorflow::int64 seed2_;
};

// 注册宏
#define REGISTER_MUSA_TRUNCATED_NORMAL_KERNEL(TYPE)       \
  REGISTER_KERNEL_BUILDER(Name("TruncatedNormal")         \
                            .Device("MUSA")               \
                            .HostMemory("shape")          \
                            .TypeConstraint<TYPE>("dtype"),\
                          MusaTruncatedNormalOp<TYPE>);

// 批量注册支持的数据类型
REGISTER_MUSA_TRUNCATED_NORMAL_KERNEL(float);
REGISTER_MUSA_TRUNCATED_NORMAL_KERNEL(double);
REGISTER_MUSA_TRUNCATED_NORMAL_KERNEL(Eigen::half);
REGISTER_MUSA_TRUNCATED_NORMAL_KERNEL(Eigen::bfloat16);

#undef REGISTER_MUSA_TRUNCATED_NORMAL_KERNEL

}  // namespace musa
}  // namespace tensorflow