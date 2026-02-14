#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/bfloat16.h"
// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "../utils/musa_guarded_philox_random.h"

#include "utils_op.h"
#include "mu/device/musa_memcpy.h"

namespace tensorflow {
namespace musa {

template <typename T>
void LaunchPhiloxNormal(musaStream_t, T*, uint64_t, const random::PhiloxRandom&);

template <typename T>
class MusaRandomStandardNormalOp : public MusaOpKernel {
  public:

    explicit MusaRandomStandardNormalOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("seed2", &seed2_));
    }

    void Compute(OpKernelContext* ctx) override {
      // Parse the input
      const Tensor& shape_tensor = ctx->input(0);
      TensorShape shape;
      OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &shape));

      // allocate the output
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
      int64_t total_elements = shape.num_elements();
      if (total_elements == 0) return;

      // Step 1: 初始化 Philox 随机数生成器（带 seed 管理）
      GuardedPhiloxRandom generator;
      generator.Init(seed_, seed2_);    // !!!Probably leads to uncontrolled context randomized?
      // OP_REQUIRES_OK(ctx, generator.Init(ctx, seed_, seed2_));
      
      // Step 2: 预分配随机数样本（避免重复计算）
      // 每 4 个输出元素需要 2 个 uint64 (Philox 一次生成 4 个 uint32)
      const int64_t samples_needed = 
          (output->NumElements() + 3) / 4 * 2;  // kResultElementCount=4
      auto philox = generator.ReserveSamples32(samples_needed);

      // Step 3: 分发到 MUSA Kernel
      // auto stream = ctx->op_device_context()->stream();
      // OP_REQUIRES_ASYNC(ctx, stream, done, 
      //     errors::Internal("No MUSA stream available"));
      auto& handle = GetHandleByCtx(ctx);
      musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

      // 根据 dtype 分发到具体实现
      LaunchPhiloxNormal<T>(
        stream, 
        output->flat<T>().data(), 
        output->NumElements(), 
        philox
      );
    }
    
  private:
    tensorflow::int64 seed_;
    tensorflow::int64 seed2_;
};

#define REGISTER_MUSA_RANDOM_STANDARD_NORMAL_KERNEL(TYPE)   \
  REGISTER_KERNEL_BUILDER(Name("RandomStandardNormal")      \
                            .Device("MUSA")                 \
                            .HostMemory("shape")            \
                            .TypeConstraint<TYPE>("dtype"), \
                          MusaRandomStandardNormalOp<TYPE>);                                 

// 执行批量注册
REGISTER_MUSA_RANDOM_STANDARD_NORMAL_KERNEL(float);
REGISTER_MUSA_RANDOM_STANDARD_NORMAL_KERNEL(double);
// REGISTER_MUSA_RANDOM_STANDARD_NORMAL_KERNEL(Eigen::half)
// REGISTER_MUSA_RANDOM_STANDARD_NORMAL_KERNEL(Eigen::bfloat16);

#undef REGISTER_MUSA_RANDOM_STANDARD_NORMAL_KERNEL

}   // namespace musa
}   // namespace tensorflow