#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace musa {

// 1. 注册 Op 定义 (复用 TF 标准定义，通常不需要重写 REGISTER_OP，但为了完整性)
// 注意：TF 核心通常已经注册了 "Placeholder"，我们主要负责注册 Kernel。
// 如果是独立插件开发，直接注册 Kernel 即可。

template <typename T>
class MusaPlaceholderOp : public OpKernel {
 public:
  explicit MusaPlaceholderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // Placeholder 可能有一些 shape 属性，但通常不需要在构造函数里做重活
  }

  void Compute(OpKernelContext* ctx) override {
    // ！！！ 核心逻辑 ！！！
    
    // 如果程序真的跑进了这个 Compute 函数，说明出大事了：
    // 用户调用了 session.run()，但是忘了在 feed_dict 里给这个 placeholder 喂数据。
    
    if (ctx->output_required(0)) {
      ctx->CtxFailure(errors::InvalidArgument(
          "You must feed a value for placeholder tensor '", name(), 
          "' with dtype ", DataTypeString(output_type(0))));
    }
    
    // 没错，这里不需要调用 mudnn，不需要 allocate_output
    // 因为如果有数据喂进来，TF 框架层会在这个 Op 运行前就把 output 替换掉。
  }
};

// 2. 注册 Kernel
// 我们告诉 TF：如果有人把 Placeholder 放在了 MUSA 设备上，请用这个 Kernel (虽然它只是个报错复读机)
#define REGISTER_PLACEHOLDER(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(Name("Placeholder")                       \
                              .Device("MUSA")                       \
                              .TypeConstraint<TYPE>("dtype"),       \
                          MusaPlaceholderOp<TYPE>);

REGISTER_PLACEHOLDER(float);
REGISTER_PLACEHOLDER(double);
REGISTER_PLACEHOLDER(Eigen::half);
REGISTER_PLACEHOLDER(bfloat16);
REGISTER_PLACEHOLDER(int32);
REGISTER_PLACEHOLDER(int64);
REGISTER_PLACEHOLDER(bool);

#undef REGISTER_PLACEHOLDER

}  // namespace musa
}  // namespace tensorflow
