#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/util/bcast.h"

#include "utils_op.h"
#include <mudnn.h>

namespace tensorflow {
namespace musa {

// 1. 注册 MusaMaximum Op
REGISTER_OP("MusaMaximum")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {float, half, bfloat16, double, int32, int64}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

template <typename T>
class MusaMaximumOp : public MusaOpKernel {
 public:
  explicit MusaMaximumOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // 调试日志
    fprintf(stderr, ">>> [MUSA_TRACE_AUTO] %s\n", name().c_str());

    const Tensor& input_x = ctx->input(0);
    const Tensor& input_y = ctx->input(1);

    // --- 步骤 A: 处理广播 (Broadcasting) ---
    // Maximum 支持类似 [1, 10] 和 [5, 10] 的比较，结果是 [5, 10]
    BCast bcast(BCast::Vec(input_x.shape().dim_sizes()),
                BCast::Vec(input_y.shape().dim_sizes()));
    
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument("Incompatible shapes: ",
                                        input_x.shape().DebugString(), " vs ",
                                        input_y.shape().DebugString()));

    TensorShape output_shape = BCast::ToShape(bcast.output_shape());
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) return;

    // --- 步骤 B: 准备 MUSA 环境 ---
    auto& handle = GetHandleByCtx(ctx);

    // 根据 utils_op.h，逐元素二元运算使用 mBinary
    mBinary binary_op;
    
    // 设置模式为 MAX
    // 参考 utils_op.h 中的 BINARY_MODE = ::musa::dnn::Binary::Mode
    binary_op.SetMode(BINARY_MODE::MAX);

    // --- 步骤 C: 创建 mTensor ---
    mTensor mt_x = CreateMTensor(input_x, format_);
    mTensor mt_y = CreateMTensor(input_y, format_);
    mTensor mt_out = CreateMTensor(*output, format_);

    // --- 步骤 D: 执行计算 ---
    // 二元算子通常不需要额外的 MemoryMaintainer，除非涉及复杂的广播
    // 这里的 Run 接收 (handle, output, input_x, input_y)
    auto status = binary_op.Run(handle, mt_out, mt_x, mt_y);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Maximum execution failed. Status: ", (int)status));
  }
};

// 2. 注册 Kernel (同时绑定标准 Maximum 和自定义 MusaMaximum)
#define REGISTER_MAXIMUM(TYPE)                                      \
  REGISTER_KERNEL_BUILDER(Name("Maximum")                           \
                              .Device("MUSA")                       \
                              .TypeConstraint<TYPE>("T"),           \
                          MusaMaximumOp<TYPE>);                     \
  REGISTER_KERNEL_BUILDER(Name("MusaMaximum")                       \
                              .Device("MUSA")                       \
                              .TypeConstraint<TYPE>("T"),           \
                          MusaMaximumOp<TYPE>);

REGISTER_MAXIMUM(float);
REGISTER_MAXIMUM(Eigen::half);
REGISTER_MAXIMUM(bfloat16);
REGISTER_MAXIMUM(int32);
REGISTER_MAXIMUM(int64);

#undef REGISTER_MAXIMUM

}  // namespace musa
}  // namespace tensorflow
