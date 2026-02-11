#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h" 
#include "utils_op.h" 

namespace tensorflow {
namespace musa {

/**
 * 1. 前向算子 (使用 Unary SIGMOID)
 */
template <typename T>
class MusaSigmoidOp : public MusaOpKernel {
public:
    explicit MusaSigmoidOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& input = ctx->input(0);
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
        if (input.NumElements() == 0) return;

        auto& handle = GetHandleByCtx(ctx);
        auto in_mt = CreateMTensor(input, format_);
        auto out_mt = CreateMTensor(*output, format_);

        ::musa::dnn::Unary op;
        MTOP_CHECK_OK(op.SetMode(::musa::dnn::Unary::Mode::SIGMOID), "Set Sigmoid", ctx);
        MTOP_CHECK_OK_RUN(op.Run(handle, out_mt, in_mt), "Sigmoid Forward Run", ctx);
    }
};

/**
 * 2. 反向算子 (使用原生 Binary SIGMOID_BW)
 */
template <typename T>
class MusaSigmoidGradOp : public MusaOpKernel {
public:
    explicit MusaSigmoidGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        // 输出 Trace 信息方便调试
        fprintf(stderr, ">>> [MUSA_NATIVE_BW] %s\n", name().c_str());
        
        const Tensor& y = ctx->input(0);   // 前向输出 Sigmoid(x)
        const Tensor& dy = ctx->input(1);  // 传回来的梯度
        
        Tensor* dz = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, y.shape(), &dz));

        if (y.NumElements() == 0) return;

        auto& handle = GetHandleByCtx(ctx);
        auto y_mt = CreateMTensor(y, format_);
        auto dy_mt = CreateMTensor(dy, format_);
        auto dz_mt = CreateMTensor(*dz, format_);

        // 核心：根据你的 grep，它属于 Binary 类
        ::musa::dnn::Binary op;
        MTOP_CHECK_OK(op.SetMode(::musa::dnn::Binary::Mode::SIGMOID_BW), "Set Sigmoid_BW", ctx);
        
        // 执行原生反向：dz = sigmoid_bw(y, dy)
        // muDNN Binary 的典型参数顺序: (handle, output, input1, input2)
        // 对于反向算子，input1 是前向输出 y，input2 是梯度 dy
        // 尝试交换 y_mt 和 dy_mt 的顺序
		MTOP_CHECK_OK_RUN(op.Run(handle, dz_mt, dy_mt, y_mt), "Sigmoid_BW Native Run", ctx);

	}
};
// 注册	
REGISTER_KERNEL_BUILDER(
    Name("Sigmoid").Device("MUSA").TypeConstraint<float>("T"),
    MusaSigmoidOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("SigmoidGrad").Device("MUSA").TypeConstraint<float>("T"),
    MusaSigmoidGradOp<float>);

}  // namespace musa
}  // namespace tensorflow

