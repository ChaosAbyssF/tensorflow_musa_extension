#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaRealDivOp : public MusaOpKernel {
 public:
  explicit MusaRealDivOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& dividend = ctx->input(0);  // 被除数
    const Tensor& divisor = ctx->input(1);   // 除数

    // 形状广播检查
    const int dims0 = dividend.dims();
    const int dims1 = divisor.dims();
    const int out_dims = std::max(dims0, dims1);
    TensorShape output_shape;

    for (int i = 0; i < out_dims; ++i) {
      int d0 = (i < out_dims - dims0) ? 1 : dividend.dim_size(i - (out_dims - dims0));
      int d1 = (i < out_dims - dims1) ? 1 : divisor.dim_size(i - (out_dims - dims1));

      if (d0 == d1) {
        output_shape.AddDim(d0);
      } else if (d0 == 1) {
        output_shape.AddDim(d1);
      } else if (d1 == 1) {
        output_shape.AddDim(d0);
      } else {
        ctx->CtxFailure(errors::InvalidArgument("Incompatible shapes: ",
                                                dividend.shape().DebugString(), " and ",
                                                divisor.shape().DebugString()));
        return;
      }
    }

    // 分配输出张量
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));

    // 处理空张量情况
    if (dividend.NumElements() == 0 || divisor.NumElements() == 0 || 
        output_shape.num_elements() == 0) {
      return;
    }

    // 获取MUSA句柄并创建mTensor
    auto& handle = GetHandleByCtx(ctx);
    mTensor t_dividend = CreateMTensor(dividend, format_);
    mTensor t_divisor = CreateMTensor(divisor, format_);
    mTensor t_out = CreateMTensor(*out, format_);

    // 使用muDNN的Binary操作，设置模式为除法
    ::musa::dnn::Binary op;
    op.SetMode(::musa::dnn::Binary::Mode::DIV);  // 设置为DIV模式

    // 执行除法运算
    auto status = op.Run(handle, t_out, t_dividend, t_divisor);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA RealDiv execution failed."));
  }
};

#define REGISTER_MUSA_REAL_DIV(TYPE)                                          \
  REGISTER_KERNEL_BUILDER(Name("RealDiv").Device("MUSA").TypeConstraint<TYPE>("T"), MusaRealDivOp<TYPE>);

REGISTER_MUSA_REAL_DIV(float);
REGISTER_MUSA_REAL_DIV(double);
REGISTER_MUSA_REAL_DIV(Eigen::half);    
REGISTER_MUSA_REAL_DIV(bfloat16);       
REGISTER_MUSA_REAL_DIV(int32);
REGISTER_MUSA_REAL_DIV(int64);

} // namespace musa
} // namespace tensorflow
