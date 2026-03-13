#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "../utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaFusedRealDivAddOp : public MusaOpKernel {
 public:
  explicit MusaFusedRealDivAddOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0); // Input from RealDiv
    const Tensor& y = ctx->input(1); // Divisor
    const Tensor& z = ctx->input(2); // Input for Add

    // Broadcast is complex for 3 inputs, here implementation using sequential Run 
    // to reuse existing musa dnn binary kernels.
    // In a production environment, you would use a single fused kernel.
    
    // 1. x / y
    Tensor temp_div;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(x.dtype(), x.shape(), &temp_div));
    
    auto& handle = GetHandleByCtx(ctx);
    mTensor t_x = CreateMTensor(x, format_);
    mTensor t_y = CreateMTensor(y, format_);
    mTensor t_temp_div = CreateMTensor(temp_div, format_);

    ::musa::dnn::Binary div_op;
    div_op.SetMode(::musa::dnn::Binary::Mode::DIV);
    auto status = div_op.Run(handle, t_temp_div, t_x, t_y);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA FusedRealDivAdd: Div execution failed."));

    // 2. temp_div + z
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &out));
    mTensor t_z = CreateMTensor(z, format_);
    mTensor t_out = CreateMTensor(*out, format_);

    ::musa::dnn::Binary add_op;
    add_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    status = add_op.Run(handle, t_out, t_temp_div, t_z);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA FusedRealDivAdd: Add execution failed."));
  }
};

#define REGISTER_MUSA_FUSED_REAL_DIV_ADD(TYPE)                           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("MusafusedRealDivAdd").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaFusedRealDivAddOp<TYPE>);

REGISTER_MUSA_FUSED_REAL_DIV_ADD(float);
REGISTER_MUSA_FUSED_REAL_DIV_ADD(double);

}  // namespace musa
}  // namespace tensorflow
