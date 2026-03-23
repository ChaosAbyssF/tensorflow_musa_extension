#include "../array/musa_fill_functor.h"
#include "../utils_op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace musa {

template <typename T>
void LaunchSigmoidCalibrationKernel(const void*, const void*, void*, int,
                                    musaStream_t);

/**
 * MusaSigmoidCalibrationOp
 *
 * Performs fusion: S / (S + Scale * (1 - S))
 * where S = Sigmoid(x)
 *
 * Note that this formula is equivalent to `1 / (1 + Scale * exp(-x))`
 *
 * This implements the specific activation logic from the given graph.
 */
template <typename T>
class MusaSigmoidCalibrationOp : public MusaOpKernel {
 public:
  explicit MusaSigmoidCalibrationOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& scale = ctx->input(1);

    if (input.NumElements() == 0) return;

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    useMudnn(ctx, input, scale, output);
  }

 private:
  void useKernel(OpKernelContext* ctx, const Tensor& input, const Tensor& scale,
                 Tensor* output) {
    // Compute S / (S + Scale * (1 - S))
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchSigmoidCalibrationKernel<T>(
        input.flat<T>().data(), scale.flat<T>().data(),
        output->flat<T>().data(), static_cast<int>(input.NumElements()),
        stream);
  }

  void useMudnn(OpKernelContext* ctx, const Tensor& input, const Tensor& scale,
                Tensor* output) {
    // 1. Run Sigmoid
    Tensor sigmoid_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), input.shape(),
                         &sigmoid_tensor));
    auto sigmoid_mt = CreateMTensor(sigmoid_tensor, format_);
    auto& handle = GetHandleByCtx(ctx);
    auto in_mt = CreateMTensor(input, format_);

    ::musa::dnn::Unary sigmoid_op;
    MTOP_CHECK_OK(sigmoid_op.SetMode(::musa::dnn::Unary::Mode::SIGMOID),
                  "Set Sigmoid", ctx);
    MTOP_CHECK_OK_RUN(sigmoid_op.Run(handle, sigmoid_mt, in_mt),
                      "Sigmoid Forward Run", ctx);

    // 2. Run 1 - S
    Tensor one_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), input.shape(),
                         &one_tensor));
    auto one_mt = CreateMTensor(one_tensor, format_);
    mFill fill_op;
    MTOP_CHECK_OK(fill_op.SetValue(1.0f), "Set Fill Value", ctx);
    MTOP_CHECK_OK(fill_op.Run(handle, one_mt), "Fill One Tensor", ctx);
    Tensor one_minus_s_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), input.shape(),
                         &one_minus_s_tensor));
    auto one_minus_s_mt = CreateMTensor(one_minus_s_tensor, format_);
    ::musa::dnn::Binary sub_op;
    MTOP_CHECK_OK(sub_op.SetMode(::musa::dnn::Binary::Mode::SUB), "Set Sub",
                  ctx);
    MTOP_CHECK_OK_RUN(sub_op.Run(handle, one_minus_s_mt, one_mt, sigmoid_mt),
                      "One Minus S Run", ctx);

    // 3. Run Scale * (1 - S)
    Tensor scale_times_one_minus_s_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), input.shape(),
                         &scale_times_one_minus_s_tensor));
    auto scale_times_one_minus_s_mt =
      CreateMTensor(scale_times_one_minus_s_tensor, format_);
    auto scale_mt = CreateMTensor(scale, format_);
    ::musa::dnn::Binary mul_op;
    MTOP_CHECK_OK(mul_op.SetMode(::musa::dnn::Binary::Mode::MUL), "Set Mul",
                  ctx);
    MTOP_CHECK_OK_RUN(
        mul_op.Run(handle, scale_times_one_minus_s_mt, scale_mt, one_minus_s_mt),
        "Scale Times One Minus S Run", ctx);

    // 4. Run S + Scale * (1 - S)
    Tensor denominator_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), input.shape(),
                         &denominator_tensor));
    auto denominator_mt = CreateMTensor(denominator_tensor, format_);
    ::musa::dnn::Binary add_op;
    MTOP_CHECK_OK(add_op.SetMode(::musa::dnn::Binary::Mode::ADD), "Set Add",
                  ctx);
    MTOP_CHECK_OK_RUN(add_op.Run(handle, denominator_mt, sigmoid_mt,
                                 scale_times_one_minus_s_mt),
                      "Denominator Run", ctx);

    // 5. Run S / (S + Scale * (1 - S))
    ::musa::dnn::Binary div_op;
    auto out_mt = CreateMTensor(*output, format_);
    MTOP_CHECK_OK(div_op.SetMode(::musa::dnn::Binary::Mode::DIV), "Set Div",
                  ctx);
    MTOP_CHECK_OK_RUN(div_op.Run(handle, out_mt, sigmoid_mt, denominator_mt),
                      "Final Division Run", ctx);
  }
};

#define REGISTER_MUSA_SIGMOID_CALIBRATION(type)                                \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MusaSigmoidCalibration").Device("MUSA").TypeConstraint<type>("T"), \
      MusaSigmoidCalibrationOp<type>);

REGISTER_MUSA_SIGMOID_CALIBRATION(float);
REGISTER_MUSA_SIGMOID_CALIBRATION(Eigen::half);
REGISTER_MUSA_SIGMOID_CALIBRATION(bfloat16);
REGISTER_MUSA_SIGMOID_CALIBRATION(double);

}  // namespace musa
}  // namespace tensorflow
