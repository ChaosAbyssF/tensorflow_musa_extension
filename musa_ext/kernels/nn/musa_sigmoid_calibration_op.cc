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

    // Compute S / (S + Scale * (1 - S))
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchSigmoidCalibrationKernel<T>(
        input.flat<T>().data(), scale.flat<T>().data(),
        output->flat<T>().data(), static_cast<int>(input.NumElements()),
        stream);
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
