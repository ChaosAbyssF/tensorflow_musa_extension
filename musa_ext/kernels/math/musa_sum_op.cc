#include <mudnn.h>

#include "../utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "musa_reduce_functor.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSumOp : public MusaOpKernel {
 public:
  explicit MusaSumOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  }

  // Sum is computationally intensive (reduction operation)
  // Mark as expensive to enable optimal scheduling (async execution)
  // Expected improvement: Better overlapping with other operations
  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& axes_tensor = ctx->input(1);

    if (input.NumElements() == 0) {
      ctx->set_output(0, input);
      return;
    }

    int64_t num_axes = axes_tensor.NumElements();
    std::vector<int> reduce_dims;
    gtl::InlinedVector<bool, 4> bitmap(input.dims(), false);

    if (num_axes > 0) {
      if (axes_tensor.dtype() == DT_INT32) {
        auto axes_flat = axes_tensor.flat<int32>();
        for (int64_t i = 0; i < num_axes; ++i) {
          int32 index = axes_flat(i);
          if (index < 0) index += input.dims();
          if (index >= 0 && index < input.dims() && !bitmap[index]) {
            bitmap[index] = true;
            reduce_dims.push_back(static_cast<int>(index));
          }
        }
      } else if (axes_tensor.dtype() == DT_INT64) {
        auto axes_flat = axes_tensor.flat<int64>();
        for (int64_t i = 0; i < num_axes; ++i) {
          int64 index = axes_flat(i);
          if (index < 0) index += input.dims();
          if (index >= 0 && index < input.dims() && !bitmap[index]) {
            bitmap[index] = true;
            reduce_dims.push_back(static_cast<int>(index));
          }
        }
      } else {
        OP_REQUIRES(ctx, false,
                    errors::InvalidArgument(
                        "reduction_indices must be int32 or int64"));
      }
    }

    TensorShape output_shape;
    TensorShape musa_output_shape;
    int64_t reduce_elements = 1;

    for (int d = 0; d < input.dims(); ++d) {
      if (bitmap[d]) {
        reduce_elements *= input.dim_size(d);
        if (keep_dims_) {
          output_shape.AddDim(1);
        }
        musa_output_shape.AddDim(1);
      } else {
        output_shape.AddDim(input.dim_size(d));
        musa_output_shape.AddDim(input.dim_size(d));
      }
    }

    if (reduce_elements == 1) {
      Tensor output;
      // zero-copy: assign new output_shape, underlying GPU memory still points
      // to input
      bool success = output.CopyFrom(input, output_shape);
      OP_REQUIRES(ctx, success,
                  errors::Internal("MUSA Reduce: Tensor::CopyFrom failed."));
      ctx->set_output(0, output);
      return;
    }

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));

    if (out->NumElements() == 0) return;

    if (reduce_elements == 0) return;

    Tensor out_reshaped(out->dtype());
    OP_REQUIRES(ctx, out_reshaped.CopyFrom(*out, musa_output_shape),
                errors::Internal("Reshape failed."));

    // bf16 inputs are promoted to fp32 inside the helper; other dtypes go
    // straight to muDNN Reduce. This matters most for tf.clip_by_global_norm
    // (sum of squared bf16 grads) and for any bf16 tf.reduce_sum on tensors
    // with more than a few thousand contributors.
    OP_REQUIRES_OK(
        ctx, RunReduceWithFP32Promotion(
                 ctx, input, &out_reshaped, ::musa::dnn::Reduce::Mode::ADD,
                 reduce_dims.data(), static_cast<int>(reduce_dims.size()),
                 "MUSA muDNN Reduce Sum execution failed. Status: "));
  }

 private:
  bool keep_dims_;
};

#define REGISTER_MUSA_SUM(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(Name("Sum")                           \
                              .Device("MUSA")                   \
                              .TypeConstraint<TYPE>("T")        \
                              .TypeConstraint<int32>("Tidx")    \
                              .HostMemory("reduction_indices"), \
                          MusaSumOp<TYPE>);                     \
  REGISTER_KERNEL_BUILDER(Name("Sum")                           \
                              .Device("MUSA")                   \
                              .TypeConstraint<TYPE>("T")        \
                              .TypeConstraint<int64>("Tidx")    \
                              .HostMemory("reduction_indices"), \
                          MusaSumOp<TYPE>);

REGISTER_MUSA_SUM(float);
REGISTER_MUSA_SUM(Eigen::half);
REGISTER_MUSA_SUM(bfloat16);
REGISTER_MUSA_SUM(double);
REGISTER_MUSA_SUM(int32);
REGISTER_MUSA_SUM(int64);

#undef REGISTER_MUSA_SUM

}  // namespace musa
}  // namespace tensorflow
