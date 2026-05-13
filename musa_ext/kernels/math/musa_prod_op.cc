#include <mudnn.h>

#include "../utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "musa_reduce_functor.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaProdOp : public MusaOpKernel {
 public:
  explicit MusaProdOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  }

  // Prod is a reduction operation - computationally intensive
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
      }
    }

    TensorShape output_shape;
    TensorShape musa_output_shape;
    int64_t reduce_elements = 1;

    for (int d = 0; d < input.dims(); ++d) {
      if (bitmap[d]) {
        reduce_elements *= input.dim_size(d);
        if (keep_dims_) output_shape.AddDim(1);
        musa_output_shape.AddDim(1);
      } else {
        output_shape.AddDim(input.dim_size(d));
        musa_output_shape.AddDim(input.dim_size(d));
      }
    }

    if (reduce_elements == 1) {
      Tensor output;
      bool success = output.CopyFrom(input, output_shape);
      OP_REQUIRES(ctx, success,
                  errors::Internal("MUSA Reduce: Tensor::CopyFrom failed."));
      ctx->set_output(0, output);
      // return early for this trivial case to avoid unnecessary setup and
      // kernel launch
      return;
    }

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));
    if (out->NumElements() == 0 || reduce_elements == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    Tensor out_reshaped(out->dtype());
    OP_REQUIRES(ctx, out_reshaped.CopyFrom(*out, musa_output_shape),
                errors::Internal("Reshape failed."));

    // bf16 inputs are promoted to fp32 inside the helper. CreateMTensor
    // already sets contiguous-stride descriptors via muDNN's SetNdInfo
    // (with rank+dims; muDNN fills strides for the contiguous layout),
    // which subsumes the explicit SafeSetShape that used to live here.
    OP_REQUIRES_OK(
        ctx, RunReduceWithFP32Promotion(
                 ctx, input, &out_reshaped, ::musa::dnn::Reduce::Mode::PROD,
                 reduce_dims.data(), static_cast<int>(reduce_dims.size()),
                 "MUSA Reduce Prod failed. Status: "));
  }

 private:
  bool keep_dims_;
};

#define REGISTER_MUSA_PROD(TYPE)                                \
  REGISTER_KERNEL_BUILDER(Name("Prod")                          \
                              .Device("MUSA")                   \
                              .TypeConstraint<TYPE>("T")        \
                              .TypeConstraint<int32>("Tidx")    \
                              .HostMemory("reduction_indices"), \
                          MusaProdOp<TYPE>);                    \
  REGISTER_KERNEL_BUILDER(Name("Prod")                          \
                              .Device("MUSA")                   \
                              .TypeConstraint<TYPE>("T")        \
                              .TypeConstraint<int64>("Tidx")    \
                              .HostMemory("reduction_indices"), \
                          MusaProdOp<TYPE>);

REGISTER_MUSA_PROD(float);
REGISTER_MUSA_PROD(double);
REGISTER_MUSA_PROD(int32);
REGISTER_MUSA_PROD(int64);
REGISTER_MUSA_PROD(Eigen::half);
REGISTER_MUSA_PROD(Eigen::bfloat16);

}  // namespace musa
}  // namespace tensorflow
