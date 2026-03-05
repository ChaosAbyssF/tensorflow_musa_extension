#include <cstdint>
#include <limits>

#include <musa_runtime.h>

#include "mu/device/musa_memset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "../utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T, typename TIndex>
void LaunchIsNonZeroCount(const T* input, TIndex* output, int n,
                          musaStream_t stream);

REGISTER_OP("MusaIsNonZeroCount")
    .Input("input: T")
    .Output("count: Tidx")
    .Attr("T: {float, double, half, bfloat16, int32, int64, bool}")
    .Attr("Tidx: {int32}")
    .SetShapeFn(shape_inference::ScalarShape);

template <typename T, typename TIndex>
class MusaIsNonZeroCountOp : public MusaOpKernel {
 public:
  explicit MusaIsNonZeroCountOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));

    const int64_t element_count = input.NumElements();
    const int64_t max_elements = std::numeric_limits<int>::max();
    OP_REQUIRES(
      ctx, element_count <= max_elements,
      errors::InvalidArgument(strings::StrCat(
        "MUSA IsNonZeroCount supports at most ", max_elements, " elements.")));
    const int elements = static_cast<int>(element_count);

    const musaStream_t stream = GetMusaStreamByCtx(ctx);
    const size_t output_bytes = output->TotalBytes();
    mStatus memset_status = MemsetAsync(
        const_cast<char*>(output->tensor_data().data()), 0, output_bytes, stream);
    OP_REQUIRES(ctx, memset_status == mStatus::SUCCESS,
                errors::Internal("MUSA IsNonZeroCount failed to clear the output tensor."));

    if (elements == 0) {
      return;
    }

    LaunchIsNonZeroCount<T, TIndex>(
        input.flat<T>().data(), output->flat<TIndex>().data(), elements, stream);
  }
};

#define REGISTER_MUSA_IS_NON_ZERO_COUNT(TYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("MusaIsNonZeroCount")                          \
                              .Device("MUSA")                                \
                              .TypeConstraint<TYPE>("T")                     \
                              .TypeConstraint<int32>("Tidx"),                 \
                          MusaIsNonZeroCountOp<TYPE, int32>)

REGISTER_MUSA_IS_NON_ZERO_COUNT(float);
REGISTER_MUSA_IS_NON_ZERO_COUNT(double);
REGISTER_MUSA_IS_NON_ZERO_COUNT(Eigen::half);
REGISTER_MUSA_IS_NON_ZERO_COUNT(bfloat16);
REGISTER_MUSA_IS_NON_ZERO_COUNT(int32);
REGISTER_MUSA_IS_NON_ZERO_COUNT(int64);
REGISTER_MUSA_IS_NON_ZERO_COUNT(bool);

#undef REGISTER_MUSA_IS_NON_ZERO_COUNT

}  // namespace musa
}  // namespace tensorflow
