#include "../utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
void LaunchWeightedSum3Kernel(const T* a, const T* b, const T* c,
                              const T* alpha, const T* beta, const T* gamma,
                              T* output, int num_elements, musaStream_t stream);

template <typename T>
class MusaWeightedSum3Op : public MusaOpKernel {
 public:
  explicit MusaWeightedSum3Op(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);
    const Tensor& c = ctx->input(2);
    const Tensor& alpha = ctx->input(3);
    const Tensor& beta = ctx->input(4);
    const Tensor& gamma = ctx->input(5);

    OP_REQUIRES(ctx, alpha.NumElements() == 1,
                errors::InvalidArgument("alpha must be a scalar"));
    OP_REQUIRES(ctx, beta.NumElements() == 1,
                errors::InvalidArgument("beta must be a scalar"));
    OP_REQUIRES(ctx, gamma.NumElements() == 1,
                errors::InvalidArgument("gamma must be a scalar"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, a.shape(), &output));

    if (a.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchWeightedSum3Kernel<T>(
        a.flat<T>().data(), b.flat<T>().data(), c.flat<T>().data(),
        alpha.flat<T>().data(), beta.flat<T>().data(), gamma.flat<T>().data(),
        output->flat<T>().data(), a.NumElements(), stream);
  }
};

#define REGISTER_WEIGHTED_SUM3_KERNEL(TYPE)                              \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("MusaWeightedSum3").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaWeightedSum3Op<TYPE>);

REGISTER_WEIGHTED_SUM3_KERNEL(float);
REGISTER_WEIGHTED_SUM3_KERNEL(Eigen::half);
REGISTER_WEIGHTED_SUM3_KERNEL(bfloat16);
REGISTER_WEIGHTED_SUM3_KERNEL(double);

}  // namespace musa
}  // namespace tensorflow