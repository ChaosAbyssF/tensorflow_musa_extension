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

    useKernel(ctx, a, b, c, alpha, beta, gamma, output);
  }

 private:
  void useKernel(OpKernelContext* ctx, const Tensor& a, const Tensor& b,
                 const Tensor& c, const Tensor& alpha, const Tensor& beta,
                 const Tensor& gamma, Tensor* output) {
    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    LaunchWeightedSum3Kernel<T>(
        a.flat<T>().data(), b.flat<T>().data(), c.flat<T>().data(),
        alpha.flat<T>().data(), beta.flat<T>().data(), gamma.flat<T>().data(),
        output->flat<T>().data(), a.NumElements(), stream);
  }

  // void useMudnn(OpKernelContext* ctx, const Tensor& a, const Tensor& b,
  //               const Tensor& c, const Tensor& alpha, const Tensor& beta,
  //               const Tensor& gamma, Tensor* output) {
  //   auto& handle = GetHandleByCtx(ctx);
  //   mTensor a_mt = CreateMTensor(a);
  //   mTensor b_mt = CreateMTensor(b);
  //   mTensor c_mt = CreateMTensor(c);

  //   Tensor* temp;
  //   OP_REQUIRES_OK(ctx, ctx->allocate_output(0, a.shape(), &temp));
  //   mTensor temp_mt = CreateMTensor(*temp);

  //   // Perform output = alpha * a + beta * b + gamma * c
  //   mBinary op;
  //   MTOP_CHECK_OK(op.SetMode(mBinary::Mode::ADD), "Set Add Mode", ctx);
  //   MTOP_CHECK_OK(op.SetAlpha(alpha.flat<T>().data()[0]), "Set Alpha", ctx);
  //   MTOP_CHECK_OK(op.SetBeta(beta.flat<T>().data()[0]), "Set Beta", ctx);
  //   MTOP_CHECK_OK_RUN(op.Run(handle, temp_mt, a_mt, b_mt), "Add Forward Run",
  //                     ctx);

  //   mTensor output_mt = CreateMTensor(*output);
  //   MTOP_CHECK_OK(op.SetAlpha(gamma.flat<T>().data()[0]), "Set Gamma", ctx);
  //   MTOP_CHECK_OK_RUN(op.Run(handle, output_mt, temp_mt, c_mt),
  //                     "Add Gamma Forward Run", ctx);
  // }
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
