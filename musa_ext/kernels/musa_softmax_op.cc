#include "utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace musa {

// 1. 实现部分
namespace {

class MusaSoftmaxCall : public MusaOpKernel {
 public:
  explicit MusaSoftmaxCall(OpKernelConstruction* context) : MusaOpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);

    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(logits_in.shape()),
                errors::InvalidArgument("logits must have >= 1 dimension, got ",
                                        logits_in.shape().DebugString()));

    Tensor* softmax_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->forward_input_or_allocate_output(
                       {0}, 0, logits_in.shape(), &softmax_out));

    if (logits_in.NumElements() == 0) return;

    auto in = CreateMTensor(logits_in, format_);
    auto out = CreateMTensor(*softmax_out, format_);

    auto& h = GetHandleByCtx(context);
    mSoftmax softmax_op;
    this->Operate(softmax_op);

    int axis = static_cast<int>(logits_in.dims() - 1);
    MTOP_CHECK_OK(softmax_op.SetDim(axis), "SetDim", context);
    MTOP_CHECK_OK(softmax_op.SetAlgorithm(mSoftmax::Algorithm::ACCURATE),
                  "SetAlgorithm", context);

    MTOP_CHECK_OK_RUN(softmax_op.Run(h, out, in), "RunSoftmax", context);
  }

  virtual void Operate(mSoftmax& op) = 0;
};

template <SOFTMAX_MODE m>
class MusaSoftmaxOp : public MusaSoftmaxCall {
 public:
  using MusaSoftmaxCall::MusaSoftmaxCall;
  void Operate(mSoftmax& op) override {
    op.SetMode(m);
  }
};

}  // namespace anonymous

// 2. 核心修改：模仿 Add 的宏定义，直接使用原生 REGISTER_KERNEL_BUILDER
// 这里的 Device("MUSA") 必须和你 Add 算子里的字符串完全一致
#define REGISTER_MUSA_SOFTMAX_TYPE(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(                                                      \
      Name("Softmax").Device("MUSA").TypeConstraint<TYPE>("T"),                 \
      MusaSoftmaxOp<SOFTMAX_MODE::SOFTMAX>);                                     \
  REGISTER_KERNEL_BUILDER(                                                      \
      Name("LogSoftmax").Device("MUSA").TypeConstraint<TYPE>("T"),              \
      MusaSoftmaxOp<SOFTMAX_MODE::LOGSOFTMAX>);

// 3. 像 Add 一样直接在全局/命名空间域注册，不包裹在任何自定义函数块里
REGISTER_MUSA_SOFTMAX_TYPE(float);
REGISTER_MUSA_SOFTMAX_TYPE(Eigen::half);
REGISTER_MUSA_SOFTMAX_TYPE(bfloat16);
REGISTER_MUSA_SOFTMAX_TYPE(double);
REGISTER_MUSA_SOFTMAX_TYPE(int32);
REGISTER_MUSA_SOFTMAX_TYPE(int64);

#undef REGISTER_MUSA_SOFTMAX_TYPE

}  // namespace musa
}  // namespace tensorflow
