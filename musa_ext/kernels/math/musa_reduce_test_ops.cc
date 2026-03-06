#include <mudnn.h>

#include "musa_reduce_functor.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "../utils_op.h"

// 1. ReduceTestOp
REGISTER_OP("ReduceTest")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {float, half, bfloat16}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
      ::tensorflow::shape_inference::DimensionHandle d0 = c->Dim(input, 0);
      c->set_output(0, c->MakeShape({d0}));
      return ::tensorflow::Status::OK();
    });

// 2. ReduceBmmTestOp
REGISTER_OP("ReduceBmmTest")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("T: {float, half, bfloat16}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle a_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a_shape));
      ::tensorflow::shape_inference::ShapeHandle b_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b_shape));
      c->set_output(0, c->MakeShape({c->Dim(a_shape, 0), c->Dim(b_shape, 1)}));
      return ::tensorflow::Status::OK();
    });

namespace tensorflow {
namespace musa {

template <typename T>
class ReduceTestOp : public MusaOpKernel {
 public:
  explicit ReduceTestOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    OP_REQUIRES(ctx, input.dims() == 2,
                errors::InvalidArgument("ReduceTest only supports 2D input"));

    TensorShape output_shape({input.dim_size(0)});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (input.NumElements() == 0) return;

    mTensor mt_input = CreateMTensor(input, format_);
    mTensor mt_output = CreateMTensor(*output, format_);

    int reduce_dims[] = {1};
    OP_REQUIRES_OK(ctx, ReduceFunctor::Compute<T>(
                            ctx, &mt_output, &mt_input,
                            ::musa::dnn::Reduce::Mode::ADD, reduce_dims, 1,
                            "ReduceTest execution failed"));
  }
};

template <typename T>
class ReduceBmmTestOp : public MusaOpKernel {
 public:
  explicit ReduceBmmTestOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);
    OP_REQUIRES(ctx, a.dims() == 2 && b.dims() == 2,
                errors::InvalidArgument("Inputs must be 2D"));
    OP_REQUIRES(ctx, b.dim_size(0) == 1,
                errors::InvalidArgument(
                    "Input B must have 1 row for this test op"));

    TensorShape reduced_shape({a.dim_size(0), 1});
    Tensor reduced_a;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(a.dtype(), reduced_shape, &reduced_a));

    mTensor mt_a = CreateMTensor(a, format_);
    mTensor mt_reduced_a = CreateMTensor(reduced_a, format_);

    int reduce_dims[] = {1};
    OP_REQUIRES_OK(ctx, ReduceFunctor::Compute<T>(
                            ctx, &mt_reduced_a, &mt_a,
                            ::musa::dnn::Reduce::Mode::ADD, reduce_dims, 1,
                            "ReduceBmmTest Reduce step failed"));

    TensorShape out_shape({a.dim_size(0), b.dim_size(1)});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    mTensor mt_b = CreateMTensor(b, format_);
    mTensor mt_output = CreateMTensor(*output, format_);

    mBatchMatMul matmul_op;
    auto status = matmul_op.Run(handle, mt_output, mt_reduced_a, mt_b);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("ReduceBmmTest BMM step failed. Status: ",
                                 (int)status));
  }
};

#define REGISTER_REDUCE_TEST(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("ReduceTest").Device("MUSA").TypeConstraint<TYPE>("T"),   \
      ::tensorflow::musa::ReduceTestOp<TYPE>);                      \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("ReduceTest").Device("CPU").TypeConstraint<TYPE>("T"),    \
      ::tensorflow::musa::ReduceTestOp<TYPE>)

REGISTER_REDUCE_TEST(float);
REGISTER_REDUCE_TEST(Eigen::half);
REGISTER_REDUCE_TEST(bfloat16);

#define REGISTER_REDUCE_BMM_TEST(TYPE)                               \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("ReduceBmmTest").Device("MUSA").TypeConstraint<TYPE>("T"), \
      ::tensorflow::musa::ReduceBmmTestOp<TYPE>);                    \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("ReduceBmmTest").Device("CPU").TypeConstraint<TYPE>("T"),  \
      ::tensorflow::musa::ReduceBmmTestOp<TYPE>)

REGISTER_REDUCE_BMM_TEST(float);
REGISTER_REDUCE_BMM_TEST(Eigen::half);
REGISTER_REDUCE_BMM_TEST(bfloat16);

}  // namespace musa
}  // namespace tensorflow
