#include <mudnn.h>
#include <mudnn_xmma.h>
#include <musa_runtime.h>

#include "../utils_op.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {

// MusaPerTokenFFN: performs BatchMatMul(A, W0) + BiasAdd0 + Gelu +
// BatchMatMul(..., W1) + BiasAdd1
template <typename T>
class MusaPerTokenFFNOp : public MusaOpKernel {
 public:
  explicit MusaPerTokenFFNOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    MUSA_KERNEL_TIMING_GUARD(ctx);

    const Tensor& x = ctx->input(0);
    const Tensor& w0 = ctx->input(1);
    const Tensor& b0 = ctx->input(2);
    const Tensor& w1 = ctx->input(3);
    const Tensor& b1 = ctx->input(4);

    // Validate shapes via MatMul broadcast helper for the two matmuls
    MatMulBCast bcast0(x.shape().dim_sizes(), w0.shape().dim_sizes());
    OP_REQUIRES(ctx, bcast0.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes for BMM0: ", x.shape().DebugString(),
                    " vs ", w0.shape().DebugString()));

    // Output shape after first matmul
    TensorShape mm0_shape = bcast0.output_batch_shape();
    mm0_shape.AddDim(x.dim_size(x.dims() - 2));
    mm0_shape.AddDim(w0.dim_size(w0.dims() - 1));

    // Temp tensor for first matmul
    Tensor mm0_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(x.dtype(), mm0_shape, &mm0_tensor));

    if (mm0_tensor.NumElements() == 0) {
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, mm0_shape, &out));
      return;
    }

    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(false);
    mTensor mt_x = CreateMTensor(x);
    mTensor mt_w0 = CreateMTensor(w0);
    mTensor mt_mm0 = CreateMTensor(mm0_tensor);

    ::musa::dnn::Status status;

    // Use MatMul or BatchMatMul
    if (x.dims() == 2 && w0.dims() == 2) {
      mMatMul op;
      op.SetTranspose(false, false);
      op.SetAlpha(1.0);
      op.SetBeta(0.0);
      status = op.Run(handle, mt_mm0, mt_x, mt_w0);
    } else {
      mBatchMatMul op;
      op.SetTranspose(false, false);
      op.SetAlpha(1.0);
      op.SetBeta(0.0);
      int64_t out_batch = bcast0.output_batch_shape().num_elements();

      auto ReshapeTo3D = [out_batch](mTensor& mt, const Tensor& t) {
        int64_t dims = t.dims();
        int64_t rows = t.dim_size(dims - 2);
        int64_t cols = t.dim_size(dims - 1);
        int64_t batch = t.NumElements() / (rows * cols);
        if (dims != 3 || (batch == 1 && out_batch > 1)) {
          mt.SetNdInfo(
              {batch == 1 && out_batch > 1 ? out_batch : batch, rows, cols},
              {batch == 1 && out_batch > 1 ? 0 : rows * cols, cols, 1});
        }
      };
      ReshapeTo3D(mt_x, x);
      ReshapeTo3D(mt_w0, w0);
      mt_mm0.SetNdInfo(
          {out_batch, x.dim_size(x.dims() - 2), w0.dim_size(w0.dims() - 1)},
          {x.dim_size(x.dims() - 2) * w0.dim_size(w0.dims() - 1),
           w0.dim_size(w0.dims() - 1), 1});
      status = op.Run(handle, mt_mm0, mt_x, mt_w0);
    }

    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal(
            "MUSA BatchMatMul execution failed in PerTokenFFN (BMM0)."));

    // BiasAdd0
    Tensor mm0_with_bias;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(x.dtype(), mm0_shape, &mm0_with_bias));
    mTensor mt_bias0 = CreateMTensor(b0);
    mTensor mt_mm0_with_bias = CreateMTensor(mm0_with_bias);

    // Prepare bias ndinfo
    int channel_dim = mm0_shape.dims() - 1;
    int dims_cnt = mm0_shape.dims();
    std::vector<int64_t> b_dims(dims_cnt, 1);
    std::vector<int64_t> b_strides(dims_cnt, 0);
    b_dims[channel_dim] = b0.dim_size(0);
    b_strides[channel_dim] = 1;
    mt_bias0.SetNdInfo(dims_cnt, b_dims.data(), b_strides.data());

    mBinary bias_op;
    bias_op.SetMode(::musa::dnn::Binary::Mode::ADD);
    status = bias_op.Run(handle, mt_mm0_with_bias, mt_mm0, mt_bias0);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA BiasAdd0 failed in PerTokenFFN."));

    // GELU
    mUnary gelu_op;
    gelu_op.SetMode(UNARY_MODE::GELU);
    mTensor mt_gelu_out = CreateMTensor(mm0_with_bias);
    status = gelu_op.Run(handle, mt_gelu_out, mt_mm0_with_bias);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA GELU failed in PerTokenFFN."));

    // Second BMM: gelu_out * w1
    Tensor mm1_tensor;
    MatMulBCast bcast1(mm0_with_bias.shape().dim_sizes(),
                       w1.shape().dim_sizes());
    OP_REQUIRES(ctx, bcast1.IsValid(),
                errors::InvalidArgument("Incompatible shapes for BMM1: ",
                                        mm0_with_bias.shape().DebugString(),
                                        " vs ", w1.shape().DebugString()));

    TensorShape mm1_shape = bcast1.output_batch_shape();
    mm1_shape.AddDim(mm0_with_bias.dim_size(mm0_with_bias.dims() - 2));
    mm1_shape.AddDim(w1.dim_size(w1.dims() - 1));

    OP_REQUIRES_OK(ctx, ctx->allocate_temp(x.dtype(), mm1_shape, &mm1_tensor));
    mTensor mt_mm1 = CreateMTensor(mm1_tensor);
    mTensor mt_gelu = CreateMTensor(mm0_with_bias);
    mTensor mt_w1 = CreateMTensor(w1);

    if (mm0_with_bias.dims() == 2 && w1.dims() == 2) {
      mMatMul op;
      op.SetTranspose(false, false);
      op.SetAlpha(1.0);
      op.SetBeta(0.0);
      status = op.Run(handle, mt_mm1, mt_gelu, mt_w1);
    } else {
      mBatchMatMul op;
      op.SetTranspose(false, false);
      op.SetAlpha(1.0);
      op.SetBeta(0.0);
      int64_t out_batch = bcast1.output_batch_shape().num_elements();
      auto ReshapeTo3D = [out_batch](mTensor& mt, const Tensor& t) {
        int64_t dims = t.dims();
        int64_t rows = t.dim_size(dims - 2);
        int64_t cols = t.dim_size(dims - 1);
        int64_t batch = t.NumElements() / (rows * cols);
        if (dims != 3 || (batch == 1 && out_batch > 1)) {
          mt.SetNdInfo(
              {batch == 1 && out_batch > 1 ? out_batch : batch, rows, cols},
              {batch == 1 && out_batch > 1 ? 0 : rows * cols, cols, 1});
        }
      };
      ReshapeTo3D(mt_gelu, mm0_with_bias);
      ReshapeTo3D(mt_w1, w1);
      mt_mm1.SetNdInfo(
          {out_batch, mm0_with_bias.dim_size(mm0_with_bias.dims() - 2),
           w1.dim_size(w1.dims() - 1)},
          {mm0_with_bias.dim_size(mm0_with_bias.dims() - 2) *
               w1.dim_size(w1.dims() - 1),
           w1.dim_size(w1.dims() - 1), 1});
      status = op.Run(handle, mt_mm1, mt_gelu, mt_w1);
    }

    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal(
            "MUSA BatchMatMul execution failed in PerTokenFFN (BMM1)."));

    // BiasAdd1 -> final output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, mm1_shape, &output));
    mTensor mt_out = CreateMTensor(*output);
    mTensor mt_bias1 = CreateMTensor(b1);

    int channel_dim1 = mm1_shape.dims() - 1;
    int dims_cnt1 = mm1_shape.dims();
    std::vector<int64_t> b1_dims(dims_cnt1, 1);
    std::vector<int64_t> b1_strides(dims_cnt1, 0);
    b1_dims[channel_dim1] = b1.dim_size(0);
    b1_strides[channel_dim1] = 1;
    mt_bias1.SetNdInfo(dims_cnt1, b1_dims.data(), b1_strides.data());

    mBinary bias_op1;
    bias_op1.SetMode(::musa::dnn::Binary::Mode::ADD);
    status = bias_op1.Run(handle, mt_out, mt_mm1, mt_bias1);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA BiasAdd1 failed in PerTokenFFN."));
  }
};

#define REGISTER_MUSA_PER_TOKEN_FFN(TYPE)                               \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("MusaPerTokenFFN").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaPerTokenFFNOp<TYPE>);

REGISTER_MUSA_PER_TOKEN_FFN(float);
REGISTER_MUSA_PER_TOKEN_FFN(Eigen::half);
REGISTER_MUSA_PER_TOKEN_FFN(bfloat16);
REGISTER_MUSA_PER_TOKEN_FFN(double);

#undef REGISTER_MUSA_PER_TOKEN_FFN

}  // namespace musa

REGISTER_OP("MusaPerTokenFFN")
    .Input("x: T")
    .Input("w0: T")
    .Input("b0: T")
    .Input("w1: T")
    .Input("b1: T")
    .Output("y: T")
    .Attr("T: {float, double, half, bfloat16}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // Conservatively set output shape to input `x` with last dim unknown;
      c->set_output(0, c->input(0));
      return Status::OK();
    });

}  // namespace tensorflow
