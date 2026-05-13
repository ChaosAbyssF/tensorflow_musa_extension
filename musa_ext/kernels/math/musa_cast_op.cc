#include <cstdlib>
#include <string>

#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"

// Vectorized fast-path launchers for the four mixed-precision Cast dtype
// pairs that dominate Keras mixed_bfloat16 / mixed_float16 training. Defined
// in musa_cast_kernels.mu; the .mu side handles intrinsics and uint4
// reinterprets so this .cc does not need to pull in __mt_bfloat16 / __half
// headers.
extern "C" {
void LaunchMusaCastBfloat16ToFloat32(const void* src, void* dst, int64_t n,
                                      musaStream_t stream);
void LaunchMusaCastFloat32ToBfloat16(const void* src, void* dst, int64_t n,
                                      musaStream_t stream);
void LaunchMusaCastHalfToFloat32(const void* src, void* dst, int64_t n,
                                  musaStream_t stream);
void LaunchMusaCastFloat32ToHalf(const void* src, void* dst, int64_t n,
                                  musaStream_t stream);
}

namespace tensorflow {
namespace musa {

namespace {

inline bool UseCastCustomKernelFastPath() {
  const char* env = std::getenv("MUSA_CAST_ENABLE_CUSTOM_KERNEL");
  if (env == nullptr || std::string(env).empty()) return true;
  const std::string value(env);
  return !(value == "0" || value == "false" || value == "FALSE" ||
           value == "off" || value == "OFF" || value == "no" || value == "NO");
}

// Returns true if the (src, dst) dtype pair has a vectorized fast-path
// kernel and we successfully dispatched it. Returns false to let the caller
// fall back to the generic muDNN CAST.
bool TryLaunchMusaCastFastPath(OpKernelContext* ctx, DataType src_dtype,
                                DataType dst_dtype, const Tensor& inp,
                                Tensor* output) {
  if (!UseCastCustomKernelFastPath()) return false;
  musaStream_t stream = GetMusaStreamByCtx(ctx);
  if (stream == nullptr) return false;

  const int64_t n = inp.NumElements();
  const void* src = inp.tensor_data().data();
  void* dst =
      const_cast<void*>(static_cast<const void*>(output->tensor_data().data()));

  if (src_dtype == DT_BFLOAT16 && dst_dtype == DT_FLOAT) {
    LaunchMusaCastBfloat16ToFloat32(src, dst, n, stream);
  } else if (src_dtype == DT_FLOAT && dst_dtype == DT_BFLOAT16) {
    LaunchMusaCastFloat32ToBfloat16(src, dst, n, stream);
  } else if (src_dtype == DT_HALF && dst_dtype == DT_FLOAT) {
    LaunchMusaCastHalfToFloat32(src, dst, n, stream);
  } else if (src_dtype == DT_FLOAT && dst_dtype == DT_HALF) {
    LaunchMusaCastFloat32ToHalf(src, dst, n, stream);
  } else {
    return false;
  }

  const musaError_t launch_status = musaGetLastError();
  if (launch_status != musaSuccess) {
    ctx->CtxFailure(errors::Internal("MUSA Cast fast path launch failed: ",
                                      musaGetErrorString(launch_status)));
    // Returning true here means "do not fall through" — we already set a
    // failure on ctx. Returning false would re-invoke muDNN on top of an
    // error which doesn't help anyone.
    return true;
  }
  return true;
}

}  // namespace

class MusaCastOp : public MusaOpKernel {
 public:
  explicit MusaCastOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("SrcT", &external_src_dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("DstT", &external_dst_dtype_));
    // Cache identity check for zero-copy fast path (matches TensorFlow's
    // CastOpBase)
    is_identity_cast_ = (external_src_dtype_ == external_dst_dtype_);
  }

  // Cast is element-wise - lightweight
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inp = ctx->input(0);

    // Zero-copy fast path for identity cast (SrcT == DstT)
    // This matches TensorFlow's official CastOpBase behavior:
    // - Uses reference counting to manage shared buffer
    // - TensorFlow's runtime ensures safety via copy-on-write semantics
    if (is_identity_cast_) {
      ctx->set_output(0, inp);
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, inp.shape(), &output));

    if (inp.NumElements() == 0) {
      // No need to run muDNN for empty tensors. Just return the zero-element
      // output tensor (already allocated above).
      return;
    }

    // Vectorized custom fast path for the bf16<->fp32 and fp16<->fp32 dtype
    // pairs. These four casts dominate mixed-precision training (every
    // dtype boundary in Keras mixed_bfloat16 / mixed_float16 issues one),
    // so taking them out of muDNN avoids one Unary descriptor setup per
    // boundary on top of cutting the inner-loop work to a single uint4
    // load + store per 8 elements.
    if (TryLaunchMusaCastFastPath(ctx, external_src_dtype_, external_dst_dtype_,
                                    inp, output)) {
      return;
    }

    auto in_mt = CreateMTensor(inp);
    auto out_mt = CreateMTensor(*output);

    // BOOL format workaround for muDNN
    if (inp.dtype() == DT_BOOL) {
      in_mt.SetFormat(mFormat::NCHW);
    }

    mHandle& h = GetHandleByCtx(ctx);
    ::musa::dnn::Unary op;

    auto m_status = op.SetMode(::musa::dnn::Unary::Mode::CAST);
    OP_REQUIRES(ctx, m_status == mStatus::SUCCESS,
                errors::Internal("muDNN Unary SetMode failed in Cast"));

    m_status = op.Run(h, out_mt, in_mt);

    if (m_status != mStatus::SUCCESS) {
      LOG(ERROR) << "MUSA Cast Run failed! Src: "
                 << DataTypeString(external_src_dtype_)
                 << " -> Dst: " << DataTypeString(external_dst_dtype_)
                 << " | Status: " << static_cast<int>(m_status);

      ctx->SetStatus(errors::Internal("MUSA Cast Run failed. Status code: ",
                                      static_cast<int>(m_status)));
      return;
    }
  }

 private:
  DataType external_src_dtype_;
  DataType external_dst_dtype_;
  bool is_identity_cast_;  // Cached flag for zero-copy optimization
};

#define REGISTER_CAST_MUSA(SrcT, DstT)                       \
  REGISTER_KERNEL_BUILDER(Name("Cast")                       \
                              .Device(DEVICE_MTGPU)          \
                              .TypeConstraint<SrcT>("SrcT")  \
                              .TypeConstraint<DstT>("DstT"), \
                          MusaCastOp);

REGISTER_CAST_MUSA(bool, bool);
REGISTER_CAST_MUSA(bool, int32);
REGISTER_CAST_MUSA(bool, int64);
REGISTER_CAST_MUSA(bool, Eigen::half);
REGISTER_CAST_MUSA(bool, bfloat16);
REGISTER_CAST_MUSA(bool, float);
REGISTER_CAST_MUSA(bool, double);

REGISTER_CAST_MUSA(int32, bool);
REGISTER_CAST_MUSA(int32, int32);
REGISTER_CAST_MUSA(int32, int64);
REGISTER_CAST_MUSA(int32, Eigen::half);
REGISTER_CAST_MUSA(int32, bfloat16);
REGISTER_CAST_MUSA(int32, float);
REGISTER_CAST_MUSA(int32, double);

REGISTER_CAST_MUSA(int64, bool);
REGISTER_CAST_MUSA(int64, int32);
REGISTER_CAST_MUSA(int64, int64);
REGISTER_CAST_MUSA(int64, Eigen::half);
REGISTER_CAST_MUSA(int64, bfloat16);
REGISTER_CAST_MUSA(int64, float);
REGISTER_CAST_MUSA(int64, double);

REGISTER_CAST_MUSA(Eigen::half, bool);
REGISTER_CAST_MUSA(Eigen::half, int32);
REGISTER_CAST_MUSA(Eigen::half, int64);
REGISTER_CAST_MUSA(Eigen::half, Eigen::half);
REGISTER_CAST_MUSA(Eigen::half, bfloat16);
REGISTER_CAST_MUSA(Eigen::half, float);
REGISTER_CAST_MUSA(Eigen::half, double);

REGISTER_CAST_MUSA(bfloat16, bool);
REGISTER_CAST_MUSA(bfloat16, int32);
REGISTER_CAST_MUSA(bfloat16, int64);
REGISTER_CAST_MUSA(bfloat16, Eigen::half);
REGISTER_CAST_MUSA(bfloat16, bfloat16);
REGISTER_CAST_MUSA(bfloat16, float);
REGISTER_CAST_MUSA(bfloat16, double);

REGISTER_CAST_MUSA(float, bool);
REGISTER_CAST_MUSA(float, int32);
REGISTER_CAST_MUSA(float, int64);
REGISTER_CAST_MUSA(float, Eigen::half);
REGISTER_CAST_MUSA(float, bfloat16);
REGISTER_CAST_MUSA(float, float);
REGISTER_CAST_MUSA(float, double);

REGISTER_CAST_MUSA(double, bool);
REGISTER_CAST_MUSA(double, int32);
REGISTER_CAST_MUSA(double, int64);
REGISTER_CAST_MUSA(double, Eigen::half);
REGISTER_CAST_MUSA(double, bfloat16);
REGISTER_CAST_MUSA(double, float);
REGISTER_CAST_MUSA(double, double);

}  // namespace musa
}  // namespace tensorflow
