#ifndef MUSA_PLUGIN_SRC_KERNELS_MUSA_REDUCE_FUNCTOR_H_
#define MUSA_PLUGIN_SRC_KERNELS_MUSA_REDUCE_FUNCTOR_H_

#include <functional>
#include <memory>

#include "musa_cast_functor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

namespace internal_reduce {

// Build a MemoryMaintainer backed by the OpKernelContext's allocator. Shared
// by all reduce paths (direct, fp32-promoted bf16, and the legacy
// ReduceFunctor::Compute).
inline ::musa::dnn::MemoryMaintainer MakeMemMaintainer(OpKernelContext* ctx) {
  tensorflow::Allocator* tf_allocator =
      ctx->device()->GetAllocator(tensorflow::AllocatorAttributes());
  auto alloc_func =
      [tf_allocator](
          size_t size) -> std::unique_ptr<void, std::function<void(void*)>> {
    void* ptr = tf_allocator->AllocateRaw(256, size);
    std::function<void(void*)> deleter = [tf_allocator](void* p) {
      if (p) tf_allocator->DeallocateRaw(p);
    };
    return std::unique_ptr<void, std::function<void(void*)>>(ptr, deleter);
  };
  return ::musa::dnn::MemoryMaintainer(alloc_func);
}

// Run muDNN Cast on mTensors backed by the given Tensors. The destination
// Tensor must already be allocated to the same shape with the target dtype.
inline Status CastTensor(OpKernelContext* ctx, const Tensor& src, Tensor* dst,
                         const char* op_label) {
  auto& handle = GetHandleByCtx(ctx);
  mTensor src_mt = CreateMTensor(src);
  mTensor dst_mt = CreateMTensor(*dst);

  ::musa::dnn::Unary op;
  auto status = op.SetMode(::musa::dnn::Unary::Mode::CAST);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return errors::Internal(op_label, " SetMode failed. Status: ",
                            static_cast<int>(status));
  }
  status = op.Run(handle, dst_mt, src_mt);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return errors::Internal(op_label, " Run failed. Status: ",
                            static_cast<int>(status));
  }
  return Status();
}

}  // namespace internal_reduce

// Run muDNN Reduce with fp32 promotion for bf16 inputs.
//
// bf16 reductions through muDNN accumulate in bf16, which loses precision
// after only a few thousand contributions (7-bit mantissa). Stock TensorFlow
// promotes bf16 reductions to fp32 internally for exactly this reason. This
// helper applies the same promotion on the MUSA path:
//
//   bf16 input -> Cast(fp32) -> Reduce(fp32) -> Cast(bf16) -> bf16 output
//
// Non-bf16 inputs (fp32, fp64, fp16, int*) skip the promotion and run the
// reduce directly. fp16 is *not* promoted by default — historically muDNN's
// half reduction is acceptable, and the memory cost of an fp32 input copy
// would be doubled for the same shape.
//
// Args:
//   input            - source Tensor (any reduce-compatible dtype).
//   output_reshaped  - destination Tensor, already sized so that reduced
//                      dimensions are explicit "1"s (matches muDNN's
//                      expected output layout). Typically constructed via
//                      Tensor::CopyFrom(*real_output, reshape_with_ones)
//                      by the caller so the actual output buffer is shared.
//   mode             - muDNN Reduce mode (ADD/MEAN/PROD/MIN/MAX/AND/OR/...).
//   reduce_dims      - dimension indices to reduce over.
//   reduce_dim_count - length of reduce_dims.
//   error_prefix     - prefix string for error messages.
inline Status RunReduceWithFP32Promotion(OpKernelContext* ctx,
                                         const Tensor& input,
                                         Tensor* output_reshaped,
                                         ::musa::dnn::Reduce::Mode mode,
                                         const int* reduce_dims,
                                         int reduce_dim_count,
                                         const char* error_prefix) {
  auto& handle = GetHandleByCtx(ctx);
  ::musa::dnn::MemoryMaintainer mm = internal_reduce::MakeMemMaintainer(ctx);

  // Direct path for non-bf16 inputs.
  if (input.dtype() != DT_BFLOAT16) {
    mTensor in_mt = CreateMTensor(input);
    mTensor out_mt = CreateMTensor(*output_reshaped);
    mReduce op;
    op.SetMode(mode);
    op.SetDim(reduce_dim_count, reduce_dims);
    auto status = op.Run(handle, out_mt, in_mt, mm);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal(error_prefix, static_cast<int>(status));
    }
    return Status();
  }

  // bf16 promotion path: input -> Cast -> fp32 temp -> Reduce(fp32) ->
  // fp32 temp -> Cast -> bf16 output. Memory cost is one fp32-sized input
  // copy plus one fp32-sized output copy, freed when this function returns
  // because ctx->allocate_temp tensors are scoped to the op call.
  Tensor input_fp32;
  TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_FLOAT, input.shape(), &input_fp32));
  Tensor output_fp32;
  TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_FLOAT, output_reshaped->shape(),
                                        &output_fp32));

  TF_RETURN_IF_ERROR(internal_reduce::CastTensor(ctx, input, &input_fp32,
                                                  "Reduce bf16->fp32 cast"));

  {
    mTensor in_mt = CreateMTensor(input_fp32);
    mTensor out_mt = CreateMTensor(output_fp32);
    mReduce op;
    op.SetMode(mode);
    op.SetDim(reduce_dim_count, reduce_dims);
    auto status = op.Run(handle, out_mt, in_mt, mm);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal(error_prefix, "(fp32 reduce) ",
                              static_cast<int>(status));
    }
  }

  return internal_reduce::CastTensor(ctx, output_fp32, output_reshaped,
                                      "Reduce fp32->bf16 cast");
}

struct ReduceFunctor {
  template <typename T>
  static Status Compute(OpKernelContext* ctx, mTensor* output, mTensor* input,
                        ::musa::dnn::Reduce::Mode mode, const int* reduce_dims,
                        int reduce_dim_count, const char* error_prefix) {
    auto& handle = GetHandleByCtx(ctx);

    mReduce op;
    op.SetMode(mode);
    op.SetDim(reduce_dim_count, reduce_dims);

    ::musa::dnn::MemoryMaintainer mm = internal_reduce::MakeMemMaintainer(ctx);

    auto status = op.Run(handle, *output, *input, mm);
    if (status != ::musa::dnn::Status::SUCCESS) {
      return errors::Internal(error_prefix, static_cast<int>(status));
    }
    return Status();
  }
};

// bf16 specialization: the legacy version in this file took bare mTensors
// without backing storage and called CastFunctor / Compute<float> on them,
// which doesn't work — CastFunctor never allocates GPU memory and the
// recursive Compute<float> call receives uninitialized descriptors. The
// surface-level "bf16-as-fp32" intent was never actually exercised.
//
// Callers that want fp32-promoted bf16 reductions should prefer the new
// RunReduceWithFP32Promotion helper above, which takes Tensors directly so
// it can allocate fp32 temporaries with the correct shape. This
// specialization is kept (and now correctly implemented) so existing call
// sites that pass mTensors through ReduceFunctor::Compute<bfloat16>
// (musa_all_op.cc, musa_einsum_op.cc) at least get the muDNN-native bf16
// reduce instead of crashing. They should be migrated to the helper if
// fp32 precision matters for those ops.
template <>
inline Status ReduceFunctor::Compute<bfloat16>(
    OpKernelContext* ctx, mTensor* output_mt, mTensor* input_mt,
    ::musa::dnn::Reduce::Mode mode, const int* reduce_dims,
    int reduce_dim_count, const char* error_prefix) {
  auto& handle = GetHandleByCtx(ctx);
  ::musa::dnn::MemoryMaintainer mm = internal_reduce::MakeMemMaintainer(ctx);

  mReduce op;
  op.SetMode(mode);
  op.SetDim(reduce_dim_count, reduce_dims);

  auto status = op.Run(handle, *output_mt, *input_mt, mm);
  if (status != ::musa::dnn::Status::SUCCESS) {
    return errors::Internal(error_prefix, "(bf16 reduce) ",
                            static_cast<int>(status));
  }
  return Status();
}

}  // namespace musa
}  // namespace tensorflow

#endif  // MUSA_PLUGIN_SRC_KERNELS_MUSA_REDUCE_FUNCTOR_H_