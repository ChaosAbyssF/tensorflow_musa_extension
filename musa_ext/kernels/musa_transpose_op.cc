/* Copyright @2020-2022 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved. */

#include "utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/bfloat16.h"

#include <mudnn.h>
#include <vector>

namespace tensorflow {
namespace musa {

template <typename T>
class MusaTransposeOp : public MusaOpKernel {
 public:
  explicit MusaTransposeOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // 调试日志 (可选)
    // fprintf(stderr, ">>> [MUSA_TRACE_AUTO] %s\n", name().c_str());

    const Tensor& input = ctx->input(0);
    const Tensor& perm_tensor = ctx->input(1);
    const int dims = input.dims();

    // ============================================================
    // 1. 校验 perm 参数的合法性
    // ============================================================
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(perm_tensor.shape()),
                errors::InvalidArgument("perm must be rank 1"));
    OP_REQUIRES(ctx, dims == perm_tensor.NumElements(),
                errors::InvalidArgument("transpose expects a vector of size ", input.dims(),
                                        ". But input(1) is a vector of size ",
                                        perm_tensor.NumElements()));

    // 获取 perm 数据 (支持 int32 或 int64)
    std::vector<int64_t> permutation_64;
    permutation_64.reserve(dims);
    
    // 用于检查是否有重复维度
    std::vector<bool> bits(dims, false);
    bool is_identity = true;
    TensorShape output_shape;

    // 统一转为 int64 处理
    auto process_perm = [&](int64_t d, int i) {
      OP_REQUIRES(ctx, d >= 0 && d < dims,
                  errors::InvalidArgument(d, " is out of range [0 .. ", dims, ")"));
      OP_REQUIRES(ctx, !bits[d],
                  errors::InvalidArgument(d, " is duplicated in the permutation."));
      
      bits[d] = true;
      permutation_64.push_back(d);
      output_shape.AddDim(input.dim_size(d));
      
      if (d != i) {
        is_identity = false;
      }
    };

    if (perm_tensor.dtype() == DT_INT32) {
      auto Vperm = perm_tensor.vec<int32>();
      for (int i = 0; i < dims; ++i) process_perm(Vperm(i), i);
    } else {
      auto Vperm = perm_tensor.vec<int64_t>();
      for (int i = 0; i < dims; ++i) process_perm(Vperm(i), i);
    }
    
    // 如果上面的 OP_REQUIRES 失败，直接返回
    if (!ctx->status().ok()) return;

    // ============================================================
    // 2. 零拷贝优化 (Zero Copy Optimization)
    // ============================================================
    // 如果是恒等变换(perm=[0,1,2...])，或者维度<=1，直接透传输入到输出
    // 注意：NCHW <-> NHWC 转换时 is_identity 会是 false，所以不会误触这里
    if (dims <= 1 || is_identity) {
      ctx->set_output(0, input);
      return;
    }

    // ============================================================
    // 3. 分配输出显存
    // ============================================================
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) return;

    // ============================================================
    // 4. 调用 muDNN 执行转置
    // ============================================================
    mHandle& h = GetHandleByCtx(ctx);

    // 【修正】：直接使用 format_，不要手动强制切换 NCHW/NHWC
    // Transpose 的物理搬运会由 permutation_64 参数控制，不需要欺骗 mTensor
    mTensor in_mt = CreateMTensor(input, format_);
    mTensor out_mt = CreateMTensor(*output, format_);

    ::musa::dnn::Permute pop;

    // 配置维度和步长
    if (::musa::dnn::Status::SUCCESS != 
        pop.ConfigDimStride(out_mt, in_mt, 
                            static_cast<int>(permutation_64.size()), 
                            permutation_64.data())) {
      ctx->CtxFailure(errors::Internal("muDNN Permute ConfigDimStride failed!"));
      return;
    }

    // 执行计算
    if (::musa::dnn::Status::SUCCESS != pop.Run(h, out_mt, in_mt)) {
      ctx->CtxFailure(errors::Internal("muDNN Permute Run failed!"));
      return;
    }
  }
};

// ============================================================
// 5. 算子注册
// ============================================================
#define REGISTER_MUSA_TRANSPOSE(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(Name("Transpose")                               \
                              .Device("MUSA")                             \
                              .TypeConstraint<TYPE>("T")                  \
                              .HostMemory("perm"),                        \
                          MusaTransposeOp<TYPE>);

// 注册常用数据类型
REGISTER_MUSA_TRANSPOSE(float);
REGISTER_MUSA_TRANSPOSE(double);
REGISTER_MUSA_TRANSPOSE(Eigen::half);
REGISTER_MUSA_TRANSPOSE(bfloat16);
REGISTER_MUSA_TRANSPOSE(int32);
REGISTER_MUSA_TRANSPOSE(int64);
REGISTER_MUSA_TRANSPOSE(bool); // 很多时候 mask 也需要转置

#undef REGISTER_MUSA_TRANSPOSE

}  // namespace musa
}  // namespace tensorflow
