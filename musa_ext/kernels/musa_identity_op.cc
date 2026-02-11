/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved. */

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/public/version.h"
#include "utils_op.h" 
#include "mu/device/musa_memcpy.h"

namespace tensorflow {
namespace musa {

// =================================================================
// 1. Identity / _Arg / _Retval 统一实现
// =================================================================
class MusaIdentityOp : public OpKernel {
 public:
  explicit MusaIdentityOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // 获取实际输出的数量
    int num_outputs = context->num_outputs();
    int num_inputs = context->num_inputs();

    if (num_outputs > 0) {
      // 只有在确实有输入时才调用 input(0)，防止 (0 vs 0) 崩溃
      if (num_inputs > 0) {
        for (int i = 0; i < num_outputs && i < num_inputs; ++i) {
          context->set_output(i, context->input(i));
        }
      } else {
        // 如果没有输入但有输出（常见于 _Arg），这通常由 TF 运行时自动填充输出
        // 我们在这里保持沉默，不触发 Check failed
        VLOG(1) << ">>>>> [MUSA] Identity/Arg with 0 inputs, skipping manual set_output";
      }
    }
  }
};





// 3.1 注册普通 Tensor 的 Identity, , _Retval
// =================================================================
// 3.1 注册普通 Tensor 的 Identity, IdentityN, _Retval
// =================================================================

// 注意：每一行的末尾都必须有反斜杠 \ 且后面不能有空格
#define REGISTER_MUSA_BASE_OPS(type)                                          \
  REGISTER_KERNEL_BUILDER(Name("Identity")                                    \
                              .Device("MUSA")                                 \
                              .TypeConstraint<type>("T"),                     \
                          MusaIdentityOp);                                    \
  REGISTER_KERNEL_BUILDER(Name("IdentityN")                                   \
                              .Device("MUSA")                                 \
                              .TypeConstraint<type>("T"),                     \
                          MusaIdentityOp);                                    \
  REGISTER_KERNEL_BUILDER(Name("Snapshot")                                    \
                              .Device("MUSA")                                 \
                              .TypeConstraint<type>("T"),                     \
                          MusaIdentityOp);                                 
// 此时再调用就不会报错了
REGISTER_MUSA_BASE_OPS(float);
REGISTER_MUSA_BASE_OPS(double);
REGISTER_MUSA_BASE_OPS(Eigen::half);
REGISTER_MUSA_BASE_OPS(int32);
REGISTER_MUSA_BASE_OPS(int64);
REGISTER_MUSA_BASE_OPS(bool);




} // namespace musa
} // namespace tensorflow



