/* Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
 * rights reserved.
 */

#include "utils_op.h" // 确保包含新版工具头文件
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include <vector>

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSplitOp : public OpKernel {
 public:
  explicit MusaSplitOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* context) override {
    // Split 的输入顺序是: 0: split_dim, 1: value
    const Tensor& split_dim_tensor = context->input(0);
    const Tensor& input = context->input(1);
    const TensorShape& input_shape = input.shape();
    const int32 num_split = context->num_outputs();

    // 校验 split_dim 是否为标量
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(split_dim_tensor.shape()),
                errors::InvalidArgument("split_dim must be a scalar, but got rank ",
                                        split_dim_tensor.shape().dims()));

    const int32 split_dim_orig = split_dim_tensor.flat<int32>()(0);
    const int32 split_dim = split_dim_orig < 0 ? split_dim_orig + input.dims() : split_dim_orig;

    OP_REQUIRES(context, 0 <= split_dim && split_dim < input.dims(),
                errors::InvalidArgument("Split dim ", split_dim_orig, " out of range"));

    OP_REQUIRES(context, num_split > 0,
                errors::InvalidArgument("Number of ways to split must be > 0"));

    const int64_t input_size_split_dim = input_shape.dim_size(split_dim);
    OP_REQUIRES(context, input_size_split_dim % num_split == 0,
                errors::InvalidArgument("Number of ways to split must evenly divide the split dimension"));

    // 特殊情况处理
    if (num_split == 1) {
      context->set_output(0, input);
      return;
    }

    const int64_t delta = input_size_split_dim / num_split;
    auto& h = GetHandleByCtx(context);
    ::musa::dnn::Permute op;

    // 准备 Slice 需要的参数
    std::vector<int64_t> starts_mt(input.dims(), 0);
    std::vector<int64_t> strides_mt(input.dims(), 1); // 默认步长为 1

    TensorShape out_shape = input_shape;
    out_shape.set_dim(split_dim, delta);

    for (int i = 0; i < num_split; ++i) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(i, out_shape, &output));

      if (delta == 0) continue;

      auto in_mt = CreateMTensor(input);
      auto out_mt = CreateMTensor(*output);

      // 设置当前分片的起始偏移
      starts_mt[split_dim] = i * delta;

      // 使用 Permute 算子实现 Slice 逻辑
      MTOP_CHECK_OK(op.ConfigDimStrideForSlice(out_mt, in_mt, starts_mt.data(), strides_mt.data()),
                    "ConfigDimStrideForSlice", context);
      
      MTOP_CHECK_OK_RUN(op.Run(h, out_mt, in_mt), "Split Run", context);
    }
  }
};

#define REGISTER_MUSA_SPLIT(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Split")                   \
                              .Device("MUSA")             \
                              .TypeConstraint<type>("T")  \
                              .HostMemory("split_dim"),   \
                          MusaSplitOp<type>)

// 注册常用类型
REGISTER_MUSA_SPLIT(float);
REGISTER_MUSA_SPLIT(double);
REGISTER_MUSA_SPLIT(Eigen::half);
REGISTER_MUSA_SPLIT(Eigen::bfloat16);
REGISTER_MUSA_SPLIT(int32);
REGISTER_MUSA_SPLIT(int64);
REGISTER_MUSA_SPLIT(bool);
REGISTER_MUSA_SPLIT(uint8);

#undef REGISTER_MUSA_SPLIT

}  // namespace musa
}  // namespace tensorflow
