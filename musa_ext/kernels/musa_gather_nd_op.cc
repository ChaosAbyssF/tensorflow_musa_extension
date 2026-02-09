/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved. */
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T, typename IndexT>
class MusaGatherNdOp : public MusaOpKernel {
 public:
  explicit MusaGatherNdOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& params = ctx->input(0);
    const Tensor& indices = ctx->input(1);

    const int64_t params_dims = params.dims();
    const int64_t indices_dims = indices.dims();
    const int64_t index_depth = indices.dim_size(indices_dims - 1);

    // 1. 形状校验：GatherNd 的索引深度不能超过参数维度
    OP_REQUIRES(ctx, index_depth <= params_dims,
                errors::InvalidArgument("index_depth (", index_depth, 
                                        ") must be <= params_dims (", params_dims, ")"));

    // 2. 计算输出 Shape
    // 公式：OutputShape = indices.shape[:-1] + params.shape[index_depth:]
    TensorShape output_shape;
    for (int i = 0; i < indices_dims - 1; ++i) {
      output_shape.AddDim(indices.dim_size(i));
    }
    for (int i = index_depth; i < params_dims; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) return;

    // 3. 准备 MUSA 资源
    auto& handle = GetHandleByCtx(ctx);
    mTensor t_params = CreateMTensor(params);
    mTensor t_indices = CreateMTensor(indices);
    mTensor t_output = CreateMTensor(*output);

    // 4. 配置并执行 GatherX (GATHER_ND 模式)
    mGatherX op; // 对应 ::musa::dnn::GatherX
    
    // 设置模式为 GATHER_ND
    auto status = op.SetMode(::musa::dnn::GatherX::Mode::GATHER_ND);
    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("muDNN GatherX SetMode(GATHER_ND) failed"));

    // 注意：按照头文件 Run(handle, out, index, in) 的顺序调用
    status = op.Run(handle, t_output, t_indices, t_params);

    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("MUSA muDNN GatherNd (GatherX) execution failed. Status: ",
                                static_cast<int>(status)));
  }
};

// =====================================================================
// 5. 算子注册 (支持 Wide&Deep 常用类型)
// =====================================================================

#define REGISTER_MUSA_GATHER_ND(type, itype)                         \
  REGISTER_KERNEL_BUILDER(Name("GatherNd")                           \
                              .Device(DEVICE_MTGPU)                  \
                              .TypeConstraint<type>("Tparams")       \
                              .TypeConstraint<itype>("Tindices"),    \
                          MusaGatherNdOp<type, itype>);

// 注册 float 配合 int32/int64 索引
REGISTER_MUSA_GATHER_ND(float, int32);
REGISTER_MUSA_GATHER_ND(float, int64);

// 注册半精度 (Embedding 常用)
REGISTER_MUSA_GATHER_ND(Eigen::half, int32);
REGISTER_MUSA_GATHER_ND(Eigen::half, int64);
REGISTER_MUSA_GATHER_ND(bfloat16, int32);
REGISTER_MUSA_GATHER_ND(bfloat16, int64);

// 注册 int32/int64 数据类型
REGISTER_MUSA_GATHER_ND(int32, int32);
REGISTER_MUSA_GATHER_ND(int32, int64);
REGISTER_MUSA_GATHER_ND(int64, int32);
REGISTER_MUSA_GATHER_ND(int64, int64);

} // namespace musa
} // namespace tensorflow
