/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved. */

#include <mudnn.h>
#include <mudnn_tensor.h>
#include "utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace musa {
namespace {

template <typename T, typename Tmultiples>
class MusaTileOp : public MusaOpKernel {
 public:
  explicit MusaTileOp(OpKernelConstruction* context) : MusaOpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& multiples = context->input(1);

    const int input_dims = input.dims();
    const Tmultiples* m_data = multiples.flat<Tmultiples>().data();

    // 1. 计算输出形状
    TensorShape output_shape;
    bool need_tile = false;
    for (int i = 0; i < input_dims; ++i) {
      Tmultiples m = m_data[i];
      output_shape.AddDim(input.dim_size(i) * m);
      if (m != 1) need_tile = true;
    }

    // 处理标量或无需平铺的情况
    if (input_dims == 0 || !need_tile) {
      context->set_output(0, input);
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) return;

    // 2. MUSA 核心逻辑：使用 Permute + Stride 0 广播
    auto& h = GetHandleByCtx(context);
    auto in_mt = CreateMTensor(input);
    auto out_mt = CreateMTensor(*output);

    // 计算逻辑步长 (Broadcasting Strides)
    std::vector<int64_t> b_dims;
    std::vector<int64_t> b_strides;
    int64_t stride = 1;
    std::vector<int64_t> orig_strides(input_dims);

    for (int i = input_dims - 1; i >= 0; --i) {
      orig_strides[i] = (input.dim_size(i) == 1) ? 0 : stride;
      stride *= input.dim_size(i);
    }

    for (int i = 0; i < input_dims; ++i) {
      b_dims.push_back(output_shape.dim_size(i));
      b_strides.push_back(input.dim_size(i) == 1 ? 0 : orig_strides[i]);
    }

    MTOP_CHECK_OK(in_mt.SetNdInfo(input_dims, b_dims.data(), b_strides.data()),
                  "Tile SetNdInfo", context);

    // 3. 执行物理平铺
    ::musa::dnn::Permute op;
    MTOP_CHECK_OK_RUN(op.Run(h, out_mt, in_mt), "Permute Run for Tile", context);
  }
};

// =====================================================================
// 4. 注册 6 种数据类型 (T) 及其对应的 2 种索引类型 (Tmultiples)
// =====================================================================
#define REGISTER_MUSA_TILE_ALL_TYPES(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Tile")                                   \
                              .Device(DEVICE_MTGPU)                      \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<int32>("Tmultiples")       \
                              .HostMemory("multiples"),                  \
                          MusaTileOp<type, int32>);                      \
  REGISTER_KERNEL_BUILDER(Name("Tile")                                   \
                              .Device(DEVICE_MTGPU)                      \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<int64>("Tmultiples")       \
                              .HostMemory("multiples"),                  \
                          MusaTileOp<type, int64>);

// 6 种核心类型：float, half, double, int32, int64, bool
REGISTER_MUSA_TILE_ALL_TYPES(float);
REGISTER_MUSA_TILE_ALL_TYPES(Eigen::half);
REGISTER_MUSA_TILE_ALL_TYPES(double);
REGISTER_MUSA_TILE_ALL_TYPES(int32);
REGISTER_MUSA_TILE_ALL_TYPES(int64);
REGISTER_MUSA_TILE_ALL_TYPES(bool);

#undef REGISTER_MUSA_TILE_ALL_TYPES

}  // namespace
}  // namespace musa
}  // namespace tensorflow
