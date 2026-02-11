/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved. */

#include "utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "mu/device/musa_device.h" 
#include <mudnn.h>

namespace tensorflow {
namespace musa {

template <typename T>
class MusaWhereOp : public MusaOpKernel {
 public:
  explicit MusaWhereOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    int64_t rank = input.dims();

    // 0. 空输入处理
    if (input.NumElements() == 0) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({0, rank}), &output));
      return;
    }

    MusaDevice* musa_device = reinterpret_cast<MusaDevice*>(ctx->device());
    auto& h = musa_device->mudnn_handle();

    // 1. 创建 Input mTensor (直接调用 utils)
    auto input_mt = CreateMTensor(input);

    // 2. 准备 MemoryMaintainer (依然需要保持，因为这是 Where 算子的特殊性)
    size_t captured_size = 0;
    void* scratch_ptr = nullptr;
    Tensor scratch_tensor_holder; // 放在这里防止析构

    auto mm = musa_device->GetMemMaintainer(
        [ctx, &captured_size, &scratch_ptr, &scratch_tensor_holder](size_t size) -> ::musa::dnn::MemoryHandler {
            captured_size = size;
            Status s = ctx->allocate_temp(DT_INT8, TensorShape({static_cast<int64_t>(size)}), &scratch_tensor_holder);
            if (!s.ok()) return ::musa::dnn::MemoryHandler(nullptr, [](void*){});
            
            scratch_ptr = const_cast<char*>(scratch_tensor_holder.tensor_data().data());
            return ::musa::dnn::MemoryHandler(scratch_ptr, [](void*) {});
        }
    );

    // 3. 运行 Nonzero
    ::musa::dnn::Nonzero op;
    ::musa::dnn::Tensor out_mt; 
    out_mt.SetType(mType::INT64); // 输出索引固定为 INT64

    auto status = op.Run(h, out_mt, input_mt, mm);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS, errors::Internal("MUSA Nonzero run failed"));

    // 4. 同步以获取 size
    musaStreamSynchronize(h.GetStream());

    // 5. 处理输出
    if (captured_size == 0) {
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({0, rank}), &output));
        return;
    }

    size_t element_size = sizeof(int64_t);
    size_t num_nonzero = captured_size / (rank * element_size);

    TensorShape output_shape({static_cast<int64_t>(num_nonzero), rank});
    Tensor* final_output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &final_output));

    // 6. 拷贝数据
    auto dst_ptr = final_output->tensor_data().data();
    musaMemcpyAsync((void*)dst_ptr, scratch_ptr, captured_size, musaMemcpyDeviceToDevice, h.GetStream());
  }
};

#define REGISTER_WHERE(T) \
  REGISTER_KERNEL_BUILDER(Name("Where").Device(DEVICE_MTGPU).TypeConstraint<T>("T"), MusaWhereOp<T>);

REGISTER_WHERE(float);
REGISTER_WHERE(double);
REGISTER_WHERE(int32);
REGISTER_WHERE(int64);
REGISTER_WHERE(bool);
REGISTER_WHERE(Eigen::half);
REGISTER_WHERE(Eigen::bfloat16);

#undef REGISTER_WHERE

}  // namespace musa
}  // namespace tensorflow