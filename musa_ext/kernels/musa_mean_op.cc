#include <iostream>
#include <vector>
#include <memory>
#include <functional>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/bounds_check.h"

// 引入 MUSA 头文件
#include "mu/device/musa_memcpy.h"
#include "utils_op.h"
#include <mudnn.h>

namespace tensorflow {
namespace musa {

REGISTER_OP("MusaMean")
    .Input("input: T")
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("Tidx: {int32, int64}")
    .SetShapeFn(shape_inference::ReductionShape);

template <typename T, typename Tidx>
class MusaMeanOp : public MusaOpKernel {
 public:
  explicit MusaMeanOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  }

  void Compute(OpKernelContext* ctx) override {
    // 调试日志 (可选保留，方便验证是否运行)
    //fprintf(stderr, ">>> [MUSA_TRACE_AUTO] %s\n", name().c_str());

    const Tensor& input = ctx->input(0);
    const Tensor& axes_tensor = ctx->input(1);

    if (input.NumElements() == 0) {
      ctx->set_output(0, input);
      return;
    }

    // --- 1. 解析 Reduce 维度 (完全参考 ReduceSum) ---
    int64_t num_axes = axes_tensor.NumElements();
    std::vector<int> reduce_dims;
    // 使用 TF 的 bitmap 辅助工具
    gtl::InlinedVector<bool, 4> bitmap(input.dims(), false);

    if (num_axes == 0) {
      // 如果 axes 为空，通常意味着 Reduce 所有维度 (Global Mean)
      for (int i = 0; i < input.dims(); ++i) {
        bitmap[i] = true;
        reduce_dims.push_back(i);
      }
    } else {
      auto axes_flat = axes_tensor.flat<Tidx>();
      for (int64_t i = 0; i < num_axes; i++) {
        Tidx index = axes_flat(i);
        if (index < 0) index += input.dims(); // 处理负索引
        if (index >= 0 && index < input.dims() && !bitmap[index]) {
          bitmap[index] = true;
          reduce_dims.push_back(static_cast<int>(index));
        }
      }
    }

    // --- 2. 计算输出 Shape ---
    TensorShape output_shape;       // TF 逻辑输出形状 (根据 keep_dims 变化)
    TensorShape musa_output_shape;  // MUSA 物理输出形状 (总是 keep_dims=true)
    int64_t reduce_elements = 1;

    for (int d = 0; d < input.dims(); ++d) {
      if (bitmap[d]) {
        reduce_elements *= input.dim_size(d);
        if (keep_dims_) output_shape.AddDim(1);
        musa_output_shape.AddDim(1); // muDNN 总是需要看到维度变成 1
      } else {
        output_shape.AddDim(input.dim_size(d));
        musa_output_shape.AddDim(input.dim_size(d));
      }
    }

    // --- 3. 分配输出内存 ---
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));

    if (out->NumElements() == 0 || reduce_elements == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    // 特殊情况：如果是 Identity (没有维度被 reduce)，直接拷贝
    if (reduce_elements == 1) {
       musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());
       MusaMemcpyAsyncD2D(const_cast<char*>(out->tensor_data().data()),
                          input.tensor_data().data(),
                          input.TotalBytes(), stream);
       return;
    }

    // --- 4. 准备 muDNN 计算 ---
    
    // 创建一个 View (视图)，让 muDNN 以为输出是 keep_dims=true 的形状
    // 这样数据指针没变，但形状对了
    Tensor out_reshaped(out->dtype());
    OP_REQUIRES(ctx, out_reshaped.CopyFrom(*out, musa_output_shape),
                errors::Internal("Reshape failed."));

    mTensor t_in = CreateMTensor(input, format_);
    mTensor t_out = CreateMTensor(out_reshaped, format_);

    mReduce op;
    // 【关键】设置模式为 MEAN
    op.SetMode(::musa::dnn::Reduce::Mode::MEAN);
    // 【关键】显式设置 Reduce 维度 (之前漏了这个)
    op.SetDim(reduce_dims.size(), reduce_dims.data());

    // --- 5. 配置显存分配器 (之前漏了这个导致 SegFault) ---
    // 这是为了给 muDNN 提供临时 Workspace 空间
    tensorflow::Allocator* tf_allocator = ctx->device()->GetAllocator(tensorflow::AllocatorAttributes());

    auto alloc_func = [tf_allocator](size_t size) -> std::unique_ptr<void, std::function<void(void*)>> {
        void* ptr = tf_allocator->AllocateRaw(256, size);
        std::function<void(void*)> deleter = [tf_allocator](void* p) {
            if (p) tf_allocator->DeallocateRaw(p);
        };
        return std::unique_ptr<void, std::function<void(void*)>>(ptr, deleter);
    };

    ::musa::dnn::MemoryMaintainer mm(alloc_func);

    // --- 6. 执行 ---
    auto status = op.Run(handle, t_out, t_in, mm);
    
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS, 
                errors::Internal("MUSA Mean execution failed. Status: ", (int)status));
  }

 private:
  bool keep_dims_;
};

// 注册 Kernel
#define REGISTER_MEAN_KERNEL(T, Tidx)                               \
  REGISTER_KERNEL_BUILDER(Name("Mean")                              \
                              .Device("MUSA")                       \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<Tidx>("Tidx")         \
                              .HostMemory("reduction_indices"),     \
                          MusaMeanOp<T, Tidx>);                     \
  REGISTER_KERNEL_BUILDER(Name("MusaMean")                          \
                              .Device("MUSA")                       \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<Tidx>("Tidx")         \
                              .HostMemory("reduction_indices"),     \
                          MusaMeanOp<T, Tidx>);

REGISTER_MEAN_KERNEL(float, int32);
REGISTER_MEAN_KERNEL(float, int64);
REGISTER_MEAN_KERNEL(Eigen::half, int32);
REGISTER_MEAN_KERNEL(Eigen::half, int64);
REGISTER_MEAN_KERNEL(bfloat16, int32);
REGISTER_MEAN_KERNEL(bfloat16, int64);
// 如果需要 double 支持，可以加上
// REGISTER_MEAN_KERNEL(double, int32);
// REGISTER_MEAN_KERNEL(double, int64);

#undef REGISTER_MEAN_KERNEL

}  // namespace musa
}  // namespace tensorflow
