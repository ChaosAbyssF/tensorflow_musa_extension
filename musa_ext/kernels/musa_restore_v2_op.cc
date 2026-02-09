#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor.h" 

namespace tensorflow {
namespace musa {

class MusaRestoreV2Op : public OpKernel {
 public:
  explicit MusaRestoreV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& prefix = ctx->input(0);
    const Tensor& tensor_names = ctx->input(1);
    const Tensor& shape_and_slices = ctx->input(2);

    OP_REQUIRES(ctx, prefix.NumElements() == 1,
                errors::InvalidArgument("prefix must have 1 element"));
    const string& prefix_str = prefix.flat<tstring>()(0);

    const auto& names_flat = tensor_names.flat<tstring>();
    const auto& slices_flat = shape_and_slices.flat<tstring>();

    int num_tensors = tensor_names.NumElements();
    OP_REQUIRES(ctx, shape_and_slices.NumElements() == num_tensors,
                errors::InvalidArgument("shape_and_slices must match tensor_names size"));
    OP_REQUIRES(ctx, dtypes_.size() == num_tensors,
                errors::InvalidArgument("dtypes must match tensor_names size"));

    BundleReader reader(ctx->env(), prefix_str);
    OP_REQUIRES_OK(ctx, reader.status());

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream != nullptr, errors::Internal("No MUSA stream available"));

    for (int i = 0; i < num_tensors; ++i) {
      const string& name = names_flat(i);
      const string& slice_spec = slices_flat(i);
      DataType dtype = dtypes_[i];

      // 1. 获取完整 Shape
      TensorShape full_shape;
      OP_REQUIRES_OK(ctx, reader.LookupTensorShape(name, &full_shape));

      // 2. 解析切片逻辑
      TensorShape target_shape;
      TensorSlice slice(0);
      bool is_slice = !slice_spec.empty();

      if (is_slice) {
        OP_REQUIRES_OK(ctx, TensorSlice::Parse(slice_spec, &slice));
        OP_REQUIRES_OK(ctx, slice.SliceTensorShape(full_shape, &target_shape));
      } else {
        target_shape = full_shape;
      }

      // 3. 分配 MUSA 输出内存
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, target_shape, &output_tensor));

      if (target_shape.num_elements() == 0) continue;

      // 4. 分配 CPU 临时内存 (始终分配 Full Shape 以避免溢出)
      Tensor cpu_full_tensor;
      AllocatorAttributes cpu_alloc_attr;
      cpu_alloc_attr.set_on_host(true);
      cpu_alloc_attr.set_gpu_compatible(true); // 建议加上，虽然 BlockHostUntilDone 会掩盖未对齐的问题，但加上更好
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(dtype, full_shape, &cpu_full_tensor, cpu_alloc_attr));

      // 5. 读取全量数据
      OP_REQUIRES_OK(ctx, reader.Lookup(name, &cpu_full_tensor));

      // 6. 计算源数据指针和偏移
      const char* src_ptr_base = reinterpret_cast<const char*>(cpu_full_tensor.data());
      const char* src_ptr = src_ptr_base;
      uint64 copy_size_bytes = target_shape.num_elements() * DataTypeSize(dtype);

      if (is_slice) {
        // 简单检查连续性
        bool is_contiguous = true;
        for (int d = 1; d < full_shape.dims(); ++d) {
          if (slice.length(d) != full_shape.dim_size(d)) {
            is_contiguous = false;
            break;
          }
        }
        
        if (!is_contiguous) {
           OP_REQUIRES(ctx, false, errors::Unimplemented(
               "MusaRestoreV2 currently only supports contiguous slices (slicing on the first dimension)."));
        }

        int64 start_idx = slice.start(0);
        int64 stride = 1;
        for (int d = 1; d < full_shape.dims(); ++d) {
          stride *= full_shape.dim_size(d);
        }
        
        int64 offset_bytes = start_idx * stride * DataTypeSize(dtype);
        src_ptr += offset_bytes;
      }

      // 7. 内存拷贝到 MUSA
      void* dst_ptr = output_tensor->data();
      se::DeviceMemoryBase dst_mem(dst_ptr, copy_size_bytes);
      stream->ThenMemcpy(&dst_mem, src_ptr, copy_size_bytes);

      // =================================================================================
      // 关键修复：同步等待！
      // 必须等待 GPU 拷贝完成，才能让 cpu_full_tensor 析构，否则 GPU 会读取已释放的内存。
      // =================================================================================
      OP_REQUIRES_OK(ctx, stream->BlockHostUntilDone());
    }
  }

 private:
  std::vector<DataType> dtypes_;
};

REGISTER_KERNEL_BUILDER(Name("RestoreV2")
                            .Device("MUSA")
                            .HostMemory("prefix")
                            .HostMemory("tensor_names")
                            .HostMemory("shape_and_slices"),
                        MusaRestoreV2Op);

} // namespace musa
} // namespace tensorflow
