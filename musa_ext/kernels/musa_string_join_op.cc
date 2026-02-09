#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace musa {

class MusaStringJoinOp : public OpKernel {
 public:
  explicit MusaStringJoinOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // 1. 获取分隔符属性 (separator)
    OP_REQUIRES_OK(ctx, ctx->GetAttr("separator", &separator_));
  }

  void Compute(OpKernelContext* ctx) override {
    // 2. 获取输入列表
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs));
    
    // 3. 校验输入
    const int num_inputs = inputs.size();
    if (num_inputs == 0) return; // 没数据就不干活

    const Tensor& input0 = inputs[0];
    const int64 num_elements = input0.NumElements();
    
    // 简单校验：这里假设所有输入 Tensor 形状一致
    // (完整版 TF 算子支持广播，但在 MUSA 适配初期，通常先支持同形状)
    for (int i = 1; i < num_inputs; ++i) {
      OP_REQUIRES(ctx, inputs[i].shape() == input0.shape(),
                  errors::InvalidArgument("MUSA StringJoin currently requires all inputs to have the same shape."));
    }

    // 4. 分配输出 (输出也是 String，所以也是 HostMemory)
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input0.shape(), &output));
    
    // 获取扁平化的数据视图
    auto output_flat = output->flat<tstring>();
    
    // 5. 执行拼接 (在 CPU 上进行)
    for (int64 i = 0; i < num_elements; ++i) {
      // 获取第一个字符串
      tstring& out_s = output_flat(i);
      out_s = inputs[0].flat<tstring>()(i);
      
      // 循环拼接剩下的
      for (int j = 1; j < num_inputs; ++j) {
        if (!separator_.empty()) {
          out_s.append(separator_);
        }
        out_s.append(inputs[j].flat<tstring>()(i));
      }
    }
    
    // 注意：因为输入输出都在 HostMemory，所以这里不需要 musaMemcpy
  }

 private:
  string separator_;
};

// 6. 注册 Kernel (关键步骤)
// 使用 .HostMemory("inputs") 和 .HostMemory("output")
// 告诉 TF：虽然我在 MUSA 设备上，但我的数据全都在 CPU 内存里，别往显存拷！
#define REGISTER_MUSA_KERNEL()                      \
  REGISTER_KERNEL_BUILDER(Name("StringJoin")        \
                              .Device("MUSA")       \
                              .HostMemory("inputs") \
                              .HostMemory("output"),\
                          MusaStringJoinOp);

REGISTER_MUSA_KERNEL();

}  // namespace musa
}  // namespace tensorflow
