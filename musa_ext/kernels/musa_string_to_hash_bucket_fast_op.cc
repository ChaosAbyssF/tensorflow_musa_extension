#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/hash/hash.h" // 使用基础 Hash 库，保证编译通过

#include <musa_runtime.h> // 引入 MUSA 运行时 API，用于内存拷贝

namespace tensorflow {
namespace musa {

class MusaStringToHashBucketFastOp : public OpKernel {
 public:
  explicit MusaStringToHashBucketFastOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // 1. 读取属性：一共要分多少个桶
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_buckets", &num_buckets_));
  }

  void Compute(OpKernelContext* ctx) override {
    // --- 步骤 A: 获取输入 (CPU 端) ---
    // 注意：注册时我们会用 .HostMemory("input")，所以这里读到的是 CPU 内存
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<tstring>();

    // --- 步骤 B: 分配输出 (GPU 端) ---
    // 注意：输出是 int64，我们希望它直接在显存里，方便后续 GPU 算子使用
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor->shape(), &output_tensor));

    // 如果输入为空，直接返回
    if (input_tensor->NumElements() == 0) return;

    int64 N = input_tensor->NumElements();
    
    // --- 步骤 C: 在 CPU 上计算 Hash ---
    // 策略：先在 CPU 上申请一块临时内存存放计算结果
    std::vector<int64> host_output(N); 

    for (int64 i = 0; i < N; ++i) {
      const tstring& s = input_flat(i);
      
      // 使用 Hash64 计算指纹
      uint64 hash = tensorflow::Hash64(s.data(), s.size());
      
      // 取模运算，算出 Bucket ID
      host_output[i] = static_cast<int64>(hash % num_buckets_);
    }

    // --- 步骤 D: 数据搬运 (CPU -> GPU) ---
    // 获取输出 Tensor 在显存中的物理地址
    int64* device_ptr = output_tensor->flat<int64>().data();
    
    // 调用 MUSA API，把 CPU 算好的结果一次性拷贝到显存
    // 这种做法比在 GPU 上强行处理字符串要快得多
    musaError_t err = musaMemcpy(device_ptr, 
                                 host_output.data(), 
                                 N * sizeof(int64), 
                                 musaMemcpyHostToDevice);

    OP_REQUIRES(ctx, err == musaSuccess,
                errors::Internal("MUSA memcpy failed: ", musaGetErrorString(err)));
  }

 private:
  int64 num_buckets_;
};

// 注册 Kernel
// ！！！关键点！！！
// .HostMemory("input")：告诉 TF，输入字符串留在 CPU。
// (注意：我们没有写 .HostMemory("output")，这意味着输出默认会在 GPU 上)
#define REGISTER_MUSA_KERNEL()                        \
  REGISTER_KERNEL_BUILDER(Name("StringToHashBucketFast") \
                              .Device("MUSA")            \
                              .HostMemory("input"),      \
                          MusaStringToHashBucketFastOp);

REGISTER_MUSA_KERNEL();

}  // namespace musa
}  // namespace tensorflow
