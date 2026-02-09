#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

// MUSA muDNN 核心头文件
#include <mudnn.h>
#include <mudnn_xmma.h> // 包含非 Batch 版 MatMul 定义
#include "utils_op.h"

namespace tensorflow {
namespace musa {

// === 1. 算子注册 (Op Registration) ===

REGISTER_OP("MusaBatchMatMulV2")
    .Input("x: T")
    .Input("y: T")
    .Output("output: T")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .SetShapeFn(shape_inference::BatchMatMulV2Shape);

REGISTER_OP("MusaMatMul")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("T: {float, double, half, bfloat16}")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .SetShapeFn(shape_inference::MatMulShape);

// === 2. 算子实现 (Op Implementation) ===

template <typename T>
class MusaMatMulOp : public MusaOpKernel {
 public:
  explicit MusaMatMulOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    trans_a_ = false;
    trans_b_ = false;

    // 处理标准 MatMul 属性
    if (ctx->HasAttr("transpose_a")) ctx->GetAttr("transpose_a", &trans_a_);
    if (ctx->HasAttr("transpose_b")) ctx->GetAttr("transpose_b", &trans_b_);

    // 处理 BatchMatMulV2 属性 (adj_x -> transpose_a)
    bool adj_x = false;
    bool adj_y = false;
    if (ctx->GetAttr("adj_x", &adj_x).ok()) trans_a_ = adj_x;
    if (ctx->GetAttr("adj_y", &adj_y).ok()) trans_b_ = adj_y;
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    // 形状校验与广播计算
    MatMulBCast bcast(in0.shape().dim_sizes(), in1.shape().dim_sizes());
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument("Incompatible shapes: ",
                                        in0.shape().DebugString(), " vs ",
                                        in1.shape().DebugString()));

    // 矩阵维度提取
    int64 d0 = in0.dim_size(in0.dims() - 2);
    int64 d1 = in0.dim_size(in0.dims() - 1);
    int64 d2 = in1.dim_size(in1.dims() - 2);
    int64 d3 = in1.dim_size(in1.dims() - 1);

    int64 m = trans_a_ ? d1 : d0;
    int64 k = trans_a_ ? d0 : d1;
    int64 n = trans_b_ ? d2 : d3;
    int64 k_check = trans_b_ ? d3 : d2;

    OP_REQUIRES(ctx, k == k_check,
                errors::InvalidArgument("Matrix size-incompatible: In[0] mismatch In[1]"));

    // 输出 Tensor 分配
    TensorShape out_shape = bcast.output_batch_shape();
    out_shape.AddDim(m);
    out_shape.AddDim(n);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(false);
    mTensor mt_a = CreateMTensor(in0);
    mTensor mt_b = CreateMTensor(in1);
    mTensor mt_out = CreateMTensor(*out);

    // Lambda: 针对 Batch 接口的 2D 降级补齐
    auto FixToBatchFormat = [](mTensor& mt, const Tensor& t) {
      if (t.dims() == 2) {
        int64_t rows = t.dim_size(0);
        int64_t cols = t.dim_size(1);
        mt.SetNdInfo({1, rows, cols}, {rows * cols, cols, 1});
      }
    };

    // --- 🚀 核心分流逻辑 (Dispatch Logic) ---
    ::musa::dnn::Status status;

    if (in0.dims() == 2 && in1.dims() == 2) {
        // [路径 A] 调用高精度非 Batch MatMul (针对 2D 全连接等场景)
        mMatMul op; 
        op.SetTranspose(trans_a_, trans_b_);
        op.SetAlpha(1.0);
        op.SetBeta(0.0);
        
        // 直接运行，不进行 Batch 维度的伪造
        status = op.Run(handle, mt_out, mt_a, mt_b);
        
        OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                    errors::Internal("MUSA MatMul (2D High Precision) execution failed. Status: ", (int)status));
    } else {
        // [路径 B] 调用 BatchMatMul 处理多维张量
        mBatchMatMul op;
        op.SetTranspose(trans_a_, trans_b_);
        op.SetAlpha(1.0);
        op.SetBeta(0.0);

        // 统一格式化为 Batch 布局
        FixToBatchFormat(mt_a, in0);
        FixToBatchFormat(mt_b, in1);
        FixToBatchFormat(mt_out, *out);

        status = op.Run(handle, mt_out, mt_a, mt_b);

        OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                    errors::Internal("MUSA BatchMatMul execution failed. Status: ", (int)status));
    }
    
    VLOG(1) << "MUSA MatMul execution finished successfully.";
  }

 private:
  bool trans_a_ = false;
  bool trans_b_ = false;
};

// === 3. 算子内核注册 (Kernel Registration) ===

#define REGISTER_MUSA_MATMUL_ALL(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(Name("MatMul")                                \
                              .Device("MUSA")                           \
                              .TypeConstraint<TYPE>("T"),               \
                          MusaMatMulOp<TYPE>);                          \
  REGISTER_KERNEL_BUILDER(Name("BatchMatMulV2")                         \
                              .Device("MUSA")                           \
                              .TypeConstraint<TYPE>("T"),               \
                          MusaMatMulOp<TYPE>);                          \
  REGISTER_KERNEL_BUILDER(Name("MusaMatMul")                            \
                              .Device("MUSA")                           \
                              .TypeConstraint<TYPE>("T"),               \
                          MusaMatMulOp<TYPE>);                          \
  REGISTER_KERNEL_BUILDER(Name("MusaBatchMatMulV2")                     \
                              .Device("MUSA")                           \
                              .TypeConstraint<TYPE>("T"),               \
                          MusaMatMulOp<TYPE>);

REGISTER_MUSA_MATMUL_ALL(float);
REGISTER_MUSA_MATMUL_ALL(double);
REGISTER_MUSA_MATMUL_ALL(Eigen::half);
REGISTER_MUSA_MATMUL_ALL(bfloat16);

#undef REGISTER_MUSA_MATMUL_ALL

}  // namespace musa
}  // namespace tensorflow