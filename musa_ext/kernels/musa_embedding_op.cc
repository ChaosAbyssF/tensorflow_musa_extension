#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "mu/device/musa_device.h"
#include <musa_runtime.h>
#include "utils_op.h"
#include <cstdint>

namespace tensorflow {

    // ==========================================
    // 1. 定义算子接口 (REGISTER_OP)
    // ==========================================
    REGISTER_OP("MusaEmbedding")
        .Input("t: T")                   // 数据输入 (Params)
        .Input("tindices: Tindices")     // 索引输入 (Indices)
        .Output("output: T")             // 输出 (Output)
        .Attr("T: type")                 // 数据类型属性
        .Attr("Tindices: {int32, int64}")// 索引类型属性
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            shape_inference::ShapeHandle params_shape;
            TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &params_shape));
            shape_inference::ShapeHandle ids_shape = c->input(1);
            shape_inference::ShapeHandle sub_shape;
            TF_RETURN_IF_ERROR(c->Subshape(params_shape, 1, &sub_shape));
            shape_inference::ShapeHandle out;
            TF_RETURN_IF_ERROR(c->Concatenate(ids_shape, sub_shape, &out));
            c->set_output(0, out);
            return Status::OK();
        });

    // ==========================================
    // 2. 声明外部 Launcher 函数 (对应 .mu 文件)
    // ==========================================
    template <typename T, typename Tidx>
    void LaunchEmbeddingLookup(
        const T* params,
        const Tidx* ids,
        T* output,
        int vocab_size,
        int embedding_dim,
        int total_count,
        musaStream_t stream
    );

    // ==========================================
    // 3. OpKernel 实现类
    // ==========================================
    template <typename T, typename Tidx>
    class MusaEmbeddingOp : public OpKernel {
    public:
        explicit MusaEmbeddingOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

        void Compute(OpKernelContext* ctx) override {
            const Tensor& params = ctx->input(0);
            const Tensor& ids = ctx->input(1);

            // 类型检查 (双重保险)
            OP_REQUIRES(ctx, params.dtype() == DataTypeToEnum<T>::value,
                errors::InvalidArgument("Input 0 (params) dtype mismatch"));
            OP_REQUIRES(ctx, ids.dtype() == DataTypeToEnum<Tidx>::value,
                errors::InvalidArgument("Input 1 (indices) dtype mismatch"));

            OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(params.shape()),
                errors::InvalidArgument("params must be at least 1 dim"));

            TensorShape output_shape(ids.shape());
            for (int i = 1; i < params.dims(); ++i) {
                output_shape.AddDim(params.dim_size(i));
            }

            Tensor* output = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

            if (output->NumElements() <= 0) return;

            int vocab_size = params.dim_size(0);
            int total_count = output->NumElements();
            int embedding_dim = total_count / ids.NumElements(); // 更稳健的计算方式

            auto* musa_dev = musa::GetDeviceByCtx(ctx);
            musaStream_t stream = musa_dev->GetStream();

            // =======================================================
            // 核心修正：使用 .flat<T>().data() 而不是 reinterpret_cast
            // 这更加安全，且与官方实现保持一致
            // =======================================================
            const T* d_params = params.flat<T>().data();
            const Tidx* d_ids = ids.flat<Tidx>().data();
            T* d_output = output->flat<T>().data();

            LaunchEmbeddingLookup<T, Tidx>(
                d_params, d_ids, d_output,
                vocab_size, embedding_dim, total_count, stream
            );
        }
    };

    // ==========================================
    // 4. 注册 Kernel (覆盖所有类型组合)
    // ==========================================
    
    // 定义一个宏来简化注册：为一个数据类型 T 注册 int32 和 int64 两种索引
#define REGISTER_MUSA_EMBEDDING_ALL_INDICES(T)                           \
  REGISTER_KERNEL_BUILDER(Name("MusaEmbedding")                          \
                              .Device("MUSA")                            \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<int32>("Tindices"),        \
                          MusaEmbeddingOp<T, int32>);                    \
  REGISTER_KERNEL_BUILDER(Name("MusaEmbedding")                          \
                              .Device("MUSA")                            \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<int64>("Tindices"),        \
                          MusaEmbeddingOp<T, int64>);

    // -----------------------------------------------------
    // A. 浮点数类型 (Float, Double)
    // -----------------------------------------------------
    REGISTER_MUSA_EMBEDDING_ALL_INDICES(float);
    REGISTER_MUSA_EMBEDDING_ALL_INDICES(double);

    // -----------------------------------------------------
    // B. 整数类型 (Int32, Int64)
    // -----------------------------------------------------
    // 【关键修改】补全这部分，解决之前的 "Could not find device" 报错
    REGISTER_MUSA_EMBEDDING_ALL_INDICES(int32_t);
    REGISTER_MUSA_EMBEDDING_ALL_INDICES(int64_t);

    // -----------------------------------------------------
    // C. (可选) 半精度浮点
    // -----------------------------------------------------
    // #include <Eigen/Core>
    // REGISTER_MUSA_EMBEDDING_ALL_INDICES(Eigen::half);

} // namespace tensorflow
