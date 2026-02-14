/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved. */

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <random>
#include <limits>

#include "utils_op.h"
#include "mu/device/musa_memcpy.h"

namespace tensorflow {
namespace musa {

namespace {

template <typename T>
class MusaRandomOp : public MusaOpKernel {
 public:
  explicit MusaRandomOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    fprintf(stderr, ">>> [MUSA_TRACE_AUTO] %s - Start\n", name().c_str());

    const Tensor& shape_tensor = ctx->input(0);
    TensorShape out_shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &out_shape));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

    if (output->NumElements() == 0) return;

    Tensor tmp_host_out;
    AllocatorAttributes host_attr;
    host_attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(output->dtype(), out_shape, &tmp_host_out, host_attr));

    T* cpu_ptr = tmp_host_out.flat<T>().data();
    std::random_device rd;
    std::mt19937 gen(rd());

    if (name().find("UniformInt") != std::string::npos) {
      T min_val = static_cast<T>(0);
      T max_val = static_cast<T>(255);

      // 根据 OpDef 提取输入
      if (name().find("Stateless") != std::string::npos) {
          // StatelessRandomUniformIntV2: 0:shape, 1:key, 2:counter, 3:alg, 4:minval, 5:maxval
          if (ctx->num_inputs() >= 6) {
              min_val = ctx->input(4).flat<T>()(0);
              max_val = ctx->input(5).flat<T>()(0);
          }
      } else {
          // RandomUniformInt: 0:shape, 1:minval, 2:maxval
          if (ctx->num_inputs() >= 3) {
              min_val = ctx->input(1).flat<T>()(0);
              max_val = ctx->input(2).flat<T>()(0);
          }
      }

      if (static_cast<float>(min_val) >= static_cast<float>(max_val)) {
          ctx->CtxFailure(__FILE__, __LINE__, errors::InvalidArgument("minval must be < maxval"));
          return;
      }

      using DistType = typename std::conditional<std::is_integral<T>::value, T, int32>::type;
      std::uniform_int_distribution<DistType> dist(static_cast<DistType>(min_val), 
                                                   static_cast<DistType>(max_val) - 1);
      for (int i = 0; i < output->NumElements(); ++i) {
        cpu_ptr[i] = static_cast<T>(dist(gen));
      }
    } else if (name().find("Normal") != std::string::npos) {
      std::normal_distribution<float> dist(0.0f, 1.0f);
      for (int i = 0; i < output->NumElements(); ++i) {
        cpu_ptr[i] = static_cast<T>(dist(gen));
      }
    } else {
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      for (int i = 0; i < output->NumElements(); ++i) {
        cpu_ptr[i] = static_cast<T>(dist(gen));
      }
    }

    size_t total_bytes = output->NumElements() * sizeof(T);
    mStatus status = MusaMemcpyH2D(output->data(), tmp_host_out.data(), total_bytes);

    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("MUSA Random H2D copy failed for ", name()));

    musaDeviceSynchronize();
    fprintf(stderr, ">>> [MUSA_TRACE_AUTO] %s - Success (%s)\n", 
            name().c_str(), DataTypeString(output->dtype()).c_str());
  }
};

} // namespace anonymous

// ============================================================
// 完整注册宏 - 修正参数名为 minval/maxval (单数)
// ============================================================
#define REGISTER_MUSA_RANDOM_KERNELS(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(Name("RandomUniform")                                \
                              .Device("MUSA")                                  \
                              .HostMemory("shape")                             \
                              .TypeConstraint<TYPE>("dtype"),                  \
                          MusaRandomOp<TYPE>);                                 \
                                                                               \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomUniform")                       \
                              .Device("MUSA")                                  \
                              .HostMemory("shape")                             \
                              .HostMemory("seed")                              \
                              .TypeConstraint<TYPE>("dtype"),                  \
                          MusaRandomOp<TYPE>);                                 \
                                                                               \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomUniformV2")                     \
                              .Device("MUSA")                                  \
                              .HostMemory("shape")                             \
                              .HostMemory("key")                               \
                              .HostMemory("counter")                           \
                              .HostMemory("alg")                               \
                              .TypeConstraint<TYPE>("dtype"),                  \
                          MusaRandomOp<TYPE>);                                 \
                                                                               \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomNormalV2")                      \
                              .Device("MUSA")                                  \
                              .HostMemory("shape")                             \
                              .HostMemory("key")                               \
                              .HostMemory("counter")                           \
                              .HostMemory("alg")                               \
                              .TypeConstraint<TYPE>("dtype"),                  \
                          MusaRandomOp<TYPE>);                                 \
                                                                               \
  REGISTER_KERNEL_BUILDER(Name("StatelessTruncatedNormalV2")                   \
                              .Device("MUSA")                                  \
                              .HostMemory("shape")                             \
                              .HostMemory("key")                               \
                              .HostMemory("counter")                           \
                              .HostMemory("alg")                               \
                              .TypeConstraint<TYPE>("dtype"),                  \
                          MusaRandomOp<TYPE>);                                 \
                                                                               \
  /* 修正处：HostMemory 参数名必须与 OpDef 严格一致 */                             \
  REGISTER_KERNEL_BUILDER(Name("RandomUniformInt")                             \
                              .Device("MUSA")                                  \
                              .HostMemory("shape")                             \
                              .HostMemory("minval")                            \
                              .HostMemory("maxval")                            \
                              .TypeConstraint<TYPE>("Tout"),                   \
                          MusaRandomOp<TYPE>);                                 \
                                                                               \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomUniformIntV2")                  \
                              .Device("MUSA")                                  \
                              .HostMemory("shape")                             \
                              .HostMemory("key")                               \
                              .HostMemory("counter")                           \
                              .HostMemory("alg")                               \
                              .HostMemory("minval")                            \
                              .HostMemory("maxval")                            \
                              .TypeConstraint<TYPE>("dtype"),                  \
                          MusaRandomOp<TYPE>);
  //                                                                              \
  // REGISTER_KERNEL_BUILDER(Name("RandomStandardNormal")                         \
  //                             .Device("MUSA")                                  \
  //                             .HostMemory("shape")                             \
  //                             .TypeConstraint<TYPE>("dtype"),                  \
  //                         MusaRandomOp<TYPE>);                                 \
  //                                                                              \
  // REGISTER_KERNEL_BUILDER(Name("TruncatedNormal")                              \
  //                             .Device("MUSA")                                  \
  //                             .HostMemory("shape")                             \
  //                             .TypeConstraint<TYPE>("dtype"),                  \
  //                         MusaRandomOp<TYPE>);                                 \

// 执行批量注册
REGISTER_MUSA_RANDOM_KERNELS(float);
REGISTER_MUSA_RANDOM_KERNELS(double);
REGISTER_MUSA_RANDOM_KERNELS(Eigen::half);
REGISTER_MUSA_RANDOM_KERNELS(Eigen::bfloat16);
REGISTER_MUSA_RANDOM_KERNELS(int32);
REGISTER_MUSA_RANDOM_KERNELS(int64);

#undef REGISTER_MUSA_RANDOM_KERNELS

} // namespace musa
} // namespace tensorflow
