#include "musa_where_op.h"

#include <utility>

#include "../utils_op.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaWhereOp : public MusaOpKernel {
 public:
  explicit MusaWhereOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const int input_dims = input.dims();
    if (input.NumElements() == 0) {
      // Handle the case where there are no elements in the input tensor.
      Tensor* out = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  0, TensorShape({0, input_dims}), &out));
      return;
    }

    ComputeType(context, input, input_dims);
  }

  void ComputeType(OpKernelContext* context, const Tensor& input,
                   int input_dims) {
    musaStream_t stream = GetMusaStreamByCtx(context);
    const int64 num_items = input.NumElements();

    Tensor flat_indices_t;
    OP_REQUIRES_OK(
      context,
      context->allocate_temp(DataTypeToEnum<int64>::value,
                   TensorShape({num_items}), &flat_indices_t));
    int64* d_flat_indices = flat_indices_t.flat<int64>().data();

    Tensor selected_count_t;
    OP_REQUIRES_OK(context,
             context->allocate_temp(DataTypeToEnum<int64>::value,
                        TensorShape({1}), &selected_count_t));
    int64* d_num_selected = selected_count_t.flat<int64>().data();

    const size_t select_workspace_bytes =
      GetSelectTrueIndicesWorkspaceSize<T, int64>(
        input.flat<T>().data(), d_flat_indices, d_num_selected,
        static_cast<int>(num_items), stream);
    Tensor select_workspace_t;
    void* select_workspace_ptr = nullptr;
    if (select_workspace_bytes > 0) {
      OP_REQUIRES_OK(context,
             context->allocate_temp(
               DT_UINT8,
               TensorShape({static_cast<int64>(select_workspace_bytes)}),
               &select_workspace_t));
      select_workspace_ptr =
        static_cast<void*>(select_workspace_t.flat<uint8_t>().data());
    }

    LaunchSelectTrueIndicesKernel<T, int64>(
      input.flat<T>().data(), d_flat_indices, d_num_selected,
      static_cast<int>(num_items), select_workspace_ptr,
      select_workspace_bytes, stream);

    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    alloc_attr.set_gpu_compatible(true);

    // Copy selected row count from device for output shape inference.
    Tensor num_true_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<int64>::value, TensorShape({1}),
                                &num_true_tensor, alloc_attr));

    auto m_err = musaMemcpyAsync(num_true_tensor.flat<int64>().data(),
                   d_num_selected, sizeof(int64),
                   musaMemcpyDeviceToHost, stream);
    OP_REQUIRES(context, m_err == musaSuccess,
          errors::Internal("WhereOp: musaMemcpyAsync failed: ",
                   musaGetErrorString(m_err)));
    m_err = musaStreamSynchronize(stream);
    OP_REQUIRES(context, m_err == musaSuccess,
          errors::Internal("WhereOp: musaStreamSynchronize failed: ",
                   musaGetErrorString(m_err)));

    const int64 num_true = *num_true_tensor.flat<int64>().data();
    if (num_true == 0) {
      Tensor* out = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  0, TensorShape({0, input_dims}), &out));
      return;
    }

    // Next is to compute `where`, given the number of true elements.
    Tensor* output = nullptr;
    TensorShape output_shape;
    OP_REQUIRES_OK(context,
                   output_shape.AddDimWithStatus(static_cast<int64>(num_true)));
    OP_REQUIRES_OK(
        context, output_shape.AddDimWithStatus(static_cast<int64>(input_dims)));
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

#define HANDLE_DIM(NDIM)                                            \
  case NDIM: {                                                      \
    Status where_status = Where::PropagateFromSelectedIndices<NDIM, T, int64>( \
        context, input.tensor<T, NDIM>(), d_flat_indices, num_true,             \
        output->matrix<int64>());                                                \
    OP_REQUIRES_OK(context, where_status);                          \
                                                                    \
  } break
    switch (input_dims) {
      case 0:
        break;  // For a scalar input, output shape is [num_true, 0]. No
                // coordinates to write.
        HANDLE_DIM(1);
        HANDLE_DIM(2);
        HANDLE_DIM(3);
        HANDLE_DIM(4);
        HANDLE_DIM(5);
        HANDLE_DIM(6);
        HANDLE_DIM(7);
        HANDLE_DIM(8);
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "WhereOp: Unhandled input dimensions: ", input_dims));
    }
#undef HANDLE_DIM
  }

  bool IsExpensive() override { return true; }
};

#define REGISTER_MUSA_WHERE_OP(TYPE)                                \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Where").Device(DEVICE_MTGPU).TypeConstraint<TYPE>("T"), \
      MusaWhereOp<TYPE>)

REGISTER_MUSA_WHERE_OP(float);
REGISTER_MUSA_WHERE_OP(double);
REGISTER_MUSA_WHERE_OP(int8);
REGISTER_MUSA_WHERE_OP(uint8);
REGISTER_MUSA_WHERE_OP(int16);
REGISTER_MUSA_WHERE_OP(uint16);
REGISTER_MUSA_WHERE_OP(int32);
REGISTER_MUSA_WHERE_OP(int64);
REGISTER_MUSA_WHERE_OP(bfloat16);
REGISTER_MUSA_WHERE_OP(bool);

#undef REGISTER_MUSA_WHERE_OP

}  // namespace musa
}  // namespace tensorflow
