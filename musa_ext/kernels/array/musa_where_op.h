#ifndef TENSORFLOW_MUSA_KERNELS_ARRAY_MUSA_WHERE_OP_H_
#define TENSORFLOW_MUSA_KERNELS_ARRAY_MUSA_WHERE_OP_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../math/musa_reduce_functor.h"
#include "../utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace musa {

// Count Non Zero within the input tensor
template <typename T, typename TIndex>
size_t GetIsNonZeroCountWorkspaceSize(const T* input, TIndex* output, int n,
                                      musaStream_t stream);

template <typename T, typename TIndex>
void LaunchIsNonZeroCount(const T* input, TIndex* output, int n,
                          void* temp_storage, size_t temp_storage_bytes,
                          musaStream_t stream);

template <typename T, typename TIndex>
size_t GetSelectTrueIndicesWorkspaceSize(const T* input, TIndex* output_indices,
                                         TIndex* num_selected, int num_items,
                                         musaStream_t stream);

template <typename T, typename TIndex>
void LaunchSelectTrueIndicesKernel(const T* input, TIndex* output_indices,
                                   TIndex* num_selected, int num_items,
                                   void* temp_storage,
                                   size_t temp_storage_bytes,
                                   musaStream_t stream);

template <int NDIM, typename TIndex>
void LaunchPropagateWhereIndicesFromFlatKernel(const TIndex* flat_indices,
                         TIndex* output,
                         const TIndex* strides_host,
                         int num_selected,
                         musaStream_t stream);

template <typename T, typename TIndex>
struct NumTrue {
  static Status Compute(OpKernelContext* ctx,
                        typename TTypes<T>::ConstFlat input,
                        typename TTypes<TIndex>::UnalignedScalar num_true) {
    musaStream_t mstream = GetMusaStreamByCtx(ctx);
    const T* input_data = reinterpret_cast<const T*>(input.data());
    TIndex* num_true_data = num_true.data();

    if (input.size() == 0) {
      *num_true_data = static_cast<TIndex>(0);
      return Status::OK();
    }

    Tensor count64_wrapper;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<TIndex>::value,
                                          TensorShape({1}), &count64_wrapper));
    TIndex* count_device = count64_wrapper.flat<TIndex>().data();

    const size_t count_workspace_bytes =
      GetIsNonZeroCountWorkspaceSize<T, TIndex>(
        input_data, count_device, static_cast<int>(input.size()), mstream);
    Tensor count_workspace_t;
    void* count_workspace_ptr = nullptr;
    if (count_workspace_bytes > 0) {
      TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DT_UINT8,
        TensorShape({static_cast<int64_t>(count_workspace_bytes)}),
        &count_workspace_t));
      count_workspace_ptr =
        static_cast<void*>(count_workspace_t.flat<uint8_t>().data());
    }

    LaunchIsNonZeroCount<T, TIndex>(input_data, count_device,
                    static_cast<int>(input.size()),
                    count_workspace_ptr,
                    count_workspace_bytes, mstream);

    auto m_err = musaMemcpyAsync(num_true_data, count_device, sizeof(TIndex),
                                 musaMemcpyDeviceToHost, mstream);
    if (m_err != musaSuccess) {
      return errors::Internal("WhereOp: musaMemcpyAsync failed: ",
                              musaGetErrorString(m_err));
    }
    m_err = musaStreamSynchronize(mstream);
    if (m_err != musaSuccess) {
      return errors::Internal("WhereOp: musaStreamSynchronize failed: ",
                              musaGetErrorString(m_err));
    }

    return Status::OK();
  }
};

template <typename TIndex, typename T, int NDIM>
Eigen::array<TIndex, NDIM> CalculateStrides(
    typename TTypes<T, NDIM>::ConstTensor input) {
  const Eigen::DSizes<Eigen::DenseIndex, NDIM> dims = input.dimensions();
  Eigen::array<TIndex, NDIM> strides;
  EIGEN_STATIC_ASSERT((static_cast<int>(decltype(input)::Layout) ==
                       static_cast<int>(Eigen::RowMajor)),
                      INTERNAL_ERROR_INPUT_SHOULD_BE_ROWMAJOR);
  strides[NDIM - 1] = 1;
  for (int i = NDIM - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

// Be advised: The original TF implementation has an extra template parameter
// called `IsConvertibleToBool`, which considered data types that cannot be
// directly converted to bool, namely complex types. For now we only consider
// real number cases.
struct Where {
  template <int NDIM, typename T, typename TIndex>
  static Status Compute(OpKernelContext* ctx,
                        typename TTypes<T, NDIM>::ConstTensor input,
                        typename TTypes<TIndex>::Matrix output) {
    if (output.dimension(0) == 0) {
      return Status::OK();
    }

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    const int64_t num_items = input.size();

    Tensor flat_indices_t;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
      DataTypeToEnum<TIndex>::value, TensorShape({output.dimension(0)}),
      &flat_indices_t));
    TIndex* d_flat_indices = flat_indices_t.flat<TIndex>().data();

    Tensor selected_count_t;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<TIndex>::value,
                        TensorShape({1}), &selected_count_t));
    TIndex* d_num_selected = selected_count_t.flat<TIndex>().data();

    const size_t select_workspace_bytes =
        GetSelectTrueIndicesWorkspaceSize<T, TIndex>(
            input.data(), d_flat_indices, d_num_selected,
            static_cast<int>(num_items), stream);
    Tensor select_workspace_t;
    void* select_workspace_ptr = nullptr;
    if (select_workspace_bytes > 0) {
      TF_RETURN_IF_ERROR(ctx->allocate_temp(
          DT_UINT8,
          TensorShape({static_cast<int64_t>(select_workspace_bytes)}),
          &select_workspace_t));
      select_workspace_ptr =
          static_cast<void*>(select_workspace_t.flat<uint8_t>().data());
    }

    LaunchSelectTrueIndicesKernel<T, TIndex>(
      input.data(), d_flat_indices, d_num_selected,
      static_cast<int>(num_items), select_workspace_ptr,
      select_workspace_bytes, stream);

    const Eigen::array<TIndex, NDIM> strides =
        CalculateStrides<TIndex, T, NDIM>(input);
    LaunchPropagateWhereIndicesFromFlatKernel<NDIM, TIndex>(
      d_flat_indices, output.data(), strides.data(),
      static_cast<int>(output.dimension(0)), stream);

    return Status::OK();
  }
};

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_KERNELS_ARRAY_MUSA_WHERE_OP_H_
