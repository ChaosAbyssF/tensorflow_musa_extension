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
  static Status PropagateFromSelectedIndices(
      OpKernelContext* ctx, typename TTypes<T, NDIM>::ConstTensor input,
      const TIndex* d_flat_indices, TIndex num_selected,
      typename TTypes<TIndex>::Matrix output) {
    if (output.dimension(0) == 0) {
      return Status::OK();
    }

    musaStream_t stream = GetMusaStreamByCtx(ctx);

    const Eigen::array<TIndex, NDIM> strides =
        CalculateStrides<TIndex, T, NDIM>(input);
    LaunchPropagateWhereIndicesFromFlatKernel<NDIM, TIndex>(
      d_flat_indices, output.data(), strides.data(),
      static_cast<int>(num_selected), stream);

    return Status::OK();
  }
};

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_KERNELS_ARRAY_MUSA_WHERE_OP_H_
