#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/cub.cuh>

#include <algorithm>
#include <stdexcept>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;
}

void cubDeviceRadixSortPairs(const CubDeviceRadixSortPairsPlan& plan,
                             const Tensor& temp_storage,
                             const Tensor& keys_in,
                             Tensor& keys_out,
                             const Tensor& values_in,
                             Tensor& values_out,
                             Stream& stream) {
    validateSortPairs(keys_in, keys_out, values_in, values_out, plan.num_items, plan.begin_bit, plan.end_bit);
    if (keys_in.getPlacement() != plan.placement || keys_in.getDataType() != plan.key_dtype || values_in.getDataType() != plan.value_dtype) {
        throw std::invalid_argument("CUB radix-sort-pairs plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    const int cub_items = checkedCubNumItems(plan.num_items);
    if (plan.num_items == 0) {
        return;
    }

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch_key = [&]<typename KeyT>() -> void {
        auto launch_value = [&]<typename ValueT>() -> void {
            if (plan.order == CubSortOrder::Ascending) {
                CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp_storage_ptr,
                                                           temp_storage_bytes,
                                                           keys_in.getMemPtr<KeyT>(),
                                                           keys_out.getMemPtr<KeyT>(),
                                                           values_in.getMemPtr<ValueT>(),
                                                           values_out.getMemPtr<ValueT>(),
                                                           cub_items,
                                                           plan.begin_bit,
                                                           plan.end_bit,
                                                           stream.getStream()));
            } else {
                CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(temp_storage_ptr,
                                                                     temp_storage_bytes,
                                                                     keys_in.getMemPtr<KeyT>(),
                                                                     keys_out.getMemPtr<KeyT>(),
                                                                     values_in.getMemPtr<ValueT>(),
                                                                     values_out.getMemPtr<ValueT>(),
                                                                     cub_items,
                                                                     plan.begin_bit,
                                                                     plan.end_bit,
                                                                     stream.getStream()));
            }
        };
        dispatchSortValueDType(plan.value_dtype, launch_value);
    };

    dispatchSortKeyDType(plan.key_dtype, launch_key);
}

void cubDeviceRadixSortPairs(const Tensor& temp_storage,
                             size_t temp_storage_bytes,
                             const Tensor& keys_in,
                             Tensor& keys_out,
                             const Tensor& values_in,
                             Tensor& values_out,
                             uint64_t num_items,
                             Stream& stream,
                             CubSortOrder order,
                             int begin_bit,
                             int end_bit) {
    CubDeviceRadixSortPairsPlan plan = prepareCubDeviceRadixSortPairs(keys_in, keys_out, values_in, values_out, num_items, order, begin_bit, end_bit);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB radix-sort-pairs requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceRadixSortPairs(plan, temp_storage, keys_in, keys_out, values_in, values_out, stream);
}

}  // namespace ThorImplementation
