#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/cub.cuh>

#include <algorithm>
#include <stdexcept>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;
}

CubDeviceRadixSortPairsPlan prepareCubDeviceRadixSortPairs(const Tensor& keys_in,
                                                           const Tensor& keys_out,
                                                           const Tensor& values_in,
                                                           const Tensor& values_out,
                                                           uint64_t num_items,
                                                           CubSortOrder order,
                                                           int begin_bit,
                                                           int end_bit) {
    end_bit = fullEndBitFor(keys_in.getDataType(), end_bit);
    validateSortPairs(keys_in, keys_out, values_in, values_out, num_items, begin_bit, end_bit);
    const int cub_items = checkedCubNumItems(num_items);

    size_t bytes = 1;
    if (num_items != 0) {
        auto query_key = [&]<typename KeyT>() -> size_t {
            auto query_value = [&]<typename ValueT>() -> size_t {
                size_t queried_bytes = 0;
                if (order == CubSortOrder::Ascending) {
                    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,
                                                               queried_bytes,
                                                               keys_in.getMemPtr<KeyT>(),
                                                               const_cast<KeyT*>(keys_out.getMemPtr<KeyT>()),
                                                               values_in.getMemPtr<ValueT>(),
                                                               const_cast<ValueT*>(values_out.getMemPtr<ValueT>()),
                                                               cub_items,
                                                               begin_bit,
                                                               end_bit));
                } else {
                    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(nullptr,
                                                                         queried_bytes,
                                                                         keys_in.getMemPtr<KeyT>(),
                                                                         const_cast<KeyT*>(keys_out.getMemPtr<KeyT>()),
                                                                         values_in.getMemPtr<ValueT>(),
                                                                         const_cast<ValueT*>(values_out.getMemPtr<ValueT>()),
                                                                         cub_items,
                                                                         begin_bit,
                                                                         end_bit));
                }
                return queried_bytes;
            };
            return dispatchSortValueDType(values_in.getDataType(), query_value);
        };
        bytes = dispatchSortKeyDType(keys_in.getDataType(), query_key);
    }

    CubDeviceRadixSortPairsPlan plan;
    plan.placement = keys_in.getPlacement();
    plan.key_dtype = keys_in.getDataType();
    plan.value_dtype = values_in.getDataType();
    plan.num_items = num_items;
    plan.order = order;
    plan.begin_bit = begin_bit;
    plan.end_bit = end_bit;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceRadixSortPairsTempBytes(const Tensor& keys_in,
                                        const Tensor& keys_out,
                                        const Tensor& values_in,
                                        const Tensor& values_out,
                                        uint64_t num_items,
                                        CubSortOrder order,
                                        int begin_bit,
                                        int end_bit) {
    return prepareCubDeviceRadixSortPairs(keys_in, keys_out, values_in, values_out, num_items, order, begin_bit, end_bit).temp_storage_bytes;
}


}  // namespace ThorImplementation
