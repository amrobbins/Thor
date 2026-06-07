#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/cub.cuh>

#include <algorithm>
#include <stdexcept>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;
}

CubDeviceRadixSortKeysPlan prepareCubDeviceRadixSortKeys(const Tensor& keys_in,
                                                         const Tensor& keys_out,
                                                         uint64_t num_items,
                                                         CubSortOrder order,
                                                         int begin_bit,
                                                         int end_bit) {
    end_bit = fullEndBitFor(keys_in.getDataType(), end_bit);
    validateSortKeys(keys_in, keys_out, num_items, begin_bit, end_bit);
    const int cub_items = checkedCubNumItems(num_items);

    size_t bytes = 1;
    if (num_items != 0) {
        auto query = [&]<typename KeyT>() -> size_t {
            size_t queried_bytes = 0;
            if (order == CubSortOrder::Ascending) {
                CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr,
                                                          queried_bytes,
                                                          keys_in.getMemPtr<KeyT>(),
                                                          const_cast<KeyT*>(keys_out.getMemPtr<KeyT>()),
                                                          cub_items,
                                                          begin_bit,
                                                          end_bit));
            } else {
                CUDA_CHECK(cub::DeviceRadixSort::SortKeysDescending(nullptr,
                                                                    queried_bytes,
                                                                    keys_in.getMemPtr<KeyT>(),
                                                                    const_cast<KeyT*>(keys_out.getMemPtr<KeyT>()),
                                                                    cub_items,
                                                                    begin_bit,
                                                                    end_bit));
            }
            return queried_bytes;
        };
        bytes = dispatchSortKeyDType(keys_in.getDataType(), query);
    }

    CubDeviceRadixSortKeysPlan plan;
    plan.placement = keys_in.getPlacement();
    plan.key_dtype = keys_in.getDataType();
    plan.num_items = num_items;
    plan.order = order;
    plan.begin_bit = begin_bit;
    plan.end_bit = end_bit;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceRadixSortKeysTempBytes(const Tensor& keys_in,
                                       const Tensor& keys_out,
                                       uint64_t num_items,
                                       CubSortOrder order,
                                       int begin_bit,
                                       int end_bit) {
    return prepareCubDeviceRadixSortKeys(keys_in, keys_out, num_items, order, begin_bit, end_bit).temp_storage_bytes;
}


}  // namespace ThorImplementation
