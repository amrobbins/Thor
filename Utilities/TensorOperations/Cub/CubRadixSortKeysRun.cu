#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/cub.cuh>

#include <algorithm>
#include <stdexcept>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;
}

void cubDeviceRadixSortKeys(const CubDeviceRadixSortKeysPlan& plan,
                            const Tensor& temp_storage,
                            const Tensor& keys_in,
                            Tensor& keys_out,
                            Stream& stream) {
    validateSortKeys(keys_in, keys_out, plan.num_items, plan.begin_bit, plan.end_bit);
    if (keys_in.getPlacement() != plan.placement || keys_in.getDataType() != plan.key_dtype) {
        throw std::invalid_argument("CUB radix-sort-keys plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    const int cub_items = checkedCubNumItems(plan.num_items);
    if (plan.num_items == 0) {
        return;
    }

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch = [&]<typename KeyT>() -> void {
        if (plan.order == CubSortOrder::Ascending) {
            CUDA_CHECK(cub::DeviceRadixSort::SortKeys(temp_storage_ptr,
                                                      temp_storage_bytes,
                                                      keys_in.getMemPtr<KeyT>(),
                                                      keys_out.getMemPtr<KeyT>(),
                                                      cub_items,
                                                      plan.begin_bit,
                                                      plan.end_bit,
                                                      stream.getStream()));
        } else {
            CUDA_CHECK(cub::DeviceRadixSort::SortKeysDescending(temp_storage_ptr,
                                                                temp_storage_bytes,
                                                                keys_in.getMemPtr<KeyT>(),
                                                                keys_out.getMemPtr<KeyT>(),
                                                                cub_items,
                                                                plan.begin_bit,
                                                                plan.end_bit,
                                                                stream.getStream()));
        }
    };

    dispatchSortKeyDType(plan.key_dtype, launch);
}

void cubDeviceRadixSortKeys(const Tensor& temp_storage,
                            size_t temp_storage_bytes,
                            const Tensor& keys_in,
                            Tensor& keys_out,
                            uint64_t num_items,
                            Stream& stream,
                            CubSortOrder order,
                            int begin_bit,
                            int end_bit) {
    CubDeviceRadixSortKeysPlan plan = prepareCubDeviceRadixSortKeys(keys_in, keys_out, num_items, order, begin_bit, end_bit);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB radix-sort-keys requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceRadixSortKeys(plan, temp_storage, keys_in, keys_out, stream);
}


}  // namespace ThorImplementation
