#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/cub.cuh>

#include <algorithm>
#include <stdexcept>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;

template <typename OffsetT>
const OffsetT* endOffsetsPtr(const Tensor& segment_offsets) {
    return segment_offsets.getMemPtr<OffsetT>() + 1;
}

}  // namespace

void cubDeviceSegmentedRadixSortPairs(const CubDeviceSegmentedRadixSortPairsPlan& plan,
                                      const Tensor& temp_storage,
                                      const Tensor& keys_in,
                                      Tensor& keys_out,
                                      const Tensor& values_in,
                                      Tensor& values_out,
                                      const Tensor& segment_offsets,
                                      Stream& stream) {
    validateSegmentedSortPairs(
        keys_in, keys_out, values_in, values_out, segment_offsets, plan.num_items, plan.num_segments, plan.begin_bit, plan.end_bit);
    if (keys_in.getPlacement() != plan.placement || keys_in.getDataType() != plan.key_dtype ||
        values_in.getDataType() != plan.value_dtype || segment_offsets.getDataType() != plan.offset_dtype) {
        throw std::invalid_argument("CUB segmented radix-sort-pairs plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    const int64_t cub_items = checkedCubInt64Count(plan.num_items, "num_items");
    const int64_t cub_segments = checkedCubInt64Count(plan.num_segments, "num_segments");
    if (plan.num_items == 0 || plan.num_segments == 0) {
        return;
    }

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch_key = [&]<typename KeyT>() -> void {
        auto launch_value = [&]<typename ValueT>() -> void {
            auto launch_offset = [&]<typename OffsetT>() -> void {
                if (plan.order == CubSortOrder::Ascending) {
                    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(temp_storage_ptr,
                                                                        temp_storage_bytes,
                                                                        keys_in.getMemPtr<KeyT>(),
                                                                        keys_out.getMemPtr<KeyT>(),
                                                                        values_in.getMemPtr<ValueT>(),
                                                                        values_out.getMemPtr<ValueT>(),
                                                                        cub_items,
                                                                        cub_segments,
                                                                        segment_offsets.getMemPtr<OffsetT>(),
                                                                        endOffsetsPtr<OffsetT>(segment_offsets),
                                                                        plan.begin_bit,
                                                                        plan.end_bit,
                                                                        stream.getStream()));
                } else {
                    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(temp_storage_ptr,
                                                                                  temp_storage_bytes,
                                                                                  keys_in.getMemPtr<KeyT>(),
                                                                                  keys_out.getMemPtr<KeyT>(),
                                                                                  values_in.getMemPtr<ValueT>(),
                                                                                  values_out.getMemPtr<ValueT>(),
                                                                                  cub_items,
                                                                                  cub_segments,
                                                                                  segment_offsets.getMemPtr<OffsetT>(),
                                                                                  endOffsetsPtr<OffsetT>(segment_offsets),
                                                                                  plan.begin_bit,
                                                                                  plan.end_bit,
                                                                                  stream.getStream()));
                }
            };
            dispatchSegmentOffsetDType(plan.offset_dtype, launch_offset);
        };
        dispatchSortValueDType(plan.value_dtype, launch_value);
    };

    dispatchSortKeyDType(plan.key_dtype, launch_key);
}

void cubDeviceSegmentedRadixSortPairs(const Tensor& temp_storage,
                                      size_t temp_storage_bytes,
                                      const Tensor& keys_in,
                                      Tensor& keys_out,
                                      const Tensor& values_in,
                                      Tensor& values_out,
                                      const Tensor& segment_offsets,
                                      uint64_t num_items,
                                      uint64_t num_segments,
                                      Stream& stream,
                                      CubSortOrder order,
                                      int begin_bit,
                                      int end_bit) {
    CubDeviceSegmentedRadixSortPairsPlan plan = prepareCubDeviceSegmentedRadixSortPairs(
        keys_in, keys_out, values_in, values_out, segment_offsets, num_items, num_segments, order, begin_bit, end_bit);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB segmented radix-sort-pairs requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceSegmentedRadixSortPairs(plan, temp_storage, keys_in, keys_out, values_in, values_out, segment_offsets, stream);
}

}  // namespace ThorImplementation
