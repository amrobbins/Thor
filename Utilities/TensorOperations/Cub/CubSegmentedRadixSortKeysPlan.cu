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

CubDeviceSegmentedRadixSortKeysPlan prepareCubDeviceSegmentedRadixSortKeys(const Tensor& keys_in,
                                                                           const Tensor& keys_out,
                                                                           const Tensor& segment_offsets,
                                                                           uint64_t num_items,
                                                                           uint64_t num_segments,
                                                                           CubSortOrder order,
                                                                           int begin_bit,
                                                                           int end_bit) {
    end_bit = fullEndBitFor(keys_in.getDataType(), end_bit);
    validateSegmentedSortKeys(keys_in, keys_out, segment_offsets, num_items, num_segments, begin_bit, end_bit);
    const int64_t cub_items = checkedCubInt64Count(num_items, "num_items");
    const int64_t cub_segments = checkedCubInt64Count(num_segments, "num_segments");

    size_t bytes = 1;
    if (num_items != 0 && num_segments != 0) {
        auto query_key = [&]<typename KeyT>() -> size_t {
            auto query_offset = [&]<typename OffsetT>() -> size_t {
                size_t queried_bytes = 0;
                if (order == CubSortOrder::Ascending) {
                    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(nullptr,
                                                                       queried_bytes,
                                                                       keys_in.getMemPtr<KeyT>(),
                                                                       const_cast<KeyT*>(keys_out.getMemPtr<KeyT>()),
                                                                       cub_items,
                                                                       cub_segments,
                                                                       segment_offsets.getMemPtr<OffsetT>(),
                                                                       endOffsetsPtr<OffsetT>(segment_offsets),
                                                                       begin_bit,
                                                                       end_bit));
                } else {
                    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeysDescending(nullptr,
                                                                                 queried_bytes,
                                                                                 keys_in.getMemPtr<KeyT>(),
                                                                                 const_cast<KeyT*>(keys_out.getMemPtr<KeyT>()),
                                                                                 cub_items,
                                                                                 cub_segments,
                                                                                 segment_offsets.getMemPtr<OffsetT>(),
                                                                                 endOffsetsPtr<OffsetT>(segment_offsets),
                                                                                 begin_bit,
                                                                                 end_bit));
                }
                return queried_bytes;
            };
            return dispatchSegmentOffsetDType(segment_offsets.getDataType(), query_offset);
        };
        bytes = dispatchSortKeyDType(keys_in.getDataType(), query_key);
    }

    CubDeviceSegmentedRadixSortKeysPlan plan;
    plan.placement = keys_in.getPlacement();
    plan.key_dtype = keys_in.getDataType();
    plan.offset_dtype = segment_offsets.getDataType();
    plan.num_items = num_items;
    plan.num_segments = num_segments;
    plan.order = order;
    plan.begin_bit = begin_bit;
    plan.end_bit = end_bit;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceSegmentedRadixSortKeysTempBytes(const Tensor& keys_in,
                                                const Tensor& keys_out,
                                                const Tensor& segment_offsets,
                                                uint64_t num_items,
                                                uint64_t num_segments,
                                                CubSortOrder order,
                                                int begin_bit,
                                                int end_bit) {
    return prepareCubDeviceSegmentedRadixSortKeys(
               keys_in, keys_out, segment_offsets, num_items, num_segments, order, begin_bit, end_bit)
        .temp_storage_bytes;
}


}  // namespace ThorImplementation
