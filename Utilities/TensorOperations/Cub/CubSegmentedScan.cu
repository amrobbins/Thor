#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/device/device_segmented_scan.cuh>

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

CubDeviceSegmentedExclusiveSumPlan prepareCubDeviceSegmentedExclusiveSum(const Tensor& input,
                                                                         const Tensor& output,
                                                                         const Tensor& segment_offsets,
                                                                         uint64_t num_items,
                                                                         uint64_t num_segments) {
    validateSegmentedExclusiveSum(input, output, segment_offsets, num_items, num_segments);
    const int64_t cub_segments = checkedCubInt64Count(num_segments, "num_segments");

    size_t bytes = 1;
    if (num_items != 0 && num_segments != 0) {
        auto query_value = [&]<typename T>() -> size_t {
            auto query_offset = [&]<typename OffsetT>() -> size_t {
                size_t queried_bytes = 0;
                CUDA_CHECK(cub::DeviceSegmentedScan::ExclusiveSegmentedSum(nullptr,
                                                                           queried_bytes,
                                                                           input.getMemPtr<T>(),
                                                                           const_cast<T*>(output.getMemPtr<T>()),
                                                                           segment_offsets.getMemPtr<OffsetT>(),
                                                                           endOffsetsPtr<OffsetT>(segment_offsets),
                                                                           cub_segments));
                return queried_bytes;
            };
            return dispatchSegmentOffsetDType(segment_offsets.getDataType(), query_offset);
        };
        bytes = dispatchScanDType(input.getDataType(), query_value);
    }

    CubDeviceSegmentedExclusiveSumPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.offset_dtype = segment_offsets.getDataType();
    plan.num_items = num_items;
    plan.num_segments = num_segments;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceSegmentedExclusiveSumTempBytes(const Tensor& input,
                                               const Tensor& output,
                                               const Tensor& segment_offsets,
                                               uint64_t num_items,
                                               uint64_t num_segments) {
    return prepareCubDeviceSegmentedExclusiveSum(input, output, segment_offsets, num_items, num_segments).temp_storage_bytes;
}

void cubDeviceSegmentedExclusiveSum(const CubDeviceSegmentedExclusiveSumPlan& plan,
                                    const Tensor& temp_storage,
                                    const Tensor& input,
                                    Tensor& output,
                                    const Tensor& segment_offsets,
                                    Stream& stream) {
    validateSegmentedExclusiveSum(input, output, segment_offsets, plan.num_items, plan.num_segments);
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype ||
        segment_offsets.getDataType() != plan.offset_dtype) {
        throw std::invalid_argument("CUB segmented exclusive-sum plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    const int64_t cub_segments = checkedCubInt64Count(plan.num_segments, "num_segments");
    if (plan.num_items == 0 || plan.num_segments == 0) {
        return;
    }

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch_value = [&]<typename T>() -> void {
        auto launch_offset = [&]<typename OffsetT>() -> void {
            CUDA_CHECK(cub::DeviceSegmentedScan::ExclusiveSegmentedSum(temp_storage_ptr,
                                                                       temp_storage_bytes,
                                                                       input.getMemPtr<T>(),
                                                                       output.getMemPtr<T>(),
                                                                       segment_offsets.getMemPtr<OffsetT>(),
                                                                       endOffsetsPtr<OffsetT>(segment_offsets),
                                                                       cub_segments,
                                                                       stream.getStream()));
        };
        dispatchSegmentOffsetDType(plan.offset_dtype, launch_offset);
    };

    dispatchScanDType(plan.dtype, launch_value);
}

void cubDeviceSegmentedExclusiveSum(const Tensor& temp_storage,
                                    size_t temp_storage_bytes,
                                    const Tensor& input,
                                    Tensor& output,
                                    const Tensor& segment_offsets,
                                    uint64_t num_items,
                                    uint64_t num_segments,
                                    Stream& stream) {
    CubDeviceSegmentedExclusiveSumPlan plan = prepareCubDeviceSegmentedExclusiveSum(input, output, segment_offsets, num_items, num_segments);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB segmented exclusive-sum requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceSegmentedExclusiveSum(plan, temp_storage, input, output, segment_offsets, stream);
}

}  // namespace ThorImplementation
