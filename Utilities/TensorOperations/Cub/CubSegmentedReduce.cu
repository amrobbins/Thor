#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/device/device_segmented_reduce.cuh>

#include <algorithm>
#include <stdexcept>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;

template <typename OffsetT>
const OffsetT* endOffsetsPtr(const Tensor& segment_offsets) {
    return segment_offsets.getMemPtr<OffsetT>() + 1;
}

template <typename T, typename OffsetT>
size_t querySegmentedReduceSumBytes(const Tensor& input,
                                    const Tensor& output,
                                    const Tensor& segment_offsets,
                                    int64_t cub_segments) {
    size_t queried_bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(nullptr,
                                               queried_bytes,
                                               input.getMemPtr<T>(),
                                               const_cast<T*>(output.getMemPtr<T>()),
                                               cub_segments,
                                               segment_offsets.getMemPtr<OffsetT>(),
                                               endOffsetsPtr<OffsetT>(segment_offsets)));
    return queried_bytes;
}

template <typename T, typename OffsetT>
size_t querySegmentedReduceMaxBytes(const Tensor& input,
                                    const Tensor& output,
                                    const Tensor& segment_offsets,
                                    int64_t cub_segments) {
    size_t queried_bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedReduce::Max(nullptr,
                                               queried_bytes,
                                               input.getMemPtr<T>(),
                                               const_cast<T*>(output.getMemPtr<T>()),
                                               cub_segments,
                                               segment_offsets.getMemPtr<OffsetT>(),
                                               endOffsetsPtr<OffsetT>(segment_offsets)));
    return queried_bytes;
}

template <typename T, typename OffsetT>
size_t querySegmentedReduceMinBytes(const Tensor& input,
                                    const Tensor& output,
                                    const Tensor& segment_offsets,
                                    int64_t cub_segments) {
    size_t queried_bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedReduce::Min(nullptr,
                                               queried_bytes,
                                               input.getMemPtr<T>(),
                                               const_cast<T*>(output.getMemPtr<T>()),
                                               cub_segments,
                                               segment_offsets.getMemPtr<OffsetT>(),
                                               endOffsetsPtr<OffsetT>(segment_offsets)));
    return queried_bytes;
}

template <typename T, typename OffsetT>
void launchSegmentedReduceSum(void* temp_storage_ptr,
                              size_t temp_storage_bytes,
                              const Tensor& input,
                              Tensor& output,
                              const Tensor& segment_offsets,
                              int64_t cub_segments,
                              Stream& stream) {
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(temp_storage_ptr,
                                               temp_storage_bytes,
                                               input.getMemPtr<T>(),
                                               output.getMemPtr<T>(),
                                               cub_segments,
                                               segment_offsets.getMemPtr<OffsetT>(),
                                               endOffsetsPtr<OffsetT>(segment_offsets),
                                               stream.getStream()));
}

template <typename T, typename OffsetT>
void launchSegmentedReduceMax(void* temp_storage_ptr,
                              size_t temp_storage_bytes,
                              const Tensor& input,
                              Tensor& output,
                              const Tensor& segment_offsets,
                              int64_t cub_segments,
                              Stream& stream) {
    CUDA_CHECK(cub::DeviceSegmentedReduce::Max(temp_storage_ptr,
                                               temp_storage_bytes,
                                               input.getMemPtr<T>(),
                                               output.getMemPtr<T>(),
                                               cub_segments,
                                               segment_offsets.getMemPtr<OffsetT>(),
                                               endOffsetsPtr<OffsetT>(segment_offsets),
                                               stream.getStream()));
}

template <typename T, typename OffsetT>
void launchSegmentedReduceMin(void* temp_storage_ptr,
                              size_t temp_storage_bytes,
                              const Tensor& input,
                              Tensor& output,
                              const Tensor& segment_offsets,
                              int64_t cub_segments,
                              Stream& stream) {
    CUDA_CHECK(cub::DeviceSegmentedReduce::Min(temp_storage_ptr,
                                               temp_storage_bytes,
                                               input.getMemPtr<T>(),
                                               output.getMemPtr<T>(),
                                               cub_segments,
                                               segment_offsets.getMemPtr<OffsetT>(),
                                               endOffsetsPtr<OffsetT>(segment_offsets),
                                               stream.getStream()));
}

}  // namespace

CubDeviceSegmentedReduceSumPlan prepareCubDeviceSegmentedReduceSum(const Tensor& input,
                                                                   const Tensor& output,
                                                                   const Tensor& segment_offsets,
                                                                   uint64_t num_items,
                                                                   uint64_t num_segments) {
    validateSegmentedReduceSum(input, output, segment_offsets, num_items, num_segments);
    const int64_t cub_segments = checkedCubInt64Count(num_segments, "num_segments");

    size_t bytes = 1;
    if (num_items != 0 && num_segments != 0) {
        auto query_value = [&]<typename T>() -> size_t {
            auto query_offset = [&]<typename OffsetT>() -> size_t {
                return querySegmentedReduceSumBytes<T, OffsetT>(input, output, segment_offsets, cub_segments);
            };
            return dispatchSegmentOffsetDType(segment_offsets.getDataType(), query_offset);
        };
        bytes = dispatchReduceDType(input.getDataType(), query_value);
    }

    CubDeviceSegmentedReduceSumPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.offset_dtype = segment_offsets.getDataType();
    plan.num_items = num_items;
    plan.num_segments = num_segments;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceSegmentedReduceSumTempBytes(const Tensor& input,
                                            const Tensor& output,
                                            const Tensor& segment_offsets,
                                            uint64_t num_items,
                                            uint64_t num_segments) {
    return prepareCubDeviceSegmentedReduceSum(input, output, segment_offsets, num_items, num_segments).temp_storage_bytes;
}

void cubDeviceSegmentedReduceSum(const CubDeviceSegmentedReduceSumPlan& plan,
                                 const Tensor& temp_storage,
                                 const Tensor& input,
                                 Tensor& output,
                                 const Tensor& segment_offsets,
                                 Stream& stream) {
    validateSegmentedReduceSum(input, output, segment_offsets, plan.num_items, plan.num_segments);
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype ||
        segment_offsets.getDataType() != plan.offset_dtype) {
        throw std::invalid_argument("CUB segmented reduce-sum plan is not compatible with the provided tensors.");
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
            launchSegmentedReduceSum<T, OffsetT>(
                temp_storage_ptr, temp_storage_bytes, input, output, segment_offsets, cub_segments, stream);
        };
        dispatchSegmentOffsetDType(plan.offset_dtype, launch_offset);
    };

    dispatchReduceDType(plan.dtype, launch_value);
}

void cubDeviceSegmentedReduceSum(const Tensor& temp_storage,
                                 size_t temp_storage_bytes,
                                 const Tensor& input,
                                 Tensor& output,
                                 const Tensor& segment_offsets,
                                 uint64_t num_items,
                                 uint64_t num_segments,
                                 Stream& stream) {
    CubDeviceSegmentedReduceSumPlan plan = prepareCubDeviceSegmentedReduceSum(input, output, segment_offsets, num_items, num_segments);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB segmented reduce-sum requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceSegmentedReduceSum(plan, temp_storage, input, output, segment_offsets, stream);
}

CubDeviceSegmentedReduceMaxPlan prepareCubDeviceSegmentedReduceMax(const Tensor& input,
                                                                   const Tensor& output,
                                                                   const Tensor& segment_offsets,
                                                                   uint64_t num_items,
                                                                   uint64_t num_segments) {
    validateSegmentedReduceMax(input, output, segment_offsets, num_items, num_segments);
    const int64_t cub_segments = checkedCubInt64Count(num_segments, "num_segments");

    size_t bytes = 1;
    if (num_items != 0 && num_segments != 0) {
        auto query_value = [&]<typename T>() -> size_t {
            auto query_offset = [&]<typename OffsetT>() -> size_t {
                return querySegmentedReduceMaxBytes<T, OffsetT>(input, output, segment_offsets, cub_segments);
            };
            return dispatchSegmentOffsetDType(segment_offsets.getDataType(), query_offset);
        };
        bytes = dispatchReduceDType(input.getDataType(), query_value);
    }

    CubDeviceSegmentedReduceMaxPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.offset_dtype = segment_offsets.getDataType();
    plan.num_items = num_items;
    plan.num_segments = num_segments;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceSegmentedReduceMaxTempBytes(const Tensor& input,
                                            const Tensor& output,
                                            const Tensor& segment_offsets,
                                            uint64_t num_items,
                                            uint64_t num_segments) {
    return prepareCubDeviceSegmentedReduceMax(input, output, segment_offsets, num_items, num_segments).temp_storage_bytes;
}

void cubDeviceSegmentedReduceMax(const CubDeviceSegmentedReduceMaxPlan& plan,
                                 const Tensor& temp_storage,
                                 const Tensor& input,
                                 Tensor& output,
                                 const Tensor& segment_offsets,
                                 Stream& stream) {
    validateSegmentedReduceMax(input, output, segment_offsets, plan.num_items, plan.num_segments);
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype ||
        segment_offsets.getDataType() != plan.offset_dtype) {
        throw std::invalid_argument("CUB segmented reduce-max plan is not compatible with the provided tensors.");
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
            launchSegmentedReduceMax<T, OffsetT>(
                temp_storage_ptr, temp_storage_bytes, input, output, segment_offsets, cub_segments, stream);
        };
        dispatchSegmentOffsetDType(plan.offset_dtype, launch_offset);
    };

    dispatchReduceDType(plan.dtype, launch_value);
}

void cubDeviceSegmentedReduceMax(const Tensor& temp_storage,
                                 size_t temp_storage_bytes,
                                 const Tensor& input,
                                 Tensor& output,
                                 const Tensor& segment_offsets,
                                 uint64_t num_items,
                                 uint64_t num_segments,
                                 Stream& stream) {
    CubDeviceSegmentedReduceMaxPlan plan = prepareCubDeviceSegmentedReduceMax(input, output, segment_offsets, num_items, num_segments);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB segmented reduce-max requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceSegmentedReduceMax(plan, temp_storage, input, output, segment_offsets, stream);
}


CubDeviceSegmentedReduceMinPlan prepareCubDeviceSegmentedReduceMin(const Tensor& input,
                                                                   const Tensor& output,
                                                                   const Tensor& segment_offsets,
                                                                   uint64_t num_items,
                                                                   uint64_t num_segments) {
    validateSegmentedReduceMin(input, output, segment_offsets, num_items, num_segments);
    const int64_t cub_segments = checkedCubInt64Count(num_segments, "num_segments");

    size_t bytes = 1;
    if (num_items != 0 && num_segments != 0) {
        auto query_value = [&]<typename T>() -> size_t {
            auto query_offset = [&]<typename OffsetT>() -> size_t {
                return querySegmentedReduceMinBytes<T, OffsetT>(input, output, segment_offsets, cub_segments);
            };
            return dispatchSegmentOffsetDType(segment_offsets.getDataType(), query_offset);
        };
        bytes = dispatchReduceDType(input.getDataType(), query_value);
    }

    CubDeviceSegmentedReduceMinPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.offset_dtype = segment_offsets.getDataType();
    plan.num_items = num_items;
    plan.num_segments = num_segments;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceSegmentedReduceMinTempBytes(const Tensor& input,
                                            const Tensor& output,
                                            const Tensor& segment_offsets,
                                            uint64_t num_items,
                                            uint64_t num_segments) {
    return prepareCubDeviceSegmentedReduceMin(input, output, segment_offsets, num_items, num_segments).temp_storage_bytes;
}

void cubDeviceSegmentedReduceMin(const CubDeviceSegmentedReduceMinPlan& plan,
                                 const Tensor& temp_storage,
                                 const Tensor& input,
                                 Tensor& output,
                                 const Tensor& segment_offsets,
                                 Stream& stream) {
    validateSegmentedReduceMin(input, output, segment_offsets, plan.num_items, plan.num_segments);
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype ||
        segment_offsets.getDataType() != plan.offset_dtype) {
        throw std::invalid_argument("CUB segmented reduce-min plan is not compatible with the provided tensors.");
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
            launchSegmentedReduceMin<T, OffsetT>(
                temp_storage_ptr, temp_storage_bytes, input, output, segment_offsets, cub_segments, stream);
        };
        dispatchSegmentOffsetDType(plan.offset_dtype, launch_offset);
    };

    dispatchReduceDType(plan.dtype, launch_value);
}

void cubDeviceSegmentedReduceMin(const Tensor& temp_storage,
                                 size_t temp_storage_bytes,
                                 const Tensor& input,
                                 Tensor& output,
                                 const Tensor& segment_offsets,
                                 uint64_t num_items,
                                 uint64_t num_segments,
                                 Stream& stream) {
    CubDeviceSegmentedReduceMinPlan plan = prepareCubDeviceSegmentedReduceMin(input, output, segment_offsets, num_items, num_segments);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB segmented reduce-min requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceSegmentedReduceMin(plan, temp_storage, input, output, segment_offsets, stream);
}

}  // namespace ThorImplementation
