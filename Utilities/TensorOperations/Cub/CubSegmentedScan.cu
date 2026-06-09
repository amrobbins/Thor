#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/device/device_segmented_scan.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;

template <typename OffsetT>
const OffsetT* endOffsetsPtr(const Tensor& segment_offsets) {
    return segment_offsets.getMemPtr<OffsetT>() + 1;
}

template <typename OffsetT>
struct UniformSegmentOffsetOp {
    OffsetT segment_size;

    __host__ __device__ OffsetT operator()(OffsetT segment) const { return segment * segment_size; }
};

template <typename OffsetT>
auto makeUniformSegmentBeginOffsets(OffsetT segment_size) {
    return thrust::make_transform_iterator(thrust::counting_iterator<OffsetT>(0), UniformSegmentOffsetOp<OffsetT>{segment_size});
}

void validateUniformSegmentedSum(const Tensor& input,
                                 const Tensor& output,
                                 uint64_t num_items,
                                 uint64_t num_segments,
                                 uint64_t segment_size,
                                 const char* op_name) {
    validateExclusiveSum(input, output, num_items);
    if (num_segments == 0) {
        if (num_items != 0) {
            throw std::invalid_argument(std::string(op_name) + " requires num_items to be zero when num_segments is zero.");
        }
        return;
    }
    if (segment_size == 0) {
        throw std::invalid_argument(std::string(op_name) + " requires non-zero segment_size when num_segments is non-zero.");
    }
    if (num_segments > std::numeric_limits<uint64_t>::max() / segment_size || num_segments * segment_size != num_items) {
        throw std::invalid_argument(std::string(op_name) + " requires num_items == num_segments * segment_size.");
    }
    static_cast<void>(checkedCubInt64Count(num_segments, "num_segments"));
    static_cast<void>(checkedCubInt64Count(segment_size, "segment_size"));
}



template <typename T>
__host__ __device__ float segmentedScanToFloat(T value) {
    return static_cast<float>(value);
}

template <>
__host__ __device__ float segmentedScanToFloat<__half>(__half value) {
    return __half2float(value);
}

template <>
__host__ __device__ float segmentedScanToFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__host__ __device__ T segmentedScanFromFloat(float value) {
    return static_cast<T>(value);
}

template <>
__host__ __device__ __half segmentedScanFromFloat<__half>(float value) {
    return __float2half(value);
}

template <>
__host__ __device__ __nv_bfloat16 segmentedScanFromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename T>
struct CubSegmentedScanSumOp {
    __host__ __device__ T operator()(const T& a, const T& b) const {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            return segmentedScanFromFloat<T>(segmentedScanToFloat(a) + segmentedScanToFloat(b));
        } else {
            return a + b;
        }
    }
};

template <typename T>
struct CubSegmentedScanProductOp {
    __host__ __device__ T operator()(const T& a, const T& b) const {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            return segmentedScanFromFloat<T>(segmentedScanToFloat(a) * segmentedScanToFloat(b));
        } else {
            return a * b;
        }
    }
};

template <typename T>
struct CubSegmentedScanMinOp {
    __host__ __device__ T operator()(const T& a, const T& b) const {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            return segmentedScanToFloat(b) < segmentedScanToFloat(a) ? b : a;
        } else {
            return b < a ? b : a;
        }
    }
};

template <typename T>
struct CubSegmentedScanMaxOp {
    __host__ __device__ T operator()(const T& a, const T& b) const {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            return segmentedScanToFloat(a) < segmentedScanToFloat(b) ? b : a;
        } else {
            return a < b ? b : a;
        }
    }
};

template <typename T>
T segmentedScanZero() {
    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
        return segmentedScanFromFloat<T>(0.0f);
    } else {
        return T{0};
    }
}

template <typename T>
T segmentedScanOne() {
    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
        return segmentedScanFromFloat<T>(1.0f);
    } else {
        return T{1};
    }
}

template <typename T>
T segmentedScanPositiveInfinityOrMax() {
    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
        return segmentedScanFromFloat<T>(std::numeric_limits<float>::infinity());
    } else if constexpr (std::is_floating_point_v<T>) {
        return std::numeric_limits<T>::infinity();
    } else {
        return std::numeric_limits<T>::max();
    }
}

template <typename T>
T segmentedScanNegativeInfinityOrLowest() {
    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
        return segmentedScanFromFloat<T>(-std::numeric_limits<float>::infinity());
    } else if constexpr (std::is_floating_point_v<T>) {
        return -std::numeric_limits<T>::infinity();
    } else {
        return std::numeric_limits<T>::lowest();
    }
}

template <typename T>
T segmentedScanIdentity(CubScanOp op) {
    switch (op) {
        case CubScanOp::Sum:
            return segmentedScanZero<T>();
        case CubScanOp::Product:
            return segmentedScanOne<T>();
        case CubScanOp::Min:
            return segmentedScanPositiveInfinityOrMax<T>();
        case CubScanOp::Max:
            return segmentedScanNegativeInfinityOrLowest<T>();
    }
    throw std::invalid_argument("Unsupported CUB segmented scan op.");
}

template <typename T, typename Fn>
decltype(auto) dispatchSegmentedScanOperator(CubScanOp op, Fn&& fn) {
    switch (op) {
        case CubScanOp::Sum:
            return fn(CubSegmentedScanSumOp<T>{}, segmentedScanIdentity<T>(op));
        case CubScanOp::Min:
            return fn(CubSegmentedScanMinOp<T>{}, segmentedScanIdentity<T>(op));
        case CubScanOp::Max:
            return fn(CubSegmentedScanMaxOp<T>{}, segmentedScanIdentity<T>(op));
        case CubScanOp::Product:
            return fn(CubSegmentedScanProductOp<T>{}, segmentedScanIdentity<T>(op));
    }
    throw std::invalid_argument("Unsupported CUB segmented scan op.");
}

template <typename T, typename ScanOpT>
size_t queryUniformSegmentedScan(const Tensor& input,
                                 const Tensor& output,
                                 uint64_t num_segments,
                                 uint64_t segment_size,
                                 CubScanMode mode,
                                 ScanOpT scan_op,
                                 T init) {
    const int64_t cub_segments = checkedCubInt64Count(num_segments, "num_segments");
    const int64_t cub_segment_size = checkedCubInt64Count(segment_size, "segment_size");
    auto begin_offsets = makeUniformSegmentBeginOffsets<int64_t>(cub_segment_size);
    auto end_offsets = begin_offsets + 1;
    size_t queried_bytes = 0;
    if (mode == CubScanMode::Exclusive) {
        CUDA_CHECK(cub::DeviceSegmentedScan::ExclusiveSegmentedScan(nullptr,
                                                                    queried_bytes,
                                                                    input.getMemPtr<T>(),
                                                                    const_cast<T*>(output.getMemPtr<T>()),
                                                                    begin_offsets,
                                                                    end_offsets,
                                                                    cub_segments,
                                                                    scan_op,
                                                                    init));
    } else if (mode == CubScanMode::Inclusive) {
        CUDA_CHECK(cub::DeviceSegmentedScan::InclusiveSegmentedScan(nullptr,
                                                                    queried_bytes,
                                                                    input.getMemPtr<T>(),
                                                                    const_cast<T*>(output.getMemPtr<T>()),
                                                                    begin_offsets,
                                                                    end_offsets,
                                                                    cub_segments,
                                                                    scan_op));
    } else {
        throw std::invalid_argument("Unsupported CUB segmented scan mode.");
    }
    return queried_bytes;
}

template <typename T, typename ScanOpT>
void launchUniformSegmentedScan(const CubDeviceSegmentedUniformScanPlan& plan,
                                const Tensor& temp_storage,
                                const Tensor& input,
                                Tensor& output,
                                Stream& stream,
                                ScanOpT scan_op,
                                T init) {
    const int64_t cub_segments = checkedCubInt64Count(plan.num_segments, "num_segments");
    const int64_t cub_segment_size = checkedCubInt64Count(plan.segment_size, "segment_size");
    auto begin_offsets = makeUniformSegmentBeginOffsets<int64_t>(cub_segment_size);
    auto end_offsets = begin_offsets + 1;
    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;
    if (plan.mode == CubScanMode::Exclusive) {
        CUDA_CHECK(cub::DeviceSegmentedScan::ExclusiveSegmentedScan(temp_storage_ptr,
                                                                    temp_storage_bytes,
                                                                    input.getMemPtr<T>(),
                                                                    output.getMemPtr<T>(),
                                                                    begin_offsets,
                                                                    end_offsets,
                                                                    cub_segments,
                                                                    scan_op,
                                                                    init,
                                                                    stream.getStream()));
    } else if (plan.mode == CubScanMode::Inclusive) {
        CUDA_CHECK(cub::DeviceSegmentedScan::InclusiveSegmentedScan(temp_storage_ptr,
                                                                    temp_storage_bytes,
                                                                    input.getMemPtr<T>(),
                                                                    output.getMemPtr<T>(),
                                                                    begin_offsets,
                                                                    end_offsets,
                                                                    cub_segments,
                                                                    scan_op,
                                                                    stream.getStream()));
    } else {
        throw std::invalid_argument("Unsupported CUB segmented scan mode.");
    }
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

CubDeviceSegmentedInclusiveSumPlan prepareCubDeviceSegmentedInclusiveSum(const Tensor& input,
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
                CUDA_CHECK(cub::DeviceSegmentedScan::InclusiveSegmentedSum(nullptr,
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

    CubDeviceSegmentedInclusiveSumPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.offset_dtype = segment_offsets.getDataType();
    plan.num_items = num_items;
    plan.num_segments = num_segments;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceSegmentedInclusiveSumTempBytes(const Tensor& input,
                                               const Tensor& output,
                                               const Tensor& segment_offsets,
                                               uint64_t num_items,
                                               uint64_t num_segments) {
    return prepareCubDeviceSegmentedInclusiveSum(input, output, segment_offsets, num_items, num_segments).temp_storage_bytes;
}

void cubDeviceSegmentedInclusiveSum(const CubDeviceSegmentedInclusiveSumPlan& plan,
                                    const Tensor& temp_storage,
                                    const Tensor& input,
                                    Tensor& output,
                                    const Tensor& segment_offsets,
                                    Stream& stream) {
    validateSegmentedExclusiveSum(input, output, segment_offsets, plan.num_items, plan.num_segments);
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype ||
        segment_offsets.getDataType() != plan.offset_dtype) {
        throw std::invalid_argument("CUB segmented inclusive-sum plan is not compatible with the provided tensors.");
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
            CUDA_CHECK(cub::DeviceSegmentedScan::InclusiveSegmentedSum(temp_storage_ptr,
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

void cubDeviceSegmentedInclusiveSum(const Tensor& temp_storage,
                                    size_t temp_storage_bytes,
                                    const Tensor& input,
                                    Tensor& output,
                                    const Tensor& segment_offsets,
                                    uint64_t num_items,
                                    uint64_t num_segments,
                                    Stream& stream) {
    CubDeviceSegmentedInclusiveSumPlan plan = prepareCubDeviceSegmentedInclusiveSum(input, output, segment_offsets, num_items, num_segments);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("temp_storage_bytes is smaller than the prepared CUB segmented inclusive-sum requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    cubDeviceSegmentedInclusiveSum(plan, temp_storage, input, output, segment_offsets, stream);
}

CubDeviceSegmentedUniformExclusiveSumPlan prepareCubDeviceSegmentedUniformExclusiveSum(const Tensor& input,
                                                                                       const Tensor& output,
                                                                                       uint64_t num_items,
                                                                                       uint64_t num_segments,
                                                                                       uint64_t segment_size) {
    validateUniformSegmentedSum(input, output, num_items, num_segments, segment_size, "CUB segmented-uniform exclusive-sum");

    size_t bytes = 1;
    if (num_items != 0 && num_segments != 0) {
        auto query_value = [&]<typename T>() -> size_t {
            const int64_t cub_segments = checkedCubInt64Count(num_segments, "num_segments");
            const int64_t cub_segment_size = checkedCubInt64Count(segment_size, "segment_size");
            auto begin_offsets = makeUniformSegmentBeginOffsets<int64_t>(cub_segment_size);
            auto end_offsets = begin_offsets + 1;
            size_t queried_bytes = 0;
            CUDA_CHECK(cub::DeviceSegmentedScan::ExclusiveSegmentedSum(nullptr,
                                                                       queried_bytes,
                                                                       input.getMemPtr<T>(),
                                                                       const_cast<T*>(output.getMemPtr<T>()),
                                                                       begin_offsets,
                                                                       end_offsets,
                                                                       cub_segments));
            return queried_bytes;
        };
        bytes = dispatchScanDType(input.getDataType(), query_value);
    }

    CubDeviceSegmentedUniformExclusiveSumPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.num_items = num_items;
    plan.num_segments = num_segments;
    plan.segment_size = segment_size;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceSegmentedUniformExclusiveSumTempBytes(const Tensor& input,
                                                      const Tensor& output,
                                                      uint64_t num_items,
                                                      uint64_t num_segments,
                                                      uint64_t segment_size) {
    return prepareCubDeviceSegmentedUniformExclusiveSum(input, output, num_items, num_segments, segment_size).temp_storage_bytes;
}

void cubDeviceSegmentedUniformExclusiveSum(const CubDeviceSegmentedUniformExclusiveSumPlan& plan,
                                           const Tensor& temp_storage,
                                           const Tensor& input,
                                           Tensor& output,
                                           Stream& stream) {
    validateUniformSegmentedSum(input, output, plan.num_items, plan.num_segments, plan.segment_size, "CUB segmented-uniform exclusive-sum");
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype) {
        throw std::invalid_argument("CUB segmented-uniform exclusive-sum plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    if (plan.num_items == 0 || plan.num_segments == 0) {
        return;
    }

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch_value = [&]<typename T>() -> void {
        const int64_t cub_segments = checkedCubInt64Count(plan.num_segments, "num_segments");
        const int64_t cub_segment_size = checkedCubInt64Count(plan.segment_size, "segment_size");
        auto begin_offsets = makeUniformSegmentBeginOffsets<int64_t>(cub_segment_size);
        auto end_offsets = begin_offsets + 1;
        CUDA_CHECK(cub::DeviceSegmentedScan::ExclusiveSegmentedSum(temp_storage_ptr,
                                                                   temp_storage_bytes,
                                                                   input.getMemPtr<T>(),
                                                                   output.getMemPtr<T>(),
                                                                   begin_offsets,
                                                                   end_offsets,
                                                                   cub_segments,
                                                                   stream.getStream()));
    };

    dispatchScanDType(plan.dtype, launch_value);
}

CubDeviceSegmentedUniformInclusiveSumPlan prepareCubDeviceSegmentedUniformInclusiveSum(const Tensor& input,
                                                                                       const Tensor& output,
                                                                                       uint64_t num_items,
                                                                                       uint64_t num_segments,
                                                                                       uint64_t segment_size) {
    validateUniformSegmentedSum(input, output, num_items, num_segments, segment_size, "CUB segmented-uniform inclusive-sum");

    size_t bytes = 1;
    if (num_items != 0 && num_segments != 0) {
        auto query_value = [&]<typename T>() -> size_t {
            const int64_t cub_segments = checkedCubInt64Count(num_segments, "num_segments");
            const int64_t cub_segment_size = checkedCubInt64Count(segment_size, "segment_size");
            auto begin_offsets = makeUniformSegmentBeginOffsets<int64_t>(cub_segment_size);
            auto end_offsets = begin_offsets + 1;
            size_t queried_bytes = 0;
            CUDA_CHECK(cub::DeviceSegmentedScan::InclusiveSegmentedSum(nullptr,
                                                                       queried_bytes,
                                                                       input.getMemPtr<T>(),
                                                                       const_cast<T*>(output.getMemPtr<T>()),
                                                                       begin_offsets,
                                                                       end_offsets,
                                                                       cub_segments));
            return queried_bytes;
        };
        bytes = dispatchScanDType(input.getDataType(), query_value);
    }

    CubDeviceSegmentedUniformInclusiveSumPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.num_items = num_items;
    plan.num_segments = num_segments;
    plan.segment_size = segment_size;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceSegmentedUniformInclusiveSumTempBytes(const Tensor& input,
                                                      const Tensor& output,
                                                      uint64_t num_items,
                                                      uint64_t num_segments,
                                                      uint64_t segment_size) {
    return prepareCubDeviceSegmentedUniformInclusiveSum(input, output, num_items, num_segments, segment_size).temp_storage_bytes;
}

void cubDeviceSegmentedUniformInclusiveSum(const CubDeviceSegmentedUniformInclusiveSumPlan& plan,
                                           const Tensor& temp_storage,
                                           const Tensor& input,
                                           Tensor& output,
                                           Stream& stream) {
    validateUniformSegmentedSum(input, output, plan.num_items, plan.num_segments, plan.segment_size, "CUB segmented-uniform inclusive-sum");
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype) {
        throw std::invalid_argument("CUB segmented-uniform inclusive-sum plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    if (plan.num_items == 0 || plan.num_segments == 0) {
        return;
    }

    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;

    auto launch_value = [&]<typename T>() -> void {
        const int64_t cub_segments = checkedCubInt64Count(plan.num_segments, "num_segments");
        const int64_t cub_segment_size = checkedCubInt64Count(plan.segment_size, "segment_size");
        auto begin_offsets = makeUniformSegmentBeginOffsets<int64_t>(cub_segment_size);
        auto end_offsets = begin_offsets + 1;
        CUDA_CHECK(cub::DeviceSegmentedScan::InclusiveSegmentedSum(temp_storage_ptr,
                                                                   temp_storage_bytes,
                                                                   input.getMemPtr<T>(),
                                                                   output.getMemPtr<T>(),
                                                                   begin_offsets,
                                                                   end_offsets,
                                                                   cub_segments,
                                                                   stream.getStream()));
    };

    dispatchScanDType(plan.dtype, launch_value);
}


CubDeviceSegmentedUniformScanPlan prepareCubDeviceSegmentedUniformScan(const Tensor& input,
                                                                       const Tensor& output,
                                                                       uint64_t num_items,
                                                                       uint64_t num_segments,
                                                                       uint64_t segment_size,
                                                                       CubScanOp op,
                                                                       CubScanMode mode) {
    validateUniformSegmentedSum(input, output, num_items, num_segments, segment_size, "CUB segmented-uniform scan");
    if (!isCubScanDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB segmented-uniform scan dtype " + dtypeName(input.getDataType()) + ".");
    }

    size_t bytes = 1;
    if (num_items != 0 && num_segments != 0) {
        auto query_value = [&]<typename T>() -> size_t {
            auto query_op = [&](auto scan_op, T init) -> size_t {
                return queryUniformSegmentedScan<T>(input, output, num_segments, segment_size, mode, scan_op, init);
            };
            return dispatchSegmentedScanOperator<T>(op, query_op);
        };
        bytes = dispatchScanDType(input.getDataType(), query_value);
    }

    CubDeviceSegmentedUniformScanPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.num_items = num_items;
    plan.num_segments = num_segments;
    plan.segment_size = segment_size;
    plan.op = op;
    plan.mode = mode;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceSegmentedUniformScanTempBytes(const Tensor& input,
                                              const Tensor& output,
                                              uint64_t num_items,
                                              uint64_t num_segments,
                                              uint64_t segment_size,
                                              CubScanOp op,
                                              CubScanMode mode) {
    return prepareCubDeviceSegmentedUniformScan(input, output, num_items, num_segments, segment_size, op, mode).temp_storage_bytes;
}

void cubDeviceSegmentedUniformScan(const CubDeviceSegmentedUniformScanPlan& plan,
                                   const Tensor& temp_storage,
                                   const Tensor& input,
                                   Tensor& output,
                                   Stream& stream) {
    validateUniformSegmentedSum(input, output, plan.num_items, plan.num_segments, plan.segment_size, "CUB segmented-uniform scan");
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype) {
        throw std::invalid_argument("CUB segmented-uniform scan plan is not compatible with the provided tensors.");
    }
    if (!isCubScanDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB segmented-uniform scan dtype " + dtypeName(input.getDataType()) + ".");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    if (plan.num_items == 0 || plan.num_segments == 0) {
        return;
    }

    auto launch_value = [&]<typename T>() -> void {
        auto launch_op = [&](auto scan_op, T init) -> void {
            launchUniformSegmentedScan<T>(plan, temp_storage, input, output, stream, scan_op, init);
        };
        dispatchSegmentedScanOperator<T>(plan.op, launch_op);
    };

    dispatchScanDType(plan.dtype, launch_value);
}

}  // namespace ThorImplementation
