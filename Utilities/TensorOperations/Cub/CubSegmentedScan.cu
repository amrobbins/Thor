#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include "Utilities/Expression/CudaHelpers.h"

#include <cub/device/device_segmented_scan.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
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
struct UniformSegmentReverseIndexOp {
    OffsetT segment_size;

    __host__ __device__ OffsetT operator()(OffsetT logical_index) const {
        const OffsetT segment = logical_index / segment_size;
        const OffsetT inner = logical_index - segment * segment_size;
        return segment * segment_size + (segment_size - OffsetT{1} - inner);
    }
};

template <typename OffsetT>
auto makeUniformSegmentBeginOffsets(OffsetT segment_size) {
    return thrust::make_transform_iterator(thrust::counting_iterator<OffsetT>(0), UniformSegmentOffsetOp<OffsetT>{segment_size});
}

template <typename OffsetT>
auto makeUniformSegmentReverseIndices(OffsetT segment_size) {
    return thrust::make_transform_iterator(thrust::counting_iterator<OffsetT>(0), UniformSegmentReverseIndexOp<OffsetT>{segment_size});
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

template <typename T, typename ScanOpT, typename InputIt, typename OutputIt, typename BeginOffsetIt, typename EndOffsetIt>
size_t queryUniformSegmentedScanIterator(InputIt input_begin,
                                         OutputIt output_begin,
                                         BeginOffsetIt begin_offsets,
                                         EndOffsetIt end_offsets,
                                         int64_t cub_segments,
                                         CubScanMode mode,
                                         ScanOpT scan_op,
                                         T init) {
    size_t queried_bytes = 0;
    if (mode == CubScanMode::Exclusive) {
        CUDA_CHECK(cub::DeviceSegmentedScan::ExclusiveSegmentedScan(nullptr,
                                                                    queried_bytes,
                                                                    input_begin,
                                                                    output_begin,
                                                                    begin_offsets,
                                                                    end_offsets,
                                                                    cub_segments,
                                                                    scan_op,
                                                                    init));
    } else if (mode == CubScanMode::Inclusive) {
        CUDA_CHECK(cub::DeviceSegmentedScan::InclusiveSegmentedScan(
            nullptr, queried_bytes, input_begin, output_begin, begin_offsets, end_offsets, cub_segments, scan_op));
    } else {
        throw std::invalid_argument("Unsupported CUB segmented scan mode.");
    }
    return queried_bytes;
}

template <typename T, typename ScanOpT>
size_t queryUniformSegmentedScan(const Tensor& input,
                                 const Tensor& output,
                                 uint64_t num_segments,
                                 uint64_t segment_size,
                                 CubScanMode mode,
                                 CubScanDirection direction,
                                 ScanOpT scan_op,
                                 T init) {
    const int64_t cub_segments = checkedCubInt64Count(num_segments, "num_segments");
    const int64_t cub_segment_size = checkedCubInt64Count(segment_size, "segment_size");
    auto begin_offsets = makeUniformSegmentBeginOffsets<int64_t>(cub_segment_size);
    auto end_offsets = begin_offsets + 1;
    if (direction == CubScanDirection::Forward) {
        return queryUniformSegmentedScanIterator<T>(input.getMemPtr<T>(),
                                                   const_cast<T*>(output.getMemPtr<T>()),
                                                   begin_offsets,
                                                   end_offsets,
                                                   cub_segments,
                                                   mode,
                                                   scan_op,
                                                   init);
    }
    if (direction == CubScanDirection::Reverse) {
        auto input_indices = makeUniformSegmentReverseIndices<int64_t>(cub_segment_size);
        auto output_indices = makeUniformSegmentReverseIndices<int64_t>(cub_segment_size);
        auto input_begin = thrust::make_permutation_iterator(input.getMemPtr<T>(), input_indices);
        auto output_begin = thrust::make_permutation_iterator(const_cast<T*>(output.getMemPtr<T>()), output_indices);
        return queryUniformSegmentedScanIterator<T>(
            input_begin, output_begin, begin_offsets, end_offsets, cub_segments, mode, scan_op, init);
    }
    throw std::invalid_argument("Unsupported CUB segmented scan direction.");
}

template <typename T, typename ScanOpT, typename InputIt, typename OutputIt, typename BeginOffsetIt, typename EndOffsetIt>
void launchUniformSegmentedScanIterator(void* temp_storage_ptr,
                                        size_t temp_storage_bytes,
                                        InputIt input_begin,
                                        OutputIt output_begin,
                                        BeginOffsetIt begin_offsets,
                                        EndOffsetIt end_offsets,
                                        int64_t cub_segments,
                                        cudaStream_t stream,
                                        CubScanMode mode,
                                        ScanOpT scan_op,
                                        T init) {
    if (mode == CubScanMode::Exclusive) {
        CUDA_CHECK(cub::DeviceSegmentedScan::ExclusiveSegmentedScan(temp_storage_ptr,
                                                                    temp_storage_bytes,
                                                                    input_begin,
                                                                    output_begin,
                                                                    begin_offsets,
                                                                    end_offsets,
                                                                    cub_segments,
                                                                    scan_op,
                                                                    init,
                                                                    stream));
    } else if (mode == CubScanMode::Inclusive) {
        CUDA_CHECK(cub::DeviceSegmentedScan::InclusiveSegmentedScan(temp_storage_ptr,
                                                                    temp_storage_bytes,
                                                                    input_begin,
                                                                    output_begin,
                                                                    begin_offsets,
                                                                    end_offsets,
                                                                    cub_segments,
                                                                    scan_op,
                                                                    stream));
    } else {
        throw std::invalid_argument("Unsupported CUB segmented scan mode.");
    }
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
    if (plan.direction == CubScanDirection::Forward) {
        launchUniformSegmentedScanIterator<T>(temp_storage_ptr,
                                             temp_storage_bytes,
                                             input.getMemPtr<T>(),
                                             output.getMemPtr<T>(),
                                             begin_offsets,
                                             end_offsets,
                                             cub_segments,
                                             stream.getStream(),
                                             plan.mode,
                                             scan_op,
                                             init);
        return;
    }
    if (plan.direction == CubScanDirection::Reverse) {
        auto input_indices = makeUniformSegmentReverseIndices<int64_t>(cub_segment_size);
        auto output_indices = makeUniformSegmentReverseIndices<int64_t>(cub_segment_size);
        auto input_begin = thrust::make_permutation_iterator(input.getMemPtr<T>(), input_indices);
        auto output_begin = thrust::make_permutation_iterator(output.getMemPtr<T>(), output_indices);
        launchUniformSegmentedScanIterator<T>(temp_storage_ptr,
                                             temp_storage_bytes,
                                             input_begin,
                                             output_begin,
                                             begin_offsets,
                                             end_offsets,
                                             cub_segments,
                                             stream.getStream(),
                                             plan.mode,
                                             scan_op,
                                             init);
        return;
    }
    throw std::invalid_argument("Unsupported CUB segmented scan direction.");
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
                                                                       CubScanMode mode,
                                                                       CubScanDirection direction) {
    validateUniformSegmentedSum(input, output, num_items, num_segments, segment_size, "CUB segmented-uniform scan");
    if (!isCubScanDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB segmented-uniform scan dtype " + dtypeName(input.getDataType()) + ".");
    }

    size_t bytes = 1;
    if (num_items != 0 && num_segments != 0) {
        auto query_value = [&]<typename T>() -> size_t {
            auto query_op = [&](auto scan_op, T init) -> size_t {
                return queryUniformSegmentedScan<T>(input, output, num_segments, segment_size, mode, direction, scan_op, init);
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
    plan.direction = direction;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceSegmentedUniformScanTempBytes(const Tensor& input,
                                              const Tensor& output,
                                              uint64_t num_items,
                                              uint64_t num_segments,
                                              uint64_t segment_size,
                                              CubScanOp op,
                                              CubScanMode mode,
                                              CubScanDirection direction) {
    return prepareCubDeviceSegmentedUniformScan(input, output, num_items, num_segments, segment_size, op, mode, direction).temp_storage_bytes;
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


namespace {

template <typename OffsetT>
void validateRaggedReverseRange(uint64_t num_items, uint64_t num_segments) {
    if (num_items > static_cast<uint64_t>(std::numeric_limits<OffsetT>::max()) ||
        num_segments > static_cast<uint64_t>(std::numeric_limits<OffsetT>::max())) {
        throw std::invalid_argument("CUB segmented reverse scan range exceeds the segment-offset dtype range.");
    }
}

template <typename OffsetT>
struct RaggedReverseBeginOffsetOp {
    const OffsetT* offsets;
    OffsetT num_items;
    OffsetT num_segments;

    __host__ __device__ OffsetT operator()(OffsetT segment) const {
        const OffsetT original_segment = num_segments - OffsetT{1} - segment;
        return num_items - offsets[original_segment + OffsetT{1}];
    }
};

template <typename OffsetT>
struct RaggedReverseEndOffsetOp {
    const OffsetT* offsets;
    OffsetT num_items;
    OffsetT num_segments;

    __host__ __device__ OffsetT operator()(OffsetT segment) const {
        const OffsetT original_segment = num_segments - OffsetT{1} - segment;
        return num_items - offsets[original_segment];
    }
};

template <typename OffsetT>
auto makeRaggedReverseBeginOffsets(const OffsetT* offsets, uint64_t num_items, uint64_t num_segments) {
    validateRaggedReverseRange<OffsetT>(num_items, num_segments);
    return thrust::make_transform_iterator(
        thrust::counting_iterator<OffsetT>(0),
        RaggedReverseBeginOffsetOp<OffsetT>{offsets, static_cast<OffsetT>(num_items), static_cast<OffsetT>(num_segments)});
}

template <typename OffsetT>
auto makeRaggedReverseEndOffsets(const OffsetT* offsets, uint64_t num_items, uint64_t num_segments) {
    validateRaggedReverseRange<OffsetT>(num_items, num_segments);
    return thrust::make_transform_iterator(
        thrust::counting_iterator<OffsetT>(0),
        RaggedReverseEndOffsetOp<OffsetT>{offsets, static_cast<OffsetT>(num_items), static_cast<OffsetT>(num_segments)});
}

template <typename T, typename ScanOpT, typename InputIt, typename OutputIt, typename BeginOffsetIt, typename EndOffsetIt>
size_t querySegmentedScanIterator(InputIt input_begin,
                                  OutputIt output_begin,
                                  BeginOffsetIt begin_offsets,
                                  EndOffsetIt end_offsets,
                                  int64_t cub_segments,
                                  CubScanMode mode,
                                  ScanOpT scan_op,
                                  T init) {
    size_t queried_bytes = 0;
    if (mode == CubScanMode::Exclusive) {
        CUDA_CHECK(cub::DeviceSegmentedScan::ExclusiveSegmentedScan(nullptr,
                                                                    queried_bytes,
                                                                    input_begin,
                                                                    output_begin,
                                                                    begin_offsets,
                                                                    end_offsets,
                                                                    cub_segments,
                                                                    scan_op,
                                                                    init));
    } else if (mode == CubScanMode::Inclusive) {
        CUDA_CHECK(cub::DeviceSegmentedScan::InclusiveSegmentedScan(
            nullptr, queried_bytes, input_begin, output_begin, begin_offsets, end_offsets, cub_segments, scan_op));
    } else {
        throw std::invalid_argument("Unsupported CUB segmented scan mode.");
    }
    return queried_bytes;
}

template <typename T, typename OffsetT, typename ScanOpT>
size_t querySegmentedScan(const Tensor& input,
                          const Tensor& output,
                          const Tensor& segment_offsets,
                          uint64_t num_items,
                          uint64_t num_segments,
                          CubScanMode mode,
                          CubScanDirection direction,
                          ScanOpT scan_op,
                          T init) {
    const int64_t cub_segments = checkedCubInt64Count(num_segments, "num_segments");
    const OffsetT* offsets = segment_offsets.getMemPtr<OffsetT>();
    if (direction == CubScanDirection::Forward) {
        return querySegmentedScanIterator<T>(input.getMemPtr<T>(),
                                             const_cast<T*>(output.getMemPtr<T>()),
                                             offsets,
                                             offsets + 1,
                                             cub_segments,
                                             mode,
                                             scan_op,
                                             init);
    }
    if (direction == CubScanDirection::Reverse) {
        auto input_begin = thrust::make_reverse_iterator(input.getMemPtr<T>() + num_items);
        auto output_begin = thrust::make_reverse_iterator(const_cast<T*>(output.getMemPtr<T>()) + num_items);
        auto begin_offsets = makeRaggedReverseBeginOffsets<OffsetT>(offsets, num_items, num_segments);
        auto end_offsets = makeRaggedReverseEndOffsets<OffsetT>(offsets, num_items, num_segments);
        return querySegmentedScanIterator<T>(input_begin, output_begin, begin_offsets, end_offsets, cub_segments, mode, scan_op, init);
    }
    throw std::invalid_argument("Unsupported CUB segmented scan direction.");
}

template <typename T, typename ScanOpT, typename InputIt, typename OutputIt, typename BeginOffsetIt, typename EndOffsetIt>
void launchSegmentedScanIterator(void* temp_storage_ptr,
                                 size_t temp_storage_bytes,
                                 InputIt input_begin,
                                 OutputIt output_begin,
                                 BeginOffsetIt begin_offsets,
                                 EndOffsetIt end_offsets,
                                 int64_t cub_segments,
                                 cudaStream_t stream,
                                 CubScanMode mode,
                                 ScanOpT scan_op,
                                 T init) {
    if (mode == CubScanMode::Exclusive) {
        CUDA_CHECK(cub::DeviceSegmentedScan::ExclusiveSegmentedScan(temp_storage_ptr,
                                                                    temp_storage_bytes,
                                                                    input_begin,
                                                                    output_begin,
                                                                    begin_offsets,
                                                                    end_offsets,
                                                                    cub_segments,
                                                                    scan_op,
                                                                    init,
                                                                    stream));
    } else if (mode == CubScanMode::Inclusive) {
        CUDA_CHECK(cub::DeviceSegmentedScan::InclusiveSegmentedScan(temp_storage_ptr,
                                                                    temp_storage_bytes,
                                                                    input_begin,
                                                                    output_begin,
                                                                    begin_offsets,
                                                                    end_offsets,
                                                                    cub_segments,
                                                                    scan_op,
                                                                    stream));
    } else {
        throw std::invalid_argument("Unsupported CUB segmented scan mode.");
    }
}

template <typename T, typename OffsetT, typename ScanOpT>
void launchSegmentedScan(const CubDeviceSegmentedScanPlan& plan,
                         const Tensor& temp_storage,
                         const Tensor& input,
                         Tensor& output,
                         const Tensor& segment_offsets,
                         Stream& stream,
                         ScanOpT scan_op,
                         T init) {
    const int64_t cub_segments = checkedCubInt64Count(plan.num_segments, "num_segments");
    const OffsetT* offsets = segment_offsets.getMemPtr<OffsetT>();
    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    size_t temp_storage_bytes = plan.temp_storage_bytes;
    if (plan.direction == CubScanDirection::Forward) {
        launchSegmentedScanIterator<T>(temp_storage_ptr,
                                       temp_storage_bytes,
                                       input.getMemPtr<T>(),
                                       output.getMemPtr<T>(),
                                       offsets,
                                       offsets + 1,
                                       cub_segments,
                                       stream.getStream(),
                                       plan.mode,
                                       scan_op,
                                       init);
        return;
    }
    if (plan.direction == CubScanDirection::Reverse) {
        auto input_begin = thrust::make_reverse_iterator(input.getMemPtr<T>() + plan.num_items);
        auto output_begin = thrust::make_reverse_iterator(output.getMemPtr<T>() + plan.num_items);
        auto begin_offsets = makeRaggedReverseBeginOffsets<OffsetT>(offsets, plan.num_items, plan.num_segments);
        auto end_offsets = makeRaggedReverseEndOffsets<OffsetT>(offsets, plan.num_items, plan.num_segments);
        launchSegmentedScanIterator<T>(temp_storage_ptr,
                                       temp_storage_bytes,
                                       input_begin,
                                       output_begin,
                                       begin_offsets,
                                       end_offsets,
                                       cub_segments,
                                       stream.getStream(),
                                       plan.mode,
                                       scan_op,
                                       init);
        return;
    }
    throw std::invalid_argument("Unsupported CUB segmented scan direction.");
}

void validateSegmentedScan(const Tensor& input,
                           const Tensor& output,
                           const Tensor& segment_offsets,
                           uint64_t num_items,
                           uint64_t num_segments) {
    validateSegmentedExclusiveSum(input, output, segment_offsets, num_items, num_segments);
    if (!isCubScanDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB segmented scan dtype " + dtypeName(input.getDataType()) + ".");
    }
}

}  // namespace

CubDeviceSegmentedScanPlan prepareCubDeviceSegmentedScan(const Tensor& input,
                                                         const Tensor& output,
                                                         const Tensor& segment_offsets,
                                                         uint64_t num_items,
                                                         uint64_t num_segments,
                                                         CubScanOp op,
                                                         CubScanMode mode,
                                                         CubScanDirection direction) {
    validateSegmentedScan(input, output, segment_offsets, num_items, num_segments);

    size_t bytes = 1;
    if (num_items != 0 && num_segments != 0) {
        auto query_value = [&]<typename T>() -> size_t {
            auto query_offset = [&]<typename OffsetT>() -> size_t {
                auto query_op = [&](auto scan_op, T init) -> size_t {
                    return querySegmentedScan<T, OffsetT>(
                        input, output, segment_offsets, num_items, num_segments, mode, direction, scan_op, init);
                };
                return dispatchSegmentedScanOperator<T>(op, query_op);
            };
            return dispatchSegmentOffsetDType(segment_offsets.getDataType(), query_offset);
        };
        bytes = dispatchScanDType(input.getDataType(), query_value);
    }

    CubDeviceSegmentedScanPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.offset_dtype = segment_offsets.getDataType();
    plan.num_items = num_items;
    plan.num_segments = num_segments;
    plan.op = op;
    plan.mode = mode;
    plan.direction = direction;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceSegmentedScanTempBytes(const Tensor& input,
                                       const Tensor& output,
                                       const Tensor& segment_offsets,
                                       uint64_t num_items,
                                       uint64_t num_segments,
                                       CubScanOp op,
                                       CubScanMode mode,
                                       CubScanDirection direction) {
    return prepareCubDeviceSegmentedScan(input, output, segment_offsets, num_items, num_segments, op, mode, direction).temp_storage_bytes;
}

void cubDeviceSegmentedScan(const CubDeviceSegmentedScanPlan& plan,
                            const Tensor& temp_storage,
                            const Tensor& input,
                            Tensor& output,
                            const Tensor& segment_offsets,
                            Stream& stream) {
    validateSegmentedScan(input, output, segment_offsets, plan.num_items, plan.num_segments);
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype ||
        segment_offsets.getDataType() != plan.offset_dtype) {
        throw std::invalid_argument("CUB segmented scan plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    if (plan.num_items == 0 || plan.num_segments == 0) {
        return;
    }

    auto launch_value = [&]<typename T>() -> void {
        auto launch_offset = [&]<typename OffsetT>() -> void {
            auto launch_op = [&](auto scan_op, T init) -> void {
                launchSegmentedScan<T, OffsetT>(plan, temp_storage, input, output, segment_offsets, stream, scan_op, init);
            };
            dispatchSegmentedScanOperator<T>(plan.op, launch_op);
        };
        dispatchSegmentOffsetDType(plan.offset_dtype, launch_offset);
    };

    dispatchScanDType(plan.dtype, launch_value);
}

namespace {

template <typename T>
struct CubSegmentedArgScanPair {
    T value;
    uint32_t index;
};

template <typename T>
struct CubSegmentedArgScanInputOp {
    const T* input;
    uint32_t num_items;
    uint32_t segment_size;
    bool uniform_segmented;
    CubScanDirection direction;

    __host__ __device__ uint32_t physicalIndex(uint32_t logical_index) const {
        if (direction == CubScanDirection::Forward) {
            return logical_index;
        }
        if (uniform_segmented) {
            const uint32_t segment = logical_index / segment_size;
            const uint32_t inner = logical_index - segment * segment_size;
            return segment * segment_size + (segment_size - 1U - inner);
        }
        return num_items - 1U - logical_index;
    }

    __host__ __device__ CubSegmentedArgScanPair<T> operator()(uint32_t logical_index) const {
        const uint32_t physical = physicalIndex(logical_index);
        return CubSegmentedArgScanPair<T>{input[physical], physical};
    }
};

template <typename T>
struct CubSegmentedArgScanMinOp {
    __host__ __device__ CubSegmentedArgScanPair<T> operator()(const CubSegmentedArgScanPair<T>& a, const CubSegmentedArgScanPair<T>& b) const {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            return segmentedScanToFloat(b.value) < segmentedScanToFloat(a.value) ? b : a;
        } else {
            return b.value < a.value ? b : a;
        }
    }
};

template <typename T>
struct CubSegmentedArgScanMaxOp {
    __host__ __device__ CubSegmentedArgScanPair<T> operator()(const CubSegmentedArgScanPair<T>& a, const CubSegmentedArgScanPair<T>& b) const {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            return segmentedScanToFloat(a.value) < segmentedScanToFloat(b.value) ? b : a;
        } else {
            return a.value < b.value ? b : a;
        }
    }
};

template <typename T>
CubSegmentedArgScanPair<T> segmentedArgScanIdentity(CubArgScanOp op) {
    switch (op) {
        case CubArgScanOp::ArgMin:
            return CubSegmentedArgScanPair<T>{segmentedScanPositiveInfinityOrMax<T>(), UINT32_MAX};
        case CubArgScanOp::ArgMax:
            return CubSegmentedArgScanPair<T>{segmentedScanNegativeInfinityOrLowest<T>(), UINT32_MAX};
    }
    throw std::invalid_argument("Unsupported CUB segmented arg scan op.");
}

template <typename T, typename Fn>
decltype(auto) dispatchSegmentedArgScanOperator(CubArgScanOp op, Fn&& fn) {
    switch (op) {
        case CubArgScanOp::ArgMin:
            return fn(CubSegmentedArgScanMinOp<T>{}, segmentedArgScanIdentity<T>(op));
        case CubArgScanOp::ArgMax:
            return fn(CubSegmentedArgScanMaxOp<T>{}, segmentedArgScanIdentity<T>(op));
    }
    throw std::invalid_argument("Unsupported CUB segmented arg scan op.");
}

inline size_t alignCubSegmentedArgScanBytes(size_t bytes) { return (bytes + size_t{255}) & ~size_t{255}; }

template <typename T>
size_t segmentedArgScanPairStorageBytes(uint64_t num_items) {
    if (num_items > std::numeric_limits<size_t>::max() / sizeof(CubSegmentedArgScanPair<T>)) {
        throw std::invalid_argument("CUB segmented arg scan pair workspace size overflow.");
    }
    return alignCubSegmentedArgScanBytes(static_cast<size_t>(num_items) * sizeof(CubSegmentedArgScanPair<T>));
}

template <typename T>
__global__ void extractSegmentedArgScanIndicesKernel(const CubSegmentedArgScanPair<T>* pairs,
                                                     uint32_t* output,
                                                     uint32_t num_items,
                                                     uint32_t segment_size,
                                                     bool uniform_segmented,
                                                     CubScanDirection direction) {
    const uint32_t logical = blockIdx.x * blockDim.x + threadIdx.x;
    if (logical >= num_items) {
        return;
    }
    uint32_t physical = logical;
    if (direction == CubScanDirection::Reverse) {
        if (uniform_segmented) {
            const uint32_t segment = logical / segment_size;
            const uint32_t inner = logical - segment * segment_size;
            physical = segment * segment_size + (segment_size - 1U - inner);
        } else {
            physical = num_items - 1U - logical;
        }
    }
    output[physical] = pairs[logical].index;
}

template <typename T, typename ScanOpT, typename InputIt, typename BeginOffsetIt, typename EndOffsetIt>
size_t querySegmentedArgScanIterator(InputIt input_begin,
                                     CubSegmentedArgScanPair<T>* output_begin,
                                     BeginOffsetIt begin_offsets,
                                     EndOffsetIt end_offsets,
                                     int64_t cub_segments,
                                     CubScanMode mode,
                                     ScanOpT scan_op,
                                     CubSegmentedArgScanPair<T> init) {
    size_t queried_bytes = 0;
    if (mode == CubScanMode::Exclusive) {
        CUDA_CHECK(cub::DeviceSegmentedScan::ExclusiveSegmentedScan(nullptr,
                                                                    queried_bytes,
                                                                    input_begin,
                                                                    output_begin,
                                                                    begin_offsets,
                                                                    end_offsets,
                                                                    cub_segments,
                                                                    scan_op,
                                                                    init));
    } else if (mode == CubScanMode::Inclusive) {
        CUDA_CHECK(cub::DeviceSegmentedScan::InclusiveSegmentedScan(
            nullptr, queried_bytes, input_begin, output_begin, begin_offsets, end_offsets, cub_segments, scan_op));
    } else {
        throw std::invalid_argument("Unsupported CUB segmented arg scan mode.");
    }
    return queried_bytes;
}

template <typename T, typename ScanOpT>
size_t queryUniformSegmentedArgScan(const Tensor& input,
                                    uint64_t num_items,
                                    uint64_t num_segments,
                                    uint64_t segment_size,
                                    CubScanMode mode,
                                    CubScanDirection direction,
                                    ScanOpT scan_op,
                                    CubSegmentedArgScanPair<T> init) {
    const int64_t cub_segments = checkedCubInt64Count(num_segments, "num_segments");
    const int64_t cub_segment_size = checkedCubInt64Count(segment_size, "segment_size");
    auto begin_offsets = makeUniformSegmentBeginOffsets<int64_t>(cub_segment_size);
    auto end_offsets = begin_offsets + 1;
    auto input_begin = thrust::make_transform_iterator(
        thrust::counting_iterator<uint32_t>(0),
        CubSegmentedArgScanInputOp<T>{input.getMemPtr<T>(), static_cast<uint32_t>(checkedCubNumItems(num_items)), static_cast<uint32_t>(cub_segment_size), true, direction});
    CubSegmentedArgScanPair<T>* output_begin = nullptr;
    return querySegmentedArgScanIterator<T>(input_begin, output_begin, begin_offsets, end_offsets, cub_segments, mode, scan_op, init);
}

template <typename T, typename OffsetT, typename ScanOpT>
size_t querySegmentedArgScan(const Tensor& input,
                             const Tensor& segment_offsets,
                             uint64_t num_items,
                             uint64_t num_segments,
                             CubScanMode mode,
                             CubScanDirection direction,
                             ScanOpT scan_op,
                             CubSegmentedArgScanPair<T> init) {
    const int64_t cub_segments = checkedCubInt64Count(num_segments, "num_segments");
    const OffsetT* offsets = segment_offsets.getMemPtr<OffsetT>();
    auto input_begin = thrust::make_transform_iterator(
        thrust::counting_iterator<uint32_t>(0),
        CubSegmentedArgScanInputOp<T>{input.getMemPtr<T>(), static_cast<uint32_t>(checkedCubNumItems(num_items)), 0U, false, direction});
    CubSegmentedArgScanPair<T>* output_begin = nullptr;
    if (direction == CubScanDirection::Forward) {
        return querySegmentedArgScanIterator<T>(input_begin, output_begin, offsets, offsets + 1, cub_segments, mode, scan_op, init);
    }
    if (direction == CubScanDirection::Reverse) {
        auto begin_offsets = makeRaggedReverseBeginOffsets<OffsetT>(offsets, num_items, num_segments);
        auto end_offsets = makeRaggedReverseEndOffsets<OffsetT>(offsets, num_items, num_segments);
        return querySegmentedArgScanIterator<T>(input_begin, output_begin, begin_offsets, end_offsets, cub_segments, mode, scan_op, init);
    }
    throw std::invalid_argument("Unsupported CUB segmented arg scan direction.");
}

template <typename T, typename ScanOpT, typename InputIt, typename BeginOffsetIt, typename EndOffsetIt>
void launchSegmentedArgScanIterator(void* temp_storage_ptr,
                                    size_t temp_storage_bytes,
                                    InputIt input_begin,
                                    CubSegmentedArgScanPair<T>* output_begin,
                                    BeginOffsetIt begin_offsets,
                                    EndOffsetIt end_offsets,
                                    int64_t cub_segments,
                                    cudaStream_t stream,
                                    CubScanMode mode,
                                    ScanOpT scan_op,
                                    CubSegmentedArgScanPair<T> init) {
    if (mode == CubScanMode::Exclusive) {
        CUDA_CHECK(cub::DeviceSegmentedScan::ExclusiveSegmentedScan(temp_storage_ptr,
                                                                    temp_storage_bytes,
                                                                    input_begin,
                                                                    output_begin,
                                                                    begin_offsets,
                                                                    end_offsets,
                                                                    cub_segments,
                                                                    scan_op,
                                                                    init,
                                                                    stream));
    } else if (mode == CubScanMode::Inclusive) {
        CUDA_CHECK(cub::DeviceSegmentedScan::InclusiveSegmentedScan(temp_storage_ptr,
                                                                    temp_storage_bytes,
                                                                    input_begin,
                                                                    output_begin,
                                                                    begin_offsets,
                                                                    end_offsets,
                                                                    cub_segments,
                                                                    scan_op,
                                                                    stream));
    } else {
        throw std::invalid_argument("Unsupported CUB segmented arg scan mode.");
    }
}

template <typename T, typename ScanOpT>
void launchUniformSegmentedArgScan(const CubDeviceSegmentedUniformArgScanPlan& plan,
                                   const Tensor& temp_storage,
                                   const Tensor& input,
                                   Tensor& output,
                                   Stream& stream,
                                   ScanOpT scan_op,
                                   CubSegmentedArgScanPair<T> init) {
    const int64_t cub_segments = checkedCubInt64Count(plan.num_segments, "num_segments");
    const int64_t cub_segment_size = checkedCubInt64Count(plan.segment_size, "segment_size");
    auto begin_offsets = makeUniformSegmentBeginOffsets<int64_t>(cub_segment_size);
    auto end_offsets = begin_offsets + 1;
    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    const size_t pair_bytes = segmentedArgScanPairStorageBytes<T>(plan.num_items);
    auto* pair_output = reinterpret_cast<CubSegmentedArgScanPair<T>*>(temp_storage_ptr);
    void* cub_temp = static_cast<void*>(static_cast<unsigned char*>(temp_storage_ptr) + pair_bytes);
    const size_t cub_temp_bytes = plan.temp_storage_bytes - pair_bytes;
    auto input_begin = thrust::make_transform_iterator(
        thrust::counting_iterator<uint32_t>(0),
        CubSegmentedArgScanInputOp<T>{input.getMemPtr<T>(), static_cast<uint32_t>(checkedCubNumItems(plan.num_items)), static_cast<uint32_t>(cub_segment_size), true, plan.direction});
    launchSegmentedArgScanIterator<T>(cub_temp,
                                      cub_temp_bytes,
                                      input_begin,
                                      pair_output,
                                      begin_offsets,
                                      end_offsets,
                                      cub_segments,
                                      stream.getStream(),
                                      plan.mode,
                                      scan_op,
                                      init);
    const uint32_t threads = 256;
    const uint32_t items = static_cast<uint32_t>(checkedCubNumItems(plan.num_items));
    const uint32_t blocks = (items + threads - 1U) / threads;
    extractSegmentedArgScanIndicesKernel<T><<<blocks, threads, 0, stream.getStream()>>>(
        pair_output, output.getMemPtr<uint32_t>(), items, static_cast<uint32_t>(cub_segment_size), true, plan.direction);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T, typename OffsetT, typename ScanOpT>
void launchSegmentedArgScan(const CubDeviceSegmentedArgScanPlan& plan,
                            const Tensor& temp_storage,
                            const Tensor& input,
                            Tensor& output,
                            const Tensor& segment_offsets,
                            Stream& stream,
                            ScanOpT scan_op,
                            CubSegmentedArgScanPair<T> init) {
    const int64_t cub_segments = checkedCubInt64Count(plan.num_segments, "num_segments");
    const OffsetT* offsets = segment_offsets.getMemPtr<OffsetT>();
    void* temp_storage_ptr = mutableCubTempStoragePtr(temp_storage);
    const size_t pair_bytes = segmentedArgScanPairStorageBytes<T>(plan.num_items);
    auto* pair_output = reinterpret_cast<CubSegmentedArgScanPair<T>*>(temp_storage_ptr);
    void* cub_temp = static_cast<void*>(static_cast<unsigned char*>(temp_storage_ptr) + pair_bytes);
    const size_t cub_temp_bytes = plan.temp_storage_bytes - pair_bytes;
    auto input_begin = thrust::make_transform_iterator(
        thrust::counting_iterator<uint32_t>(0),
        CubSegmentedArgScanInputOp<T>{input.getMemPtr<T>(), static_cast<uint32_t>(checkedCubNumItems(plan.num_items)), 0U, false, plan.direction});
    if (plan.direction == CubScanDirection::Forward) {
        launchSegmentedArgScanIterator<T>(cub_temp,
                                          cub_temp_bytes,
                                          input_begin,
                                          pair_output,
                                          offsets,
                                          offsets + 1,
                                          cub_segments,
                                          stream.getStream(),
                                          plan.mode,
                                          scan_op,
                                          init);
    } else if (plan.direction == CubScanDirection::Reverse) {
        auto begin_offsets = makeRaggedReverseBeginOffsets<OffsetT>(offsets, plan.num_items, plan.num_segments);
        auto end_offsets = makeRaggedReverseEndOffsets<OffsetT>(offsets, plan.num_items, plan.num_segments);
        launchSegmentedArgScanIterator<T>(cub_temp,
                                          cub_temp_bytes,
                                          input_begin,
                                          pair_output,
                                          begin_offsets,
                                          end_offsets,
                                          cub_segments,
                                          stream.getStream(),
                                          plan.mode,
                                          scan_op,
                                          init);
    } else {
        throw std::invalid_argument("Unsupported CUB segmented arg scan direction.");
    }
    const uint32_t threads = 256;
    const uint32_t items = static_cast<uint32_t>(checkedCubNumItems(plan.num_items));
    const uint32_t blocks = (items + threads - 1U) / threads;
    extractSegmentedArgScanIndicesKernel<T><<<blocks, threads, 0, stream.getStream()>>>(
        pair_output, output.getMemPtr<uint32_t>(), items, 0U, false, plan.direction);
    CUDA_CHECK(cudaGetLastError());
}

void validateUniformSegmentedArgScan(const Tensor& input,
                                     const Tensor& output,
                                     uint64_t num_items,
                                     uint64_t num_segments,
                                     uint64_t segment_size,
                                     const char* op_name) {
    requireDenseContiguousGpuTensor(input, "segmented arg scan input");
    requireDenseContiguousGpuTensor(output, "segmented arg scan output");
    requireSameGpuPlacement(input, output, "segmented arg scan input", "segmented arg scan output");
    requireStorageForNumItems(input, "segmented arg scan input", num_items);
    requireStorageForNumItems(output, "segmented arg scan output", num_items);
    if (output.getDataType() != DataType::UINT32) {
        throw std::invalid_argument("CUB segmented arg scan output dtype must be UINT32.");
    }
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

void validateSegmentedArgScan(const Tensor& input,
                              const Tensor& output,
                              const Tensor& segment_offsets,
                              uint64_t num_items,
                              uint64_t num_segments) {
    requireDenseContiguousGpuTensor(input, "segmented arg scan input");
    requireDenseContiguousGpuTensor(output, "segmented arg scan output");
    requireStorageForNumItems(input, "segmented arg scan input", num_items);
    requireStorageForNumItems(output, "segmented arg scan output", num_items);
    validateSegmentOffsets(input, segment_offsets, num_items, num_segments, "segmented arg scan input");
    requireSameGpuPlacement(input, output, "segmented arg scan input", "segmented arg scan output");
    if (output.getDataType() != DataType::UINT32) {
        throw std::invalid_argument("CUB segmented arg scan output dtype must be UINT32.");
    }
    if (!isCubScanDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB segmented arg scan dtype " + dtypeName(input.getDataType()) + ".");
    }
}

}  // namespace

CubDeviceSegmentedUniformArgScanPlan prepareCubDeviceSegmentedUniformArgScan(const Tensor& input,
                                                                             const Tensor& output,
                                                                             uint64_t num_items,
                                                                             uint64_t num_segments,
                                                                             uint64_t segment_size,
                                                                             CubArgScanOp op,
                                                                             CubScanMode mode,
                                                                             CubScanDirection direction) {
    validateUniformSegmentedArgScan(input, output, num_items, num_segments, segment_size, "CUB segmented-uniform arg scan");
    if (!isCubScanDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB segmented-uniform arg scan dtype " + dtypeName(input.getDataType()) + ".");
    }
    size_t bytes = 1;
    if (num_items != 0 && num_segments != 0) {
        auto query_value = [&]<typename T>() -> size_t {
            auto query_op = [&](auto scan_op, CubSegmentedArgScanPair<T> init) -> size_t {
                return segmentedArgScanPairStorageBytes<T>(num_items) +
                       queryUniformSegmentedArgScan<T>(input, num_items, num_segments, segment_size, mode, direction, scan_op, init);
            };
            return dispatchSegmentedArgScanOperator<T>(op, query_op);
        };
        bytes = dispatchScanDType(input.getDataType(), query_value);
    }
    CubDeviceSegmentedUniformArgScanPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.num_items = num_items;
    plan.num_segments = num_segments;
    plan.segment_size = segment_size;
    plan.op = op;
    plan.mode = mode;
    plan.direction = direction;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceSegmentedUniformArgScanTempBytes(const Tensor& input,
                                                 const Tensor& output,
                                                 uint64_t num_items,
                                                 uint64_t num_segments,
                                                 uint64_t segment_size,
                                                 CubArgScanOp op,
                                                 CubScanMode mode,
                                                 CubScanDirection direction) {
    return prepareCubDeviceSegmentedUniformArgScan(input, output, num_items, num_segments, segment_size, op, mode, direction).temp_storage_bytes;
}

void cubDeviceSegmentedUniformArgScan(const CubDeviceSegmentedUniformArgScanPlan& plan,
                                      const Tensor& temp_storage,
                                      const Tensor& input,
                                      Tensor& output,
                                      Stream& stream) {
    validateUniformSegmentedArgScan(input, output, plan.num_items, plan.num_segments, plan.segment_size, "CUB segmented-uniform arg scan");
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype) {
        throw std::invalid_argument("CUB segmented-uniform arg scan plan is not compatible with the provided tensors.");
    }
    if (!isCubScanDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB segmented-uniform arg scan dtype " + dtypeName(input.getDataType()) + ".");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    if (plan.num_items == 0 || plan.num_segments == 0) {
        return;
    }
    auto launch_value = [&]<typename T>() -> void {
        auto launch_op = [&](auto scan_op, CubSegmentedArgScanPair<T> init) -> void {
            launchUniformSegmentedArgScan<T>(plan, temp_storage, input, output, stream, scan_op, init);
        };
        dispatchSegmentedArgScanOperator<T>(plan.op, launch_op);
    };
    dispatchScanDType(plan.dtype, launch_value);
}

CubDeviceSegmentedArgScanPlan prepareCubDeviceSegmentedArgScan(const Tensor& input,
                                                               const Tensor& output,
                                                               const Tensor& segment_offsets,
                                                               uint64_t num_items,
                                                               uint64_t num_segments,
                                                               CubArgScanOp op,
                                                               CubScanMode mode,
                                                               CubScanDirection direction) {
    validateSegmentedArgScan(input, output, segment_offsets, num_items, num_segments);
    size_t bytes = 1;
    if (num_items != 0 && num_segments != 0) {
        auto query_value = [&]<typename T>() -> size_t {
            auto query_offset = [&]<typename OffsetT>() -> size_t {
                auto query_op = [&](auto scan_op, CubSegmentedArgScanPair<T> init) -> size_t {
                    return segmentedArgScanPairStorageBytes<T>(num_items) +
                           querySegmentedArgScan<T, OffsetT>(input, segment_offsets, num_items, num_segments, mode, direction, scan_op, init);
                };
                return dispatchSegmentedArgScanOperator<T>(op, query_op);
            };
            return dispatchSegmentOffsetDType(segment_offsets.getDataType(), query_offset);
        };
        bytes = dispatchScanDType(input.getDataType(), query_value);
    }
    CubDeviceSegmentedArgScanPlan plan;
    plan.placement = input.getPlacement();
    plan.dtype = input.getDataType();
    plan.offset_dtype = segment_offsets.getDataType();
    plan.num_items = num_items;
    plan.num_segments = num_segments;
    plan.op = op;
    plan.mode = mode;
    plan.direction = direction;
    plan.temp_storage_bytes = std::max<size_t>(bytes, 1);
    return plan;
}

size_t cubDeviceSegmentedArgScanTempBytes(const Tensor& input,
                                          const Tensor& output,
                                          const Tensor& segment_offsets,
                                          uint64_t num_items,
                                          uint64_t num_segments,
                                          CubArgScanOp op,
                                          CubScanMode mode,
                                          CubScanDirection direction) {
    return prepareCubDeviceSegmentedArgScan(input, output, segment_offsets, num_items, num_segments, op, mode, direction).temp_storage_bytes;
}

void cubDeviceSegmentedArgScan(const CubDeviceSegmentedArgScanPlan& plan,
                               const Tensor& temp_storage,
                               const Tensor& input,
                               Tensor& output,
                               const Tensor& segment_offsets,
                               Stream& stream) {
    validateSegmentedArgScan(input, output, segment_offsets, plan.num_items, plan.num_segments);
    if (input.getPlacement() != plan.placement || input.getDataType() != plan.dtype || segment_offsets.getDataType() != plan.offset_dtype) {
        throw std::invalid_argument("CUB segmented arg scan plan is not compatible with the provided tensors.");
    }
    requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    if (plan.num_items == 0 || plan.num_segments == 0) {
        return;
    }
    auto launch_value = [&]<typename T>() -> void {
        auto launch_offset = [&]<typename OffsetT>() -> void {
            auto launch_op = [&](auto scan_op, CubSegmentedArgScanPair<T> init) -> void {
                launchSegmentedArgScan<T, OffsetT>(plan, temp_storage, input, output, segment_offsets, stream, scan_op, init);
            };
            dispatchSegmentedArgScanOperator<T>(plan.op, launch_op);
        };
        dispatchSegmentOffsetDType(plan.offset_dtype, launch_offset);
    };
    dispatchScanDType(plan.dtype, launch_value);
}

}  // namespace ThorImplementation
