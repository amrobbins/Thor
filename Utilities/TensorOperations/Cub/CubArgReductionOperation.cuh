#pragma once

#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubReductionIndexing.cuh"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/iterator>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ThorImplementation::CubReductionInternal {

struct ArgReductionCandidateFp32 {
    uint64_t index;
    float value;
};

static_assert(std::is_trivially_copyable_v<ArgReductionCandidateFp32>);

struct ArgMinimumCandidateFp32 {
    __host__ __device__ ArgReductionCandidateFp32 operator()(const ArgReductionCandidateFp32& lhs,
                                                              const ArgReductionCandidateFp32& rhs) const {
        const bool lhs_nan = lhs.value != lhs.value;
        const bool rhs_nan = rhs.value != rhs.value;
        if (lhs_nan || rhs_nan) {
            if (lhs_nan && rhs_nan) {
                return lhs.index <= rhs.index ? lhs : rhs;
            }
            return lhs_nan ? lhs : rhs;
        }
        if (lhs.value < rhs.value) {
            return lhs;
        }
        if (rhs.value < lhs.value) {
            return rhs;
        }
        return lhs.index <= rhs.index ? lhs : rhs;
    }
};

struct ArgMaximumCandidateFp32 {
    __host__ __device__ ArgReductionCandidateFp32 operator()(const ArgReductionCandidateFp32& lhs,
                                                              const ArgReductionCandidateFp32& rhs) const {
        const bool lhs_nan = lhs.value != lhs.value;
        const bool rhs_nan = rhs.value != rhs.value;
        if (lhs_nan || rhs_nan) {
            if (lhs_nan && rhs_nan) {
                return lhs.index <= rhs.index ? lhs : rhs;
            }
            return lhs_nan ? lhs : rhs;
        }
        if (lhs.value > rhs.value) {
            return lhs;
        }
        if (rhs.value > lhs.value) {
            return rhs;
        }
        return lhs.index <= rhs.index ? lhs : rhs;
    }
};

inline __host__ __device__ void storeArgIndexAsRuntimeDType(void* output,
                                                            DataType output_dtype,
                                                            uint64_t output_index,
                                                            uint64_t value) {
    switch (output_dtype) {
        case DataType::UINT32:
            static_cast<uint32_t*>(output)[output_index] = static_cast<uint32_t>(value);
            return;
        case DataType::UINT64:
            static_cast<uint64_t*>(output)[output_index] = value;
            return;
        default:
#if defined(__CUDA_ARCH__)
            asm("trap;");
#endif
            return;
    }
}

struct StoreArgReductionResultRuntime {
    void* value_output;
    DataType value_output_dtype;
    void* index_output;
    DataType index_output_dtype;

    template <typename IndexT>
    __host__ __device__ void operator()(IndexT output_index, ArgReductionCandidateFp32 candidate) const {
        const uint64_t index = static_cast<uint64_t>(output_index);
        if (value_output != nullptr) {
            storeFp32AsRuntimeDType(value_output, value_output_dtype, index, candidate.value);
        }
        if (index_output != nullptr) {
            storeArgIndexAsRuntimeDType(index_output, index_output_dtype, index, candidate.index);
        }
    }
};

inline auto makeRuntimeArgReductionOutputIterator(Tensor* value_output, Tensor* index_output) {
    return cuda::make_tabulate_output_iterator(StoreArgReductionResultRuntime{
        value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
        value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
        index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
        index_output == nullptr ? DataType::UINT32 : index_output->getDataType()});
}

template <typename InputT>
struct DeviceArgCandidateInput {
    const InputT* input;

    __host__ __device__ ArgReductionCandidateFp32 operator()(int64_t logical_index) const {
        const uint64_t index = static_cast<uint64_t>(logical_index);
        return ArgReductionCandidateFp32{index, ToFp32<InputT>{}(input[index])};
    }
};

template <typename InputT>
struct ContiguousArgCandidateInput {
    const InputT* input;
    uint64_t reduction_size;

    __host__ __device__ ArgReductionCandidateFp32 operator()(int64_t logical_index) const {
        const uint64_t index = static_cast<uint64_t>(logical_index);
        return ArgReductionCandidateFp32{index % reduction_size, ToFp32<InputT>{}(input[index])};
    }
};

template <typename InputT>
struct StridedArgCandidateInput {
    const InputT* input;
    uint64_t reduction_size;
    CubReductionDeviceIndexing indexing;

    __host__ __device__ ArgReductionCandidateFp32 operator()(int64_t logical_index) const {
        const uint64_t index = static_cast<uint64_t>(logical_index);
        const uint64_t output_index = index / reduction_size;
        const uint64_t reduction_index = index - output_index * reduction_size;
        const uint64_t physical_index = mapLogicalReductionIndex(indexing, output_index, reduction_index);
        return ArgReductionCandidateFp32{reduction_index, ToFp32<InputT>{}(input[physical_index])};
    }
};

template <typename InputT>
auto makeDeviceArgCandidateIterator(const Tensor& input) {
    return thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0),
                                           DeviceArgCandidateInput<InputT>{input.getMemPtr<InputT>()});
}

template <typename InputT>
auto makeContiguousArgCandidateIterator(const Tensor& input, const CubReductionGeometry& geometry) {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<int64_t>(0),
        ContiguousArgCandidateInput<InputT>{input.getMemPtr<InputT>(), geometry.reduction_size});
}

template <typename InputT>
auto makeStridedArgCandidateIterator(const Tensor& input, const CubReductionGeometry& geometry) {
    return thrust::make_transform_iterator(
        thrust::counting_iterator<int64_t>(0),
        StridedArgCandidateInput<InputT>{input.getMemPtr<InputT>(), geometry.reduction_size, geometry.device_indexing});
}

template <typename InputT, typename ReductionOpT>
size_t queryArgReductionBytesForInput(const Tensor& input,
                                      Tensor* value_output,
                                      Tensor* index_output,
                                      const CubReductionGeometry& geometry,
                                      ReductionOpT reduction_op,
                                      ArgReductionCandidateFp32 init,
                                      cudaStream_t stream) {
    using AccumulatorT = std::decay_t<decltype(std::declval<ReductionOpT>()(
        std::declval<ArgReductionCandidateFp32>(), std::declval<ArgReductionCandidateFp32>()))>;
    static_assert(std::is_same_v<AccumulatorT, ArgReductionCandidateFp32>,
                  "CUB arg reductions must preserve the FP32 candidate state.");

    size_t queried_bytes = 0;
    auto output_iterator = makeRuntimeArgReductionOutputIterator(value_output, index_output);

    switch (geometry.path) {
        case CubReductionPath::DeviceTransformReduce: {
            auto input_iterator = makeDeviceArgCandidateIterator<InputT>(input);
            CUDA_CHECK(cub::DeviceReduce::Reduce(nullptr,
                                                 queried_bytes,
                                                 input_iterator,
                                                 output_iterator,
                                                 static_cast<int64_t>(geometry.input_elements),
                                                 reduction_op,
                                                 init,
                                                 stream));
            break;
        }
        case CubReductionPath::ContiguousFixedSegment: {
            auto input_iterator = makeContiguousArgCandidateIterator<InputT>(input, geometry);
            CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                          queried_bytes,
                                                          input_iterator,
                                                          output_iterator,
                                                          static_cast<int64_t>(geometry.output_elements),
                                                          static_cast<int>(geometry.reduction_size),
                                                          reduction_op,
                                                          init,
                                                          stream));
            break;
        }
        case CubReductionPath::StridedFixedSegment: {
            auto input_iterator = makeStridedArgCandidateIterator<InputT>(input, geometry);
            CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                          queried_bytes,
                                                          input_iterator,
                                                          output_iterator,
                                                          static_cast<int64_t>(geometry.output_elements),
                                                          static_cast<int>(geometry.reduction_size),
                                                          reduction_op,
                                                          init,
                                                          stream));
            break;
        }
        case CubReductionPath::OffsetSegmented:
            throw std::logic_error("Dense CUB arg reduction received offset-segmented geometry.");
    }

    return std::max<size_t>(queried_bytes, 1);
}

template <typename InputT, typename ReductionOpT>
void launchArgReductionForInput(const Tensor& temp_storage,
                                size_t temp_storage_bytes,
                                const Tensor& input,
                                Tensor* value_output,
                                Tensor* index_output,
                                const CubReductionGeometry& geometry,
                                ReductionOpT reduction_op,
                                ArgReductionCandidateFp32 init,
                                cudaStream_t stream) {
    void* temp_storage_ptr = const_cast<void*>(static_cast<const void*>(temp_storage.getMemPtr<void>()));
    auto output_iterator = makeRuntimeArgReductionOutputIterator(value_output, index_output);

    switch (geometry.path) {
        case CubReductionPath::DeviceTransformReduce: {
            auto input_iterator = makeDeviceArgCandidateIterator<InputT>(input);
            CUDA_CHECK(cub::DeviceReduce::Reduce(temp_storage_ptr,
                                                 temp_storage_bytes,
                                                 input_iterator,
                                                 output_iterator,
                                                 static_cast<int64_t>(geometry.input_elements),
                                                 reduction_op,
                                                 init,
                                                 stream));
            break;
        }
        case CubReductionPath::ContiguousFixedSegment: {
            auto input_iterator = makeContiguousArgCandidateIterator<InputT>(input, geometry);
            CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(temp_storage_ptr,
                                                          temp_storage_bytes,
                                                          input_iterator,
                                                          output_iterator,
                                                          static_cast<int64_t>(geometry.output_elements),
                                                          static_cast<int>(geometry.reduction_size),
                                                          reduction_op,
                                                          init,
                                                          stream));
            break;
        }
        case CubReductionPath::StridedFixedSegment: {
            auto input_iterator = makeStridedArgCandidateIterator<InputT>(input, geometry);
            CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(temp_storage_ptr,
                                                          temp_storage_bytes,
                                                          input_iterator,
                                                          output_iterator,
                                                          static_cast<int64_t>(geometry.output_elements),
                                                          static_cast<int>(geometry.reduction_size),
                                                          reduction_op,
                                                          init,
                                                          stream));
            break;
        }
        case CubReductionPath::OffsetSegmented:
            throw std::logic_error("Dense CUB arg reduction received offset-segmented geometry.");
    }
}

template <typename ReductionOpT>
size_t queryOperationArgReductionBytes(const Tensor& input,
                                       Tensor* value_output,
                                       Tensor* index_output,
                                       const CubReductionGeometry& geometry,
                                       ReductionOpT reduction_op,
                                       ArgReductionCandidateFp32 init,
                                       const Stream& stream) {
    auto dispatch_input = [&]<typename InputT>() -> size_t {
        return queryArgReductionBytesForInput<InputT>(input,
                                                      value_output,
                                                      index_output,
                                                      geometry,
                                                      reduction_op,
                                                      init,
                                                      stream.getStream());
    };
    return dispatchReductionInputDType(input.getDataType(), dispatch_input);
}

template <typename ReductionOpT>
void launchOperationArgReduction(const Tensor& temp_storage,
                                 size_t temp_storage_bytes,
                                 const Tensor& input,
                                 Tensor* value_output,
                                 Tensor* index_output,
                                 const CubReductionGeometry& geometry,
                                 ReductionOpT reduction_op,
                                 ArgReductionCandidateFp32 init,
                                 Stream& stream) {
    auto dispatch_input = [&]<typename InputT>() -> void {
        launchArgReductionForInput<InputT>(temp_storage,
                                           temp_storage_bytes,
                                           input,
                                           value_output,
                                           index_output,
                                           geometry,
                                           reduction_op,
                                           init,
                                           stream.getStream());
    };
    dispatchReductionInputDType(input.getDataType(), dispatch_input);
}

}  // namespace ThorImplementation::CubReductionInternal
