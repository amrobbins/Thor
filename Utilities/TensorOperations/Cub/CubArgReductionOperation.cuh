#pragma once

#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/iterator>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ThorImplementation::CubReductionInternal {

// 1261: the CUB dense ARG adapters (direct and interval-composed, CubArgReductionMin / Max.cu) are deleted; dense ARG
// runs the retained planned passes. What remains are the FP32 candidate rules and runtime output stores that the
// offset-segmented ARG kernels (CubArgReductionOffsetSegmented.cu) use.

struct ArgReductionCandidateFp32 {
    uint64_t index;
    float value;
};

template <typename IndexT>
struct DenseArgReductionCandidateFp32 {
    IndexT index;
    float value;
};

static_assert(std::is_trivially_copyable_v<ArgReductionCandidateFp32>);
static_assert(std::is_trivially_copyable_v<DenseArgReductionCandidateFp32<uint32_t>>);
static_assert(std::is_trivially_copyable_v<DenseArgReductionCandidateFp32<uint64_t>>);

struct ArgMinimumCandidateFp32 {
    template <typename CandidateT>
    __host__ __device__ CandidateT operator()(const CandidateT& lhs, const CandidateT& rhs) const {
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
    template <typename CandidateT>
    __host__ __device__ CandidateT operator()(const CandidateT& lhs, const CandidateT& rhs) const {
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

inline __host__ __device__ void storeArgIndexAsRuntimeDType(void* output, DataType output_dtype, uint64_t output_index, uint64_t value) {
    switch (output_dtype) {
        case DataType::UINT32:
            static_cast<uint32_t*>(output)[output_index] = static_cast<uint32_t>(value);
            return;
        case DataType::UINT64:
            static_cast<uint64_t*>(output)[output_index] = value;
            return;
        default:
            return;
    }
}

struct StoreArgReductionResultRuntime {
    void* value_output;
    DataType value_output_dtype;
    void* index_output;
    DataType index_output_dtype;

    template <typename OutputIndexT, typename CandidateT>
    __host__ __device__ void operator()(OutputIndexT output_index, CandidateT candidate) const {
        const uint64_t index = static_cast<uint64_t>(output_index);
        if (value_output != nullptr) {
            storeFp32AsRuntimeDType(value_output, value_output_dtype, index, candidate.value);
        }
        if (index_output != nullptr) {
            storeArgIndexAsRuntimeDType(index_output, index_output_dtype, index, static_cast<uint64_t>(candidate.index));
        }
    }
};

inline auto makeRuntimeArgReductionOutputIterator(Tensor* value_output, Tensor* index_output) {
    return cuda::make_tabulate_output_iterator(
        StoreArgReductionResultRuntime{value_output == nullptr ? nullptr : value_output->getMemPtr<void>(),
                                       value_output == nullptr ? DataType::FP32 : value_output->getDataType(),
                                       index_output == nullptr ? nullptr : index_output->getMemPtr<void>(),
                                       index_output == nullptr ? DataType::UINT32 : index_output->getDataType()});
}

inline auto makeRuntimeArgReductionOutputIterator(std::optional<DataType> value_output_dtype,
                                                   std::optional<DataType> index_output_dtype) {
    return cuda::make_tabulate_output_iterator(
        StoreArgReductionResultRuntime{nullptr,
                                       value_output_dtype.value_or(DataType::FP32),
                                       nullptr,
                                       index_output_dtype.value_or(DataType::UINT32)});
}

}  // namespace ThorImplementation::CubReductionInternal
