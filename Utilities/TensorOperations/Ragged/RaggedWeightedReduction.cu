#include "Utilities/TensorOperations/Ragged/RaggedWeightedReduction.h"

#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;
using CubReductionInternal::dispatchReductionInputDType;
using CubReductionInternal::ToFp32;

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kTargetItemsPerThread = 8;
constexpr uint32_t kMaxPartialBlocks = 4096;

struct WeightedStatistics {
    float numerator;
    float denominator;
};
static_assert(sizeof(WeightedStatistics) == 2 * sizeof(float));

struct AddWeightedStatistics {
    __host__ __device__ __forceinline__ WeightedStatistics operator()(WeightedStatistics lhs,
                                                                      WeightedStatistics rhs) const {
        return {lhs.numerator + rhs.numerator, lhs.denominator + rhs.denominator};
    }
};

uint32_t partialBlockCount(uint64_t scalar_count) {
    if (scalar_count == 0) return 1;
    const uint64_t target_scalars_per_block =
        static_cast<uint64_t>(kBlockSize) * kTargetItemsPerThread;
    const uint64_t requested_blocks = 1 + (scalar_count - 1) / target_scalars_per_block;
    return static_cast<uint32_t>(std::min<uint64_t>(requested_blocks, kMaxPartialBlocks));
}

__device__ __forceinline__ void storeFinalStatistics(WeightedStatistics total,
                                                      float* numerator,
                                                      float* denominator) {
    *denominator = total.denominator;
    // WeightedMean defines zero total weight as no contribution. Preserve that
    // contract even when ignored values contain NaN and 0 * NaN made one or
    // more numerator partials non-finite.
    *numerator = total.denominator == 0.0f ? 0.0f : total.numerator;
}

template <typename ValueT, typename WeightT, typename IndexT>
__global__ void weightedStatisticsPartialsKernel(const ValueT* values,
                                                 const WeightT* weights,
                                                 WeightedStatistics* partials,
                                                 IndexT active_scalar_count,
                                                 float* numerator,
                                                 float* denominator) {
    using BlockReduce = cub::BlockReduce<WeightedStatistics, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage reduction_storage;

    WeightedStatistics local{0.0f, 0.0f};
    const IndexT first = static_cast<IndexT>(blockIdx.x * kBlockSize + threadIdx.x);
    const IndexT stride = static_cast<IndexT>(gridDim.x * kBlockSize);
    IndexT i = first;
    while (i < active_scalar_count) {
        const float weight = ToFp32<WeightT>{}(weights[i]);
        local.numerator += ToFp32<ValueT>{}(values[i]) * weight;
        local.denominator += weight;
        if (stride >= active_scalar_count - i) break;
        i += stride;
    }

    const WeightedStatistics block_statistics =
        BlockReduce(reduction_storage).Reduce(local, AddWeightedStatistics{});
    if (threadIdx.x == 0) {
        // Small active prefixes need only one CTA. Finish the sufficient
        // statistics here instead of writing a workspace partial and launching
        // a second kernel solely to read that one partial back.
        if (gridDim.x == 1) {
            storeFinalStatistics(block_statistics, numerator, denominator);
        } else {
            partials[blockIdx.x] = block_statistics;
        }
    }
}

__global__ void finalizeWeightedStatisticsKernel(const WeightedStatistics* partials,
                                                 uint32_t partial_count,
                                                 float* numerator,
                                                 float* denominator) {
    using BlockReduce = cub::BlockReduce<WeightedStatistics, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage reduction_storage;

    WeightedStatistics local{0.0f, 0.0f};
    for (uint32_t i = threadIdx.x; i < partial_count; i += kBlockSize) {
        local.numerator += partials[i].numerator;
        local.denominator += partials[i].denominator;
    }

    const WeightedStatistics total = BlockReduce(reduction_storage).Reduce(local, AddWeightedStatistics{});
    if (threadIdx.x == 0) storeFinalStatistics(total, numerator, denominator);
}

void validate(const Tensor& values,
              const Tensor& weights,
              const Tensor& partial_statistics,
              const Tensor& numerator,
              const Tensor& denominator,
              uint64_t active_value_count,
              uint64_t max_total_values,
              uint64_t elements_per_value) {
    requireDenseContiguousGpuTensor(values, "ragged WeightedMean values");
    requireDenseContiguousGpuTensor(weights, "ragged WeightedMean weights");
    requireDenseContiguousGpuTensor(partial_statistics, "ragged WeightedMean partial statistics");
    requireDenseContiguousGpuTensor(numerator, "ragged WeightedMean numerator");
    requireDenseContiguousGpuTensor(denominator, "ragged WeightedMean denominator");
    requireSameGpuPlacement(values, weights, "ragged WeightedMean values", "ragged WeightedMean weights");
    requireSameGpuPlacement(values,
                            partial_statistics,
                            "ragged WeightedMean values",
                            "ragged WeightedMean partial statistics");
    requireSameGpuPlacement(values, numerator, "ragged WeightedMean values", "ragged WeightedMean numerator");
    requireSameGpuPlacement(values, denominator, "ragged WeightedMean values", "ragged WeightedMean denominator");
    if (!CubSegmentedReduction::isInputDataTypeSupported(values.getDataType()) ||
        !CubSegmentedReduction::isInputDataTypeSupported(weights.getDataType())) {
        throw std::invalid_argument("ragged WeightedMean values and weights must use a CUB floating reduction dtype.");
    }
    if (numerator.getDataType() != DataType::FP32 || numerator.getTotalNumElements() != 1 ||
        denominator.getDataType() != DataType::FP32 || denominator.getTotalNumElements() != 1) {
        throw std::invalid_argument("ragged WeightedMean sufficient statistics must be FP32 scalars.");
    }
    if (max_total_values == 0)
        throw std::invalid_argument("ragged WeightedMean max_total_values must be non-zero.");
    if (elements_per_value == 0)
        throw std::invalid_argument("ragged WeightedMean elements_per_value must be non-zero.");
    if (active_value_count > max_total_values)
        throw std::invalid_argument("ragged WeightedMean active_value_count exceeds packed capacity.");
    if (values.getDimensions() != weights.getDimensions())
        throw std::invalid_argument("ragged WeightedMean values and weights dimensions must match.");
    const std::vector<uint64_t>& dimensions = values.getDimensions();
    if (dimensions.empty() || dimensions.front() != max_total_values)
        throw std::invalid_argument(
            "ragged WeightedMean packed tensors must use max_total_values as their leading dimension.");
    uint64_t inferred_elements_per_value = 1;
    for (size_t axis = 1; axis < dimensions.size(); ++axis) {
        if (dimensions[axis] == 0 ||
            inferred_elements_per_value > std::numeric_limits<uint64_t>::max() / dimensions[axis]) {
            throw std::overflow_error("ragged WeightedMean trailing element count overflows uint64_t.");
        }
        inferred_elements_per_value *= dimensions[axis];
    }
    if (inferred_elements_per_value != elements_per_value)
        throw std::invalid_argument("ragged WeightedMean elements_per_value does not match packed tensor dimensions.");
    if (max_total_values > std::numeric_limits<uint64_t>::max() / elements_per_value)
        throw std::overflow_error("ragged WeightedMean packed scalar capacity overflows uint64_t.");
    const uint64_t max_scalars = max_total_values * elements_per_value;
    if (values.getTotalNumElements() != max_scalars || weights.getTotalNumElements() != max_scalars)
        throw std::invalid_argument("ragged WeightedMean packed tensors do not match the declared capacity.");
    requireStorageForNumItems(values, "ragged WeightedMean values", max_scalars);
    requireStorageForNumItems(weights, "ragged WeightedMean weights", max_scalars);

    const TensorDescriptor expected_workspace =
        raggedWeightedMeanStatisticsWorkspaceDescriptor(max_total_values, elements_per_value);
    if (partial_statistics.getDescriptor() != expected_workspace) {
        throw std::invalid_argument("ragged WeightedMean partial-statistics workspace has the wrong descriptor.");
    }
}

template <typename ValueT, typename IndexT>
void dispatchWeightAndLaunch(const Tensor& values,
                             const Tensor& weights,
                             Tensor& partial_statistics,
                             IndexT active_scalar_count,
                             uint32_t partial_count,
                             Tensor& numerator,
                             Tensor& denominator,
                             Stream& stream) {
    const ValueT* values_ptr = values.getMemPtr<ValueT>();
    auto launch = [&]<typename WeightT>() {
        const WeightT* weights_ptr = weights.getMemPtr<WeightT>();
        weightedStatisticsPartialsKernel<ValueT, WeightT, IndexT>
            <<<partial_count, kBlockSize, 0, stream.getStream()>>>(
                values_ptr,
                weights_ptr,
                reinterpret_cast<WeightedStatistics*>(partial_statistics.getMemPtr<float>()),
                active_scalar_count,
                numerator.getMemPtr<float>(),
                denominator.getMemPtr<float>());
        CUDA_CHECK(cudaPeekAtLastError());
    };
    dispatchReductionInputDType(weights.getDataType(), launch);
}

template <typename IndexT>
void dispatchValueAndLaunch(const Tensor& values,
                            const Tensor& weights,
                            Tensor& partial_statistics,
                            IndexT active_scalar_count,
                            uint32_t partial_count,
                            Tensor& numerator,
                            Tensor& denominator,
                            Stream& stream) {
    auto launch = [&]<typename ValueT>() {
        dispatchWeightAndLaunch<ValueT, IndexT>(values,
                                                weights,
                                                partial_statistics,
                                                active_scalar_count,
                                                partial_count,
                                                numerator,
                                                denominator,
                                                stream);
    };
    dispatchReductionInputDType(values.getDataType(), launch);
}

void dispatchIndexAndLaunch(const Tensor& values,
                            const Tensor& weights,
                            Tensor& partial_statistics,
                            uint64_t active_scalar_count,
                            uint32_t partial_count,
                            Tensor& numerator,
                            Tensor& denominator,
                            Stream& stream) {
    if (active_scalar_count <= std::numeric_limits<uint32_t>::max()) {
        dispatchValueAndLaunch<uint32_t>(values,
                                         weights,
                                         partial_statistics,
                                         static_cast<uint32_t>(active_scalar_count),
                                         partial_count,
                                         numerator,
                                         denominator,
                                         stream);
    } else {
        dispatchValueAndLaunch<uint64_t>(values,
                                         weights,
                                         partial_statistics,
                                         active_scalar_count,
                                         partial_count,
                                         numerator,
                                         denominator,
                                         stream);
    }
}

}  // namespace

TensorDescriptor raggedWeightedMeanStatisticsWorkspaceDescriptor(uint64_t max_total_values,
                                                                  uint64_t elements_per_value) {
    if (max_total_values == 0)
        throw std::invalid_argument("ragged WeightedMean max_total_values must be non-zero.");
    if (elements_per_value == 0)
        throw std::invalid_argument("ragged WeightedMean elements_per_value must be non-zero.");
    if (max_total_values > std::numeric_limits<uint64_t>::max() / elements_per_value)
        throw std::overflow_error("ragged WeightedMean packed scalar capacity overflows uint64_t.");
    const uint64_t max_scalars = max_total_values * elements_per_value;
    return TensorDescriptor(DataType::FP32, {partialBlockCount(max_scalars), 2});
}

void raggedWeightedMeanStatistics(const Tensor& values,
                                  const Tensor& weights,
                                  Tensor& partial_statistics,
                                  Tensor& numerator,
                                  Tensor& denominator,
                                  uint64_t active_value_count,
                                  uint64_t max_total_values,
                                  uint64_t elements_per_value,
                                  Stream& stream) {
    validate(values,
             weights,
             partial_statistics,
             numerator,
             denominator,
             active_value_count,
             max_total_values,
             elements_per_value);

    const uint64_t active_scalar_count = active_value_count * elements_per_value;
    const uint32_t partial_count = partialBlockCount(active_scalar_count);
    if (partial_count > partial_statistics.getDimensions().front())
        throw std::logic_error("ragged WeightedMean runtime partial count exceeds its capacity-sized workspace.");

    dispatchIndexAndLaunch(values,
                           weights,
                           partial_statistics,
                           active_scalar_count,
                           partial_count,
                           numerator,
                           denominator,
                           stream);

    if (partial_count > 1) {
        finalizeWeightedStatisticsKernel<<<1, kBlockSize, 0, stream.getStream()>>>(
            reinterpret_cast<const WeightedStatistics*>(partial_statistics.getMemPtr<float>()),
            partial_count,
            numerator.getMemPtr<float>(),
            denominator.getMemPtr<float>());
        CUDA_CHECK(cudaPeekAtLastError());
    }
}

}  // namespace ThorImplementation
