#include "Utilities/TensorOperations/Ragged/RaggedWeightedReduction.h"

#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"
#include "Utilities/TensorOperations/Ragged/RowPartition.h"

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
constexpr uint32_t kMaxBlocks = 4096;

template <typename ValueT, typename WeightT, typename OffsetT>
__global__ void weightedStatisticsKernel(const ValueT* values,
                                         const WeightT* weights,
                                         const OffsetT* offsets,
                                         float* numerator,
                                         float* denominator,
                                         uint64_t valid_row_count,
                                         uint64_t elements_per_value) {
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage numerator_storage;
    __shared__ typename BlockReduce::TempStorage denominator_storage;

    for (uint64_t row = blockIdx.x; row < valid_row_count; row += gridDim.x) {
        const uint64_t scalar_begin = static_cast<uint64_t>(offsets[row]) * elements_per_value;
        const uint64_t scalar_end = static_cast<uint64_t>(offsets[row + 1]) * elements_per_value;

        float local_num = 0.0f;
        float local_den = 0.0f;
        for (uint64_t i = scalar_begin + threadIdx.x; i < scalar_end; i += blockDim.x) {
            const float w = ToFp32<WeightT>{}(weights[i]);
            local_num += ToFp32<ValueT>{}(values[i]) * w;
            local_den += w;
        }

        const float block_num = BlockReduce(numerator_storage).Sum(local_num);
        __syncthreads();
        const float block_den = BlockReduce(denominator_storage).Sum(local_den);
        if (threadIdx.x == 0) {
            atomicAdd(numerator, block_num);
            atomicAdd(denominator, block_den);
        }
        __syncthreads();
    }
}

void validate(const Tensor& values,
              const Tensor& weights,
              const Tensor& offsets,
              const Tensor& numerator,
              const Tensor& denominator,
              uint64_t valid_row_count,
              uint64_t max_total_values,
              uint64_t elements_per_value) {
    requireDenseContiguousGpuTensor(values, "ragged WeightedMean values");
    requireDenseContiguousGpuTensor(weights, "ragged WeightedMean weights");
    requireDenseContiguousGpuTensor(offsets, "ragged WeightedMean offsets");
    requireDenseContiguousGpuTensor(numerator, "ragged WeightedMean numerator");
    requireDenseContiguousGpuTensor(denominator, "ragged WeightedMean denominator");
    requireSameGpuPlacement(values, weights, "ragged WeightedMean values", "ragged WeightedMean weights");
    requireSameGpuPlacement(values, offsets, "ragged WeightedMean values", "ragged WeightedMean offsets");
    requireSameGpuPlacement(values, numerator, "ragged WeightedMean values", "ragged WeightedMean numerator");
    requireSameGpuPlacement(values, denominator, "ragged WeightedMean values", "ragged WeightedMean denominator");
    if (!CubSegmentedReduction::isInputDataTypeSupported(values.getDataType()) ||
        !CubSegmentedReduction::isInputDataTypeSupported(weights.getDataType())) {
        throw std::invalid_argument("ragged WeightedMean values and weights must use a CUB floating reduction dtype.");
    }
    if (!isRowPartitionOffsetDTypeSupported(offsets.getDataType()))
        throw std::invalid_argument("ragged WeightedMean offsets must be UINT32 or UINT64.");
    if (numerator.getDataType() != DataType::FP32 || numerator.getTotalNumElements() != 1 ||
        denominator.getDataType() != DataType::FP32 || denominator.getTotalNumElements() != 1) {
        throw std::invalid_argument("ragged WeightedMean sufficient statistics must be FP32 scalars.");
    }
    if (valid_row_count == 0)
        throw std::invalid_argument("ragged WeightedMean valid_row_count must be non-zero.");
    if (max_total_values == 0)
        throw std::invalid_argument("ragged WeightedMean max_total_values must be non-zero.");
    if (elements_per_value == 0)
        throw std::invalid_argument("ragged WeightedMean elements_per_value must be non-zero.");
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
    requireStorageForNumItems(offsets, "ragged WeightedMean offsets", valid_row_count + 1);
}


__global__ void canonicalizeZeroWeightStatisticsKernel(float* numerator, const float* denominator) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && *denominator == 0.0f)
        *numerator = 0.0f;
}

template <typename ValueT, typename OffsetT>
void dispatchWeightAndLaunch(const Tensor& values,
                             const Tensor& weights,
                             const Tensor& offsets,
                             Tensor& numerator,
                             Tensor& denominator,
                             uint64_t valid_row_count,
                             uint64_t elements_per_value,
                             Stream& stream) {
    const ValueT* values_ptr = values.getMemPtr<ValueT>();
    const OffsetT* offsets_ptr = offsets.getMemPtr<OffsetT>();
    auto launch = [&]<typename WeightT>() {
        const WeightT* weights_ptr = weights.getMemPtr<WeightT>();
        const uint32_t blocks =
            static_cast<uint32_t>(std::min<uint64_t>(valid_row_count, kMaxBlocks));
        weightedStatisticsKernel<ValueT, WeightT, OffsetT><<<blocks, kBlockSize, 0, stream.getStream()>>>(
            values_ptr,
            weights_ptr,
            offsets_ptr,
            numerator.getMemPtr<float>(),
            denominator.getMemPtr<float>(),
            valid_row_count,
            elements_per_value);
        CUDA_CHECK(cudaPeekAtLastError());
    };
    dispatchReductionInputDType(weights.getDataType(), launch);
}

template <typename OffsetT>
void dispatchValueAndLaunch(const Tensor& values,
                            const Tensor& weights,
                            const Tensor& offsets,
                            Tensor& numerator,
                            Tensor& denominator,
                            uint64_t valid_row_count,
                            uint64_t elements_per_value,
                            Stream& stream) {
    auto launch = [&]<typename ValueT>() {
        dispatchWeightAndLaunch<ValueT, OffsetT>(values,
                                                  weights,
                                                  offsets,
                                                  numerator,
                                                  denominator,
                                                  valid_row_count,
                                                  elements_per_value,
                                                  stream);
    };
    dispatchReductionInputDType(values.getDataType(), launch);
}

}  // namespace

void raggedWeightedMeanStatistics(const Tensor& values,
                                  const Tensor& weights,
                                  const Tensor& offsets,
                                  Tensor& numerator,
                                  Tensor& denominator,
                                  uint64_t valid_row_count,
                                  uint64_t max_total_values,
                                  uint64_t elements_per_value,
                                  Stream& stream) {
    validate(values,
             weights,
             offsets,
             numerator,
             denominator,
             valid_row_count,
             max_total_values,
             elements_per_value);
    CUDA_CHECK(cudaMemsetAsync(numerator.getMemPtr<void>(), 0, sizeof(float), stream.getStream()));
    CUDA_CHECK(cudaMemsetAsync(denominator.getMemPtr<void>(), 0, sizeof(float), stream.getStream()));
    if (offsets.getDataType() == DataType::UINT32) {
        dispatchValueAndLaunch<uint32_t>(values,
                                         weights,
                                         offsets,
                                         numerator,
                                         denominator,
                                         valid_row_count,
                                         elements_per_value,
                                         stream);
    } else if (offsets.getDataType() == DataType::UINT64) {
        dispatchValueAndLaunch<uint64_t>(values,
                                         weights,
                                         offsets,
                                         numerator,
                                         denominator,
                                         valid_row_count,
                                         elements_per_value,
                                         stream);
    } else {
        throw std::invalid_argument("ragged WeightedMean offsets must be UINT32 or UINT64.");
    }
    canonicalizeZeroWeightStatisticsKernel<<<1, 1, 0, stream.getStream()>>>(
        numerator.getMemPtr<float>(), denominator.getMemPtr<float>());
    CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace ThorImplementation
