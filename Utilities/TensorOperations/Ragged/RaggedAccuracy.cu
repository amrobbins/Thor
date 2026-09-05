#include "Utilities/TensorOperations/Ragged/RaggedAccuracy.h"

#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"
#include "Utilities/TensorOperations/Ragged/RowPartition.h"

#include <cub/block/block_reduce.cuh>
#include <cuda_fp16.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation {
namespace {
using CubReductionInternal::ToFp32;
using namespace CubDevicePrimitiveSupport;

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kMaxBlocks = 4096;

bool isPredictionDTypeSupported(DataType dtype) {
    return dtype == DataType::FP16 || dtype == DataType::FP32;
}

bool isBinaryLabelDTypeSupported(DataType dtype) {
    return dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 ||
           dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 ||
           dtype == DataType::FP16 || dtype == DataType::FP32;
}

bool isPerClassLabelDTypeSupported(DataType dtype) { return isBinaryLabelDTypeSupported(dtype); }

bool isClassIndexLabelDTypeSupported(DataType dtype) {
    return dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 ||
           dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32;
}

template <typename Fn>
decltype(auto) dispatchPredictionDType(DataType dtype, Fn&& fn) {
    switch (dtype) {
        case DataType::FP16: return fn.template operator()<__half>();
        case DataType::FP32: return fn.template operator()<float>();
        default: throw std::invalid_argument("Ragged accuracy predictions must be FP16 or FP32.");
    }
}

template <typename Fn>
decltype(auto) dispatchBinaryOrPerClassLabelDType(DataType dtype, Fn&& fn) {
    switch (dtype) {
        case DataType::UINT8: return fn.template operator()<uint8_t>();
        case DataType::UINT16: return fn.template operator()<uint16_t>();
        case DataType::UINT32: return fn.template operator()<uint32_t>();
        case DataType::INT8: return fn.template operator()<int8_t>();
        case DataType::INT16: return fn.template operator()<int16_t>();
        case DataType::INT32: return fn.template operator()<int32_t>();
        case DataType::FP16: return fn.template operator()<__half>();
        case DataType::FP32: return fn.template operator()<float>();
        default: throw std::invalid_argument("Ragged accuracy labels use an unsupported dtype.");
    }
}

template <typename Fn>
decltype(auto) dispatchClassIndexLabelDType(DataType dtype, Fn&& fn) {
    switch (dtype) {
        case DataType::UINT8: return fn.template operator()<uint8_t>();
        case DataType::UINT16: return fn.template operator()<uint16_t>();
        case DataType::UINT32: return fn.template operator()<uint32_t>();
        case DataType::INT8: return fn.template operator()<int8_t>();
        case DataType::INT16: return fn.template operator()<int16_t>();
        case DataType::INT32: return fn.template operator()<int32_t>();
        default: throw std::invalid_argument("Ragged categorical class-index labels must use an integer dtype.");
    }
}

template <typename PredictionT, typename LabelT, typename OffsetT>
__global__ void binaryAccuracyKernel(const PredictionT* predictions,
                                     const LabelT* labels,
                                     const OffsetT* offsets,
                                     float* correct_count,
                                     float* token_count,
                                     uint64_t valid_row_count) {
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage correct_storage;

    for (uint64_t row = blockIdx.x; row < valid_row_count; row += gridDim.x) {
        const uint64_t begin = static_cast<uint64_t>(offsets[row]);
        const uint64_t end = static_cast<uint64_t>(offsets[row + 1]);
        float local_correct = 0.0f;
        for (uint64_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
            const float prediction = ToFp32<PredictionT>{}(predictions[token]);
            const float label = ToFp32<LabelT>{}(labels[token]);
            const float predicted_label = prediction >= 0.5f ? 1.0f : 0.0f;
            local_correct += predicted_label == label ? 1.0f : 0.0f;
        }

        const float block_correct = BlockReduce(correct_storage).Sum(local_correct);
        if (threadIdx.x == 0) {
            atomicAdd(correct_count, block_correct);
            atomicAdd(token_count, static_cast<float>(end - begin));
        }
        __syncthreads();
    }
}

template <typename T>
__device__ uint64_t tokenArgmax(const T* values, uint64_t token, uint64_t num_classes) {
    const uint64_t base = token * num_classes;
    uint64_t best_index = 0;
    float best_value = ToFp32<T>{}(values[base]);
    for (uint64_t c = 1; c < num_classes; ++c) {
        const float value = ToFp32<T>{}(values[base + c]);
        // Match ordinary argmax first-occurrence tie behavior.
        if (value > best_value) {
            best_value = value;
            best_index = c;
        }
    }
    return best_index;
}

template <typename PredictionT, typename LabelT, typename OffsetT, bool ClassIndexLabels>
__global__ void categoricalAccuracyKernel(const PredictionT* predictions,
                                          const LabelT* labels,
                                          const OffsetT* offsets,
                                          float* correct_count,
                                          float* token_count,
                                          uint64_t valid_row_count,
                                          uint64_t num_classes) {
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage correct_storage;

    for (uint64_t row = blockIdx.x; row < valid_row_count; row += gridDim.x) {
        const uint64_t begin = static_cast<uint64_t>(offsets[row]);
        const uint64_t end = static_cast<uint64_t>(offsets[row + 1]);
        float local_correct = 0.0f;
        for (uint64_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
            const uint64_t predicted_class = tokenArgmax(predictions, token, num_classes);
            bool correct = false;
            if constexpr (ClassIndexLabels) {
                const int64_t true_class = static_cast<int64_t>(labels[token]);
                correct = true_class >= 0 && static_cast<uint64_t>(true_class) == predicted_class;
            } else {
                const uint64_t true_class = tokenArgmax(labels, token, num_classes);
                correct = predicted_class == true_class;
            }
            local_correct += correct ? 1.0f : 0.0f;
        }

        const float block_correct = BlockReduce(correct_storage).Sum(local_correct);
        if (threadIdx.x == 0) {
            atomicAdd(correct_count, block_correct);
            atomicAdd(token_count, static_cast<float>(end - begin));
        }
        __syncthreads();
    }
}

void validateCommon(const Tensor& predictions,
                    const Tensor& labels,
                    const Tensor& offsets,
                    const Tensor& correct_count,
                    const Tensor& token_count,
                    uint64_t valid_row_count,
                    uint64_t max_total_values) {
    requireDenseContiguousGpuTensor(predictions, "ragged accuracy predictions");
    requireDenseContiguousGpuTensor(labels, "ragged accuracy labels");
    requireDenseContiguousGpuTensor(offsets, "ragged accuracy offsets");
    requireDenseContiguousGpuTensor(correct_count, "ragged accuracy correct count");
    requireDenseContiguousGpuTensor(token_count, "ragged accuracy token count");
    requireSameGpuPlacement(predictions, labels, "ragged accuracy predictions", "ragged accuracy labels");
    requireSameGpuPlacement(predictions, offsets, "ragged accuracy predictions", "ragged accuracy offsets");
    requireSameGpuPlacement(predictions, correct_count, "ragged accuracy predictions", "ragged accuracy correct count");
    requireSameGpuPlacement(predictions, token_count, "ragged accuracy predictions", "ragged accuracy token count");
    if (!isPredictionDTypeSupported(predictions.getDataType()))
        throw std::invalid_argument("Ragged accuracy predictions must be FP16 or FP32.");
    if (!isRowPartitionOffsetDTypeSupported(offsets.getDataType()))
        throw std::invalid_argument("Ragged accuracy offsets must be UINT32 or UINT64.");
    if (correct_count.getDataType() != DataType::FP32 || correct_count.getTotalNumElements() != 1 ||
        token_count.getDataType() != DataType::FP32 || token_count.getTotalNumElements() != 1) {
        throw std::invalid_argument("Ragged accuracy sufficient statistics must be FP32 scalars.");
    }
    if (valid_row_count == 0)
        throw std::invalid_argument("Ragged accuracy valid_row_count must be non-zero.");
    if (max_total_values == 0)
        throw std::invalid_argument("Ragged accuracy max_total_values must be non-zero.");
    const std::vector<uint64_t>& prediction_dims = predictions.getDimensions();
    const std::vector<uint64_t>& label_dims = labels.getDimensions();
    if (prediction_dims.empty() || prediction_dims.front() != max_total_values ||
        label_dims.empty() || label_dims.front() != max_total_values) {
        throw std::invalid_argument("Ragged accuracy packed tensors must use max_total_values as the leading dimension.");
    }
    const std::vector<uint64_t>& offset_dims = offsets.getDimensions();
    if (offset_dims.size() != 1 || valid_row_count + 1 > offset_dims.front())
        throw std::invalid_argument("Ragged accuracy offsets do not cover valid_row_count rows.");
    requireStorageForNumItems(offsets, "ragged accuracy offsets", valid_row_count + 1);
}

template <typename OffsetT>
void launchBinaryForOffset(const Tensor& predictions,
                           const Tensor& labels,
                           const Tensor& offsets,
                           Tensor& correct_count,
                           Tensor& token_count,
                           uint64_t valid_row_count,
                           Stream& stream) {
    auto launch_prediction = [&]<typename PredictionT>() {
        auto launch_label = [&]<typename LabelT>() {
            const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(valid_row_count, kMaxBlocks));
            binaryAccuracyKernel<PredictionT, LabelT, OffsetT><<<blocks, kBlockSize, 0, stream.getStream()>>>(
                predictions.getMemPtr<PredictionT>(),
                labels.getMemPtr<LabelT>(),
                offsets.getMemPtr<OffsetT>(),
                correct_count.getMemPtr<float>(),
                token_count.getMemPtr<float>(),
                valid_row_count);
            CUDA_CHECK(cudaPeekAtLastError());
        };
        dispatchBinaryOrPerClassLabelDType(labels.getDataType(), launch_label);
    };
    dispatchPredictionDType(predictions.getDataType(), launch_prediction);
}

template <typename OffsetT>
void launchCategoricalForOffset(const Tensor& predictions,
                                const Tensor& labels,
                                const Tensor& offsets,
                                Tensor& correct_count,
                                Tensor& token_count,
                                uint64_t valid_row_count,
                                uint64_t num_classes,
                                RaggedCategoricalLabelFormat label_format,
                                Stream& stream) {
    auto launch_prediction = [&]<typename PredictionT>() {
        if (label_format == RaggedCategoricalLabelFormat::CLASS_INDEX) {
            auto launch_label = [&]<typename LabelT>() {
                const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(valid_row_count, kMaxBlocks));
                categoricalAccuracyKernel<PredictionT, LabelT, OffsetT, true><<<blocks, kBlockSize, 0, stream.getStream()>>>(
                    predictions.getMemPtr<PredictionT>(),
                    labels.getMemPtr<LabelT>(),
                    offsets.getMemPtr<OffsetT>(),
                    correct_count.getMemPtr<float>(),
                    token_count.getMemPtr<float>(),
                    valid_row_count,
                    num_classes);
                CUDA_CHECK(cudaPeekAtLastError());
            };
            dispatchClassIndexLabelDType(labels.getDataType(), launch_label);
        } else {
            auto launch_label = [&]<typename LabelT>() {
                const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(valid_row_count, kMaxBlocks));
                categoricalAccuracyKernel<PredictionT, LabelT, OffsetT, false><<<blocks, kBlockSize, 0, stream.getStream()>>>(
                    predictions.getMemPtr<PredictionT>(),
                    labels.getMemPtr<LabelT>(),
                    offsets.getMemPtr<OffsetT>(),
                    correct_count.getMemPtr<float>(),
                    token_count.getMemPtr<float>(),
                    valid_row_count,
                    num_classes);
                CUDA_CHECK(cudaPeekAtLastError());
            };
            dispatchBinaryOrPerClassLabelDType(labels.getDataType(), launch_label);
        }
    };
    dispatchPredictionDType(predictions.getDataType(), launch_prediction);
}

}  // namespace

void raggedBinaryAccuracyStatistics(const Tensor& predictions,
                                    const Tensor& labels,
                                    const Tensor& offsets,
                                    Tensor& correct_count,
                                    Tensor& token_count,
                                    uint64_t valid_row_count,
                                    uint64_t max_total_values,
                                    Stream& stream) {
    validateCommon(predictions, labels, offsets, correct_count, token_count, valid_row_count, max_total_values);
    if (!isBinaryLabelDTypeSupported(labels.getDataType()))
        throw std::invalid_argument("Ragged BinaryAccuracy labels use an unsupported dtype.");
    if (predictions.getDimensions() != std::vector<uint64_t>{max_total_values, 1} ||
        labels.getDimensions() != std::vector<uint64_t>{max_total_values, 1}) {
        throw std::invalid_argument("Ragged BinaryAccuracy predictions and labels must contain one scalar per packed token.");
    }
    requireStorageForNumItems(predictions, "ragged BinaryAccuracy predictions", max_total_values);
    requireStorageForNumItems(labels, "ragged BinaryAccuracy labels", max_total_values);

    CUDA_CHECK(cudaMemsetAsync(correct_count.getMemPtr<float>(), 0, sizeof(float), stream.getStream()));
    CUDA_CHECK(cudaMemsetAsync(token_count.getMemPtr<float>(), 0, sizeof(float), stream.getStream()));
    if (offsets.getDataType() == DataType::UINT32) {
        launchBinaryForOffset<uint32_t>(predictions, labels, offsets, correct_count, token_count, valid_row_count, stream);
    } else {
        launchBinaryForOffset<uint64_t>(predictions, labels, offsets, correct_count, token_count, valid_row_count, stream);
    }
}

void raggedCategoricalAccuracyStatistics(const Tensor& predictions,
                                         const Tensor& labels,
                                         const Tensor& offsets,
                                         Tensor& correct_count,
                                         Tensor& token_count,
                                         uint64_t valid_row_count,
                                         uint64_t max_total_values,
                                         uint64_t num_classes,
                                         RaggedCategoricalLabelFormat label_format,
                                         Stream& stream) {
    validateCommon(predictions, labels, offsets, correct_count, token_count, valid_row_count, max_total_values);
    if (num_classes < 2)
        throw std::invalid_argument("Ragged CategoricalAccuracy requires at least two classes.");
    if (predictions.getDimensions() != std::vector<uint64_t>{max_total_values, num_classes})
        throw std::invalid_argument("Ragged CategoricalAccuracy predictions must have trailing class width num_classes.");
    if (max_total_values > std::numeric_limits<uint64_t>::max() / num_classes)
        throw std::overflow_error("Ragged CategoricalAccuracy packed prediction element count overflows uint64_t.");
    requireStorageForNumItems(
        predictions, "ragged CategoricalAccuracy predictions", max_total_values * num_classes);

    if (label_format == RaggedCategoricalLabelFormat::CLASS_INDEX) {
        if (!isClassIndexLabelDTypeSupported(labels.getDataType()))
            throw std::invalid_argument("Ragged CategoricalAccuracy class-index labels must use an integer dtype.");
        if (labels.getDimensions() != std::vector<uint64_t>{max_total_values, 1})
            throw std::invalid_argument("Ragged CategoricalAccuracy class-index labels must contain one scalar per packed token.");
        requireStorageForNumItems(labels, "ragged CategoricalAccuracy class-index labels", max_total_values);
    } else {
        if (!isPerClassLabelDTypeSupported(labels.getDataType()))
            throw std::invalid_argument("Ragged CategoricalAccuracy per-class labels use an unsupported dtype.");
        if (labels.getDimensions() != std::vector<uint64_t>{max_total_values, num_classes})
            throw std::invalid_argument("Ragged CategoricalAccuracy per-class labels must match prediction class width.");
        requireStorageForNumItems(
            labels, "ragged CategoricalAccuracy per-class labels", max_total_values * num_classes);
    }

    CUDA_CHECK(cudaMemsetAsync(correct_count.getMemPtr<float>(), 0, sizeof(float), stream.getStream()));
    CUDA_CHECK(cudaMemsetAsync(token_count.getMemPtr<float>(), 0, sizeof(float), stream.getStream()));
    if (offsets.getDataType() == DataType::UINT32) {
        launchCategoricalForOffset<uint32_t>(predictions,
                                             labels,
                                             offsets,
                                             correct_count,
                                             token_count,
                                             valid_row_count,
                                             num_classes,
                                             label_format,
                                             stream);
    } else {
        launchCategoricalForOffset<uint64_t>(predictions,
                                             labels,
                                             offsets,
                                             correct_count,
                                             token_count,
                                             valid_row_count,
                                             num_classes,
                                             label_format,
                                             stream);
    }
}

}  // namespace ThorImplementation
