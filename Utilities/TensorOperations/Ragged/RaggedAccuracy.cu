#include "Utilities/TensorOperations/Ragged/RaggedAccuracy.h"

#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"

#include <cub/block/block_reduce.cuh>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;
using CubReductionInternal::ToFp32;

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kTargetItemsPerThread = 8;
constexpr uint32_t kMaxPartialBlocks = 4096;

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

uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char* what) {
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs)
        throw std::overflow_error(what);
    return lhs * rhs;
}

uint32_t partialBlockCount(uint64_t work_items) {
    if (work_items == 0) return 1;
    const uint64_t target_items_per_block =
        static_cast<uint64_t>(kBlockSize) * kTargetItemsPerThread;
    const uint64_t requested_blocks = 1 + (work_items - 1) / target_items_per_block;
    return static_cast<uint32_t>(std::min<uint64_t>(requested_blocks, kMaxPartialBlocks));
}

uint32_t categoricalLanesPerToken(uint64_t num_classes) {
    if (num_classes <= 2) return 2;
    if (num_classes <= 4) return 4;
    if (num_classes <= 8) return 8;
    if (num_classes <= 16) return 16;
    return 32;
}

uint32_t categoricalPartialBlockCount(uint64_t active_value_count, uint64_t num_classes) {
    if (active_value_count == 0) return 1;
    const uint64_t scalar_work = checkedMultiply(
        active_value_count, num_classes, "Ragged CategoricalAccuracy scalar work overflows uint64_t.");
    const uint64_t lanes_per_token = categoricalLanesPerToken(num_classes);
    const uint64_t groups_per_block = kBlockSize / lanes_per_token;
    const uint64_t max_useful_blocks = 1 + (active_value_count - 1) / groups_per_block;
    return static_cast<uint32_t>(
        std::min<uint64_t>(partialBlockCount(scalar_work), max_useful_blocks));
}

__device__ __forceinline__ void storeFinalStatistics(uint64_t correct,
                                                      uint64_t tokens,
                                                      float* correct_count,
                                                      float* token_count) {
    *correct_count = static_cast<float>(correct);
    *token_count = static_cast<float>(tokens);
}

template <typename PredictionT, typename LabelT, typename IndexT>
__global__ void binaryAccuracyPartialsKernel(const PredictionT* predictions,
                                             const LabelT* labels,
                                             uint64_t* partial_correct_counts,
                                             IndexT active_value_count,
                                             float* correct_count,
                                             float* token_count) {
    using BlockReduce = cub::BlockReduce<IndexT, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage reduction_storage;

    IndexT local_correct = 0;
    const IndexT first = static_cast<IndexT>(blockIdx.x * kBlockSize + threadIdx.x);
    const IndexT stride = static_cast<IndexT>(gridDim.x * kBlockSize);
    IndexT token = first;
    while (token < active_value_count) {
        const float prediction = ToFp32<PredictionT>{}(predictions[token]);
        const float label = ToFp32<LabelT>{}(labels[token]);
        const float predicted_label = prediction >= 0.5f ? 1.0f : 0.0f;
        local_correct += predicted_label == label ? 1 : 0;
        if (stride >= active_value_count - token) break;
        token += stride;
    }

    const IndexT block_correct = BlockReduce(reduction_storage).Sum(local_correct);
    if (threadIdx.x == 0) {
        if (gridDim.x == 1) {
            storeFinalStatistics(static_cast<uint64_t>(block_correct),
                                 static_cast<uint64_t>(active_value_count),
                                 correct_count,
                                 token_count);
        } else {
            partial_correct_counts[blockIdx.x] = static_cast<uint64_t>(block_correct);
        }
    }
}

template <typename IndexT>
struct ArgMaxCandidate {
    float value;
    IndexT index;
};

template <typename IndexT>
__device__ __forceinline__ constexpr IndexT invalidArgMaxIndex() {
    return ~static_cast<IndexT>(0);
}

template <typename IndexT>
__device__ __forceinline__ ArgMaxCandidate<IndexT> betterArgMaxCandidate(ArgMaxCandidate<IndexT> lhs,
                                                                        ArgMaxCandidate<IndexT> rhs) {
    if (rhs.index == invalidArgMaxIndex<IndexT>()) return lhs;
    if (lhs.index == invalidArgMaxIndex<IndexT>()) return rhs;
    if (rhs.value > lhs.value || (rhs.value == lhs.value && rhs.index < lhs.index)) return rhs;
    return lhs;
}

template <uint32_t LanesPerToken>
__device__ __forceinline__ uint32_t groupMask() {
    static_assert(LanesPerToken == 2 || LanesPerToken == 4 || LanesPerToken == 8 ||
                  LanesPerToken == 16 || LanesPerToken == 32);
    if constexpr (LanesPerToken == 32) {
        return 0xffffffffu;
    } else {
        constexpr uint32_t base_mask = (1u << LanesPerToken) - 1u;
        const uint32_t group_in_warp = (threadIdx.x & 31u) / LanesPerToken;
        return base_mask << (group_in_warp * LanesPerToken);
    }
}

template <typename IndexT>
__device__ __forceinline__ IndexT shuffleDownIndex(uint32_t mask,
                                                   IndexT value,
                                                   uint32_t delta,
                                                   uint32_t width) {
    if constexpr (sizeof(IndexT) == sizeof(uint32_t)) {
        return static_cast<IndexT>(
            __shfl_down_sync(mask, static_cast<uint32_t>(value), delta, width));
    } else {
        return static_cast<IndexT>(
            __shfl_down_sync(mask, static_cast<unsigned long long>(value), delta, width));
    }
}

template <typename IndexT>
__device__ __forceinline__ IndexT shuffleIndex(uint32_t mask,
                                               IndexT value,
                                               uint32_t source_lane,
                                               uint32_t width) {
    if constexpr (sizeof(IndexT) == sizeof(uint32_t)) {
        return static_cast<IndexT>(
            __shfl_sync(mask, static_cast<uint32_t>(value), source_lane, width));
    } else {
        return static_cast<IndexT>(
            __shfl_sync(mask, static_cast<unsigned long long>(value), source_lane, width));
    }
}

template <uint32_t LanesPerToken, typename T, typename IndexT>
__device__ __forceinline__ IndexT groupArgmax(const T* values,
                                              IndexT token,
                                              IndexT num_classes) {
    const uint32_t lane = threadIdx.x & (LanesPerToken - 1u);
    const uint32_t mask = groupMask<LanesPerToken>();
    const IndexT base = token * num_classes;

    float first_value = lane == 0 ? ToFp32<T>{}(values[base]) : 0.0f;
    first_value = __shfl_sync(mask, first_value, 0, LanesPerToken);
    // Ordinary serial argmax initializes from class 0. If that value is NaN,
    // every subsequent `value > best_value` comparison is false, so class 0
    // wins regardless of later values. Preserve that behavior exactly.
    if (isnan(first_value)) return static_cast<IndexT>(0);

    ArgMaxCandidate<IndexT> best{0.0f, invalidArgMaxIndex<IndexT>()};
    IndexT c = static_cast<IndexT>(lane);
    if (lane == 0) {
        best = {first_value, static_cast<IndexT>(0)};
        c += static_cast<IndexT>(LanesPerToken);
    }
    while (c < num_classes) {
        const float value = ToFp32<T>{}(values[base + c]);
        if (!isnan(value))
            best = betterArgMaxCandidate(best, ArgMaxCandidate<IndexT>{value, c});
        const IndexT remaining = num_classes - c;
        if (remaining <= static_cast<IndexT>(LanesPerToken)) break;
        c += static_cast<IndexT>(LanesPerToken);
    }

    for (uint32_t delta = LanesPerToken / 2; delta != 0; delta >>= 1) {
        ArgMaxCandidate<IndexT> other{
            __shfl_down_sync(mask, best.value, delta, LanesPerToken),
            shuffleDownIndex(mask, best.index, delta, LanesPerToken)};
        best = betterArgMaxCandidate(best, other);
    }
    return shuffleIndex(mask, best.index, 0, LanesPerToken);
}

template <uint32_t LanesPerToken,
          typename PredictionT,
          typename LabelT,
          typename IndexT,
          bool ClassIndexLabels>
__global__ void categoricalAccuracyPartialsKernel(const PredictionT* predictions,
                                                  const LabelT* labels,
                                                  uint64_t* partial_correct_counts,
                                                  IndexT active_value_count,
                                                  IndexT num_classes,
                                                  float* correct_count,
                                                  float* token_count) {
    using BlockReduce = cub::BlockReduce<IndexT, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage reduction_storage;

    constexpr uint32_t groups_per_block = kBlockSize / LanesPerToken;
    const uint32_t lane = threadIdx.x & (LanesPerToken - 1u);
    const uint32_t group_in_block = threadIdx.x / LanesPerToken;
    const IndexT first_token = static_cast<IndexT>(blockIdx.x * groups_per_block + group_in_block);
    const IndexT token_stride = static_cast<IndexT>(gridDim.x * groups_per_block);

    IndexT local_correct = 0;
    IndexT token = first_token;
    while (token < active_value_count) {
        const IndexT predicted_class = groupArgmax<LanesPerToken>(predictions, token, num_classes);
        bool correct = false;
        if constexpr (ClassIndexLabels) {
            if (lane == 0) {
                const int64_t true_class = static_cast<int64_t>(labels[token]);
                correct = true_class >= 0 && static_cast<uint64_t>(true_class) ==
                                                  static_cast<uint64_t>(predicted_class);
            }
        } else {
            const IndexT true_class = groupArgmax<LanesPerToken>(labels, token, num_classes);
            if (lane == 0) correct = predicted_class == true_class;
        }
        if (lane == 0 && correct) ++local_correct;

        if (token_stride >= active_value_count - token) break;
        token += token_stride;
    }

    const IndexT block_correct = BlockReduce(reduction_storage).Sum(local_correct);
    if (threadIdx.x == 0) {
        if (gridDim.x == 1) {
            storeFinalStatistics(static_cast<uint64_t>(block_correct),
                                 static_cast<uint64_t>(active_value_count),
                                 correct_count,
                                 token_count);
        } else {
            partial_correct_counts[blockIdx.x] = static_cast<uint64_t>(block_correct);
        }
    }
}

__global__ void finalizeAccuracyStatisticsKernel(const uint64_t* partial_correct_counts,
                                                 uint32_t partial_count,
                                                 uint64_t active_value_count,
                                                 float* correct_count,
                                                 float* token_count) {
    using BlockReduce = cub::BlockReduce<uint64_t, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage reduction_storage;

    uint64_t local_correct = 0;
    for (uint32_t i = threadIdx.x; i < partial_count; i += kBlockSize)
        local_correct += partial_correct_counts[i];

    const uint64_t total_correct = BlockReduce(reduction_storage).Sum(local_correct);
    if (threadIdx.x == 0)
        storeFinalStatistics(total_correct, active_value_count, correct_count, token_count);
}

void validateCommon(const Tensor& predictions,
                    const Tensor& labels,
                    const Tensor& partial_correct_counts,
                    const Tensor& correct_count,
                    const Tensor& token_count,
                    uint64_t active_value_count,
                    uint64_t max_total_values,
                    uint64_t workspace_num_classes) {
    requireDenseContiguousGpuTensor(predictions, "ragged accuracy predictions");
    requireDenseContiguousGpuTensor(labels, "ragged accuracy labels");
    requireDenseContiguousGpuTensor(partial_correct_counts, "ragged accuracy partial correct counts");
    requireDenseContiguousGpuTensor(correct_count, "ragged accuracy correct count");
    requireDenseContiguousGpuTensor(token_count, "ragged accuracy token count");
    requireSameGpuPlacement(predictions, labels, "ragged accuracy predictions", "ragged accuracy labels");
    requireSameGpuPlacement(predictions,
                            partial_correct_counts,
                            "ragged accuracy predictions",
                            "ragged accuracy partial correct counts");
    requireSameGpuPlacement(predictions, correct_count, "ragged accuracy predictions", "ragged accuracy correct count");
    requireSameGpuPlacement(predictions, token_count, "ragged accuracy predictions", "ragged accuracy token count");
    if (!isPredictionDTypeSupported(predictions.getDataType()))
        throw std::invalid_argument("Ragged accuracy predictions must be FP16 or FP32.");
    if (correct_count.getDataType() != DataType::FP32 || correct_count.getTotalNumElements() != 1 ||
        token_count.getDataType() != DataType::FP32 || token_count.getTotalNumElements() != 1) {
        throw std::invalid_argument("Ragged accuracy sufficient statistics must be FP32 scalars.");
    }
    if (max_total_values == 0)
        throw std::invalid_argument("Ragged accuracy max_total_values must be non-zero.");
    if (active_value_count > max_total_values)
        throw std::invalid_argument("Ragged accuracy active_value_count exceeds packed capacity.");
    const std::vector<uint64_t>& prediction_dims = predictions.getDimensions();
    const std::vector<uint64_t>& label_dims = labels.getDimensions();
    if (prediction_dims.empty() || prediction_dims.front() != max_total_values ||
        label_dims.empty() || label_dims.front() != max_total_values) {
        throw std::invalid_argument("Ragged accuracy packed tensors must use max_total_values as the leading dimension.");
    }
    const TensorDescriptor expected_workspace =
        raggedAccuracyStatisticsWorkspaceDescriptor(max_total_values, workspace_num_classes);
    if (partial_correct_counts.getDescriptor() != expected_workspace)
        throw std::invalid_argument("Ragged accuracy partial-correct-count workspace has the wrong descriptor.");
}

template <typename PredictionT, typename LabelT, typename IndexT>
void launchBinaryTyped(const Tensor& predictions,
                       const Tensor& labels,
                       Tensor& partial_correct_counts,
                       IndexT active_value_count,
                       uint32_t partial_count,
                       Tensor& correct_count,
                       Tensor& token_count,
                       Stream& stream) {
    binaryAccuracyPartialsKernel<PredictionT, LabelT, IndexT><<<partial_count, kBlockSize, 0, stream.getStream()>>>(
        predictions.getMemPtr<PredictionT>(),
        labels.getMemPtr<LabelT>(),
        partial_correct_counts.getMemPtr<uint64_t>(),
        active_value_count,
        correct_count.getMemPtr<float>(),
        token_count.getMemPtr<float>());
    CUDA_CHECK(cudaPeekAtLastError());
}

template <typename IndexT>
void launchBinaryForIndex(const Tensor& predictions,
                          const Tensor& labels,
                          Tensor& partial_correct_counts,
                          IndexT active_value_count,
                          uint32_t partial_count,
                          Tensor& correct_count,
                          Tensor& token_count,
                          Stream& stream) {
    auto launch_prediction = [&]<typename PredictionT>() {
        auto launch_label = [&]<typename LabelT>() {
            launchBinaryTyped<PredictionT, LabelT, IndexT>(predictions,
                                                           labels,
                                                           partial_correct_counts,
                                                           active_value_count,
                                                           partial_count,
                                                           correct_count,
                                                           token_count,
                                                           stream);
        };
        dispatchBinaryOrPerClassLabelDType(labels.getDataType(), launch_label);
    };
    dispatchPredictionDType(predictions.getDataType(), launch_prediction);
}

template <uint32_t LanesPerToken,
          typename PredictionT,
          typename LabelT,
          typename IndexT,
          bool ClassIndexLabels>
void launchCategoricalTyped(const Tensor& predictions,
                            const Tensor& labels,
                            Tensor& partial_correct_counts,
                            IndexT active_value_count,
                            IndexT num_classes,
                            uint32_t partial_count,
                            Tensor& correct_count,
                            Tensor& token_count,
                            Stream& stream) {
    categoricalAccuracyPartialsKernel<LanesPerToken, PredictionT, LabelT, IndexT, ClassIndexLabels>
        <<<partial_count, kBlockSize, 0, stream.getStream()>>>(
            predictions.getMemPtr<PredictionT>(),
            labels.getMemPtr<LabelT>(),
            partial_correct_counts.getMemPtr<uint64_t>(),
            active_value_count,
            num_classes,
            correct_count.getMemPtr<float>(),
            token_count.getMemPtr<float>());
    CUDA_CHECK(cudaPeekAtLastError());
}

template <typename PredictionT, typename LabelT, typename IndexT, bool ClassIndexLabels>
void dispatchCategoricalLaneCount(const Tensor& predictions,
                                  const Tensor& labels,
                                  Tensor& partial_correct_counts,
                                  IndexT active_value_count,
                                  IndexT num_classes,
                                  uint32_t partial_count,
                                  Tensor& correct_count,
                                  Tensor& token_count,
                                  Stream& stream) {
    if (num_classes <= static_cast<IndexT>(2)) {
        launchCategoricalTyped<2, PredictionT, LabelT, IndexT, ClassIndexLabels>(
            predictions, labels, partial_correct_counts, active_value_count, num_classes, partial_count,
            correct_count, token_count, stream);
    } else if (num_classes <= static_cast<IndexT>(4)) {
        launchCategoricalTyped<4, PredictionT, LabelT, IndexT, ClassIndexLabels>(
            predictions, labels, partial_correct_counts, active_value_count, num_classes, partial_count,
            correct_count, token_count, stream);
    } else if (num_classes <= static_cast<IndexT>(8)) {
        launchCategoricalTyped<8, PredictionT, LabelT, IndexT, ClassIndexLabels>(
            predictions, labels, partial_correct_counts, active_value_count, num_classes, partial_count,
            correct_count, token_count, stream);
    } else if (num_classes <= static_cast<IndexT>(16)) {
        launchCategoricalTyped<16, PredictionT, LabelT, IndexT, ClassIndexLabels>(
            predictions, labels, partial_correct_counts, active_value_count, num_classes, partial_count,
            correct_count, token_count, stream);
    } else {
        launchCategoricalTyped<32, PredictionT, LabelT, IndexT, ClassIndexLabels>(
            predictions, labels, partial_correct_counts, active_value_count, num_classes, partial_count,
            correct_count, token_count, stream);
    }
}

template <typename IndexT>
void launchCategoricalForIndex(const Tensor& predictions,
                               const Tensor& labels,
                               Tensor& partial_correct_counts,
                               IndexT active_value_count,
                               IndexT num_classes,
                               uint32_t partial_count,
                               Tensor& correct_count,
                               Tensor& token_count,
                               RaggedCategoricalLabelFormat label_format,
                               Stream& stream) {
    auto launch_prediction = [&]<typename PredictionT>() {
        if (label_format == RaggedCategoricalLabelFormat::CLASS_INDEX) {
            auto launch_label = [&]<typename LabelT>() {
                dispatchCategoricalLaneCount<PredictionT, LabelT, IndexT, true>(
                    predictions, labels, partial_correct_counts, active_value_count, num_classes,
                    partial_count, correct_count, token_count, stream);
            };
            dispatchClassIndexLabelDType(labels.getDataType(), launch_label);
        } else {
            auto launch_label = [&]<typename LabelT>() {
                dispatchCategoricalLaneCount<PredictionT, LabelT, IndexT, false>(
                    predictions, labels, partial_correct_counts, active_value_count, num_classes,
                    partial_count, correct_count, token_count, stream);
            };
            dispatchBinaryOrPerClassLabelDType(labels.getDataType(), launch_label);
        }
    };
    dispatchPredictionDType(predictions.getDataType(), launch_prediction);
}

void finalizeIfNeeded(const Tensor& partial_correct_counts,
                      uint32_t partial_count,
                      uint64_t active_value_count,
                      Tensor& correct_count,
                      Tensor& token_count,
                      Stream& stream) {
    if (partial_count <= 1) return;
    finalizeAccuracyStatisticsKernel<<<1, kBlockSize, 0, stream.getStream()>>>(
        partial_correct_counts.getMemPtr<uint64_t>(),
        partial_count,
        active_value_count,
        correct_count.getMemPtr<float>(),
        token_count.getMemPtr<float>());
    CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace

TensorDescriptor raggedAccuracyStatisticsWorkspaceDescriptor(uint64_t max_total_values,
                                                              uint64_t num_classes) {
    if (max_total_values == 0)
        throw std::invalid_argument("Ragged accuracy max_total_values must be non-zero.");
    if (num_classes == 0)
        throw std::invalid_argument("Ragged accuracy workspace num_classes must be non-zero.");
    const uint32_t max_partials = num_classes == 1
                                      ? partialBlockCount(max_total_values)
                                      : categoricalPartialBlockCount(max_total_values, num_classes);
    return TensorDescriptor(DataType::UINT64, {max_partials});
}

void raggedBinaryAccuracyStatistics(const Tensor& predictions,
                                    const Tensor& labels,
                                    Tensor& partial_correct_counts,
                                    Tensor& correct_count,
                                    Tensor& token_count,
                                    uint64_t active_value_count,
                                    uint64_t max_total_values,
                                    Stream& stream) {
    validateCommon(predictions,
                   labels,
                   partial_correct_counts,
                   correct_count,
                   token_count,
                   active_value_count,
                   max_total_values,
                   1);
    if (!isBinaryLabelDTypeSupported(labels.getDataType()))
        throw std::invalid_argument("Ragged BinaryAccuracy labels use an unsupported dtype.");
    if (predictions.getDimensions() != std::vector<uint64_t>{max_total_values, 1} ||
        labels.getDimensions() != std::vector<uint64_t>{max_total_values, 1}) {
        throw std::invalid_argument("Ragged BinaryAccuracy predictions and labels must contain one scalar per packed token.");
    }
    requireStorageForNumItems(predictions, "ragged BinaryAccuracy predictions", max_total_values);
    requireStorageForNumItems(labels, "ragged BinaryAccuracy labels", max_total_values);

    if (active_value_count == 0) {
        CUDA_CHECK(cudaMemsetAsync(correct_count.getMemPtr<float>(), 0, sizeof(float), stream.getStream()));
        CUDA_CHECK(cudaMemsetAsync(token_count.getMemPtr<float>(), 0, sizeof(float), stream.getStream()));
        return;
    }

    const uint32_t partial_count = partialBlockCount(active_value_count);
    if (partial_count > partial_correct_counts.getDimensions().front())
        throw std::logic_error("Ragged BinaryAccuracy runtime partial count exceeds workspace capacity.");
    if (active_value_count <= std::numeric_limits<uint32_t>::max()) {
        launchBinaryForIndex<uint32_t>(predictions,
                                       labels,
                                       partial_correct_counts,
                                       static_cast<uint32_t>(active_value_count),
                                       partial_count,
                                       correct_count,
                                       token_count,
                                       stream);
    } else {
        launchBinaryForIndex<uint64_t>(predictions,
                                       labels,
                                       partial_correct_counts,
                                       active_value_count,
                                       partial_count,
                                       correct_count,
                                       token_count,
                                       stream);
    }
    finalizeIfNeeded(partial_correct_counts, partial_count, active_value_count, correct_count, token_count, stream);
}

void raggedCategoricalAccuracyStatistics(const Tensor& predictions,
                                         const Tensor& labels,
                                         Tensor& partial_correct_counts,
                                         Tensor& correct_count,
                                         Tensor& token_count,
                                         uint64_t active_value_count,
                                         uint64_t max_total_values,
                                         uint64_t num_classes,
                                         RaggedCategoricalLabelFormat label_format,
                                         Stream& stream) {
    validateCommon(predictions,
                   labels,
                   partial_correct_counts,
                   correct_count,
                   token_count,
                   active_value_count,
                   max_total_values,
                   num_classes);
    if (num_classes < 2)
        throw std::invalid_argument("Ragged CategoricalAccuracy requires at least two classes.");
    if (predictions.getDimensions() != std::vector<uint64_t>{max_total_values, num_classes})
        throw std::invalid_argument("Ragged CategoricalAccuracy predictions must have trailing class width num_classes.");
    const uint64_t max_scalar_count = checkedMultiply(
        max_total_values, num_classes, "Ragged CategoricalAccuracy packed prediction element count overflows uint64_t.");
    requireStorageForNumItems(predictions, "ragged CategoricalAccuracy predictions", max_scalar_count);

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
        requireStorageForNumItems(labels, "ragged CategoricalAccuracy per-class labels", max_scalar_count);
    }

    if (active_value_count == 0) {
        CUDA_CHECK(cudaMemsetAsync(correct_count.getMemPtr<float>(), 0, sizeof(float), stream.getStream()));
        CUDA_CHECK(cudaMemsetAsync(token_count.getMemPtr<float>(), 0, sizeof(float), stream.getStream()));
        return;
    }

    const uint64_t active_scalar_count = checkedMultiply(
        active_value_count, num_classes, "Ragged CategoricalAccuracy active scalar count overflows uint64_t.");
    const uint32_t partial_count = categoricalPartialBlockCount(active_value_count, num_classes);
    if (partial_count > partial_correct_counts.getDimensions().front())
        throw std::logic_error("Ragged CategoricalAccuracy runtime partial count exceeds workspace capacity.");

    if (active_scalar_count <= std::numeric_limits<uint32_t>::max()) {
        launchCategoricalForIndex<uint32_t>(predictions,
                                            labels,
                                            partial_correct_counts,
                                            static_cast<uint32_t>(active_value_count),
                                            static_cast<uint32_t>(num_classes),
                                            partial_count,
                                            correct_count,
                                            token_count,
                                            label_format,
                                            stream);
    } else {
        launchCategoricalForIndex<uint64_t>(predictions,
                                            labels,
                                            partial_correct_counts,
                                            active_value_count,
                                            num_classes,
                                            partial_count,
                                            correct_count,
                                            token_count,
                                            label_format,
                                            stream);
    }
    finalizeIfNeeded(partial_correct_counts, partial_count, active_value_count, correct_count, token_count, stream);
}

}  // namespace ThorImplementation
