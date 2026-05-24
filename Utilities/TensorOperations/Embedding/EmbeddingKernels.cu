#include "Utilities/TensorOperations/Embedding/EmbeddingKernels.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ThorImplementation {
namespace {

using DataType = TensorDescriptor::DataType;

constexpr uint32_t WARP_SIZE_EMBEDDING = 32;
constexpr uint32_t WARPS_PER_BLOCK = 8;
constexpr uint32_t THREADS_PER_BLOCK = WARP_SIZE_EMBEDDING * WARPS_PER_BLOCK;
constexpr uint32_t MAX_GRID_BLOCKS = 131072;
// A lane could hold substantially more in registers, but using a wide per-lane
// struct makes each warp instruction touch sparse memory sectors for wide rows.
// Keep each lane copy to one 16-byte vector so every substep is a contiguous
// 512-byte warp transaction: lane 0 copies bytes 0..15, lane 1 copies 16..31,
// etc. Wider embedding rows iterate over additional coalesced warp substeps.
constexpr uint32_t MAX_REGISTER_BYTES_PER_LANE = 64u * sizeof(uint32_t);
constexpr uint32_t MAX_COALESCED_BYTES_PER_LANE = 16;

static_assert(THREADS_PER_BLOCK == 256);
static_assert(MAX_REGISTER_BYTES_PER_LANE == 256);
static_assert(MAX_COALESCED_BYTES_PER_LANE == 16);

template <typename T, int Elements>
struct alignas((Elements * sizeof(T) >= MAX_COALESCED_BYTES_PER_LANE)
                   ? MAX_COALESCED_BYTES_PER_LANE
                   : Elements * sizeof(T)) LaneVector {
    T value[Elements];
};

template <typename T, int Elements>
__device__ __forceinline__ void copyFixed(const T* __restrict__ src, T* __restrict__ dst) {
    LaneVector<T, Elements> tmp = *reinterpret_cast<const LaneVector<T, Elements>*>(src);
    *reinterpret_cast<LaneVector<T, Elements>*>(dst) = tmp;
}

template <typename T, int Elements>
__device__ __forceinline__ void zeroFixed(T* __restrict__ dst) {
    LaneVector<T, Elements> tmp{};
    *reinterpret_cast<LaneVector<T, Elements>*>(dst) = tmp;
}

template <typename T>
__device__ __forceinline__ void copyTail(const T* __restrict__ src, T* __restrict__ dst, uint32_t elements) {
    if constexpr (sizeof(T) <= 2) {
        if (elements >= 128) {
            copyFixed<T, 128>(src, dst);
            src += 128;
            dst += 128;
            elements -= 128;
        }
    }
    if (elements >= 64) {
        copyFixed<T, 64>(src, dst);
        src += 64;
        dst += 64;
        elements -= 64;
    }
    if (elements >= 32) {
        copyFixed<T, 32>(src, dst);
        src += 32;
        dst += 32;
        elements -= 32;
    }
    if (elements >= 16) {
        copyFixed<T, 16>(src, dst);
        src += 16;
        dst += 16;
        elements -= 16;
    }
    if (elements >= 8) {
        copyFixed<T, 8>(src, dst);
        src += 8;
        dst += 8;
        elements -= 8;
    }
    if (elements >= 4) {
        copyFixed<T, 4>(src, dst);
        src += 4;
        dst += 4;
        elements -= 4;
    }
    if (elements >= 2) {
        copyFixed<T, 2>(src, dst);
        src += 2;
        dst += 2;
        elements -= 2;
    }
    if (elements >= 1) {
        copyFixed<T, 1>(src, dst);
    }
}

template <typename T>
__device__ __forceinline__ void zeroTail(T* __restrict__ dst, uint32_t elements) {
    if constexpr (sizeof(T) <= 2) {
        if (elements >= 128) {
            zeroFixed<T, 128>(dst);
            dst += 128;
            elements -= 128;
        }
    }
    if (elements >= 64) {
        zeroFixed<T, 64>(dst);
        dst += 64;
        elements -= 64;
    }
    if (elements >= 32) {
        zeroFixed<T, 32>(dst);
        dst += 32;
        elements -= 32;
    }
    if (elements >= 16) {
        zeroFixed<T, 16>(dst);
        dst += 16;
        elements -= 16;
    }
    if (elements >= 8) {
        zeroFixed<T, 8>(dst);
        dst += 8;
        elements -= 8;
    }
    if (elements >= 4) {
        zeroFixed<T, 4>(dst);
        dst += 4;
        elements -= 4;
    }
    if (elements >= 2) {
        zeroFixed<T, 2>(dst);
        dst += 2;
        elements -= 2;
    }
    if (elements >= 1) {
        zeroFixed<T, 1>(dst);
    }
}

__device__ __forceinline__ uint64_t warpBroadcastU64(uint64_t value, uint32_t sourceLane) {
    uint32_t lo = static_cast<uint32_t>(value);
    uint32_t hi = static_cast<uint32_t>(value >> 32);
    lo = __shfl_sync(0xffffffffu, lo, sourceLane);
    hi = __shfl_sync(0xffffffffu, hi, sourceLane);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

template <typename IndexT, typename ValueT, uint32_t ElementsPerLane>
__global__ void embeddingForwardWarpKernel(const IndexT* __restrict__ indices,
                                           const ValueT* __restrict__ weights,
                                           ValueT* __restrict__ output,
                                           uint64_t numIndices,
                                           uint64_t vocabularySize,
                                           uint64_t embeddingDim,
                                           uint64_t paddingIndex,
                                           bool hasPaddingIndex) {
    static_assert(!std::is_signed_v<IndexT>, "Embedding indices are unsigned-only.");
    static_assert(ElementsPerLane > 0, "ElementsPerLane must be non-zero.");
    static_assert(ElementsPerLane * sizeof(ValueT) <= MAX_COALESCED_BYTES_PER_LANE,
                  "Embedding lane load must fit in one coalesced 16-byte vector step.");

    const uint32_t lane = threadIdx.x & (WARP_SIZE_EMBEDDING - 1);
    const uint32_t warpInBlock = threadIdx.x >> 5;
    const uint64_t globalWarp = static_cast<uint64_t>(blockIdx.x) * WARPS_PER_BLOCK + warpInBlock;
    const uint64_t totalWarps = static_cast<uint64_t>(gridDim.x) * WARPS_PER_BLOCK;
    constexpr uint64_t kElementsPerWarpIteration = static_cast<uint64_t>(ElementsPerLane) * WARP_SIZE_EMBEDDING;

    for (uint64_t token = globalWarp; token < numIndices; token += totalWarps) {
        uint64_t row = 0;
        if (lane == 0) {
            row = static_cast<uint64_t>(indices[token]);
        }
        row = warpBroadcastU64(row, 0);

        const bool zeroRow = row >= vocabularySize || (hasPaddingIndex && row == paddingIndex);
        ValueT* __restrict__ outBase = output + token * embeddingDim;

        for (uint64_t dimBase = 0; dimBase < embeddingDim; dimBase += kElementsPerWarpIteration) {
            const uint64_t laneBase = dimBase + static_cast<uint64_t>(lane) * ElementsPerLane;
            if (laneBase >= embeddingDim) {
                continue;
            }
            const uint64_t remaining = embeddingDim - laneBase;
            const uint32_t elements = static_cast<uint32_t>(remaining < ElementsPerLane ? remaining : ElementsPerLane);
            ValueT* __restrict__ outPtr = outBase + laneBase;
            if (zeroRow) {
                if (elements == ElementsPerLane) {
                    zeroFixed<ValueT, ElementsPerLane>(outPtr);
                } else {
                    zeroTail<ValueT>(outPtr, elements);
                }
            } else {
                const ValueT* __restrict__ srcPtr = weights + row * embeddingDim + laneBase;
                if (elements == ElementsPerLane) {
                    copyFixed<ValueT, ElementsPerLane>(srcPtr, outPtr);
                } else {
                    copyTail<ValueT>(srcPtr, outPtr, elements);
                }
            }
        }
    }
}

template <typename IndexT>
__global__ void embeddingSparseSgdUpdateFp32Kernel(const IndexT* __restrict__ indices,
                                                  const float* __restrict__ outputGradient,
                                                  float* __restrict__ weights,
                                                  uint64_t numIndices,
                                                  uint64_t vocabularySize,
                                                  uint64_t embeddingDim,
                                                  float step,
                                                  uint64_t paddingIndex,
                                                  bool hasPaddingIndex) {
    static_assert(!std::is_signed_v<IndexT>, "Embedding indices are unsigned-only.");
    const uint64_t totalElements = numIndices * embeddingDim;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

    for (uint64_t linear = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; linear < totalElements;
         linear += stride) {
        const uint64_t indexElement = linear / embeddingDim;
        const uint64_t embeddingElement = linear - indexElement * embeddingDim;

        const uint64_t row = static_cast<uint64_t>(indices[indexElement]);
        if (row >= vocabularySize)
            continue;
        if (hasPaddingIndex && row == paddingIndex)
            continue;

        atomicAdd(weights + row * embeddingDim + embeddingElement, -step * outputGradient[linear]);
    }
}

uint32_t gridForWarps(uint64_t tokens) {
    const uint64_t blocks = (tokens + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    return static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(blocks, 1), MAX_GRID_BLOCKS));
}

uint32_t gridForElements(uint64_t elements) {
    const uint64_t blocks = (elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    return static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(blocks, 1), MAX_GRID_BLOCKS));
}

bool isSupportedEmbeddingIndexDtype(DataType dtype) {
    switch (dtype) {
        case DataType::UINT32:
        case DataType::UINT64:
            return true;
        default:
            return false;
    }
}

bool isSupportedEmbeddingValueDtype(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

std::string dataTypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

void validateDenseContiguous(const Tensor& tensor, const std::string& name) {
    if (tensor.hasCustomStrides() || !tensor.isDenseContiguous()) {
        throw std::invalid_argument("Embedding " + name + " tensor must be dense contiguous in this backend slice.");
    }
}

template <typename IndexT, typename ValueT, uint32_t ElementsPerLane>
void launchEmbeddingForwardWarpTyped(const Tensor& indices,
                                     const Tensor& weights,
                                     Tensor& output,
                                     std::optional<uint64_t> paddingIndex,
                                     Stream stream) {
    const uint64_t numIndices = indices.getTotalNumElements();
    const std::vector<uint64_t> weightDims = weights.getDimensions();
    const uint64_t vocabularySize = weightDims[0];
    const uint64_t embeddingDim = weightDims[1];

    if (numIndices == 0)
        return;

    embeddingForwardWarpKernel<IndexT, ValueT, ElementsPerLane><<<gridForWarps(numIndices), THREADS_PER_BLOCK, 0, stream.getStream()>>>(
        indices.getMemPtr<IndexT>(),
        weights.getMemPtr<ValueT>(),
        output.getMemPtr<ValueT>(),
        numIndices,
        vocabularySize,
        embeddingDim,
        paddingIndex.value_or(0),
        paddingIndex.has_value());
    CUDA_CHECK(cudaPeekAtLastError());
}

uint32_t nextPowerOfTwoAtMost8(uint64_t v) {
    if (v <= 1) return 1;
    if (v <= 2) return 2;
    if (v <= 4) return 4;
    return 8;
}

template <typename IndexT, typename ValueT>
void dispatchEmbeddingForwardTyped(const Tensor& indices,
                                   const Tensor& weights,
                                   Tensor& output,
                                   std::optional<uint64_t> paddingIndex,
                                   Stream stream) {
    const uint64_t embeddingDim = weights.getDimensions()[1];
    constexpr uint32_t maxElementsPerLane = MAX_COALESCED_BYTES_PER_LANE / sizeof(ValueT);
    static_assert(maxElementsPerLane >= 1, "Embedding value dtype is too wide for the coalesced lane copy size.");
    static_assert(maxElementsPerLane <= 8, "Embedding forward intentionally caps lane copies to 16 bytes.");

    const uint64_t firstIterationDims =
        std::min<uint64_t>(embeddingDim, static_cast<uint64_t>(maxElementsPerLane) * WARP_SIZE_EMBEDDING);
    const uint32_t requestedElementsPerLane =
        nextPowerOfTwoAtMost8((firstIterationDims + WARP_SIZE_EMBEDDING - 1) / WARP_SIZE_EMBEDDING);
    const uint32_t elementsPerLane = std::min<uint32_t>(requestedElementsPerLane, maxElementsPerLane);

    switch (elementsPerLane) {
        case 1:
            launchEmbeddingForwardWarpTyped<IndexT, ValueT, 1>(indices, weights, output, paddingIndex, stream);
            return;
        case 2:
            launchEmbeddingForwardWarpTyped<IndexT, ValueT, 2>(indices, weights, output, paddingIndex, stream);
            return;
        case 4:
            launchEmbeddingForwardWarpTyped<IndexT, ValueT, 4>(indices, weights, output, paddingIndex, stream);
            return;
        case 8:
            if constexpr (maxElementsPerLane >= 8) {
                launchEmbeddingForwardWarpTyped<IndexT, ValueT, 8>(indices, weights, output, paddingIndex, stream);
                return;
            } else {
                THOR_UNREACHABLE();
            }
        default:
            THOR_UNREACHABLE();
    }
}

template <typename IndexT>
void dispatchSparseSgdUpdateTyped(const Tensor& indices,
                                  const Tensor& outputGradient,
                                  Tensor& weights,
                                  float step,
                                  std::optional<uint64_t> paddingIndex,
                                  Stream stream) {
    const uint64_t numIndices = indices.getTotalNumElements();
    const std::vector<uint64_t> weightDims = weights.getDimensions();
    const uint64_t vocabularySize = weightDims[0];
    const uint64_t embeddingDim = weightDims[1];

    if (numIndices == 0)
        return;

    const uint64_t totalElements = numIndices * embeddingDim;
    embeddingSparseSgdUpdateFp32Kernel<IndexT><<<gridForElements(totalElements), THREADS_PER_BLOCK, 0, stream.getStream()>>>(
        indices.getMemPtr<IndexT>(),
        outputGradient.getMemPtr<float>(),
        weights.getMemPtr<float>(),
        numIndices,
        vocabularySize,
        embeddingDim,
        step,
        paddingIndex.value_or(0),
        paddingIndex.has_value());
    CUDA_CHECK(cudaPeekAtLastError());
}

template <typename IndexT>
void dispatchEmbeddingForwardValueDtype(const Tensor& indices,
                                        const Tensor& weights,
                                        Tensor& output,
                                        std::optional<uint64_t> paddingIndex,
                                        Stream stream) {
    switch (weights.getDataType()) {
        case DataType::FP16:
            dispatchEmbeddingForwardTyped<IndexT, __half>(indices, weights, output, paddingIndex, stream);
            return;
        case DataType::BF16:
            dispatchEmbeddingForwardTyped<IndexT, __nv_bfloat16>(indices, weights, output, paddingIndex, stream);
            return;
        case DataType::FP32:
            dispatchEmbeddingForwardTyped<IndexT, float>(indices, weights, output, paddingIndex, stream);
            return;
        default:
            throw std::invalid_argument("Embedding forward weights dtype must be fp16, bf16, or fp32. Got " +
                                        dataTypeName(weights.getDataType()) + ".");
    }
}

void validateEmbeddingForwardInputs(const Tensor& indices, const Tensor& weights, const Tensor& output) {
    if (indices.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        weights.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        output.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("Embedding forward tensors must all live on GPU.");
    }
    if (indices.getPlacement() != weights.getPlacement() || indices.getPlacement() != output.getPlacement()) {
        throw std::invalid_argument("Embedding forward tensors must be on the same GPU placement.");
    }
    if (!isSupportedEmbeddingIndexDtype(indices.getDataType())) {
        throw std::invalid_argument("Embedding indices dtype must be uint32 or uint64. Got " + dataTypeName(indices.getDataType()) + ".");
    }
    if (!isSupportedEmbeddingValueDtype(weights.getDataType())) {
        throw std::invalid_argument("Embedding weights dtype must be fp16, bf16, or fp32. Got " + dataTypeName(weights.getDataType()) +
                                    ".");
    }
    if (output.getDataType() != weights.getDataType()) {
        throw std::invalid_argument("Embedding output dtype must match weights dtype in this backend slice.");
    }
    const std::vector<uint64_t> weightDims = weights.getDimensions();
    if (weightDims.size() != 2 || weightDims[0] == 0 || weightDims[1] == 0) {
        throw std::invalid_argument("Embedding weights tensor must have shape [vocabulary_size, embedding_dim].");
    }
    std::vector<uint64_t> expectedOutputDims = indices.getDimensions();
    expectedOutputDims.push_back(weightDims[1]);
    if (output.getDimensions() != expectedOutputDims) {
        throw std::invalid_argument("Embedding output dimensions must equal indices dimensions with embedding_dim appended.");
    }
    validateDenseContiguous(indices, "indices");
    validateDenseContiguous(weights, "weights");
    validateDenseContiguous(output, "output");
}

}  // namespace

void launchEmbeddingForward(const Tensor& indices,
                            const Tensor& weights,
                            Tensor& output,
                            std::optional<uint64_t> paddingIndex,
                            Stream stream) {
    validateEmbeddingForwardInputs(indices, weights, output);

    switch (indices.getDataType()) {
        case DataType::UINT32:
            dispatchEmbeddingForwardValueDtype<uint32_t>(indices, weights, output, paddingIndex, stream);
            return;
        case DataType::UINT64:
            dispatchEmbeddingForwardValueDtype<uint64_t>(indices, weights, output, paddingIndex, stream);
            return;
        default:
            THOR_UNREACHABLE();
    }
}

void launchEmbeddingSparseSgdUpdate(const Tensor& indices,
                                    const Tensor& outputGradient,
                                    Tensor& weights,
                                    float step,
                                    std::optional<uint64_t> paddingIndex,
                                    Stream stream) {
    if (step == 0.0f)
        return;
    if (indices.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        outputGradient.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
        weights.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("Embedding sparse SGD tensors must all live on GPU.");
    }
    if (indices.getPlacement() != outputGradient.getPlacement() || indices.getPlacement() != weights.getPlacement()) {
        throw std::invalid_argument("Embedding sparse SGD tensors must be on the same GPU placement.");
    }
    if (!isSupportedEmbeddingIndexDtype(indices.getDataType())) {
        throw std::invalid_argument("Embedding sparse SGD indices dtype must be uint32 or uint64. Got " +
                                    dataTypeName(indices.getDataType()) + ".");
    }
    if (weights.getDataType() != DataType::FP32 || outputGradient.getDataType() != DataType::FP32) {
        throw std::invalid_argument(
            "Embedding sparse SGD update currently supports fp32 weights and fp32 output gradients. "
            "The forward path supports fp16/bf16/fp32; mixed precision sparse optimizer state is the next backend slice.");
    }
    const std::vector<uint64_t> weightDims = weights.getDimensions();
    if (weightDims.size() != 2 || weightDims[0] == 0 || weightDims[1] == 0) {
        throw std::invalid_argument("Embedding sparse SGD weights tensor must have shape [vocabulary_size, embedding_dim].");
    }
    std::vector<uint64_t> expectedGradientDims = indices.getDimensions();
    expectedGradientDims.push_back(weightDims[1]);
    if (outputGradient.getDimensions() != expectedGradientDims) {
        throw std::invalid_argument("Embedding sparse SGD output gradient dimensions must equal indices dimensions with embedding_dim appended.");
    }
    validateDenseContiguous(indices, "indices");
    validateDenseContiguous(outputGradient, "output_gradient");
    validateDenseContiguous(weights, "weights");

    switch (indices.getDataType()) {
        case DataType::UINT32:
            dispatchSparseSgdUpdateTyped<uint32_t>(indices, outputGradient, weights, step, paddingIndex, stream);
            return;
        case DataType::UINT64:
            dispatchSparseSgdUpdateTyped<uint64_t>(indices, outputGradient, weights, step, paddingIndex, stream);
            return;
        default:
            THOR_UNREACHABLE();
    }
}

}  // namespace ThorImplementation
