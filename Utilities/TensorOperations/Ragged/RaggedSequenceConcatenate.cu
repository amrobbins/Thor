#include "Utilities/TensorOperations/Ragged/RaggedSequenceConcatenate.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxPairsPerBlock = 8;
constexpr uint32_t kMaxPortableBlocks = 65535;
constexpr uint64_t kMaxUint32 = 0xFFFFFFFFULL;

struct PairCopyMetadata {
    const unsigned char *source;
    unsigned char *destination;
    uint64_t byteCount;
    uint32_t copyWidthBytes;
};

template <typename OffsetT, typename RowIndexT>
__device__ __forceinline__ OffsetT offsetAt(void *const *offsets, uint32_t input, RowIndexT row) {
    return reinterpret_cast<const OffsetT *>(offsets[input])[row];
}

/**
 * Resolve one input row's source range and destination placement in a single
 * pass over the input partitions.
 *
 * The concatenated row begins after every value in preceding rows. For inputs
 * before `input`, the values from the current row also precede this input, so
 * their row-end offsets contribute directly. For the current and later inputs,
 * only their row-start offsets contribute:
 *
 *   outputBegin = sum(i < input, offsets[i][row + 1])
 *               + sum(i >= input, offsets[i][row])
 *
 * This is algebraically identical to the previous
 *   sum(all row starts) + sum(prior row lengths)
 * formulation, but avoids rereading prior inputs' row starts and ends. The
 * API validates that the concatenated max_total_values fits the selected
 * offsets dtype, so carrying these quantities as OffsetT does not narrow the
 * valid production range.
 */
template <typename OffsetT, typename RowIndexT>
__device__ __forceinline__ void rowPlacement(void *const *inputOffsets,
                                             uint32_t numInputs,
                                             uint32_t input,
                                             RowIndexT row,
                                             OffsetT &sourceBegin,
                                             OffsetT &rowLength,
                                             OffsetT &outputBegin) {
    sourceBegin = 0;
    OffsetT sourceEnd = 0;
    outputBegin = 0;

    for (uint32_t i = 0; i < numInputs; ++i) {
        if (i < input) {
            outputBegin += offsetAt<OffsetT>(inputOffsets, i, row + 1);
            continue;
        }

        const OffsetT begin = offsetAt<OffsetT>(inputOffsets, i, row);
        outputBegin += begin;
        if (i == input) {
            sourceBegin = begin;
            sourceEnd = offsetAt<OffsetT>(inputOffsets, i, row + 1);
        }
    }

    rowLength = sourceEnd - sourceBegin;
}

__device__ __forceinline__ uint32_t widestAlignedCopyWidth(const unsigned char *source,
                                                            const unsigned char *destination,
                                                            uint64_t byteCount) {
    const uintptr_t combined = reinterpret_cast<uintptr_t>(source) |
                               reinterpret_cast<uintptr_t>(destination) |
                               static_cast<uintptr_t>(byteCount);
    if ((combined & 15U) == 0) return 16;
    if ((combined & 7U) == 0) return 8;
    if ((combined & 3U) == 0) return 4;
    if ((combined & 1U) == 0) return 2;
    return 1;
}

template <typename CopyT, typename ItemIndexT, uint32_t LanesPerPair>
__device__ __forceinline__ void copyAligned(const unsigned char *source,
                                            unsigned char *destination,
                                            ItemIndexT byteCount,
                                            uint32_t lane) {
    const auto *typedSource = reinterpret_cast<const CopyT *>(source);
    auto *typedDestination = reinterpret_cast<CopyT *>(destination);
    const ItemIndexT items = byteCount / sizeof(CopyT);
    for (ItemIndexT item = static_cast<ItemIndexT>(lane); item < items; item += LanesPerPair) {
        typedDestination[item] = typedSource[item];
    }
}

template <typename ItemIndexT, uint32_t LanesPerPair>
__device__ __forceinline__ void copyPairIndexed(const PairCopyMetadata &metadata, uint32_t lane) {
    const ItemIndexT byteCount = static_cast<ItemIndexT>(metadata.byteCount);
    switch (metadata.copyWidthBytes) {
        case 16:
            copyAligned<uint4, ItemIndexT, LanesPerPair>(metadata.source, metadata.destination, byteCount, lane);
            break;
        case 8:
            copyAligned<uint64_t, ItemIndexT, LanesPerPair>(metadata.source, metadata.destination, byteCount, lane);
            break;
        case 4:
            copyAligned<uint32_t, ItemIndexT, LanesPerPair>(metadata.source, metadata.destination, byteCount, lane);
            break;
        case 2:
            copyAligned<uint16_t, ItemIndexT, LanesPerPair>(metadata.source, metadata.destination, byteCount, lane);
            break;
        default:
            copyAligned<uint8_t, ItemIndexT, LanesPerPair>(metadata.source, metadata.destination, byteCount, lane);
            break;
    }
}

template <uint32_t LanesPerPair>
__device__ __forceinline__ void copyPair(const PairCopyMetadata &metadata, uint32_t lane) {
    // Keep the ordinary copy loop entirely 32-bit. Leave enough headroom for
    // item += LanesPerPair so an extreme near-4-GiB byte copy cannot wrap the
    // loop counter; only genuinely huge rows take the 64-bit fallback.
    if (metadata.byteCount <= kMaxUint32 - LanesPerPair) {
        copyPairIndexed<uint32_t, LanesPerPair>(metadata, lane);
    } else {
        copyPairIndexed<uint64_t, LanesPerPair>(metadata, lane);
    }
}

template <typename OffsetT, typename PairIndexT, uint32_t PairsPerBlock>
__global__ void concatenateValuesKernel(unsigned char *outputValues,
                                        unsigned char *const *inputValues,
                                        void *const *inputOffsets,
                                        uint32_t numInputs,
                                        uint64_t bytesPerValue,
                                        PairIndexT batchSize) {
    static_assert(PairsPerBlock >= 1 && PairsPerBlock <= kMaxPairsPerBlock,
                  "RaggedSequenceConcatenate pairs per CTA are out of range.");
    static_assert(kThreads % PairsPerBlock == 0,
                  "RaggedSequenceConcatenate pair groups must tile a CTA.");
    constexpr uint32_t kLanesPerPair = kThreads / PairsPerBlock;

    __shared__ PairCopyMetadata pairMetadata[PairsPerBlock];

    const uint32_t pairSlot = threadIdx.x / kLanesPerPair;
    const uint32_t lane = threadIdx.x - pairSlot * kLanesPerPair;
    const PairIndexT totalPairs = batchSize * static_cast<PairIndexT>(numInputs);
    const uint32_t pairStride = gridDim.x * PairsPerBlock;

    PairIndexT pairBase = static_cast<PairIndexT>(blockIdx.x) * PairsPerBlock;
    while (pairBase < totalPairs) {
        if (lane == 0) {
            PairCopyMetadata metadata{nullptr, nullptr, 0, 1};
            if (pairSlot < totalPairs - pairBase) {
                const PairIndexT pair = pairBase + static_cast<PairIndexT>(pairSlot);
                const PairIndexT row = pair / static_cast<PairIndexT>(numInputs);
                const uint32_t input = static_cast<uint32_t>(pair - row * static_cast<PairIndexT>(numInputs));
                OffsetT sourceBegin = 0;
                OffsetT rowLength = 0;
                OffsetT outputBegin = 0;
                rowPlacement<OffsetT>(inputOffsets, numInputs, input, row, sourceBegin, rowLength, outputBegin);

                if (rowLength != 0) {
                    metadata.source = inputValues[input] + static_cast<uint64_t>(sourceBegin) * bytesPerValue;
                    metadata.destination = outputValues + static_cast<uint64_t>(outputBegin) * bytesPerValue;
                    metadata.byteCount = static_cast<uint64_t>(rowLength) * bytesPerValue;
                    metadata.copyWidthBytes =
                        widestAlignedCopyWidth(metadata.source, metadata.destination, metadata.byteCount);
                }
            }
            pairMetadata[pairSlot] = metadata;
        }

        // One producer lane resolves each pair. Shared memory is the broadcast.
        // The 8-pair specialization maps one warp to each shared slot, so a
        // warp barrier is sufficient there; wider pair groups span warps and
        // require a CTA barrier.
        if constexpr (kLanesPerPair == 32) {
            __syncwarp();
        } else {
            __syncthreads();
        }
        const PairCopyMetadata metadata = pairMetadata[pairSlot];

        // Protect shared-slot reuse on a grid-stride iteration. Keep this before
        // the actual copy: once all consumers have loaded metadata into registers,
        // the next iteration may reuse shared memory while this copy is in flight.
        const bool hasNextPairBase = static_cast<PairIndexT>(pairStride) < totalPairs - pairBase;
        if (hasNextPairBase) {
            if constexpr (kLanesPerPair == 32) {
                __syncwarp();
            } else {
                __syncthreads();
            }
        }

        if (metadata.byteCount != 0) copyPair<kLanesPerPair>(metadata, lane);
        if (!hasNextPairBase) break;
        pairBase += static_cast<PairIndexT>(pairStride);
    }
}

template <typename OffsetT, typename PairIndexT, uint32_t PairsPerBlock>
__global__ void splitGradientKernel(unsigned char *const *inputGradients,
                                    const unsigned char *outputGradient,
                                    void *const *inputOffsets,
                                    uint32_t numInputs,
                                    uint64_t bytesPerValue,
                                    PairIndexT batchSize) {
    static_assert(PairsPerBlock >= 1 && PairsPerBlock <= kMaxPairsPerBlock,
                  "RaggedSequenceConcatenate pairs per CTA are out of range.");
    static_assert(kThreads % PairsPerBlock == 0,
                  "RaggedSequenceConcatenate pair groups must tile a CTA.");
    constexpr uint32_t kLanesPerPair = kThreads / PairsPerBlock;

    __shared__ PairCopyMetadata pairMetadata[PairsPerBlock];

    const uint32_t pairSlot = threadIdx.x / kLanesPerPair;
    const uint32_t lane = threadIdx.x - pairSlot * kLanesPerPair;
    const PairIndexT totalPairs = batchSize * static_cast<PairIndexT>(numInputs);
    const uint32_t pairStride = gridDim.x * PairsPerBlock;

    PairIndexT pairBase = static_cast<PairIndexT>(blockIdx.x) * PairsPerBlock;
    while (pairBase < totalPairs) {
        if (lane == 0) {
            PairCopyMetadata metadata{nullptr, nullptr, 0, 1};
            if (pairSlot < totalPairs - pairBase) {
                const PairIndexT pair = pairBase + static_cast<PairIndexT>(pairSlot);
                const PairIndexT row = pair / static_cast<PairIndexT>(numInputs);
                const uint32_t input = static_cast<uint32_t>(pair - row * static_cast<PairIndexT>(numInputs));
                unsigned char *inputGradient = inputGradients[input];
                if (inputGradient != nullptr) {
                    OffsetT destinationBegin = 0;
                    OffsetT rowLength = 0;
                    OffsetT outputBegin = 0;
                    rowPlacement<OffsetT>(
                        inputOffsets, numInputs, input, row, destinationBegin, rowLength, outputBegin);

                    if (rowLength != 0) {
                        metadata.source =
                            outputGradient + static_cast<uint64_t>(outputBegin) * bytesPerValue;
                        metadata.destination =
                            inputGradient + static_cast<uint64_t>(destinationBegin) * bytesPerValue;
                        metadata.byteCount = static_cast<uint64_t>(rowLength) * bytesPerValue;
                        metadata.copyWidthBytes =
                            widestAlignedCopyWidth(metadata.source, metadata.destination, metadata.byteCount);
                    }
                }
            }
            pairMetadata[pairSlot] = metadata;
        }

        if constexpr (kLanesPerPair == 32) {
            __syncwarp();
        } else {
            __syncthreads();
        }
        const PairCopyMetadata metadata = pairMetadata[pairSlot];

        const bool hasNextPairBase = static_cast<PairIndexT>(pairStride) < totalPairs - pairBase;
        if (hasNextPairBase) {
            if constexpr (kLanesPerPair == 32) {
                __syncwarp();
            } else {
                __syncthreads();
            }
        }

        if (metadata.byteCount != 0) copyPair<kLanesPerPair>(metadata, lane);
        if (!hasNextPairBase) break;
        pairBase += static_cast<PairIndexT>(pairStride);
    }
}

uint64_t pairCount(uint64_t batchSize, uint32_t numInputs) {
    if (numInputs == 0) throw std::invalid_argument("RaggedSequenceConcatenate requires at least one input.");
    if (batchSize > std::numeric_limits<uint64_t>::max() / numInputs) {
        throw std::invalid_argument("RaggedSequenceConcatenate batch/input pair count overflow.");
    }
    return batchSize * static_cast<uint64_t>(numInputs);
}

template <uint32_t PairsPerBlock>
uint32_t blocksForPairs(uint64_t pairs) {
    const uint64_t blocks = pairs / PairsPerBlock + (pairs % PairsPerBlock != 0 ? 1 : 0);
    return static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(blocks, 1), kMaxPortableBlocks));
}

uint64_t bytesPerValue(std::size_t valueElementSizeBytes, uint64_t elementsPerValue) {
    if (elementsPerValue > std::numeric_limits<uint64_t>::max() / valueElementSizeBytes) {
        throw std::invalid_argument("RaggedSequenceConcatenate value byte width overflow.");
    }
    return static_cast<uint64_t>(valueElementSizeBytes) * elementsPerValue;
}

void validateOffsetSize(std::size_t bytes) {
    if (bytes != sizeof(uint32_t) && bytes != sizeof(uint64_t)) {
        throw std::invalid_argument("RaggedSequenceConcatenate offsets must use UINT32 or UINT64 storage.");
    }
}

template <typename OffsetT, typename PairIndexT, uint32_t PairsPerBlock>
void launchForwardGrouped(void *outputValues,
                          void *inputValues[],
                          void *inputOffsets[],
                          uint32_t numInputs,
                          uint64_t valueBytes,
                          PairIndexT batchSize,
                          uint64_t pairs,
                          Stream stream) {
    concatenateValuesKernel<OffsetT, PairIndexT, PairsPerBlock>
        <<<blocksForPairs<PairsPerBlock>(pairs), kThreads, 0, stream.getStream()>>>(
            static_cast<unsigned char *>(outputValues),
            reinterpret_cast<unsigned char **>(inputValues),
            inputOffsets,
            numInputs,
            valueBytes,
            batchSize);
    CUDA_CHECK(cudaGetLastError());
}

template <typename OffsetT, typename PairIndexT>
void launchForwardIndexed(void *outputValues,
                          void *inputValues[],
                          void *inputOffsets[],
                          uint32_t numInputs,
                          uint64_t valueBytes,
                          PairIndexT batchSize,
                          uint64_t pairs,
                          Stream stream) {
    // Keep at least about 64 CTAs available once the pair count is large enough,
    // while assigning all 256 CTA lanes to the few-pair case. This avoids the
    // occupancy collapse that would result from unconditionally packing eight
    // pairs into every CTA.
    if (pairs < 128) {
        launchForwardGrouped<OffsetT, PairIndexT, 1>(
            outputValues, inputValues, inputOffsets, numInputs, valueBytes, batchSize, pairs, stream);
    } else if (pairs < 256) {
        launchForwardGrouped<OffsetT, PairIndexT, 2>(
            outputValues, inputValues, inputOffsets, numInputs, valueBytes, batchSize, pairs, stream);
    } else if (pairs < 512) {
        launchForwardGrouped<OffsetT, PairIndexT, 4>(
            outputValues, inputValues, inputOffsets, numInputs, valueBytes, batchSize, pairs, stream);
    } else {
        launchForwardGrouped<OffsetT, PairIndexT, 8>(
            outputValues, inputValues, inputOffsets, numInputs, valueBytes, batchSize, pairs, stream);
    }
}

template <typename OffsetT>
void launchForwardTyped(void *outputValues,
                        void *inputValues[],
                        void *inputOffsets[],
                        uint32_t numInputs,
                        std::size_t valueElementSizeBytes,
                        uint64_t elementsPerValue,
                        uint64_t batchSize,
                        Stream stream) {
    const uint64_t pairs = pairCount(batchSize, numInputs);
    const uint64_t valueBytes = bytesPerValue(valueElementSizeBytes, elementsPerValue);

    // Pair and row arithmetic is 32-bit for the normal path. Preserve the
    // utility's full uint64_t API with a fallback only when the flattened pair
    // count genuinely cannot be represented in 32 bits.
    if (pairs <= kMaxUint32) {
        launchForwardIndexed<OffsetT, uint32_t>(outputValues,
                                                inputValues,
                                                inputOffsets,
                                                numInputs,
                                                valueBytes,
                                                static_cast<uint32_t>(batchSize),
                                                pairs,
                                                stream);
    } else {
        launchForwardIndexed<OffsetT, uint64_t>(
            outputValues, inputValues, inputOffsets, numInputs, valueBytes, batchSize, pairs, stream);
    }
}

template <typename OffsetT, typename PairIndexT, uint32_t PairsPerBlock>
void launchBackwardGrouped(void *inputGradients[],
                           const void *outputGradient,
                           void *inputOffsets[],
                           uint32_t numInputs,
                           uint64_t valueBytes,
                           PairIndexT batchSize,
                           uint64_t pairs,
                           Stream stream) {
    splitGradientKernel<OffsetT, PairIndexT, PairsPerBlock>
        <<<blocksForPairs<PairsPerBlock>(pairs), kThreads, 0, stream.getStream()>>>(
            reinterpret_cast<unsigned char **>(inputGradients),
            static_cast<const unsigned char *>(outputGradient),
            inputOffsets,
            numInputs,
            valueBytes,
            batchSize);
    CUDA_CHECK(cudaGetLastError());
}

template <typename OffsetT, typename PairIndexT>
void launchBackwardIndexed(void *inputGradients[],
                           const void *outputGradient,
                           void *inputOffsets[],
                           uint32_t numInputs,
                           uint64_t valueBytes,
                           PairIndexT batchSize,
                           uint64_t pairs,
                           Stream stream) {
    if (pairs < 128) {
        launchBackwardGrouped<OffsetT, PairIndexT, 1>(
            inputGradients, outputGradient, inputOffsets, numInputs, valueBytes, batchSize, pairs, stream);
    } else if (pairs < 256) {
        launchBackwardGrouped<OffsetT, PairIndexT, 2>(
            inputGradients, outputGradient, inputOffsets, numInputs, valueBytes, batchSize, pairs, stream);
    } else if (pairs < 512) {
        launchBackwardGrouped<OffsetT, PairIndexT, 4>(
            inputGradients, outputGradient, inputOffsets, numInputs, valueBytes, batchSize, pairs, stream);
    } else {
        launchBackwardGrouped<OffsetT, PairIndexT, 8>(
            inputGradients, outputGradient, inputOffsets, numInputs, valueBytes, batchSize, pairs, stream);
    }
}

template <typename OffsetT>
void launchBackwardTyped(void *inputGradients[],
                         const void *outputGradient,
                         void *inputOffsets[],
                         uint32_t numInputs,
                         std::size_t valueElementSizeBytes,
                         uint64_t elementsPerValue,
                         uint64_t batchSize,
                         Stream stream) {
    const uint64_t pairs = pairCount(batchSize, numInputs);
    const uint64_t valueBytes = bytesPerValue(valueElementSizeBytes, elementsPerValue);

    if (pairs <= kMaxUint32) {
        launchBackwardIndexed<OffsetT, uint32_t>(inputGradients,
                                                 outputGradient,
                                                 inputOffsets,
                                                 numInputs,
                                                 valueBytes,
                                                 static_cast<uint32_t>(batchSize),
                                                 pairs,
                                                 stream);
    } else {
        launchBackwardIndexed<OffsetT, uint64_t>(
            inputGradients, outputGradient, inputOffsets, numInputs, valueBytes, batchSize, pairs, stream);
    }
}

}  // namespace

void launchRaggedSequenceConcatenate(void *output_values,
                                     void *input_values[],
                                     void *input_offsets[],
                                     uint32_t num_inputs,
                                     std::size_t value_element_size_bytes,
                                     uint64_t elements_per_value,
                                     std::size_t offsets_element_size_bytes,
                                     uint64_t batch_size,
                                     Stream stream) {
    if (num_inputs < 2) throw std::invalid_argument("RaggedSequenceConcatenate requires at least two inputs.");
    if (value_element_size_bytes == 0 || elements_per_value == 0) {
        throw std::invalid_argument("RaggedSequenceConcatenate values must have non-zero element geometry.");
    }
    validateOffsetSize(offsets_element_size_bytes);
    ScopedGpu scopedGpu(stream.getGpuNum());
    if (offsets_element_size_bytes == sizeof(uint32_t)) {
        launchForwardTyped<uint32_t>(output_values,
                                     input_values,
                                     input_offsets,
                                     num_inputs,
                                     value_element_size_bytes,
                                     elements_per_value,
                                     batch_size,
                                     stream);
    } else {
        launchForwardTyped<uint64_t>(output_values,
                                     input_values,
                                     input_offsets,
                                     num_inputs,
                                     value_element_size_bytes,
                                     elements_per_value,
                                     batch_size,
                                     stream);
    }
}

void launchRaggedSequenceConcatenateBackward(void *input_gradients[],
                                             const void *output_gradient,
                                             void *input_offsets[],
                                             uint32_t num_inputs,
                                             std::size_t value_element_size_bytes,
                                             uint64_t elements_per_value,
                                             std::size_t offsets_element_size_bytes,
                                             uint64_t batch_size,
                                             Stream stream) {
    if (num_inputs < 2) throw std::invalid_argument("RaggedSequenceConcatenate backward requires at least two inputs.");
    if (value_element_size_bytes == 0 || elements_per_value == 0) {
        throw std::invalid_argument("RaggedSequenceConcatenate backward values must have non-zero element geometry.");
    }
    validateOffsetSize(offsets_element_size_bytes);
    ScopedGpu scopedGpu(stream.getGpuNum());
    if (offsets_element_size_bytes == sizeof(uint32_t)) {
        launchBackwardTyped<uint32_t>(input_gradients,
                                      output_gradient,
                                      input_offsets,
                                      num_inputs,
                                      value_element_size_bytes,
                                      elements_per_value,
                                      batch_size,
                                      stream);
    } else {
        launchBackwardTyped<uint64_t>(input_gradients,
                                      output_gradient,
                                      input_offsets,
                                      num_inputs,
                                      value_element_size_bytes,
                                      elements_per_value,
                                      batch_size,
                                      stream);
    }
}
