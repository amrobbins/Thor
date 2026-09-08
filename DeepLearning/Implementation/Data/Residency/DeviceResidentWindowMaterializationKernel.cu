#include "DeepLearning/Implementation/Data/Residency/DeviceResidentWindowMaterializationKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/LowPrecisionFloat.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>
#include <cuda/std/bit>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorPlacement;

namespace {

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kMaxRowsPerBlock = kThreadsPerBlock;
constexpr uint64_t kTargetBytesPerLane = 32;
constexpr uint32_t kMaxPortableBlocks = 65535;

static_assert(sizeof(ulonglong4_32a) == 32);
static_assert(alignof(ulonglong4_32a) == 32);

uint64_t dataTypeBytes(DataType dataType) {
    switch (dataType) {
        case DataType::BOOLEAN:
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return 1;
        case DataType::FP16:
        case DataType::BF16:
        case DataType::INT16:
        case DataType::UINT16:
            return 2;
        case DataType::FP32:
        case DataType::INT32:
        case DataType::UINT32:
            return 4;
        case DataType::FP64:
        case DataType::INT64:
        case DataType::UINT64:
            return 8;
        default:
            break;
    }
    throw std::runtime_error("Unsupported compact resident window data type.");
}

template <typename T>
uint64_t scalarBits(T value) {
    uint64_t bits = 0;
    static_assert(sizeof(T) <= sizeof(bits));
    std::memcpy(&bits, &value, sizeof(T));
    return bits;
}

uint64_t padBits(DataType dataType, double value) {
    if (value == 0.0) return 0;
    switch (dataType) {
        case DataType::BOOLEAN:
            return scalarBits(static_cast<uint8_t>(value != 0.0));
        case DataType::INT8:
            return scalarBits(static_cast<int8_t>(value));
        case DataType::UINT8:
            return scalarBits(static_cast<uint8_t>(value));
        case DataType::INT16:
            return scalarBits(static_cast<int16_t>(value));
        case DataType::UINT16:
            return scalarBits(static_cast<uint16_t>(value));
        case DataType::INT32:
            return scalarBits(static_cast<int32_t>(value));
        case DataType::UINT32:
            return scalarBits(static_cast<uint32_t>(value));
        case DataType::INT64:
            return scalarBits(static_cast<int64_t>(value));
        case DataType::UINT64:
            return scalarBits(static_cast<uint64_t>(value));
        case DataType::FP16:
            return scalarBits(ThorLowPrecision::doubleToFp16Bits(value));
        case DataType::BF16:
            return scalarBits(ThorLowPrecision::doubleToBf16Bits(value));
        case DataType::FP8_E4M3:
            return scalarBits(ThorLowPrecision::doubleToFp8E4M3Bits(value));
        case DataType::FP8_E5M2:
            return scalarBits(ThorLowPrecision::doubleToFp8E5M2Bits(value));
        case DataType::FP32:
            return scalarBits(static_cast<float>(value));
        case DataType::FP64:
            return scalarBits(value);
        default:
            break;
    }
    throw std::runtime_error("Unsupported compact resident non-zero window padding data type.");
}

__device__ __forceinline__ uint32_t widestAlignedCopyWidth(const uint8_t *source,
                                                            const uint8_t *destination) {
    const uintptr_t combined = reinterpret_cast<uintptr_t>(source) |
                               reinterpret_cast<uintptr_t>(destination);
    if ((combined & 31U) == 0) return 32;
    if ((combined & 15U) == 0) return 16;
    if ((combined & 7U) == 0) return 8;
    if ((combined & 3U) == 0) return 4;
    if ((combined & 1U) == 0) return 2;
    return 1;
}

__device__ __forceinline__ uint32_t widestAlignedFillWidth(const uint8_t *destination) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(destination);
    if ((address & 7U) == 0) return 8;
    if ((address & 3U) == 0) return 4;
    if ((address & 1U) == 0) return 2;
    return 1;
}

template <uint32_t NumBytes>
struct ExactTailBytes {
    uint8_t bytes[NumBytes];
};

template <uint32_t NumBytes>
struct RawPacketBytes {
    uint8_t bytes[NumBytes];
};

template <uint32_t TailBytes, typename CopyT>
__device__ __forceinline__ void storeExactTailFromPacket(CopyT packet,
                                                         CopyT *destination) {
    static_assert(TailBytes > 0);
    static_assert(TailBytes < sizeof(CopyT));
    const RawPacketBytes<sizeof(CopyT)> packetBytes =
        cuda::std::bit_cast<RawPacketBytes<sizeof(CopyT)>>(packet);
    ExactTailBytes<TailBytes> tail;
#pragma unroll
    for (uint32_t byte = 0; byte < TailBytes; ++byte) {
        tail.bytes[byte] = packetBytes.bytes[byte];
    }
    *reinterpret_cast<ExactTailBytes<TailBytes> *>(destination) = tail;
}

template <typename CopyT>
__device__ __attribute__((noinline)) void copyExactTailPacket(const CopyT *source,
                                                              CopyT *destination,
                                                              uint32_t tailBytes) {
    static_assert(sizeof(CopyT) == 2 || sizeof(CopyT) == 4 || sizeof(CopyT) == 8 ||
                  sizeof(CopyT) == 16 || sizeof(CopyT) == 32);
    if (tailBytes == 0 || tailBytes >= sizeof(CopyT)) return;
    const CopyT packet = *source;
#define THOR_WINDOW_TAIL_CASE(N)                         \
    case N:                                              \
        if constexpr (N < sizeof(CopyT)) {               \
            storeExactTailFromPacket<N>(packet, destination); \
        }                                                \
        return
    switch (tailBytes) {
        THOR_WINDOW_TAIL_CASE(1);
        THOR_WINDOW_TAIL_CASE(2);
        THOR_WINDOW_TAIL_CASE(3);
        THOR_WINDOW_TAIL_CASE(4);
        THOR_WINDOW_TAIL_CASE(5);
        THOR_WINDOW_TAIL_CASE(6);
        THOR_WINDOW_TAIL_CASE(7);
        THOR_WINDOW_TAIL_CASE(8);
        THOR_WINDOW_TAIL_CASE(9);
        THOR_WINDOW_TAIL_CASE(10);
        THOR_WINDOW_TAIL_CASE(11);
        THOR_WINDOW_TAIL_CASE(12);
        THOR_WINDOW_TAIL_CASE(13);
        THOR_WINDOW_TAIL_CASE(14);
        THOR_WINDOW_TAIL_CASE(15);
        THOR_WINDOW_TAIL_CASE(16);
        THOR_WINDOW_TAIL_CASE(17);
        THOR_WINDOW_TAIL_CASE(18);
        THOR_WINDOW_TAIL_CASE(19);
        THOR_WINDOW_TAIL_CASE(20);
        THOR_WINDOW_TAIL_CASE(21);
        THOR_WINDOW_TAIL_CASE(22);
        THOR_WINDOW_TAIL_CASE(23);
        THOR_WINDOW_TAIL_CASE(24);
        THOR_WINDOW_TAIL_CASE(25);
        THOR_WINDOW_TAIL_CASE(26);
        THOR_WINDOW_TAIL_CASE(27);
        THOR_WINDOW_TAIL_CASE(28);
        THOR_WINDOW_TAIL_CASE(29);
        THOR_WINDOW_TAIL_CASE(30);
        THOR_WINDOW_TAIL_CASE(31);
        default:
            return;
    }
#undef THOR_WINDOW_TAIL_CASE
}

template <typename CopyT, typename ItemIndexT, uint32_t LanesPerRow>
__device__ __forceinline__ void copyAlignedWindowSpan(const uint8_t *sourceBytes,
                                                      uint8_t *destinationBytes,
                                                      ItemIndexT byteCount,
                                                      uint32_t lane) {
    const CopyT *__restrict__ source = reinterpret_cast<const CopyT *>(sourceBytes);
    CopyT *__restrict__ destination = reinterpret_cast<CopyT *>(destinationBytes);
    const ItemIndexT itemCount = byteCount / static_cast<ItemIndexT>(sizeof(CopyT));

    ItemIndexT item = static_cast<ItemIndexT>(lane);
    while (item < itemCount) {
        destination[item] = source[item];
        const ItemIndexT laneStride = static_cast<ItemIndexT>(LanesPerRow);
        if (itemCount - item <= laneStride) break;
        item += laneStride;
    }

    if constexpr (sizeof(CopyT) > 1) {
        if (lane == 0) {
            const uint32_t tailBytes = static_cast<uint32_t>(
                byteCount % static_cast<ItemIndexT>(sizeof(CopyT)));
            if (tailBytes != 0) {
                copyExactTailPacket(source + itemCount, destination + itemCount, tailBytes);
            }
        }
    }
}

template <typename ItemIndexT, uint32_t LanesPerRow>
__device__ __forceinline__ void copyWindowSpan(const uint8_t *source,
                                               uint8_t *destination,
                                               ItemIndexT byteCount,
                                               uint32_t lane) {
    if (byteCount == 0) return;
    switch (widestAlignedCopyWidth(source, destination)) {
        case 32:
            copyAlignedWindowSpan<ulonglong4_32a, ItemIndexT, LanesPerRow>(
                source, destination, byteCount, lane);
            break;
        case 16:
            copyAlignedWindowSpan<uint4, ItemIndexT, LanesPerRow>(
                source, destination, byteCount, lane);
            break;
        case 8:
            copyAlignedWindowSpan<uint64_t, ItemIndexT, LanesPerRow>(
                source, destination, byteCount, lane);
            break;
        case 4:
            copyAlignedWindowSpan<uint32_t, ItemIndexT, LanesPerRow>(
                source, destination, byteCount, lane);
            break;
        case 2:
            copyAlignedWindowSpan<uint16_t, ItemIndexT, LanesPerRow>(
                source, destination, byteCount, lane);
            break;
        default:
            copyAlignedWindowSpan<uint8_t, ItemIndexT, LanesPerRow>(
                source, destination, byteCount, lane);
            break;
    }
}

__host__ __device__ uint64_t repeatedPadWord(uint64_t valueBits, uint64_t elementBytes) {
    if (elementBytes == 8) return valueBits;
    uint64_t word = 0;
    const uint64_t mask = (uint64_t{1} << (elementBytes * 8)) - 1;
    const uint64_t element = valueBits & mask;
    for (uint64_t byte = 0; byte < 8; byte += elementBytes) {
        word |= element << (byte * 8);
    }
    return word;
}

template <typename StoreT, typename ItemIndexT, uint32_t LanesPerRow>
__device__ __forceinline__ void fillAlignedRepeatedPattern(uint8_t *destinationBytes,
                                                           ItemIndexT byteCount,
                                                           uint64_t repeatedWord,
                                                           uint32_t lane) {
    StoreT pattern;
    if constexpr (sizeof(StoreT) == 8) {
        pattern = static_cast<StoreT>(repeatedWord);
    } else if constexpr (sizeof(StoreT) == 4) {
        pattern = static_cast<StoreT>(repeatedWord & 0xffffffffU);
    } else if constexpr (sizeof(StoreT) == 2) {
        pattern = static_cast<StoreT>(repeatedWord & 0xffffU);
    } else {
        pattern = static_cast<StoreT>(repeatedWord & 0xffU);
    }

    StoreT *__restrict__ destination = reinterpret_cast<StoreT *>(destinationBytes);
    const ItemIndexT itemCount = byteCount / static_cast<ItemIndexT>(sizeof(StoreT));
    ItemIndexT item = static_cast<ItemIndexT>(lane);
    while (item < itemCount) {
        destination[item] = pattern;
        const ItemIndexT laneStride = static_cast<ItemIndexT>(LanesPerRow);
        if (itemCount - item <= laneStride) break;
        item += laneStride;
    }

    if constexpr (sizeof(StoreT) > 1) {
        if (lane == 0) {
            const uint32_t tailBytes = static_cast<uint32_t>(
                byteCount % static_cast<ItemIndexT>(sizeof(StoreT)));
            uint8_t *tail = destinationBytes + itemCount * sizeof(StoreT);
#pragma unroll
            for (uint32_t byte = 0; byte < sizeof(StoreT) - 1; ++byte) {
                if (byte < tailBytes) {
                    tail[byte] = static_cast<uint8_t>((repeatedWord >> (8 * byte)) & 0xffU);
                }
            }
        }
    }
}

template <typename ItemIndexT, uint32_t LanesPerRow>
__device__ __forceinline__ void fillRepeatedPattern(uint8_t *destination,
                                                    ItemIndexT byteCount,
                                                    uint64_t repeatedWord,
                                                    uint32_t lane) {
    if (byteCount == 0) return;
    switch (widestAlignedFillWidth(destination)) {
        case 8:
            fillAlignedRepeatedPattern<uint64_t, ItemIndexT, LanesPerRow>(
                destination, byteCount, repeatedWord, lane);
            break;
        case 4:
            fillAlignedRepeatedPattern<uint32_t, ItemIndexT, LanesPerRow>(
                destination, byteCount, repeatedWord, lane);
            break;
        case 2:
            fillAlignedRepeatedPattern<uint16_t, ItemIndexT, LanesPerRow>(
                destination, byteCount, repeatedWord, lane);
            break;
        default:
            fillAlignedRepeatedPattern<uint8_t, ItemIndexT, LanesPerRow>(
                destination, byteCount, repeatedWord, lane);
            break;
    }
}

template <typename PlanT,
          typename RowIndexT,
          typename ItemIndexT,
          uint32_t RowsPerBlock,
          bool MaterializeMask>
__global__ void materializeResolvedWindowKernel(
    const uint8_t *__restrict__ source,
    const PlanT *__restrict__ rowPlans,
    uint8_t *__restrict__ destination,
    RowIndexT logicalRows,
    ItemIndexT windowLength,
    ItemIndexT sourceStepBytes,
    uint64_t repeatedPaddingWord) {
    static_assert(RowsPerBlock >= 1 && RowsPerBlock <= kMaxRowsPerBlock);
    static_assert(kThreadsPerBlock % RowsPerBlock == 0);
    constexpr uint32_t kLanesPerRow = kThreadsPerBlock / RowsPerBlock;

    const uint32_t rowSlot = threadIdx.x / kLanesPerRow;
    const uint32_t lane = threadIdx.x - rowSlot * kLanesPerRow;
    const uint32_t rowStride = gridDim.x * RowsPerBlock;

    RowIndexT rowBase = static_cast<RowIndexT>(blockIdx.x) * RowsPerBlock;
    while (rowBase < logicalRows) {
        if (static_cast<RowIndexT>(rowSlot) < logicalRows - rowBase) {
            const RowIndexT batchRow = rowBase + static_cast<RowIndexT>(rowSlot);
            const PlanT plan = rowPlans[batchRow];
            const ItemIndexT validStepBegin = static_cast<ItemIndexT>(plan.validStepBegin);
            const ItemIndexT validStepCount = static_cast<ItemIndexT>(plan.validStepCount);

            if constexpr (MaterializeMask) {
                uint8_t *rowDestination =
                    destination + static_cast<uint64_t>(batchRow) * windowLength;
                const ItemIndexT validStepEnd = validStepBegin + validStepCount;
                fillRepeatedPattern<ItemIndexT, kLanesPerRow>(
                    rowDestination, validStepBegin, uint64_t{0}, lane);
                fillRepeatedPattern<ItemIndexT, kLanesPerRow>(
                    rowDestination + validStepBegin, validStepCount,
                    uint64_t{0x0101010101010101ULL}, lane);
                fillRepeatedPattern<ItemIndexT, kLanesPerRow>(
                    rowDestination + validStepEnd,
                    windowLength - validStepEnd,
                    uint64_t{0}, lane);
            } else {
                const ItemIndexT rowBytes = windowLength * sourceStepBytes;
                const ItemIndexT prefixBytes = validStepBegin * sourceStepBytes;
                const ItemIndexT validBytes = validStepCount * sourceStepBytes;
                const ItemIndexT validEndBytes = prefixBytes + validBytes;
                uint8_t *rowDestination =
                    destination + static_cast<uint64_t>(batchRow) * rowBytes;

                fillRepeatedPattern<ItemIndexT, kLanesPerRow>(
                    rowDestination, prefixBytes, repeatedPaddingWord, lane);
                if (validBytes != 0) {
                    copyWindowSpan<ItemIndexT, kLanesPerRow>(
                        source + plan.sourceOffsetBytes,
                        rowDestination + prefixBytes,
                        validBytes,
                        lane);
                }
                fillRepeatedPattern<ItemIndexT, kLanesPerRow>(
                    rowDestination + validEndBytes,
                    rowBytes - validEndBytes,
                    repeatedPaddingWord,
                    lane);
            }
        }

        if (static_cast<RowIndexT>(rowStride) >= logicalRows - rowBase) break;
        rowBase += static_cast<RowIndexT>(rowStride);
    }
}

uint32_t rowsPerBlockFor(uint64_t rowBytes, uint64_t logicalRows) {
    uint32_t rowsByPayload = 1;
    if (rowBytes <= kTargetBytesPerLane) {
        rowsByPayload = 256;
    } else if (rowBytes <= 2 * kTargetBytesPerLane) {
        rowsByPayload = 128;
    } else if (rowBytes <= 4 * kTargetBytesPerLane) {
        rowsByPayload = 64;
    } else if (rowBytes <= 8 * kTargetBytesPerLane) {
        rowsByPayload = 32;
    } else if (rowBytes <= 16 * kTargetBytesPerLane) {
        rowsByPayload = 16;
    } else if (rowBytes <= 32 * kTargetBytesPerLane) {
        rowsByPayload = 8;
    } else if (rowBytes <= 64 * kTargetBytesPerLane) {
        rowsByPayload = 4;
    } else if (rowBytes <= 128 * kTargetBytesPerLane) {
        rowsByPayload = 2;
    }

    uint32_t rowsByParallelism = 1;
    if (logicalRows >= 16384) {
        rowsByParallelism = 256;
    } else if (logicalRows >= 8192) {
        rowsByParallelism = 128;
    } else if (logicalRows >= 4096) {
        rowsByParallelism = 64;
    } else if (logicalRows >= 2048) {
        rowsByParallelism = 32;
    } else if (logicalRows >= 1024) {
        rowsByParallelism = 16;
    } else if (logicalRows >= 512) {
        rowsByParallelism = 8;
    } else if (logicalRows >= 256) {
        rowsByParallelism = 4;
    } else if (logicalRows >= 128) {
        rowsByParallelism = 2;
    }
    return std::min(rowsByPayload, rowsByParallelism);
}

template <uint32_t RowsPerBlock>
uint32_t blocksForRows(uint64_t logicalRows) {
    const uint64_t blocks =
        logicalRows / RowsPerBlock + (logicalRows % RowsPerBlock != 0 ? 1 : 0);
    return static_cast<uint32_t>(
        std::min<uint64_t>(std::max<uint64_t>(blocks, 1), kMaxPortableBlocks));
}

template <typename PlanT,
          typename RowIndexT,
          typename ItemIndexT,
          uint32_t RowsPerBlock,
          bool MaterializeMask>
void launchGrouped(const uint8_t *source,
                   const PlanT *rowPlans,
                   uint8_t *destination,
                   RowIndexT logicalRows,
                   ItemIndexT windowLength,
                   ItemIndexT sourceStepBytes,
                   uint64_t repeatedPaddingWord,
                   cudaStream_t stream) {
    const uint32_t blocks = blocksForRows<RowsPerBlock>(static_cast<uint64_t>(logicalRows));
    materializeResolvedWindowKernel<
        PlanT, RowIndexT, ItemIndexT, RowsPerBlock, MaterializeMask>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(
            source,
            rowPlans,
            destination,
            logicalRows,
            windowLength,
            sourceStepBytes,
            repeatedPaddingWord);
    CUDA_CHECK(cudaGetLastError());
}

template <typename PlanT, typename RowIndexT, typename ItemIndexT, bool MaterializeMask>
void launchForGrouping(const uint8_t *source,
                       const PlanT *rowPlans,
                       uint8_t *destination,
                       RowIndexT logicalRows,
                       ItemIndexT windowLength,
                       ItemIndexT sourceStepBytes,
                       uint64_t repeatedPaddingWord,
                       uint64_t rowBytes,
                       cudaStream_t stream) {
#define THOR_WINDOW_GROUP_CASE(N)                                                    \
    case N:                                                                          \
        launchGrouped<PlanT, RowIndexT, ItemIndexT, N, MaterializeMask>(             \
            source, rowPlans, destination, logicalRows, windowLength, sourceStepBytes, \
            repeatedPaddingWord, stream);                                             \
        break
    switch (rowsPerBlockFor(rowBytes, static_cast<uint64_t>(logicalRows))) {
        THOR_WINDOW_GROUP_CASE(256);
        THOR_WINDOW_GROUP_CASE(128);
        THOR_WINDOW_GROUP_CASE(64);
        THOR_WINDOW_GROUP_CASE(32);
        THOR_WINDOW_GROUP_CASE(16);
        THOR_WINDOW_GROUP_CASE(8);
        THOR_WINDOW_GROUP_CASE(4);
        THOR_WINDOW_GROUP_CASE(2);
        THOR_WINDOW_GROUP_CASE(1);
        default:
            throw std::runtime_error("Invalid compact resident window row grouping.");
    }
#undef THOR_WINDOW_GROUP_CASE
}

template <typename PlanT, typename RowIndexT, typename ItemIndexT>
void launchTyped(const uint8_t *source,
                 const PlanT *rowPlans,
                 uint8_t *destination,
                 RowIndexT logicalRows,
                 ItemIndexT windowLength,
                 ItemIndexT sourceStepBytes,
                 uint64_t repeatedPaddingWord,
                 uint64_t rowBytes,
                 bool materializeMask,
                 cudaStream_t stream) {
    if (materializeMask) {
        launchForGrouping<PlanT, RowIndexT, ItemIndexT, true>(
            source, rowPlans, destination, logicalRows, windowLength, sourceStepBytes,
            repeatedPaddingWord, rowBytes, stream);
    } else {
        launchForGrouping<PlanT, RowIndexT, ItemIndexT, false>(
            source, rowPlans, destination, logicalRows, windowLength, sourceStepBytes,
            repeatedPaddingWord, rowBytes, stream);
    }
}

void validateTensor(const Tensor &tensor, TensorPlacement placement, const char *name) {
    if (!tensor.isInitialized() || tensor.getPlacement() != placement ||
        tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::runtime_error(std::string("Invalid compact resident ") + name + " tensor.");
    }
}

}  // namespace

void launchDeviceResidentWindowMaterializationKernel(
    const Tensor &sourceStorage,
    const Tensor &rowPlans,
    uint64_t logicalRows,
    const DeviceResidentWindowMaterializationSpec &spec,
    Tensor &destination,
    Stream &stream) {
    THOR_THROW_IF_FALSE(destination.isInitialized());
    THOR_THROW_IF_FALSE(rowPlans.isInitialized());
    const TensorPlacement placement = destination.getPlacement();
    validateTensor(rowPlans, placement, "window row-plan");
    // Session-owned plans may be UINT64 alias views into the consolidated
    // selection-metadata upload allocation.  Direct kernel tests and standalone
    // callers may continue to provide byte tensors.
    THOR_THROW_IF_FALSE(
        rowPlans.getDataType() == DataType::UINT8 ||
        rowPlans.getDataType() == DataType::UINT64);
    if (!spec.materializeMask) validateTensor(sourceStorage, placement, "source storage");

    THOR_THROW_IF_FALSE(spec.windowLength > 0);
    THOR_THROW_IF_FALSE(spec.sourceStepBytes > 0);
    const std::vector<uint64_t> destinationDims = destination.getDimensions();
    THOR_THROW_IF_FALSE(!destinationDims.empty());
    const uint64_t batchCapacity = destinationDims.front();
    THOR_THROW_IF_FALSE(logicalRows <= batchCapacity);

    THOR_THROW_IF_FALSE(
        spec.windowLength <= std::numeric_limits<uint64_t>::max() / spec.sourceStepBytes);
    const uint64_t payloadRowBytes = spec.windowLength * spec.sourceStepBytes;
    const uint64_t rowBytes = spec.materializeMask ? spec.windowLength : payloadRowBytes;
    THOR_THROW_IF_FALSE(
        batchCapacity == 0 || rowBytes <= std::numeric_limits<uint64_t>::max() / batchCapacity);
    if (spec.materializeMask) {
        THOR_THROW_IF_FALSE(destination.getDataType() == DataType::UINT8);
        THOR_THROW_IF_FALSE(destination.getArraySizeInBytes() == batchCapacity * spec.windowLength);
    } else {
        THOR_THROW_IF_FALSE(destination.getDataType() == spec.dataType);
        THOR_THROW_IF_FALSE(destination.getArraySizeInBytes() == batchCapacity * payloadRowBytes);
    }

    const bool usePlan32 = spec.windowLength <= std::numeric_limits<uint32_t>::max();
    const uint64_t planBytes = usePlan32 ? sizeof(DeviceResidentWindowRowPlan32)
                                         : sizeof(DeviceResidentWindowRowPlan64);
    THOR_THROW_IF_FALSE(
        logicalRows == 0 || planBytes <= std::numeric_limits<uint64_t>::max() / logicalRows);
    THOR_THROW_IF_FALSE(rowPlans.getArraySizeInBytes() == logicalRows * planBytes);
    if (logicalRows == 0) return;

    const uint64_t elementBytes = dataTypeBytes(spec.dataType);
    THOR_THROW_IF_FALSE(spec.sourceStepBytes % elementBytes == 0);
    const uint64_t paddingWord = repeatedPadWord(padBits(spec.dataType, spec.padValue), elementBytes);
    const uint8_t *source = sourceStorage.isInitialized()
                                ? static_cast<const uint8_t *>(sourceStorage.getMemPtr())
                                : nullptr;
    uint8_t *destinationBytes = static_cast<uint8_t *>(destination.getMemPtr());
    const bool useRow32 = logicalRows <= std::numeric_limits<uint32_t>::max();
    const bool useItem32 = rowBytes <= std::numeric_limits<uint32_t>::max() &&
                           spec.sourceStepBytes <= std::numeric_limits<uint32_t>::max();

#define THOR_LAUNCH_PLAN(PlanT, plansPtr)                                                       \
    do {                                                                                       \
        if (useRow32 && useItem32) {                                                           \
            launchTyped<PlanT, uint32_t, uint32_t>(                                            \
                source, plansPtr, destinationBytes, static_cast<uint32_t>(logicalRows),        \
                static_cast<uint32_t>(spec.windowLength),                                      \
                static_cast<uint32_t>(spec.sourceStepBytes), paddingWord, rowBytes,            \
                spec.materializeMask, stream.getStream());                                     \
        } else if (useRow32) {                                                                 \
            launchTyped<PlanT, uint32_t, uint64_t>(                                            \
                source, plansPtr, destinationBytes, static_cast<uint32_t>(logicalRows),        \
                spec.windowLength, spec.sourceStepBytes, paddingWord, rowBytes,                \
                spec.materializeMask, stream.getStream());                                     \
        } else if (useItem32) {                                                                \
            launchTyped<PlanT, uint64_t, uint32_t>(                                            \
                source, plansPtr, destinationBytes, logicalRows,                               \
                static_cast<uint32_t>(spec.windowLength),                                      \
                static_cast<uint32_t>(spec.sourceStepBytes), paddingWord, rowBytes,            \
                spec.materializeMask, stream.getStream());                                     \
        } else {                                                                               \
            launchTyped<PlanT, uint64_t, uint64_t>(                                            \
                source, plansPtr, destinationBytes, logicalRows, spec.windowLength,            \
                spec.sourceStepBytes, paddingWord, rowBytes,                                   \
                spec.materializeMask, stream.getStream());                                     \
        }                                                                                      \
    } while (false)

    if (usePlan32) {
        THOR_LAUNCH_PLAN(
            DeviceResidentWindowRowPlan32,
            reinterpret_cast<const DeviceResidentWindowRowPlan32 *>(rowPlans.getMemPtr()));
    } else {
        THOR_LAUNCH_PLAN(
            DeviceResidentWindowRowPlan64,
            reinterpret_cast<const DeviceResidentWindowRowPlan64 *>(rowPlans.getMemPtr()));
    }
#undef THOR_LAUNCH_PLAN
}
