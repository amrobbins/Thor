#include "DeepLearning/Implementation/Data/Residency/DeviceResidentWindowMaterializationKernel.h"

#include "DeepLearning/Implementation/Data/Residency/DeviceResidentPacketGrid.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/LowPrecisionFloat.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorPlacement;
using ThorImplementation::TensorSpan;
using namespace DeviceResidentPacketGrid;

namespace {

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

// The 8 bytes that repeat through padding: the pad element, repeated.
uint64_t repeatedPadWord(uint64_t valueBits, uint64_t elementBytes) {
    if (elementBytes == 8) return valueBits;
    uint64_t word = 0;
    const uint64_t mask = (uint64_t{1} << (elementBytes * 8)) - 1;
    const uint64_t element = valueBits & mask;
    for (uint64_t byte = 0; byte < 8; byte += elementBytes) {
        word |= element << (byte * 8);
    }
    return word;
}

// A row plan as the kernel reads it: 64-bit words, two for a 32-bit-step plan (the offset, then begin in the low half
// and count in the high half) and three for a 64-bit-step plan.
static_assert(offsetof(DeviceResidentWindowRowPlan32, sourceOffsetBytes) == 0);
static_assert(offsetof(DeviceResidentWindowRowPlan32, validStepBegin) == 8);
static_assert(offsetof(DeviceResidentWindowRowPlan32, validStepCount) == 12);
static_assert(offsetof(DeviceResidentWindowRowPlan64, sourceOffsetBytes) == 0);
static_assert(offsetof(DeviceResidentWindowRowPlan64, validStepBegin) == 8);
static_assert(offsetof(DeviceResidentWindowRowPlan64, validStepCount) == 16);

template <typename PlanT>
struct PlanSteps;

template <>
struct PlanSteps<DeviceResidentWindowRowPlan32> {
    uint64_t beginAndCount;
    __device__ __forceinline__ static PlanSteps load(const uint64_t *__restrict__ planWords, uint64_t row) {
        return PlanSteps{planWords[2 * row + 1]};
    }
    __device__ __forceinline__ static uint64_t loadSourceOffsetBytes(const uint64_t *__restrict__ planWords, uint64_t row) {
        return planWords[2 * row];
    }
    __device__ __forceinline__ uint32_t begin() const { return static_cast<uint32_t>(beginAndCount); }
    __device__ __forceinline__ uint32_t count() const { return static_cast<uint32_t>(beginAndCount >> 32); }
};

template <>
struct PlanSteps<DeviceResidentWindowRowPlan64> {
    uint64_t validStepBegin;
    uint64_t validStepCount;
    __device__ __forceinline__ static PlanSteps load(const uint64_t *__restrict__ planWords, uint64_t row) {
        return PlanSteps{planWords[3 * row + 1], planWords[3 * row + 2]};
    }
    __device__ __forceinline__ static uint64_t loadSourceOffsetBytes(const uint64_t *__restrict__ planWords, uint64_t row) {
        return planWords[3 * row];
    }
    __device__ __forceinline__ uint64_t begin() const { return validStepBegin; }
    __device__ __forceinline__ uint64_t count() const { return validStepCount; }
};

// The padding packet at packetInRow of a row. A row starts on an element boundary and the padding word repeats every
// 8 bytes, so the packet is the word from the packet's byte offset modulo 8 on; a packet wider than the word repeats
// it.
__device__ __forceinline__ uint4 paddingPacket(uint4 *, uint64_t word, uint64_t) {
    const uint32_t low = static_cast<uint32_t>(word);
    const uint32_t high = static_cast<uint32_t>(word >> 32);
    return make_uint4(low, high, low, high);
}
__device__ __forceinline__ uint2 paddingPacket(uint2 *, uint64_t word, uint64_t) {
    return make_uint2(static_cast<uint32_t>(word), static_cast<uint32_t>(word >> 32));
}
__device__ __forceinline__ uint32_t paddingPacket(uint32_t *, uint64_t word, uint64_t packetInRow) {
    return static_cast<uint32_t>(word >> ((packetInRow & 1) * 32));
}
__device__ __forceinline__ uint16_t paddingPacket(uint16_t *, uint64_t word, uint64_t packetInRow) {
    return static_cast<uint16_t>(word >> ((packetInRow & 3) * 16));
}
__device__ __forceinline__ uint8_t paddingPacket(uint8_t *, uint64_t word, uint64_t packetInRow) {
    return static_cast<uint8_t>(word >> ((packetInRow & 7) * 8));
}

__device__ __forceinline__ uint4 choosePacket(bool first, uint4 a, uint4 b) {
    return make_uint4(first ? a.x : b.x, first ? a.y : b.y, first ? a.z : b.z, first ? a.w : b.w);
}
__device__ __forceinline__ uint2 choosePacket(bool first, uint2 a, uint2 b) {
    return make_uint2(first ? a.x : b.x, first ? a.y : b.y);
}
__device__ __forceinline__ uint32_t choosePacket(bool first, uint32_t a, uint32_t b) { return first ? a : b; }
__device__ __forceinline__ uint16_t choosePacket(bool first, uint16_t a, uint16_t b) { return first ? a : b; }
__device__ __forceinline__ uint8_t choosePacket(bool first, uint8_t a, uint8_t b) { return first ? a : b; }

// The mask bytes of the 8 steps from firstStep on, as one word (the first step in the low byte): 1 for the steps in
// [validBegin, validEnd), 0 for the others.
template <typename IndexT>
__device__ __forceinline__ uint64_t maskWord(IndexT firstStep, IndexT validBegin, IndexT validEnd) {
    constexpr uint64_t kOnes = 0x0101010101010101ULL;
    // Of the 8 steps, how many are before the valid steps, and how many are before their end.
    const uint32_t before = firstStep >= validBegin ? 0 : static_cast<uint32_t>(validBegin - firstStep < 8 ? validBegin - firstStep : 8);
    const uint32_t through = firstStep >= validEnd ? 0 : static_cast<uint32_t>(validEnd - firstStep < 8 ? validEnd - firstStep : 8);
    const uint64_t beforeOnes = before >= 8 ? kOnes : kOnes & ((uint64_t(1) << (8 * before)) - 1);
    const uint64_t throughOnes = through >= 8 ? kOnes : kOnes & ((uint64_t(1) << (8 * through)) - 1);
    return throughOnes & ~beforeOnes;
}

// The steps a mask packet takes from one row: the packet's first byte is step firstStep, and the steps in
// [validBegin, validEnd) are valid.
template <typename IndexT>
struct MaskOfOneRow {
    IndexT firstStep;
    IndexT validBegin;
    IndexT validEnd;
    // The mask bytes of the packet's 8 bytes from byte on.
    __device__ __forceinline__ uint64_t word(IndexT byte) const { return maskWord<IndexT>(firstStep + byte, validBegin, validEnd); }
};

// The steps of a mask packet that starts in one row and may end in the next: each row gives the bytes that are its
// own valid steps.
template <typename IndexT>
struct MaskOfTwoRows {
    MaskOfOneRow<IndexT> firstRow;
    MaskOfOneRow<IndexT> lastRow;
    __device__ __forceinline__ uint64_t word(IndexT byte) const { return firstRow.word(byte) | lastRow.word(byte); }
};

// The mask packet of steps (MaskOfOneRow or MaskOfTwoRows): a step is a byte, so a packet holds sizeof(PacketT) steps.
template <typename Steps>
__device__ __forceinline__ uint4 maskPacket(uint4 *, const Steps &steps) {
    const uint64_t low = steps.word(0);
    const uint64_t high = steps.word(8);
    return make_uint4(static_cast<uint32_t>(low), static_cast<uint32_t>(low >> 32), static_cast<uint32_t>(high),
                      static_cast<uint32_t>(high >> 32));
}
template <typename Steps>
__device__ __forceinline__ uint2 maskPacket(uint2 *, const Steps &steps) {
    const uint64_t word = steps.word(0);
    return make_uint2(static_cast<uint32_t>(word), static_cast<uint32_t>(word >> 32));
}
template <typename Steps>
__device__ __forceinline__ uint32_t maskPacket(uint32_t *, const Steps &steps) {
    return static_cast<uint32_t>(steps.word(0));
}
template <typename Steps>
__device__ __forceinline__ uint16_t maskPacket(uint16_t *, const Steps &steps) {
    return static_cast<uint16_t>(steps.word(0));
}
template <typename Steps>
__device__ __forceinline__ uint8_t maskPacket(uint8_t *, const Steps &steps) {
    return static_cast<uint8_t>(steps.word(0));
}

/**
 * Materialize window payloads as one flat array of packets, like the row gather (DeviceResidentRowGatherKernel.cu,
 * whose launch geometry and load order this follows). A batch row is windowLength steps: the steps from
 * validStepBegin on, validStepCount of them, come from the source at the row plan's sourceOffsetBytes, and the steps
 * around them are padding.
 *
 * A step is packetsPerStep packets. Every row, every step and every plan's source offset starts on a PacketT
 * boundary, so a packet is all source or all padding and every access is a full aligned packet. A thread issues the
 * plan loads of all its packets before it reads any plan, then all their source loads before it stores any packet.
 * No access is under a test of its own: a packet past the end is the last packet again, a padding packet reads
 * source packet 0 (the launcher requires one) and stores the padding, and every packet is stored.
 */
template <typename PacketT, typename IndexT, typename RowOfPacket, typename PlanT, uint32_t PacketsPerThread>
__global__ void __launch_bounds__(kThreadsPerBlock)
    windowPacketsKernel(const PacketT *__restrict__ source,
                        PacketT *__restrict__ destination,
                        const uint64_t *__restrict__ rowPlanWords,
                        IndexT totalPackets,
                        IndexT packetsPerRow,
                        RowOfPacket rowOfPacket,
                        IndexT packetsPerStep,
                        uint64_t paddingWord) {
    constexpr IndexT tilePackets = static_cast<IndexT>(kThreadsPerBlock) * PacketsPerThread;
    const IndexT tileStride = static_cast<IndexT>(gridDim.x) * tilePackets;
    const IndexT lastPacket = totalPackets - 1;
    IndexT tileStart = static_cast<IndexT>(blockIdx.x) * tilePackets;
    while (tileStart < totalPackets) {
        // Plan loads only: nothing here reads a loaded plan, so the thread's plans are in flight together.
        IndexT packet[PacketsPerThread];
        IndexT packetInRow[PacketsPerThread];
        PlanSteps<PlanT> steps[PacketsPerThread];
        uint64_t sourceOffsetBytes[PacketsPerThread];
#pragma unroll
        for (uint32_t k = 0; k < PacketsPerThread; ++k) {
            const IndexT ownPacket = tileStart + static_cast<IndexT>(k * kThreadsPerBlock + threadIdx.x);
            packet[k] = ownPacket < totalPackets ? ownPacket : lastPacket;
            const IndexT row = rowOfPacket(packet[k]);
            packetInRow[k] = packet[k] - row * packetsPerRow;
            steps[k] = PlanSteps<PlanT>::load(rowPlanWords, row);
            sourceOffsetBytes[k] = PlanSteps<PlanT>::loadSourceOffsetBytes(rowPlanWords, row);
        }
        // Source loads: the first use of the plans. No loaded packet is read before the stores.
        PacketT value[PacketsPerThread];
        bool fromSource[PacketsPerThread];
#pragma unroll
        for (uint32_t k = 0; k < PacketsPerThread; ++k) {
            const IndexT validPackets = static_cast<IndexT>(steps[k].count()) * packetsPerStep;
            // Packets past the row's first valid packet; wraps to a large value before it.
            const IndexT packetInValid = packetInRow[k] - static_cast<IndexT>(steps[k].begin()) * packetsPerStep;
            fromSource[k] = packetInValid < validPackets;
            value[k] = source[fromSource[k] ? sourceOffsetBytes[k] / sizeof(PacketT) + packetInValid : 0];
        }
#pragma unroll
        for (uint32_t k = 0; k < PacketsPerThread; ++k) {
            destination[packet[k]] = choosePacket(
                fromSource[k], value[k], paddingPacket(static_cast<PacketT *>(nullptr), paddingWord, packetInRow[k]));
        }
        // Break before the increment so the 32-bit index cannot wrap on the last tile.
        if (totalPackets - tileStart <= tileStride) break;
        tileStart += tileStride;
    }
}

/**
 * Materialize window masks: the batch is one byte per step, 1 where the step is valid. The batch is one flat array of
 * packets from its first byte on, whatever the window length, and a packet is computed from the row plans; there is
 * no source. A thread issues the plan loads of all its packets before it reads any plan. A packet past the end is the
 * last packet again.
 *
 * CrossesRows false: the window length is a whole number of packets, so every packet lies in one row.
 * CrossesRows true: it is not. A packet is no longer than a row, so it lies in one row or ends in the next, and it is
 * computed from those two rows' plans. The first row gives its steps from the packet's first step on; no step past
 * the row's end is valid (a plan's valid steps are steps of its row), so it gives the next row's bytes nothing. The
 * next row's step 0 is the packet's byte stepsToNextRow, so in the packet its valid steps are stepsToNextRow further
 * on; for a packet that lies in one row that is past every byte of the packet, and the plan loaded for it is the
 * first row's again. The batch need not be a whole number of packets: the tailBytes bytes after the last whole
 * packet are stored one each by the first threads of block 0, so nothing past the batch is written.
 *
 * rowOfByte divides a byte's position in the batch by the window length.
 */
template <typename PacketT, typename IndexT, typename RowOfByte, typename PlanT, uint32_t PacketsPerThread, bool CrossesRows>
__global__ void __launch_bounds__(kThreadsPerBlock)
    maskPacketsKernel(PacketT *__restrict__ destination,
                      const uint64_t *__restrict__ rowPlanWords,
                      IndexT totalPackets,
                      IndexT windowLength,
                      RowOfByte rowOfByte,
                      uint32_t tailBytes) {
    constexpr IndexT tilePackets = static_cast<IndexT>(kThreadsPerBlock) * PacketsPerThread;
    const IndexT tileStride = static_cast<IndexT>(gridDim.x) * tilePackets;
    const IndexT lastPacket = totalPackets - 1;
    IndexT tileStart = static_cast<IndexT>(blockIdx.x) * tilePackets;
    while (tileStart < totalPackets) {
        // Plan loads only: nothing here reads a loaded plan.
        IndexT packet[PacketsPerThread];
        IndexT firstStep[PacketsPerThread];
        PlanSteps<PlanT> firstRowSteps[PacketsPerThread];
        [[maybe_unused]] PlanSteps<PlanT> lastRowSteps[PacketsPerThread];
#pragma unroll
        for (uint32_t k = 0; k < PacketsPerThread; ++k) {
            const IndexT ownPacket = tileStart + static_cast<IndexT>(k * kThreadsPerBlock + threadIdx.x);
            packet[k] = ownPacket < totalPackets ? ownPacket : lastPacket;
            const IndexT firstByte = packet[k] * static_cast<IndexT>(sizeof(PacketT));
            const IndexT firstRow = rowOfByte(firstByte);
            firstStep[k] = firstByte - firstRow * windowLength;
            firstRowSteps[k] = PlanSteps<PlanT>::load(rowPlanWords, firstRow);
            if constexpr (CrossesRows) {
                // The packet's last byte is in the next row when fewer than a packet of steps are left in this one.
                const IndexT lastRow =
                    windowLength - firstStep[k] < static_cast<IndexT>(sizeof(PacketT)) ? firstRow + 1 : firstRow;
                lastRowSteps[k] = PlanSteps<PlanT>::load(rowPlanWords, lastRow);
            }
        }
#pragma unroll
        for (uint32_t k = 0; k < PacketsPerThread; ++k) {
            const IndexT firstRowBegin = static_cast<IndexT>(firstRowSteps[k].begin());
            const MaskOfOneRow<IndexT> firstRow{
                firstStep[k], firstRowBegin, firstRowBegin + static_cast<IndexT>(firstRowSteps[k].count())};
            if constexpr (CrossesRows) {
                // The next row's step 0 is this byte of the packet.
                const IndexT stepsToNextRow = windowLength - firstStep[k];
                const IndexT lastRowBegin = stepsToNextRow + static_cast<IndexT>(lastRowSteps[k].begin());
                const MaskOfOneRow<IndexT> lastRow{0, lastRowBegin, lastRowBegin + static_cast<IndexT>(lastRowSteps[k].count())};
                destination[packet[k]] = maskPacket(static_cast<PacketT *>(nullptr), MaskOfTwoRows<IndexT>{firstRow, lastRow});
            } else {
                destination[packet[k]] = maskPacket(static_cast<PacketT *>(nullptr), firstRow);
            }
        }
        // Break before the increment so the 32-bit index cannot wrap on the last tile.
        if (totalPackets - tileStart <= tileStride) break;
        tileStart += tileStride;
    }
    if constexpr (CrossesRows) {
        if (blockIdx.x == 0 && threadIdx.x < tailBytes) {
            const IndexT byte = totalPackets * static_cast<IndexT>(sizeof(PacketT)) + static_cast<IndexT>(threadIdx.x);
            const IndexT row = rowOfByte(byte);
            const PlanSteps<PlanT> steps = PlanSteps<PlanT>::load(rowPlanWords, row);
            const uint64_t stepInValid = static_cast<uint64_t>(byte - row * windowLength) - static_cast<uint64_t>(steps.begin());
            reinterpret_cast<uint8_t *>(destination)[byte] = stepInValid < static_cast<uint64_t>(steps.count()) ? 1 : 0;
        }
    }
}
// The source's 16-byte words are aligned, and a window's bytes start at any byte of the source, so the source bytes
// for a packet's valid bytes lie in one word or in two consecutive ones: first holds the first of them, last the
// last (first again when one word holds them all). shift is the byte of the word at which the packet's byte 0 would
// be read were it valid, so byte j of the packet is byte shift + j of the two words, counted from the first when
// shift + j is below 16 and from the last otherwise.
struct SourceWords {
    uint64_t first;
    uint64_t last;
    uint32_t shift;
};

// The words that hold the source bytes for the bytes a packet takes from one row, whose valid bytes start at
// sourceOffsetBytes of the source. Only words with a byte the packet takes are named, so nothing else of the source
// is read: a packet that takes no byte of the row names word 0 twice. A word may end up to 15 bytes past the source,
// inside the padding every tensor is allocated with.
template <typename IndexT>
__device__ __forceinline__ SourceWords sourceWordsOf(uint64_t sourceOffsetBytes, const MaskOfOneRow<IndexT> &row) {
    constexpr IndexT kPacketBytes = static_cast<IndexT>(sizeof(uint4));
    // The packet's bytes [firstValid, endValid) are valid bytes of the row.
    const IndexT untilBegin = row.validBegin - row.firstStep;
    const IndexT untilEnd = row.validEnd - row.firstStep;
    const IndexT firstValid = row.firstStep >= row.validBegin ? 0 : untilBegin < kPacketBytes ? untilBegin : kPacketBytes;
    const IndexT endValid = row.firstStep >= row.validEnd ? 0 : untilEnd < kPacketBytes ? untilEnd : kPacketBytes;
    const bool takesBytes = firstValid < endValid;
    // Where the packet's byte 0 would be read. It wraps for a packet that starts before the valid bytes; the bytes
    // the packet takes are then at or after sourceOffsetBytes all the same.
    const uint64_t byteZero = sourceOffsetBytes + row.firstStep - row.validBegin;
    return SourceWords{takesBytes ? (byteZero + firstValid) / kPacketBytes : 0,
                       takesBytes ? (byteZero + endValid - 1) / kPacketBytes : 0,
                       static_cast<uint32_t>(byteZero) & 15U};
}

// The packet with the bytes of chosen where the row has valid bytes (the row's mask packet, 1 per valid byte) and the
// bytes of other elsewhere.
__device__ __forceinline__ uint4 chooseBytes(uint4 validBytes, uint4 chosen, uint4 other) {
    const uint4 all = make_uint4(validBytes.x * 0xffU, validBytes.y * 0xffU, validBytes.z * 0xffU, validBytes.w * 0xffU);
    return make_uint4((chosen.x & all.x) | (other.x & ~all.x),
                      (chosen.y & all.y) | (other.y & ~all.y),
                      (chosen.z & all.z) | (other.z & ~all.z),
                      (chosen.w & all.w) | (other.w & ~all.w));
}

/**
 * Materialize window payloads whose steps or source offsets are on no wider boundary than a byte, 16 bytes a packet.
 * The batch is one flat array of 16-byte packets from its first byte on, as for a mask (maskPacketsKernel), and a
 * packet's bytes are chosen one by one: a valid byte of its row from the source, any other byte from the padding.
 *
 * The source bytes for a packet start at any byte, so they are read as the one or two aligned words that hold them
 * and shifted (sourceWordsOf, bytesFrom). The padding packet is the same for every packet: a row and a packet both
 * start on an element boundary, and the padding word repeats every 8 bytes.
 *
 * A thread takes one packet: it issues its plan loads, then its source loads (the first use of the plans), then
 * stores. No load is under a test: a packet past the end is the last packet again, and a packet with no valid byte
 * reads word 0 of the source.
 *
 * CrossesRows false: the row is a whole number of packets, so every packet lies in one row.
 * CrossesRows true: a packet lies in one row or ends in the next (the launcher requires rows of at least a packet).
 * The next row's byte 0 is the packet's byte bytesToNextRow, so its valid bytes and their source are counted from
 * there; for a packet that lies in one row they are past the packet (it takes none, and reads word 0), and the plan
 * loaded for it is the first row's again. The tailBytes bytes after the last whole packet are copied one each by the
 * first threads of block 0.
 *
 * rowOfByte divides a byte's position in the batch by the row's bytes.
 */
template <typename IndexT, typename RowOfByte, typename PlanT, bool CrossesRows>
__global__ void __launch_bounds__(kThreadsPerBlock)
    windowBytesKernel(const uint4 *__restrict__ source,
                      uint4 *__restrict__ destination,
                      const uint64_t *__restrict__ rowPlanWords,
                      IndexT totalPackets,
                      IndexT rowBytes,
                      RowOfByte rowOfByte,
                      IndexT stepBytes,
                      uint64_t paddingWord,
                      uint32_t tailBytes) {
    constexpr IndexT kPacketBytes = static_cast<IndexT>(sizeof(uint4));
    const IndexT tileStride = static_cast<IndexT>(gridDim.x) * static_cast<IndexT>(kThreadsPerBlock);
    const IndexT lastPacket = totalPackets - 1;
    const uint4 padding = paddingPacket(static_cast<uint4 *>(nullptr), paddingWord, 0);
    IndexT tileStart = static_cast<IndexT>(blockIdx.x) * static_cast<IndexT>(kThreadsPerBlock);
    while (tileStart < totalPackets) {
        const IndexT ownPacket = tileStart + static_cast<IndexT>(threadIdx.x);
        const IndexT packet = ownPacket < totalPackets ? ownPacket : lastPacket;
        const IndexT firstByte = packet * kPacketBytes;
        const IndexT firstRow = rowOfByte(firstByte);
        const IndexT byteInRow = firstByte - firstRow * rowBytes;

        // Plan loads only: nothing here reads a loaded plan.
        const PlanSteps<PlanT> firstRowSteps = PlanSteps<PlanT>::load(rowPlanWords, firstRow);
        const uint64_t firstRowOffset = PlanSteps<PlanT>::loadSourceOffsetBytes(rowPlanWords, firstRow);
        [[maybe_unused]] PlanSteps<PlanT> lastRowSteps;
        [[maybe_unused]] uint64_t lastRowOffset;
        if constexpr (CrossesRows) {
            // The packet's last byte is in the next row when less than a packet is left of this one.
            const IndexT lastRow = rowBytes - byteInRow < kPacketBytes ? firstRow + 1 : firstRow;
            lastRowSteps = PlanSteps<PlanT>::load(rowPlanWords, lastRow);
            lastRowOffset = PlanSteps<PlanT>::loadSourceOffsetBytes(rowPlanWords, lastRow);
        }

        // Source loads: the first use of the plans. No loaded word is read before the last load is issued.
        const IndexT firstRowBegin = static_cast<IndexT>(firstRowSteps.begin()) * stepBytes;
        const MaskOfOneRow<IndexT> firstRowBytes{
            byteInRow, firstRowBegin, firstRowBegin + static_cast<IndexT>(firstRowSteps.count()) * stepBytes};
        const SourceWords firstRowWords = sourceWordsOf(firstRowOffset, firstRowBytes);
        const uint4 firstRowFirstWord = source[firstRowWords.first];
        const uint4 firstRowLastWord = source[firstRowWords.last];
        [[maybe_unused]] MaskOfOneRow<IndexT> lastRowBytes;
        [[maybe_unused]] SourceWords lastRowWords;
        [[maybe_unused]] uint4 lastRowFirstWord;
        [[maybe_unused]] uint4 lastRowLastWord;
        if constexpr (CrossesRows) {
            const IndexT bytesToNextRow = rowBytes - byteInRow;
            const IndexT lastRowBegin = bytesToNextRow + static_cast<IndexT>(lastRowSteps.begin()) * stepBytes;
            lastRowBytes = MaskOfOneRow<IndexT>{0, lastRowBegin, lastRowBegin + static_cast<IndexT>(lastRowSteps.count()) * stepBytes};
            lastRowWords = sourceWordsOf(lastRowOffset, lastRowBytes);
            lastRowFirstWord = source[lastRowWords.first];
            lastRowLastWord = source[lastRowWords.last];
        }

        uint4 value = chooseBytes(maskPacket(static_cast<uint4 *>(nullptr), firstRowBytes),
                                  bytesFrom(firstRowFirstWord, firstRowLastWord, firstRowWords.shift),
                                  padding);
        if constexpr (CrossesRows) {
            value = chooseBytes(maskPacket(static_cast<uint4 *>(nullptr), lastRowBytes),
                                bytesFrom(lastRowFirstWord, lastRowLastWord, lastRowWords.shift),
                                value);
        }
        destination[packet] = value;
        // Break before the increment so the 32-bit index cannot wrap on the last tile.
        if (totalPackets - tileStart <= tileStride) break;
        tileStart += tileStride;
    }
    if constexpr (CrossesRows) {
        if (blockIdx.x == 0 && threadIdx.x < tailBytes) {
            const IndexT byte = totalPackets * kPacketBytes + static_cast<IndexT>(threadIdx.x);
            const IndexT row = rowOfByte(byte);
            const IndexT byteInRow = byte - row * rowBytes;
            const PlanSteps<PlanT> steps = PlanSteps<PlanT>::load(rowPlanWords, row);
            const uint64_t offset = PlanSteps<PlanT>::loadSourceOffsetBytes(rowPlanWords, row);
            const IndexT byteInValid = byteInRow - static_cast<IndexT>(steps.begin()) * stepBytes;
            const bool fromSource = byteInValid < static_cast<IndexT>(steps.count()) * stepBytes;
            const uint8_t sourceByte = reinterpret_cast<const uint8_t *>(source)[fromSource ? offset + byteInValid : 0];
            reinterpret_cast<uint8_t *>(destination)[byte] =
                fromSource ? sourceByte : paddingPacket(static_cast<uint8_t *>(nullptr), paddingWord, byteInRow);
        }
    }
}

// The most a thread takes with the 32-bit packet index, once the batch is past kSpreadMaxBlocks blocks at that depth.
// A packet carries its index, its place in the row and its plan words until they are used, so 8 would pass 64
// registers.
constexpr uint32_t kMaxPacketsPerThread = 4;

template <typename PacketT, typename PlanT, uint32_t PacketsPerThread>
void launchPayloadWithPacketsPerThread(const PacketT *source,
                                       PacketT *destination,
                                       const uint64_t *rowPlanWords,
                                       uint64_t totalPackets,
                                       uint64_t packetsPerRow,
                                       uint64_t packetsPerStep,
                                       uint64_t paddingWord,
                                       cudaStream_t stream) {
    windowPacketsKernel<PacketT, uint32_t, RowOfPacket32, PlanT, PacketsPerThread>
        <<<blocksFor(totalPackets, PacketsPerThread), kThreadsPerBlock, 0, stream>>>(
            source,
            destination,
            rowPlanWords,
            static_cast<uint32_t>(totalPackets),
            static_cast<uint32_t>(packetsPerRow),
            RowOfPacket32{makeFastDivisor32(static_cast<uint32_t>(packetsPerRow))},
            static_cast<uint32_t>(packetsPerStep),
            paddingWord);
}

// A 64-bit-step plan (a window of more than 2^32 steps) and a batch whose packets or bytes pass 2^32 take the 64-bit
// packet index, one packet per thread: out of the ordinary range.
template <typename PacketT, typename PlanT>
void launchPayloadWithPacket(const uint8_t *source,
                             uint8_t *destination,
                             const uint64_t *rowPlanWords,
                             uint64_t logicalRows,
                             uint64_t rowBytes,
                             uint64_t stepBytes,
                             uint64_t paddingWord,
                             cudaStream_t stream) {
    const uint64_t packetsPerRow = rowBytes / sizeof(PacketT);
    const uint64_t packetsPerStep = stepBytes / sizeof(PacketT);
    THOR_THROW_IF_FALSE(packetsPerRow != 0 && logicalRows <= std::numeric_limits<uint64_t>::max() / packetsPerRow);
    const uint64_t totalPackets = logicalRows * packetsPerRow;
    const auto *sourcePackets = reinterpret_cast<const PacketT *>(source);
    auto *destinationPackets = reinterpret_cast<PacketT *>(destination);

    if constexpr (sizeof(PlanT) == sizeof(DeviceResidentWindowRowPlan32)) {
        const uint32_t packetsPerThread = spreadPacketsPerThread(totalPackets, kMaxPacketsPerThread);
        if (packetIndexFitsIn32Bits(totalPackets, packetsPerThread)) {
            if (packetsPerThread == 1) {
                launchPayloadWithPacketsPerThread<PacketT, PlanT, 1>(
                    sourcePackets, destinationPackets, rowPlanWords, totalPackets, packetsPerRow, packetsPerStep, paddingWord,
                    stream);
            } else if (packetsPerThread == 2) {
                launchPayloadWithPacketsPerThread<PacketT, PlanT, 2>(
                    sourcePackets, destinationPackets, rowPlanWords, totalPackets, packetsPerRow, packetsPerStep, paddingWord,
                    stream);
            } else {
                launchPayloadWithPacketsPerThread<PacketT, PlanT, kMaxPacketsPerThread>(
                    sourcePackets, destinationPackets, rowPlanWords, totalPackets, packetsPerRow, packetsPerStep, paddingWord,
                    stream);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    windowPacketsKernel<PacketT, uint64_t, RowOfPacket64, PlanT, 1><<<blocksFor(totalPackets, 1), kThreadsPerBlock, 0, stream>>>(
        sourcePackets, destinationPackets, rowPlanWords, totalPackets, packetsPerRow, RowOfPacket64{packetsPerRow}, packetsPerStep,
        paddingWord);
    CUDA_CHECK(cudaGetLastError());
}

template <typename PacketT, typename PlanT, uint32_t PacketsPerThread, bool CrossesRows>
void launchMaskWithPacketsPerThread(PacketT *destination,
                                    const uint64_t *rowPlanWords,
                                    uint64_t totalPackets,
                                    uint64_t windowLength,
                                    uint32_t tailBytes,
                                    cudaStream_t stream) {
    maskPacketsKernel<PacketT, uint32_t, RowOfPacket32, PlanT, PacketsPerThread, CrossesRows>
        <<<blocksFor(totalPackets, PacketsPerThread), kThreadsPerBlock, 0, stream>>>(
            destination,
            rowPlanWords,
            static_cast<uint32_t>(totalPackets),
            static_cast<uint32_t>(windowLength),
            RowOfPacket32{makeFastDivisor32(static_cast<uint32_t>(windowLength))},
            tailBytes);
}

// The 32-bit index needs every byte position of the batch to fit, not only every packet, and a row's length more (a
// step of the next row counted from the packet's first byte).
template <typename PacketT, typename PlanT, bool CrossesRows>
void launchMaskWithRows(PacketT *destination,
                        const uint64_t *rowPlanWords,
                        uint64_t totalBytes,
                        uint64_t windowLength,
                        cudaStream_t stream) {
    const uint64_t totalPackets = totalBytes / sizeof(PacketT);
    const uint32_t tailBytes = static_cast<uint32_t>(totalBytes % sizeof(PacketT));

    if constexpr (sizeof(PlanT) == sizeof(DeviceResidentWindowRowPlan32)) {
        const uint32_t packetsPerThread = spreadPacketsPerThread(totalPackets, kMaxPacketsPerThread);
        if (packetIndexFitsIn32Bits(totalPackets, packetsPerThread) &&
            totalBytes <= std::numeric_limits<uint32_t>::max() - windowLength) {
            if (packetsPerThread == 1) {
                launchMaskWithPacketsPerThread<PacketT, PlanT, 1, CrossesRows>(
                    destination, rowPlanWords, totalPackets, windowLength, tailBytes, stream);
            } else if (packetsPerThread == 2) {
                launchMaskWithPacketsPerThread<PacketT, PlanT, 2, CrossesRows>(
                    destination, rowPlanWords, totalPackets, windowLength, tailBytes, stream);
            } else {
                launchMaskWithPacketsPerThread<PacketT, PlanT, kMaxPacketsPerThread, CrossesRows>(
                    destination, rowPlanWords, totalPackets, windowLength, tailBytes, stream);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    maskPacketsKernel<PacketT, uint64_t, RowOfPacket64, PlanT, 1, CrossesRows>
        <<<blocksFor(totalPackets, 1), kThreadsPerBlock, 0, stream>>>(
            destination, rowPlanWords, totalPackets, windowLength, RowOfPacket64{windowLength}, tailBytes);
    CUDA_CHECK(cudaGetLastError());
}

template <typename PacketT, typename PlanT>
void launchMaskWithPacket(uint8_t *destination,
                          const uint64_t *rowPlanWords,
                          uint64_t logicalRows,
                          uint64_t windowLength,
                          cudaStream_t stream) {
    THOR_THROW_IF_FALSE(windowLength >= sizeof(PacketT) && logicalRows <= std::numeric_limits<uint64_t>::max() / windowLength);
    const uint64_t totalBytes = logicalRows * windowLength;
    auto *destinationPackets = reinterpret_cast<PacketT *>(destination);
    if (windowLength % sizeof(PacketT) == 0) {
        launchMaskWithRows<PacketT, PlanT, false>(destinationPackets, rowPlanWords, totalBytes, windowLength, stream);
    } else {
        launchMaskWithRows<PacketT, PlanT, true>(destinationPackets, rowPlanWords, totalBytes, windowLength, stream);
    }
}

// The 32-bit index needs every byte position of the batch to fit, and a row's length more (a byte of the next row
// counted from the packet's first byte). One packet per thread at every size.
template <typename PlanT, bool CrossesRows>
void launchPayloadBytesWithRows(const uint4 *source,
                                uint4 *destination,
                                const uint64_t *rowPlanWords,
                                uint64_t totalBytes,
                                uint64_t rowBytes,
                                uint64_t stepBytes,
                                uint64_t paddingWord,
                                cudaStream_t stream) {
    const uint64_t totalPackets = totalBytes / sizeof(uint4);
    const uint32_t tailBytes = static_cast<uint32_t>(totalBytes % sizeof(uint4));

    if constexpr (sizeof(PlanT) == sizeof(DeviceResidentWindowRowPlan32)) {
        if (packetIndexFitsIn32Bits(totalPackets, 1) && totalBytes <= std::numeric_limits<uint32_t>::max() - rowBytes) {
            windowBytesKernel<uint32_t, RowOfPacket32, PlanT, CrossesRows><<<blocksFor(totalPackets, 1), kThreadsPerBlock, 0, stream>>>(
                source,
                destination,
                rowPlanWords,
                static_cast<uint32_t>(totalPackets),
                static_cast<uint32_t>(rowBytes),
                RowOfPacket32{makeFastDivisor32(static_cast<uint32_t>(rowBytes))},
                static_cast<uint32_t>(stepBytes),
                paddingWord,
                tailBytes);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    windowBytesKernel<uint64_t, RowOfPacket64, PlanT, CrossesRows><<<blocksFor(totalPackets, 1), kThreadsPerBlock, 0, stream>>>(
        source, destination, rowPlanWords, totalPackets, rowBytes, RowOfPacket64{rowBytes}, stepBytes, paddingWord, tailBytes);
    CUDA_CHECK(cudaGetLastError());
}

// A payload whose steps or source offsets are on no wider boundary than a byte still moves 16-byte packets
// (windowBytesKernel) when both tensors start on one and a row holds one.
template <typename PlanT>
void launchPayloadBytes(const uint8_t *source,
                        uint64_t sourceBytes,
                        uint8_t *destination,
                        const uint64_t *rowPlanWords,
                        uint64_t logicalRows,
                        uint64_t rowBytes,
                        uint64_t stepBytes,
                        uint64_t paddingWord,
                        cudaStream_t stream) {
    THOR_THROW_IF_FALSE(sourceBytes != 0 && rowBytes >= sizeof(uint4) && logicalRows <= std::numeric_limits<uint64_t>::max() / rowBytes);
    const auto *sourceWords = reinterpret_cast<const uint4 *>(source);
    auto *destinationPackets = reinterpret_cast<uint4 *>(destination);
    const uint64_t totalBytes = logicalRows * rowBytes;
    if (rowBytes % sizeof(uint4) == 0) {
        launchPayloadBytesWithRows<PlanT, false>(
            sourceWords, destinationPackets, rowPlanWords, totalBytes, rowBytes, stepBytes, paddingWord, stream);
    } else {
        launchPayloadBytesWithRows<PlanT, true>(
            sourceWords, destinationPackets, rowPlanWords, totalBytes, rowBytes, stepBytes, paddingWord, stream);
    }
}

// Calls launch with a null pointer to the packet type of packetBytes bytes.
template <typename Launch>
void withPacketType(uint64_t packetBytes, Launch &&launch) {
    switch (packetBytes) {
        case 16:
            launch(static_cast<uint4 *>(nullptr));
            return;
        case 8:
            launch(static_cast<uint2 *>(nullptr));
            return;
        case 4:
            launch(static_cast<uint32_t *>(nullptr));
            return;
        case 2:
            launch(static_cast<uint16_t *>(nullptr));
            return;
        default:
            launch(static_cast<uint8_t *>(nullptr));
            return;
    }
}

// The largest batch whose payload moves in its aligned packets of 1 or 2 bytes; a larger one, with both tensors on a
// 16-byte boundary and rows of 16 bytes or more, moves in 16-byte packets (windowBytesKernel). Measured on the RTX 5090
// (patches 1542, 1545, 1546; plans in L2): 1- and 2-byte packets 0.02-0.17 us faster up to 256 KiB, level or slower at
// 400 KiB, slower from 512 KiB, and the 16-byte packets 0.4-3 us faster from 1.5 MiB, half-valid windows most.
constexpr uint64_t kNarrowPayloadMaxBatchBytes = 256 * 1024;

template <typename PlanT>
void launchWithPlan(bool materializeMask,
                    uint64_t packetBytes,
                    const uint8_t *source,
                    uint64_t sourceBytes,
                    uint8_t *destination,
                    const uint64_t *rowPlanWords,
                    uint64_t logicalRows,
                    uint64_t windowLength,
                    uint64_t stepBytes,
                    uint64_t paddingWord,
                    cudaStream_t stream) {
    const uint64_t rowBytes = windowLength * stepBytes;
    if (!materializeMask && packetBytes <= 2 && rowBytes >= sizeof(uint4) && logicalRows > kNarrowPayloadMaxBatchBytes / rowBytes &&
        reinterpret_cast<uintptr_t>(source) % sizeof(uint4) == 0 && reinterpret_cast<uintptr_t>(destination) % sizeof(uint4) == 0) {
        launchPayloadBytes<PlanT>(
            source, sourceBytes, destination, rowPlanWords, logicalRows, windowLength * stepBytes, stepBytes, paddingWord, stream);
        return;
    }
    withPacketType(packetBytes, [&](auto *packetType) {
        using PacketT = std::remove_pointer_t<decltype(packetType)>;
        if (materializeMask) {
            launchMaskWithPacket<PacketT, PlanT>(destination, rowPlanWords, logicalRows, windowLength, stream);
        } else {
            launchPayloadWithPacket<PacketT, PlanT>(
                source, destination, rowPlanWords, logicalRows, windowLength * stepBytes, stepBytes, paddingWord, stream);
        }
    });
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
    const TensorSpan &rowPlans,
    uint64_t logicalRows,
    const DeviceResidentWindowMaterializationSpec &spec,
    Tensor &destination,
    Stream &stream) {
    THOR_THROW_IF_FALSE(destination.isInitialized());
    rowPlans.validate("Compact resident window row-plan");
    const TensorPlacement placement = destination.getPlacement();
    validateTensor(rowPlans.tensor, placement, "window row-plan");
    // Session-owned plans are a UINT64 span of the consolidated selection-metadata upload allocation, placed on an
    // 8-byte boundary after the row indices. Direct kernel tests and standalone callers may pass whole byte tensors.
    THOR_THROW_IF_FALSE(
        rowPlans.getDataType() == DataType::UINT8 ||
        rowPlans.getDataType() == DataType::UINT64);
    THOR_THROW_IF_FALSE(reinterpret_cast<uintptr_t>(rowPlans.address()) % alignof(uint64_t) == 0);
    if (!spec.materializeMask) validateTensor(sourceStorage, placement, "source storage");

    THOR_THROW_IF_FALSE(spec.windowLength > 0);
    THOR_THROW_IF_FALSE(spec.sourceStepBytes > 0);
    THOR_THROW_IF_FALSE(spec.sourceOffsetAlignmentBytes > 0 &&
                        (spec.sourceOffsetAlignmentBytes & (spec.sourceOffsetAlignmentBytes - 1)) == 0);
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
    THOR_THROW_IF_FALSE(rowPlans.bytes() == logicalRows * planBytes);
    if (logicalRows == 0) return;

    const uint64_t elementBytes = dataTypeBytes(spec.dataType);
    THOR_THROW_IF_FALSE(spec.sourceStepBytes % elementBytes == 0);
    const uint64_t paddingWord = repeatedPadWord(padBits(spec.dataType, spec.padValue), elementBytes);
    const uint8_t *source = spec.materializeMask ? nullptr : static_cast<const uint8_t *>(sourceStorage.getMemPtr());
    uint8_t *destinationBytes = static_cast<uint8_t *>(destination.getMemPtr());
    const uint64_t *rowPlanWords = static_cast<const uint64_t *>(rowPlans.address());
    const uint64_t sourceBytes = spec.materializeMask ? 0 : sourceStorage.getArraySizeInBytes();

    // The widest packet, up to 16 bytes. A mask: on whose boundary the batch starts, and no longer than a row. A
    // payload: on whose boundaries everything the kernel addresses starts, which is the steps (so the rows too, and
    // the edges of the valid steps), the plans' source offsets, the source and the batch; and no wider than the
    // source, of which a padding packet reads the first packet. A large payload for which that is 1 or 2 bytes is
    // launched with 16-byte packets all the same where it can be (launchWithPlan).
    uint64_t alignment = reinterpret_cast<uintptr_t>(destinationBytes) | 16;
    if (!spec.materializeMask) {
        alignment |= spec.sourceStepBytes | spec.sourceOffsetAlignmentBytes | reinterpret_cast<uintptr_t>(source);
    }
    uint64_t packetBytes = alignment & (~alignment + 1);
    const uint64_t packetLimitBytes = spec.materializeMask ? spec.windowLength : sourceBytes;
    while (packetBytes > packetLimitBytes) packetBytes /= 2;

    if (usePlan32) {
        launchWithPlan<DeviceResidentWindowRowPlan32>(
            spec.materializeMask, packetBytes, source, sourceBytes, destinationBytes, rowPlanWords, logicalRows, spec.windowLength,
            spec.sourceStepBytes, paddingWord, stream.getStream());
    } else {
        launchWithPlan<DeviceResidentWindowRowPlan64>(
            spec.materializeMask, packetBytes, source, sourceBytes, destinationBytes, rowPlanWords, logicalRows, spec.windowLength,
            spec.sourceStepBytes, paddingWord, stream.getStream());
    }
}
