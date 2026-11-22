#include "DeepLearning/Implementation/Data/Residency/DeviceResidentRaggedMaterializationKernel.h"

#include "DeepLearning/Implementation/Data/Residency/DeviceResidentPacketGrid.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorPlacement;
using ThorImplementation::TensorSpan;
using namespace DeviceResidentPacketGrid;

namespace {

// The index of a record's first value: the 8 bytes at referenceByte of the record store, which is at any byte (records
// are packed). The store's 8-byte words are aligned, so the bytes lie in one word or run into the next. The next word
// is inside the store either way: the record's value count follows the reference.
__device__ __forceinline__ uint64_t loadFirstValue(const uint64_t *__restrict__ recordWords, uint64_t referenceByte) {
    const uint64_t word = referenceByte / sizeof(uint64_t);
    return bytesFrom(recordWords[word], recordWords[word + 1], static_cast<uint32_t>(referenceByte % sizeof(uint64_t)));
}

/**
 * Materialize the rows of a ragged field. Batch row r takes the values of record rowIndices[r]: their count is
 * offsets[r + 1] - offsets[r], they start at the value the record's reference names, and they go to value offsets[r]
 * of the batch. A value is packetsPerValue packets, and both value stores start on a packet boundary, so every row
 * of both is whole aligned packets (as in the row gather, DeviceResidentRowGatherKernel.cu).
 *
 * Rows differ in length, so the grid is over rows: a block takes 256 >> lanesPerRowLog2 consecutive rows and gives
 * each 1 << lanesPerRowLog2 consecutive threads, its lanes. In one pass the lanes of a row take its next
 * lanes x PacketsPerLane packets, consecutive lanes consecutive packets, PacketsPerLane packets each; a row longer
 * than that takes more passes. The launcher sizes a pass to the store's mean row.
 *
 * A thread issues its index and offset loads, then the record load (the first use of the index), then the value
 * loads of a pass (the first use of the record) before it stores any. No load is under a test: a packet past the
 * row's end reads packet 0 of the value store (the launcher requires one) and is not stored.
 */
template <typename PacketT, typename OffsetT, typename LengthT, uint32_t PacketsPerLane>
__global__ void __launch_bounds__(kThreadsPerBlock)
    raggedPacketsKernel(const uint64_t *__restrict__ recordWords,
                        const PacketT *__restrict__ values,
                        PacketT *__restrict__ destination,
                        const uint64_t *__restrict__ rowIndices,
                        const OffsetT *__restrict__ offsets,
                        uint64_t logicalRows,
                        uint64_t recordSizeBytes,
                        uint64_t referenceOffsetBytes,
                        LengthT packetsPerValue,
                        uint32_t lanesPerRowLog2) {
    const uint32_t lanesPerRow = 1U << lanesPerRowLog2;
    const uint32_t rowsPerBlock = kThreadsPerBlock >> lanesPerRowLog2;
    const LengthT lane = static_cast<LengthT>(threadIdx.x & (lanesPerRow - 1U));
    const LengthT packetsPerPass = static_cast<LengthT>(lanesPerRow) * PacketsPerLane;
    const uint64_t rowStride = static_cast<uint64_t>(gridDim.x) * rowsPerBlock;
    for (uint64_t row = static_cast<uint64_t>(blockIdx.x) * rowsPerBlock + (threadIdx.x >> lanesPerRowLog2); row < logicalRows;
         row += rowStride) {
        const uint64_t index = rowIndices[row];
        const OffsetT firstOffset = offsets[row];
        const OffsetT endOffset = offsets[row + 1];
        const uint64_t firstValue = loadFirstValue(recordWords, index * recordSizeBytes + referenceOffsetBytes);

        const LengthT rowPackets = static_cast<LengthT>(endOffset - firstOffset) * packetsPerValue;
        const LengthT destinationPacket = static_cast<LengthT>(firstOffset) * packetsPerValue;
        const uint64_t sourcePacket = firstValue * packetsPerValue;
        LengthT pass = 0;
        while (pass < rowPackets) {
            // Value loads only: no loaded packet is read before the stores.
            LengthT packetInRow[PacketsPerLane];
            bool inRow[PacketsPerLane];
            PacketT value[PacketsPerLane];
#pragma unroll
            for (uint32_t k = 0; k < PacketsPerLane; ++k) {
                packetInRow[k] = pass + static_cast<LengthT>(k * lanesPerRow) + lane;
                inRow[k] = packetInRow[k] < rowPackets;
                value[k] = values[inRow[k] ? sourcePacket + packetInRow[k] : 0];
            }
#pragma unroll
            for (uint32_t k = 0; k < PacketsPerLane; ++k) {
                if (inRow[k]) destination[destinationPacket + packetInRow[k]] = value[k];
            }
            // Break before the increment so the 32-bit packet count cannot wrap on the last pass.
            if (rowPackets - pass <= packetsPerPass) break;
            pass += packetsPerPass;
        }
    }
}

// The n-th set bit of a 4-bit count, counting from its lowest bit up (0 if it has no such bit).
__device__ __forceinline__ uint32_t nthSetBit(uint32_t fourBits, uint32_t n) {
    uint32_t found = 0;
    uint32_t seen = 0;
#pragma unroll
    for (uint32_t bit = 0; bit < 4; ++bit) {
        const uint32_t isSet = (fourBits >> bit) & 1U;
        found = isSet != 0 && seen == n ? bit : found;
        seen += isSet;
    }
    return found;
}

// Store the packets of a lane's pass that are in its row: the 16 bytes from byte packetShift of the packet's first word
// on.
template <typename LengthT, uint32_t PacketsPerLane>
__device__ __forceinline__ void storeShiftedPackets(uint4 *__restrict__ destination,
                                                    LengthT firstPacket,
                                                    const LengthT (&packet)[PacketsPerLane],
                                                    const bool (&inRow)[PacketsPerLane],
                                                    const uint4 (&firstWord)[PacketsPerLane],
                                                    const uint4 (&lastWord)[PacketsPerLane],
                                                    uint32_t packetShift) {
#pragma unroll
    for (uint32_t k = 0; k < PacketsPerLane; ++k) {
        if (inRow[k]) destination[firstPacket + packet[k]] = bytesFrom(firstWord[k], lastWord[k], packetShift);
    }
}

/**
 * Materialize the rows of a ragged field in 16-byte packets whatever byte its rows start and end at (the launcher
 * takes it for large batches of values on a 1- or 2-byte boundary). The grid, the lanes, the passes and the load
 * order are raggedPacketsKernel's.
 *
 * A row is the batch bytes [firstByte, endByte), which start and end at any byte, and every store is an aligned
 * 1, 2, 4, 8 or 16 bytes of one row. A row that reaches a 16-byte boundary of the batch is its head (the bytes
 * before its first boundary, at most 15), its whole 16-byte packets, and its tail (the bytes after its last
 * boundary, at most 15). The head is one piece per set bit of its length, the smaller pieces first: each piece ends
 * where the next larger boundary begins, so each is aligned. The tail is one piece per set bit of its length, the
 * larger pieces first. A row inside one packet is split the same way at the most aligned byte it holds.
 *
 * The packets are what the lanes share out, PacketsPerLane to a lane in a pass, as in raggedPacketsKernel. The
 * pieces (8 a row at most) ride along: in a pass a lane also takes one piece if the row has one left for it, and
 * the launcher sizes the lanes to the packets.
 *
 * The source bytes start at any byte of the value store and are read as aligned words and shifted (bytesFrom). A
 * packet's are the 16-byte word that holds the first of them and, unless they are that word, the next; every packet
 * of a row starts at the same byte of its word. A piece's are the 8-byte words that hold its first and its last
 * byte, and it stores the first bytes of the result. A packet past the row's end reads word 0 of the value store
 * and is not stored. A word may end up to 15 bytes past the store, inside the padding every tensor is allocated
 * with.
 *
 * In a pass a lane issues its packets' loads, under no test, before anything waits for memory. A lane with a piece
 * then loads and stores it, with the packets' words on their way, and stores its packets. The packets' stores are
 * written on both sides of the piece's test, so their loads have a reader on each side of it and stay ahead of it.
 * With one reader, behind the test, the compiler moved the loads down to it, behind the piece's store, and such a
 * lane waited for memory twice (the SASS of the one-packet-per-lane kernel of patch 1552).
 */
template <typename OffsetT, typename LengthT, uint32_t PacketsPerLane>
__global__ void __launch_bounds__(kThreadsPerBlock)
    raggedBytesKernel(const uint64_t *__restrict__ recordWords,
                      const uint4 *__restrict__ values,
                      uint4 *__restrict__ destination,
                      const uint64_t *__restrict__ rowIndices,
                      const OffsetT *__restrict__ offsets,
                      uint64_t logicalRows,
                      uint64_t recordSizeBytes,
                      uint64_t referenceOffsetBytes,
                      LengthT valueBytes,
                      uint32_t lanesPerRowLog2) {
    constexpr LengthT kPacketBytes = static_cast<LengthT>(sizeof(uint4));
    const uint32_t lanesPerRow = 1U << lanesPerRowLog2;
    const uint32_t rowsPerBlock = kThreadsPerBlock >> lanesPerRowLog2;
    const uint32_t lane = threadIdx.x & (lanesPerRow - 1U);
    const LengthT packetsPerPass = static_cast<LengthT>(lanesPerRow) * PacketsPerLane;
    const uint64_t rowStride = static_cast<uint64_t>(gridDim.x) * rowsPerBlock;
    const uint64_t *const valueWords = reinterpret_cast<const uint64_t *>(values);
    uint8_t *const destinationBytes = reinterpret_cast<uint8_t *>(destination);
    for (uint64_t row = static_cast<uint64_t>(blockIdx.x) * rowsPerBlock + (threadIdx.x >> lanesPerRowLog2); row < logicalRows;
         row += rowStride) {
        const uint64_t index = rowIndices[row];
        const OffsetT firstOffset = offsets[row];
        const OffsetT endOffset = offsets[row + 1];
        const uint64_t firstValue = loadFirstValue(recordWords, index * recordSizeBytes + referenceOffsetBytes);

        const LengthT firstByte = static_cast<LengthT>(firstOffset) * valueBytes;
        const LengthT endByte = static_cast<LengthT>(endOffset) * valueBytes;
        const uint64_t sourceByte = firstValue * valueBytes;
        // Where the head ends and the tail begins: the row's first and last 16-byte boundaries, or, for a row
        // inside one packet, both at its most aligned byte. The first and end bytes of such a row differ only in
        // their low four bits; the highest bit they differ in is that byte's alignment.
        const LengthT firstBoundary = (firstByte + (kPacketBytes - 1)) & ~(kPacketBytes - 1);
        const uint32_t differing = (static_cast<uint32_t>(firstByte ^ endByte) & 15U) | 1U;
        const uint32_t splitLog2 = differing >= 8U ? 3U : differing >= 4U ? 2U : differing >= 2U ? 1U : 0U;
        const LengthT split = endByte & ~((static_cast<LengthT>(1) << splitLog2) - 1);
        const bool reachesBoundary = firstBoundary <= endByte;
        const LengthT headEnd = reachesBoundary ? firstBoundary : split;
        const LengthT tailBegin = reachesBoundary ? endByte & ~(kPacketBytes - 1) : split;
        const uint32_t headBytes = static_cast<uint32_t>(headEnd - firstByte);
        const uint32_t tailBytes = static_cast<uint32_t>(endByte - tailBegin);
        const uint32_t headPieces = static_cast<uint32_t>(__popc(headBytes));
        const uint32_t rowPieces = headPieces + static_cast<uint32_t>(__popc(tailBytes));
        // The row's packets: the first is packet firstPacket of the batch, and its bytes start at byte packetShift
        // of word firstSourceWord of the value store.
        const LengthT rowPackets = (tailBegin - headEnd) / kPacketBytes;
        const LengthT firstPacket = headEnd / kPacketBytes;
        const uint64_t firstSourceWord = (sourceByte + headBytes) / sizeof(uint4);
        const uint32_t packetShift = static_cast<uint32_t>(sourceByte + headBytes) & 15U;
        LengthT pass = 0;
        // The piece lane 0 takes in this pass.
        uint32_t firstPiece = 0;
        while (pass < rowPackets || firstPiece < rowPieces) {
            LengthT packet[PacketsPerLane];
            bool inRow[PacketsPerLane];
            uint4 firstWord[PacketsPerLane];
            uint4 lastWord[PacketsPerLane];
#pragma unroll
            for (uint32_t k = 0; k < PacketsPerLane; ++k) {
                packet[k] = pass + static_cast<LengthT>(k * lanesPerRow + lane);
                inRow[k] = packet[k] < rowPackets;
                const uint64_t word = inRow[k] ? firstSourceWord + packet[k] : 0;
                firstWord[k] = values[word];
                lastWord[k] = values[inRow[k] && packetShift != 0 ? word + 1 : word];
            }
            const uint32_t piece = firstPiece + lane;
            if (piece < rowPieces) {
                const bool inHead = piece < headPieces;
                const uint32_t pieceLog2 = nthSetBit(inHead ? headBytes : tailBytes, inHead ? piece : piece - headPieces);
                // The batch byte the piece starts at: a head piece after the head's smaller pieces, a tail piece
                // after the tail's larger ones.
                const LengthT pieceByte = inHead ? firstByte + (headBytes & ((1U << pieceLog2) - 1U))
                                                 : tailBegin + (tailBytes & ~((2U << pieceLog2) - 1U));
                const uint64_t pieceSourceByte = sourceByte + (pieceByte - firstByte);
                const uint64_t bytes = bytesFrom(valueWords[pieceSourceByte / sizeof(uint64_t)],
                                                 valueWords[(pieceSourceByte + ((1U << pieceLog2) - 1U)) / sizeof(uint64_t)],
                                                 static_cast<uint32_t>(pieceSourceByte) & 7U);
                if (pieceLog2 == 3) {
                    *reinterpret_cast<uint64_t *>(destinationBytes + pieceByte) = bytes;
                } else if (pieceLog2 == 2) {
                    *reinterpret_cast<uint32_t *>(destinationBytes + pieceByte) = static_cast<uint32_t>(bytes);
                } else if (pieceLog2 == 1) {
                    *reinterpret_cast<uint16_t *>(destinationBytes + pieceByte) = static_cast<uint16_t>(bytes);
                } else {
                    destinationBytes[pieceByte] = static_cast<uint8_t>(bytes);
                }
                storeShiftedPackets(destination, firstPacket, packet, inRow, firstWord, lastWord, packetShift);
            } else {
                storeShiftedPackets(destination, firstPacket, packet, inRow, firstWord, lastWord, packetShift);
            }
            pass += packetsPerPass;
            firstPiece += lanesPerRow;
        }
    }
}

// The most packets a lane takes in a pass, once the batch is past kSpreadMaxBlocks blocks at that depth: aligned
// packets, and shifted ones (a shifted packet holds two 16-byte words until its store).
constexpr uint32_t kMaxPacketsPerLane = 4;
constexpr uint32_t kMaxShiftedPacketsPerLane = 2;

// The lanes of a row, as a power of two up to a whole block: the fewest whose pass holds the store's mean row.
uint32_t lanesPerRowLog2For(uint64_t meanRowPackets, uint32_t packetsPerLane) {
    uint32_t lanesPerRowLog2 = 0;
    while ((kThreadsPerBlock >> lanesPerRowLog2) > 1 && (static_cast<uint64_t>(packetsPerLane) << lanesPerRowLog2) < meanRowPackets) {
        ++lanesPerRowLog2;
    }
    return lanesPerRowLog2;
}

// Blocks of a launch over logicalRows rows: one per kThreadsPerBlock >> lanesPerRowLog2 rows, up to kMaxBlocks.
uint32_t blocksForRows(uint64_t logicalRows, uint32_t lanesPerRowLog2) {
    const uint64_t rowsPerBlock = kThreadsPerBlock >> lanesPerRowLog2;
    return static_cast<uint32_t>(std::min(kMaxBlocks, (logicalRows + rowsPerBlock - 1) / rowsPerBlock));
}

// Everything a launch is given but its packet, offset and length types.
struct RaggedLaunch {
    const uint64_t *recordWords;
    const uint8_t *values;
    uint8_t *destination;
    const uint64_t *rowIndices;
    const void *offsets;
    uint64_t logicalRows;
    uint64_t recordSizeBytes;
    uint64_t referenceOffsetBytes;
    uint64_t valueBytes;
    // The store's mean row and the batch tensor's capacity, in bytes.
    uint64_t meanRowBytes;
    uint64_t destinationBytes;
    cudaStream_t stream;
};

template <typename PacketT, typename OffsetT, typename LengthT, uint32_t PacketsPerLane>
void launchWithPacketsPerLane(const RaggedLaunch &launch, uint32_t lanesPerRowLog2) {
    raggedPacketsKernel<PacketT, OffsetT, LengthT, PacketsPerLane>
        <<<blocksForRows(launch.logicalRows, lanesPerRowLog2), kThreadsPerBlock, 0, launch.stream>>>(
            launch.recordWords,
            reinterpret_cast<const PacketT *>(launch.values),
            reinterpret_cast<PacketT *>(launch.destination),
            launch.rowIndices,
            static_cast<const OffsetT *>(launch.offsets),
            launch.logicalRows,
            launch.recordSizeBytes,
            launch.referenceOffsetBytes,
            static_cast<LengthT>(launch.valueBytes / sizeof(PacketT)),
            lanesPerRowLog2);
    CUDA_CHECK(cudaGetLastError());
}

// A lane takes the fewest packets (1, 2, 4) that keep the launch within kSpreadMaxBlocks blocks, as a thread does in
// the row gather. A row's packets are counted in 32 bits when the batch tensor holds few enough of them that a pass
// cannot wrap; a larger batch tensor takes 64-bit counts, one packet per lane: out of the ordinary range.
// The store's mean row in packets of packetBytes.
uint64_t meanRowPacketsOf(const RaggedLaunch &launch, uint64_t packetBytes) {
    return std::max<uint64_t>(1, (launch.meanRowBytes + packetBytes - 1) / packetBytes);
}

template <typename PacketT, typename OffsetT>
void launchWithPacket(const RaggedLaunch &launch) {
    const uint64_t meanRowPackets = meanRowPacketsOf(launch, sizeof(PacketT));
    const uint64_t destinationPackets = launch.destinationBytes / sizeof(PacketT);
    if (destinationPackets > std::numeric_limits<uint32_t>::max() - kThreadsPerBlock * kMaxPacketsPerLane) {
        launchWithPacketsPerLane<PacketT, OffsetT, uint64_t, 1>(launch, lanesPerRowLog2For(meanRowPackets, 1));
        return;
    }
    uint32_t packetsPerLane = 1;
    while (packetsPerLane < kMaxPacketsPerLane &&
           blocksForRows(launch.logicalRows, lanesPerRowLog2For(meanRowPackets, packetsPerLane)) > kSpreadMaxBlocks) {
        packetsPerLane *= 2;
    }
    const uint32_t lanesPerRowLog2 = lanesPerRowLog2For(meanRowPackets, packetsPerLane);
    if (packetsPerLane == 1) {
        launchWithPacketsPerLane<PacketT, OffsetT, uint32_t, 1>(launch, lanesPerRowLog2);
    } else if (packetsPerLane == 2) {
        launchWithPacketsPerLane<PacketT, OffsetT, uint32_t, 2>(launch, lanesPerRowLog2);
    } else {
        launchWithPacketsPerLane<PacketT, OffsetT, uint32_t, kMaxPacketsPerLane>(launch, lanesPerRowLog2);
    }
}

template <typename OffsetT, typename LengthT, uint32_t PacketsPerLane>
void launchBytesWithPacketsPerLane(const RaggedLaunch &launch, uint32_t lanesPerRowLog2) {
    raggedBytesKernel<OffsetT, LengthT, PacketsPerLane>
        <<<blocksForRows(launch.logicalRows, lanesPerRowLog2), kThreadsPerBlock, 0, launch.stream>>>(
            launch.recordWords,
            reinterpret_cast<const uint4 *>(launch.values),
            reinterpret_cast<uint4 *>(launch.destination),
            launch.rowIndices,
            static_cast<const OffsetT *>(launch.offsets),
            launch.logicalRows,
            launch.recordSizeBytes,
            launch.referenceOffsetBytes,
            static_cast<LengthT>(launch.valueBytes),
            lanesPerRowLog2);
    CUDA_CHECK(cudaGetLastError());
}

// Rows that start and end at any byte: 16-byte packets with the rows' heads and tails in aligned pieces
// (raggedBytesKernel). A pass is sized to the mean row's packets, with no fewer lanes than the mean row has pieces
// (a head and a tail are 2 pieces each on average, and a row has no more pieces than bytes). A lane takes two
// packets where that makes a launch past kSpreadMaxBlocks blocks smaller. Batch bytes are counted in 32 bits when
// the batch tensor's bytes, rounded up to a packet, fit, and otherwise in 64 bits, one packet per lane.
template <typename OffsetT>
void launchBytes(const RaggedLaunch &launch) {
    const uint64_t meanRowPackets = launch.meanRowBytes / sizeof(uint4);
    const uint64_t meanRowPieces = std::min<uint64_t>(launch.meanRowBytes, 4);
    const auto lanesPerRowLog2At = [&](uint32_t packetsPerLane) {
        return std::max(lanesPerRowLog2For(meanRowPackets, packetsPerLane), lanesPerRowLog2For(meanRowPieces, 1));
    };
    if (launch.destinationBytes > std::numeric_limits<uint32_t>::max() - sizeof(uint4)) {
        launchBytesWithPacketsPerLane<OffsetT, uint64_t, 1>(launch, lanesPerRowLog2At(1));
        return;
    }
    if (blocksForRows(launch.logicalRows, lanesPerRowLog2At(1)) <= kSpreadMaxBlocks ||
        lanesPerRowLog2At(kMaxShiftedPacketsPerLane) == lanesPerRowLog2At(1)) {
        launchBytesWithPacketsPerLane<OffsetT, uint32_t, 1>(launch, lanesPerRowLog2At(1));
    } else {
        launchBytesWithPacketsPerLane<OffsetT, uint32_t, kMaxShiftedPacketsPerLane>(launch, lanesPerRowLog2At(kMaxShiftedPacketsPerLane));
    }
}

template <typename OffsetT>
void launchWithOffsets(const RaggedLaunch &launch) {
    // The widest packet, up to 16 bytes, on whose boundaries every value of both stores starts.
    const uint64_t alignment =
        launch.valueBytes | reinterpret_cast<uintptr_t>(launch.values) | reinterpret_cast<uintptr_t>(launch.destination) | 16;
    const uint64_t packetBytes = alignment & (~alignment + 1);
    // Packets of 1 and 2 bytes are the faster way for a batch that the most of them a lane takes keep within
    // kSpreadMaxBlocks blocks. A larger batch of them is bound by its instructions, and where both stores start on a
    // 16-byte boundary its rows move in 16-byte packets instead (measured on both sides of the bound, patch 1553).
    const bool storesStartOnWords = reinterpret_cast<uintptr_t>(launch.values) % sizeof(uint4) == 0 &&
                                    reinterpret_cast<uintptr_t>(launch.destination) % sizeof(uint4) == 0;
    if (packetBytes < sizeof(uint32_t) && storesStartOnWords &&
        blocksForRows(launch.logicalRows, lanesPerRowLog2For(meanRowPacketsOf(launch, packetBytes), kMaxPacketsPerLane)) >
            kSpreadMaxBlocks) {
        launchBytes<OffsetT>(launch);
        return;
    }
    switch (packetBytes) {
        case 16:
            launchWithPacket<uint4, OffsetT>(launch);
            return;
        case 8:
            launchWithPacket<uint2, OffsetT>(launch);
            return;
        case 4:
            launchWithPacket<uint32_t, OffsetT>(launch);
            return;
        case 2:
            launchWithPacket<uint16_t, OffsetT>(launch);
            return;
        default:
            launchWithPacket<uint8_t, OffsetT>(launch);
            return;
    }
}

}  // namespace

void launchDeviceResidentRaggedMaterializationKernel(
    const Tensor &recordStorage,
    const Tensor &packedValuesStorage,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t referenceOffsetBytes,
    uint64_t storedValueCount,
    uint64_t valueBytes,
    uint64_t logicalRows,
    Tensor &destinationValues,
    const Tensor &destinationOffsets,
    const TensorSpan &rowIndicesDevice,
    Stream &stream) {
    THOR_THROW_IF_FALSE(recordStorage.isInitialized());
    THOR_THROW_IF_FALSE(packedValuesStorage.isInitialized() || storedValueCount == 0);
    THOR_THROW_IF_FALSE(destinationValues.isInitialized());
    THOR_THROW_IF_FALSE(destinationOffsets.isInitialized());
    rowIndicesDevice.validate("Device resident ragged materialization row indices");
    THOR_THROW_IF_FALSE(
        recordStorage.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    THOR_THROW_IF_FALSE(!packedValuesStorage.isInitialized() ||
                        packedValuesStorage.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(destinationValues.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(destinationOffsets.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(rowIndicesDevice.getPlacement() == recordStorage.getPlacement());
    THOR_THROW_IF_FALSE(recordStorage.getDataType() == DataType::UINT8);
    THOR_THROW_IF_FALSE(rowIndicesDevice.getDataType() == DataType::UINT64);
    THOR_THROW_IF_FALSE(recordStorage.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(destinationOffsets.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(!destinationValues.getDimensions().empty());
    THOR_THROW_IF_FALSE(numExamples > 0);
    THOR_THROW_IF_FALSE(recordSizeBytes >= 2 * sizeof(uint64_t));
    THOR_THROW_IF_FALSE(referenceOffsetBytes <= recordSizeBytes - 2 * sizeof(uint64_t));
    THOR_THROW_IF_FALSE(valueBytes > 0);
    THOR_THROW_IF_FALSE(
        storedValueCount <= std::numeric_limits<uint64_t>::max() / valueBytes);
    THOR_THROW_IF_FALSE(
        (packedValuesStorage.isInitialized() ? packedValuesStorage.getArraySizeInBytes() : 0) ==
        storedValueCount * valueBytes);
    THOR_THROW_IF_FALSE(
        recordStorage.getArraySizeInBytes() == numExamples * recordSizeBytes);
    // The kernel reads a reference as aligned 8-byte words of the record store.
    THOR_THROW_IF_FALSE(reinterpret_cast<uintptr_t>(recordStorage.getMemPtr()) % alignof(uint64_t) == 0);

    const uint64_t batchSize = rowIndicesDevice.count;
    THOR_THROW_IF_FALSE(batchSize > 0);
    THOR_THROW_IF_FALSE(logicalRows >= 1 && logicalRows <= batchSize);
    THOR_THROW_IF_FALSE(destinationOffsets.getDimensions().front() == batchSize + 1);
    THOR_THROW_IF_FALSE(destinationValues.getDimensions().front() > 0);
    const DataType offsetType = destinationOffsets.getDataType();
    if (offsetType != DataType::UINT32 && offsetType != DataType::UINT64) {
        throw std::runtime_error("Device resident ragged offsets must use canonical UINT32 or UINT64 dtype.");
    }
    if (offsetType == DataType::UINT32) {
        THOR_THROW_IF_FALSE(destinationValues.getDimensions().front() <= std::numeric_limits<uint32_t>::max());
    }
    if (storedValueCount == 0) return;

    RaggedLaunch launch;
    launch.recordWords = static_cast<const uint64_t *>(recordStorage.getMemPtr());
    launch.values = static_cast<const uint8_t *>(packedValuesStorage.getMemPtr());
    launch.destination = static_cast<uint8_t *>(destinationValues.getMemPtr());
    launch.rowIndices = rowIndicesDevice.data<uint64_t>();
    launch.offsets = destinationOffsets.getMemPtr();
    launch.logicalRows = logicalRows;
    launch.recordSizeBytes = recordSizeBytes;
    launch.referenceOffsetBytes = referenceOffsetBytes;
    launch.valueBytes = valueBytes;
    const uint64_t storedValueBytes = storedValueCount * valueBytes;
    launch.meanRowBytes = storedValueBytes / numExamples + (storedValueBytes % numExamples != 0 ? 1 : 0);
    launch.destinationBytes = destinationValues.getArraySizeInBytes();
    launch.stream = stream.getStream();
    if (offsetType == DataType::UINT32) {
        launchWithOffsets<uint32_t>(launch);
    } else {
        launchWithOffsets<uint64_t>(launch);
    }
}
