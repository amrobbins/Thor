#include "DeepLearning/Implementation/Data/Residency/DeviceResidentRowGatherKernel.h"

#include "DeepLearning/Implementation/Data/Residency/DeviceResidentPacketGrid.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cstdint>
#include <limits>

using namespace DeviceResidentPacketGrid;

namespace {

/**
 * Gather rows as one flat array of packets. Every source and destination row starts on a PacketT boundary (the width
 * divides the row bytes, the source stride and both base addresses), so each row is a whole number of packets and
 * every access is a full aligned packet: no tails and no per-row width choice. Consecutive threads take consecutive
 * packets, so a warp reads a contiguous run of a row and writes a contiguous run of the batch. Source row i starts
 * sourceRowPackets * i packets into source; destination rows are back to back.
 *
 * A load holds a warp back only where its result is first used. So each thread issues the row-index loads of all its
 * PacketsPerThread packets before it reads any index, then all their source loads before it stores any packet: a tile
 * waits on memory twice (the indices, then the rows), not twice per packet.
 * That order holds only while no load and no store is under a test of its own. A tested access needs a predicate
 * register until its store, there are seven, and past that the compiler finishes one packet (load, then store) to
 * free one before it issues the other loads. So:
 *  - a packet past the end is the last packet again: every thread that takes it reads and writes the same bytes;
 *  - a row index outside the source reads row 0 (the launcher requires a source row);
 *  - the stores are untested when all of the thread's row indices are inside the source, which is one test per
 *    thread. Only a thread with an index outside the source tests each store, and leaves that row untouched.
 */
template <typename PacketT, typename IndexT, typename RowOfPacket, uint32_t PacketsPerThread>
__global__ void __launch_bounds__(kThreadsPerBlock)
    gatherPacketsKernel(const PacketT *__restrict__ source,
                        PacketT *__restrict__ destination,
                        const uint64_t *__restrict__ rowIndices,
                        IndexT totalPackets,
                        IndexT packetsPerRow,
                        RowOfPacket rowOfPacket,
                        uint64_t sourceRows,
                        uint64_t sourceRowPackets) {
    constexpr IndexT tilePackets = static_cast<IndexT>(kThreadsPerBlock) * PacketsPerThread;
    const IndexT tileStride = static_cast<IndexT>(gridDim.x) * tilePackets;
    const IndexT lastPacket = totalPackets - 1;
    IndexT tileStart = static_cast<IndexT>(blockIdx.x) * tilePackets;
    while (tileStart < totalPackets) {
        // Index loads only: nothing here reads a loaded index, so the thread's indices are in flight together.
        IndexT packet[PacketsPerThread];
        IndexT packetInRow[PacketsPerThread];
        uint64_t sourceRow[PacketsPerThread];
#pragma unroll
        for (uint32_t k = 0; k < PacketsPerThread; ++k) {
            const IndexT ownPacket = tileStart + static_cast<IndexT>(k * kThreadsPerBlock + threadIdx.x);
            packet[k] = ownPacket < totalPackets ? ownPacket : lastPacket;
            const IndexT row = rowOfPacket(packet[k]);
            packetInRow[k] = packet[k] - row * packetsPerRow;
            sourceRow[k] = rowIndices[row];
        }
        // Source loads: the first use of the indices. No loaded packet is read before the stores.
        PacketT value[PacketsPerThread];
        bool everyRowInSource = true;
#pragma unroll
        for (uint32_t k = 0; k < PacketsPerThread; ++k) {
            const bool inSource = sourceRow[k] < sourceRows;
            everyRowInSource &= inSource;
            value[k] = source[(inSource ? sourceRow[k] : 0) * sourceRowPackets + packetInRow[k]];
        }
        if (everyRowInSource) {
#pragma unroll
            for (uint32_t k = 0; k < PacketsPerThread; ++k) destination[packet[k]] = value[k];
        } else {
#pragma unroll
            for (uint32_t k = 0; k < PacketsPerThread; ++k) {
                if (sourceRow[k] < sourceRows) destination[packet[k]] = value[k];
            }
        }
        // Break before the increment so the 32-bit index cannot wrap on the last tile.
        if (totalPackets - tileStart <= tileStride) break;
        tileStart += tileStride;
    }
}

// The most a thread takes, once the batch is past kSpreadMaxBlocks blocks at that depth and bound by bandwidth: 64
// bytes of 16-byte packets, 8 narrower packets. The 64-bit packet index (over 2^32 packets in one batch) takes 4 of
// any width: its indices and row offsets are twice as wide, and 8 would pass 64 registers.
template <typename PacketT, typename IndexT>
constexpr uint32_t maxPacketsPerThread() {
    return sizeof(PacketT) >= 16 || sizeof(IndexT) > 4 ? 4 : 8;
}

template <typename PacketT, uint32_t PacketsPerThread>
void launchWithPacketsPerThread(const PacketT *source,
                                PacketT *destination,
                                const uint64_t *rowIndices,
                                uint64_t totalPackets,
                                uint64_t packetsPerRow,
                                uint64_t sourceRows,
                                uint64_t sourceRowPackets,
                                uint32_t blocks,
                                cudaStream_t stream) {
    gatherPacketsKernel<PacketT, uint32_t, RowOfPacket32, PacketsPerThread><<<blocks, kThreadsPerBlock, 0, stream>>>(
        source,
        destination,
        rowIndices,
        static_cast<uint32_t>(totalPackets),
        static_cast<uint32_t>(packetsPerRow),
        RowOfPacket32{makeFastDivisor32(static_cast<uint32_t>(packetsPerRow))},
        sourceRows,
        sourceRowPackets);
}

template <typename PacketT>
void launchWithPacket(const uint8_t *source,
                      uint8_t *destination,
                      const uint64_t *rowIndices,
                      uint64_t logicalRows,
                      uint64_t rowBytes,
                      uint64_t sourceRows,
                      uint64_t sourceStrideBytes,
                      cudaStream_t stream) {
    constexpr uint32_t maxPacketsPerThread32 = maxPacketsPerThread<PacketT, uint32_t>();
    constexpr uint32_t packetsPerThread64 = maxPacketsPerThread<PacketT, uint64_t>();
    const uint64_t packetsPerRow = rowBytes / sizeof(PacketT);
    const uint64_t sourceRowPackets = sourceStrideBytes / sizeof(PacketT);
    THOR_THROW_IF_FALSE(packetsPerRow != 0 && logicalRows <= std::numeric_limits<uint64_t>::max() / packetsPerRow);
    const uint64_t totalPackets = logicalRows * packetsPerRow;
    const auto *sourcePackets = reinterpret_cast<const PacketT *>(source);
    auto *destinationPackets = reinterpret_cast<PacketT *>(destination);

    const uint32_t packetsPerThread = spreadPacketsPerThread(totalPackets, maxPacketsPerThread32);
    const uint32_t blocks = blocksFor(totalPackets, packetsPerThread);
    if (!packetIndexFitsIn32Bits(totalPackets, packetsPerThread)) {
        const uint32_t blocks64 = blocksFor(totalPackets, packetsPerThread64);
        gatherPacketsKernel<PacketT, uint64_t, RowOfPacket64, packetsPerThread64><<<blocks64, kThreadsPerBlock, 0, stream>>>(
            sourcePackets, destinationPackets, rowIndices, totalPackets, packetsPerRow, RowOfPacket64{packetsPerRow}, sourceRows,
            sourceRowPackets);
    } else if (packetsPerThread == 1) {
        launchWithPacketsPerThread<PacketT, 1>(
            sourcePackets, destinationPackets, rowIndices, totalPackets, packetsPerRow, sourceRows, sourceRowPackets, blocks,
            stream);
    } else if (packetsPerThread == 2) {
        launchWithPacketsPerThread<PacketT, 2>(
            sourcePackets, destinationPackets, rowIndices, totalPackets, packetsPerRow, sourceRows, sourceRowPackets, blocks,
            stream);
    } else if (packetsPerThread == 4) {
        launchWithPacketsPerThread<PacketT, 4>(
            sourcePackets, destinationPackets, rowIndices, totalPackets, packetsPerRow, sourceRows, sourceRowPackets, blocks,
            stream);
    } else {
        launchWithPacketsPerThread<PacketT, maxPacketsPerThread32>(
            sourcePackets, destinationPackets, rowIndices, totalPackets, packetsPerRow, sourceRows, sourceRowPackets, blocks,
            stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launchDeviceResidentRowGatherKernel(const uint8_t *source,
                                         uint64_t sourceRows,
                                         uint64_t sourceStrideBytes,
                                         uint8_t *destination,
                                         const uint64_t *rowIndices,
                                         uint64_t logicalRows,
                                         uint64_t rowBytes,
                                         cudaStream_t stream) {
    THOR_THROW_IF_FALSE(logicalRows >= 1 && sourceRows >= 1);
    THOR_THROW_IF_FALSE(rowBytes >= 1 && rowBytes <= sourceStrideBytes);

    // The widest packet, up to 16 bytes, that divides the row bytes, the source stride and both base addresses: then
    // every row of both sides starts on a packet boundary.
    const uint64_t alignment = rowBytes | sourceStrideBytes | reinterpret_cast<uintptr_t>(source) |
                               reinterpret_cast<uintptr_t>(destination) | 16;
    switch (alignment & (~alignment + 1)) {
        case 16:
            launchWithPacket<uint4>(source, destination, rowIndices, logicalRows, rowBytes, sourceRows, sourceStrideBytes, stream);
            return;
        case 8:
            launchWithPacket<uint2>(source, destination, rowIndices, logicalRows, rowBytes, sourceRows, sourceStrideBytes, stream);
            return;
        case 4:
            launchWithPacket<uint32_t>(source, destination, rowIndices, logicalRows, rowBytes, sourceRows, sourceStrideBytes, stream);
            return;
        case 2:
            launchWithPacket<uint16_t>(source, destination, rowIndices, logicalRows, rowBytes, sourceRows, sourceStrideBytes, stream);
            return;
        default:
            launchWithPacket<uint8_t>(source, destination, rowIndices, logicalRows, rowBytes, sourceRows, sourceStrideBytes, stream);
            return;
    }
}
