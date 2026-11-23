#include "Utilities/TensorOperations/Misc/Concatenate.h"
#include "Utilities/TensorOperations/Misc/Split.h"

#include "DeepLearning/Implementation/Data/Residency/DeviceResidentPacketGrid.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/TensorOperations/DataTypeConversions/ElementConverter.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <iterator>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace DeviceResidentPacketGrid;
using ThorImplementation::DataType;
using ThorImplementation::ElementConverter;
using ThorImplementation::loadConvertBytes;
using ThorImplementation::storeConvertBytes;

namespace {

uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char *what) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
        throw std::invalid_argument(what);
    return lhs * rhs;
}

uint64_t ceilDiv(uint64_t numerator, uint64_t denominator) {
    return numerator / denominator + static_cast<uint64_t>(numerator % denominator != 0);
}

// One array's bytes of a packed slice, for the byte kernels: the arrays' spans follow each other.
struct ByteSpan {
    uint64_t spanBytes;
    uint64_t packedOffsetBytes;
};

/**
 * The dense concatenate and its reverse, the split, move the spans of a packed tensor: for every outer slice s, array
 * a holds spanBytes[a] bytes at s * spanBytes[a] of its own store and at s * packedSliceBytes + packedOffsetBytes[a]
 * of the packed tensor. A launch holds up to kMaxArraysPerLaunch arrays of one packet width, and its grid is over the
 * packed tensor's slices (slicesKernel); more arrays, or arrays of different widths, are more launches on the stream.
 * Odd spans past a size move in shifted 16-byte packets instead (concatenateBytesKernel), with a run of blocks per
 * array: every array gets the blocks from its firstBlock to the next array's, and a block finds its array by comparing
 * its index with the first blocks of the launch's arrays, all kernel parameters.
 */
constexpr uint32_t kMaxArraysPerLaunch = 16;

// The array whose run of blocks holds this block: the last whose first block is at or before it, read at fixed indices.
template <typename ArrayT, typename LaunchT>
__device__ __forceinline__ ArrayT arrayOfBlock(const LaunchT &launch) {
    ArrayT array = launch.array[0];
#pragma unroll
    for (uint32_t a = 1; a < kMaxArraysPerLaunch; ++a) {
        if (a < launch.arrays && blockIdx.x >= launch.array[a].firstBlock) array = launch.array[a];
    }
    return array;
}

// The slice of a packet index: by the array's fast divisor in 32 bits, by division past them.
template <typename IndexT>
__device__ __forceinline__ IndexT sliceOfPacket(IndexT packet, const FastDivisor32 &sliceOf, uint64_t packetsPerSlice) {
    if constexpr (sizeof(IndexT) == 8) {
        return packet / packetsPerSlice;
    } else {
        return RowOfPacket32{sliceOf}(packet);
    }
}

/**
 * The concatenate and the split, with the grid over the packed tensor. A run of blocks per array reached the packed
 * tensor at a stride: splitting, where spans are short, a warp's load touched a sector per packet and used a part of
 * it, and the blocks of the array beside it asked for the same sectors again (the split of 8 arrays of 16 bytes took
 * 4.3 us where its concatenate took 3.4, patch 1557); concatenating, every resident block wrote span-long runs across
 * the whole packed tensor at once, 4 to 9% over copy from 33 MiB (patch 1559). Here consecutive threads take
 * consecutive packets of the packed tensor (where the launch's arrays follow each other in it), and each array's store
 * is read or written in runs of a span: measured 0.1-0.5 us faster at small sizes both ways, and the concatenate of
 * 33-67 MiB from 9% to 1-3% over copy (patches 1560, 1562).
 *
 * slicePackets are the packets of one slice that the launch's arrays hold. A slice of at most a block's threads of
 * them (columnPackets = slicePackets) is a column of the block, and a block takes columns = 256 / slicePackets whole
 * slices; a longer slice is chunks = ceil(slicePackets / 256) blocks of one column of 256 packets (columnPackets =
 * 256), the last part full. Thread t of chunk c is packet c * 256 + t % columnPackets of slice t / columnPackets of
 * the block's slices. A thread takes PacketsPerThread slices, columns apart, and is at the same place of every one of
 * them: it finds its array once, and no packet needs a division. In a slice of up to 256 packets the array of each
 * packet is a byte of the launch (arrayOfPacket); in a longer one a binary search of the arrays' first packets (four
 * steps for 16 arrays). Both read the launch, a __grid_constant__ parameter, at a thread's own index: constant-memory
 * loads, and a warp's packets name a few arrays at most. (A chain of selects over the 16 arrays cost a hundred
 * instructions per thread before the first load, which a 1- or 2-byte packet did not repay, patch 1559.)
 *
 * The threads past columns * columnPackets, and past the end of a long slice, have no packet and leave. A slice past
 * the last is the last slice again, so no load or store is under a test, and a thread issues all its loads before any
 * store.
 */
struct SlicesArray {
    uint8_t *split = nullptr;
    uint64_t packedOffsetPackets = 0;
    uint64_t spanPackets = 0;
    // The array's first packet among the launch's packets of a slice.
    uint64_t firstPacket = 0;
};

struct SlicesLaunch {
    SlicesArray array[kMaxArraysPerLaunch];
    uint8_t *packed = nullptr;
    uint64_t packedSlicePackets = 0;
    uint64_t slices = 0;
    uint64_t slicePackets = 0;
    // The launch's array of each packet of a slice, for slices of up to 256 packets.
    uint8_t arrayOfPacket[kThreadsPerBlock];
    uint32_t columnPackets = 0;
    uint32_t columns = 0;
    uint32_t chunks = 1;
    uint32_t arrays = 0;
    // Divide a thread index by columnPackets and a block index by chunks.
    FastDivisor32 columnOf;
    FastDivisor32 chunkOf;
};

template <typename PacketT, typename IndexT, uint32_t PacketsPerThread, bool PackedIsDestination, bool LongSlices>
__global__ void __launch_bounds__(kThreadsPerBlock) slicesKernel(const __grid_constant__ SlicesLaunch launch) {
    if (threadIdx.x >= launch.columns * launch.columnPackets) return;
    const uint32_t column = RowOfPacket32{launch.columnOf}(threadIdx.x);
    // A slice of up to 256 packets is one block's column: the block is its group and needs no division (the chunk's
    // division, test and branch cost 0.05-0.12 us at small sizes when every slice took them, patch 1563).
    uint32_t group = blockIdx.x;
    IndexT packetInSlice = static_cast<IndexT>(threadIdx.x - column * launch.columnPackets);
    uint32_t arrayIndex = 0;
    if constexpr (!LongSlices) {
        arrayIndex = launch.arrayOfPacket[packetInSlice];
    } else {
        group = RowOfPacket32{launch.chunkOf}(blockIdx.x);
        packetInSlice += static_cast<IndexT>(blockIdx.x - group * launch.chunks) * kThreadsPerBlock;
        if (packetInSlice >= static_cast<IndexT>(launch.slicePackets)) return;
        // The last array whose first packet is at or before the thread's: steps of the largest power of two below
        // kMaxArraysPerLaunch, then halves, reach every array.
        constexpr uint32_t kFirstStep = [] {
            uint32_t step = 1;
            while (step * 2 < kMaxArraysPerLaunch) step *= 2;
            return step;
        }();
#pragma unroll
        for (uint32_t step = kFirstStep; step != 0; step /= 2) {
            if (arrayIndex + step < launch.arrays && packetInSlice >= static_cast<IndexT>(launch.array[arrayIndex + step].firstPacket)) {
                arrayIndex += step;
            }
        }
    }
    const SlicesArray &array = launch.array[arrayIndex];
    const PacketT *const source = reinterpret_cast<const PacketT *>(PackedIsDestination ? array.split : launch.packed);
    PacketT *const destination = reinterpret_cast<PacketT *>(PackedIsDestination ? launch.packed : array.split);
    const IndexT packetInSpan = packetInSlice - static_cast<IndexT>(array.firstPacket);
    const IndexT packedSlicePackets = static_cast<IndexT>(launch.packedSlicePackets);
    const IndexT packedPacketInSlice = static_cast<IndexT>(array.packedOffsetPackets) + packetInSpan;
    const IndexT spanPackets = static_cast<IndexT>(array.spanPackets);
    const IndexT lastSlice = static_cast<IndexT>(launch.slices - 1);
    const IndexT firstSlice = static_cast<IndexT>(group) * (launch.columns * PacketsPerThread) + column;

    // Loads only: no loaded packet is read before the stores.
    IndexT slice[PacketsPerThread];
    IndexT packedPacket[PacketsPerThread];
    IndexT splitPacket[PacketsPerThread];
    PacketT value[PacketsPerThread];
#pragma unroll
    for (uint32_t k = 0; k < PacketsPerThread; ++k) {
        const IndexT ownSlice = firstSlice + static_cast<IndexT>(k * launch.columns);
        slice[k] = ownSlice < lastSlice ? ownSlice : lastSlice;
        packedPacket[k] = slice[k] * packedSlicePackets + packedPacketInSlice;
        splitPacket[k] = slice[k] * spanPackets + packetInSpan;
        value[k] = source[PackedIsDestination ? splitPacket[k] : packedPacket[k]];
    }
#pragma unroll
    for (uint32_t k = 0; k < PacketsPerThread; ++k) {
        destination[PackedIsDestination ? packedPacket[k] : splitPacket[k]] = value[k];
    }
}

/**
 * The same copies in 16-byte packets whatever byte the spans start and end at: for 1- and 2-byte packets past the
 * size that spreads them (the launcher's rule), where a packet a byte wide is bound by its instructions. Every store
 * is an aligned 16-byte packet of the destination (the packed tensor when concatenating, the array's store when
 * splitting), and the source bytes, which start at any byte, are read as aligned 16-byte words and shifted
 * (bytesFrom, as in the ragged and window materializations).
 *
 * The destination's rows (array a's span of slice s) follow each other with no gap, so an aligned packet that
 * starts inside one row and runs past its end holds the first bytes of the next row: concatenating, the next array's
 * row of the same slice (array 0's row of the next slice after the last array); splitting, the same array's row of the
 * next slice. Every span is at least a packet, so a packet runs into one row at most, and it is written whole from
 * the two rows' bytes. So a row's head, its bytes before its first boundary, is written by the packet of the row
 * before it, and the rows of a destination all move this way or none does (the launcher's rule). The packet that
 * holds the destination's last bytes has no row after it and stores them as aligned pieces instead. A row's packets
 * are those that start inside it, packetsPerRow = ceil(spanBytes / 16) at most, and the grid is flat over
 * outerSlices * packetsPerRow of them per array, consecutive threads consecutive packets; a packet number past the
 * row's packets is idle.
 *
 * A thread issues the loads of all its packets, the two words of the row's bytes and the two of the next row's
 * (the row's again where the packet ends in it), before it reads any; a word is loaded only where its bytes are
 * needed, so no load ends more than 15 bytes past the bytes a row holds, inside the padding every tensor is allocated
 * with.
 */
struct BytesArray {
    uint8_t *split = nullptr;
    uint64_t spanBytes = 0;
    uint64_t packedOffsetBytes = 0;
    // outerSlices * packetsPerRow: the array's packets, and so its blocks' work.
    uint64_t packets = 0;
    uint32_t packetsPerRow = 0;
    uint32_t firstBlock = 0;
    // Divides a packet index by packetsPerRow (32-bit launches).
    FastDivisor32 sliceOf;
    // Concatenating: the array whose row follows this array's in the packed tensor, and whether that row is of the
    // next slice (after the last array); last marks the last array, whose last row ends the packed tensor.
    uint8_t *nextSplit = nullptr;
    uint64_t nextSpanBytes = 0;
    uint32_t nextSliceStep = 0;
    uint32_t last = 0;
};

struct BytesLaunch {
    BytesArray array[kMaxArraysPerLaunch];
    uint8_t *packed = nullptr;
    uint64_t packedSliceBytes = 0;
    uint64_t outerSlices = 0;
    uint32_t arrays = 0;
};

// Store the first count (1 to 15) bytes of value at destination, a 16-byte boundary, as aligned pieces: 8, 4, 2 and 1
// bytes, one per set bit of count, the larger first (a tail, as in the ragged materialization).
__device__ __forceinline__ void storeLeadingBytes(uint8_t *destination, uint4 value, uint32_t count) {
    uint32_t offset = 0;
    if ((count & 8U) != 0) {
        *reinterpret_cast<uint64_t *>(destination) = static_cast<uint64_t>(value.x) | (static_cast<uint64_t>(value.y) << 32);
        offset = 8;
    }
    if ((count & 4U) != 0) {
        *reinterpret_cast<uint32_t *>(destination + offset) = offset == 0 ? value.x : value.z;
        offset += 4;
    }
    // The 32-bit word the piece starts in: offset is a multiple of 4 here, and of 2 below.
    const uint32_t wordOfHalf = (offset & 8U) != 0 ? ((offset & 4U) != 0 ? value.w : value.z) : ((offset & 4U) != 0 ? value.y : value.x);
    if ((count & 2U) != 0) {
        *reinterpret_cast<uint16_t *>(destination + offset) = static_cast<uint16_t>(wordOfHalf);
        offset += 2;
    }
    const uint32_t wordOfByte = (offset & 8U) != 0 ? ((offset & 4U) != 0 ? value.w : value.z) : ((offset & 4U) != 0 ? value.y : value.x);
    if ((count & 1U) != 0) destination[offset] = static_cast<uint8_t>(wordOfByte >> ((offset & 3U) * 8U));
}

template <typename IndexT, uint32_t PacketsPerThread, bool PackedIsDestination>
__global__ void __launch_bounds__(kThreadsPerBlock) concatenateBytesKernel(const BytesLaunch launch) {
    constexpr uint32_t kPacketBytes = sizeof(uint4);
    const BytesArray array = arrayOfBlock<BytesArray>(launch);
    uint8_t *const destinationBytes = PackedIsDestination ? launch.packed : array.split;
    const IndexT packedSliceBytes = static_cast<IndexT>(launch.packedSliceBytes);
    const IndexT spanBytes = static_cast<IndexT>(array.spanBytes);
    const IndexT packedOffsetBytes = static_cast<IndexT>(array.packedOffsetBytes);
    const IndexT lastPacket = static_cast<IndexT>(array.packets - 1);
    const IndexT tileStart = static_cast<IndexT>(blockIdx.x - array.firstBlock) * (kThreadsPerBlock * PacketsPerThread);

    // Loads only: the row's words (a) and the next row's (b).
    IndexT packetByte[PacketsPerThread];
    uint32_t rowBytes[PacketsPerThread];
    bool inRow[PacketsPerThread];
    bool straddles[PacketsPerThread];
    uint32_t shiftA[PacketsPerThread];
    uint32_t shiftB[PacketsPerThread];
    uint4 a0[PacketsPerThread];
    uint4 a1[PacketsPerThread];
    uint4 b0[PacketsPerThread];
    uint4 b1[PacketsPerThread];
#pragma unroll
    for (uint32_t k = 0; k < PacketsPerThread; ++k) {
        const IndexT ownPacket = tileStart + static_cast<IndexT>(k * kThreadsPerBlock + threadIdx.x);
        const IndexT packet = ownPacket < lastPacket ? ownPacket : lastPacket;
        const IndexT slice = sliceOfPacket<IndexT>(packet, array.sliceOf, array.packetsPerRow);
        const uint32_t packetInRow = static_cast<uint32_t>(packet - slice * static_cast<IndexT>(array.packetsPerRow));
        // The row's destination bytes [rowBegin, rowEnd), and the packet: the packetInRow-th 16-byte boundary at or
        // after rowBegin.
        const IndexT rowBegin = PackedIsDestination ? slice * packedSliceBytes + packedOffsetBytes : slice * spanBytes;
        const IndexT rowEnd = rowBegin + spanBytes;
        packetByte[k] = ((rowBegin + (kPacketBytes - 1)) & ~static_cast<IndexT>(kPacketBytes - 1)) +
                        static_cast<IndexT>(packetInRow) * kPacketBytes;
        inRow[k] = packetByte[k] < rowEnd;
        // The bytes of the row in the packet; 16 for an idle packet, which reads the row's first word and stores nothing.
        const IndexT bytesLeft = inRow[k] ? rowEnd - packetByte[k] : kPacketBytes;
        rowBytes[k] = bytesLeft < kPacketBytes ? static_cast<uint32_t>(bytesLeft) : kPacketBytes;
        const IndexT offsetInRow = inRow[k] ? packetByte[k] - rowBegin : 0;
        const uint8_t *const rowSource = PackedIsDestination
                                             ? array.split + static_cast<uint64_t>(slice) * array.spanBytes
                                             : launch.packed + static_cast<uint64_t>(slice) * launch.packedSliceBytes + array.packedOffsetBytes;
        const uint8_t *const sourceA = rowSource + offsetInRow;
        const bool lastRow = static_cast<uint64_t>(slice) + 1 == launch.outerSlices && (!PackedIsDestination || array.last != 0);
        straddles[k] = inRow[k] && rowBytes[k] < kPacketBytes && !lastRow;
        const uint8_t *const nextRowSource =
            PackedIsDestination ? array.nextSplit + (static_cast<uint64_t>(slice) + array.nextSliceStep) * array.nextSpanBytes
                                : rowSource + launch.packedSliceBytes;
        const uint8_t *const sourceB = straddles[k] ? nextRowSource : sourceA;

        shiftA[k] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(sourceA) & (kPacketBytes - 1));
        const uint4 *const wordA = reinterpret_cast<const uint4 *>(sourceA - shiftA[k]);
        a0[k] = wordA[0];
        a1[k] = wordA[shiftA[k] + rowBytes[k] > kPacketBytes ? 1 : 0];
        shiftB[k] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(sourceB) & (kPacketBytes - 1));
        const uint4 *const wordB = reinterpret_cast<const uint4 *>(sourceB - shiftB[k]);
        b0[k] = wordB[0];
        b1[k] = wordB[straddles[k] && shiftB[k] + (kPacketBytes - rowBytes[k]) > kPacketBytes ? 1 : 0];
    }
#pragma unroll
    for (uint32_t k = 0; k < PacketsPerThread; ++k) {
        const uint4 a = bytesFrom(a0[k], a1[k], shiftA[k]);
        const uint4 b = bytesFrom(b0[k], b1[k], shiftB[k]);
        // The row's rowBytes bytes then the next row's: a rotated by rowBytes puts its first rowBytes bytes last, and
        // the 16 bytes from byte 16 - rowBytes of that and b are those followed by b's first bytes.
        const uint32_t rotation = rowBytes[k] < kPacketBytes ? rowBytes[k] : 0;
        const uint4 joined = bytesFrom(bytesFrom(a, a, rotation), b, (kPacketBytes - rotation) & (kPacketBytes - 1));
        const uint4 value = rowBytes[k] < kPacketBytes ? joined : a;
        if (inRow[k]) {
            if (rowBytes[k] == kPacketBytes || straddles[k]) {
                *reinterpret_cast<uint4 *>(destinationBytes + packetByte[k]) = value;
            } else {
                storeLeadingBytes(destinationBytes + packetByte[k], value, rowBytes[k]);
            }
        }
    }
}

// An array of the operation as the host plans it: its base, its span, and the widest packet, up to 16 bytes, on whose
// boundaries its span, its offset in the packed slice, the packed slice and both stores start.
struct ArrayPlan {
    uint32_t index = 0;
    uint8_t *split = nullptr;
    uint64_t spanBytes = 0;
    uint64_t packedOffsetBytes = 0;
    uint32_t packetBytes = 1;
};

// A launch is planned on the stack: no allocation between the call and the kernel launch (the planning's heap
// vectors put the CPU's enqueue at 1.45-1.6 us a launch, patch 1557, against 1.3 for the old kernels).
struct Plan {
    uint8_t *packed = nullptr;
    uint64_t packedSliceBytes = 0;
    uint64_t outerSlices = 0;
    const std::vector<ByteSpan> *geometry = nullptr;
    const std::vector<void *> *splits = nullptr;
    cudaStream_t stream = nullptr;

    // Array i as a launch takes it; false for an array with no bytes to move (a span of zero bytes or, splitting, a
    // null store).
    bool arrayPlanOf(size_t i, ArrayPlan &array) const {
        const ByteSpan &span = (*geometry)[i];
        void *const split = (*splits)[i];
        if (span.spanBytes == 0 || split == nullptr) return false;
        array.index = static_cast<uint32_t>(i);
        array.split = static_cast<uint8_t *>(split);
        array.spanBytes = span.spanBytes;
        array.packedOffsetBytes = span.packedOffsetBytes;
        const uint64_t alignment = span.spanBytes | span.packedOffsetBytes | packedSliceBytes | reinterpret_cast<uintptr_t>(packed) |
                                   reinterpret_cast<uintptr_t>(split) | sizeof(uint4);
        array.packetBytes = static_cast<uint32_t>(alignment & (~alignment + 1));
        return true;
    }
};

// The arrays of one launch.
struct Group {
    ArrayPlan array[kMaxArraysPerLaunch];
    uint32_t count = 0;
};

// The most packets a thread takes once a launch is past kSpreadMaxBlocks blocks at that depth: the row gather's 4 of
// 16-byte packets and 8 narrower, 4 of any width with a 64-bit index.
template <typename PacketT>
constexpr uint32_t maxPacketsPerThread() {
    return sizeof(PacketT) >= 16 ? 4 : 8;
}
constexpr uint32_t kMaxPacketsPerThread64 = 4;
// Shifted 16-byte packets hold four words each until their store: 2 at most, as in the ragged materialization.
constexpr uint32_t kMaxShiftedPacketsPerThread = 2;
// Arrays of 1- and 2-byte packets stay in them while this many per thread keep them within kSpreadMaxBlocks blocks
// (the ragged materialization's bound, measured on both sides in patch 1553).
constexpr uint32_t kNarrowSpreadPacketsPerThread = 4;

uint32_t blocksForPackets(uint64_t packets, uint32_t packetsPerThread) {
    return static_cast<uint32_t>(std::min<uint64_t>(std::numeric_limits<uint32_t>::max(),
                                                    ceilDiv(packets, static_cast<uint64_t>(kThreadsPerBlock) * packetsPerThread)));
}

// The blocks of a launch: one run per array, each a whole number of tiles.
uint64_t blocksForArrays(const uint64_t *packetsPerArray, uint32_t arrays, uint32_t packetsPerThread) {
    uint64_t blocks = 0;
    for (uint32_t i = 0; i < arrays; ++i) blocks += blocksForPackets(packetsPerArray[i], packetsPerThread);
    return blocks;
}

// The fewest packets per thread, doubling from 1 to most, that keep the launch within kSpreadMaxBlocks blocks.
uint32_t spreadPacketsPerThreadFor(const uint64_t *packetsPerArray, uint32_t arrays, uint32_t most) {
    uint32_t packetsPerThread = 1;
    while (packetsPerThread < most && blocksForArrays(packetsPerArray, arrays, packetsPerThread) > kSpreadMaxBlocks) packetsPerThread *= 2;
    return packetsPerThread;
}

uint32_t gridFor(uint64_t blocks) {
    THOR_THROW_IF_FALSE(blocks >= 1 && blocks <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));
    return static_cast<uint32_t>(blocks);
}

// The packets of one slice that a group of arrays holds.
uint64_t slicePacketsOf(const Group &group, uint64_t packetBytes) {
    uint64_t packets = 0;
    for (uint32_t i = 0; i < group.count; ++i) packets += group.array[i].spanBytes / packetBytes;
    return packets;
}

template <typename PacketT, typename IndexT, uint32_t PacketsPerThread, bool PackedIsDestination, bool LongSlices>
void launchSlices(const SlicesLaunch &launch, uint32_t blocks, cudaStream_t stream) {
    slicesKernel<PacketT, IndexT, PacketsPerThread, PackedIsDestination, LongSlices><<<blocks, kThreadsPerBlock, 0, stream>>>(launch);
    CUDA_CHECK(cudaGetLastError());
}

template <typename PacketT, bool PackedIsDestination, bool LongSlices>
void launchSlicesWith(const SlicesLaunch &launch, uint32_t grid, bool index32, uint32_t packetsPerThread, cudaStream_t stream) {
    if (!index32) {
        launchSlices<PacketT, uint64_t, kMaxPacketsPerThread64, PackedIsDestination, LongSlices>(launch, grid, stream);
    } else if (packetsPerThread == 1) {
        launchSlices<PacketT, uint32_t, 1, PackedIsDestination, LongSlices>(launch, grid, stream);
    } else if (packetsPerThread == 2) {
        launchSlices<PacketT, uint32_t, 2, PackedIsDestination, LongSlices>(launch, grid, stream);
    } else if (packetsPerThread == 4 || maxPacketsPerThread<PacketT>() == 4) {
        launchSlices<PacketT, uint32_t, 4, PackedIsDestination, LongSlices>(launch, grid, stream);
    } else {
        // Only packets narrower than 16 bytes take 8 per thread.
        launchSlices<PacketT, uint32_t, sizeof(PacketT) >= 16 ? 4 : 8, PackedIsDestination, LongSlices>(launch, grid, stream);
    }
}

// Up to kMaxArraysPerLaunch arrays of one packet width, with the grid over the packed tensor's slices. A thread takes
// the fewest slices (1, 2, 4, 8) that keep the launch within kSpreadMaxBlocks blocks. The packet index is 32-bit
// while the packed tensor's packets and a block's overhang fit it.
template <typename PacketT, bool PackedIsDestination>
void launchSlicesGroup(const Plan &plan, const Group &group) {
    const uint64_t packedSlicePackets = plan.packedSliceBytes / sizeof(PacketT);
    const uint64_t packedPackets = checkedMultiply(plan.outerSlices, packedSlicePackets, "Dense concatenate packet count overflow.");
    const uint64_t slicePackets = slicePacketsOf(group, sizeof(PacketT));
    THOR_THROW_IF_FALSE(slicePackets >= 1);
    const uint32_t columnPackets = static_cast<uint32_t>(std::min<uint64_t>(slicePackets, kThreadsPerBlock));
    const uint32_t columns = kThreadsPerBlock / columnPackets;
    const uint64_t chunks = ceilDiv(slicePackets, kThreadsPerBlock);
    const auto blocksAt = [&](uint32_t packetsPerThread) {
        return ceilDiv(plan.outerSlices, static_cast<uint64_t>(columns) * packetsPerThread) * chunks;
    };
    uint32_t packetsPerThread = 1;
    while (packetsPerThread < maxPacketsPerThread<PacketT>() && blocksAt(packetsPerThread) > kSpreadMaxBlocks) packetsPerThread *= 2;
    const bool index32 = packedPackets + static_cast<uint64_t>(kThreadsPerBlock) * packetsPerThread <= std::numeric_limits<uint32_t>::max();
    if (!index32) packetsPerThread = kMaxPacketsPerThread64;
    const uint32_t grid = gridFor(blocksAt(packetsPerThread));

    SlicesLaunch launch;
    launch.packed = plan.packed;
    launch.packedSlicePackets = packedSlicePackets;
    launch.slices = plan.outerSlices;
    launch.slicePackets = slicePackets;
    launch.columnPackets = columnPackets;
    launch.columns = columns;
    launch.chunks = static_cast<uint32_t>(chunks);
    launch.arrays = group.count;
    launch.columnOf = makeFastDivisor32(columnPackets);
    launch.chunkOf = makeFastDivisor32(launch.chunks);
    std::fill(std::begin(launch.arrayOfPacket), std::end(launch.arrayOfPacket), uint8_t{0});
    uint64_t firstPacket = 0;
    for (uint32_t i = 0; i < group.count; ++i) {
        SlicesArray &array = launch.array[i];
        array.split = group.array[i].split;
        array.packedOffsetPackets = group.array[i].packedOffsetBytes / sizeof(PacketT);
        array.spanPackets = group.array[i].spanBytes / sizeof(PacketT);
        array.firstPacket = firstPacket;
        if (chunks == 1) {
            for (uint64_t packet = 0; packet < array.spanPackets; ++packet) launch.arrayOfPacket[firstPacket + packet] = static_cast<uint8_t>(i);
        }
        firstPacket += array.spanPackets;
    }
    if (chunks == 1) {
        launchSlicesWith<PacketT, PackedIsDestination, false>(launch, grid, index32, packetsPerThread, plan.stream);
    } else {
        launchSlicesWith<PacketT, PackedIsDestination, true>(launch, grid, index32, packetsPerThread, plan.stream);
    }
}

template <typename IndexT, uint32_t PacketsPerThread, bool PackedIsDestination>
void launchBytes(const BytesLaunch &launch, uint32_t blocks, cudaStream_t stream) {
    concatenateBytesKernel<IndexT, PacketsPerThread, PackedIsDestination><<<blocks, kThreadsPerBlock, 0, stream>>>(launch);
    CUDA_CHECK(cudaGetLastError());
}

// Up to kMaxArraysPerLaunch arrays in shifted 16-byte packets, as one launch. A thread takes two packets only past
// kSpreadMaxBlocks blocks of one. Byte offsets are 32-bit while the packed tensor's bytes and a tile's overhang fit.
template <bool PackedIsDestination>
void launchBytesGroup(const Plan &plan, const Group &group) {
    const uint64_t packedBytes = checkedMultiply(plan.outerSlices, plan.packedSliceBytes, "Dense concatenate byte count overflow.");
    uint64_t packetsPerArray[kMaxArraysPerLaunch];
    for (uint32_t i = 0; i < group.count; ++i) packetsPerArray[i] = plan.outerSlices * ceilDiv(group.array[i].spanBytes, sizeof(uint4));

    uint32_t packetsPerThread = spreadPacketsPerThreadFor(packetsPerArray, group.count, kMaxShiftedPacketsPerThread);
    const bool index32 = packedBytes + static_cast<uint64_t>(kThreadsPerBlock) * packetsPerThread * sizeof(uint4) <=
                         std::numeric_limits<uint32_t>::max();
    if (!index32) packetsPerThread = 1;

    const uint32_t grid = gridFor(blocksForArrays(packetsPerArray, group.count, packetsPerThread));
    BytesLaunch launch;
    launch.packed = plan.packed;
    launch.packedSliceBytes = plan.packedSliceBytes;
    launch.outerSlices = plan.outerSlices;
    launch.arrays = group.count;
    uint32_t firstBlock = 0;
    for (uint32_t i = 0; i < group.count; ++i) {
        BytesArray &array = launch.array[i];
        array.split = group.array[i].split;
        array.spanBytes = group.array[i].spanBytes;
        array.packedOffsetBytes = group.array[i].packedOffsetBytes;
        array.packetsPerRow = static_cast<uint32_t>(ceilDiv(array.spanBytes, sizeof(uint4)));
        array.packets = packetsPerArray[i];
        array.firstBlock = firstBlock;
        if (index32) array.sliceOf = makeFastDivisor32(array.packetsPerRow);
        const size_t next = (group.array[i].index + 1) % plan.geometry->size();
        array.nextSplit = static_cast<uint8_t *>((*plan.splits)[next]);
        array.nextSpanBytes = (*plan.geometry)[next].spanBytes;
        array.nextSliceStep = next == 0 ? 1 : 0;
        array.last = next == 0 ? 1 : 0;
        firstBlock += blocksForPackets(array.packets, packetsPerThread);
    }
    if (!index32) {
        launchBytes<uint64_t, 1, PackedIsDestination>(launch, grid, plan.stream);
    } else if (packetsPerThread == 1) {
        launchBytes<uint32_t, 1, PackedIsDestination>(launch, grid, plan.stream);
    } else {
        launchBytes<uint32_t, kMaxShiftedPacketsPerThread, PackedIsDestination>(launch, grid, plan.stream);
    }
}

// How a launch moves an array: in packets of 1, 2, 4, 8 or 16 bytes (the route is the packet's log2), or shifted.
constexpr uint32_t kShiftedRoute = 5;

uint32_t widthRoute(uint32_t packetBytes) {
    return packetBytes == 1 ? 0 : packetBytes == 2 ? 1 : packetBytes == 4 ? 2 : packetBytes == 8 ? 3 : 4;
}

template <bool PackedIsDestination>
void launchGroup(const Plan &plan, uint32_t route, const Group &group) {
    switch (route) {
        case 4:
            launchSlicesGroup<uint4, PackedIsDestination>(plan, group);
            return;
        case 3:
            launchSlicesGroup<uint2, PackedIsDestination>(plan, group);
            return;
        case 2:
            launchSlicesGroup<uint32_t, PackedIsDestination>(plan, group);
            return;
        case 1:
            launchSlicesGroup<uint16_t, PackedIsDestination>(plan, group);
            return;
        case 0:
            launchSlicesGroup<uint8_t, PackedIsDestination>(plan, group);
            return;
        default:
            launchBytesGroup<PackedIsDestination>(plan, group);
            return;
    }
}

/**
 * The arrays go by packet width. An array's width is the widest packet, up to 16 bytes, on whose boundaries its span,
 * its offset in the packed slice, the packed slice and both stores start, so one array's odd span does not narrow the
 * others. Arrays of 1- and 2-byte packets are fastest as such while the most packets a thread takes keep them within
 * kSpreadMaxBlocks blocks; a larger set of them moves in shifted 16-byte packets instead (measured for the ragged
 * materialization, patch 1553) when every span is at least a packet and the stores start on 16-byte boundaries, and
 * stays in narrow packets otherwise. Splitting, each array decides for itself. Concatenating, the head of a row is
 * written by the packet that ends the row before it, which belongs to another array, so either every array moves in
 * shifted packets or none does. Each route launches its arrays kMaxArraysPerLaunch at a time, the widest packets
 * first, the shifted ones last.
 */
template <bool PackedIsDestination>
void launchPlan(const Plan &plan) {
    const size_t arrays = plan.geometry->size();
    const bool packedOnWords = reinterpret_cast<uintptr_t>(plan.packed) % sizeof(uint4) == 0;
    bool storesOnWords = packedOnWords;
    bool everySpanHoldsPacket = true;
    for (const ByteSpan &span : *plan.geometry) everySpanHoldsPacket &= span.spanBytes >= sizeof(uint4);
    ArrayPlan array;
    for (size_t i = 0; i < arrays; ++i) {
        if (plan.arrayPlanOf(i, array)) storesOnWords &= reinterpret_cast<uintptr_t>(array.split) % sizeof(uint4) == 0;
    }
    const auto shiftable = [&](const ArrayPlan &candidate) {
        const bool canShift = PackedIsDestination ? storesOnWords && everySpanHoldsPacket
                                                  : packedOnWords && reinterpret_cast<uintptr_t>(candidate.split) % sizeof(uint4) == 0 &&
                                                        candidate.spanBytes >= sizeof(uint4);
        return canShift && (candidate.packetBytes < sizeof(uint32_t) || PackedIsDestination);
    };
    // The blocks the narrow arrays that could shift would take in narrow packets.
    uint64_t narrowBlocks = 0;
    for (size_t i = 0; i < arrays; ++i) {
        if (plan.arrayPlanOf(i, array) && array.packetBytes < sizeof(uint32_t) && shiftable(array)) {
            narrowBlocks += blocksForPackets(plan.outerSlices * (array.spanBytes / array.packetBytes), kNarrowSpreadPacketsPerThread);
        }
    }
    const bool shift = narrowBlocks > kSpreadMaxBlocks;
    const auto routeOf = [&](const ArrayPlan &candidate) {
        return shift && shiftable(candidate) ? kShiftedRoute : widthRoute(candidate.packetBytes);
    };

    for (const uint32_t route : {4U, 3U, 2U, 1U, 0U, kShiftedRoute}) {
        Group group;
        for (size_t i = 0; i < arrays; ++i) {
            if (!plan.arrayPlanOf(i, array) || routeOf(array) != route) continue;
            group.array[group.count++] = array;
            if (group.count == kMaxArraysPerLaunch) {
                launchGroup<PackedIsDestination>(plan, route, group);
                group.count = 0;
            }
        }
        if (group.count != 0) launchGroup<PackedIsDestination>(plan, route, group);
    }
}

// A concatenate (PackedIsDestination) or a split whose arrays all have the packed tensor's type, as byte copies of
// spans of spanElements[i] elements of elementBytes; launchTyped has checked the arrays, the counts and their products.
template <bool PackedIsDestination>
void launchSameType(void *packed, const std::vector<void *> &splits, uint64_t outerSlices, uint64_t elementBytes,
                    const std::vector<uint64_t> &spanElements, Stream stream) {
    std::vector<ByteSpan> geometry(spanElements.size());
    uint64_t packedSliceBytes = 0;
    for (size_t i = 0; i < spanElements.size(); ++i) {
        geometry[i] = ByteSpan{spanElements[i] * elementBytes, packedSliceBytes};
        packedSliceBytes += geometry[i].spanBytes;
    }

    ScopedGpu scopedGpu(stream.getGpuNum());
    Plan plan;
    plan.packed = static_cast<uint8_t *>(packed);
    plan.packedSliceBytes = packedSliceBytes;
    plan.outerSlices = outerSlices;
    plan.geometry = &geometry;
    plan.splits = &splits;
    plan.stream = stream.getStream();
    launchPlan<PackedIsDestination>(plan);
}

/**
 * The concatenate and the split when an array's element type is not the packed tensor's. One launch moves its arrays
 * (kMaxArraysPerLaunch of them) whatever their types: a thread loads packets of elements from the source, converts
 * them to the destination's type in registers and stores them, so every element is read once and written once, and no
 * array is converted ahead of the operation. The conversion is TypeConverter's (ElementConverter), so the bits are
 * those of converting the array first; an array of the packed tensor's own type is copied.
 *
 * A packet is a power of two of elements, at most the elements of 16 bytes of the wider of the array's type and the
 * packed tensor's (the width TypeConverter's kernel measured fastest, patch 1518), so each side of a packet is one
 * access of 16 bytes or fewer. An array's packet is the largest whose accesses are aligned on both sides: one that
 * divides its span, its place in the packed slice and the packed slice, in elements, and on whose boundaries both
 * stores start; down to one element, which always is.
 *
 * The grid is slicesKernel's, over the packed tensor's slices, in packets: a slice of up to 256 packets is a column
 * of a block, which takes 256 / slicePackets columns; a longer one is chunks of 256 packets. A thread has one place
 * in a column and takes it in PacketsPerThread slices, columns apart: for whole packets every load is issued before
 * the first conversion and every conversion is done before the first store; smaller packets (spans that do not hold
 * whole ones) go one after another. A place past the last slice is loaded as the thread's first and not stored.
 * One packet per thread while the launch is within kConvertSpreadMaxBlocks blocks, four past that: at one, a launch
 * of a few MiB was bound by its blocks and threads, 1.3 to 1.6 times a copy of the same bytes (patch 1576). The
 * bound is under slicesKernel's 768: a thread here does more before its first load, and at 512 blocks of one packet
 * an INT32 to FP32 launch was 0.63 us behind the byte kernel on the same grid and bytes, while at 256 blocks of
 * four it was level with it or ahead (patch 1578).
 *
 * In the short case a thread's place is read from the launch: role[thread] holds its array, its column and its packet
 * in the array's span, filled on the host so that the threads of arrays of one type and packet follow each other, all
 * columns of them, before the next type's, each type starting on a warp where a block has the room. A warp then
 * holds one type, so the switch on the array's type below, the one thing that differs between threads, is taken one
 * way by a whole warp.
 * (In slicesKernel's order, a slice after a slice, every warp of a slice shorter than a warp would hold every type and
 * run each type's conversion in turn.) In the long case the arrays follow each other in the slice in the same order
 * of types, and a thread finds its array by the binary search. Which case a launch is, is a test on the launch.
 * Indices are 64-bit: there is no division by thread, so their cost is two multiplications a thread.
 *
 * A thread's loads are inside its type's case, behind the switch (a load of its table and an indirect branch).
 * Issued ahead of it by their byte width (patch 1580) they gained 0.03 to 0.13 us on launches of one packet a thread,
 * but a load whose width is a run-time choice was not kept whole in every kernel (the 2-byte packed split kernels
 * issued a 16-byte load as three), 15 to 33% slower at 2 to 8 MiB; taken out again.
 */
#define THOR_CONCATENATE_ELEMENT_TYPES(X) \
    X(FP8_E4M3, __nv_fp8_e4m3)            \
    X(FP8_E5M2, __nv_fp8_e5m2)            \
    X(BF16, __nv_bfloat16)                \
    X(FP16, half)                         \
    X(FP32, float)                        \
    X(FP64, double)                       \
    X(INT8, int8_t)                       \
    X(INT16, int16_t)                     \
    X(INT32, int32_t)                     \
    X(INT64, int64_t)                     \
    X(UINT8, uint8_t)                     \
    X(UINT16, uint16_t)                   \
    X(UINT32, uint32_t)                   \
    X(UINT64, uint64_t)                   \
    X(BOOLEAN, bool)

constexpr uint32_t kConvertPacketBytes = 16;
constexpr uint32_t kConvertSpreadPacketsPerThread = 4;
constexpr uint64_t kConvertSpreadMaxBlocks = 256;
constexpr uint32_t kWarpThreads = 32;
constexpr uint32_t kIdleRole = 0xFFFFFFFFU;

struct ConvertArray {
    uint8_t *split = nullptr;
    uint64_t packedOffsetElements = 0;
    uint64_t spanElements = 0;
    // The array's first packet among the launch's packets of a slice (long slices).
    uint64_t firstPacket = 0;
    // The DataType of the array's elements, and the log2 of the elements of one of its packets.
    uint32_t dataType = 0;
    uint32_t packetLog2 = 0;
};

struct ConvertLaunch {
    ConvertArray array[kMaxArraysPerLaunch];
    uint8_t *packed = nullptr;
    uint64_t packedSliceElements = 0;
    uint64_t slices = 0;
    uint64_t slicePackets = 0;
    // A thread's array, its column << 8 and its packet in the span << 16, or kIdleRole (short slices).
    uint32_t role[kThreadsPerBlock];
    uint32_t columns = 1;
    uint32_t chunks = 1;
    uint32_t arrays = 0;
    // Divides a block index by chunks.
    FastDivisor32 chunkOf;
};

// Elements elements from one type to another: one access a side.
template <typename FromT, typename ToT, int Elements>
__device__ __forceinline__ void convertElements(const FromT *from, ToT *to) {
    alignas(16) FromT in[Elements];
    loadConvertBytes<Elements * static_cast<int>(sizeof(FromT))>(in, from);
    if constexpr (std::is_same_v<FromT, ToT>) {
        storeConvertBytes<Elements * static_cast<int>(sizeof(ToT))>(to, in);
    } else {
        alignas(16) ToT out[Elements];
#pragma unroll
        for (int i = 0; i < Elements; ++i) out[i] = ElementConverter<FromT, ToT>{}(in[i]);
        storeConvertBytes<Elements * static_cast<int>(sizeof(ToT))>(to, out);
    }
}

// A thread's whole packets, of Elements elements, a step of elements apart on each side: the loads, then the
// conversions, then the stores. Only the first `stored` are the thread's; the others load the first again.
template <typename FromT, typename ToT, int Elements, uint32_t PacketsPerThread>
__device__ __forceinline__ void convertWholePackets(const FromT *from, ToT *to, uint64_t fromStep, uint64_t toStep, uint32_t stored) {
    alignas(16) FromT in[PacketsPerThread][Elements];
#pragma unroll
    for (uint32_t k = 0; k < PacketsPerThread; ++k) {
        loadConvertBytes<Elements * static_cast<int>(sizeof(FromT))>(in[k], from + (k < stored ? k * fromStep : 0));
    }
    if constexpr (std::is_same_v<FromT, ToT>) {
#pragma unroll
        for (uint32_t k = 0; k < PacketsPerThread; ++k) {
            if (k < stored) storeConvertBytes<Elements * static_cast<int>(sizeof(ToT))>(to + k * toStep, in[k]);
        }
    } else {
        alignas(16) ToT out[PacketsPerThread][Elements];
#pragma unroll
        for (uint32_t k = 0; k < PacketsPerThread; ++k) {
#pragma unroll
            for (int i = 0; i < Elements; ++i) out[k][i] = ElementConverter<FromT, ToT>{}(in[k][i]);
        }
#pragma unroll
        for (uint32_t k = 0; k < PacketsPerThread; ++k) {
            if (k < stored) storeConvertBytes<Elements * static_cast<int>(sizeof(ToT))>(to + k * toStep, out[k]);
        }
    }
}

// The most elements a packet of the two types holds.
template <typename FromT, typename ToT>
__host__ __device__ constexpr uint32_t convertPacketElements() {
    return kConvertPacketBytes / static_cast<uint32_t>(sizeof(FromT) > sizeof(ToT) ? sizeof(FromT) : sizeof(ToT));
}

// A thread's packets of 1 << packetLog2 elements: whole packets together, smaller ones one after another.
template <typename FromT, typename ToT, uint32_t PacketsPerThread>
__device__ __forceinline__ void convertPackets(const FromT *from, ToT *to, uint64_t fromStep, uint64_t toStep, uint32_t stored, uint32_t packetLog2) {
    constexpr uint32_t kMostElements = convertPacketElements<FromT, ToT>();
    if ((1U << packetLog2) == kMostElements) {
        convertWholePackets<FromT, ToT, static_cast<int>(kMostElements), PacketsPerThread>(from, to, fromStep, toStep, stored);
        return;
    }
#pragma unroll 1
    for (uint32_t k = 0; k < stored; ++k) {
        const FromT *const fromAt = from + k * fromStep;
        ToT *const toAt = to + k * toStep;
        switch (packetLog2) {
            case 3:
                if constexpr (kMostElements > 8) convertElements<FromT, ToT, 8>(fromAt, toAt);
                break;
            case 2:
                if constexpr (kMostElements > 4) convertElements<FromT, ToT, 4>(fromAt, toAt);
                break;
            case 1:
                if constexpr (kMostElements > 2) convertElements<FromT, ToT, 2>(fromAt, toAt);
                break;
            default:
                convertElements<FromT, ToT, 1>(fromAt, toAt);
                break;
        }
    }
}

template <typename PackedT, uint32_t PacketsPerThread, bool PackedIsDestination>
__global__ void __launch_bounds__(kThreadsPerBlock) convertSlicesKernel(const __grid_constant__ ConvertLaunch launch) {
    uint32_t group = blockIdx.x;
    uint32_t column = 0;
    uint32_t arrayIndex = 0;
    uint64_t packetInSpan = 0;
    if (launch.chunks == 1) {
        const uint32_t role = launch.role[threadIdx.x];
        if (role == kIdleRole) return;
        arrayIndex = role & 0xFFU;
        column = (role >> 8U) & 0xFFU;
        packetInSpan = role >> 16U;
    } else {
        group = RowOfPacket32{launch.chunkOf}(blockIdx.x);
        const uint64_t packetInSlice = static_cast<uint64_t>(blockIdx.x - group * launch.chunks) * kThreadsPerBlock + threadIdx.x;
        if (packetInSlice >= launch.slicePackets) return;
        // The last array whose first packet is at or before the thread's, as in slicesKernel.
        constexpr uint32_t kFirstStep = [] {
            uint32_t step = 1;
            while (step * 2 < kMaxArraysPerLaunch) step *= 2;
            return step;
        }();
#pragma unroll
        for (uint32_t step = kFirstStep; step != 0; step /= 2) {
            if (arrayIndex + step < launch.arrays && packetInSlice >= launch.array[arrayIndex + step].firstPacket) {
                arrayIndex += step;
            }
        }
        packetInSpan = packetInSlice - launch.array[arrayIndex].firstPacket;
    }
    // The thread's slices: its first, then every columns-th, while there are slices.
    const uint64_t firstSlice = static_cast<uint64_t>(group) * (launch.columns * PacketsPerThread) + column;
    if (firstSlice >= launch.slices) return;
    uint32_t stored = 1;
#pragma unroll
    for (uint32_t k = 1; k < PacketsPerThread; ++k) {
        stored += firstSlice + static_cast<uint64_t>(k) * launch.columns < launch.slices ? 1U : 0U;
    }
    const ConvertArray &array = launch.array[arrayIndex];
    const uint64_t elementInSpan = packetInSpan << array.packetLog2;
    const uint64_t arrayElement = firstSlice * array.spanElements + elementInSpan;
    const uint64_t packedElement = firstSlice * launch.packedSliceElements + array.packedOffsetElements + elementInSpan;
    const uint64_t arrayStep = launch.columns * array.spanElements;
    const uint64_t packedStep = launch.columns * launch.packedSliceElements;
    PackedT *const packedAt = reinterpret_cast<PackedT *>(launch.packed) + packedElement;
    switch (static_cast<DataType>(array.dataType)) {
#define THOR_CONCATENATE_CONVERT_CASE(name, type)                                                                              \
    case DataType::name: {                                                                                                     \
        type *const arrayAt = reinterpret_cast<type *>(array.split) + arrayElement;                                            \
        if constexpr (PackedIsDestination) {                                                                                   \
            convertPackets<type, PackedT, PacketsPerThread>(arrayAt, packedAt, arrayStep, packedStep, stored, array.packetLog2); \
        } else {                                                                                                               \
            convertPackets<PackedT, type, PacketsPerThread>(packedAt, arrayAt, packedStep, arrayStep, stored, array.packetLog2); \
        }                                                                                                                      \
    } break;
        THOR_CONCATENATE_ELEMENT_TYPES(THOR_CONCATENATE_CONVERT_CASE)
#undef THOR_CONCATENATE_CONVERT_CASE
        default:
            break;
    }
}

uint64_t elementBytesOf(DataType dataType) {
    switch (dataType) {
#define THOR_CONCATENATE_BYTES_CASE(name, type) \
    case DataType::name:                        \
        return sizeof(type);
        THOR_CONCATENATE_ELEMENT_TYPES(THOR_CONCATENATE_BYTES_CASE)
#undef THOR_CONCATENATE_BYTES_CASE
        default:
            throw std::invalid_argument("Dense concatenate element type is not a tensor storage type.");
    }
}

// An array of a typed operation as the host plans it.
struct TypedArray {
    uint8_t *split = nullptr;
    DataType dataType = DataType::FP32;
    uint64_t spanElements = 0;
    uint64_t packedOffsetElements = 0;
    uint32_t packetLog2 = 0;
};

// The log2 of the elements of an array's packet: see convertSlicesKernel.
uint32_t packetLog2Of(const TypedArray &array, const uint8_t *packed, DataType packedDataType, uint64_t packedSliceElements) {
    const uint64_t arrayBytes = elementBytesOf(array.dataType);
    const uint64_t packedBytes = elementBytesOf(packedDataType);
    if (reinterpret_cast<uintptr_t>(array.split) % arrayBytes != 0 || reinterpret_cast<uintptr_t>(packed) % packedBytes != 0)
        throw std::invalid_argument("Dense concatenate stores must start on a boundary of their element type.");
    uint32_t packetLog2 = 0;
    while ((2ULL << packetLog2) * std::max(arrayBytes, packedBytes) <= kConvertPacketBytes) ++packetLog2;
    const uint64_t counts = array.spanElements | array.packedOffsetElements | packedSliceElements;
    for (; packetLog2 != 0; --packetLog2) {
        const uint64_t elements = 1ULL << packetLog2;
        if ((counts & (elements - 1)) == 0 && reinterpret_cast<uintptr_t>(array.split) % (elements * arrayBytes) == 0 &&
            reinterpret_cast<uintptr_t>(packed) % (elements * packedBytes) == 0) {
            break;
        }
    }
    return packetLog2;
}

template <typename PackedT, uint32_t PacketsPerThread, bool PackedIsDestination>
void launchConvert(const ConvertLaunch &launch, uint32_t blocks, cudaStream_t stream) {
    convertSlicesKernel<PackedT, PacketsPerThread, PackedIsDestination><<<blocks, kThreadsPerBlock, 0, stream>>>(launch);
    CUDA_CHECK(cudaGetLastError());
}

template <typename PackedT, bool PackedIsDestination>
void launchConvertWith(const ConvertLaunch &launch, uint32_t grid, uint32_t packetsPerThread, cudaStream_t stream) {
    if (packetsPerThread == 1) {
        launchConvert<PackedT, 1, PackedIsDestination>(launch, grid, stream);
    } else {
        launchConvert<PackedT, kConvertSpreadPacketsPerThread, PackedIsDestination>(launch, grid, stream);
    }
}

// Up to kMaxArraysPerLaunch arrays of any types as one launch. The arrays are in the order of their type and packet
// (the caller's), and the threads of a block follow it.
template <bool PackedIsDestination>
void launchConvertGroup(uint8_t *packed,
                        DataType packedDataType,
                        uint64_t packedSliceElements,
                        uint64_t outerSlices,
                        const TypedArray *arrays,
                        uint32_t count,
                        cudaStream_t stream) {
    ConvertLaunch launch;
    launch.packed = packed;
    launch.packedSliceElements = packedSliceElements;
    launch.slices = outerSlices;
    launch.arrays = count;
    uint64_t slicePackets = 0;
    for (uint32_t i = 0; i < count; ++i) {
        ConvertArray &array = launch.array[i];
        array.split = arrays[i].split;
        array.packedOffsetElements = arrays[i].packedOffsetElements;
        array.spanElements = arrays[i].spanElements;
        array.firstPacket = slicePackets;
        array.dataType = static_cast<uint32_t>(arrays[i].dataType);
        array.packetLog2 = arrays[i].packetLog2;
        slicePackets += arrays[i].spanElements >> arrays[i].packetLog2;
    }
    THOR_THROW_IF_FALSE(slicePackets >= 1);
    launch.slicePackets = slicePackets;
    const uint64_t chunks = ceilDiv(slicePackets, kThreadsPerBlock);
    THOR_THROW_IF_FALSE(chunks <= std::numeric_limits<uint32_t>::max());
    const bool longSlices = chunks > 1;
    uint32_t columns = longSlices ? 1U : kThreadsPerBlock / static_cast<uint32_t>(slicePackets);
    launch.chunks = static_cast<uint32_t>(chunks);
    launch.chunkOf = makeFastDivisor32(launch.chunks);
    std::fill(std::begin(launch.role), std::end(launch.role), kIdleRole);
    if (!longSlices) {
        // The runs of arrays of one type and packet, and the packets of a slice each holds.
        uint32_t runEnd[kMaxArraysPerLaunch];
        uint32_t runPackets[kMaxArraysPerLaunch];
        uint32_t runs = 0;
        for (uint32_t first = 0; first < count;) {
            uint32_t end = first + 1;
            while (end < count && arrays[end].dataType == arrays[first].dataType && arrays[end].packetLog2 == arrays[first].packetLog2) ++end;
            uint32_t packets = 0;
            for (uint32_t i = first; i < end; ++i) packets += static_cast<uint32_t>(arrays[i].spanElements >> arrays[i].packetLog2);
            runEnd[runs] = end;
            runPackets[runs++] = packets;
            first = end;
        }
        // Every run starts on a warp, with the most columns that leaves room for, so no warp holds two types: one that
        // does runs each type's conversion in turn, a memory round trip longer (0.36 us on a small launch, patch
        // 1576). Where not even one column fits that way the runs follow each other.
        bool onWarps = false;
        if (runs > 1) {
            for (uint32_t fewer = columns; fewer >= 1 && !onWarps; --fewer) {
                uint32_t threads = 0;
                for (uint32_t run = 0; run < runs; ++run) threads += (fewer * runPackets[run] + kWarpThreads - 1) / kWarpThreads * kWarpThreads;
                if (threads <= kThreadsPerBlock) {
                    columns = fewer;
                    onWarps = true;
                }
            }
        }
        uint32_t thread = 0;
        uint32_t first = 0;
        for (uint32_t run = 0; run < runs; ++run) {
            if (onWarps) thread = (thread + kWarpThreads - 1) / kWarpThreads * kWarpThreads;
            for (uint32_t column = 0; column < columns; ++column) {
                for (uint32_t i = first; i < runEnd[run]; ++i) {
                    const uint32_t spanPackets = static_cast<uint32_t>(arrays[i].spanElements >> arrays[i].packetLog2);
                    THOR_THROW_IF_FALSE(thread + spanPackets <= kThreadsPerBlock);
                    for (uint32_t packet = 0; packet < spanPackets; ++packet) {
                        launch.role[thread++] = i | (column << 8U) | (packet << 16U);
                    }
                }
            }
            first = runEnd[run];
        }
    }
    launch.columns = columns;
    // One packet per thread while that is within kConvertSpreadMaxBlocks, kConvertSpreadPacketsPerThread past it.
    const uint64_t blocksAtOne = checkedMultiply(ceilDiv(outerSlices, columns), chunks, "Dense concatenate block count overflow.");
    const uint32_t packetsPerThread = blocksAtOne <= kConvertSpreadMaxBlocks ? 1U : kConvertSpreadPacketsPerThread;
    const uint64_t blocks = ceilDiv(outerSlices, static_cast<uint64_t>(columns) * packetsPerThread) * chunks;
    const uint32_t grid = gridFor(blocks);
    switch (packedDataType) {
#define THOR_CONCATENATE_LAUNCH_CASE(name, type)                                                 \
    case DataType::name:                                                                         \
        launchConvertWith<type, PackedIsDestination>(launch, grid, packetsPerThread, stream); \
        break;
        THOR_CONCATENATE_ELEMENT_TYPES(THOR_CONCATENATE_LAUNCH_CASE)
#undef THOR_CONCATENATE_LAUNCH_CASE
        default:
            throw std::invalid_argument("Dense concatenate element type is not a tensor storage type.");
    }
}

// A typed concatenate (PackedIsDestination) or split: the arguments checked; the byte operation when no array's type
// differs from the packed tensor's; else the converting launches, kMaxArraysPerLaunch arrays each.
template <bool PackedIsDestination>
void launchTyped(void *packed,
                 DataType packedDataType,
                 const std::vector<void *> &splits,
                 const std::vector<DataType> &splitDataTypes,
                 uint64_t outerSlices,
                 const std::vector<uint64_t> &spanElements,
                 Stream stream) {
    if (spanElements.empty()) throw std::invalid_argument("Dense concatenate requires at least one array.");
    if (splits.size() != spanElements.size() || splitDataTypes.size() != spanElements.size())
        throw std::invalid_argument("Dense concatenate takes one array and one element type per span.");
    if (packed == nullptr) throw std::invalid_argument("Dense concatenate packed tensor is null.");
    const uint64_t packedBytes = elementBytesOf(packedDataType);
    uint64_t packedSliceElements = 0;
    bool anyElements = false;
    bool converts = false;
    for (size_t i = 0; i < spanElements.size(); ++i) {
        (void)elementBytesOf(splitDataTypes[i]);
        if (spanElements[i] > std::numeric_limits<uint64_t>::max() - packedSliceElements)
            throw std::invalid_argument("Dense concatenate packed slice element count overflow.");
        packedSliceElements += spanElements[i];
        if (PackedIsDestination && splits[i] == nullptr && spanElements[i] != 0)
            throw std::invalid_argument("Dense concatenate source array is null.");
        const bool moves = spanElements[i] != 0 && splits[i] != nullptr;
        anyElements |= moves;
        converts |= moves && splitDataTypes[i] != packedDataType;
    }
    if (packedSliceElements == 0) throw std::invalid_argument("Dense concatenate requires a non-zero packed slice size.");
    if (outerSlices == 0 || !anyElements) return;
    constexpr uint64_t kMostElementBytes = 8;
    (void)checkedMultiply(checkedMultiply(outerSlices, packedSliceElements, "Dense concatenate element count overflow."), kMostElementBytes,
                          "Dense concatenate byte count overflow.");
    if (!converts) {
        launchSameType<PackedIsDestination>(packed, splits, outerSlices, packedBytes, spanElements, stream);
        return;
    }

    ScopedGpu scopedGpu(stream.getGpuNum());
    uint8_t *const packedStore = static_cast<uint8_t *>(packed);
    TypedArray group[kMaxArraysPerLaunch];
    uint32_t count = 0;
    uint64_t packedOffsetElements = 0;
    for (size_t i = 0; i < spanElements.size(); ++i) {
        const uint64_t offset = packedOffsetElements;
        packedOffsetElements += spanElements[i];
        if (spanElements[i] == 0 || splits[i] == nullptr) continue;
        TypedArray array;
        array.split = static_cast<uint8_t *>(splits[i]);
        array.dataType = splitDataTypes[i];
        array.spanElements = spanElements[i];
        array.packedOffsetElements = offset;
        array.packetLog2 = packetLog2Of(array, packedStore, packedDataType, packedSliceElements);
        // After the last array of its type and packet, or at the end.
        uint32_t place = count;
        for (uint32_t j = 0; j < count; ++j) {
            if (group[j].dataType == array.dataType && group[j].packetLog2 == array.packetLog2) place = j + 1;
        }
        for (uint32_t j = count; j > place; --j) group[j] = group[j - 1];
        group[place] = array;
        if (++count == kMaxArraysPerLaunch) {
            launchConvertGroup<PackedIsDestination>(packedStore, packedDataType, packedSliceElements, outerSlices, group, count, stream.getStream());
            count = 0;
        }
    }
    if (count != 0) {
        launchConvertGroup<PackedIsDestination>(packedStore, packedDataType, packedSliceElements, outerSlices, group, count, stream.getStream());
    }
}

}  // namespace

void launchConcatenate(void *dest,
                       DataType destDataType,
                       const std::vector<void *> &sources,
                       const std::vector<DataType> &sourceDataTypes,
                       uint64_t outerSlices,
                       const std::vector<uint64_t> &spanElements,
                       Stream stream) {
    launchTyped<true>(dest, destDataType, sources, sourceDataTypes, outerSlices, spanElements, stream);
}

void launchSplit(const std::vector<void *> &dests,
                 const std::vector<DataType> &destDataTypes,
                 const void *source,
                 DataType sourceDataType,
                 uint64_t outerSlices,
                 const std::vector<uint64_t> &spanElements,
                 Stream stream) {
    launchTyped<false>(const_cast<void *>(source), sourceDataType, dests, destDataTypes, outerSlices, spanElements, stream);
}
