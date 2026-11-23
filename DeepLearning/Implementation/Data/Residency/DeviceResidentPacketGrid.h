#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

/**
 * The launch geometry of the residency kernels that move a batch tensor as one flat array of packets (the row gather
 * and the window materialization): a packet is 1, 2, 4, 8 or 16 bytes, every batch row is a whole number of packets,
 * consecutive threads take consecutive packets, and a thread takes PacketsPerThread packets of its tile at once.
 * The kernels whose rows differ in length (the ragged materialization, the sequence concatenate) take their block size
 * and block bounds from here, the kernels that read a source at any byte the shift of its words, and those that move
 * rows of bytes in 16-byte packets the copy of such a row (copyRowBytes).
 * For CUDA translation units only.
 */
namespace DeviceResidentPacketGrid {

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint64_t kMaxBlocks = 1ULL << 20;  // the grid-stride loop covers anything larger
// A launch takes the fewest packets per thread (1, 2, 4, ...) that keep it within this many blocks. Up to here a batch
// is under one resident wave (1020 blocks of 256 threads on the 170 SMs of an RTX 5090), so it waits on memory round
// trips, not on bandwidth: what counts is that its packets are spread over as many threads and SMs as the wave holds,
// and with fewer loads per thread the compiler has less to reorder. The dense copy's short step is the same rule with
// the same bound (ROW_COPY_SHORT_MAX_BYTES, 768 blocks of 1 KiB).
constexpr uint64_t kSpreadMaxBlocks = 768;

// Division by a divisor fixed for the launch (Granlund and Montgomery 1994, figure 4.1):
// q = (t + ((n - t) >> shift1)) >> shift2 with t = mulhi(magic, n), exact for every 32-bit n and divisor >= 1.
struct FastDivisor32 {
    uint32_t divisor = 1;
    uint32_t magic = 1;
    uint32_t shift1 = 0;
    uint32_t shift2 = 0;
};

inline FastDivisor32 makeFastDivisor32(uint32_t divisor) {
    THOR_THROW_IF_FALSE(divisor >= 1);
    uint32_t log2Ceil = 0;
    while ((uint64_t(1) << log2Ceil) < divisor) ++log2Ceil;
    FastDivisor32 fast;
    fast.divisor = divisor;
    fast.magic = static_cast<uint32_t>(((uint64_t(1) << 32) * ((uint64_t(1) << log2Ceil) - divisor)) / divisor + 1);
    fast.shift1 = std::min<uint32_t>(log2Ceil, 1);
    fast.shift2 = log2Ceil == 0 ? 0 : log2Ceil - 1;
    return fast;
}

struct RowOfPacket32 {
    FastDivisor32 packetsPerRow;
    __device__ __forceinline__ uint32_t operator()(uint32_t packet) const {
        const uint32_t t = __umulhi(packetsPerRow.magic, packet);
        return (t + ((packet - t) >> packetsPerRow.shift1)) >> packetsPerRow.shift2;
    }
};

// More than 2^32 packets in one batch tensor: 64-bit division, slow but out of the ordinary range.
struct RowOfPacket64 {
    uint64_t packetsPerRow;
    __device__ __forceinline__ uint64_t operator()(uint64_t packet) const { return packet / packetsPerRow; }
};

// Blocks of a launch over totalPackets with packetsPerThread packets per thread: one tile each, up to kMaxBlocks.
inline uint32_t blocksFor(uint64_t totalPackets, uint32_t packetsPerThread) {
    const uint64_t tilePackets = static_cast<uint64_t>(kThreadsPerBlock) * packetsPerThread;
    return static_cast<uint32_t>(std::min(kMaxBlocks, (totalPackets + tilePackets - 1) / tilePackets));
}

// The fewest packets per thread, doubling from 1 up to maxPacketsPerThread, that keep the launch within
// kSpreadMaxBlocks blocks; maxPacketsPerThread once the batch is past that at every depth and bound by bandwidth.
inline uint32_t spreadPacketsPerThread(uint64_t totalPackets, uint32_t maxPacketsPerThread) {
    uint32_t packetsPerThread = 1;
    while (packetsPerThread < maxPacketsPerThread && blocksFor(totalPackets, packetsPerThread) > kSpreadMaxBlocks) {
        packetsPerThread *= 2;
    }
    return packetsPerThread;
}

// Whether the 32-bit packet index serves the launch: the packets and the tile arithmetic (start + stride) stay below
// 2^32.
inline bool packetIndexFitsIn32Bits(uint64_t totalPackets, uint32_t packetsPerThread) {
    return totalPackets + static_cast<uint64_t>(blocksFor(totalPackets, packetsPerThread)) * kThreadsPerBlock * packetsPerThread <=
           std::numeric_limits<uint32_t>::max();
}

// Source bytes that start at any byte are read as aligned words and shifted. In 8-byte words: this is the 8 bytes
// from byte shift (0 to 7) of first on, running into last, the word after it.
__device__ __forceinline__ uint64_t bytesFrom(uint64_t first, uint64_t last, uint32_t shift) {
    const uint32_t bits = shift * 8U;
    return bits == 0 ? first : (first >> bits) | (last << (64U - bits));
}

// In 16-byte words: the 16 bytes from byte shift (0 to 15) of first on, running into last, the word after it.
__device__ __forceinline__ uint4 bytesFrom(uint4 first, uint4 last, uint32_t shift) {
    const uint32_t word[8] = {first.x, first.y, first.z, first.w, last.x, last.y, last.z, last.w};
    // Whole 32-bit words first (shift / 4, one bit of it at a time), then the bytes within a word.
    const bool skipOne = (shift & 4U) != 0;
    const bool skipTwo = (shift & 8U) != 0;
    uint32_t afterOne[7];
#pragma unroll
    for (uint32_t i = 0; i < 7; ++i) afterOne[i] = skipOne ? word[i + 1] : word[i];
    uint32_t afterTwo[5];
#pragma unroll
    for (uint32_t i = 0; i < 5; ++i) afterTwo[i] = skipTwo ? afterOne[i + 2] : afterOne[i];
    const uint32_t bits = (shift & 3U) * 8U;
    return make_uint4(__funnelshift_r(afterTwo[0], afterTwo[1], bits),
                      __funnelshift_r(afterTwo[1], afterTwo[2], bits),
                      __funnelshift_r(afterTwo[2], afterTwo[3], bits),
                      __funnelshift_r(afterTwo[3], afterTwo[4], bits));
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
 * A lane's share of a row of bytes that starts and ends at any byte, moved in 16-byte packets: the row is the bytes
 * [firstByte, endByte) of the destination store, from byte sourceByte of the value store on. Both stores start on a
 * 16-byte boundary. The row has lanesPerRow lanes, and this is lane `lane` of them.
 *
 * Every store is an aligned 1, 2, 4, 8 or 16 bytes of the row. A row that reaches a 16-byte boundary of the
 * destination is its head (the bytes before its first boundary, at most 15), its whole 16-byte packets, and its tail
 * (the bytes after its last boundary, at most 15). The head is one piece per set bit of its length, the smaller
 * pieces first: each piece ends where the next larger boundary begins, so each is aligned. The tail is one piece per
 * set bit of its length, the larger pieces first. A row inside one packet is split the same way at the most aligned
 * byte it holds.
 *
 * The packets are what the lanes share out, PacketsPerLane to a lane in a pass, consecutive lanes consecutive
 * packets; a row of more packets takes more passes. The pieces (8 a row at most) ride along: in a pass a lane also
 * takes one piece if the row has one left for it, and the launcher sizes the lanes to the packets.
 *
 * The source bytes start at any byte of the value store and are read as aligned words and shifted (bytesFrom). A
 * packet's are the 16-byte word that holds the first of them and, unless they are that word, the next; every packet
 * of a row starts at the same byte of its word. A piece's are the 8-byte words that hold its first and its last
 * byte, and it stores the first bytes of the result. A packet past the row's end reads word 0 of the value store
 * and is not stored. A word may end up to 15 bytes past the row, inside the padding every tensor is allocated with.
 *
 * In a pass a lane issues its packets' loads, under no test, before anything waits for memory. A lane with a piece
 * then loads and stores it, with the packets' words on their way, and stores its packets. The packets' stores are
 * written on both sides of the piece's test, so their loads have a reader on each side of it and stay ahead of it.
 * With one reader, behind the test, the compiler moved the loads down to it, behind the piece's store, and such a
 * lane waited for memory twice (the SASS of the one-packet-per-lane ragged kernel of patch 1552).
 */
template <typename LengthT, uint32_t PacketsPerLane>
__device__ __forceinline__ void copyRowBytes(const uint4 *__restrict__ values,
                                             uint4 *__restrict__ destination,
                                             LengthT firstByte,
                                             LengthT endByte,
                                             uint64_t sourceByte,
                                             uint32_t lane,
                                             uint32_t lanesPerRow) {
    constexpr LengthT kPacketBytes = static_cast<LengthT>(sizeof(uint4));
    const LengthT packetsPerPass = static_cast<LengthT>(lanesPerRow) * PacketsPerLane;
    const uint64_t *const valueWords = reinterpret_cast<const uint64_t *>(values);
    uint8_t *const destinationBytes = reinterpret_cast<uint8_t *>(destination);
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

}  // namespace DeviceResidentPacketGrid
