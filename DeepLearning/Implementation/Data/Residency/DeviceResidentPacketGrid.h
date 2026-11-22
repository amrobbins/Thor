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
 * The ragged materialization, whose rows differ in length, takes its block size and block bounds from here, and
 * the kernels that read a source at any byte take the shift of its words.
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

}  // namespace DeviceResidentPacketGrid
