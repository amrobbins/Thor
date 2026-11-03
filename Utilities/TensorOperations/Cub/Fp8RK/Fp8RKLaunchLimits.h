#pragma once

// 1131: the 32-bit limits the retained FP8 launchers enforce (and throw on), restated for the host-only selection
// policies so a stamp never ranks a kernel that cannot launch at that shape (UINT64 survey, 2026-11-01, gaps 1, 2, 4).
//  - reductionWorkGrid (DeepRSmallK, AlignedK, RuntimeK rotated, and the ARG kernels on it): shards, component tiles
//    and outer each <= 2^31 - 1 (static_assert against REDUCTION_WORK_GRID_MAX_EXTENT in the narrow TU).
//  - OneWindow keeps 32-bit byte offsets: outer * R * K <= 2^32 - 1.
//  - RuntimeK PacketComplete takes outer, R, K and the tile count as 32-bit operands (offsets are 64-bit) and steps
//    its row counter by W <= 8 in 32 bits: outer, K <= 2^32 - 1, R <= 2^32 - 9.

#include <cstdint>

namespace ThorImplementation::CubReductionInternal::Fp8RK {

inline constexpr uint64_t FP8_RK_MAX_UINT32 = 0xFFFFFFFFull;
inline constexpr uint64_t FP8_RK_MAX_GRID_EXTENT = (uint64_t{1} << 31) - 1;
inline constexpr uint64_t FP8_RK_PACKET_COMPLETE_MAX_ROWS = FP8_RK_MAX_UINT32 - 8;

// outer * r * k <= 2^32 - 1 without overflowing.
constexpr bool fp8RKElementsFitUint32(uint64_t outer, uint64_t r, uint64_t k) {
    if (!outer || !r || !k) return false;
    if (outer > FP8_RK_MAX_UINT32 / r) return false;
    return outer * r <= FP8_RK_MAX_UINT32 / k;
}

// The reductionWorkGrid extents (x = shards, y = component tiles, z = outer).
constexpr bool fp8RKWorkGridFits(uint64_t shards, uint64_t tiles, uint64_t outer) {
    return shards && tiles && outer && shards <= FP8_RK_MAX_GRID_EXTENT && tiles <= FP8_RK_MAX_GRID_EXTENT &&
           outer <= FP8_RK_MAX_GRID_EXTENT;
}

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
