#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace ThorImplementation {

/**
 * Build the finite packed-row capacity classes used by a ragged GEMM.
 *
 * Capacities below 16 use only their full row capacity.  Larger capacities use
 * a canonical ascending schedule rather than deriving buckets backward from the
 * allocation size:
 *
 *   - powers of two from 8 through 512,
 *   - 768 between 512 and 1024,
 *   - quarter-octave buckets from 1024 upward:
 *       B, 1.25B, 1.5B, 1.75B, 2B
 *     for each power-of-two base B,
 *   - the exact full row capacity as the final bucket when it is not already a
 *     canonical bucket.
 *
 * Above 1024 this keeps normal bucket growth at 25%, substantially reducing
 * padded GEMM work for large ragged extents while retaining a small finite set
 * of pre-tuned kernels.
 *
 * Examples:
 *   15   -> {15}
 *   16   -> {8, 16}
 *   66   -> {8, 16, 32, 64, 66}
 *   1000 -> {8, 16, 32, 64, 128, 256, 512, 768, 1000}
 *   2048 -> {8, 16, 32, 64, 128, 256, 512, 768, 1024,
 *            1280, 1536, 1792, 2048}
 */
[[nodiscard]] std::vector<uint64_t> makeRaggedMatmulCapacityBuckets(uint64_t fullCapacityRows);

/**
 * Build the packed-row capacity classes used by ragged RMSNorm.
 *
 * RMSNorm mirrors the GEMM bucket schedule exactly.  Cached cuDNN graphs remain
 * bucket-specific, while each placed/stamped RMSNorm owns one execution
 * workspace sized to the maximum requirement across its bucket family.
 */
[[nodiscard]] std::vector<uint64_t> makeRaggedRmsNormCapacityBuckets(uint64_t fullCapacityRows);

/**
 * Select the smallest cached capacity that can contain activeRows.
 *
 * activeRows must be non-zero.  A future ragged execution path should handle
 * the zero-active-row case as a no-op before asking for a capacity bucket.
 */
[[nodiscard]] uint64_t chooseRaggedMatmulCapacityBucket(uint64_t activeRows, std::span<const uint64_t> capacityBuckets);

}  // namespace ThorImplementation
