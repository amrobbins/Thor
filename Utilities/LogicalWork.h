#pragma once

#include <cstdint>
#include <limits>
#include <optional>

namespace ThorImplementation {

/**
 * Host-side accounting for the useful logical work represented by a model
 * computation.  Logical work is deliberately independent of the physical GPU
 * implementation used to execute that computation.
 *
 * The semantic contract is:
 *
 *   - floatingPointOperations counts reference-model floating-point work, not
 *     measured SM instruction issue;
 *   - bytes counts logical tensor operand reads and result writes, not measured
 *     HBM/L2/PCIe traffic;
 *   - actual semantic problem extents are used when they are already known to
 *     the runtime (for example, valid tail-batch examples and ragged active
 *     values), rather than padded/storage capacity;
 *   - implementation optimizations such as fusion, common-subexpression
 *     elimination, caching, materialization avoidance, or conditional
 *     execution elision do not reduce logical work.  Faster execution of the
 *     same logical computation should increase the reported logical work rate;
 *   - a logical operand consumed by two authored operations is two logical
 *     reads.  A logical result produced by an operation is one logical write;
 *   - an in-place operation still logically reads and writes its value, while a
 *     pure structural/view operation contributes no tensor bytes by itself;
 *   - parameters, parameter-gradient results, and retained forward values are
 *     counted when they are logical operands/results of the model computation;
 *   - optimizer updates and parameter constraints are outside the model logical
 *     work boundary.  Producing parameter gradients remains backward logical
 *     work.  This keeps accounting invariant to optimizer fusion;
 *   - workspace/scratch traffic, cache-line transactions, register spills,
 *     synchronization/event traffic, row-partition metadata management, and
 *     other implementation bookkeeping are excluded.
 *
 * For genuine model-semantic control flow whose selected branch is available
 * only on-device, accounting must remain host-side and must not introduce D2H
 * observation or synchronization.  In that case callers should use a
 * deterministic pre-submission estimate of the semantic branch work.  Unless a
 * better host-known branch probability exists, equalBranchEstimate() provides
 * the default 50/50 estimate.  This fallback is for semantic control flow, not
 * for runtime execution optimizations, which leave logical work unchanged.
 *
 * Training aggregation uses forward + backward logical work.  Validation and
 * test aggregation use forward logical work only.
 *
 * Arithmetic helpers return std::nullopt on uint64_t overflow instead of
 * throwing.  Logical-work telemetry is diagnostic: an accounting overflow must
 * never be allowed to fail already-submitted model work.
 */
struct LogicalWorkCount {
    uint64_t floatingPointOperations = 0;
    uint64_t bytes = 0;

    [[nodiscard]] constexpr bool empty() const noexcept {
        return floatingPointOperations == 0 && bytes == 0;
    }

    [[nodiscard]] static constexpr std::optional<LogicalWorkCount> tryAdd(
        LogicalWorkCount lhs, LogicalWorkCount rhs) noexcept {
        const auto flops = tryAddUint64(lhs.floatingPointOperations, rhs.floatingPointOperations);
        if (!flops.has_value()) return std::nullopt;

        const auto logicalBytes = tryAddUint64(lhs.bytes, rhs.bytes);
        if (!logicalBytes.has_value()) return std::nullopt;

        return LogicalWorkCount{.floatingPointOperations = *flops, .bytes = *logicalBytes};
    }

    [[nodiscard]] static constexpr std::optional<LogicalWorkCount> tryScale(
        LogicalWorkCount work, uint64_t factor) noexcept {
        const auto flops = tryMultiplyUint64(work.floatingPointOperations, factor);
        if (!flops.has_value()) return std::nullopt;

        const auto logicalBytes = tryMultiplyUint64(work.bytes, factor);
        if (!logicalBytes.has_value()) return std::nullopt;

        return LogicalWorkCount{.floatingPointOperations = *flops, .bytes = *logicalBytes};
    }

    /**
     * Equal-probability estimate for genuine semantic control flow when the
     * selected branch cannot be observed host-side without synchronization.
     * Each field is rounded to the nearest integer, with exact halves rounded
     * upward.  The implementation cannot overflow for uint64_t inputs.
     */
    [[nodiscard]] static constexpr LogicalWorkCount equalBranchEstimate(
        LogicalWorkCount lhs, LogicalWorkCount rhs) noexcept {
        return LogicalWorkCount{
            .floatingPointOperations = roundedUnsignedMean(lhs.floatingPointOperations, rhs.floatingPointOperations),
            .bytes = roundedUnsignedMean(lhs.bytes, rhs.bytes)};
    }

   private:
    [[nodiscard]] static constexpr std::optional<uint64_t> tryAddUint64(uint64_t lhs, uint64_t rhs) noexcept {
        if (rhs > std::numeric_limits<uint64_t>::max() - lhs) return std::nullopt;
        return lhs + rhs;
    }

    [[nodiscard]] static constexpr std::optional<uint64_t> tryMultiplyUint64(uint64_t lhs, uint64_t rhs) noexcept {
        if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) return std::nullopt;
        return lhs * rhs;
    }

    [[nodiscard]] static constexpr uint64_t roundedUnsignedMean(uint64_t lhs, uint64_t rhs) noexcept {
        // Divide before adding so max-valued branch estimates cannot overflow.
        const uint64_t halves = lhs / 2 + rhs / 2;
        const uint64_t remainders = lhs % 2 + rhs % 2;
        return halves + (remainders + 1) / 2;
    }
};

/**
 * Returns the logical payload bytes for a tensor extent already resolved to
 * semantic active elements.  elementSizeBytes is the tensor element storage
 * size; structural metadata bytes are intentionally not part of this count.
 */
[[nodiscard]] constexpr std::optional<uint64_t> tryLogicalTensorByteCount(
    uint64_t activeElements, uint64_t elementSizeBytes) noexcept {
    if (activeElements != 0 && elementSizeBytes > std::numeric_limits<uint64_t>::max() / activeElements) {
        return std::nullopt;
    }
    return activeElements * elementSizeBytes;
}

}  // namespace ThorImplementation
