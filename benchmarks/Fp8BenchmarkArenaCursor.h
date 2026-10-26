#pragma once

#include <cstdint>
#include <stdexcept>

// Host-only bookkeeping for the FP8 cold-input benchmark's single allocation.
// The backing Tensor is owned by the CUDA-facing benchmark, not this cursor.
//
// Streaming inputs advance through the arena and wrap at the current workspace
// reservation. Reusable stage views are stable, overlapping suffixes of the
// SAME allocation: only one stage+continuation runs at a time on one stream.
// No fixed input/workspace partition is necessary. The workspace high-water
// mark is monotonic: stage-written bytes never re-enter the cold input ring.
namespace ThorImplementation::CubReductionBenchmarking {

class Fp8BenchmarkArenaCursor {
public:
    enum class Lifetime { StreamingInput, ReusableStage };
    struct View {
        uint64_t offset;
        uint64_t bytes;
    };

    static constexpr uint64_t AlignmentBytes = 32;
    static constexpr uint64_t PaddingBytes = 128;

    explicit Fp8BenchmarkArenaCursor(uint64_t arena_bytes)
        : arena_bytes_(arena_bytes), workspace_start_(arena_bytes) {
        if (arena_bytes_ < AlignmentBytes + PaddingBytes ||
            arena_bytes_ % AlignmentBytes != 0)
            throw std::invalid_argument("FP8 benchmark arena must be 32-byte aligned and nonempty");
    }

    [[nodiscard]] View getBenchmarkTensorView(uint64_t bytes, Lifetime lifetime) {
        // The trailing pad and aligned capacity must fit without overflow.
        if (bytes == 0 || bytes > arena_bytes_ - PaddingBytes)
            throw std::runtime_error("FP8 benchmark tensor view exceeds arena");
        const uint64_t aligned_bytes = align32(bytes);
        if (aligned_bytes > arena_bytes_ - PaddingBytes)
            throw std::runtime_error("FP8 benchmark tensor view plus padding exceeds arena");
        const uint64_t span = aligned_bytes + PaddingBytes;

        if (lifetime == Lifetime::ReusableStage) {
            // All stages end at the same address before the guard. Their views
            // overlap deliberately; launches and continuations are serialized.
            const uint64_t offset = arena_bytes_ - span;
            if (offset < workspace_start_) workspace_start_ = offset;
            return {offset, bytes};
        }

        // A staged output must never overlap a live input. Recheck the bound
        // every call because preparing more stage plans can expand the suffix.
        if (span > workspace_start_)
            throw std::runtime_error("FP8 benchmark input cannot fit beside active stage workspace");
        uint64_t offset = cursor_;
        if (offset > workspace_start_ || span > workspace_start_ - offset) {
            offset = 0;
            ++wraps_;
        }
        cursor_ = offset + span;
        return {offset, bytes};
    }

    [[nodiscard]] uint64_t wraps() const { return wraps_; }
    [[nodiscard]] uint64_t workspaceBytes() const { return arena_bytes_ - workspace_start_; }
    [[nodiscard]] uint64_t streamingCapacity() const { return workspace_start_; }

private:
    static constexpr uint64_t align32(uint64_t x) {
        return (x + AlignmentBytes - 1) & ~(AlignmentBytes - 1);
    }

    uint64_t arena_bytes_;
    uint64_t workspace_start_;
    uint64_t cursor_ = 0;
    uint64_t wraps_ = 0;
};

}  // namespace ThorImplementation::CubReductionBenchmarking
