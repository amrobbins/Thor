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

    // 1163: views start 256-byte aligned by default (cudaMalloc's alignment, as the reduction autotuner's ring), so a
    // timed input is aligned as a real tensor's is; at 32-byte starts consecutive views of one cell sat at different
    // offsets mod 256 and alignment-sensitive kernels timed bimodally.
    static constexpr uint64_t DefaultAlignmentBytes = 256;
    static constexpr uint64_t PaddingBytes = 128;

    explicit Fp8BenchmarkArenaCursor(uint64_t arena_bytes, uint64_t alignment_bytes = DefaultAlignmentBytes)
        : arena_bytes_(arena_bytes), workspace_start_(arena_bytes), alignment_(alignment_bytes) {
        if (alignment_ < 32 || (alignment_ & (alignment_ - 1)) != 0)
            throw std::invalid_argument("FP8 benchmark arena alignment must be a power of two >= 32");
        if (arena_bytes_ < alignment_ + PaddingBytes || arena_bytes_ % alignment_ != 0)
            throw std::invalid_argument("FP8 benchmark arena must be aligned and nonempty");
    }

    [[nodiscard]] View getBenchmarkTensorView(uint64_t bytes, Lifetime lifetime) {
        // The trailing pad and aligned capacity must fit without overflow.
        if (bytes == 0 || bytes > arena_bytes_ - PaddingBytes)
            throw std::runtime_error("FP8 benchmark tensor view exceeds arena");
        // The view and its >= 128-byte guard, rounded so the next view starts aligned too.
        const uint64_t span = align(bytes + PaddingBytes);
        if (span > arena_bytes_)
            throw std::runtime_error("FP8 benchmark tensor view plus padding exceeds arena");

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
    [[nodiscard]] uint64_t align(uint64_t x) const { return (x + alignment_ - 1) & ~(alignment_ - 1); }

    uint64_t arena_bytes_;
    uint64_t workspace_start_;
    uint64_t alignment_;
    uint64_t cursor_ = 0;
    uint64_t wraps_ = 0;
};

}  // namespace ThorImplementation::CubReductionBenchmarking
