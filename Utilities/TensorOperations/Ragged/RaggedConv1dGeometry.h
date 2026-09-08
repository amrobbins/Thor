#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace ThorImplementation {

enum class RaggedConv1dPaddingMode : uint8_t {
    VALID,
    SAME_UPPER,
    CAUSAL,
    EXPLICIT,
};

struct RaggedConv1dResolvedPadding {
    uint64_t left = 0;
    uint64_t right = 0;

    bool operator==(const RaggedConv1dResolvedPadding& other) const = default;
};

/**
 * Semantic row geometry for a ragged 1D convolution.
 *
 * Unlike ConvolutionSpatial1d, SAME_UPPER is intentionally not resolved here:
 * each ragged row can have a different active length and therefore a different
 * SAME_UPPER left/right padding split. This descriptor is host-only geometry in
 * CR2.1; the existing causal execution path remains unchanged.
 */
struct RaggedConv1dGeometry {
    RaggedConv1dPaddingMode padding_mode = RaggedConv1dPaddingMode::VALID;
    uint64_t kernel_width = 1;
    int32_t stride = 1;
    int32_t dilation = 1;
    uint64_t explicit_left_padding = 0;
    uint64_t explicit_right_padding = 0;

    bool operator==(const RaggedConv1dGeometry& other) const = default;

    [[nodiscard]] static RaggedConv1dGeometry valid(uint64_t kernelWidth,
                                                     int32_t stride = 1,
                                                     int32_t dilation = 1);
    [[nodiscard]] static RaggedConv1dGeometry sameUpper(uint64_t kernelWidth,
                                                         int32_t stride = 1,
                                                         int32_t dilation = 1);
    [[nodiscard]] static RaggedConv1dGeometry causal(uint64_t kernelWidth,
                                                      int32_t stride = 1,
                                                      int32_t dilation = 1);
    [[nodiscard]] static RaggedConv1dGeometry explicitPadding(uint64_t kernelWidth,
                                                               uint64_t leftPadding,
                                                               uint64_t rightPadding,
                                                               int32_t stride = 1,
                                                               int32_t dilation = 1);

    [[nodiscard]] uint64_t effectiveKernelWidth() const;
    [[nodiscard]] RaggedConv1dResolvedPadding resolvedPadding(uint64_t rowLength) const;
    [[nodiscard]] uint64_t outputLength(uint64_t rowLength) const;

    // True iff this geometry maps every possible row length L to exactly L.
    [[nodiscard]] bool preservesRowLengths() const;

    void validate() const;
};

struct RaggedConv1dOutputCapacity {
    uint64_t max_values_per_row = 0;
    uint64_t max_total_values = 0;

    bool operator==(const RaggedConv1dOutputCapacity& other) const = default;
};

/**
 * Derive exact output offsets from authoritative host input offsets.
 *
 * inputOffsets must be a canonical row-offset vector: non-empty, beginning at
 * zero, and monotonically non-decreasing. The returned vector has the same
 * length and begins at zero.
 */
[[nodiscard]] std::vector<uint64_t> deriveRaggedConv1dOutputOffsets(
    std::span<const uint64_t> inputOffsets,
    const RaggedConv1dGeometry& geometry);

/**
 * Derive conservative placement-time output capacities from the input row
 * capacity and batch size. max_total_values is intentionally conservative:
 * every row is allowed to reach the output length implied by
 * inputMaxValuesPerRow. CR2 can tighten this later without changing semantics.
 */
[[nodiscard]] RaggedConv1dOutputCapacity deriveRaggedConv1dOutputCapacity(
    uint64_t batchSize,
    uint64_t inputMaxValuesPerRow,
    const RaggedConv1dGeometry& geometry);

}  // namespace ThorImplementation
