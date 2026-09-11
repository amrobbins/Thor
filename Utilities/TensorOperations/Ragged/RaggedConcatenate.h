#pragma once

#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ThorImplementation {
// Trailing-axis concatenate/split is currently routed a managed active-count
// carrier. CUDA does not read its payload; the authoritative host publication on
// that carrier supplies the packed-prefix launch extent.
inline constexpr RaggedPartitionRequirement kRaggedTrailingConcatenatePartitionRequirement = RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT;
}  // namespace ThorImplementation

// Static byte geometry for one source/destination participating in a trailing
// ragged concatenate. This table is built once from tensor shapes and reused by
// forward and backward CUDA launches.
struct RaggedConcatenateSpanGeometry {
    uint64_t spanBytes;
    uint64_t valueBytes;
    uint64_t outputOffsetBytes;
};

std::vector<RaggedConcatenateSpanGeometry> buildRaggedConcatenateSpanGeometry(
    std::size_t elementSizeBytes,
    uint64_t outerSlicesPerValue,
    uint64_t innerElements,
    const std::vector<uint64_t>& axisOffsets);

// Concatenate/split contiguous packed ragged values along a trailing feature
// axis while touching only [0, activeRows). activeRows is the authoritative
// host-published packed-prefix extent for the current valid-example prefix.
// Inactive packed capacity is deliberately neither read nor canonicalized.
//
// spanGeometry is a device array with numArrays entries built once from static
// tensor shape. output/source value byte strides remain 64-bit so tensors larger
// than 4 GiB are supported while ordinary span indexing stays 32-bit.
void launchRaggedConcatenate(void *dest,
                             void *source[],
                             uint64_t capacityRows,
                             uint64_t outputValueBytes,
                             uint64_t outerSlicesPerValue,
                             uint32_t numSourceArrays,
                             const RaggedConcatenateSpanGeometry spanGeometry[],
                             uint64_t activeRows,
                             Stream stream);

void launchRaggedSplit(void *dest[],
                       void *source,
                       uint64_t capacityRows,
                       uint64_t sourceValueBytes,
                       uint64_t outerSlicesPerValue,
                       uint32_t numDestArrays,
                       const RaggedConcatenateSpanGeometry spanGeometry[],
                       uint64_t activeRows,
                       Stream stream);

// Benchmark-only launch hooks that force one existing spans-per-CTA specialization.
// Production callers should use launchRaggedConcatenate()/launchRaggedSplit().
void launchRaggedConcatenateWithSpansPerCtaForBenchmark(
    void *dest, void *source[], uint64_t capacityRows, uint64_t outputValueBytes,
    uint64_t outerSlicesPerValue, uint32_t numSourceArrays,
    const RaggedConcatenateSpanGeometry spanGeometry[], uint64_t activeRows,
    uint32_t spansPerCta, Stream stream);

void launchRaggedSplitWithSpansPerCtaForBenchmark(
    void *dest[], void *source, uint64_t capacityRows, uint64_t sourceValueBytes,
    uint64_t outerSlicesPerValue, uint32_t numDestArrays,
    const RaggedConcatenateSpanGeometry spanGeometry[], uint64_t activeRows,
    uint32_t spansPerCta, Stream stream);
