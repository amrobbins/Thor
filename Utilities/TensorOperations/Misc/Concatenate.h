#pragma once

#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Static geometry for one source/destination participating in a dense
// concatenate/split. For each fixed coordinate before the concatenation axis,
// one array contributes exactly one contiguous span.
struct ConcatenateSpanGeometry {
    uint64_t spanBytes;
    uint64_t packedOffsetBytes;
};

std::vector<ConcatenateSpanGeometry> buildConcatenateSpanGeometry(
    std::size_t elementSizeBytes,
    uint64_t innerElements,
    const std::vector<uint64_t>& axisElementsPerArray);

// outerSlices is the number of valid coordinates before the concatenation axis.
// For a non-batch physical axis this can be restricted to the valid batch prefix.
void launchConcatenate(void *dest,
                       void *source[],
                       uint64_t outerSlices,
                       uint32_t numSourceArrays,
                       uint64_t packedSliceBytes,
                       const ConcatenateSpanGeometry spanGeometry[],
                       Stream stream);

// Benchmark-only launch hook that forces one existing spans-per-CTA specialization.
// Production callers should use launchConcatenate().
void launchConcatenateWithSpansPerCtaForBenchmark(void *dest,
                                                  void *source[],
                                                  uint64_t outerSlices,
                                                  uint32_t numSourceArrays,
                                                  uint64_t packedSliceBytes,
                                                  const ConcatenateSpanGeometry spanGeometry[],
                                                  uint32_t spansPerCta,
                                                  Stream stream);
