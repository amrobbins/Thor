#pragma once

#include <algorithm>
#include <cstdint>

namespace ThorConcatenateSpanGrouping {

constexpr uint32_t kThreadsPerCta = 256;
constexpr uint32_t kMaxSpansPerCta = kThreadsPerCta;
constexpr uint64_t kTargetCtasForSmallSpans = 512;
constexpr uint64_t kBaseTargetPayloadBytesPerCta = 2048;
constexpr uint64_t kTargetPayloadBytesPerWorkSpan = 128;

// Return the largest power of two <= value, clamped to the span-grouping
// specializations implemented by the concatenate/split kernels.
inline uint32_t floorPowerOfTwoSpansPerCta(uint64_t value) {
    if (value == 0) return 1;

    uint32_t result = 1;
    while (result < kMaxSpansPerCta &&
           static_cast<uint64_t>(result) * 2 <= value) {
        result *= 2;
    }
    return result;
}

// Size the fixed 256-thread CTA from useful span work instead of preserving an
// arbitrary device-wide CTA count.
//
// spanCount controls how much row/span-level parallelism is already present.
// expectedSpanBytes limits row packing so larger copies retain enough lanes.
// Grid size remains ceil(spanCount / spansPerCta); there is no ~64-CTA target,
// SM-count dependency, or wave-count dependency.
//
// The policy comes from the forced grouping benchmark sweeps:
//   spansByWork     = floorPow2(max(1, spanCount / 512))
//   payloadBudget   = max(2048, spansByWork * 128)
//   spansByPayload  = floorPow2(max(1, payloadBudget / expectedSpanBytes))
//   spansPerCta     = min(spansByWork, spansByPayload)
//
// Examples:
//   spanCount=512,  expectedSpanBytes=32  -> 1 span/CTA
//   spanCount=1024, expectedSpanBytes=511 -> 2 spans/CTA
//   spanCount=4096, expectedSpanBytes=128 -> 8 spans/CTA
//   spanCount=4096, expectedSpanBytes=512 -> 4 spans/CTA
//   spanCount=8192, expectedSpanBytes=512 -> 4 spans/CTA
inline uint32_t selectSpansPerCta(uint64_t spanCount,
                                  uint64_t expectedSpanBytes) {
    if (spanCount == 0 || expectedSpanBytes == 0) return 1;

    const uint64_t workGroupingUnits =
        std::max<uint64_t>(1, spanCount / kTargetCtasForSmallSpans);
    const uint32_t spansByWork =
        floorPowerOfTwoSpansPerCta(workGroupingUnits);

    const uint64_t payloadBudget = std::max<uint64_t>(
        kBaseTargetPayloadBytesPerCta,
        static_cast<uint64_t>(spansByWork) *
            kTargetPayloadBytesPerWorkSpan);
    const uint64_t maxSpansByPayload =
        std::max<uint64_t>(1, payloadBudget / expectedSpanBytes);
    const uint32_t spansByPayload =
        floorPowerOfTwoSpansPerCta(maxSpansByPayload);

    return std::min(spansByWork, spansByPayload);
}

}  // namespace ThorConcatenateSpanGrouping
