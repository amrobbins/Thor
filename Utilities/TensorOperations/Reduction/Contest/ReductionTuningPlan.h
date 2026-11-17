#pragma once

// The contest's shortlist and rounds at each tuning depth, shared by the RK and KR selectors of every dtype.

#include "Utilities/TensorOperations/Reduction/ReductionTuningDepth.h"

#include <cstddef>

namespace ThorImplementation::ReductionInternal {

// The autotuned shortlist per depth (ReductionTuningDepth): candidates timed and rounds per candidate.
// trimmed: how many of the fastest and of the slowest rounds each candidate drops.
struct ReductionTuningPlan {
    size_t candidates = 0;
    size_t rounds = 0;
    size_t trimmed = 1;
};
[[nodiscard]] constexpr ReductionTuningPlan reductionTuningPlan(ReductionTuningDepth depth) {
    switch (depth) {
        case ReductionTuningDepth::Quick: return {6, 5, 1};
        case ReductionTuningDepth::Regular: return {12, 7, 1};
        case ReductionTuningDepth::Ultra: return {24, 11, 2};
    }
    return {12, 7, 1};
}

}  // namespace ThorImplementation::ReductionInternal
