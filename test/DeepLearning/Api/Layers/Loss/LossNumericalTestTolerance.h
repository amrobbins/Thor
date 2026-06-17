#pragma once

#include "DeepLearning/Implementation/Layers/Loss.h"

#include <algorithm>

namespace ThorTest {

// These numerical-gradient checks compare analytic gradients against finite-difference
// references that are multiplied by Loss::getLossScalingFactor().  The original
// absolute tolerances were chosen for the historical loss scale of 4.  Preserve
// that strictness for smaller scales, but widen proportionally for larger scales
// so changing the global loss scale does not create tolerance-only failures.
inline float lossScaleAwareGradientTolerance(float baseTolerance) {
    constexpr float kReferenceLossScalingFactor = 4.0f;
    const float toleranceScale = ThorImplementation::Loss::getLossScalingFactor() / kReferenceLossScalingFactor;
    return baseTolerance * std::max(1.0f, toleranceScale);
}

}  // namespace ThorTest
