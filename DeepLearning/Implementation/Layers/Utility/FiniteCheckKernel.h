#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

constexpr uint32_t FINITE_CHECK_MAX_REPORTED_INDICES = 32;

enum class FiniteCheckSampleKind : uint32_t {
    NONE = 0,
    NAN_VALUE = 1,
    POSITIVE_INFINITY = 2,
    NEGATIVE_INFINITY = 3,
};

struct FiniteCheckResult {
    uint64_t totalNonFinite = 0;
    uint64_t nanCount = 0;
    uint64_t positiveInfinityCount = 0;
    uint64_t negativeInfinityCount = 0;
    uint64_t flatIndices[FINITE_CHECK_MAX_REPORTED_INDICES]{};
    uint32_t kinds[FINITE_CHECK_MAX_REPORTED_INDICES]{};
};

void launchFiniteCheck(const void *data,
                       DataType dataType,
                       uint64_t numElements,
                       uint32_t maxReportedIndices,
                       FiniteCheckResult *result,
                       Stream stream);

}  // namespace ThorImplementation
