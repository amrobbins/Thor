#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstdint>

namespace ThorImplementation {

// The ragged GPU diagnostic guards one packed prefix and reads only the explicit [1] active-count input.
inline constexpr RaggedPartitionRequirement kRaggedFiniteCheckGpuPartitionRequirement =
    RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT;

constexpr uint32_t FINITE_CHECK_MAX_REPORTED_INDICES = 32;

enum class FiniteCheckSampleKind : uint32_t {
    NONE = 0,
    NAN_VALUE = 1,
    POSITIVE_INFINITY = 2,
    NEGATIVE_INFINITY = 3,
};

struct FiniteCheckResult {
    uint64_t checkedElements = 0;
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

void launchRaggedFiniteCheck(const void *data,
                             DataType dataType,
                             const void *activeValueCount,
                             DataType activeCountDataType,
                             uint64_t maxTotalValues,
                             uint64_t elementsPerValue,
                             uint32_t maxReportedIndices,
                             FiniteCheckResult *result,
                             Stream stream);

}  // namespace ThorImplementation
