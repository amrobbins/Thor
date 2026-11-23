#pragma once

#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ThorImplementation {
// Trailing-axis concatenate/split is currently routed a managed active-count
// carrier. CUDA does not read its payload; the authoritative host publication on
// that carrier supplies the packed-prefix launch extent.
inline constexpr RaggedPartitionRequirement kRaggedTrailingConcatenatePartitionRequirement = RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT;
}  // namespace ThorImplementation

/**
 * A concatenate of packed ragged values along a trailing axis is a dense concatenate (launchConcatenate,
 * launchSplit): packed value r of every input is the r-th of its rows, and the concatenation axis splits each value
 * into outerSlicesPerValue slices (the product of the dimensions between the row axis and it), so the active prefix
 * of activeRows values is activeRows * outerSlicesPerValue outer slices of the dense concatenate, and array i's span
 * of a slice is its extent on the axis times the elements after it. Inactive packed capacity is neither read nor
 * written. The spans are in elements, so the inputs and the output may each have their own type.
 */
struct RaggedTrailingConcatenateGeometry {
    std::vector<uint64_t> spanElements;
    uint64_t outerSlicesPerValue = 1;

    // The dense concatenate's outer slices for an active prefix of activeRows packed values.
    uint64_t outerSlicesFor(uint64_t activeRows) const {
        if (outerSlicesPerValue != 0 && activeRows > std::numeric_limits<uint64_t>::max() / outerSlicesPerValue)
            throw std::invalid_argument("Ragged concatenate outer slice count overflow.");
        return activeRows * outerSlicesPerValue;
    }
};

// valueDimensions are the output's packed values' dimensions, the row axis first; axis is the concatenation axis,
// a trailing one; axisElementsPerInput the inputs' extents on it, which sum to the output's.
inline RaggedTrailingConcatenateGeometry buildRaggedTrailingConcatenateGeometry(const std::vector<uint64_t> &valueDimensions,
                                                                               uint32_t axis,
                                                                               const std::vector<uint64_t> &axisElementsPerInput) {
    if (axis == 0 || axis >= valueDimensions.size())
        throw std::invalid_argument("Ragged concatenate axis must name a trailing packed-value dimension.");
    const auto checkedMultiply = [](uint64_t lhs, uint64_t rhs) {
        if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
            throw std::invalid_argument("Ragged concatenate geometry overflow.");
        return lhs * rhs;
    };
    RaggedTrailingConcatenateGeometry geometry;
    for (uint32_t d = 1; d < axis; ++d) geometry.outerSlicesPerValue = checkedMultiply(geometry.outerSlicesPerValue, valueDimensions[d]);
    uint64_t innerElements = 1;
    for (size_t d = axis + 1; d < valueDimensions.size(); ++d) innerElements = checkedMultiply(innerElements, valueDimensions[d]);
    uint64_t axisElements = 0;
    for (const uint64_t elements : axisElementsPerInput) {
        if (elements > std::numeric_limits<uint64_t>::max() - axisElements)
            throw std::invalid_argument("Ragged concatenate geometry overflow.");
        axisElements += elements;
        geometry.spanElements.push_back(checkedMultiply(elements, innerElements));
    }
    if (axisElements != valueDimensions[axis])
        throw std::invalid_argument("Ragged concatenate input extents must sum to the output's on the concatenation axis.");
    return geometry;
}
