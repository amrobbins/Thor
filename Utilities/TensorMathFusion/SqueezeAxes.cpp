#include "Utilities/TensorMathFusion/SqueezeAxes.h"

#include "Utilities/TensorMathFusion/StampedEquation.h"

#include <algorithm>
#include <stdexcept>

namespace ThorImplementation {

std::vector<uint64_t> normalizeSqueezeAxesForInputDims(const std::vector<uint64_t>& input_dims, const std::vector<uint64_t>& squeeze_axes) {
    if (squeeze_axes.empty()) {
        return {};
    }

    std::vector<uint64_t> normalized = squeeze_axes;
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());

    if (normalized.size() == 1 && normalized[0] == UINT64_MAX) {
        std::vector<uint64_t> actual_axes;
        actual_axes.reserve(input_dims.size());
        for (uint64_t axis = 0; axis < input_dims.size(); ++axis) {
            if (input_dims[axis] == 1) {
                actual_axes.push_back(axis);
            }
        }
        return actual_axes;
    }

    for (uint64_t axis : normalized) {
        if (axis >= input_dims.size()) {
            throw std::runtime_error("squeeze axes are invalid for the input rank.");
        }
        if (input_dims[axis] != 1) {
            throw std::runtime_error("squeeze axes must refer to singleton dimensions.");
        }
    }

    return normalized;
}

std::vector<uint64_t> normalizedReductionUnsqueezeAxes(const std::vector<uint64_t>& input_dims,
                                                       const std::vector<uint64_t>& reduction_axes,
                                                       const std::vector<uint64_t>& squeeze_axes) {
    const std::vector<uint64_t> unsqueezed_output_dims = StampedEquation::computeReductionOutputDims(input_dims, reduction_axes, {});
    return normalizeSqueezeAxesForInputDims(unsqueezed_output_dims, squeeze_axes);
}

std::vector<uint64_t> normalizeUnsqueezeAxesForInputDims(const std::vector<uint64_t>& input_dims,
                                                         const std::vector<uint64_t>& unsqueeze_axes) {
    std::vector<uint64_t> normalized = unsqueeze_axes;
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());

    const uint64_t output_rank = input_dims.size() + normalized.size();
    for (uint64_t axis : normalized) {
        if (axis >= output_rank) {
            throw std::runtime_error("unsqueeze axes are invalid for the input rank.");
        }
    }

    return normalized;
}

}  // namespace ThorImplementation
