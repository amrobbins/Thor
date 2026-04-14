#pragma once

#include <cstdint>
#include <vector>

namespace ThorImplementation {

std::vector<uint64_t> normalizeSqueezeAxesForInputDims(const std::vector<uint64_t>& input_dims, const std::vector<uint64_t>& squeeze_axes);

std::vector<uint64_t> normalizedReductionUnsqueezeAxes(const std::vector<uint64_t>& input_dims,
                                                       const std::vector<uint64_t>& reduction_axes,
                                                       const std::vector<uint64_t>& squeeze_axes);

std::vector<uint64_t> normalizeUnsqueezeAxesForInputDims(const std::vector<uint64_t>& input_dims,
                                                         const std::vector<uint64_t>& unsqueeze_axes);

}  // namespace ThorImplementation
