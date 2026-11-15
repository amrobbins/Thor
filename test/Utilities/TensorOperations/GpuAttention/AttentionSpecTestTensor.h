#pragma once

#include "Utilities/TensorOperations/GpuAttention/CudnnAttention.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ThorImplementation {

// The physical (storage-order) dimensions of the dense tensor an attention spec describes: the spec's semantic
// [B,H,S,D] axes ordered by decreasing stride (BHSD stays BHSD, BSHD becomes [B,S,H,D]). Tests allocate a real
// tensor with these dimensions and hand it to cuDNN attention directly, as Thor does — no view.
inline std::vector<uint64_t> physicalDimensionsForAttentionSpec(const AttentionTensorSpec& spec) {
    const size_t rank = spec.dimensions.size();
    if (rank == 0 || spec.strides.size() != rank) {
        throw std::invalid_argument("attention test spec needs matching dimensions and strides");
    }
    std::vector<size_t> order(rank);
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return spec.strides[a] > spec.strides[b]; });

    std::vector<uint64_t> physical(rank);
    for (size_t i = 0; i < rank; ++i) {
        physical[i] = static_cast<uint64_t>(spec.dimensions[order[i]]);
    }
    int64_t dense = 1;
    for (size_t i = rank; i-- > 0;) {
        const size_t axis = order[i];
        if (spec.dimensions[axis] != 1 && spec.strides[axis] != dense) {
            throw std::invalid_argument("attention test spec does not describe a dense tensor");
        }
        dense *= spec.dimensions[axis];
    }
    return physical;
}

}  // namespace ThorImplementation
