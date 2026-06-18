#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cstdint>

namespace ThorImplementation {

struct RaggedRuntimeExtent {
    Tensor activeValueCount;
    uint64_t maxActiveValues = 0;
    uint64_t elementsPerValue = 1;

    [[nodiscard]] bool isInitialized() const { return activeValueCount.isInitialized(); }
    [[nodiscard]] uint64_t maxLaunchElements() const;
    [[nodiscard]] uint32_t maxGridDimX(uint32_t blockDimX) const;
};

[[nodiscard]] Tensor rowPartitionActiveValueCount(const Tensor& offsets, uint64_t batch_size);

[[nodiscard]] RaggedRuntimeExtent raggedRuntimeExtentFromOffsets(const Tensor& offsets,
                                                                 uint64_t batch_size,
                                                                 uint64_t max_total_values,
                                                                 uint64_t elements_per_value = 1);

}  // namespace ThorImplementation
