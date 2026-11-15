#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorElement.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstdint>

namespace ThorImplementation {

inline constexpr RaggedPartitionRequirement kRaggedRuntimeExtentPartitionRequirement = RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT;

struct RaggedRuntimeExtent {
    // The device-resident active value count: element B of the row-partition offsets, or a one-element count tensor.
    TensorElement activeValueCount;
    uint64_t maxActiveValues = 0;
    uint64_t elementsPerValue = 1;

    [[nodiscard]] bool isInitialized() const { return activeValueCount.isInitialized(); }
    [[nodiscard]] uint64_t maxLaunchElements() const;
    [[nodiscard]] uint32_t maxGridDimX(uint32_t blockDimX) const;
};

[[nodiscard]] RaggedRuntimeExtent raggedRuntimeExtentFromOffsets(const Tensor& offsets,
                                                                 uint64_t batch_size,
                                                                 uint64_t max_total_values,
                                                                 uint64_t elements_per_value = 1);

[[nodiscard]] RaggedRuntimeExtent raggedRuntimeExtentFromActiveValueCount(const Tensor& active_value_count,
                                                                          uint64_t max_total_values,
                                                                          uint64_t elements_per_value = 1);

}  // namespace ThorImplementation
