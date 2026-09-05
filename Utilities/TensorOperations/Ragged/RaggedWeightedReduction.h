#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstdint>

namespace ThorImplementation {

// Current reduction kernels traverse rows explicitly, so a scalar active count is not sufficient.
inline constexpr RaggedPartitionRequirement kRaggedWeightedReductionPartitionRequirement = RaggedPartitionRequirement::DEVICE_OFFSETS;

// Computes WeightedMean sufficient statistics over the active scalar prefix of
// a same-partition ragged values/weights pair. Only scalars before
// offsets[valid_row_count] are read; inactive packed capacity remains undefined.
// Both outputs are FP32 scalars and accumulation is performed in FP32.
void raggedWeightedMeanStatistics(const Tensor& values,
                                  const Tensor& weights,
                                  const Tensor& offsets,
                                  Tensor& numerator,
                                  Tensor& denominator,
                                  uint64_t valid_row_count,
                                  uint64_t max_total_values,
                                  uint64_t elements_per_value,
                                  Stream& stream);

}  // namespace ThorImplementation
