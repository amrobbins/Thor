#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstdint>

namespace ThorImplementation {

// Partial batches need offsets[valid_row_count], not merely the partition's full
// active count. The reduction consumes only that terminal offset; it does not
// traverse row boundaries.
inline constexpr RaggedPartitionRequirement kRaggedWeightedReductionPartitionRequirement = RaggedPartitionRequirement::DEVICE_OFFSETS;

// Workspace is one FP32 {numerator, denominator} pair per first-pass CTA. The
// descriptor depends only on packed capacity and can therefore be allocated once
// when the metric expression is prepared and reused by every batch.
TensorDescriptor raggedWeightedMeanStatisticsWorkspaceDescriptor(uint64_t max_total_values,
                                                                  uint64_t elements_per_value);

// Computes WeightedMean sufficient statistics over the active scalar prefix of
// a same-partition ragged values/weights pair. Only scalars before
// offsets[valid_row_count] are read; inactive packed capacity remains undefined.
// The first pass writes non-atomic per-CTA partials into partial_statistics and a
// final CTA reduces those partials into the FP32 scalar outputs.
void raggedWeightedMeanStatistics(const Tensor& values,
                                  const Tensor& weights,
                                  const Tensor& offsets,
                                  Tensor& partial_statistics,
                                  Tensor& numerator,
                                  Tensor& denominator,
                                  uint64_t valid_row_count,
                                  uint64_t max_total_values,
                                  uint64_t elements_per_value,
                                  Stream& stream);

}  // namespace ThorImplementation
