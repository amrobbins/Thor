#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstdint>

namespace ThorImplementation {

// Partial batches need the authoritative host boundary at valid_row_count, not
// merely the partition's full active count. The DynamicExpression retains the
// canonical offsets carrier for that host publication; the CUDA reduction no
// longer reads the device offsets payload.
inline constexpr RaggedPartitionRequirement kRaggedWeightedReductionPartitionRequirement = RaggedPartitionRequirement::DEVICE_OFFSETS;

// Workspace is the maximum one FP32 {numerator, denominator} pair per first-pass
// CTA required by packed capacity. Runtime launches consume only the prefix
// required by the authoritative active_value_count and can therefore reuse this
// allocation across batches with very different active extents.
TensorDescriptor raggedWeightedMeanStatisticsWorkspaceDescriptor(uint64_t max_total_values,
                                                                  uint64_t elements_per_value);

// Computes WeightedMean sufficient statistics over the first active_value_count
// packed values of a same-partition ragged values/weights pair. Inactive packed
// capacity is never read. The first pass is sized from the active scalar prefix,
// not from capacity. A one-CTA reduction writes the final FP32 statistics
// directly; larger reductions write unique non-atomic CTA partials into the
// reusable capacity-sized workspace and finish with one final CTA.
void raggedWeightedMeanStatistics(const Tensor& values,
                                  const Tensor& weights,
                                  Tensor& partial_statistics,
                                  Tensor& numerator,
                                  Tensor& denominator,
                                  uint64_t active_value_count,
                                  uint64_t max_total_values,
                                  uint64_t elements_per_value,
                                  Stream& stream);

// Benchmark-only selector/query and forcing hook. forced_partial_count must be
// non-zero and no larger than the capacity-sized production workspace. The
// exact production kernels are used; only the first-pass grid size is forced.
uint32_t raggedWeightedMeanPartialBlockCountForBenchmark(uint64_t active_scalar_count);

void raggedWeightedMeanStatisticsWithPartialCountForBenchmark(
    const Tensor& values,
    const Tensor& weights,
    Tensor& partial_statistics,
    Tensor& numerator,
    Tensor& denominator,
    uint64_t active_value_count,
    uint64_t max_total_values,
    uint64_t elements_per_value,
    uint32_t forced_partial_count,
    Stream& stream);

}  // namespace ThorImplementation
