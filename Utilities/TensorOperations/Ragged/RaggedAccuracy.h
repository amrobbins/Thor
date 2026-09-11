#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstdint>

namespace ThorImplementation {

// Accuracy depends only on the authoritative active packed-token prefix. Row
// boundaries are irrelevant to both BinaryAccuracy and CategoricalAccuracy, so
// the metric consumes HOST_EXTENT metadata and does not force a device offsets
// representation merely to discover its execution extent.
inline constexpr RaggedPartitionRequirement kRaggedAccuracyPartitionRequirement =
    RaggedPartitionRequirement::HOST_EXTENT;

enum class RaggedCategoricalLabelFormat { CLASS_INDEX, PER_CLASS };

// Capacity-sized workspace for unique first-pass CTA correct-count partials.
// The descriptor uses UINT64 so the same reusable storage serves both the
// ordinary UINT32 and large UINT64 execution paths.
TensorDescriptor raggedAccuracyStatisticsWorkspaceDescriptor(uint64_t max_total_values,
                                                              uint64_t num_classes = 1);

// Computes BinaryAccuracy sufficient statistics over the first
// active_value_count packed tokens. Inactive packed capacity is never read.
// The first pass is sized from the active prefix and emits unique, non-atomic
// CTA partials; a one-CTA workload writes the final statistics directly.
void raggedBinaryAccuracyStatistics(const Tensor& predictions,
                                    const Tensor& labels,
                                    Tensor& partial_correct_counts,
                                    Tensor& correct_count,
                                    Tensor& token_count,
                                    uint64_t active_value_count,
                                    uint64_t max_total_values,
                                    Stream& stream);

// Computes CategoricalAccuracy sufficient statistics over the first
// active_value_count packed tokens. Predictions contain num_classes trailing
// scores per token. PER_CLASS labels contain the same trailing class width;
// CLASS_INDEX labels contain one integer class index per token. Class argmax is
// evaluated cooperatively by a 2/4/8/16/32-lane group while preserving ordinary
// first-occurrence tie behavior.
void raggedCategoricalAccuracyStatistics(const Tensor& predictions,
                                         const Tensor& labels,
                                         Tensor& partial_correct_counts,
                                         Tensor& correct_count,
                                         Tensor& token_count,
                                         uint64_t active_value_count,
                                         uint64_t max_total_values,
                                         uint64_t num_classes,
                                         RaggedCategoricalLabelFormat label_format,
                                         Stream& stream);

// Benchmark-only selectors and forcing hooks. A forced value of 0 means
// production auto. These entry points launch the same production kernels and
// exist to make transition/geometry sweeps measurable without benchmark-side
// kernel copies.
uint32_t raggedBinaryAccuracyPartialBlockCountForBenchmark(uint64_t active_value_count);
uint32_t raggedCategoricalAccuracyLanesPerTokenForBenchmark(uint64_t num_classes);
uint32_t raggedCategoricalAccuracyPartialBlockCountForBenchmark(
    uint64_t active_value_count,
    uint64_t num_classes,
    uint32_t forced_lanes_per_token = 0);

void raggedBinaryAccuracyStatisticsForBenchmark(const Tensor& predictions,
                                                const Tensor& labels,
                                                Tensor& partial_correct_counts,
                                                Tensor& correct_count,
                                                Tensor& token_count,
                                                uint64_t active_value_count,
                                                uint64_t max_total_values,
                                                uint32_t forced_partial_count,
                                                Stream& stream);

void raggedCategoricalAccuracyStatisticsForBenchmark(
    const Tensor& predictions,
    const Tensor& labels,
    Tensor& partial_correct_counts,
    Tensor& correct_count,
    Tensor& token_count,
    uint64_t active_value_count,
    uint64_t max_total_values,
    uint64_t num_classes,
    RaggedCategoricalLabelFormat label_format,
    uint32_t forced_partial_count,
    uint32_t forced_lanes_per_token,
    Stream& stream);

}  // namespace ThorImplementation
