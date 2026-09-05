#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstdint>

namespace ThorImplementation {

// Current statistics kernels traverse each valid row and therefore consume individual row boundaries.
inline constexpr RaggedPartitionRequirement kRaggedAccuracyPartitionRequirement = RaggedPartitionRequirement::DEVICE_OFFSETS;

enum class RaggedCategoricalLabelFormat { CLASS_INDEX, PER_CLASS };

// Computes BinaryAccuracy sufficient statistics over active ragged tokens.
// Predictions and labels contain one scalar per packed token. Only tokens in
// logical rows [0, valid_row_count) are read; inactive packed capacity remains
// undefined. correct_count and token_count are FP32 scalars.
void raggedBinaryAccuracyStatistics(const Tensor& predictions,
                                    const Tensor& labels,
                                    const Tensor& offsets,
                                    Tensor& correct_count,
                                    Tensor& token_count,
                                    uint64_t valid_row_count,
                                    uint64_t max_total_values,
                                    Stream& stream);

// Computes CategoricalAccuracy sufficient statistics over active ragged tokens.
// Predictions contain num_classes trailing scores per token. PER_CLASS labels
// contain the same trailing class width; CLASS_INDEX labels contain one integer
// class index per token. The ragged axis is reduced only for reporting; argmax
// is always performed independently within each active token's class axis.
void raggedCategoricalAccuracyStatistics(const Tensor& predictions,
                                         const Tensor& labels,
                                         const Tensor& offsets,
                                         Tensor& correct_count,
                                         Tensor& token_count,
                                         uint64_t valid_row_count,
                                         uint64_t max_total_values,
                                         uint64_t num_classes,
                                         RaggedCategoricalLabelFormat label_format,
                                         Stream& stream);

}  // namespace ThorImplementation
