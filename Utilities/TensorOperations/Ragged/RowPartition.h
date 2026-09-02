#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"
#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <cstdint>

namespace ThorImplementation {

enum RowPartitionValidationErrorBits : uint32_t {
    ROW_PARTITION_VALID = 0U,
    ROW_PARTITION_OFFSETS_MUST_START_AT_ZERO = 1U << 0U,
    ROW_PARTITION_OFFSETS_MUST_BE_MONOTONIC = 1U << 1U,
    ROW_PARTITION_OFFSETS_EXCEED_CAPACITY = 1U << 2U,
    ROW_PARTITION_ROW_LENGTH_EXCEEDS_MAX = 1U << 3U,
    ROW_PARTITION_ROW_LENGTH_EXCEEDS_INT32 = 1U << 4U,
};

struct RowPartitionLengthsToOffsetsPlan {
    TensorPlacement placement;
    DataType dtype = kDefaultRowPartitionOffsetDataType;
    uint64_t batch_size = 0;
    size_t temp_storage_bytes = 0;
};

[[nodiscard]] bool isRowPartitionOffsetDTypeSupported(DataType dtype);

[[nodiscard]] RowPartitionLengthsToOffsetsPlan prepareRowPartitionLengthsToOffsets(const Tensor& lengths,
                                                                                  const Tensor& offsets,
                                                                                  uint64_t batch_size);

[[nodiscard]] size_t rowPartitionLengthsToOffsetsTempBytes(const Tensor& lengths,
                                                           const Tensor& offsets,
                                                           uint64_t batch_size);

void rowPartitionLengthsToOffsets(const RowPartitionLengthsToOffsetsPlan& plan,
                                  const Tensor& temp_storage,
                                  const Tensor& lengths,
                                  Tensor& offsets,
                                  Stream& stream);

void rowPartitionLengthsToOffsets(const Tensor& temp_storage,
                                  size_t temp_storage_bytes,
                                  const Tensor& lengths,
                                  Tensor& offsets,
                                  uint64_t batch_size,
                                  Stream& stream);

void rowPartitionOffsetsToLengths(const Tensor& offsets, Tensor& lengths, uint64_t batch_size, Stream& stream);

// Writes offsets[valid_row_count] * elements_per_value to one FP32 scalar.
// This is a structural statistic: it reads only the row partition, never the
// reserved packed values capacity.  It is used by ragged ratio metrics to
// count active scalar contributions without materializing an FP32 tensor of
// packed ones.
void rowPartitionActiveScalarCount(const Tensor& offsets,
                                   Tensor& active_scalar_count,
                                   uint64_t valid_row_count,
                                   uint64_t elements_per_value,
                                   Stream& stream);

// Copies a canonical [batch_size + 1] row partition while making rows at or
// after valid_row_count empty.  Values after the valid-row boundary are never
// made active by the resulting partition.
void rowPartitionClampOffsetsToValidRows(const Tensor& offsets,
                                         Tensor& clamped_offsets,
                                         uint64_t batch_size,
                                         uint64_t valid_row_count,
                                         Stream& stream);

// Converts canonical UINT32/UINT64 offsets to INT32 per-row lengths for
// backends such as cuDNN CTC. Validation is entirely device-side: malformed
// partitions, offsets beyond max_total_values, rows beyond max_allowed_length,
// and rows that cannot be represented by INT32 are reported through
// validation_error_bits. If any validation bit is set, the entire lengths
// output is zeroed before subsequent work on `stream` can consume it. No host
// readback or synchronization is performed.
void rowPartitionOffsetsToInt32LengthsChecked(const Tensor& offsets,
                                               Tensor& lengths,
                                               Tensor& validation_error_bits,
                                               uint64_t batch_size,
                                               uint64_t max_total_values,
                                               uint64_t max_allowed_length,
                                               Stream& stream);

void rowPartitionOffsetsToRowIds(const Tensor& offsets,
                                 Tensor& row_ids,
                                 uint64_t batch_size,
                                 uint64_t max_total_values,
                                 Stream& stream);

// Materializes one FP32 absolute RoPE position per logical packed ragged value. Each logical row starts at
// row_position_offsets[row] when provided, otherwise scalar_position_offset, and advances by one per value.
// Storage after offsets[batch_size] is untouched and remains undefined; consumers that need a logical aggregate
// must use the row partition rather than reducing over full packed capacity. row_position_offsets is structural
// INT32 metadata with one value per logical row.
void rowPartitionOffsetsToRopePositionIds(const Tensor& offsets,
                                          const Tensor* row_position_offsets,
                                          int64_t scalar_position_offset,
                                          Tensor& position_ids,
                                          uint64_t batch_size,
                                          uint64_t max_total_values,
                                          Stream& stream);

void rowPartitionValidateOffsetsDebug(const Tensor& offsets,
                                      Tensor& validation_error_bits,
                                      uint64_t batch_size,
                                      uint64_t max_total_values,
                                      Stream& stream);

}  // namespace ThorImplementation
