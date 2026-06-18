#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <cstdint>

namespace ThorImplementation {

enum RowPartitionValidationErrorBits : uint32_t {
    ROW_PARTITION_VALID = 0U,
    ROW_PARTITION_OFFSETS_MUST_START_AT_ZERO = 1U << 0U,
    ROW_PARTITION_OFFSETS_MUST_BE_MONOTONIC = 1U << 1U,
    ROW_PARTITION_OFFSETS_EXCEED_CAPACITY = 1U << 2U,
};

struct RowPartitionLengthsToOffsetsPlan {
    TensorPlacement placement;
    DataType dtype = DataType::UINT32;
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

void rowPartitionOffsetsToRowIds(const Tensor& offsets,
                                 Tensor& row_ids,
                                 uint64_t batch_size,
                                 uint64_t max_total_values,
                                 Stream& stream);

void rowPartitionValidateOffsetsDebug(const Tensor& offsets,
                                      Tensor& validation_error_bits,
                                      uint64_t batch_size,
                                      uint64_t max_total_values,
                                      Stream& stream);

}  // namespace ThorImplementation
