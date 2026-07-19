#pragma once

#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RowPartition.h"

#include <cstddef>
#include <cstdint>

namespace ThorImplementation {

// Explicit dense/ragged compatibility adapters. All output and temporary
// storage is caller-owned. Each conversion validates the device-resident row
// partition on `stream` and writes a RowPartitionValidationErrorBits mask to
// `validation_error_bits`. The adapter performs no logical value copies when
// that mask is non-zero; callers may inspect it asynchronously or as part of a
// larger device-side validation policy. No host readback is performed here.
struct RaggedFromDenseWithLengthsPlan {
    TensorPlacement placement;
    DataType valuesDataType = DataType::FP32;
    DataType offsetsDataType = kDefaultRowPartitionOffsetDataType;
    uint64_t batchSize = 0;
    uint64_t maxLength = 0;
    uint64_t maxTotalValues = 0;
    uint64_t elementsPerValue = 1;
    uint64_t valueElementSizeBytes = 0;
    size_t tempStorageBytes = 0;
};

[[nodiscard]] RaggedFromDenseWithLengthsPlan prepareRaggedFromDenseWithLengths(const Tensor& dense,
                                                                              const Tensor& lengths,
                                                                              const Tensor& values,
                                                                              const Tensor& offsets);

[[nodiscard]] size_t raggedFromDenseWithLengthsTempBytes(const Tensor& dense,
                                                         const Tensor& lengths,
                                                         const Tensor& values,
                                                         const Tensor& offsets);

RaggedTensor raggedFromDense(const RaggedFromDenseWithLengthsPlan& plan,
                             const Tensor& temp_storage,
                             const Tensor& dense,
                             const Tensor& lengths,
                             Tensor& values,
                             Tensor& offsets,
                             Tensor& validation_error_bits,
                             Stream& stream);

RaggedTensor raggedFromDense(const Tensor& temp_storage,
                             size_t temp_storage_bytes,
                             const Tensor& dense,
                             const Tensor& lengths,
                             Tensor& values,
                             Tensor& offsets,
                             Tensor& validation_error_bits,
                             Stream& stream);

RaggedTensor raggedFromDense(const Tensor& dense,
                             const Tensor& offsets,
                             Tensor& values,
                             Tensor& validation_error_bits,
                             Stream& stream);

void raggedToDense(const RaggedTensor& ragged,
                   Tensor& dense,
                   double padding_value,
                   Tensor& validation_error_bits,
                   Stream& stream);

}  // namespace ThorImplementation
