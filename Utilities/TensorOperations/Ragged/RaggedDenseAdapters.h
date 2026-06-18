#pragma once

#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstddef>
#include <cstdint>

namespace ThorImplementation {

struct RaggedFromDenseWithLengthsPlan {
    TensorPlacement placement;
    DataType valuesDataType = DataType::FP32;
    DataType offsetsDataType = DataType::UINT32;
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
                             Stream& stream);

RaggedTensor raggedFromDense(const Tensor& temp_storage,
                             size_t temp_storage_bytes,
                             const Tensor& dense,
                             const Tensor& lengths,
                             Tensor& values,
                             Tensor& offsets,
                             Stream& stream);

RaggedTensor raggedFromDense(const Tensor& dense, const Tensor& offsets, Tensor& values, Stream& stream);

void raggedToDense(const RaggedTensor& ragged, Tensor& dense, double padding_value, Stream& stream);

}  // namespace ThorImplementation
