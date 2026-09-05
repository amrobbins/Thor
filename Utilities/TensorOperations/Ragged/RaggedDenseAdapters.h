#pragma once

#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"
#include "Utilities/TensorOperations/Ragged/RowPartition.h"


namespace ThorImplementation {

inline constexpr RaggedPartitionRequirement kRaggedDenseAdapterPartitionRequirement = RaggedPartitionRequirement::DEVICE_OFFSETS;

// Explicit dense/ragged compatibility adapters. The ragged partition must be
// supplied explicitly and already have authoritative host offsets; adapters do
// not derive a new partition from device values or device lengths. Each
// conversion validates the device-resident execution offsets on `stream` and
// writes a RowPartitionValidationErrorBits mask to
// `validation_error_bits`. The adapter performs no logical value copies when
// that mask is non-zero; callers may inspect it asynchronously or as part of a
// larger device-side validation policy. No host readback is performed here.
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
