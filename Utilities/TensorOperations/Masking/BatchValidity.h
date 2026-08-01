#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

/**
 * Zeroes every physical row in [validExampleCount, batchCapacity).
 *
 * Batch tensors must be dense, contiguous, and row-major with the batch axis as
 * the leading dimension. Tensor dimensions and storage remain unchanged.
 */
void zeroInvalidBatchTail(Tensor& tensor, uint32_t validExampleCount, Stream stream);

/**
 * Writes an FP32 1/0 mask whose leading dimension is the physical batch
 * capacity: valid rows are one and invalid tail rows are zero.
 *
 * The mask may have singleton trailing dimensions so that an expression can
 * broadcast one validity value across every output belonging to an example.
 */
void writeBatchValidityMask(Tensor& tensor, uint32_t validExampleCount, Stream stream);

}  // namespace ThorImplementation
