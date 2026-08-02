#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

/**
 * Materialize one compact resident ragged field for a selected batch.
 *
 * The indexed record contains an unaligned UINT64 {start_value, value_count}
 * pair. Prefix construction and packed-value gathering remain entirely on the
 * device. The caller validates the selected row references/cumulative capacity
 * from the immutable host metadata captured when residency was constructed.
 * logicalRows is the number of physical rows that participate in the batch;
 * offsets for an exact partial tail are filled with the final active value
 * count through offsets[batchSize].
 */
void launchDeviceResidentRaggedMaterializationKernel(
    const ThorImplementation::Tensor &recordStorage,
    const ThorImplementation::Tensor &packedValuesStorage,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t referenceOffsetBytes,
    uint64_t storedValueCount,
    uint64_t valueBytes,
    uint64_t logicalRows,
    ThorImplementation::Tensor &destinationValues,
    ThorImplementation::Tensor &destinationOffsets,
    const ThorImplementation::Tensor &rowIndicesDevice,
    Stream &stream);
