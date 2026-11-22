#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorSpan.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

/**
 * Materialize one direct field from compact indexed records into a device batch tensor: row r of destination is the
 * fieldBytes at fieldOffsetBytes of record rowIndicesDevice[r]. Records are byte-packed, so a field can start at any
 * address; the gather picks the widest packet the layout allows for the whole launch. Only the leading logicalRows
 * batch entries are materialized; inactive destination capacity is untouched, and so is a row whose index is not
 * below numExamples.
 */
void launchDeviceResidentDirectMaterializationKernel(
    const ThorImplementation::Tensor &recordStorage,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t fieldOffsetBytes,
    uint64_t fieldBytes,
    uint64_t logicalRows,
    ThorImplementation::Tensor &destination,
    const ThorImplementation::TensorSpan &rowIndicesDevice,
    Stream &stream);
