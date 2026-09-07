#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

/**
 * Materialize one direct field from compact indexed records into a device batch
 * tensor. Records are byte-packed, so the implementation selects aligned vector
 * copies only when the complete record/field layout proves them safe and otherwise
 * falls back to byte transactions.
 */
void launchDeviceResidentDirectMaterializationKernel(
    const ThorImplementation::Tensor &recordStorage,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t fieldOffsetBytes,
    uint64_t fieldBytes,
    ThorImplementation::Tensor &destination,
    const ThorImplementation::Tensor &rowIndicesDevice,
    Stream &stream);
