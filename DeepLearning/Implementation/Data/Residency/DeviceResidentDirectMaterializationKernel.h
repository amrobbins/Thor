#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

/**
 * Materialize one direct field from compact indexed records into a device batch
 * tensor. Records are byte-packed, so the implementation intentionally copies
 * bytes rather than assuming field alignment.
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
