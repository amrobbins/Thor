#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

/**
 * Materialize one compact resident ragged field for a selected batch.
 *
 * This low-level compatibility entry point reconstructs destination offsets on
 * the device from the indexed record UINT64 {start_value, value_count} metadata,
 * then gathers packed values. Normal device-resident batch sessions already
 * publish authoritative host offsets and use the values-only entry point below
 * so the prefix is not computed twice.
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

/**
 * Gather one compact resident ragged field when destinationOffsets already
 * contains the device representation of the authoritative host partition.
 *
 * Only the reference start_value is read from resident record metadata; row
 * lengths come from destinationOffsets[row+1] - destinationOffsets[row]. The
 * offsets tensor is never modified by this entry point.
 */
void launchDeviceResidentRaggedValuesMaterializationKernel(
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
