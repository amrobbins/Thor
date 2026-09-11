#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

/**
 * Materialize one compact resident ragged field for a selected batch.
 *
 * destinationOffsets must already contain the device representation of the
 * authoritative host partition for this batch. The materializer never
 * constructs or modifies row-partition structure; it reads only start_value
 * from resident record metadata and derives each selected row length from
 * destinationOffsets[row + 1] - destinationOffsets[row].
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
    const ThorImplementation::Tensor &destinationOffsets,
    const ThorImplementation::Tensor &rowIndicesDevice,
    Stream &stream);

/** Benchmark-only launch hook that forces a specific rows-per-CTA specialization. */
void launchDeviceResidentRaggedMaterializationKernelWithRowsPerCtaForBenchmark(
    const ThorImplementation::Tensor &recordStorage,
    const ThorImplementation::Tensor &packedValuesStorage,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t referenceOffsetBytes,
    uint64_t storedValueCount,
    uint64_t valueBytes,
    uint64_t logicalRows,
    ThorImplementation::Tensor &destinationValues,
    const ThorImplementation::Tensor &destinationOffsets,
    const ThorImplementation::Tensor &rowIndicesDevice,
    uint32_t rowsPerCta,
    Stream &stream);
