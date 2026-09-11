#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

/**
 * Materialize one direct field from compact indexed records into a device batch
 * tensor. Records are byte-packed, so the implementation selects aligned vector
 * copies only when the complete record/field layout proves them safe and otherwise
 * falls back to byte transactions. Only the leading logicalRows batch
 * entries are materialized; inactive destination capacity is untouched.
 */
void launchDeviceResidentDirectMaterializationKernel(
    const ThorImplementation::Tensor &recordStorage,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t fieldOffsetBytes,
    uint64_t fieldBytes,
    uint64_t logicalRows,
    ThorImplementation::Tensor &destination,
    const ThorImplementation::Tensor &rowIndicesDevice,
    Stream &stream);

/** Benchmark-only launch hook that forces a specific rows-per-CTA specialization.
 * rowsPerCta must be one of 1,2,4,8,16,32,64,128,256. Production callers
 * should use launchDeviceResidentDirectMaterializationKernel().
 */
void launchDeviceResidentDirectMaterializationKernelWithRowsPerCtaForBenchmark(
    const ThorImplementation::Tensor &recordStorage,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t fieldOffsetBytes,
    uint64_t fieldBytes,
    uint64_t logicalRows,
    ThorImplementation::Tensor &destination,
    const ThorImplementation::Tensor &rowIndicesDevice,
    uint32_t rowsPerCta,
    Stream &stream);
