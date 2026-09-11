#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

/**
 * Gather rows from a device-resident split tensor into a device batch tensor.
 *
 * source has shape [num_examples, *example_shape]
 * destination has shape [batch_size, *example_shape]
 * rowIndicesDevice is a UINT64 device tensor with capacity [batch_size] whose
 * values are row positions in source's first dimension. Only the leading
 * logicalRows entries are consumed and inactive destination capacity is untouched.
 */
void launchDeviceResidentNamedGatherKernel(const ThorImplementation::Tensor &source,
                                           ThorImplementation::Tensor &destination,
                                           const ThorImplementation::Tensor &rowIndicesDevice,
                                           uint64_t logicalRows,
                                           Stream &stream);

/** Benchmark-only launch hook that forces a specific rows-per-CTA specialization. */
void launchDeviceResidentNamedGatherKernelWithRowsPerCtaForBenchmark(
    const ThorImplementation::Tensor &source,
    ThorImplementation::Tensor &destination,
    const ThorImplementation::Tensor &rowIndicesDevice,
    uint64_t logicalRows,
    uint32_t rowsPerCta,
    Stream &stream);
