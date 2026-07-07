#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

/**
 * Gather rows from a device-resident split tensor into a device batch tensor.
 *
 * source has shape [num_examples, *example_shape]
 * destination has shape [batch_size, *example_shape]
 * rowIndicesDevice is a UINT64 device tensor with shape [batch_size] whose
 * values are row positions in source's first dimension.
 */
void launchDeviceResidentNamedGatherKernel(const ThorImplementation::Tensor &source,
                                           ThorImplementation::Tensor &destination,
                                           const ThorImplementation::Tensor &rowIndicesDevice,
                                           Stream &stream);
