#pragma once

#include "Utilities/CudaDriver/CudaGraphDynamicGrid.h"
#include "Utilities/TensorOperations/Ragged/RuntimeExtent.h"

#include <cstdint>

namespace ThorImplementation {

[[nodiscard]] DynamicGrid1DFromScalarDescriptor raggedRuntimeExtentDynamicGrid1DDescriptor(
    const RaggedRuntimeExtent& extent,
    const DeviceUpdatableKernelNodeDeviceHandle* target_node,
    uint32_t target_block_dim_x,
    uint32_t min_grid_dim_x = 1);

}  // namespace ThorImplementation
