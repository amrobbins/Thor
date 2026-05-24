#pragma once

#include <cstdint>
#include <limits>

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/CudaDriver/CudaGraph.h"

namespace ThorImplementation {

struct DynamicGrid1DFromScalarDescriptor {
    const DeviceUpdatableKernelNodeDeviceHandle* targetNode = nullptr;
    Tensor itemCount;
    uint64_t itemsPerCount = 1;
    uint32_t targetBlockDimX = 256;
    uint32_t minGridDimX = 1;
    uint32_t maxGridDimX = std::numeric_limits<uint32_t>::max();
};

struct DynamicGrid2DFromScalarDescriptor {
    const DeviceUpdatableKernelNodeDeviceHandle* targetNode = nullptr;
    Tensor rowCount;
    uint32_t gridDimY = 1;
    uint64_t gridDimXPerRow = 1;
    uint32_t minGridDimX = 1;
    uint32_t maxGridDimX = std::numeric_limits<uint32_t>::max();
    uint32_t maxGridDimY = std::numeric_limits<uint32_t>::max();
};

void launchUpdateDeviceGrid1DFromScalar(const DynamicGrid1DFromScalarDescriptor& descriptor, Stream stream);
void launchUpdateDeviceGrid2DFromScalar(const DynamicGrid2DFromScalarDescriptor& descriptor, Stream stream);

}  // namespace ThorImplementation
