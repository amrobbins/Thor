#include "Utilities/TensorOperations/Ragged/RuntimeExtent.h"

#include "Utilities/TensorOperations/Ragged/RuntimeExtentCudaGraph.h"

#include "Utilities/TensorOperations/Ragged/RowPartition.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation {
namespace {

uint64_t checkedOffsetsElements(uint64_t batch_size) {
    if (batch_size == std::numeric_limits<uint64_t>::max()) {
        throw std::invalid_argument("ragged runtime extent batch_size overflows offsets element count.");
    }
    return batch_size + 1;
}

uint64_t checkedMul(uint64_t a, uint64_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::invalid_argument(std::string(label) + " overflows uint64_t.");
    }
    return a * b;
}

void validateOffsetsForRuntimeExtent(const Tensor& offsets, uint64_t batch_size) {
    if (!offsets.isInitialized()) {
        throw std::invalid_argument("ragged runtime extent offsets tensor is not initialized.");
    }
    if (offsets.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("ragged runtime extent offsets tensor must live on GPU memory.");
    }
    if (!offsets.isDenseContiguous()) {
        throw std::invalid_argument("ragged runtime extent offsets tensor must be dense contiguous.");
    }
    if (offsets.getNumDimensions() != 1) {
        throw std::invalid_argument("ragged runtime extent offsets tensor must be rank 1.");
    }
    if (offsets.getTotalNumElements() < checkedOffsetsElements(batch_size)) {
        throw std::invalid_argument("ragged runtime extent offsets tensor does not contain batch_size + 1 elements.");
    }
    if (!isRowPartitionOffsetDTypeSupported(offsets.getDataType())) {
        throw std::invalid_argument("ragged runtime extent offsets dtype must be UINT32 or UINT64.");
    }
}

void validateRuntimeExtent(const RaggedRuntimeExtent& extent) {
    if (!extent.activeValueCount.isInitialized()) {
        throw std::invalid_argument("ragged runtime extent activeValueCount tensor is not initialized.");
    }
    if (extent.maxActiveValues == 0) {
        throw std::invalid_argument("ragged runtime extent maxActiveValues must be non-zero.");
    }
    if (extent.elementsPerValue == 0) {
        throw std::invalid_argument("ragged runtime extent elementsPerValue must be non-zero.");
    }
    if (extent.activeValueCount.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("ragged runtime extent activeValueCount tensor must live on GPU memory.");
    }
    if (!extent.activeValueCount.isDenseContiguous()) {
        throw std::invalid_argument("ragged runtime extent activeValueCount tensor must be dense contiguous.");
    }
    if (extent.activeValueCount.getDimensions() != std::vector<uint64_t>{1}) {
        throw std::invalid_argument("ragged runtime extent activeValueCount tensor must have shape [1].");
    }
    if (!isRowPartitionOffsetDTypeSupported(extent.activeValueCount.getDataType())) {
        throw std::invalid_argument("ragged runtime extent activeValueCount dtype must be UINT32 or UINT64.");
    }
}

}  // namespace

uint64_t RaggedRuntimeExtent::maxLaunchElements() const { return checkedMul(maxActiveValues, elementsPerValue, "ragged runtime extent max launch elements"); }

uint32_t RaggedRuntimeExtent::maxGridDimX(uint32_t blockDimX) const {
    if (blockDimX == 0) {
        throw std::invalid_argument("ragged runtime extent blockDimX must be non-zero.");
    }
    const uint64_t max_items = maxLaunchElements();
    const uint64_t grid64 = (max_items + static_cast<uint64_t>(blockDimX) - 1ULL) / static_cast<uint64_t>(blockDimX);
    if (grid64 == 0 || grid64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("ragged runtime extent max gridDim.x exceeds CUDA 1D grid range.");
    }
    return static_cast<uint32_t>(grid64);
}

Tensor rowPartitionActiveValueCount(const Tensor& offsets, uint64_t batch_size) {
    validateOffsetsForRuntimeExtent(offsets, batch_size);
    return offsets.aliasView({1}, {1}, batch_size);
}

RaggedRuntimeExtent raggedRuntimeExtentFromOffsets(const Tensor& offsets,
                                                  uint64_t batch_size,
                                                  uint64_t max_total_values,
                                                  uint64_t elements_per_value) {
    if (max_total_values == 0) {
        throw std::invalid_argument("ragged runtime extent max_total_values must be non-zero.");
    }
    if (elements_per_value == 0) {
        throw std::invalid_argument("ragged runtime extent elements_per_value must be non-zero.");
    }
    RaggedRuntimeExtent extent;
    extent.activeValueCount = rowPartitionActiveValueCount(offsets, batch_size);
    extent.maxActiveValues = max_total_values;
    extent.elementsPerValue = elements_per_value;
    return extent;
}

DynamicGrid1DFromScalarDescriptor raggedRuntimeExtentDynamicGrid1DDescriptor(const RaggedRuntimeExtent& extent,
                                                                             const DeviceUpdatableKernelNodeDeviceHandle* target_node,
                                                                             uint32_t target_block_dim_x,
                                                                             uint32_t min_grid_dim_x) {
    validateRuntimeExtent(extent);
    if (target_node == nullptr || !target_node->isInitialized()) {
        throw std::invalid_argument("ragged runtime extent dynamic-grid descriptor requires an initialized target-node handle.");
    }
    if (target_node->getGpuNum() != extent.activeValueCount.getPlacement().getDeviceNum()) {
        throw std::invalid_argument("ragged runtime extent dynamic-grid target-node handle must live on the active-count GPU.");
    }
    if (target_block_dim_x == 0) {
        throw std::invalid_argument("ragged runtime extent dynamic-grid target blockDim.x must be non-zero.");
    }
    if (min_grid_dim_x == 0) {
        throw std::invalid_argument("ragged runtime extent dynamic-grid min gridDim.x must be non-zero.");
    }

    DynamicGrid1DFromScalarDescriptor descriptor;
    descriptor.targetNode = target_node;
    descriptor.itemCount = extent.activeValueCount;
    descriptor.itemsPerCount = extent.elementsPerValue;
    descriptor.targetBlockDimX = target_block_dim_x;
    descriptor.minGridDimX = min_grid_dim_x;
    descriptor.maxGridDimX = extent.maxGridDimX(target_block_dim_x);
    return descriptor;
}

}  // namespace ThorImplementation
