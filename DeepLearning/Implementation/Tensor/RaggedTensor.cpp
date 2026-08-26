#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace ThorImplementation {
namespace {

RowPartitionRuntime makeRowPartitionRuntime(const Tensor &values, const Tensor &offsets, uint64_t maxValuesPerRow = 0) {
    THOR_THROW_IF_FALSE(values.isInitialized());
    THOR_THROW_IF_FALSE(offsets.isInitialized());
    THOR_THROW_IF_FALSE(values.getPlacement() == offsets.getPlacement());

    const TensorDescriptor valuesDescriptor = values.getDescriptor();
    const TensorDescriptor offsetsDescriptor = offsets.getDescriptor();
    THOR_THROW_IF_FALSE(valuesDescriptor.getNumDimensions() >= 1);
    THOR_THROW_IF_FALSE(offsetsDescriptor.getNumDimensions() == 1);
    THOR_THROW_IF_FALSE(offsetsDescriptor.getDimensions()[0] >= 1);
    THOR_THROW_IF_FALSE(RowPartitionDescriptor::isValidOffsetsDataType(offsetsDescriptor.getDataType()));

    const uint64_t batchSize = offsetsDescriptor.getDimensions()[0] - 1;
    const uint64_t maxTotalValues = valuesDescriptor.getDimensions()[0];
    THOR_THROW_IF_FALSE(maxValuesPerRow == 0 || maxValuesPerRow <= maxTotalValues);
    return RowPartitionRuntime(
        offsets, RowPartitionDescriptor(batchSize, maxTotalValues, offsetsDescriptor.getDataType(), maxValuesPerRow));
}

RowPartitionRuntime makeBoundedRowPartitionRuntime(const Tensor& values, const Tensor& offsets, uint64_t maxValuesPerRow) {
    THOR_THROW_IF_FALSE(maxValuesPerRow > 0);
    return makeRowPartitionRuntime(values, offsets, maxValuesPerRow);
}

}  // namespace

RaggedTensor::RaggedTensor(Tensor values, Tensor offsets)
    : RaggedTensor(values, makeRowPartitionRuntime(values, offsets)) {}

RaggedTensor::RaggedTensor(Tensor values, Tensor offsets, uint64_t maxValuesPerRow)
    : RaggedTensor(values, makeBoundedRowPartitionRuntime(values, offsets, maxValuesPerRow)) {}

RaggedTensor::RaggedTensor(Tensor values, RowPartitionRuntime rowPartition)
    : values(std::move(values)), rowPartition(std::move(rowPartition)) {
    THOR_THROW_IF_FALSE(this->values.isInitialized());
    THOR_THROW_IF_FALSE(this->rowPartition.isInitialized());
    THOR_THROW_IF_FALSE(this->values.getPlacement() == this->rowPartition.getPlacement());

    const TensorDescriptor valuesDescriptor = this->values.getDescriptor();
    THOR_THROW_IF_FALSE(valuesDescriptor.getNumDimensions() >= 1);
    THOR_THROW_IF_FALSE(valuesDescriptor.getDimensions()[0] == this->rowPartition.getMaxTotalValues());

    descriptor = RaggedTensorDescriptor(valuesDescriptor, this->rowPartition.getDescriptor());
    initialized = true;
}

std::optional<uint64_t> RaggedTensor::getHostActiveValueCountIfAvailable() const {
    THOR_THROW_IF_FALSE(initialized);
    return rowPartition.getHostActiveValueCountIfAvailable();
}

RaggedRuntimeExtent RaggedTensor::getRuntimeExtent() const {
    THOR_THROW_IF_FALSE(initialized);
    uint64_t elementsPerValue = 1;
    for (uint64_t dim : descriptor.getTrailingDimensions()) {
        if (dim != 0 && elementsPerValue > std::numeric_limits<uint64_t>::max() / dim) {
            throw std::overflow_error("RaggedTensor trailing dimensions overflow uint64_t elementsPerValue.");
        }
        elementsPerValue *= dim;
    }
    return getRuntimeExtent(elementsPerValue);
}

RaggedRuntimeExtent RaggedTensor::getRuntimeExtent(uint64_t elementsPerValue) const {
    THOR_THROW_IF_FALSE(initialized);
    return rowPartition.getRuntimeExtent(elementsPerValue);
}

}  // namespace ThorImplementation
