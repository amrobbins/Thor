#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"

#include "Utilities/TensorOperations/Ragged/RuntimeExtent.h"

namespace ThorImplementation {

RaggedTensor::RaggedTensor(Tensor values, Tensor offsets) : values(values), offsets(offsets) {
    THOR_THROW_IF_FALSE(values.isInitialized());
    THOR_THROW_IF_FALSE(offsets.isInitialized());
    THOR_THROW_IF_FALSE(values.getPlacement() == offsets.getPlacement());

    TensorDescriptor valuesDescriptor = values.getDescriptor();
    TensorDescriptor offsetsDescriptor = offsets.getDescriptor();
    THOR_THROW_IF_FALSE(valuesDescriptor.getNumDimensions() >= 1);
    THOR_THROW_IF_FALSE(offsetsDescriptor.getNumDimensions() == 1);
    THOR_THROW_IF_FALSE(offsetsDescriptor.getDimensions()[0] >= 1);
    THOR_THROW_IF_FALSE(RowPartitionDescriptor::isValidOffsetsDataType(offsetsDescriptor.getDataType()));

    const uint64_t batchSize = offsetsDescriptor.getDimensions()[0] - 1;
    const uint64_t maxTotalValues = valuesDescriptor.getDimensions()[0];
    RowPartitionDescriptor rowPartition(batchSize, maxTotalValues, offsetsDescriptor.getDataType());
    descriptor = RaggedTensorDescriptor(valuesDescriptor, rowPartition);
    initialized = true;
}

Tensor RaggedTensor::getActiveValueCount() const {
    THOR_THROW_IF_FALSE(initialized);
    return rowPartitionActiveValueCount(offsets, getBatchSize());
}

RaggedRuntimeExtent RaggedTensor::getRuntimeExtent(uint64_t elementsPerValue) const {
    THOR_THROW_IF_FALSE(initialized);
    return raggedRuntimeExtentFromOffsets(offsets, getBatchSize(), getMaxTotalValues(), elementsPerValue);
}

}  // namespace ThorImplementation
