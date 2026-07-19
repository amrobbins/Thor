#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TensorOperations/Ragged/RuntimeExtent.h"

namespace ThorImplementation {

class RaggedTensor {
   public:
    RaggedTensor() = default;
    RaggedTensor(Tensor values, Tensor offsets);

    bool isInitialized() const { return initialized; }

    Tensor getValues() const {
        THOR_THROW_IF_FALSE(initialized);
        return values;
    }
    Tensor getOffsets() const {
        THOR_THROW_IF_FALSE(initialized);
        return offsets;
    }
    RaggedTensorDescriptor getDescriptor() const {
        THOR_THROW_IF_FALSE(initialized);
        return descriptor;
    }

    TensorDescriptor getValuesDescriptor() const { return getDescriptor().getValuesDescriptor(); }
    TensorDescriptor getOffsetsDescriptor() const { return getDescriptor().getOffsetsDescriptor(); }

    DataType getValuesDataType() const { return getDescriptor().getValuesDataType(); }
    DataType getOffsetsDataType() const { return getDescriptor().getOffsetsDataType(); }
    uint64_t getBatchSize() const { return getDescriptor().getBatchSize(); }
    uint64_t getMaxTotalValues() const { return getDescriptor().getMaxTotalValues(); }
    uint32_t getRaggedRank() const { return getDescriptor().getRaggedRank(); }
    TensorPlacement getPlacement() const {
        THOR_THROW_IF_FALSE(initialized);
        return values.getPlacement();
    }

    Tensor getActiveValueCount() const;
    RaggedRuntimeExtent getRuntimeExtent() const;
    RaggedRuntimeExtent getRuntimeExtent(uint64_t elementsPerValue) const;

    bool operator==(const RaggedTensor &rhs) const { return values == rhs.values && offsets == rhs.offsets; }
    bool operator!=(const RaggedTensor &rhs) const { return !(*this == rhs); }

   private:
    Tensor values;
    Tensor offsets;
    RaggedTensorDescriptor descriptor;
    bool initialized = false;
};

}  // namespace ThorImplementation
