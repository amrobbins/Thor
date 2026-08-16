#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TensorOperations/Ragged/RuntimeExtent.h"

#include <utility>

namespace ThorImplementation {

class RaggedTensor {
   public:
    RaggedTensor() = default;
    RaggedTensor(Tensor values, Tensor offsets);
    RaggedTensor(Tensor values, RowPartitionRuntime rowPartition);

    bool isInitialized() const { return initialized; }

    Tensor getValues() const {
        THOR_THROW_IF_FALSE(initialized);
        return values;
    }
    Tensor getOffsets() const { return getRowPartitionRuntime().getOffsets(); }
    RowPartitionRuntime &getRowPartitionRuntime() {
        THOR_THROW_IF_FALSE(initialized);
        return rowPartition;
    }
    const RowPartitionRuntime &getRowPartitionRuntime() const {
        THOR_THROW_IF_FALSE(initialized);
        return rowPartition;
    }
    RaggedTensor withValues(Tensor newValues) const {
        THOR_THROW_IF_FALSE(initialized);
        return RaggedTensor(std::move(newValues), rowPartition);
    }
    RaggedTensorDescriptor getDescriptor() const {
        THOR_THROW_IF_FALSE(initialized);
        return descriptor;
    }

    TensorDescriptor getValuesDescriptor() const { return getDescriptor().getValuesDescriptor(); }
    TensorDescriptor getOffsetsDescriptor() const { return getRowPartitionRuntime().getDescriptor().getOffsetsDescriptor(); }

    DataType getValuesDataType() const { return getDescriptor().getValuesDataType(); }
    DataType getOffsetsDataType() const { return getRowPartitionRuntime().getOffsetsDataType(); }
    uint64_t getBatchSize() const { return getRowPartitionRuntime().getBatchSize(); }
    uint64_t getMaxTotalValues() const { return getRowPartitionRuntime().getMaxTotalValues(); }
    uint32_t getRaggedRank() const { return getDescriptor().getRaggedRank(); }
    TensorPlacement getPlacement() const {
        THOR_THROW_IF_FALSE(initialized);
        return values.getPlacement();
    }

    Tensor getActiveValueCount() const;
    [[nodiscard]] std::optional<uint64_t> getHostActiveValueCountIfAvailable() const;
    RaggedRuntimeExtent getRuntimeExtent() const;
    RaggedRuntimeExtent getRuntimeExtent(uint64_t elementsPerValue) const;

    bool operator==(const RaggedTensor &rhs) const {
        return values == rhs.values && rowPartition.describesSamePartition(rhs.rowPartition);
    }
    bool operator!=(const RaggedTensor &rhs) const { return !(*this == rhs); }

   private:
    Tensor values;
    RowPartitionRuntime rowPartition;
    RaggedTensorDescriptor descriptor;
    bool initialized = false;
};

}  // namespace ThorImplementation
