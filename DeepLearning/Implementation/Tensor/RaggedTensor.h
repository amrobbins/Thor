#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TensorOperations/Ragged/RuntimeExtent.h"

#include <utility>
#include <vector>

namespace ThorImplementation {

// Logical rank-1 ragged tensor backed by fixed-capacity packed storage.
//
// For batch size B, the row partition is authoritative: only values in
// [0, offsets[B]) are part of the logical tensor. Packed storage in
// [offsets[B], maxTotalValues) has undefined contents and must not be inspected
// or relied on by active-aware consumers. A physical implementation that
// intentionally executes beyond offsets[B] owns sanitation of exactly the
// additional region it will read, immediately before that read. Producers do
// not owe canonical padding, and this contract continues across network I/O.
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
    bool hasMaxValuesPerRow() const { return getRowPartitionRuntime().hasMaxValuesPerRow(); }
    uint64_t getMaxValuesPerRow() const { return getRowPartitionRuntime().getMaxValuesPerRow(); }
    uint32_t getRaggedRank() const { return getDescriptor().getRaggedRank(); }
    TensorPlacement getPlacement() const {
        THOR_THROW_IF_FALSE(initialized);
        return values.getPlacement();
    }

    [[nodiscard]] std::optional<uint64_t> getHostActiveValueCountIfAvailable() const;
    [[nodiscard]] std::optional<uint64_t> getHostMaxActiveRowLengthIfAvailable() const {
        return getRowPartitionRuntime().getHostMaxActiveRowLengthIfAvailable();
    }
    [[nodiscard]] std::optional<std::vector<uint64_t>> getHostOffsetsIfAvailable() const {
        return getRowPartitionRuntime().getHostOffsetsIfAvailable();
    }
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
