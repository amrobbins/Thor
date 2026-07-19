#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ThorImplementation {

class RowPartitionDescriptor {
   public:
    RowPartitionDescriptor() = default;
    RowPartitionDescriptor(uint64_t batchSize, uint64_t maxTotalValues, DataType offsetsDataType = kDefaultRowPartitionOffsetDataType)
        : batchSize(batchSize), maxTotalValues(maxTotalValues), offsetsDataType(offsetsDataType) {
        construct();
    }

    static bool isValidOffsetsDataType(DataType dataType) { return isCanonicalRowPartitionOffsetDataType(dataType); }

    uint64_t getBatchSize() const { return batchSize; }
    uint64_t getMaxTotalValues() const { return maxTotalValues; }
    DataType getOffsetsDataType() const { return offsetsDataType; }

    TensorDescriptor getOffsetsDescriptor() const { return TensorDescriptor(offsetsDataType, {batchSize + 1}); }

    bool operator==(const RowPartitionDescriptor &rhs) const {
        return batchSize == rhs.batchSize && maxTotalValues == rhs.maxTotalValues && offsetsDataType == rhs.offsetsDataType;
    }
    bool operator!=(const RowPartitionDescriptor &rhs) const { return !(*this == rhs); }

    std::string toString() const;

   private:
    uint64_t batchSize = 0;
    uint64_t maxTotalValues = 0;
    DataType offsetsDataType = kDefaultRowPartitionOffsetDataType;

    void construct() const {
        THOR_THROW_IF_FALSE(isValidOffsetsDataType(offsetsDataType));
        THOR_THROW_IF_FALSE(maxTotalValues > 0);
        THOR_THROW_IF_FALSE(canonicalRowPartitionOffsetCanRepresent(offsetsDataType, maxTotalValues));
        THOR_THROW_IF_FALSE(batchSize + 1 > batchSize);
    }
};

class RaggedTensorDescriptor {
   public:
    RaggedTensorDescriptor() = default;
    RaggedTensorDescriptor(TensorDescriptor valuesDescriptor, RowPartitionDescriptor rowPartition, uint32_t raggedRank = 1)
        : valuesDescriptor(valuesDescriptor), rowPartition(rowPartition), raggedRank(raggedRank) {
        construct();
    }
    RaggedTensorDescriptor(DataType valuesDataType,
                           const std::vector<uint64_t> &trailingDimensions,
                           uint64_t batchSize,
                           uint64_t maxTotalValues,
                           DataType offsetsDataType = kDefaultRowPartitionOffsetDataType,
                           uint32_t raggedRank = 1)
        : valuesDescriptor(valuesDataType, makeValuesDimensions(maxTotalValues, trailingDimensions)),
          rowPartition(batchSize, maxTotalValues, offsetsDataType),
          raggedRank(raggedRank) {
        construct();
    }

    TensorDescriptor getValuesDescriptor() const { return valuesDescriptor; }
    TensorDescriptor getOffsetsDescriptor() const { return rowPartition.getOffsetsDescriptor(); }
    RowPartitionDescriptor getRowPartition() const { return rowPartition; }

    DataType getValuesDataType() const { return valuesDescriptor.getDataType(); }
    DataType getOffsetsDataType() const { return rowPartition.getOffsetsDataType(); }
    uint64_t getBatchSize() const { return rowPartition.getBatchSize(); }
    uint64_t getMaxTotalValues() const { return rowPartition.getMaxTotalValues(); }
    uint32_t getRaggedRank() const { return raggedRank; }

    std::vector<uint64_t> getValuesDimensions() const { return valuesDescriptor.getDimensions(); }
    std::vector<uint64_t> getTrailingDimensions() const;

    bool operator==(const RaggedTensorDescriptor &rhs) const {
        return valuesDescriptor == rhs.valuesDescriptor && rowPartition == rhs.rowPartition && raggedRank == rhs.raggedRank;
    }
    bool operator!=(const RaggedTensorDescriptor &rhs) const { return !(*this == rhs); }

    std::string toString() const;

   private:
    TensorDescriptor valuesDescriptor;
    RowPartitionDescriptor rowPartition;
    uint32_t raggedRank = 1;

    static std::vector<uint64_t> makeValuesDimensions(uint64_t maxTotalValues, const std::vector<uint64_t> &trailingDimensions) {
        THOR_THROW_IF_FALSE(maxTotalValues > 0);
        std::vector<uint64_t> dimensions;
        dimensions.reserve(trailingDimensions.size() + 1);
        dimensions.push_back(maxTotalValues);
        for (uint64_t dim : trailingDimensions) {
            THOR_THROW_IF_FALSE(dim > 0);
            dimensions.push_back(dim);
        }
        return dimensions;
    }

    void construct() const {
        THOR_THROW_IF_FALSE(raggedRank == 1);
        THOR_THROW_IF_FALSE(valuesDescriptor.getNumDimensions() >= 1);
        THOR_THROW_IF_FALSE(valuesDescriptor.getDimensions()[0] == rowPartition.getMaxTotalValues());
        THOR_THROW_IF_FALSE(rowPartition.getMaxTotalValues() > 0);
        THOR_THROW_IF_FALSE(RowPartitionDescriptor::isValidOffsetsDataType(rowPartition.getOffsetsDataType()));
    }
};

}  // namespace ThorImplementation
