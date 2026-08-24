#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <cstdint>
#include <string>

namespace ThorImplementation {

// Structural rank-1 row-partition bounds. maxTotalValues limits aggregate
// packed storage while maxValuesPerRow, when present, independently limits one
// logical row. maxValuesPerRow==0 means that the per-row bound is unspecified.
class RowPartitionDescriptor {
   public:
    RowPartitionDescriptor() = default;
    RowPartitionDescriptor(uint64_t batchSize,
                           uint64_t maxTotalValues,
                           DataType offsetsDataType = kDefaultRowPartitionOffsetDataType,
                           uint64_t maxValuesPerRow = 0)
        : batchSize(batchSize),
          maxTotalValues(maxTotalValues),
          maxValuesPerRow(maxValuesPerRow),
          offsetsDataType(offsetsDataType) {
        construct();
    }

    static bool isValidOffsetsDataType(DataType dataType) { return isCanonicalRowPartitionOffsetDataType(dataType); }

    uint64_t getBatchSize() const { return batchSize; }
    uint64_t getMaxTotalValues() const { return maxTotalValues; }
    bool hasMaxValuesPerRow() const { return maxValuesPerRow != 0; }
    uint64_t getMaxValuesPerRow() const {
        THOR_THROW_IF_FALSE(hasMaxValuesPerRow());
        return maxValuesPerRow;
    }
    uint64_t getMaxValuesPerRowOrZero() const { return maxValuesPerRow; }
    DataType getOffsetsDataType() const { return offsetsDataType; }

    TensorDescriptor getOffsetsDescriptor() const { return TensorDescriptor(offsetsDataType, {batchSize + 1}); }

    bool operator==(const RowPartitionDescriptor &rhs) const {
        return batchSize == rhs.batchSize && maxTotalValues == rhs.maxTotalValues &&
               maxValuesPerRow == rhs.maxValuesPerRow && offsetsDataType == rhs.offsetsDataType;
    }
    bool operator!=(const RowPartitionDescriptor &rhs) const { return !(*this == rhs); }

    std::string toString() const;

   private:
    uint64_t batchSize = 0;
    uint64_t maxTotalValues = 0;
    uint64_t maxValuesPerRow = 0;
    DataType offsetsDataType = kDefaultRowPartitionOffsetDataType;

    void construct() const {
        THOR_THROW_IF_FALSE(isValidOffsetsDataType(offsetsDataType));
        THOR_THROW_IF_FALSE(maxTotalValues > 0);
        THOR_THROW_IF_FALSE(maxValuesPerRow <= maxTotalValues);
        THOR_THROW_IF_FALSE(canonicalRowPartitionOffsetCanRepresent(offsetsDataType, maxTotalValues));
        THOR_THROW_IF_FALSE(batchSize + 1 > batchSize);
    }
};

}  // namespace ThorImplementation
