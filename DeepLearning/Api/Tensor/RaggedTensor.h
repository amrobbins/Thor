#pragma once

#include "DeepLearning/Api/DataType.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdint>
#include <vector>

#include "Utilities/TarFile/TarReader.h"
#include "Utilities/TarFile/TarWriter.h"

namespace Thor {

using RowPartitionId = uint64_t;

// Public logical ragged tensor. The logical row partition has its own identity.
// The Tensor returned by getRowPartitionToken() is a graph-local structural token;
// it is not a promise that a device [B+1] offsets allocation exists at execution
// time. RP5/RP6 physicalization materializes HOST_EXTENT, [1] active count, or
// [B+1] offsets only for consumers that explicitly request them. For batch size B,
// values[hostOffsets[B]:maxTotalValues] is inactive capacity with
// undefined contents. Callers must not depend on that storage being zero or
// otherwise canonical, including for tensors entering or leaving a Network.
class RaggedTensor {
   public:
    RaggedTensor() = default;
    RaggedTensor(Tensor values, Tensor offsets);
    RaggedTensor(Tensor values, Tensor offsets, uint64_t maxValuesPerRow);
    RaggedTensor(DataType valuesDataType,
                 const std::vector<uint64_t> &trailingDimensions,
                 uint64_t batchSize,
                 uint64_t maxTotalValues,
                 DataType offsetsDataType = ThorImplementation::kDefaultRowPartitionOffsetDataType);
    RaggedTensor(DataType valuesDataType,
                 const std::vector<uint64_t> &trailingDimensions,
                 uint64_t batchSize,
                 uint64_t maxTotalValues,
                 uint64_t maxValuesPerRow,
                 DataType offsetsDataType = ThorImplementation::kDefaultRowPartitionOffsetDataType);

    bool isInitialized() const { return initialized; }

    uint64_t getId() const {
        THOR_THROW_IF_FALSE(initialized);
        return id;
    }
    uint64_t getOriginalId() const {
        THOR_THROW_IF_FALSE(initialized);
        return originalId;
    }

    Tensor getValues() const {
        THOR_THROW_IF_FALSE(initialized);
        return values;
    }
    // Graph-local structural row-partition token. This exists for compatibility
    // with API layers that still express partition dependencies as Tensor edges.
    // It must never be interpreted as an automatically materialized device offsets
    // buffer. Prefer this name in new graph/runtime code.
    Tensor getRowPartitionToken() const {
        THOR_THROW_IF_FALSE(initialized);
        return rowPartitionToken;
    }

    // Transitional source-compatibility alias. Direct standalone RaggedTensor values
    // may still be constructed from a real offsets Tensor, but Network execution is
    // requirement-driven and may never materialize that payload on device.
    Tensor getOffsets() const { return getRowPartitionToken(); }

    // Logical row-partition identity. RP4 decouples semantic partition
    // comparisons from any physical partition representation. The compatibility
    // row-partition token seeds identity for independently-constructed siblings in
    // one graph, while partition-preserving operations propagate the identity
    // explicitly. Consumers must compare partitions through sharesPartitionWith(),
    // never by comparing token tensors.
    RowPartitionId getRowPartitionId() const {
        THOR_THROW_IF_FALSE(initialized);
        return rowPartitionId;
    }
    bool sharesPartitionWith(const RaggedTensor &other) const {
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(other.initialized);
        return rowPartitionId == other.rowPartitionId;
    }

    RaggedTensor withValues(Tensor newValues) const;

    DataType getValuesDataType() const {
        THOR_THROW_IF_FALSE(initialized);
        return values.getDataType();
    }
    DataType getOffsetsDataType() const {
        THOR_THROW_IF_FALSE(initialized);
        return rowPartitionToken.getDataType();
    }
    std::vector<uint64_t> getValuesDimensions() const {
        THOR_THROW_IF_FALSE(initialized);
        return values.getDimensions();
    }
    std::vector<uint64_t> getOffsetsDimensions() const {
        THOR_THROW_IF_FALSE(initialized);
        return rowPartitionToken.getDimensions();
    }
    std::vector<uint64_t> getTrailingDimensions() const;
    uint64_t getBatchSize() const {
        THOR_THROW_IF_FALSE(initialized);
        return batchSize;
    }
    uint64_t getMaxTotalValues() const {
        THOR_THROW_IF_FALSE(initialized);
        return maxTotalValues;
    }
    bool hasMaxValuesPerRow() const {
        THOR_THROW_IF_FALSE(initialized);
        return maxValuesPerRow != 0;
    }
    uint64_t getMaxValuesPerRow() const {
        THOR_THROW_IF_FALSE(hasMaxValuesPerRow());
        return maxValuesPerRow;
    }
    uint32_t getRaggedRank() const {
        THOR_THROW_IF_FALSE(initialized);
        return 1;
    }

    ThorImplementation::RaggedTensorDescriptor getDescriptor() const;

    bool operator==(const RaggedTensor &other) const { return id == other.id; }
    bool operator!=(const RaggedTensor &other) const { return id != other.id; }
    bool operator<(const RaggedTensor &other) const { return id < other.id; }

    nlohmann::json architectureJson() const;
    nlohmann::json serialize(thor_file::TarWriter &archiveWriter) const;
    static RaggedTensor deserialize(const nlohmann::json &j, thor_file::TarReader *archiveReader = nullptr);

    std::string getVersion() const { return "1.1.0"; }

   private:
    static bool offsetsDataTypeValid(DataType dataType) {
        return ThorImplementation::RowPartitionDescriptor::isValidOffsetsDataType(dataType);
    }
    static std::vector<uint64_t> makeValuesDimensions(uint64_t maxTotalValues, const std::vector<uint64_t> &trailingDimensions);
    void constructFromValuesAndOffsets();

    uint64_t id = 0;
    uint64_t originalId = 0;
    static std::atomic<uint64_t> nextId;

    Tensor values;
    Tensor rowPartitionToken;
    RowPartitionId rowPartitionId = 0;
    uint64_t batchSize = 0;
    uint64_t maxTotalValues = 0;
    uint64_t maxValuesPerRow = 0;
    bool initialized = false;
};

}  // namespace Thor
