#pragma once

#include "DeepLearning/Api/DataType.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Utilities/TarFile/TarReader.h"
#include "Utilities/TarFile/TarWriter.h"

namespace Thor {

using RowPartitionId = uint64_t;

// A graph-level ragged tensor has one logical geometry: [M, ...features]
// values and a [1] structural row-partition token. Runtime B, B*M and any
// physical [B+1] offsets are owned by placement and execution, not this type.
// The partition token identifies a logical partition; it does not promise
// that an offsets allocation will be materialized on device.
// Inactive physical capacity has undefined contents.
class RaggedTensor {
   public:
    RaggedTensor() = default;
    // Valid explicit-M logical constructor; the token must have shape [1].
    RaggedTensor(Tensor values, Tensor rowPartitionToken, uint64_t maxValuesPerRow);

    // Canonical logical graph construction. The values leading dimension must
    // equal M, and the structural partition token must have shape [1].
    static RaggedTensor logical(Tensor values, Tensor rowPartitionToken, uint64_t maxValuesPerRow);

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

    ThorImplementation::RaggedTensorDescriptor getDescriptorForPlacement(uint64_t placementBatchSize) const;

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
    void constructLogical();

    uint64_t id = 0;
    uint64_t originalId = 0;
    static std::atomic<uint64_t> nextId;

    Tensor values;
    Tensor rowPartitionToken;
    RowPartitionId rowPartitionId = 0;
    uint64_t maxValuesPerRow = 0;
    bool initialized = false;
};

}  // namespace Thor
