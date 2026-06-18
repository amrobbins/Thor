#pragma once

#include "DeepLearning/Api/DataType.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensorDescriptor.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdint>
#include <vector>

#include "Utilities/TarFile/TarReader.h"
#include "Utilities/TarFile/TarWriter.h"

namespace Thor {

class RaggedTensor {
   public:
    RaggedTensor() = default;
    RaggedTensor(Tensor values, Tensor offsets);
    RaggedTensor(DataType valuesDataType,
                 const std::vector<uint64_t> &trailingDimensions,
                 uint64_t batchSize,
                 uint64_t maxTotalValues,
                 DataType offsetsDataType = DataType::UINT32);

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
    Tensor getOffsets() const {
        THOR_THROW_IF_FALSE(initialized);
        return offsets;
    }

    DataType getValuesDataType() const {
        THOR_THROW_IF_FALSE(initialized);
        return values.getDataType();
    }
    DataType getOffsetsDataType() const {
        THOR_THROW_IF_FALSE(initialized);
        return offsets.getDataType();
    }
    std::vector<uint64_t> getValuesDimensions() const {
        THOR_THROW_IF_FALSE(initialized);
        return values.getDimensions();
    }
    std::vector<uint64_t> getOffsetsDimensions() const {
        THOR_THROW_IF_FALSE(initialized);
        return offsets.getDimensions();
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

    std::string getVersion() const { return "1.0.0"; }

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
    Tensor offsets;
    uint64_t batchSize = 0;
    uint64_t maxTotalValues = 0;
    bool initialized = false;
};

}  // namespace Thor
