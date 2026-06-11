#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/DataType.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cmath>
#include <utility>
#include <vector>

#include "Utilities/TarFile/TarReader.h"
#include "Utilities/TarFile/TarWriter.h"

namespace Thor {

class Network;

class Tensor {
   public:
    Tensor() : initialized(false) {}
    Tensor(DataType dataType, const std::vector<uint64_t> &dimensions)
        : id(nextId.fetch_add(1)), dataType(dataType), dimensions(dimensions), initialized(true) {
        originalId = id;
        // When dimension[0] == 0, it means copy the batch size when it is known.
        for (uint32_t i = 1; i < dimensions.size(); ++i) {
            THOR_THROW_IF_FALSE(dimensions[i] != 0);
        }
    }
    virtual ~Tensor() {}

    // Cloned tensors have identical characteristics but different id's
    Tensor clone() const { return Tensor(dataType, dimensions); }
    Tensor clone(DataType dataType) const { return Tensor(dataType, dimensions); }

    uint64_t getId() const {
        THOR_THROW_IF_FALSE(initialized);
        return id;
    }
    uint64_t getOriginalId() const {
        THOR_THROW_IF_FALSE(initialized);
        return originalId;
    }
    DataType getDataType() const {
        THOR_THROW_IF_FALSE(initialized);
        return dataType;
    }
    std::vector<uint64_t> getDimensions() const {
        THOR_THROW_IF_FALSE(initialized);
        return dimensions;
    }

    std::string getDescriptorString() const { return ThorImplementation::TensorDescriptor(dataType, dimensions).toString(); }

    bool isInitialized() const { return initialized; }

    bool operator==(const Tensor &other) const { return id == other.id; }
    bool operator!=(const Tensor &other) const { return id != other.id; }
    bool operator<(const Tensor &other) const { return id < other.id; }
    bool operator>(const Tensor &other) const { return id > other.id; }

    static bool dataTypeValid(DataType dataType) {
        switch (dataType) {
            case DataType::FP8_E4M3:
            case DataType::FP8_E5M2:
            case DataType::FP16:
            case DataType::BF16:
            case DataType::FP32:
            case DataType::FP64:
            case DataType::INT8:
            case DataType::UINT8:
            case DataType::INT16:
            case DataType::UINT16:
            case DataType::INT32:
            case DataType::UINT32:
            case DataType::INT64:
            case DataType::UINT64:
            case DataType::BOOLEAN:
                return true;
            default:
                return false;
        }
    }

    uint64_t getTotalNumElements() const {
        uint64_t elements = 1;
        for (uint32_t i = 0; i < dimensions.size(); ++i)
            elements *= dimensions[i];
        return elements;
    }

    static float getBytesPerElement(DataType dataType) { return ThorImplementation::TensorDescriptor::getElementSizeInBytes(dataType); }

    float getBytesPerElement() const { return getBytesPerElement(getDataType()); }

    uint64_t getTotalSizeInBytes() const {
        return ThorImplementation::TensorDescriptor::getArraySizeInBytes(getTotalNumElements(), getDataType());
    }

    void reshape(std::vector<uint64_t> newDimensions) {
        uint64_t oldNumElements = getTotalNumElements();
        uint64_t newNumElements = getTotalNumElements();
        THOR_THROW_IF_FALSE(oldNumElements == newNumElements);
        dimensions = newDimensions;
    }

    nlohmann::json architectureJson() const;
    nlohmann::json serialize(thor_file::TarWriter &archiveWriter) const;
    static Tensor deserialize(const nlohmann::json &j, thor_file::TarReader *archiveReader = nullptr);

    std::string getVersion() const { return "1.0.0"; }

   protected:
    void setDataType(DataType dataType) { this->dataType = dataType; }

   private:
    uint64_t id;
    uint64_t originalId;
    static std::atomic<uint64_t> nextId;

    DataType dataType;
    std::vector<uint64_t> dimensions;

    bool initialized = false;

    friend class Network;
};

}  // namespace Thor
