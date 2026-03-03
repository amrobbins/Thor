#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <nlohmann/json.hpp>

#include <assert.h>
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
    using DataType = ThorImplementation::TensorDescriptor::DataType;

    Tensor() : initialized(false) {}
    Tensor(DataType dataType, const std::vector<uint64_t> &dimensions)
        : id(nextId.fetch_add(1)), dataType(dataType), dimensions(dimensions), initialized(true) {
        originalId = id;
        // When dimension[0] == 0, it means copy the batch size when it is known.
        for (uint32_t i = 1; i < dimensions.size(); ++i) {
            assert(dimensions[i] != 0);
        }
    }
    virtual ~Tensor() {}

    // Cloned tensors have identical characteristics but different id's
    Tensor clone() const { return Tensor(dataType, dimensions); }
    Tensor clone(DataType dataType) const { return Tensor(dataType, dimensions); }

    uint64_t getId() const {
        assert(initialized);
        return id;
    }
    uint64_t getOriginalId() const {
        assert(initialized);
        return originalId;
    }
    DataType getDataType() const {
        assert(initialized);
        return dataType;
    }
    std::vector<uint64_t> getDimensions() const {
        assert(initialized);
        return dimensions;
    }

    std::string getDescriptorString() const { return ThorImplementation::TensorDescriptor(dataType, dimensions).toString(); }

    bool isInitialized() const { return initialized; }

    bool operator==(const Tensor &other) const { return id == other.id; }
    bool operator!=(const Tensor &other) const { return id != other.id; }
    bool operator<(const Tensor &other) const { return id < other.id; }
    bool operator>(const Tensor &other) const { return id > other.id; }

    static bool dataTypeValid(DataType dataType) { return dataType >= DataType::FP16 && dataType <= DataType::PACKED_BOOLEAN; }

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
        assert(oldNumElements == newNumElements);
        dimensions = newDimensions;
    }

    nlohmann::json architectureJson() const;
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

// NLOHMANN_JSON_SERIALIZE_ENUM(Tensor::DataType,
//                              {
//                                  {Tensor::DataType::PACKED_BOOLEAN, "packed_boolean"},
//                                  {Tensor::DataType::BOOLEAN, "boolean"},
//                                  {Tensor::DataType::INT8, "int8"},
//                                  {Tensor::DataType::UINT8, "uint8"},
//                                  {Tensor::DataType::INT16, "int16"},
//                                  {Tensor::DataType::UINT16, "uint16"},
//                                  {Tensor::DataType::INT32, "int32"},
//                                  {Tensor::DataType::UINT32, "uint32"},
//                                  {Tensor::DataType::INT64, "int64"},
//                                  {Tensor::DataType::UINT64, "uint64"},
//                                  {Tensor::DataType::FP16, "fp16"},
//                                  {Tensor::DataType::FP32, "fp32"},
//                                  {Tensor::DataType::FP64, "fp64"},
//                              })

}  // namespace Thor
