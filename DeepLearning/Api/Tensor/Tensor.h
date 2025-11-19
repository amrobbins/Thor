#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <nlohmann/json.hpp>

#include <assert.h>
#include <atomic>
#include <cmath>
#include <utility>
#include <vector>

namespace Thor {

class Network;

class Tensor {
   public:
    enum class DataType {
        PACKED_BOOLEAN = 7,
        BOOLEAN,
        INT8,
        UINT8,
        INT16,
        UINT16,
        INT32,
        UINT32,
        INT64,
        UINT64,
        FP8_E4M3,
        FP8_E5M2,
        BF16,
        FP16,
        FP32,
        FP64
    };

    Tensor() : initialized(false) {}
    Tensor(DataType dataType, std::vector<uint64_t> dimensions)
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

    bool isInitialized() const { return initialized; }

    bool operator==(const Tensor &other) const { return id == other.id; }
    bool operator!=(const Tensor &other) const { return id != other.id; }
    bool operator<(const Tensor &other) const { return id < other.id; }
    bool operator>(const Tensor &other) const { return id > other.id; }

    static bool dataTypeValid(DataType dataType) { return dataType >= DataType::PACKED_BOOLEAN && dataType <= DataType::FP64; }

    static ThorImplementation::TensorDescriptor::DataType convertToImplementationDataType(DataType apiDataType) {
        switch (apiDataType) {
            case DataType::INT8:
                return ThorImplementation::TensorDescriptor::DataType::INT8;
            case DataType::UINT8:
                return ThorImplementation::TensorDescriptor::DataType::UINT8;
            case DataType::INT16:
                return ThorImplementation::TensorDescriptor::DataType::INT16;
            case DataType::UINT16:
                return ThorImplementation::TensorDescriptor::DataType::UINT16;
            case DataType::INT32:
                return ThorImplementation::TensorDescriptor::DataType::INT32;
            case DataType::UINT32:
                return ThorImplementation::TensorDescriptor::DataType::UINT32;
            case DataType::INT64:
                return ThorImplementation::TensorDescriptor::DataType::INT64;
            case DataType::UINT64:
                return ThorImplementation::TensorDescriptor::DataType::UINT64;
            case DataType::FP16:
                return ThorImplementation::TensorDescriptor::DataType::FP16;
            case DataType::FP32:
                return ThorImplementation::TensorDescriptor::DataType::FP32;
            case DataType::FP64:
                return ThorImplementation::TensorDescriptor::DataType::FP64;
            case DataType::PACKED_BOOLEAN:
                return ThorImplementation::TensorDescriptor::DataType::PACKED_BOOLEAN;
            default:
                assert(false);
        }
    }

    uint64_t getTotalNumElements() const {
        uint64_t elements = 1;
        for (uint32_t i = 0; i < dimensions.size(); ++i)
            elements *= dimensions[i];
        return elements;
    }

    static double getBytesPerElement(DataType dataType) {
        switch (dataType) {
            case DataType::INT8:
                return 1;
            case DataType::UINT8:
                return 1;
            case DataType::INT16:
                return 2;
            case DataType::UINT16:
                return 2;
            case DataType::INT32:
                return 4;
            case DataType::UINT32:
                return 4;
            case DataType::INT64:
                return 8;
            case DataType::UINT64:
                return 8;
            case DataType::FP16:
                return 2;
            case DataType::FP32:
                return 4;
            case DataType::FP64:
                return 8;
            case DataType::PACKED_BOOLEAN:
                return 0.125;
            default:
                assert(false);
        }
    }

    double getBytesPerElement() const { return getBytesPerElement(getDataType()); }

    uint64_t getTotalSizeInBytes() const { return (uint64_t)ceil((double)getTotalNumElements() * getBytesPerElement()); }

    void reshape(std::vector<uint64_t> newDimensions) {
        uint64_t oldNumElements = getTotalNumElements();
        dimensions = newDimensions;
        uint64_t newNumElements = getTotalNumElements();
        assert(oldNumElements == newNumElements);
    }

    nlohmann::json serialize() const;
    static Tensor deserialize(const nlohmann::json &j);

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

NLOHMANN_JSON_SERIALIZE_ENUM(Tensor::DataType,
                             {
                                 {Tensor::DataType::PACKED_BOOLEAN, "packed_boolean"},
                                 {Tensor::DataType::BOOLEAN, "boolean"},
                                 {Tensor::DataType::INT8, "int8"},
                                 {Tensor::DataType::UINT8, "uint8"},
                                 {Tensor::DataType::INT16, "int16"},
                                 {Tensor::DataType::UINT16, "uint16"},
                                 {Tensor::DataType::INT32, "int32"},
                                 {Tensor::DataType::UINT32, "uint32"},
                                 {Tensor::DataType::INT64, "int64"},
                                 {Tensor::DataType::UINT64, "uint64"},
                                 {Tensor::DataType::FP16, "fp16"},
                                 {Tensor::DataType::FP32, "fp32"},
                                 {Tensor::DataType::FP64, "fp64"},
                             })

}  // namespace Thor
