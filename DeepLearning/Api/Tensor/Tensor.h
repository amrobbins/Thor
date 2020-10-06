#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <assert.h>
#include <atomic>
#include <utility>
#include <vector>

using std::atomic;
using std::vector;

namespace Thor {

class Network;

class Tensor {
   public:
    enum class DataType { FP32 = 2, FP16, UINT8 };

    Tensor() : initialized(false) {}
    Tensor(DataType dataType, vector<uint64_t> dimensions)
        : id(nextId.fetch_add(1)), dataType(dataType), dimensions(dimensions), initialized(true) {
        for (uint32_t i = 0; i < dimensions.size(); ++i) {
            assert(dimensions[i] != 0);
        }
    }
    virtual ~Tensor() {}

    // Cloned tensors have identical characteristics but different id's
    Tensor clone() { return Tensor(dataType, dimensions); }
    Tensor clone(DataType dataType) { return Tensor(dataType, dimensions); }

    uint64_t getId() const {
        assert(initialized);
        return id;
    }
    DataType getDataType() const {
        assert(initialized);
        return dataType;
    }
    vector<uint64_t> getDimensions() const {
        assert(initialized);
        return dimensions;
    }

    bool isInitialized() { return initialized; }

    bool operator==(const Tensor &other) const { return id == other.id; }
    bool operator!=(const Tensor &other) const { return id != other.id; }
    bool operator<(const Tensor &other) const { return id < other.id; }
    bool operator>(const Tensor &other) const { return id > other.id; }

    static bool dataTypeValid(DataType dataType) { return dataType >= DataType::FP32 && dataType <= DataType::UINT8; }

    static ThorImplementation::TensorDescriptor::DataType convertToImplementationDataType(DataType apiDataType) {
        if (apiDataType == DataType::FP32)
            return ThorImplementation::TensorDescriptor::DataType::FP32;
        else if (apiDataType == DataType::FP16)
            return ThorImplementation::TensorDescriptor::DataType::FP16;
        else if (apiDataType == DataType::UINT8)
            return ThorImplementation::TensorDescriptor::DataType::UINT8;
        else
            assert(false);
    }

    uint64_t getTotalNumElements() const {
        uint64_t elements = 1;
        for (uint32_t i = 0; i < dimensions.size(); ++i)
            elements *= dimensions[i];
        return elements;
    }

    uint64_t getTotalSizeInBytes() const {
        uint32_t bytesPerElement;
        if (dataType == DataType::FP32)
            bytesPerElement = 4;
        else if (dataType == DataType::FP16)
            bytesPerElement = 2;
        else if (dataType == DataType::UINT8)
            bytesPerElement = 1;
        else
            assert(false);
        return bytesPerElement * getTotalNumElements();
    }

   protected:
    void setDataType(DataType dataType) { this->dataType = dataType; }

   private:
    uint64_t id;
    static atomic<uint64_t> nextId;

    DataType dataType;
    vector<uint64_t> dimensions;

    bool initialized;

    friend class Network;
};

}  // namespace Thor
