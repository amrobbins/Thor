#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <assert.h>
#include <atomic>
#include <cmath>
#include <utility>
#include <vector>

using std::atomic;
using std::vector;

namespace Thor {

class Network;

class Tensor {
   public:
    enum class DataType { PACKED_BOOLEAN = 7, BOOLEAN, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FP16, FP32, FP64 };

    Tensor() : initialized(false) {}
    Tensor(DataType dataType, vector<uint64_t> dimensions)
        : id(nextId.fetch_add(1)), dataType(dataType), dimensions(dimensions), initialized(true) {
        for (uint32_t i = 0; i < dimensions.size(); ++i) {
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
    DataType getDataType() const {
        assert(initialized);
        return dataType;
    }
    vector<uint64_t> getDimensions() const {
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

    void reshape(vector<uint64_t> newDimensions) {
        uint64_t oldNumElements = getTotalNumElements();
        dimensions = newDimensions;
        uint64_t newNumElements = getTotalNumElements();
        assert(oldNumElements == newNumElements);
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
