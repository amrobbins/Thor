#pragma once

#include "DeepLearning/Implementation/Tensor/PackedBoolean.h"

#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <assert.h>
#include <cstdint>
#include <type_traits>

namespace ThorImplementation {

class TensorDescriptor {
   public:
    // TODO: add FP8_E4M3, FP8_E5M2, BF16
    enum class DataType {
        FP16 = 10,
        FP32 = 11,
        FP64 = 12,
        INT8 = 13,
        INT16 = 14,
        INT32 = 15,
        INT64 = 16,
        UINT8 = 17,
        UINT16 = 18,
        UINT32 = 19,
        UINT64 = 20,
        BOOLEAN = 21,
        // FIXME: PACKED_BOOLEAN is broken for multi-dimensional case when rows are not multiple of 8 elements
        // FIXME: to fix this I need to round each dimension to (dimensionSize+7)/8 uint8_t's and I  need to save these dimensions off
        // and use them. So that say two rows do not share bits in a single uint8_t.
        PACKED_BOOLEAN = 22
    };

    TensorDescriptor() {}

    TensorDescriptor(DataType dataType, const std::vector<uint64_t> &dimensions) : dataType(dataType), dimensions(dimensions) {
        construct();
    }

    bool operator==(const TensorDescriptor &rhs) const { return dataType == rhs.dataType && dimensions == rhs.dimensions; }
    bool operator!=(const TensorDescriptor &rhs) const { return !(*this == rhs); }

    // Gives the number of bytes to store an array of numElements of this dataType, this exists because it can give the number of bytes of
    // packed boolean, whereas bytes per element doesn't work since it is not an integer number of bytes per packed boolean.
    static uint64_t getArraySizeInBytes(uint64_t numElements, DataType dataType) {
        if (dataType == DataType::PACKED_BOOLEAN) {
            return (numElements + 7) / 8;
        } else {
            return numElements * uint32_t(getElementSizeInBytes(dataType));
        }
    }
    uint64_t getArraySizeInBytes() { return getArraySizeInBytes(totalNumElements, dataType); }
    uint64_t getArraySizeInBytes(uint64_t numElements) { return getArraySizeInBytes(numElements, dataType); }

    std::string getElementName() const { return getElementTypeName(dataType); }

    static std::string getElementTypeName(DataType dataType) {
        switch (dataType) {
            case DataType::INT8:
                return "int8";
            case DataType::UINT8:
                return "uint8";
            case DataType::FP16:
                return "fp16";
            case DataType::INT16:
                return "int16";
            case DataType::UINT16:
                return "uint16";
            case DataType::FP32:
                return "fp32";
            case DataType::INT32:
                return "int32";
            case DataType::UINT32:
                return "uint32";
            case DataType::FP64:
                return "fp64";
            case DataType::INT64:
                return "int64";
            case DataType::UINT64:
                return "uint64";
            case DataType::BOOLEAN:
                return "bool";
            case DataType::PACKED_BOOLEAN:
                return "packed_boolean";
            default:
                assert(false);
        }
        return "";
    }

    std::string getValueAsString(void *basePointer, uint64_t elementIndex) { return getValueAsString(basePointer, elementIndex, dataType); }

    static std::string getValueAsString(void *basePointer, uint64_t elementIndex, DataType dataType) {
        switch (dataType) {
            case DataType::INT8:
                return std::to_string(*((int8_t *)basePointer + elementIndex));
            case DataType::INT16:
                return std::to_string(*((int16_t *)basePointer + elementIndex));
            case DataType::INT32:
                return std::to_string(*((int32_t *)basePointer + elementIndex));
            case DataType::INT64:
                return std::to_string(*((int64_t *)basePointer + elementIndex));
            case DataType::UINT8:
                return std::to_string(*((uint8_t *)basePointer + elementIndex));
            case DataType::UINT16:
                return std::to_string(*((uint16_t *)basePointer + elementIndex));
            case DataType::UINT32:
                return std::to_string(*((uint32_t *)basePointer + elementIndex));
            case DataType::UINT64:
                return std::to_string(*((uint64_t *)basePointer + elementIndex));
            case DataType::FP16:
                return std::to_string((float)*(((half *)basePointer + elementIndex)));
            case DataType::FP32:
                return std::to_string(*((float *)basePointer + elementIndex));
            case DataType::FP64:
                return std::to_string(*((double *)basePointer + elementIndex));
            case DataType::BOOLEAN:
                return *((bool *)basePointer + elementIndex) ? "1" : "0";
            case DataType::PACKED_BOOLEAN:
                return PackedBoolean::getElement(elementIndex % 8, (uint8_t *)basePointer + elementIndex / 8) ? "1" : "0";
            default:
                assert(false);
        }
        return "";
    }

    bool isIntegralType() const { return isIntegralType(dataType); }

    static bool isIntegralType(DataType dataType) {
        switch (dataType) {
            case DataType::INT8:
            case DataType::INT16:
            case DataType::INT32:
            case DataType::INT64:
            case DataType::UINT8:
            case DataType::UINT16:
            case DataType::UINT32:
            case DataType::UINT64:
            case DataType::BOOLEAN:
            case DataType::PACKED_BOOLEAN:
                return true;
            case DataType::FP16:
            case DataType::FP32:
            case DataType::FP64:
                return false;
            default:
                assert(false);
        }
        assert(false);
        return false;
    }

    bool isBooleanType() const { return isBooleanType(dataType); }

    static bool isBooleanType(DataType dataType) { return dataType == DataType::BOOLEAN || dataType == DataType::PACKED_BOOLEAN; }

    bool isSignedType() const { return isSignedType(dataType); }

    static bool isSignedType(DataType dataType) {
        switch (dataType) {
            case DataType::INT8:
            case DataType::INT16:
            case DataType::INT32:
            case DataType::INT64:
            case DataType::FP16:
            case DataType::FP32:
            case DataType::FP64:
                return true;

            case DataType::UINT8:
            case DataType::UINT16:
            case DataType::UINT32:
            case DataType::UINT64:
            case DataType::BOOLEAN:
            case DataType::PACKED_BOOLEAN:
                return false;
            default:
                assert(false);
        }
        assert(false);
        return false;
    }

    DataType getDataType() const { return dataType; }
    std::vector<uint64_t> getDimensions() const { return dimensions; }
    uint32_t getNumDimensions() const { return dimensions.size(); }
    uint64_t getTotalNumElements() const { return totalNumElements; }

    std::string toString() const {
        std::string s = std::string("DataType ") + getElementTypeName(dataType) + "\nDimensions [";
        for (uint32_t i = 0; i < dimensions.size(); ++i) {
            s += std::to_string(dimensions[i]);
            if (i < dimensions.size() - 1)
                s += " ";
        }
        s += "]\n";
        return s;
    }

    void reshape(std::vector<uint64_t> newDimensions) {
        dimensions = newDimensions;
        uint64_t newTotalNumElements = 1;
        for (uint32_t i = 0; i < dimensions.size(); ++i)
            newTotalNumElements *= newDimensions[i];
        assert(newTotalNumElements == totalNumElements);
    }

    uint64_t getFlatIndex(std::vector<uint64_t> element) {
        assert(element.size() == dimensions.size());
        uint64_t stepSize = 1;
        uint64_t index = 0;
        for (int32_t d = dimensions.size() - 1; d >= 0; --d) {
            assert(element[d] < dimensions[d]);
            index += element[d] * stepSize;
            stepSize *= dimensions[d];
        }
        assert(stepSize != 0);
        return index;
    }

    std::vector<uint64_t> getDimensionalIndex(uint64_t flatIndex) {
        assert(flatIndex < totalNumElements);

        std::vector<uint64_t> dimensionIndex;
        for (uint32_t d = 0; d < dimensions.size(); ++d) {
            dimensionIndex.push_back(flatIndex / stridePerDimension[d]);
            flatIndex -= dimensionIndex[d] * stridePerDimension[d];
        }
        return dimensionIndex;
    }

    uint64_t getDimensionStride(uint32_t axis) {
        assert(axis < dimensions.size());
        return stridePerDimension[axis];
    }

    void *getChunkAddress(std::vector<uint64_t> leadingDimensions, void *mem) {
        assert(leadingDimensions.size() <= dimensions.size());
        uint64_t chunkOffset = 0;
        for (uint32_t i = 0; i < leadingDimensions.size(); ++i) {
            assert(leadingDimensions[i] < dimensions[i]);
            chunkOffset += stridePerDimension[i] * leadingDimensions[i];
        }
        return (uint8_t *)mem + getArraySizeInBytes(chunkOffset);
    }

    void *getElementAddress(std::vector<uint64_t> leadingDimensions, void *mem) { return getChunkAddress(leadingDimensions, mem); }

    static float getElementSizeInBytes(DataType dataType) {
        switch (dataType) {
            case DataType::BOOLEAN:
            case DataType::INT8:
            case DataType::UINT8:
                return 1;
            case DataType::FP16:
            case DataType::INT16:
            case DataType::UINT16:
                return 2;
            case DataType::FP32:
            case DataType::INT32:
            case DataType::UINT32:
                return 4;
            case DataType::FP64:
            case DataType::INT64:
            case DataType::UINT64:
                return 8;
            // The PACKED_BOOLEAN case needs to be caught and handled specially
            case DataType::PACKED_BOOLEAN:
                return 0.125f;
            default:
                assert(false);
        }
        return 0;
    }

   private:
    DataType dataType;
    std::vector<uint64_t> dimensions;
    uint64_t totalNumElements;
    std::vector<uint64_t> stridePerDimension;

    void construct() {
        assert(!dimensions.empty());

        totalNumElements = 1;
        for (uint32_t i = 0; i < dimensions.size(); ++i)
            totalNumElements *= dimensions[i];
        assert(totalNumElements > 0);

        for (int32_t i = (int)dimensions.size() - 1; i >= 0; --i)
            stridePerDimension.push_back(1);
        for (int32_t i = (int)dimensions.size() - 2; i >= 0; --i)
            stridePerDimension[i] = stridePerDimension[i + 1] * dimensions[i + 1];
    }

    float getElementSizeInBytes() { return getElementSizeInBytes(dataType); }
};

NLOHMANN_JSON_SERIALIZE_ENUM(TensorDescriptor::DataType,
                             {
                                 {TensorDescriptor::DataType::PACKED_BOOLEAN, "packed_boolean"},
                                 {TensorDescriptor::DataType::BOOLEAN, "boolean"},
                                 {TensorDescriptor::DataType::INT8, "int8"},
                                 {TensorDescriptor::DataType::UINT8, "uint8"},
                                 {TensorDescriptor::DataType::INT16, "int16"},
                                 {TensorDescriptor::DataType::UINT16, "uint16"},
                                 {TensorDescriptor::DataType::INT32, "int32"},
                                 {TensorDescriptor::DataType::UINT32, "uint32"},
                                 {TensorDescriptor::DataType::INT64, "int64"},
                                 {TensorDescriptor::DataType::UINT64, "uint64"},
                                 {TensorDescriptor::DataType::FP16, "fp16"},
                                 {TensorDescriptor::DataType::FP32, "fp32"},
                                 {TensorDescriptor::DataType::FP64, "fp64"},
                             })

}  // namespace ThorImplementation
