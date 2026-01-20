#pragma once

#include "DeepLearning/Implementation/Tensor/PackedBoolean.h"

#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"

#include <string>
#include <vector>

#include <assert.h>
#include <type_traits>

namespace ThorImplementation {

class TensorDescriptor {
   public:
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
    static unsigned long getArraySizeInBytes(long numElements, DataType dataType) {
        if (dataType == DataType::PACKED_BOOLEAN) {
            return (numElements + 7) / 8;
        } else {
            return numElements * getElementSizeInBytes(dataType);
        }
    }
    long unsigned getArraySizeInBytes() { return getArraySizeInBytes(totalNumElements, dataType); }
    long unsigned getArraySizeInBytes(long numElements) { return getArraySizeInBytes(numElements, dataType); }

    std::string getElementName() { return getElementTypeName(dataType); }

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

    std::string getValueAsString(void *basePointer, unsigned long elementIndex) {
        return getValueAsString(basePointer, elementIndex, dataType);
    }

    static std::string getValueAsString(void *basePointer, unsigned long elementIndex, DataType dataType) {
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

    bool isIntegralType() { return isIntegralType(dataType); }

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

    bool isBooleanType() { return isBooleanType(dataType); }

    static bool isBooleanType(DataType dataType) { return dataType == DataType::BOOLEAN || dataType == DataType::PACKED_BOOLEAN; }

    bool isSignedType() { return isSignedType(dataType); }

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
    std::vector<unsigned long> getDimensions() const { return dimensions; }
    unsigned int getNumDimensions() const { return dimensions.size(); }
    unsigned long getTotalNumElements() const { return totalNumElements; }

    std::string toString() const {
        std::string s = std::string("DataType ") + getElementTypeName(dataType) + "\nDimensions [";
        for (unsigned int i = 0; i < dimensions.size(); ++i) {
            s += std::to_string(dimensions[i]);
            if (i < dimensions.size() - 1)
                s += " ";
        }
        s += "]\n";
        return s;
    }

    void reshape(std::vector<unsigned long> newDimensions) {
        dimensions = newDimensions;
        unsigned long newTotalNumElements = 1;
        for (unsigned int i = 0; i < dimensions.size(); ++i)
            newTotalNumElements *= newDimensions[i];
        assert(newTotalNumElements == totalNumElements);
    }

    unsigned long getFlatIndex(std::vector<unsigned long> element) {
        assert(element.size() == dimensions.size());
        unsigned long stepSize = 1;
        unsigned long index = 0;
        for (int d = dimensions.size() - 1; d >= 0; --d) {
            assert(element[d] < dimensions[d]);
            index += element[d] * stepSize;
            stepSize *= dimensions[d];
        }
        assert(stepSize != 0);
        return index;
    }

    std::vector<unsigned long> getDimensionalIndex(unsigned long flatIndex) {
        assert(flatIndex < totalNumElements);

        std::vector<unsigned long> dimensionIndex;
        for (unsigned int d = 0; d < dimensions.size(); ++d) {
            dimensionIndex.push_back(flatIndex / stridePerDimension[d]);
            flatIndex -= dimensionIndex[d] * stridePerDimension[d];
        }
        return dimensionIndex;
    }

    unsigned long getDimensionStride(unsigned int axis) {
        assert(axis < dimensions.size());
        return stridePerDimension[axis];
    }

    void *getChunkAddress(std::vector<unsigned long> leadingDimensions, void *mem) {
        assert(leadingDimensions.size() <= dimensions.size());
        unsigned long chunkOffset = 0;
        for (unsigned int i = 0; i < leadingDimensions.size(); ++i) {
            assert(leadingDimensions[i] < dimensions[i]);
            chunkOffset += stridePerDimension[i] * leadingDimensions[i];
        }
        return (uint8_t *)mem + getArraySizeInBytes(chunkOffset);
    }

    void *getElementAddress(std::vector<unsigned long> leadingDimensions, void *mem) { return getChunkAddress(leadingDimensions, mem); }

    static int getElementSizeInBytes(DataType dataType) {
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
                assert(false);
            default:
                assert(false);
        }
        return 0;
    }

   private:
    DataType dataType;
    std::vector<unsigned long> dimensions;
    unsigned long totalNumElements;
    std::vector<unsigned long> stridePerDimension;

    void construct() {
        assert(!dimensions.empty());

        totalNumElements = 1;
        for (unsigned int i = 0; i < dimensions.size(); ++i)
            totalNumElements *= dimensions[i];
        assert(totalNumElements > 0);

        for (int i = (int)dimensions.size() - 1; i >= 0; --i)
            stridePerDimension.push_back(1);
        for (int i = (int)dimensions.size() - 2; i >= 0; --i)
            stridePerDimension[i] = stridePerDimension[i + 1] * dimensions[i + 1];
    }

    int getElementSizeInBytes() { return getElementSizeInBytes(dataType); }
};

}  // namespace ThorImplementation
