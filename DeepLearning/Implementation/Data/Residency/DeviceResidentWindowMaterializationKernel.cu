#include "DeepLearning/Implementation/Data/Residency/DeviceResidentWindowMaterializationKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/Common/LowPrecisionFloat.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

using ThorImplementation::DataType;
using ThorImplementation::Tensor;
using ThorImplementation::TensorPlacement;

namespace {

uint64_t dataTypeBytes(DataType dataType) {
    switch (dataType) {
        case DataType::BOOLEAN:
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return 1;
        case DataType::FP16:
        case DataType::BF16:
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
        default:
            break;
    }
    throw std::runtime_error("Unsupported compact resident window data type.");
}

bool isSignedInteger(DataType dataType) {
    return dataType == DataType::INT8 || dataType == DataType::INT16 ||
           dataType == DataType::INT32 || dataType == DataType::INT64;
}

bool isUnsignedInteger(DataType dataType) {
    return dataType == DataType::UINT8 || dataType == DataType::UINT16 ||
           dataType == DataType::UINT32 || dataType == DataType::UINT64;
}

template <typename T>
uint64_t scalarBits(T value) {
    uint64_t bits = 0;
    static_assert(sizeof(T) <= sizeof(bits));
    std::memcpy(&bits, &value, sizeof(T));
    return bits;
}

uint64_t padBits(DataType dataType, double value) {
    if (value == 0.0) {
        return 0;
    }
    switch (dataType) {
        case DataType::BOOLEAN:
            return scalarBits(static_cast<uint8_t>(value != 0.0));
        case DataType::INT8:
            return scalarBits(static_cast<int8_t>(value));
        case DataType::UINT8:
            return scalarBits(static_cast<uint8_t>(value));
        case DataType::INT16:
            return scalarBits(static_cast<int16_t>(value));
        case DataType::UINT16:
            return scalarBits(static_cast<uint16_t>(value));
        case DataType::INT32:
            return scalarBits(static_cast<int32_t>(value));
        case DataType::UINT32:
            return scalarBits(static_cast<uint32_t>(value));
        case DataType::INT64:
            return scalarBits(static_cast<int64_t>(value));
        case DataType::UINT64:
            return scalarBits(static_cast<uint64_t>(value));
        case DataType::FP16:
            return scalarBits(ThorLowPrecision::doubleToFp16Bits(value));
        case DataType::BF16:
            return scalarBits(ThorLowPrecision::doubleToBf16Bits(value));
        case DataType::FP8_E4M3:
            return scalarBits(ThorLowPrecision::doubleToFp8E4M3Bits(value));
        case DataType::FP8_E5M2:
            return scalarBits(ThorLowPrecision::doubleToFp8E5M2Bits(value));
        case DataType::FP32:
            return scalarBits(static_cast<float>(value));
        case DataType::FP64:
            return scalarBits(value);
        default:
            break;
    }
    throw std::runtime_error("Unsupported compact resident non-zero window padding data type.");
}

__device__ uint64_t readLittleEndianUnsigned(const uint8_t *bytes, uint64_t numBytes) {
    uint64_t value = 0;
    for (uint64_t i = 0; i < numBytes; ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

__device__ int64_t signExtend(uint64_t value, uint64_t numBytes) {
    if (numBytes == 8) {
        return static_cast<int64_t>(value);
    }
    const uint64_t bits = numBytes * 8;
    const uint64_t signBit = uint64_t{1} << (bits - 1);
    if ((value & signBit) != 0) {
        value |= (~uint64_t{0}) << bits;
    }
    return static_cast<int64_t>(value);
}

__device__ int64_t readIndex(const uint8_t *bytes, uint64_t numBytes, bool signedType, bool *valid) {
    const uint64_t raw = readLittleEndianUnsigned(bytes, numBytes);
    if (signedType) {
        return signExtend(raw, numBytes);
    }
    if (numBytes == 8 && raw > static_cast<uint64_t>(INT64_MAX)) {
        *valid = false;
        return 0;
    }
    return static_cast<int64_t>(raw);
}

__device__ int64_t findSequence(const DeviceResidentWindowSourceSequence *sequences,
                                uint64_t count,
                                uint64_t keyBits) {
    uint64_t low = 0;
    uint64_t high = count;
    while (low < high) {
        const uint64_t middle = low + (high - low) / 2;
        const uint64_t candidate = sequences[middle].keyBits;
        if (candidate < keyBits) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (low >= count || sequences[low].keyBits != keyBits) {
        return -1;
    }
    return static_cast<int64_t>(low);
}

__device__ int64_t findAffineSegment(const DeviceResidentAffineWindowSegment *segments,
                                     uint64_t count,
                                     uint64_t row) {
    uint64_t low = 0;
    uint64_t high = count;
    while (low < high) {
        const uint64_t middle = low + (high - low) / 2;
        if (segments[middle].rowStart <= row) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (low == 0) {
        return -1;
    }
    const uint64_t candidate = low - 1;
    const DeviceResidentAffineWindowSegment &segment = segments[candidate];
    if (row - segment.rowStart >= segment.count) {
        return -1;
    }
    return static_cast<int64_t>(candidate);
}

__global__ void materializeWindowKernel(
    const uint8_t *__restrict__ records,
    const uint8_t *__restrict__ source,
    const DeviceResidentWindowSourceSequence *__restrict__ sequences,
    uint64_t sequenceCount,
    const DeviceResidentAffineWindowSegment *__restrict__ affineSegments,
    uint64_t affineSegmentCount,
    const uint64_t *__restrict__ rowIndices,
    uint8_t *__restrict__ destination,
    uint64_t batchSize,
    uint64_t numExamples,
    uint64_t recordSizeBytes,
    uint64_t referenceOffsetBytes,
    uint64_t keyBytes,
    uint64_t indexBytes,
    bool signedIndex,
    bool affine,
    uint64_t windowLength,
    uint64_t sourceStepBytes,
    uint64_t elementBytes,
    uint64_t padValueBits,
    bool materializeMask) {
    const uint64_t batchRow = static_cast<uint64_t>(blockIdx.x);
    if (batchRow >= batchSize) {
        return;
    }

    __shared__ bool validReference;
    __shared__ int64_t requestedStart;
    __shared__ int64_t sequenceStart;
    __shared__ int64_t sequenceEnd;
    __shared__ uint64_t sequenceOffsetBytes;

    if (threadIdx.x == 0) {
        validReference = true;
        const uint64_t sourceRow = rowIndices[batchRow];
        uint64_t keyBits = 0;
        int64_t start = 0;
        if (sourceRow >= numExamples) {
            validReference = false;
        } else if (!affine) {
            if (records == nullptr || recordSizeBytes == 0 ||
                referenceOffsetBytes + keyBytes + indexBytes > recordSizeBytes) {
                validReference = false;
            } else {
                const uint8_t *reference =
                    records + sourceRow * recordSizeBytes + referenceOffsetBytes;
                keyBits = readLittleEndianUnsigned(reference, keyBytes);
                start = readIndex(reference + keyBytes, indexBytes, signedIndex, &validReference);
            }
        } else {
            const int64_t segmentIndex =
                findAffineSegment(affineSegments, affineSegmentCount, sourceRow);
            if (segmentIndex < 0) {
                validReference = false;
            } else {
                const DeviceResidentAffineWindowSegment &segment =
                    affineSegments[segmentIndex];
                const uint64_t localRow = sourceRow - segment.rowStart;
                // Dataset manifest validation proves this expression is in int64 range
                // for every row covered by the segment.
                keyBits = segment.keyBits;
                start = segment.base + static_cast<int64_t>(localRow) * segment.stride +
                        segment.fieldOffset;
            }
        }

        if (validReference) {
            const int64_t sequenceIndex = findSequence(sequences, sequenceCount, keyBits);
            if (sequenceIndex < 0) {
                validReference = false;
            } else {
                const DeviceResidentWindowSourceSequence &sequence =
                    sequences[sequenceIndex];
                requestedStart = start;
                sequenceStart = sequence.startIndex;
                sequenceEnd = sequence.endIndexExclusive;
                sequenceOffsetBytes = sequence.offsetBytes;
            }
        }
    }
    __syncthreads();

    if (materializeMask) {
        uint8_t *output = destination + batchRow * windowLength;
        for (uint64_t step = static_cast<uint64_t>(threadIdx.x);
             step < windowLength;
             step += static_cast<uint64_t>(blockDim.x)) {
            bool valid = validReference;
            int64_t sourceIndex = 0;
            if (valid) {
                valid = step <= static_cast<uint64_t>(INT64_MAX) &&
                        requestedStart <= INT64_MAX - static_cast<int64_t>(step);
                if (valid) {
                    sourceIndex = requestedStart + static_cast<int64_t>(step);
                    valid = sourceIndex >= sequenceStart && sourceIndex < sequenceEnd;
                }
            }
            output[step] = valid ? uint8_t{1} : uint8_t{0};
        }
        return;
    }

    const uint64_t outputBytes = windowLength * sourceStepBytes;
    uint8_t *output = destination + batchRow * outputBytes;
    for (uint64_t byteOffset = static_cast<uint64_t>(threadIdx.x);
         byteOffset < outputBytes;
         byteOffset += static_cast<uint64_t>(blockDim.x)) {
        const uint64_t step = byteOffset / sourceStepBytes;
        const uint64_t byteWithinStep = byteOffset - step * sourceStepBytes;
        bool valid = validReference;
        int64_t sourceIndex = 0;
        if (valid) {
            valid = step <= static_cast<uint64_t>(INT64_MAX) &&
                    requestedStart <= INT64_MAX - static_cast<int64_t>(step);
            if (valid) {
                sourceIndex = requestedStart + static_cast<int64_t>(step);
                valid = sourceIndex >= sequenceStart && sourceIndex < sequenceEnd;
            }
        }
        if (valid) {
            const uint64_t sourceStep = static_cast<uint64_t>(sourceIndex - sequenceStart);
            output[byteOffset] =
                source[sequenceOffsetBytes + sourceStep * sourceStepBytes + byteWithinStep];
        } else {
            const uint64_t patternByte = byteOffset % elementBytes;
            output[byteOffset] = static_cast<uint8_t>((padValueBits >> (8 * patternByte)) & 0xffu);
        }
    }
}

void validateTensor(const Tensor &tensor, TensorPlacement placement, const char *name) {
    if (!tensor.isInitialized() || tensor.getPlacement() != placement ||
        tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::runtime_error(std::string("Invalid compact resident ") + name + " tensor.");
    }
}

}  // namespace

void launchDeviceResidentWindowMaterializationKernel(
    const Tensor &recordStorage,
    const Tensor &sourceStorage,
    const Tensor &sourceSequences,
    uint64_t sourceSequenceCount,
    const Tensor &affineSegments,
    uint64_t affineSegmentCount,
    const DeviceResidentWindowMaterializationSpec &spec,
    Tensor &destination,
    const Tensor &rowIndicesDevice,
    Stream &stream) {
    THOR_THROW_IF_FALSE(destination.isInitialized());
    THOR_THROW_IF_FALSE(rowIndicesDevice.isInitialized());
    const TensorPlacement placement = destination.getPlacement();
    validateTensor(sourceStorage, placement, "source storage");
    validateTensor(sourceSequences, placement, "source sequence metadata");
    validateTensor(rowIndicesDevice, placement, "row-index");
    THOR_THROW_IF_FALSE(rowIndicesDevice.getDataType() == DataType::UINT64);
    if (spec.referenceMode == DatasetLayout::WindowedTensorReferenceMode::INDEXED) {
        validateTensor(recordStorage, placement, "record storage");
    } else {
        validateTensor(affineSegments, placement, "affine metadata");
    }
    THOR_THROW_IF_FALSE(sourceSequenceCount > 0);
    THOR_THROW_IF_FALSE(spec.numExamples > 0);
    THOR_THROW_IF_FALSE(spec.windowLength > 0);
    THOR_THROW_IF_FALSE(spec.sourceStepBytes > 0);

    const std::vector<uint64_t> destinationDims = destination.getDimensions();
    const std::vector<uint64_t> rowDims = rowIndicesDevice.getDimensions();
    THOR_THROW_IF_FALSE(!destinationDims.empty());
    THOR_THROW_IF_FALSE(rowDims.size() == 1);
    const uint64_t batchSize = destinationDims.front();
    THOR_THROW_IF_FALSE(batchSize == rowDims.front());
    if (spec.materializeMask) {
        THOR_THROW_IF_FALSE(destination.getDataType() == DataType::UINT8);
        THOR_THROW_IF_FALSE(destination.getArraySizeInBytes() == batchSize * spec.windowLength);
    } else {
        THOR_THROW_IF_FALSE(destination.getDataType() == spec.dataType);
        THOR_THROW_IF_FALSE(
            destination.getArraySizeInBytes() == batchSize * spec.windowLength * spec.sourceStepBytes);
    }

    const uint64_t keyBytes = dataTypeBytes(spec.keyDataType);
    const uint64_t indexBytes = dataTypeBytes(spec.indexDataType);
    THOR_THROW_IF_FALSE(isSignedInteger(spec.indexDataType) || isUnsignedInteger(spec.indexDataType));
    const uint64_t elementBytes = dataTypeBytes(spec.dataType);
    const uint64_t padding = padBits(spec.dataType, spec.padValue);

    constexpr int threadsPerBlock = 256;
    THOR_THROW_IF_FALSE(batchSize <= static_cast<uint64_t>(std::numeric_limits<unsigned>::max()));
    materializeWindowKernel<<<static_cast<unsigned>(batchSize), threadsPerBlock, 0, stream.getStream()>>>(
        recordStorage.isInitialized() ? static_cast<const uint8_t *>(recordStorage.getMemPtr()) : nullptr,
        static_cast<const uint8_t *>(sourceStorage.getMemPtr()),
        reinterpret_cast<const DeviceResidentWindowSourceSequence *>(sourceSequences.getMemPtr()),
        sourceSequenceCount,
        affineSegments.isInitialized()
            ? reinterpret_cast<const DeviceResidentAffineWindowSegment *>(affineSegments.getMemPtr())
            : nullptr,
        affineSegmentCount,
        rowIndicesDevice.getMemPtr<uint64_t>(),
        static_cast<uint8_t *>(destination.getMemPtr()),
        batchSize,
        spec.numExamples,
        spec.recordSizeBytes,
        spec.referenceOffsetBytes,
        keyBytes,
        indexBytes,
        isSignedInteger(spec.indexDataType),
        spec.referenceMode == DatasetLayout::WindowedTensorReferenceMode::AFFINE,
        spec.windowLength,
        spec.sourceStepBytes,
        elementBytes,
        padding,
        spec.materializeMask);
    CUDA_CHECK(cudaGetLastError());
}
