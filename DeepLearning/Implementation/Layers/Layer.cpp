#include "Layer.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <limits>
#include <stdexcept>
using namespace ThorImplementation;
using namespace std;

atomic<uint64_t> Layer::nextId(2);


uint64_t Layer::checkedLogicalByteAdd(uint64_t lhs, uint64_t rhs, const char* where) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        throw std::runtime_error(std::string(where) + " logical byte count overflow.");
    }
    return lhs + rhs;
}

uint64_t Layer::logicalBatchCapacityFromTensor(const Tensor& tensor) {
    const std::vector<uint64_t> dimensions = tensor.getDimensions();
    if (dimensions.empty()) return 0;
    return dimensions.front();
}

uint64_t Layer::logicalTensorBytesForBatch(const Tensor& tensor,
                                           uint64_t validExampleCount,
                                           uint64_t physicalBatchCapacity) {
    const uint64_t totalBytes = tensor.getArraySizeInBytes();
    if (physicalBatchCapacity == 0) return totalBytes;

    const uint64_t resolvedValidExampleCount = validExampleCount == 0 ? physicalBatchCapacity : validExampleCount;
    if (resolvedValidExampleCount > physicalBatchCapacity) {
        throw std::runtime_error("Logical byte accounting valid-example count exceeds stamped batch capacity.");
    }

    const std::vector<uint64_t> dimensions = tensor.getDimensions();
    if (dimensions.empty() || dimensions.front() != physicalBatchCapacity) {
        // Fixed-size outputs/state (for example a scalar reduction) are one
        // logical tensor for the whole submitted batch rather than one tensor
        // per example. Count them once instead of manufacturing fractional
        // per-example bytes.
        return totalBytes;
    }
    if (totalBytes % physicalBatchCapacity != 0) {
        throw std::runtime_error("Logical byte accounting tensor bytes are not divisible by stamped batch capacity.");
    }
    const uint64_t bytesPerExample = totalBytes / physicalBatchCapacity;
    if (bytesPerExample != 0 && resolvedValidExampleCount > std::numeric_limits<uint64_t>::max() / bytesPerExample) {
        throw std::runtime_error("Logical byte accounting active tensor byte count overflow.");
    }
    return bytesPerExample * resolvedValidExampleCount;
}

uint64_t Layer::logicalByteCountForward(uint64_t validExampleCount) {
    uint64_t physicalBatchCapacity = 0;
    if (featureInput.has_value()) {
        physicalBatchCapacity = logicalBatchCapacityFromTensor(featureInput.value());
    } else if (featureOutput.has_value()) {
        physicalBatchCapacity = logicalBatchCapacityFromTensor(featureOutput.value());
    }

    uint64_t bytes = 0;
    if (featureInput.has_value()) {
        bytes = checkedLogicalByteAdd(bytes,
                                      logicalTensorBytesForBatch(featureInput.value(), validExampleCount, physicalBatchCapacity),
                                      "Layer forward");
    }
    if (featureOutput.has_value()) {
        bytes = checkedLogicalByteAdd(bytes,
                                      logicalTensorBytesForBatch(featureOutput.value(), validExampleCount, physicalBatchCapacity),
                                      "Layer forward");
    }
    return bytes;
}

uint64_t Layer::logicalByteCountBackward(uint64_t validExampleCount) {
    uint64_t physicalBatchCapacity = 0;
    if (featureInput.has_value()) {
        physicalBatchCapacity = logicalBatchCapacityFromTensor(featureInput.value());
    } else if (errorInput.has_value()) {
        physicalBatchCapacity = logicalBatchCapacityFromTensor(errorInput.value());
    } else if (errorOutput.has_value()) {
        physicalBatchCapacity = logicalBatchCapacityFromTensor(errorOutput.value());
    }

    uint64_t bytes = 0;
    if (errorInput.has_value()) {
        bytes = checkedLogicalByteAdd(bytes,
                                      logicalTensorBytesForBatch(errorInput.value(), validExampleCount, physicalBatchCapacity),
                                      "Layer backward");
    }
    if (errorOutput.has_value()) {
        bytes = checkedLogicalByteAdd(bytes,
                                      logicalTensorBytesForBatch(errorOutput.value(), validExampleCount, physicalBatchCapacity),
                                      "Layer backward");
    }
    return bytes;
}


cudnnTensorDescriptor_t Layer::createCudnnTensorDescriptor(vector<unsigned long> featureInputDimensions,
                                                           DataType dataType) {
    cudnnTensorDescriptor_t descriptor;

    cudnnStatus_t cudnnStatus = cudnnCreateTensorDescriptor(&descriptor);
    THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
    // Tensors must have at least 4 dimensions and not more than CUDNN_DIM_MAX, per cudnn.
    // Unused dimensions will be set to size 1.
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnSetTensorNdDescriptor
    THOR_THROW_IF_FALSE(featureInputDimensions.size() <= CUDNN_DIM_MAX);
    vector<int> dimensionsMin4;
    vector<int> noGapsStride;
    for (unsigned int i = 0; i < featureInputDimensions.size(); ++i) {
        dimensionsMin4.push_back(featureInputDimensions[i]);
        // no overflow:
        THOR_THROW_IF_FALSE(dimensionsMin4.back() == (long)featureInputDimensions[i]);
        noGapsStride.push_back(1);
    }

    while (dimensionsMin4.size() < 4) {
        dimensionsMin4.push_back(1);
        noGapsStride.push_back(1);
    }

    for (int i = (int)dimensionsMin4.size() - 2; i >= 0; --i) {
        noGapsStride[i] = noGapsStride[i + 1] * dimensionsMin4[i + 1];
    }

    cudnnStatus = cudnnSetTensorNdDescriptor(
        descriptor, CudnnHelper::getCudnnDataType(dataType), dimensionsMin4.size(), dimensionsMin4.data(), noGapsStride.data());
    THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);

    return descriptor;
}
