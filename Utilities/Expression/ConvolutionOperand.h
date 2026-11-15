#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorSpan.h"
#include "Utilities/TensorOperations/DeepLearning/CudnnTensorContract.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ThorImplementation {

// One cuDNN convolution operand (x, w, y, dy, dx or dW): dense storage and the dimensions cuDNN sees, so that a padded
// ragged value can be convolved at any width of its family without a view. A whole Tensor converts with its own
// dimensions; a padded ragged value at width W is the first B*C*W elements of its reserved storage, seen as
// [B, C, 1, W] (PaddedRaggedSequence::paddedValuesForWidth / paddedDimensionsForWidth).
struct ConvolutionOperand {
    TensorSpan storage;
    std::vector<uint64_t> dims;

    ConvolutionOperand(const Tensor& tensor) : storage(tensor), dims(tensor.getDimensions()) {}
    // Throws std::invalid_argument unless the span holds exactly the dimensions' element count.
    ConvolutionOperand(TensorSpan storage, std::vector<uint64_t> dims) : storage(std::move(storage)), dims(std::move(dims)) {
        uint64_t elements = 1;
        for (uint64_t dim : this->dims) {
            elements *= dim;
        }
        if (this->dims.empty() || elements != this->storage.count) {
            throw std::invalid_argument("ConvolutionOperand span of " + std::to_string(this->storage.count) +
                                        " elements does not match its " + std::to_string(elements) + "-element dimensions.");
        }
    }

    [[nodiscard]] DataType getDataType() const { return storage.getDataType(); }
    [[nodiscard]] TensorPlacement getPlacement() const { return storage.getPlacement(); }
    [[nodiscard]] TensorDescriptor descriptor() const { return TensorDescriptor(getDataType(), dims); }
    [[nodiscard]] uint64_t bytes() const { return storage.bytes(); }
    [[nodiscard]] void* address() const { return storage.address(); }
};

// Throws std::invalid_argument naming the operation and operand unless the operand meets the cuDNN contract: dense
// storage, starting on a 16-byte boundary.
inline void requireCudnnContractOperand(const ConvolutionOperand& operand, std::string_view operation, std::string_view name) {
    requireCudnnContractTensor(operand.storage.tensor, operation, name);
    if (reinterpret_cast<uintptr_t>(operand.address()) % CUDNN_TENSOR_ALIGNMENT_BYTES != 0) {
        throw std::invalid_argument(std::string(operation) + " operand '" + std::string(name) + "' must start on a " +
                                    std::to_string(CUDNN_TENSOR_ALIGNMENT_BYTES) + "-byte boundary.");
    }
}

}  // namespace ThorImplementation
