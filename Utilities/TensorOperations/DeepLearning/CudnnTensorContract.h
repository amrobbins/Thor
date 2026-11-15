#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ThorImplementation {

// cuDNN graphs describe each operand by its dimensions and dense strides, with the frontend's
// default 16-byte pointer alignment. Every tensor handed to a cuDNN stage must be dense and
// 16-byte aligned. Every Thor Tensor is: it is never a view and starts on its allocation's
// (at least 256-byte) boundary. The checks keep the contract explicit at cuDNN's door.
inline constexpr uint64_t CUDNN_TENSOR_ALIGNMENT_BYTES = 16;

inline bool meetsCudnnTensorContract(const Tensor &tensor) {
    return tensor.isInitialized() && tensor.isDenseContiguous() && tensor.isAligned(CUDNN_TENSOR_ALIGNMENT_BYTES);
}

// Throws std::invalid_argument naming the operation and operand unless `tensor` meets the contract.
inline void requireCudnnContractTensor(const Tensor &tensor, std::string_view operation, std::string_view operand) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument(std::string(operation) + " tensor '" + std::string(operand) + "' is not initialized.");
    }
    if (!tensor.isDenseContiguous()) {
        throw std::invalid_argument(std::string(operation) + " tensor '" + std::string(operand) +
                                    "' must be dense: cuDNN reads and writes it with dense strides, so a strided view is not "
                                    "supported. Materialize the view first.");
    }
    if (!tensor.isAligned(CUDNN_TENSOR_ALIGNMENT_BYTES)) {
        throw std::invalid_argument(std::string(operation) + " tensor '" + std::string(operand) + "' must start on a " +
                                    std::to_string(CUDNN_TENSOR_ALIGNMENT_BYTES) + "-byte boundary; it is aligned to " +
                                    std::to_string(tensor.alignmentBytes(CUDNN_TENSOR_ALIGNMENT_BYTES)) +
                                    " bytes. Materialize the view first.");
    }
}

}  // namespace ThorImplementation
