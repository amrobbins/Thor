#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ThorImplementation {

// `count` consecutive elements of a tensor, starting `offset` elements past its first element (getMemPtr), addressed by
// offset instead of by a view. A Tensor is always dense, so these are its elements in order. Examples:
// the row indices and each window plan packed into one selection-metadata buffer, so that one upload moves them all;
// the part of a grouped-MATMUL operand that one strided-batched group reads or writes. Not a Tensor: only Thor's own
// host code and kernels take one, and they address it through data<T>() / address().
struct TensorSpan {
    Tensor tensor;
    uint64_t offset = 0;
    uint64_t count = 0;

    TensorSpan() = default;
    // A whole tensor converts implicitly: all of its elements (an uninitialized tensor gives an empty span).
    TensorSpan(Tensor tensor) : tensor(std::move(tensor)), count(storageElements(this->tensor)) {}
    TensorSpan(Tensor tensor, uint64_t offset, uint64_t count) : tensor(std::move(tensor)), offset(offset), count(count) {
        validate("TensorSpan");
    }

    // A tensor's element count, 0 when it is uninitialized.
    [[nodiscard]] static uint64_t storageElements(const Tensor& tensor) {
        return tensor.isInitialized() ? tensor.getTotalNumElements() : 0;
    }

    [[nodiscard]] bool isInitialized() const { return tensor.isInitialized(); }
    [[nodiscard]] DataType getDataType() const { return tensor.getDataType(); }
    [[nodiscard]] TensorPlacement getPlacement() const { return tensor.getPlacement(); }
    [[nodiscard]] uint64_t elementBytes() const {
        return tensor.isInitialized() ? tensor.getArraySizeInBytes() / tensor.getTotalNumElements() : 0;
    }
    [[nodiscard]] uint64_t bytes() const { return count * elementBytes(); }

    template <typename T>
    [[nodiscard]] T* data() const {
        return const_cast<T*>(tensor.getMemPtr<T>()) + offset;
    }
    [[nodiscard]] void* address() const {
        void* start = const_cast<void*>(tensor.getMemPtr<void>());
        return offset == 0 ? start : static_cast<uint8_t*>(start) + offset * elementBytes();
    }

    // Throws std::invalid_argument (prefixed with `what`) unless the tensor is initialized and the span lies inside its
    // storage.
    void validate(const std::string& what) const {
        if (!tensor.isInitialized()) {
            throw std::invalid_argument(what + " tensor is not initialized.");
        }
        const uint64_t storage = storageElements(tensor);
        if (offset > storage || count > storage - offset) {
            throw std::invalid_argument(what + " span of " + std::to_string(count) + " elements at offset " +
                                        std::to_string(offset) + " is outside its tensor's " + std::to_string(storage) +
                                        "-element storage.");
        }
    }
};

}  // namespace ThorImplementation
