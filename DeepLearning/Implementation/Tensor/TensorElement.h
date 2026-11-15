#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace ThorImplementation {

// One element of a dense tensor, addressed by index instead of by a one-element view: for example the active value
// count of a ragged batch, which is element B of its [B + 1] row-partition offsets. Kernels read it through
// pointer<T>() (the tensor's start plus the index).
struct TensorElement {
    Tensor tensor;
    uint64_t index = 0;

    TensorElement() = default;
    // A one-element tensor converts implicitly: it is its own element 0.
    TensorElement(Tensor tensor, uint64_t index = 0) : tensor(std::move(tensor)), index(index) {}

    [[nodiscard]] bool isInitialized() const { return tensor.isInitialized(); }
    [[nodiscard]] DataType getDataType() const { return tensor.getDataType(); }
    [[nodiscard]] TensorPlacement getPlacement() const { return tensor.getPlacement(); }

    template <typename T>
    [[nodiscard]] const T* pointer() const {
        return tensor.getMemPtr<T>() + index;
    }
    [[nodiscard]] const void* address() const {
        const uint64_t elementBytes = tensor.getArraySizeInBytes() / tensor.getTotalNumElements();
        return static_cast<const uint8_t*>(tensor.getMemPtr<void>()) + index * elementBytes;
    }

    // Throws std::invalid_argument (prefixed with `what`) unless the tensor is initialized and dense and the index is
    // inside it.
    void validate(const std::string& what) const {
        if (!tensor.isInitialized()) {
            throw std::invalid_argument(what + " tensor is not initialized.");
        }
        if (!tensor.isDenseContiguous()) {
            throw std::invalid_argument(what + " tensor must be dense contiguous.");
        }
        if (index >= tensor.getTotalNumElements()) {
            throw std::invalid_argument(what + " element " + std::to_string(index) + " is outside its " +
                                        std::to_string(tensor.getTotalNumElements()) + "-element tensor.");
        }
    }
};

}  // namespace ThorImplementation
