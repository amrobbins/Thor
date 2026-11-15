#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorSpan.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ThorImplementation {

// A strided region of a tensor's storage: dimensions with element strides, starting at its storage's first element. It
// is what an expression strided view (a GLU half, a slice, an einsum permutation) is at run time, kept out of the
// Tensor type: only the consumers that read strides take one (fused kernels, matmul operands, reductions); every other
// consumer gets a dense tensor. Its storage is every element the layout reaches, gaps included.
//
// A span reads through the same accessors as a Tensor (getDimensions(), getStridesElements(), getMemPtr<T>(), ...), so
// code that reads a tensor reads a span unchanged, and a whole tensor converts implicitly. A span is a non-owning
// reference to its storage, so a const span may still be written through (an output span).
//
// bytesAs() reads part of a dense span's bytes as a dense span of another dtype (the slots of a reduction autotuner's
// byte arenas); getStorage() then counts that span's storage in the source tensor's elements, not its own.
// withLayout() reads a dense span's elements under another layout (a strided span inside a span). asDense() reads a
// permuted dense span's elements as dense, in storage order (1365: a value reduction of the same bytes).
class StridedSpan {
   public:
    // An empty span (no tensor): a placeholder, as an uninitialized Tensor is.
    StridedSpan() = default;
    // A whole tensor, with its own dimensions and strides (an uninitialized tensor gives an empty span).
    StridedSpan(const Tensor& tensor) : storage(tensor) {
        if (tensor.isInitialized()) {
            dims = tensor.getDimensions();
            strides = tensor.getStridesElements();
            dataType = tensor.getDataType();
        }
    }
    // `dims` / `strides` from element `offset` of `source`. Throws std::invalid_argument unless the layout stays inside
    // the source's storage.
    StridedSpan(const Tensor& source, uint64_t offset, std::vector<uint64_t> dims, std::vector<uint64_t> strides)
        : dims(std::move(dims)), strides(std::move(strides)), dataType(source.getDataType()) {
        if (this->dims.empty() || this->dims.size() != this->strides.size()) {
            throw std::invalid_argument("StridedSpan requires dimensions and strides of the same non-zero rank.");
        }
        storage = TensorSpan(source, offset, reachedElements(this->dims, this->strides));
    }

    // The bytes [byteOffset, byteOffset + elements * sizeof(dtype)) of this dense span, read as a dense `dtype` span of
    // `newDims`. Throws std::invalid_argument unless this span is dense, the range lies inside its bytes and the start
    // is aligned to the element size.
    [[nodiscard]] StridedSpan bytesAs(uint64_t byteOffset, DataType dtype, std::vector<uint64_t> newDims) const {
        if (!isInitialized() || !isDenseContiguous()) {
            throw std::invalid_argument("StridedSpan::bytesAs requires an initialized dense span.");
        }
        const uint64_t sourceElementBytes = storage.elementBytes();
        const uint64_t elementBytes = bytesOf(dtype, 1);
        StridedSpan out;
        out.strides = denseStrides(newDims);
        const uint64_t bytes = bytesOf(dtype, reachedElements(newDims, out.strides));
        const uint64_t available = getArraySizeInBytes();
        const uint64_t startByte = storage.offset * sourceElementBytes + byteOffset;
        if (byteOffset > available || bytes > available - byteOffset || startByte % sourceElementBytes != 0 ||
            (reinterpret_cast<uintptr_t>(getMemPtr<uint8_t>()) + byteOffset) % elementBytes != 0) {
            throw std::invalid_argument("StridedSpan::bytesAs range is outside the span or misaligned for its dtype.");
        }
        out.storage = TensorSpan(storage.tensor, startByte / sourceElementBytes, (bytes + sourceElementBytes - 1) / sourceElementBytes);
        out.dims = std::move(newDims);
        out.dataType = dtype;
        return out;
    }

    // This dense span's elements read under another layout (`newDims` with element `newStrides`) from its first element:
    // a strided span inside a span, e.g. of bytes read as a dtype with bytesAs(). Throws std::invalid_argument unless
    // this span is dense and the layout reaches no element past its end.
    [[nodiscard]] StridedSpan withLayout(std::vector<uint64_t> newDims, std::vector<uint64_t> newStrides) const {
        if (!isInitialized() || !isDenseContiguous()) {
            throw std::invalid_argument("StridedSpan::withLayout requires an initialized dense span.");
        }
        if (newDims.empty() || newDims.size() != newStrides.size()) {
            throw std::invalid_argument("StridedSpan::withLayout requires dimensions and strides of the same non-zero rank.");
        }
        if (reachedElements(newDims, newStrides) > getTotalNumElements()) {
            throw std::invalid_argument("StridedSpan::withLayout layout reaches past the span's last element.");
        }
        StridedSpan out = *this;
        out.dims = std::move(newDims);
        out.strides = std::move(newStrides);
        return out;
    }

    // 1391 (phase 7 A): the elements of this span's dtype from its first element to the end of its source tensor: what
    // a layout from the same first element may reach (withSourceLayout).
    [[nodiscard]] uint64_t sourceElementsFromFirst() const {
        if (!isInitialized()) return 0;
        const uint64_t sourceBytes = storage.tensor.getArraySizeInBytes();
        const uint64_t startByte = storage.offset * storage.elementBytes();
        const uint64_t elementBytes = bytesOf(dataType, 1);
        return startByte >= sourceBytes || elementBytes == 0 ? 0 : (sourceBytes - startByte) / elementBytes;
    }

    // 1391 (phase 7 A): this span's source read under another layout from the same first element. The layout may reach
    // elements past this span's own (the gap after its last row) but none past its source tensor's last element; a
    // consumer that reads them must not let them change its result. Throws std::invalid_argument when the layout would
    // leave the source.
    [[nodiscard]] StridedSpan withSourceLayout(std::vector<uint64_t> newDims, std::vector<uint64_t> newStrides) const {
        if (!isInitialized()) {
            throw std::invalid_argument("StridedSpan::withSourceLayout requires an initialized span.");
        }
        if (newDims.empty() || newDims.size() != newStrides.size()) {
            throw std::invalid_argument(
                "StridedSpan::withSourceLayout requires dimensions and strides of the same non-zero rank.");
        }
        const uint64_t reached = reachedElements(newDims, newStrides);
        if (reached > sourceElementsFromFirst()) {
            throw std::invalid_argument("StridedSpan::withSourceLayout layout reaches past its source tensor.");
        }
        const uint64_t sourceElementBytes = storage.elementBytes();
        StridedSpan out = *this;
        out.storage = TensorSpan(storage.tensor, storage.offset,
                                 (bytesOf(dataType, reached) + sourceElementBytes - 1) / sourceElementBytes);
        out.dims = std::move(newDims);
        out.strides = std::move(newStrides);
        return out;
    }

    // 1365: this span's elements read as a dense span of `newDims` in storage order: the same storage from the same
    // first element, so element order changes. For an order-insensitive consumer such as a value reduction, a
    // permuted dense tensor is the dense tensor of its bytes. Throws std::invalid_argument unless the layout is a
    // compact permutation (isCompactPermutation) and `newDims` has its element count.
    [[nodiscard]] StridedSpan asDense(std::vector<uint64_t> newDims) const {
        if (!isInitialized() || !isCompactPermutation()) {
            throw std::invalid_argument("StridedSpan::asDense requires an initialized compact-permutation span.");
        }
        StridedSpan out = *this;
        out.strides = denseStrides(newDims);
        if (newDims.empty() || reachedElements(newDims, out.strides) != getTotalNumElements()) {
            throw std::invalid_argument("StridedSpan::asDense requires dimensions with the span's element count.");
        }
        out.dims = std::move(newDims);
        return out;
    }

    // True when the layout is a permuted dense tensor: its non-singleton axes, ordered by stride, have the dense
    // strides of their extents, so it reaches each of getTotalNumElements() elements exactly once. Extent-one axes may
    // carry any stride.
    [[nodiscard]] bool isCompactPermutation() const {
        std::vector<size_t> order;
        for (size_t axis = 0; axis < dims.size(); ++axis) {
            if (dims[axis] > 1) order.push_back(axis);
        }
        std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) { return strides[lhs] > strides[rhs]; });
        uint64_t expected = 1;
        for (size_t position = order.size(); position-- > 0;) {
            const size_t axis = order[position];
            if (strides[axis] != expected || expected > std::numeric_limits<uint64_t>::max() / dims[axis]) return false;
            expected *= dims[axis];
        }
        return true;
    }

    // The elements a layout reaches from its first element, gaps included.
    [[nodiscard]] static uint64_t reachedElements(const std::vector<uint64_t>& dims, const std::vector<uint64_t>& strides) {
        uint64_t last = 0;
        for (size_t axis = 0; axis < dims.size(); ++axis) {
            if (dims[axis] == 0) {
                throw std::invalid_argument("StridedSpan dimensions must be non-zero.");
            }
            const uint64_t steps = dims[axis] - 1;
            if (steps != 0 && strides[axis] > (std::numeric_limits<uint64_t>::max() - last) / steps) {
                throw std::invalid_argument("StridedSpan layout overflows uint64_t.");
            }
            last += steps * strides[axis];
        }
        return last + 1;
    }

    [[nodiscard]] static std::vector<uint64_t> denseStrides(const std::vector<uint64_t>& dims) {
        std::vector<uint64_t> out(dims.size(), 1);
        for (size_t axis = dims.size(); axis-- > 1;) {
            out[axis - 1] = out[axis] * dims[axis];
        }
        return out;
    }

    // The storage the layout reaches: its source tensor, the element its first element is at, and how many it reaches.
    [[nodiscard]] const TensorSpan& getStorage() const { return storage; }
    // The source tensor's own view of its storage, for identity (getTensorId) and lifetime, never for addressing.
    [[nodiscard]] const Tensor& getSourceTensor() const { return storage.tensor; }
    // True when the span is the whole of its source tensor, in the source's own layout.
    [[nodiscard]] bool isWholeTensor() const {
        return storage.offset == 0 && dataType == storage.tensor.getDataType() && dims == storage.tensor.getDimensionsRef() &&
               strides == storage.tensor.getStridesElements();
    }

    [[nodiscard]] bool isInitialized() const { return storage.isInitialized(); }
    [[nodiscard]] const std::vector<uint64_t>& getDimensions() const { return dims; }
    [[nodiscard]] const std::vector<uint64_t>& getStridesElements() const { return strides; }
    // True when the layout is dense row-major (it may still start partway into its source).
    [[nodiscard]] bool isDenseContiguous() const { return strides == denseStrides(dims); }
    [[nodiscard]] uint64_t getTotalNumElements() const {
        uint64_t elements = 1;
        for (uint64_t dim : dims) elements *= dim;
        return elements;
    }
    [[nodiscard]] DataType getDataType() const { return dataType; }
    [[nodiscard]] TensorPlacement getPlacement() const { return storage.getPlacement(); }
    [[nodiscard]] TensorDescriptor getDescriptor() const { return TensorDescriptor(getDataType(), dims); }
    // The bytes of the logical elements (as for a Tensor, not the gaps between them).
    [[nodiscard]] uint64_t getArraySizeInBytes() const { return bytesOf(dataType, getTotalNumElements()); }
    // The span's first element.
    template <typename T = void>
    [[nodiscard]] T* getMemPtr() const {
        return static_cast<T*>(storage.address());
    }
    // The largest power of two, at most `limit`, that the first element's address is a multiple of.
    [[nodiscard]] uint64_t alignmentBytes(uint64_t limit = 256) const {
        if (limit == 0 || (limit & (limit - 1)) != 0) {
            throw std::invalid_argument("StridedSpan::alignmentBytes limit must be a power of two.");
        }
        const uintptr_t address = reinterpret_cast<uintptr_t>(storage.address());
        uint64_t alignment = 1;
        while (alignment < limit && (address & static_cast<uintptr_t>(alignment * 2 - 1)) == 0) alignment *= 2;
        return alignment;
    }

    // A dense span read under other dimensions with the same element count, as Tensor::reshape.
    void reshape(const std::vector<uint64_t>& newDims) {
        std::vector<uint64_t> newStrides = denseStrides(newDims);
        if (!isDenseContiguous() || reachedElements(newDims, newStrides) != getTotalNumElements()) {
            throw std::invalid_argument("StridedSpan::reshape requires a dense span and the same element count.");
        }
        dims = newDims;
        strides = std::move(newStrides);
    }

   private:
    [[nodiscard]] static uint64_t bytesOf(DataType dtype, uint64_t elements) {
        return TensorDescriptor(dtype, {elements}).getArraySizeInBytes();
    }

    TensorSpan storage;
    std::vector<uint64_t> dims;
    std::vector<uint64_t> strides;
    DataType dataType = DataType::UINT8;
};

}  // namespace ThorImplementation
