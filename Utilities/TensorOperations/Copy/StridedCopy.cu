#include "Utilities/TensorOperations/Copy/StridedCopy.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

namespace ThorImplementation {
namespace {

constexpr uint32_t kMaxTensorViewRank = 32;

struct TensorViewLayout {
    uint32_t rank = 0;
    uint64_t dimensions[kMaxTensorViewRank]{};
    uint64_t strides[kMaxTensorViewRank]{};
};

template <typename Element>
__global__ void materializeTensorViewKernel(const Element* source, Element* destination, uint64_t numElements, TensorViewLayout layout) {
    const uint64_t linearIndex = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linearIndex >= numElements) {
        return;
    }

    uint64_t remaining = linearIndex;
    uint64_t sourceIndex = 0;
    for (uint32_t reverseDimension = 0; reverseDimension < layout.rank; ++reverseDimension) {
        const uint32_t dimension = layout.rank - 1 - reverseDimension;
        const uint64_t coordinate = remaining % layout.dimensions[dimension];
        remaining /= layout.dimensions[dimension];
        sourceIndex += coordinate * layout.strides[dimension];
    }
    destination[linearIndex] = source[sourceIndex];
}

uint64_t checkedProduct(const std::vector<uint64_t>& values, size_t begin, size_t end) {
    uint64_t product = 1;
    for (size_t i = begin; i < end; ++i) {
        if (values[i] != 0 && product > std::numeric_limits<uint64_t>::max() / values[i]) {
            throw std::runtime_error("Tensor-view materialization dimension product overflow.");
        }
        product *= values[i];
    }
    return product;
}

bool tryMaterializePitchedView(const Tensor& source,
                               Tensor& destination,
                               const std::vector<uint64_t>& dimensions,
                               const std::vector<uint64_t>& strides,
                               uint64_t elementSizeBytes,
                               Stream stream) {
    size_t denseSuffixStart = dimensions.size();
    uint64_t expectedStride = 1;
    for (size_t i = dimensions.size(); i > 0; --i) {
        const size_t dimension = i - 1;
        if (strides[dimension] != expectedStride) {
            break;
        }
        denseSuffixStart = dimension;
        if (dimensions[dimension] != 0 && expectedStride > std::numeric_limits<uint64_t>::max() / dimensions[dimension]) {
            throw std::runtime_error("Tensor-view materialization stride overflow.");
        }
        expectedStride *= dimensions[dimension];
    }

    if (denseSuffixStart == 0 || denseSuffixStart == dimensions.size()) {
        return false;
    }

    for (size_t i = 0; i + 1 < denseSuffixStart; ++i) {
        if (dimensions[i + 1] != 0 && strides[i + 1] > std::numeric_limits<uint64_t>::max() / dimensions[i + 1]) {
            throw std::runtime_error("Tensor-view materialization prefix-stride overflow.");
        }
        if (strides[i] != dimensions[i + 1] * strides[i + 1]) {
            return false;
        }
    }

    const uint64_t rowElements = checkedProduct(dimensions, denseSuffixStart, dimensions.size());
    const uint64_t rowCount = checkedProduct(dimensions, 0, denseSuffixStart);
    if (rowElements > std::numeric_limits<size_t>::max() / elementSizeBytes ||
        strides[denseSuffixStart - 1] > std::numeric_limits<size_t>::max() / elementSizeBytes ||
        rowCount > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Tensor-view materialization pitched-copy size overflow.");
    }

    const size_t rowBytes = static_cast<size_t>(rowElements * elementSizeBytes);
    const size_t sourcePitchBytes = static_cast<size_t>(strides[denseSuffixStart - 1] * elementSizeBytes);
    if (sourcePitchBytes < rowBytes) {
        return false;
    }

    CUDA_CHECK(cudaMemcpy2DAsync(destination.getMemPtr<void>(),
                                 rowBytes,
                                 source.getMemPtr<void>(),
                                 sourcePitchBytes,
                                 rowBytes,
                                 static_cast<size_t>(rowCount),
                                 cudaMemcpyDeviceToDevice,
                                 stream.getStream()));
    return true;
}

template <typename Element>
void launchMaterializeTensorView(
    const Tensor& source, Tensor& destination, uint64_t numElements, const TensorViewLayout& layout, Stream stream) {
    constexpr uint32_t threadsPerBlock = 256;
    const uint64_t blocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Tensor-view materialization requires more CUDA blocks than a one-dimensional launch supports.");
    }

    materializeTensorViewKernel<Element><<<static_cast<uint32_t>(blocks), threadsPerBlock, 0, stream.getStream()>>>(
        static_cast<const Element*>(source.getMemPtr<void>()), static_cast<Element*>(destination.getMemPtr<void>()), numElements, layout);
    CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace

void materializeTensorViewAsync(const Tensor& source, Tensor& destination, Stream stream) {
    if (source.getDescriptor() != destination.getDescriptor()) {
        throw std::runtime_error("Tensor-view materialization requires identical source and destination descriptors. Source=" +
                                 source.getDescriptor().toString() + ", destination=" + destination.getDescriptor().toString() + ".");
    }
    if (source.getPlacement() != destination.getPlacement() || source.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::runtime_error("Tensor-view materialization requires source and destination on the same GPU placement.");
    }
    if (!destination.isDenseContiguous()) {
        throw std::runtime_error("Tensor-view materialization requires a dense contiguous destination tensor.");
    }

    const uint64_t numElements = source.getTotalNumElements();
    if (numElements == 0) {
        return;
    }

    ScopedGpu scopedGpu(source.getPlacement().getDeviceNum());
    if (source.isDenseContiguous()) {
        CUDA_CHECK(cudaMemcpyAsync(destination.getMemPtr<void>(),
                                   source.getMemPtr<void>(),
                                   source.getArraySizeInBytes(),
                                   cudaMemcpyDeviceToDevice,
                                   stream.getStream()));
        return;
    }

    const std::vector<uint64_t> dimensions = source.getDimensions();
    const std::vector<uint64_t> strides = source.getStridesElements();
    if (dimensions.empty() || dimensions.size() != strides.size()) {
        throw std::runtime_error("Tensor-view materialization received invalid source dimensions or strides.");
    }
    if (dimensions.size() > kMaxTensorViewRank) {
        throw std::runtime_error("Tensor-view materialization supports ranks up to " + std::to_string(kMaxTensorViewRank) +
                                 ", but received rank " + std::to_string(dimensions.size()) + ".");
    }

    const uint64_t arraySizeBytes = source.getArraySizeInBytes();
    if (arraySizeBytes % numElements != 0) {
        throw std::runtime_error("Tensor-view materialization could not determine the tensor element size.");
    }
    const uint64_t elementSizeBytes = arraySizeBytes / numElements;
    if (tryMaterializePitchedView(source, destination, dimensions, strides, elementSizeBytes, stream)) {
        return;
    }

    TensorViewLayout layout;
    layout.rank = static_cast<uint32_t>(dimensions.size());
    for (uint32_t dimension = 0; dimension < layout.rank; ++dimension) {
        layout.dimensions[dimension] = dimensions[dimension];
        layout.strides[dimension] = strides[dimension];
    }

    switch (elementSizeBytes) {
        case 1:
            launchMaterializeTensorView<uint8_t>(source, destination, numElements, layout, stream);
            break;
        case 2:
            launchMaterializeTensorView<uint16_t>(source, destination, numElements, layout, stream);
            break;
        case 4:
            launchMaterializeTensorView<uint32_t>(source, destination, numElements, layout, stream);
            break;
        case 8:
            launchMaterializeTensorView<uint64_t>(source, destination, numElements, layout, stream);
            break;
        default:
            throw std::runtime_error("Tensor-view materialization does not support the tensor element size " +
                                     std::to_string(elementSizeBytes) + " bytes.");
    }
}

}  // namespace ThorImplementation
