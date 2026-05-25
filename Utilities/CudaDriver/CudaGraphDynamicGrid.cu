#include "Utilities/CudaDriver/CudaGraphDynamicGrid.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation {
namespace {


bool isSupportedCountDType(DataType dtype) {
    return dtype == DataType::UINT16 || dtype == DataType::UINT32 || dtype == DataType::UINT64;
}

std::string dtypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

void validateCountTensor(const Tensor& tensor, int gpuNum, const char* label) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument(std::string("CUDA graph dynamic-grid ") + label + " tensor is not initialized.");
    }
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU || tensor.getPlacement().getDeviceNum() != gpuNum) {
        throw std::invalid_argument(std::string("CUDA graph dynamic-grid ") + label + " tensor must live on the target GPU.");
    }
    if (tensor.hasCustomStrides() || !tensor.isDenseContiguous()) {
        throw std::invalid_argument(std::string("CUDA graph dynamic-grid ") + label + " tensor must be dense contiguous.");
    }
    if (tensor.getDimensions() != std::vector<uint64_t>{1}) {
        throw std::invalid_argument(std::string("CUDA graph dynamic-grid ") + label + " tensor must have shape [1].");
    }
    if (!isSupportedCountDType(tensor.getDataType())) {
        throw std::invalid_argument(std::string("CUDA graph dynamic-grid ") + label + " tensor dtype " + dtypeName(tensor.getDataType()) +
                                    " is unsupported. Supported dtypes are uint16, uint32, and uint64.");
    }
}

void validateDescriptorCommon(const DeviceUpdatableKernelNodeDeviceHandle* targetNode, const Tensor& countTensor, Stream stream, const char* label) {
    if (targetNode == nullptr || !targetNode->isInitialized()) {
        throw std::invalid_argument(std::string("CUDA graph dynamic-grid ") + label + " requires an initialized target-node device handle.");
    }
    if (!stream.isInitialized()) {
        throw std::invalid_argument(std::string("CUDA graph dynamic-grid ") + label + " requires an initialized stream.");
    }
    if (targetNode->getGpuNum() != stream.getGpuNum()) {
        throw std::invalid_argument(std::string("CUDA graph dynamic-grid ") + label + " stream must be on the target-node GPU.");
    }
    validateCountTensor(countTensor, stream.getGpuNum(), label);
}

__device__ __forceinline__ unsigned long long checkedMulU64(unsigned long long a, unsigned long long b) {
    if (a != 0ULL && b > 0xffffffffffffffffULL / a) {
        asm("trap;");
    }
    return a * b;
}

__device__ __forceinline__ unsigned int checkedGridDim(unsigned long long value, unsigned int minGrid, unsigned int maxGrid) {
    if (minGrid == 0U || maxGrid == 0U || minGrid > maxGrid) {
        asm("trap;");
    }
    unsigned long long clamped = value;
    if (clamped < static_cast<unsigned long long>(minGrid)) {
        clamped = static_cast<unsigned long long>(minGrid);
    }
    if (clamped > static_cast<unsigned long long>(maxGrid) || clamped > 0xffffffffULL) {
        asm("trap;");
    }
    return static_cast<unsigned int>(clamped);
}

__device__ __forceinline__ cudaGraphDeviceNode_t loadTargetNode(const cudaGraphDeviceNode_t* targetNode) {
    if (targetNode == nullptr) {
        asm("trap;");
    }
    cudaGraphDeviceNode_t node = *targetNode;
    if (node == nullptr) {
        asm("trap;");
    }
    return node;
}

template <typename CountT>
__global__ void updateDeviceGrid1DFromScalarKernel(const cudaGraphDeviceNode_t* targetNode,
                                                  const CountT* itemCount,
                                                  unsigned long long itemsPerCount,
                                                  unsigned int targetBlockDimX,
                                                  unsigned int minGridDimX,
                                                  unsigned int maxGridDimX) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    if (itemCount == nullptr || targetBlockDimX == 0U || itemsPerCount == 0ULL) {
        asm("trap;");
    }

    const unsigned long long count = static_cast<unsigned long long>(itemCount[0]);
    const unsigned long long items = checkedMulU64(count, itemsPerCount);
    const unsigned long long grid64 = (items + static_cast<unsigned long long>(targetBlockDimX) - 1ULL) /
                                      static_cast<unsigned long long>(targetBlockDimX);
    const unsigned int gridX = checkedGridDim(grid64, minGridDimX, maxGridDimX);
    cudaError_t status = cudaGraphKernelNodeSetGridDim(loadTargetNode(targetNode), dim3(gridX, 1U, 1U));
    if (status != cudaSuccess) {
        asm("trap;");
    }
}

template <typename CountT>
__global__ void updateDeviceGrid2DFromScalarKernel(const cudaGraphDeviceNode_t* targetNode,
                                                  const CountT* rowCount,
                                                  unsigned long long gridDimXPerRow,
                                                  unsigned int gridDimY,
                                                  unsigned int minGridDimX,
                                                  unsigned int maxGridDimX,
                                                  unsigned int maxGridDimY) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    if (rowCount == nullptr || gridDimXPerRow == 0ULL || gridDimY == 0U || gridDimY > maxGridDimY) {
        asm("trap;");
    }

    const unsigned long long rows = static_cast<unsigned long long>(rowCount[0]);
    const unsigned long long gridX64 = checkedMulU64(rows, gridDimXPerRow);
    const unsigned int gridX = checkedGridDim(gridX64, minGridDimX, maxGridDimX);
    cudaError_t status = cudaGraphKernelNodeSetGridDim(loadTargetNode(targetNode), dim3(gridX, gridDimY, 1U));
    if (status != cudaSuccess) {
        asm("trap;");
    }
}

template <typename CountT>
void launchUpdateDeviceGrid1DTyped(const DynamicGrid1DFromScalarDescriptor& descriptor, Stream stream) {
    const CountT* itemCount = descriptor.itemCount.getMemPtr<CountT>();
    updateDeviceGrid1DFromScalarKernel<CountT><<<1, 1, 0, stream.getStream()>>>(descriptor.targetNode->devicePtr(),
                                                                                itemCount,
                                                                                descriptor.itemsPerCount,
                                                                                descriptor.targetBlockDimX,
                                                                                descriptor.minGridDimX,
                                                                                descriptor.maxGridDimX);
    CUDA_CHECK(cudaGetLastError());
}

template <typename CountT>
void launchUpdateDeviceGrid2DTyped(const DynamicGrid2DFromScalarDescriptor& descriptor, Stream stream) {
    const CountT* rowCount = descriptor.rowCount.getMemPtr<CountT>();
    updateDeviceGrid2DFromScalarKernel<CountT><<<1, 1, 0, stream.getStream()>>>(descriptor.targetNode->devicePtr(),
                                                                                rowCount,
                                                                                descriptor.gridDimXPerRow,
                                                                                descriptor.gridDimY,
                                                                                descriptor.minGridDimX,
                                                                                descriptor.maxGridDimX,
                                                                                descriptor.maxGridDimY);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launchUpdateDeviceGrid1DFromScalar(const DynamicGrid1DFromScalarDescriptor& descriptor, Stream stream) {
    validateDescriptorCommon(descriptor.targetNode, descriptor.itemCount, stream, "1D update");
    if (descriptor.itemsPerCount == 0) {
        throw std::invalid_argument("CUDA graph dynamic-grid 1D update itemsPerCount must be non-zero.");
    }
    if (descriptor.targetBlockDimX == 0) {
        throw std::invalid_argument("CUDA graph dynamic-grid 1D update targetBlockDimX must be non-zero.");
    }
    if (descriptor.minGridDimX == 0 || descriptor.maxGridDimX == 0 || descriptor.minGridDimX > descriptor.maxGridDimX) {
        throw std::invalid_argument("CUDA graph dynamic-grid 1D update requires 0 < minGridDimX <= maxGridDimX.");
    }

    ScopedGpu scoped(stream.getGpuNum());
    switch (descriptor.itemCount.getDataType()) {
        case DataType::UINT16:
            launchUpdateDeviceGrid1DTyped<uint16_t>(descriptor, stream);
            break;
        case DataType::UINT32:
            launchUpdateDeviceGrid1DTyped<uint32_t>(descriptor, stream);
            break;
        case DataType::UINT64:
            launchUpdateDeviceGrid1DTyped<uint64_t>(descriptor, stream);
            break;
        default:
            throw std::invalid_argument("CUDA graph dynamic-grid 1D update encountered unsupported count dtype.");
    }
}

void launchUpdateDeviceGrid2DFromScalar(const DynamicGrid2DFromScalarDescriptor& descriptor, Stream stream) {
    validateDescriptorCommon(descriptor.targetNode, descriptor.rowCount, stream, "2D update");
    if (descriptor.gridDimXPerRow == 0) {
        throw std::invalid_argument("CUDA graph dynamic-grid 2D update gridDimXPerRow must be non-zero.");
    }
    if (descriptor.gridDimY == 0) {
        throw std::invalid_argument("CUDA graph dynamic-grid 2D update gridDimY must be non-zero.");
    }
    if (descriptor.minGridDimX == 0 || descriptor.maxGridDimX == 0 || descriptor.minGridDimX > descriptor.maxGridDimX) {
        throw std::invalid_argument("CUDA graph dynamic-grid 2D update requires 0 < minGridDimX <= maxGridDimX.");
    }
    if (descriptor.gridDimY > descriptor.maxGridDimY) {
        throw std::invalid_argument("CUDA graph dynamic-grid 2D update gridDimY exceeds maxGridDimY.");
    }

    ScopedGpu scoped(stream.getGpuNum());
    switch (descriptor.rowCount.getDataType()) {
        case DataType::UINT16:
            launchUpdateDeviceGrid2DTyped<uint16_t>(descriptor, stream);
            break;
        case DataType::UINT32:
            launchUpdateDeviceGrid2DTyped<uint32_t>(descriptor, stream);
            break;
        case DataType::UINT64:
            launchUpdateDeviceGrid2DTyped<uint64_t>(descriptor, stream);
            break;
        default:
            throw std::invalid_argument("CUDA graph dynamic-grid 2D update encountered unsupported count dtype.");
    }
}

}  // namespace ThorImplementation
