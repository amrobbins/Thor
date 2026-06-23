#include "Utilities/TensorOperations/Ensemble/EnsembleWeightedMean.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Common/ScopedGpu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation::Ensemble {
namespace {

template <typename T>
__device__ double toDoubleDevice(T value) {
    return static_cast<double>(value);
}

template <>
__device__ double toDoubleDevice<half>(half value) {
    return static_cast<double>(__half2float(value));
}

template <>
__device__ double toDoubleDevice<__nv_bfloat16>(__nv_bfloat16 value) {
    return static_cast<double>(__bfloat162float(value));
}

template <typename T>
__device__ T fromDoubleDevice(double value) {
    return static_cast<T>(value);
}

template <>
__device__ half fromDoubleDevice<half>(double value) {
    return __float2half(static_cast<float>(value));
}

template <>
__device__ __nv_bfloat16 fromDoubleDevice<__nv_bfloat16>(double value) {
    return __float2bfloat16(static_cast<float>(value));
}

template <typename T>
__global__ void weightedMeanKernel(T* destination,
                                   const T* const* sources,
                                   const double* weights,
                                   double inverseWeightSum,
                                   uint32_t numSources,
                                   uint64_t numElements) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t element = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         element < numElements;
         element += stride) {
        double accumulator = 0.0;
        for (uint32_t sourceIndex = 0; sourceIndex < numSources; ++sourceIndex) {
            accumulator += weights[sourceIndex] * toDoubleDevice<T>(sources[sourceIndex][element]);
        }
        destination[element] = fromDoubleDevice<T>(accumulator * inverseWeightSum);
    }
}

template <typename T>
void launchTypedWeightedMean(Tensor destination,
                             const std::vector<Tensor>& sources,
                             const std::vector<double>& weights,
                             double weightSum,
                             Stream stream) {
    const uint32_t numSources = static_cast<uint32_t>(sources.size());
    const uint64_t numElements = destination.getTotalNumElements();

    std::vector<const T*> sourcePtrsHost;
    sourcePtrsHost.reserve(sources.size());
    for (const Tensor& source : sources) {
        sourcePtrsHost.push_back(source.getMemPtr<T>());
    }

    const T** sourcePtrsDevice = nullptr;
    double* weightsDevice = nullptr;

    ScopedGpu scopedGpu(stream.getGpuNum());
    cudaError_t status = cudaMallocAsync(reinterpret_cast<void**>(&sourcePtrsDevice), sizeof(T*) * numSources, stream.getStream());
    THOR_THROW_IF_FALSE(status == cudaSuccess);
    status = cudaMallocAsync(reinterpret_cast<void**>(&weightsDevice), sizeof(double) * numSources, stream.getStream());
    THOR_THROW_IF_FALSE(status == cudaSuccess);

    status = cudaMemcpyAsync(sourcePtrsDevice,
                             sourcePtrsHost.data(),
                             sizeof(T*) * numSources,
                             cudaMemcpyHostToDevice,
                             stream.getStream());
    THOR_THROW_IF_FALSE(status == cudaSuccess);
    status = cudaMemcpyAsync(weightsDevice,
                             weights.data(),
                             sizeof(double) * numSources,
                             cudaMemcpyHostToDevice,
                             stream.getStream());
    THOR_THROW_IF_FALSE(status == cudaSuccess);

    const uint32_t blockSize = 256;
    uint64_t blocks64 = (numElements + blockSize - 1) / blockSize;
    if (blocks64 == 0) {
        blocks64 = 1;
    }
    const uint32_t gridSize = static_cast<uint32_t>(std::min<uint64_t>(blocks64, 65535));
    weightedMeanKernel<T><<<gridSize, blockSize, 0, stream.getStream()>>>(
        destination.getMemPtr<T>(),
        sourcePtrsDevice,
        weightsDevice,
        1.0 / weightSum,
        numSources,
        numElements);
    status = cudaGetLastError();
    THOR_THROW_IF_FALSE(status == cudaSuccess);

    status = cudaFreeAsync(const_cast<T**>(sourcePtrsDevice), stream.getStream());
    THOR_THROW_IF_FALSE(status == cudaSuccess);
    status = cudaFreeAsync(weightsDevice, stream.getStream());
    THOR_THROW_IF_FALSE(status == cudaSuccess);
}

void validateWeightedMeanInputs(Tensor destination, const std::vector<Tensor>& sources, const std::vector<double>& weights, Stream stream) {
    if (sources.empty()) {
        throw std::runtime_error("Ensemble weighted-mean aggregation requires at least one source tensor.");
    }
    if (sources.size() != weights.size()) {
        throw std::runtime_error("Ensemble weighted-mean aggregation requires one weight per source tensor.");
    }
    if (destination.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::runtime_error("Ensemble weighted-mean destination tensor must be GPU-resident.");
    }
    if (destination.getPlacement().getDeviceNum() != stream.getGpuNum()) {
        throw std::runtime_error("Ensemble weighted-mean destination tensor must be on the aggregation stream GPU.");
    }

    const TensorDescriptor destinationDescriptor = destination.getDescriptor();
    double weightSum = 0.0;
    for (size_t i = 0; i < sources.size(); ++i) {
        const Tensor& source = sources[i];
        if (source.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
            throw std::runtime_error("Ensemble weighted-mean source tensors must be GPU-resident.");
        }
        if (source.getPlacement().getDeviceNum() != destination.getPlacement().getDeviceNum()) {
            throw std::runtime_error("Ensemble weighted-mean currently requires all source tensors to live on the destination GPU.");
        }
        if (source.getDescriptor() != destinationDescriptor) {
            throw std::runtime_error("Ensemble weighted-mean source and destination descriptors must match.");
        }
        if (!std::isfinite(weights[i]) || weights[i] <= 0.0) {
            throw std::runtime_error("Ensemble weighted-mean source weights must be finite and positive.");
        }
        weightSum += weights[i];
    }
    if (!std::isfinite(weightSum) || weightSum <= 0.0) {
        throw std::runtime_error("Ensemble weighted-mean source weights must have a positive finite sum.");
    }
}

}  // namespace

void launchWeightedMean(Tensor destination,
                        const std::vector<Tensor>& sources,
                        const std::vector<double>& weights,
                        Stream stream) {
    validateWeightedMeanInputs(destination, sources, weights, stream);

    double weightSum = 0.0;
    for (double weight : weights) {
        weightSum += weight;
    }

    switch (destination.getDescriptor().getDataType()) {
        case DataType::FP16:
            launchTypedWeightedMean<half>(destination, sources, weights, weightSum, stream);
            return;
        case DataType::FP32:
            launchTypedWeightedMean<float>(destination, sources, weights, weightSum, stream);
            return;
        case DataType::FP64:
            launchTypedWeightedMean<double>(destination, sources, weights, weightSum, stream);
            return;
        case DataType::BF16:
            launchTypedWeightedMean<__nv_bfloat16>(destination, sources, weights, weightSum, stream);
            return;
        default:
            throw std::runtime_error("Ensemble weighted-mean aggregation currently supports FP16, BF16, FP32, and FP64 outputs, not " +
                                     TensorDescriptor::getElementTypeName(destination.getDescriptor().getDataType()) + ".");
    }
}

}  // namespace ThorImplementation::Ensemble
