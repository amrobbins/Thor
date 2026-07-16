#include "DeepLearning/Implementation/Layers/Utility/FiniteCheckKernel.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ThorImplementation {
namespace {

template <typename T>
__device__ __forceinline__ double finiteCheckToDouble(T value) {
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __nv_fp8_e4m3> ||
                  std::is_same_v<T, __nv_fp8_e5m2>) {
        return static_cast<double>(static_cast<float>(value));
    } else {
        return static_cast<double>(value);
    }
}

template <typename T>
__global__ void finiteCheckKernel(const T *data,
                                  uint64_t numElements,
                                  uint32_t maxReportedIndices,
                                  FiniteCheckResult *result) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t index = first; index < numElements; index += stride) {
        const double value = finiteCheckToDouble(data[index]);
        FiniteCheckSampleKind kind = FiniteCheckSampleKind::NONE;
        if (isnan(value)) {
            kind = FiniteCheckSampleKind::NAN_VALUE;
            atomicAdd(reinterpret_cast<unsigned long long *>(&result->nanCount), 1ULL);
        } else if (isinf(value)) {
            if (value < 0.0) {
                kind = FiniteCheckSampleKind::NEGATIVE_INFINITY;
                atomicAdd(reinterpret_cast<unsigned long long *>(&result->negativeInfinityCount), 1ULL);
            } else {
                kind = FiniteCheckSampleKind::POSITIVE_INFINITY;
                atomicAdd(reinterpret_cast<unsigned long long *>(&result->positiveInfinityCount), 1ULL);
            }
        }

        if (kind == FiniteCheckSampleKind::NONE)
            continue;

        const unsigned long long sample =
            atomicAdd(reinterpret_cast<unsigned long long *>(&result->totalNonFinite), 1ULL);
        if (sample < maxReportedIndices) {
            result->flatIndices[sample] = index;
            result->kinds[sample] = static_cast<uint32_t>(kind);
        }
    }
}

template <typename T>
void launchTyped(const void *data,
                 uint64_t numElements,
                 uint32_t maxReportedIndices,
                 FiniteCheckResult *result,
                 Stream stream) {
    constexpr uint32_t threads = 256;
    const uint64_t requestedBlocks = (numElements + threads - 1) / threads;
    const uint32_t blocks = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(requestedBlocks, 65535)));
    finiteCheckKernel<T><<<blocks, threads, 0, stream.getStream()>>>(
        static_cast<const T *>(data), numElements, maxReportedIndices, result);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launchFiniteCheck(const void *data,
                       DataType dataType,
                       uint64_t numElements,
                       uint32_t maxReportedIndices,
                       FiniteCheckResult *result,
                       Stream stream) {
    THOR_THROW_IF_FALSE(data != nullptr);
    THOR_THROW_IF_FALSE(result != nullptr);
    THOR_THROW_IF_FALSE(numElements > 0);
    THOR_THROW_IF_FALSE(maxReportedIndices <= FINITE_CHECK_MAX_REPORTED_INDICES);

    switch (dataType) {
        case DataType::FP8_E4M3:
            launchTyped<__nv_fp8_e4m3>(data, numElements, maxReportedIndices, result, stream);
            return;
        case DataType::FP8_E5M2:
            launchTyped<__nv_fp8_e5m2>(data, numElements, maxReportedIndices, result, stream);
            return;
        case DataType::FP16:
            launchTyped<half>(data, numElements, maxReportedIndices, result, stream);
            return;
        case DataType::BF16:
            launchTyped<__nv_bfloat16>(data, numElements, maxReportedIndices, result, stream);
            return;
        case DataType::FP32:
            launchTyped<float>(data, numElements, maxReportedIndices, result, stream);
            return;
        case DataType::FP64:
            launchTyped<double>(data, numElements, maxReportedIndices, result, stream);
            return;
        default:
            throw std::invalid_argument("FiniteCheck GPU kernel only accepts floating-point tensor storage types.");
    }
}

}  // namespace ThorImplementation
