#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"

#include <cuda_runtime_api.h>

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace ThorImplementation::CubReductionInternal {

// CUDA identity and resource geometry are process-lifetime device properties. Cache them by ordinal so every
// selection-cache hit is free of CUDA property queries. No selection or physical-plan state is cached here.
DenseReductionDeviceIdentity denseReductionDeviceIdentity(int gpu) {
    static std::mutex mutex;
    static std::unordered_map<int, DenseReductionDeviceIdentity> identities;
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto it = identities.find(gpu); it != identities.end()) return it->second;
    cudaDeviceProp properties{};
    const cudaError_t status = cudaGetDeviceProperties(&properties, gpu);
    if (status != cudaSuccess || properties.multiProcessorCount <= 0)
        throw std::runtime_error("Dense reduction selection could not query the CUDA device identity");
    DenseReductionDeviceIdentity identity;
    static_assert(sizeof(properties.uuid.bytes) == 16);
    std::memcpy(identity.uuid.data(), properties.uuid.bytes, identity.uuid.size());
    identity.ordinal = gpu;
    identity.sm_count = static_cast<uint32_t>(properties.multiProcessorCount);
    identity.compute_major = static_cast<uint32_t>(properties.major);
    identity.compute_minor = static_cast<uint32_t>(properties.minor);
    return identities.emplace(gpu, identity).first->second;
}

}  // namespace ThorImplementation::CubReductionInternal
