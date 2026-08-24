#include "Utilities/Common/PersistingL2Cache.h"

#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

namespace ThorImplementation {
namespace {

class DeviceRestoreGuard {
   public:
    DeviceRestoreGuard() = default;
    ~DeviceRestoreGuard() {
        if (switched && previous_gpu >= 0)
            (void)cudaSetDevice(previous_gpu);
    }

    DeviceRestoreGuard(const DeviceRestoreGuard&) = delete;
    DeviceRestoreGuard& operator=(const DeviceRestoreGuard&) = delete;

    int previous_gpu = -1;
    bool switched = false;
};

std::string cudaFailureDetail(const char* operation, cudaError_t status) {
    std::ostringstream message;
    message << operation << " failed with ";
    const char* name = cudaGetErrorName(status);
    const char* description = cudaGetErrorString(status);
    message << (name != nullptr ? name : "cudaErrorUnknown") << " (" << static_cast<int>(status) << "): "
            << (description != nullptr ? description : "<no description>");
    return message.str();
}

PersistingL2OperationResult invalidArgument(std::string detail) {
    return PersistingL2OperationResult{
        .status = PersistingL2OperationStatus::INVALID_ARGUMENT,
        .cuda_status = cudaErrorInvalidValue,
        .detail = std::move(detail),
    };
}

PersistingL2OperationResult unsupported(std::string detail, cudaError_t status = cudaSuccess) {
    return PersistingL2OperationResult{
        .status = PersistingL2OperationStatus::UNSUPPORTED,
        .cuda_status = status,
        .detail = std::move(detail),
    };
}

PersistingL2OperationResult cudaFailure(const char* operation, cudaError_t status) {
    return PersistingL2OperationResult{
        .status = PersistingL2OperationStatus::CUDA_ERROR,
        .cuda_status = status,
        .detail = cudaFailureDetail(operation, status),
    };
}

PersistingL2OperationResult success() { return {}; }

PersistingL2OperationResult selectDevice(int gpu_num, DeviceRestoreGuard& restore) noexcept {
    if (gpu_num < 0)
        return invalidArgument("gpu_num must be non-negative");

    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess)
        return cudaFailure("cudaGetDeviceCount", status);
    if (gpu_num >= device_count) {
        return invalidArgument("gpu_num " + std::to_string(gpu_num) + " is outside the available CUDA device range [0, " +
                               std::to_string(device_count) + ")");
    }

    status = cudaGetDevice(&restore.previous_gpu);
    if (status != cudaSuccess)
        return cudaFailure("cudaGetDevice", status);

    restore.switched = restore.previous_gpu != gpu_num;
    if (restore.switched) {
        status = cudaSetDevice(gpu_num);
        if (status != cudaSuccess)
            return cudaFailure("cudaSetDevice", status);
    }
    return success();
}

PersistingL2OperationResult capabilitiesFailure(const PersistingL2Capabilities& capabilities) {
    if (!capabilities.query_succeeded) {
        return PersistingL2OperationResult{
            .status = capabilities.cuda_status == cudaErrorInvalidValue ? PersistingL2OperationStatus::INVALID_ARGUMENT
                                                                        : PersistingL2OperationStatus::CUDA_ERROR,
            .cuda_status = capabilities.cuda_status,
            .detail = capabilities.detail,
        };
    }
    return unsupported(capabilities.detail, capabilities.cuda_status);
}

}  // namespace

PersistingL2Capabilities queryPersistingL2Capabilities(int gpu_num) noexcept {
    PersistingL2Capabilities capabilities;
    DeviceRestoreGuard restore;
    PersistingL2OperationResult selected = selectDevice(gpu_num, restore);
    if (!selected.succeeded()) {
        capabilities.cuda_status = selected.cuda_status;
        capabilities.detail = std::move(selected.detail);
        return capabilities;
    }

    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDeviceProperties(&properties, gpu_num);
    if (status != cudaSuccess) {
        capabilities.cuda_status = status;
        capabilities.detail = cudaFailureDetail("cudaGetDeviceProperties", status);
        return capabilities;
    }

    capabilities.compute_capability_major = properties.major;
    capabilities.compute_capability_minor = properties.minor;
    capabilities.l2_bytes = properties.l2CacheSize > 0 ? static_cast<uint64_t>(properties.l2CacheSize) : 0;
    capabilities.max_persisting_bytes =
        properties.persistingL2CacheMaxSize > 0 ? static_cast<uint64_t>(properties.persistingL2CacheMaxSize) : 0;
    capabilities.max_access_policy_window_bytes =
        properties.accessPolicyMaxWindowSize > 0 ? static_cast<uint64_t>(properties.accessPolicyMaxWindowSize) : 0;
    capabilities.query_succeeded = true;

    if (properties.major < 8) {
        capabilities.detail = "persisting L2 cache access policy requires CUDA compute capability 8.0 or newer";
        return capabilities;
    }
    if (capabilities.max_access_policy_window_bytes == 0) {
        capabilities.detail = "CUDA reports max access-policy window size of zero; persisting L2 access policy is unavailable";
        return capabilities;
    }
    if (capabilities.max_persisting_bytes == 0) {
        capabilities.detail =
            "CUDA reports max persisting L2 set-aside size of zero; this can occur on unsupported or MIG configurations";
        return capabilities;
    }

    size_t current_persisting_bytes = 0;
    status = cudaDeviceGetLimit(&current_persisting_bytes, cudaLimitPersistingL2CacheSize);
    if (status != cudaSuccess) {
        capabilities.query_succeeded = false;
        capabilities.cuda_status = status;
        capabilities.detail = cudaFailureDetail("cudaDeviceGetLimit(cudaLimitPersistingL2CacheSize)", status);
        return capabilities;
    }
    capabilities.current_persisting_bytes = static_cast<uint64_t>(current_persisting_bytes);
    capabilities.supported = true;
    capabilities.detail.clear();
    return capabilities;
}

PersistingL2OperationResult trySetPersistingL2SetAsideBytes(int gpu_num, uint64_t bytes) noexcept {
    const PersistingL2Capabilities capabilities = queryPersistingL2Capabilities(gpu_num);
    if (!capabilities.supported)
        return capabilitiesFailure(capabilities);
    if (bytes > capabilities.max_persisting_bytes) {
        return invalidArgument("requested persisting L2 set-aside of " + std::to_string(bytes) +
                               " bytes exceeds device maximum of " + std::to_string(capabilities.max_persisting_bytes) +
                               " bytes");
    }
    if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return invalidArgument("requested persisting L2 set-aside does not fit in size_t");

    DeviceRestoreGuard restore;
    PersistingL2OperationResult selected = selectDevice(gpu_num, restore);
    if (!selected.succeeded())
        return selected;

    const cudaError_t status = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, static_cast<size_t>(bytes));
    if (status == cudaErrorUnsupportedLimit)
        return unsupported(cudaFailureDetail("cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)", status), status);
    if (status != cudaSuccess)
        return cudaFailure("cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)", status);
    return success();
}

PersistingL2OperationResult trySetPersistingL2AccessPolicyWindow(
    int gpu_num, cudaStream_t stream, const void* base, uint64_t bytes, float hit_ratio) noexcept {
    if (stream == nullptr)
        return invalidArgument("stream must be initialized");
    if (base == nullptr)
        return invalidArgument("persisting L2 access-policy window base must not be null");
    if (bytes == 0)
        return invalidArgument("persisting L2 access-policy window bytes must be greater than zero");
    if (!std::isfinite(hit_ratio) || hit_ratio < 0.0f || hit_ratio > 1.0f)
        return invalidArgument("persisting L2 access-policy hit_ratio must be finite and in [0, 1]");

    const PersistingL2Capabilities capabilities = queryPersistingL2Capabilities(gpu_num);
    if (!capabilities.supported)
        return capabilitiesFailure(capabilities);
    if (bytes > capabilities.max_access_policy_window_bytes) {
        return invalidArgument("requested persisting L2 access-policy window of " + std::to_string(bytes) +
                               " bytes exceeds device maximum of " +
                               std::to_string(capabilities.max_access_policy_window_bytes) + " bytes");
    }
    if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return invalidArgument("requested persisting L2 access-policy window does not fit in size_t");

    DeviceRestoreGuard restore;
    PersistingL2OperationResult selected = selectDevice(gpu_num, restore);
    if (!selected.succeeded())
        return selected;

    cudaStreamAttrValue attribute{};
    attribute.accessPolicyWindow.base_ptr = const_cast<void*>(base);
    attribute.accessPolicyWindow.num_bytes = static_cast<size_t>(bytes);
    attribute.accessPolicyWindow.hitRatio = hit_ratio;
    attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    const cudaError_t status =
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attribute);
    if (status == cudaErrorNotSupported)
        return unsupported(cudaFailureDetail("cudaStreamSetAttribute(cudaStreamAttributeAccessPolicyWindow)", status), status);
    if (status != cudaSuccess)
        return cudaFailure("cudaStreamSetAttribute(cudaStreamAttributeAccessPolicyWindow)", status);
    return success();
}

PersistingL2OperationResult tryClearPersistingL2AccessPolicyWindow(int gpu_num, cudaStream_t stream) noexcept {
    if (stream == nullptr)
        return invalidArgument("stream must be initialized");

    DeviceRestoreGuard restore;
    PersistingL2OperationResult selected = selectDevice(gpu_num, restore);
    if (!selected.succeeded())
        return selected;

    cudaStreamAttrValue attribute{};
    attribute.accessPolicyWindow.base_ptr = nullptr;
    attribute.accessPolicyWindow.num_bytes = 0;
    attribute.accessPolicyWindow.hitRatio = 0.0f;
    attribute.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
    attribute.accessPolicyWindow.missProp = cudaAccessPropertyNormal;

    const cudaError_t status =
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attribute);
    if (status == cudaErrorNotSupported)
        return unsupported(cudaFailureDetail("cudaStreamSetAttribute(cudaStreamAttributeAccessPolicyWindow)", status), status);
    if (status != cudaSuccess)
        return cudaFailure("cudaStreamSetAttribute(cudaStreamAttributeAccessPolicyWindow)", status);
    return success();
}

PersistingL2OperationResult tryResetPersistingL2Cache(int gpu_num) noexcept {
    const PersistingL2Capabilities capabilities = queryPersistingL2Capabilities(gpu_num);
    if (!capabilities.supported)
        return capabilitiesFailure(capabilities);

    DeviceRestoreGuard restore;
    PersistingL2OperationResult selected = selectDevice(gpu_num, restore);
    if (!selected.succeeded())
        return selected;

    const cudaError_t status = cudaCtxResetPersistingL2Cache();
    if (status == cudaErrorNotSupported)
        return unsupported(cudaFailureDetail("cudaCtxResetPersistingL2Cache", status), status);
    if (status != cudaSuccess)
        return cudaFailure("cudaCtxResetPersistingL2Cache", status);
    return success();
}

}  // namespace ThorImplementation
