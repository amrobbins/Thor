#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace ThorImplementation {

// Runtime capabilities for CUDA's persisting-L2 access policy.  L2 cache
// persistence is a performance hint, not a correctness requirement, so callers
// need enough information to make an explicit fallback decision rather than
// treating an unsupported configuration as a fatal error.
struct PersistingL2Capabilities {
    bool query_succeeded = false;
    bool supported = false;

    uint64_t l2_bytes = 0;
    uint64_t max_persisting_bytes = 0;
    uint64_t max_access_policy_window_bytes = 0;
    uint64_t current_persisting_bytes = 0;

    int compute_capability_major = 0;
    int compute_capability_minor = 0;

    cudaError_t cuda_status = cudaSuccess;
    std::string detail;
};

enum class PersistingL2OperationStatus : uint8_t {
    SUCCESS,
    UNSUPPORTED,
    INVALID_ARGUMENT,
    CUDA_ERROR,
};

struct PersistingL2OperationResult {
    PersistingL2OperationStatus status = PersistingL2OperationStatus::SUCCESS;
    cudaError_t cuda_status = cudaSuccess;
    std::string detail;

    [[nodiscard]] bool succeeded() const { return status == PersistingL2OperationStatus::SUCCESS; }
    [[nodiscard]] bool unsupported() const { return status == PersistingL2OperationStatus::UNSUPPORTED; }
    explicit operator bool() const { return succeeded(); }
};

// Querying is deliberately non-throwing.  Devices before compute capability
// 8.0, MIG configurations with no set-aside, and other unsupported CUDA
// configurations are reported through the returned capabilities.
[[nodiscard]] PersistingL2Capabilities queryPersistingL2Capabilities(int gpu_num) noexcept;

// Set the device/context-wide L2 set-aside.  The requested value must not
// exceed max_persisting_bytes.  CUDA may reject this operation in configurations
// such as MPS even when access-policy windows themselves are supported.
[[nodiscard]] PersistingL2OperationResult trySetPersistingL2SetAsideBytes(int gpu_num, uint64_t bytes) noexcept;

// Apply a persisting access-policy window to one stream.  bytes must be nonzero
// and no larger than max_access_policy_window_bytes; hit_ratio must be in [0,1].
// Misses use cudaAccessPropertyStreaming, matching NVIDIA's recommended
// persisting-window pattern.
[[nodiscard]] PersistingL2OperationResult trySetPersistingL2AccessPolicyWindow(
    int gpu_num, cudaStream_t stream, const void* base, uint64_t bytes, float hit_ratio) noexcept;

// Disable the stream's access-policy window.  This stops assigning new
// persisting accesses but does not globally evict/reset already-persisting L2
// lines; use tryResetPersistingL2Cache() for that device-wide operation.
[[nodiscard]] PersistingL2OperationResult tryClearPersistingL2AccessPolicyWindow(int gpu_num, cudaStream_t stream) noexcept;

// Reset persisting cache lines in the current CUDA context for gpu_num back to
// normal cache status.  This is context/device-wide and therefore intentionally
// separate from the per-stream clear operation.
[[nodiscard]] PersistingL2OperationResult tryResetPersistingL2Cache(int gpu_num) noexcept;

}  // namespace ThorImplementation
