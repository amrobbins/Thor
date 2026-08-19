#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace ThorImplementation {

/**
 * Common contract for cuDNN Frontend execution workspace.
 *
 * Cached cuDNN graphs/plans are immutable execution metadata and may be shared
 * across compatible placed operations. Workspace is mutable execution scratch
 * and must instead be owned by the placed/stamped execution object (or another
 * serialization domain that guarantees non-overlap). A global graph cache must
 * never be the owner of workspace used by independently schedulable executions.
 *
 * All cuDNN Frontend graph caches must obey this ownership split. Cache entries
 * use CudnnCachedExecutionPlan so execution workspace cannot accidentally become
 * part of the shared cache object again.
 */

template <typename GraphT>
struct CudnnCachedExecutionPlan {
    std::shared_ptr<GraphT> graph;
    int64_t workspaceBytes = 0;
};

[[nodiscard]] uint64_t checkedCudnnWorkspaceSizeInBytes(int64_t reportedBytes, std::string_view operationName);

void validateCudnnExecutionWorkspace(const std::optional<Tensor>& workspace,
                                     uint64_t requiredBytes,
                                     int gpuNum,
                                     std::string_view operationName);

/**
 * Validate caller-owned workspace and return the mutable scratch pointer expected by cuDNN.
 * The optional is intentionally non-const: exposing mutable execution scratch must not
 * cast away constness from the caller-owned Tensor.
 * Returns nullptr when requiredBytes == 0. Supplying an optional workspace in
 * that case is allowed, but if supplied it must still satisfy the GPU/UINT8
 * workspace contract.
 */
[[nodiscard]] void* cudnnExecutionWorkspacePointer(std::optional<Tensor>& workspace,
                                                   uint64_t requiredBytes,
                                                   int gpuNum,
                                                   std::string_view operationName);

}  // namespace ThorImplementation
