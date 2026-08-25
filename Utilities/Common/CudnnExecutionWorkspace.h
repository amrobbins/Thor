#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cstdint>
#include <optional>
#include <string_view>

namespace ThorImplementation {

/**
 * Common validation helpers for caller-owned cuDNN execution workspace.
 *
 * This header deliberately has no cached-execution-plan abstraction. Thor's
 * accelerator backend policy permits process-global sharing only for immutable
 * selection/configuration recipes. Backend executable graphs/plans, descriptors,
 * and workspace are execution-local state owned by each independently executable
 * operation.
 */

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
