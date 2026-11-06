#pragma once

#include "Utilities/TensorOperations/Cub/DenseReductionPhysicalSelector.h"

#include <cstdint>
#include <limits>
#include <optional>

class Stream;

namespace ThorImplementation { enum class CubReductionOp : uint8_t; }

namespace ThorImplementation::CubReductionInternal {

// Shared translation of public VALUE semantics into the existing FP32 RK
// production selection policy; called by both the planner and FP8 lookahead.
[[nodiscard]] DenseRKValueOperation denseRKValueOperation(CubReductionOp operation);

/**
 * Owns the CUDA-aware selection facts for a production GPU (its tuning stream and multiprocessor count). The object
 * is intentionally neither copyable nor movable: context() points at the stream. Keep it alive for every select/plan
 * call. The FP8 cache-miss benchmark uses context() to select hypothetical FP32 physical passes through the
 * production policy without constructing an FP32 stamp.
 */
class DenseProductionPhysicalSelectionContext {
   public:
    DenseProductionPhysicalSelectionContext(DataType original_input_dtype,
                                             DenseRKValueOperation operation,
                                             const Stream& stream,
                                             uint64_t original_reduction_size = 0,
                                             float output_scale = 1.0f,
                                             std::optional<CubReductionOp> original_value_operation = std::nullopt,
                                             uint64_t workspace_budget_bytes = std::numeric_limits<uint64_t>::max(),
                                             CubReductionTuningDepth tuning_depth = CubReductionTuningDepth::Regular,
                                             uint64_t original_input_alignment_bytes = 16);

    DenseProductionPhysicalSelectionContext(const DenseProductionPhysicalSelectionContext&) = delete;
    DenseProductionPhysicalSelectionContext& operator=(const DenseProductionPhysicalSelectionContext&) = delete;
    DenseProductionPhysicalSelectionContext(DenseProductionPhysicalSelectionContext&&) = delete;
    DenseProductionPhysicalSelectionContext& operator=(DenseProductionPhysicalSelectionContext&&) = delete;

    [[nodiscard]] const DensePhysicalSelectionContext& context() const noexcept { return context_; }

   private:
    DensePhysicalSelectionContext context_;
};

}  // namespace ThorImplementation::CubReductionInternal
