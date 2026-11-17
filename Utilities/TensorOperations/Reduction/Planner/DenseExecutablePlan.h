#pragma once

// A stamp's dense plan as bound passes (DenseExecutablePass: physical plan, buffers and scratch) and their
// read-only diagnostics.

#include "DeepLearning/Implementation/Tensor/StridedSpan.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TensorOperations/Reduction/Kernels/StageSemantics.h"
#include "Utilities/TensorOperations/Reduction/Layout/ReductionGeometry.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionFamilies.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation {

namespace ReductionInternal {
struct Fp8ArgPhysicalPlan;
struct Fp32ArgPhysicalPlan;

/**
 * One bound ordinary-dense physical pass.
 *
 * The family-local physical plan stays type-erased at the execution-plan level. role is explicit so semantic placement
 * never depends on reducer-owned continuation state. The pass owns both any persistent intermediate output and any
 * direct-backend temporary storage queried at stamp time, so execution needs no reducer-owned continuation state.
 */
struct DenseExecutablePass {
    std::shared_ptr<const DenseReductionPhysicalPlan> physical_plan;
    DensePassRole role = DensePassRole::Complete;
    StridedSpan input;
    mutable Tensor output;
    ReductionGeometry geometry;

    // Persistent storage owned by this physical pass. intermediate_bytes accounts for a non-final aggregate output;
    // temp_storage_bytes/temp_storage are direct-backend scratch queried at stamp time. Modern RK kernels currently
    // need no separate launch scratch. Keeping both here makes the flat pass representation sufficient for R/KR/RK.
    size_t intermediate_bytes = 0;
    size_t temp_storage_bytes = 0;
    Tensor temp_storage;
    // output and temp_storage as the spans the launch takes, made once at stamp.
    StridedSpan output_span;
    StridedSpan temp_storage_span;

    [[nodiscard]] size_t workspaceBytes() const {
        if (intermediate_bytes > std::numeric_limits<size_t>::max() - temp_storage_bytes) {
            throw std::overflow_error("Dense executable pass workspace size overflows size_t.");
        }
        return intermediate_bytes + temp_storage_bytes;
    }
};

/** Flat ordered physical-pass representation for ordinary dense execution. */
struct DenseExecutablePlan {
    std::vector<DenseExecutablePass> passes;
    size_t workspace_size_bytes = 0;
};

}  // namespace ReductionInternal

/** Read-only view of one already stamped dense pass (no GPU allocation or selection).
 * Fields that do not apply to the selected physical family remain nullopt; there is no invented packet, CTA, or shard
 * geometry for FP8.
 */
struct DensePhysicalPassDiagnostic {
    std::string family;
    ReductionInternal::DensePassRole role;
    DataType input_dtype;
    DataType output_dtype;
    std::vector<uint64_t> input_dimensions;
    std::vector<uint64_t> output_dimensions;
    size_t intermediate_bytes = 0;
    size_t scratch_bytes = 0;
    std::optional<size_t> packet_bytes;
    std::optional<uint32_t> block_threads;
    std::optional<uint64_t> stage_blocks;
    std::optional<uint64_t> shards_per_output;
};

}  // namespace ThorImplementation
