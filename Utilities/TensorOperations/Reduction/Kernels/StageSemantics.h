#pragma once

// Shared CUDA-free semantic pass descriptor. Host-only FP8 RK planning needs
// the same operation placement rules as CUDA launchers, but must not pull
// Tensor/CUDA headers into its independent translation unit.
#include <cstdint>

namespace ThorImplementation {

/**
 * Semantic I/O role of one physical reduction pass.
 *
 * This role says only whether a pass consumes original input or an intermediate aggregate and whether it produces
 * another aggregate or the requested final output. It does not select reduction topology or constrain pass ordering.
 */
enum class ReductionPassRole : uint8_t {
    Complete = 0,
    First = 1,
    Intermediate = 2,
    Final = 3,
};

enum class ReductionOp : uint8_t;
namespace ReductionInternal {
// Physical execution and dense composition share one semantic pass-role vocabulary.
using ReductionStageRole = ReductionPassRole;

enum class ReductionStageInputTransform : uint8_t {
    Identity = 0,
    AbsoluteValue = 1,
    Square = 2,
};

enum class ReductionStageCombine : uint8_t {
    Sum = 0,
    Product = 1,
    Minimum = 2,
    Maximum = 3,
};

enum class ReductionStageFinalize : uint8_t {
    Identity = 0,
    Divide = 1,
    SquareRoot = 2,
};

/**
 * Mathematical semantics of one physical value-reduction pass.
 *
 * Public ReductionOp intentionally describes the complete operation, while a composed dense reduction must apply
 * the input transform only on its first pass and the output finalizer only on its last pass.  This descriptor separates
 * those concerns without adding more public reduction operations or changing the direct reducer families.
 *
 * finalize_divisor is consulted only for Divide and is the total original reduction-domain size, not the current
 * physical stage's reduction size.
 */
struct ReductionStageSemantics {
    ReductionStageInputTransform input_transform = ReductionStageInputTransform::Identity;
    ReductionStageCombine combine = ReductionStageCombine::Sum;
    ReductionStageFinalize finalize = ReductionStageFinalize::Identity;
    uint64_t finalize_divisor = 1;
};

[[nodiscard]] ReductionStageSemantics makeValueReductionStageSemantics(ReductionOp op,
                                                                           ReductionStageRole role,
                                                                           uint64_t total_reduction_size);

 } // namespace ReductionInternal
} // namespace ThorImplementation
