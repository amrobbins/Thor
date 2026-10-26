#pragma once

// Shared CUDA-free semantic pass descriptor. Host-only FP8 RK planning needs
// the same operation placement rules as CUDA launchers, but must not pull
// Tensor/CUDA headers into its independent translation unit.
#include "Utilities/TensorOperations/Cub/CubReductionPassRole.h"
#include <cstdint>

namespace ThorImplementation {
enum class CubReductionOp : uint8_t;
namespace CubReductionInternal {
// Physical execution and dense composition share one semantic pass-role vocabulary.
using CubReductionStageRole = CubReductionPassRole;

enum class CubReductionStageInputTransform : uint8_t {
    Identity = 0,
    AbsoluteValue = 1,
    Square = 2,
};

enum class CubReductionStageCombine : uint8_t {
    Sum = 0,
    Product = 1,
    Minimum = 2,
    Maximum = 3,
};

enum class CubReductionStageFinalize : uint8_t {
    Identity = 0,
    Divide = 1,
    SquareRoot = 2,
};

/**
 * Mathematical semantics of one physical value-reduction pass.
 *
 * Public CubReductionOp intentionally describes the complete operation, while a composed dense reduction must apply
 * the input transform only on its first pass and the output finalizer only on its last pass.  This descriptor separates
 * those concerns without adding more public reduction operations or changing the direct reducer families.
 *
 * finalize_divisor is consulted only for Divide and is the total original reduction-domain size, not the current
 * physical stage's reduction size.
 */
struct CubReductionStageSemantics {
    CubReductionStageInputTransform input_transform = CubReductionStageInputTransform::Identity;
    CubReductionStageCombine combine = CubReductionStageCombine::Sum;
    CubReductionStageFinalize finalize = CubReductionStageFinalize::Identity;
    uint64_t finalize_divisor = 1;
};

[[nodiscard]] CubReductionStageSemantics makeValueReductionStageSemantics(CubReductionOp op,
                                                                           CubReductionStageRole role,
                                                                           uint64_t total_reduction_size);

 } // namespace CubReductionInternal
} // namespace ThorImplementation
