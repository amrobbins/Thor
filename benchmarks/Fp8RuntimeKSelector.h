#pragma once
// Backward-compatible benchmark namespace facade. The retained inventory,
// candidate legality, and frozen ranker live in the production reduction tree.
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RuntimeKSelector.h"

namespace ThorImplementation::CubReductionBenchmarking {
namespace Fp8RuntimeKPolicy = ::ThorImplementation::CubReductionInternal::Fp8RK::Fp8RuntimeKPolicy;
} // namespace ThorImplementation::CubReductionBenchmarking
