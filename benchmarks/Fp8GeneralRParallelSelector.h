#pragma once
// Backward-compatible benchmark namespace facade. The retained inventory,
// candidate legality, and frozen ranker live in the production reduction tree.
#include "benchmarks/Fp8OneWindowSelector.h"
#include "benchmarks/Fp8DeepRSmallKSelector.h"
#include "benchmarks/Fp8AlignedKSelector.h"
#include "benchmarks/Fp8RuntimeKSelector.h"
#include "benchmarks/Fp8GeometryRankerModel.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8GeneralRParallelSelector.h"

namespace ThorImplementation::CubReductionBenchmarking {
namespace Fp8GeneralRParallelPolicy = ::ThorImplementation::CubReductionInternal::Fp8RK::Fp8GeneralRParallelPolicy;
} // namespace ThorImplementation::CubReductionBenchmarking
