#pragma once

#if !defined(THOR_GTEST) && !defined(THOR_EXPRESSION_TEST_HOOKS_IMPLEMENTATION)
#error "ExpressionTestHooks.h is internal test support and is not part of Thor's production API."
#endif

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/LogicalWork.h"

namespace ThorImplementation::detail {

// Read-only observability hooks for expression/compiler architecture tests.
// These delegate directly to the production planner/specializer and must not
// acquire behavior that exists only for tests.
struct EquationCompilerPlanForTests {
    std::vector<PhysicalExecutionStage> stages;
    std::vector<CompiledStageOutput> final_outputs;
};

struct FusedStageFlopCountsForTests {
    uint64_t physical_flops = 0;
    uint64_t logical_flops = 0;
};

// Exposes stage producers and named final value bindings from one production
// planning pass. It does not compile, execute, or mutate the plan.
[[nodiscard]] EquationCompilerPlanForTests planEquationCompilerForTests(const PhysicalOutputs& outputs);

// Computes both the post-CSE physical fused-stage FLOP estimate and the
// pre-CSE authored logical FLOP estimate using production accounting
// formulas. This is observability only; it does not compile or execute kernels.
[[nodiscard]] FusedStageFlopCountsForTests fusedStageFlopCountsForTests(
    const PhysicalExecutionStage& stage,
    const std::vector<std::vector<uint64_t>>& stage_input_dims);

// Computes authored logical FLOPs and logical tensor bytes for one fused stage.
// Byte accounting is fixed-shape in LWA-3A; runtime-active ragged byte
// specialization is added later without changing this semantic definition.
[[nodiscard]] LogicalWorkCount fusedStageLogicalWorkForTests(
    const PhysicalExecutionStage& stage,
    const std::vector<std::vector<uint64_t>>& stage_input_dims);

// Computes the fixed-shape logical FLOPs/bytes sidecar for one compiled
// dedicated stage using the same production accounting formulas used while
// stamping. Runtime-partition stages intentionally return zero from this static
// helper; their exact active logical work is supplied by stamped runtime sidecars.
[[nodiscard]] LogicalWorkCount dedicatedStageLogicalWorkForTests(
    const CompiledExecutionStage& stage,
    const std::vector<std::vector<uint64_t>>& stage_input_dims,
    const std::vector<std::optional<DataType>>& stage_input_dtypes);

// Exercises the production pre-planner GEMM/activation specialization without
// constructing a device-bound FusedEquation. No alternate test rewrite exists.
[[nodiscard]] PhysicalOutputs optimizeGemmPatternsForTests(
    const PhysicalOutputs& outputs,
    const std::unordered_map<std::string, Tensor>& named_inputs);

}  // namespace ThorImplementation::detail
