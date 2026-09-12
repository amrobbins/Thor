#pragma once

#if !defined(THOR_GTEST) && !defined(THOR_EXPRESSION_TEST_HOOKS_IMPLEMENTATION)
#error "ExpressionTestHooks.h is internal test support and is not part of Thor's production API."
#endif

#include <string>
#include <unordered_map>
#include <vector>

#include "Utilities/Expression/EquationCompiler.h"

namespace ThorImplementation::detail {

// Read-only observability hooks for expression/compiler architecture tests.
// These delegate directly to the production planner/specializer and must not
// acquire behavior that exists only for tests.
struct EquationCompilerPlanForTests {
    std::vector<PhysicalExecutionStage> stages;
    std::vector<CompiledStageOutput> final_outputs;
};

// Exposes stage producers and named final value bindings from one production
// planning pass. It does not compile, execute, or mutate the plan.
[[nodiscard]] EquationCompilerPlanForTests planEquationCompilerForTests(const PhysicalOutputs& outputs);

// Exercises the production pre-planner GEMM/activation specialization without
// constructing a device-bound FusedEquation. No alternate test rewrite exists.
[[nodiscard]] PhysicalOutputs optimizeGemmPatternsForTests(
    const PhysicalOutputs& outputs,
    const std::unordered_map<std::string, Tensor>& named_inputs);

}  // namespace ThorImplementation::detail
