#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/ExecutionDiagnostics.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Einsum/EinsumPlanner.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ThorImplementation {

class StampedExecutionPlan;

// Algebraic lowering selected by einsum. Physical execution is always owned by
// Expression/StampedExecutionPlan.
enum class EinsumExecutionPath {
    GENERIC,
    GEMM,
    BATCHED_GEMM,
    PAIR_PRODUCT,
    // A selected exact/bridge multi-operand contraction tree lowered entirely
    // into the existing Expression primitives. Individual internal nodes may
    // be GEMM, batched GEMM, pair product, and/or CUB pre-reductions.
    EXACT_CONTRACTION,

    // A selected bounded-beam contraction tree for 7+ operands, lowered through
    // the same pair execution machinery as exact/bridge contraction trees.
    BEAM_CONTRACTION,
};

// Why a stamped operation is executing through the whole-equation generic
// expression. NONE means an optimized execution path was selected. The
// distinctions are intentionally observable so production fallback can be
// audited without conflating benign unary execution with a failed optimized
// multi-operand lowering.
enum class EinsumGenericExecutionReason {
    NONE,

    // Explicit correctness/reference surface requested by stampGenericReference().
    EXPLICIT_REFERENCE,

    // Unary permutation/reduction uses the generic Expression formulation as
    // its native implementation; there is no multi-operand broadcast product.
    UNARY_DIRECT,

    // The binary planner selected GENERAL because no GEMM/batched-GEMM or
    // pair-product realization was preferred/available.
    BINARY_GENERAL_PLAN,

    // The binary planner selected an optimized realization, but runtime
    // Expression lowering could not reconstruct it.
    BINARY_OPTIMIZED_LOWERING_UNAVAILABLE,

    // Exact/bridge/beam planning did not produce a representable contraction
    // tree. This can occur only through the planner's overflow-safe fallback.
    MULTI_OPERAND_PLAN_UNAVAILABLE,

    // A contraction tree was selected, but one of its speculative physical
    // candidates could not be reconstructed by the execution backend.
    MULTI_OPERAND_TREE_LOWERING_UNAVAILABLE,
};

class StampedEinsum;

/**
 * Describes a GPU einsum operation.
 *
 * The textual equation is parsed and planned at stamp time because concrete
 * tensor dimensions are required to resolve broadcasting and ellipses. The
 * resulting operation is lowered completely into an Expression graph. Matrix
 * contractions use Expression::matmul; all independent sums use
 * Expression::reduce_sum and therefore the centralized CubReduction backend.
 */
class Einsum {
   public:
    explicit Einsum(std::string equation);

    [[nodiscard]] const std::string& getEquation() const { return equation; }

    [[nodiscard]] std::shared_ptr<StampedEinsum> stamp(const std::vector<Tensor>& inputs, const Stream& stream) const;
    [[nodiscard]] std::shared_ptr<StampedEinsum> stamp(const std::vector<Tensor>& inputs,
                                                       const Tensor& preallocated_output,
                                                       const Stream& stream) const;

    // Implementation-layer surface for equations that have already been
    // normalized against concrete physical tensor shapes.  This is used by
    // DeepLearning layers to prepend Thor's implicit runtime batch axis without
    // consuming a user-visible ASCII label or changing ellipsis placement.
    // The supplied equation is validated again against inputs before stamping.
    [[nodiscard]] static std::shared_ptr<StampedEinsum> stampResolvedEquation(
        const ResolvedEinsumEquation& resolved_equation,
        const std::vector<Tensor>& inputs,
        const Tensor& preallocated_output,
        const Stream& stream);

    // Diagnostic/reference surface used to compare optimized lowering against
    // the original whole-equation broadcast-product + reduction implementation.
    // This deliberately bypasses pair and contraction-tree lowering while preserving
    // parsing, dtype, accumulation, and output semantics. It is not intended as
    // a production execution policy.
    [[nodiscard]] std::shared_ptr<StampedEinsum> stampGenericReference(
        const std::vector<Tensor>& inputs, const Stream& stream) const;

   private:
    std::string equation;
};

/**
 * Fully stamped einsum execution.
 *
 * Einsum owns no CUDA kernels, helper streams, reductions, or GEMM launch
 * machinery. run()/runOn() delegate directly to the stamped Expression DAG.
 */
class StampedEinsum {
   public:
    void run();
    void runOn(Stream& run_stream) const;

    [[nodiscard]] Tensor getOutputTensor() const { return output; }
    [[nodiscard]] const EinsumPlan& getPlan() const { return plan; }
    [[nodiscard]] EinsumExecutionPath getExecutionPath() const { return execution_path; }
    [[nodiscard]] EinsumGenericExecutionReason getGenericExecutionReason() const {
        return generic_execution_reason;
    }
    [[nodiscard]] bool isWholeEquationGenericFallback() const {
        return execution_path == EinsumExecutionPath::GENERIC &&
               generic_execution_reason != EinsumGenericExecutionReason::EXPLICIT_REFERENCE &&
               generic_execution_reason != EinsumGenericExecutionReason::UNARY_DIRECT;
    }
    [[nodiscard]] bool usesStandaloneReduction() const;
    [[nodiscard]] bool usesStridedBatchedGemm() const { return uses_strided_batched_gemm; }
    [[nodiscard]] std::vector<CubReductionPath> getStandaloneReductionPaths() const;
    [[nodiscard]] std::optional<CubReductionPath> getStandaloneReductionPath() const;
    [[nodiscard]] std::vector<std::string> getExpressionStageKindNames() const;
    [[nodiscard]] std::vector<StampedMatmulStageDiagnostic> getExpressionMatmulStageDiagnostics() const;

    // Public for consistency with Thor's other stamped execution objects; callers
    // normally obtain this through Einsum::stamp().
    StampedEinsum(EinsumPlan plan,
                  Tensor output,
                  const Stream& stream,
                  EinsumExecutionPath execution_path,
                  EinsumGenericExecutionReason generic_execution_reason,
                  bool uses_strided_batched_gemm,
                  std::shared_ptr<StampedExecutionPlan> execution);

   private:
    EinsumPlan plan;
    Tensor output;
    Stream stream;
    EinsumExecutionPath execution_path = EinsumExecutionPath::GENERIC;
    EinsumGenericExecutionReason generic_execution_reason = EinsumGenericExecutionReason::NONE;
    bool uses_strided_batched_gemm = false;
    std::shared_ptr<StampedExecutionPlan> execution;
};

}  // namespace ThorImplementation
