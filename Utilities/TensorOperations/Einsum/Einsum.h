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
                  bool uses_strided_batched_gemm,
                  std::shared_ptr<StampedExecutionPlan> execution);

   private:
    EinsumPlan plan;
    Tensor output;
    Stream stream;
    EinsumExecutionPath execution_path = EinsumExecutionPath::GENERIC;
    bool uses_strided_batched_gemm = false;
    std::shared_ptr<StampedExecutionPlan> execution;
};

}  // namespace ThorImplementation
