#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Einsum/EinsumPlanner.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ThorImplementation {

class StampedExecutionPlan;
class StampedMatmul;
class CublasKernel;

// Concrete execution path chosen after an einsum has been parsed/planned and
// stamped against physical tensors.
enum class EinsumExecutionPath {
    GENERIC,
    GEMM,
    BATCHED_GEMM,
};

class StampedEinsum;

/**
 * Describes a GPU einsum operation.
 *
 * The textual equation is parsed and planned at stamp time because concrete
 * tensor dimensions are required to resolve broadcasting and ellipses.
 * Standalone reductions in the generic path are always delegated to the
 * centralized CubReduction utility.  Matrix K reductions are intrinsic to
 * GEMM and therefore are not standalone reduction stages.
 */
class Einsum {
   public:
    explicit Einsum(std::string equation);

    [[nodiscard]] const std::string& getEquation() const { return equation; }

    [[nodiscard]] std::shared_ptr<StampedEinsum> stamp(const std::vector<Tensor>& inputs, const Stream& stream) const;
    [[nodiscard]] std::shared_ptr<StampedEinsum> stamp(const std::vector<Tensor>& inputs,
                                                       const Tensor& preallocated_output,
                                                       const Stream& stream) const;

   private:
    std::string equation;
};

/**
 * Fully planned/stamped einsum execution.
 *
 * run()/runOn() allocate nothing and perform no planner or kernel-selection
 * work.  Generic standalone sums are represented by cub_reduction and run
 * only through StampedCubReduction.
 */
class StampedEinsum {
   public:
    void run();
    void runOn(Stream& run_stream) const;

    [[nodiscard]] Tensor getOutputTensor() const { return output; }
    [[nodiscard]] const EinsumPlan& getPlan() const { return plan; }
    [[nodiscard]] EinsumExecutionPath getExecutionPath() const { return execution_path; }
    [[nodiscard]] bool usesStandaloneReduction() const { return cub_reduction != nullptr; }
    [[nodiscard]] bool usesStridedBatchedGemm() const { return batched_matrix_kernel != nullptr; }
    [[nodiscard]] std::optional<CubReductionPath> getStandaloneReductionPath() const {
        if (!cub_reduction) {
            return std::nullopt;
        }
        return cub_reduction->getPath();
    }

    // Public for consistency with Thor's other stamped execution objects; callers
    // normally obtain this through Einsum::stamp().
    StampedEinsum(EinsumPlan plan,
                  std::vector<Tensor> inputs,
                  Tensor output,
                  const Stream& stream,
                  EinsumExecutionPath execution_path,
                  std::shared_ptr<StampedExecutionPlan> generic_preparation,
                  std::shared_ptr<StampedCubReduction> cub_reduction,
                  std::vector<std::shared_ptr<StampedMatmul>> matrix_batches,
                  std::shared_ptr<CublasKernel> batched_matrix_kernel,
                  std::optional<Tensor> matrix_workspace);

   private:
    EinsumPlan plan;
    std::vector<Tensor> inputs;
    Tensor output;
    Stream stream;
    EinsumExecutionPath execution_path = EinsumExecutionPath::GENERIC;
    std::shared_ptr<StampedExecutionPlan> generic_preparation;
    std::shared_ptr<StampedCubReduction> cub_reduction;
    std::vector<std::shared_ptr<StampedMatmul>> matrix_batches;
    std::shared_ptr<CublasKernel> batched_matrix_kernel;
    std::optional<Tensor> matrix_workspace;
};

}  // namespace ThorImplementation
