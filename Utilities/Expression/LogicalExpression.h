#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Utilities/Expression/LogicalExpressionFwd.h"
#include "Utilities/Expression/Expression.h"

namespace ThorImplementation {

class CudaKernelExpression;
enum class LogicalDependencyKind : uint8_t {
    Lhs = 0,
    Rhs,
    Aux,
    Alpha,
    Beta,
    MatmulEpilogueAux,
    RopeEffectiveSequenceLength,
    RopePositionIds,
    AttentionSeqLenQ,
    AttentionSeqLenKv,
    AttentionRaggedOffsetQ,
    AttentionRaggedOffsetKv,
    AttentionPageTableK,
    AttentionPageTableV,
    AttentionDropoutSeed,
    AttentionDropoutOffset,
    AttentionDescaleQ,
    AttentionDescaleK,
    AttentionDescaleV,
    AttentionDescaleS,
    AttentionScaleS,
    AttentionScaleO,
    AttentionAmaxS,
    AttentionAmaxO,
    Count,
};

struct LogicalDependency {
    LogicalDependencyKind kind = LogicalDependencyKind::Lhs;
    uint32_t ordinal = 0;
    LogicalExpression node;
};


// Authored identity for one custom-CUDA invocation. Multiple applications may
// share the exact same immutable kernel specification and logical inputs while
// remaining distinct executions; multiple outputs of one application share one
// LogicalCudaKernelApplication handle.
class LogicalCudaKernelApplication final {
   public:
    [[nodiscard]] static LogicalCudaKernelApplicationPtr create(
        std::shared_ptr<const CudaKernelExpression> specification,
        std::vector<LogicalExpression> inputs);

    [[nodiscard]] const std::shared_ptr<const CudaKernelExpression>& specification() const { return specification_; }
    [[nodiscard]] const std::vector<LogicalExpression>& inputs() const { return inputs_; }

    LogicalCudaKernelApplication(const LogicalCudaKernelApplication&) = delete;
    LogicalCudaKernelApplication& operator=(const LogicalCudaKernelApplication&) = delete;
    LogicalCudaKernelApplication(LogicalCudaKernelApplication&&) = delete;
    LogicalCudaKernelApplication& operator=(LogicalCudaKernelApplication&&) = delete;

   private:
    LogicalCudaKernelApplication(std::shared_ptr<const CudaKernelExpression> specification,
                                 std::vector<LogicalExpression> inputs);

    const std::shared_ptr<const CudaKernelExpression> specification_;
    const std::vector<LogicalExpression> inputs_;
};

struct LogicalInputBinding {
    std::string name;
    NamedInput::Kind kind = NamedInput::Kind::Tensor;

    bool operator==(const LogicalInputBinding&) const = default;
};

class LogicalExpressionNode final {
   public:
    [[nodiscard]] static LogicalExpression create(
        ExprNode semantics,
        std::vector<LogicalDependency> dependencies = {},
        std::optional<LogicalInputBinding> input_binding = std::nullopt,
        std::optional<LogicalInputBinding> ragged_runtime_offsets_binding = std::nullopt,
        LogicalCudaKernelApplicationPtr cuda_kernel_application = nullptr);

    [[nodiscard]] ExprOp op() const { return semantics_.op; }
    [[nodiscard]] const ExprNode& semantics() const { return semantics_; }
    [[nodiscard]] const std::vector<LogicalDependency>& dependencies() const { return dependencies_; }
    [[nodiscard]] const std::optional<LogicalInputBinding>& inputBinding() const { return input_binding_; }
    [[nodiscard]] const std::optional<LogicalInputBinding>& raggedRuntimeOffsetsBinding() const {
        return ragged_runtime_offsets_binding_;
    }
    [[nodiscard]] const LogicalCudaKernelApplicationPtr& cudaKernelApplication() const {
        return cuda_kernel_application_;
    }

    [[nodiscard]] const LogicalExpression& dependency(LogicalDependencyKind kind, uint32_t ordinal = 0) const;
    [[nodiscard]] bool hasDependency(LogicalDependencyKind kind, uint32_t ordinal = 0) const;

    LogicalExpressionNode(const LogicalExpressionNode&) = delete;
    LogicalExpressionNode& operator=(const LogicalExpressionNode&) = delete;
    LogicalExpressionNode(LogicalExpressionNode&&) = delete;
    LogicalExpressionNode& operator=(LogicalExpressionNode&&) = delete;

   private:
    LogicalExpressionNode(ExprNode semantics,
                          std::vector<LogicalDependency> dependencies,
                          std::optional<LogicalInputBinding> input_binding,
                          std::optional<LogicalInputBinding> ragged_runtime_offsets_binding,
                          LogicalCudaKernelApplicationPtr cuda_kernel_application);

    const ExprNode semantics_;
    const std::vector<LogicalDependency> dependencies_;
    const std::optional<LogicalInputBinding> input_binding_;
    const std::optional<LogicalInputBinding> ragged_runtime_offsets_binding_;
    const LogicalCudaKernelApplicationPtr cuda_kernel_application_;
};

[[nodiscard]] LogicalExpression makeLogicalInput(
    std::string name,
    NamedInput::Kind kind = NamedInput::Kind::Tensor,
    std::optional<DataType> input_tensor_dtype = std::nullopt,
    std::optional<DataType> compute_dtype = std::nullopt,
    std::optional<DataType> output_dtype = std::nullopt);
[[nodiscard]] LogicalExpression makeLogicalScalar(double value,
                                                  std::optional<DataType> compute_dtype = std::nullopt,
                                                  std::optional<DataType> output_dtype = std::nullopt);
[[nodiscard]] LogicalExpression makeLogicalUnary(ExprOp op, const LogicalExpression& input, ExprNode semantics = ExprNode{});
[[nodiscard]] LogicalExpression makeLogicalBinary(ExprOp op,
                                                   const LogicalExpression& lhs,
                                                   const LogicalExpression& rhs,
                                                   ExprNode semantics = ExprNode{});
[[nodiscard]] LogicalExpression makeLogicalTernary(ExprOp op,
                                                    const LogicalExpression& lhs,
                                                    const LogicalExpression& rhs,
                                                    const LogicalExpression& aux,
                                                    ExprNode semantics = ExprNode{});



class LogicalExpressionLowerer final {
   public:
    [[nodiscard]] static PhysicalOutputs lower(const std::vector<LogicalNamedOutput>& outputs);
};


class LogicalExpressionImporter final {
   public:
    explicit LogicalExpressionImporter(const PhysicalExpression& source);

    // Graph-scoped import is the only public import operation. All roots from one
    // PhysicalExpression must pass through this importer context together so
    // physical sharing is preserved as logical identity.
    [[nodiscard]] std::vector<LogicalNamedOutput> importOutputs(const std::vector<NamedOutput>& outputs);

   private:
    // Recursive implementation detail only. Exposing node-index import publicly
    // makes it too easy for callers to create one importer per root and thereby
    // destroy shared ancestry at the physical->logical boundary.
    [[nodiscard]] LogicalExpression importNode(uint32_t node_index);

    const PhysicalExpression& source_;
    std::vector<LogicalExpression> memo_;
    std::vector<uint8_t> visit_state_;
    std::vector<LogicalCudaKernelApplicationPtr> cuda_applications_;
    std::vector<std::vector<uint32_t>> cuda_application_input_nodes_;
};

void validateLogicalGraph(const std::vector<LogicalNamedOutput>& outputs);

[[nodiscard]] const char* logicalDependencyKindName(LogicalDependencyKind kind);

}  // namespace ThorImplementation
