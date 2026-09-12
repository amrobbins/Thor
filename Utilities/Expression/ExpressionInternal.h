#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Utilities/Expression/Expression.h"

namespace ThorImplementation {

enum class LogicalDependencyKind : uint8_t;

struct ExpressionInputAnnotations {
    std::optional<DataType> computeDataType;
    std::optional<DataType> outputDataType;

    bool operator==(const ExpressionInputAnnotations&) const = default;
};

// Internal logical-expression inspection boundary.
//
// Already-authored physical graphs enter logical form only through
// Outputs::fromPhysicalOutputs(), which scopes one importer to the whole graph.
// Root-by-root physical import and logical->physical->logical builder shortcuts
// are deliberately not exposed here.
class ExpressionInternalAccess final {
   public:
    [[nodiscard]] static bool hasRoot(const Expression& expression) noexcept;
    [[nodiscard]] static ExprOp rootOp(const Expression& expression);
    [[nodiscard]] static const ExprNode& rootSemantics(const Expression& expression);

    // Narrow immutable backend-semantic rewrite used by the optional cuDNN
    // RMSNorm+Swish fusion. This is intentionally not a generic root-mutation
    // escape hatch: it accepts only an RMSNORM root and preserves every logical
    // dependency/binding handle exactly.
    [[nodiscard]] static Expression withRmsNormFusedActivation(
        const Expression& expression, CudnnRmsNormFusedActivation fused_activation);

    // Narrow read-only dependency access for internal graph inspection and
    // invariant tests. Returning Expressions keeps physical node indices out of
    // the long-term introspection contract.
    [[nodiscard]] static std::optional<Expression> lhsDependency(const Expression& expression);
    [[nodiscard]] static std::optional<Expression> rhsDependency(const Expression& expression);
    [[nodiscard]] static std::optional<Expression> auxDependency(const Expression& expression);
    [[nodiscard]] static std::optional<Expression> alphaDependency(const Expression& expression);
    [[nodiscard]] static std::optional<Expression> betaDependency(const Expression& expression);
    [[nodiscard]] static std::optional<Expression> logicalDependency(const Expression& expression,
                                                                     LogicalDependencyKind kind,
                                                                     uint32_t ordinal = 0);
    [[nodiscard]] static std::optional<Expression> ropeEffectiveSequenceLengthDependency(const Expression& expression);
    [[nodiscard]] static std::optional<Expression> ropePositionIdsDependency(const Expression& expression);
    [[nodiscard]] static LogicalCudaKernelApplicationPtr cudaKernelApplication(const Expression& expression);
    [[nodiscard]] static std::optional<Expression> cudaKernelApplicationInput(const Expression& expression, size_t ordinal);

    // Returns the semantic dtype annotations from every distinct logical input
    // node bound to input_name. The traversal remains entirely in logical IR,
    // visits each logical node at most once, and follows custom-CUDA application
    // inputs as well as ordinary expression dependencies. Callers that require a
    // particular input are responsible for their layer-specific missing/inconsistent
    // annotation diagnostics.
    [[nodiscard]] static std::vector<ExpressionInputAnnotations> inputAnnotations(
        const Expression& expression, const std::string& input_name);
};

}  // namespace ThorImplementation
