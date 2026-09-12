#include "Utilities/Expression/ExpressionInternal.h"

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <utility>

#include "Utilities/Expression/LogicalExpression.h"

namespace ThorImplementation {
namespace {

void validateLogicalRoot(const Expression& expression, const char* caller) {
    if (!ExpressionInternalAccess::hasRoot(expression)) {
        throw std::invalid_argument(std::string(caller) + " requires a non-null logical Expression root.");
    }
}

}  // namespace

bool ExpressionInternalAccess::hasRoot(const Expression& expression) noexcept { return static_cast<bool>(expression.node); }

ExprOp ExpressionInternalAccess::rootOp(const Expression& expression) { return rootSemantics(expression).op; }

const ExprNode& ExpressionInternalAccess::rootSemantics(const Expression& expression) {
    validateLogicalRoot(expression, "rootSemantics");
    return expression.node->semantics();
}

Expression ExpressionInternalAccess::withRmsNormFusedActivation(
    const Expression& expression, CudnnRmsNormFusedActivation fused_activation) {
    validateLogicalRoot(expression, "withRmsNormFusedActivation");
    if (expression.node->op() != ExprOp::RMSNORM) {
        throw std::invalid_argument("withRmsNormFusedActivation requires an RMSNORM root.");
    }

    ExprNode semantics = expression.node->semantics();
    semantics.rms_norm_fused_activation = fused_activation;
    return Expression(LogicalExpressionNode::create(std::move(semantics),
                                                    expression.node->dependencies(),
                                                    expression.node->inputBinding(),
                                                    expression.node->raggedRuntimeOffsetsBinding(),
                                                    expression.node->cudaKernelApplication()));
}

std::optional<Expression> ExpressionInternalAccess::lhsDependency(const Expression& expression) {
    validateLogicalRoot(expression, "lhsDependency");
    if (!expression.node->hasDependency(LogicalDependencyKind::Lhs)) {
        return std::nullopt;
    }
    return Expression(expression.node->dependency(LogicalDependencyKind::Lhs));
}

std::optional<Expression> ExpressionInternalAccess::rhsDependency(const Expression& expression) {
    validateLogicalRoot(expression, "rhsDependency");
    if (!expression.node->hasDependency(LogicalDependencyKind::Rhs)) {
        return std::nullopt;
    }
    return Expression(expression.node->dependency(LogicalDependencyKind::Rhs));
}

std::optional<Expression> ExpressionInternalAccess::auxDependency(const Expression& expression) {
    validateLogicalRoot(expression, "auxDependency");
    if (!expression.node->hasDependency(LogicalDependencyKind::Aux)) {
        return std::nullopt;
    }
    return Expression(expression.node->dependency(LogicalDependencyKind::Aux));
}

std::optional<Expression> ExpressionInternalAccess::alphaDependency(const Expression& expression) {
    validateLogicalRoot(expression, "alphaDependency");
    if (!expression.node->hasDependency(LogicalDependencyKind::Alpha)) {
        return std::nullopt;
    }
    return Expression(expression.node->dependency(LogicalDependencyKind::Alpha));
}

std::optional<Expression> ExpressionInternalAccess::betaDependency(const Expression& expression) {
    validateLogicalRoot(expression, "betaDependency");
    if (!expression.node->hasDependency(LogicalDependencyKind::Beta)) {
        return std::nullopt;
    }
    return Expression(expression.node->dependency(LogicalDependencyKind::Beta));
}

std::optional<Expression> ExpressionInternalAccess::logicalDependency(const Expression& expression,
                                                                     LogicalDependencyKind kind,
                                                                     uint32_t ordinal) {
    validateLogicalRoot(expression, "logicalDependency");
    if (!expression.node->hasDependency(kind, ordinal)) {
        return std::nullopt;
    }
    return Expression(expression.node->dependency(kind, ordinal));
}

std::optional<Expression> ExpressionInternalAccess::ropeEffectiveSequenceLengthDependency(const Expression& expression) {
    validateLogicalRoot(expression, "ropeEffectiveSequenceLengthDependency");
    if (!expression.node->hasDependency(LogicalDependencyKind::RopeEffectiveSequenceLength)) {
        return std::nullopt;
    }
    return Expression(expression.node->dependency(LogicalDependencyKind::RopeEffectiveSequenceLength));
}

std::optional<Expression> ExpressionInternalAccess::ropePositionIdsDependency(const Expression& expression) {
    validateLogicalRoot(expression, "ropePositionIdsDependency");
    if (!expression.node->hasDependency(LogicalDependencyKind::RopePositionIds)) {
        return std::nullopt;
    }
    return Expression(expression.node->dependency(LogicalDependencyKind::RopePositionIds));
}

LogicalCudaKernelApplicationPtr ExpressionInternalAccess::cudaKernelApplication(const Expression& expression) {
    validateLogicalRoot(expression, "cudaKernelApplication");
    return expression.node->cudaKernelApplication();
}

std::optional<Expression> ExpressionInternalAccess::cudaKernelApplicationInput(const Expression& expression, size_t ordinal) {
    validateLogicalRoot(expression, "cudaKernelApplicationInput");
    const LogicalCudaKernelApplicationPtr& application = expression.node->cudaKernelApplication();
    if (!application) {
        return std::nullopt;
    }
    if (ordinal >= application->inputs().size()) {
        throw std::out_of_range("cudaKernelApplicationInput ordinal is out of range.");
    }
    return Expression(application->inputs()[ordinal]);
}

std::vector<ExpressionInputAnnotations> ExpressionInternalAccess::inputAnnotations(
    const Expression& expression, const std::string& input_name) {
    validateLogicalRoot(expression, "inputAnnotations");

    std::vector<ExpressionInputAnnotations> annotations;
    std::vector<LogicalExpression> pending{expression.node};
    std::unordered_set<const LogicalExpressionNode*> visited_nodes;
    std::unordered_set<const LogicalCudaKernelApplication*> visited_applications;

    while (!pending.empty()) {
        LogicalExpression node = std::move(pending.back());
        pending.pop_back();
        if (!node || !visited_nodes.insert(node.get()).second) {
            continue;
        }

        if (node->inputBinding() && node->inputBinding()->name == input_name) {
            annotations.push_back(ExpressionInputAnnotations{node->semantics().compute_dtype,
                                                             node->semantics().output_dtype});
        }

        for (const LogicalDependency& dependency : node->dependencies()) {
            pending.push_back(dependency.node);
        }

        const LogicalCudaKernelApplicationPtr& application = node->cudaKernelApplication();
        if (application && visited_applications.insert(application.get()).second) {
            for (const LogicalExpression& input : application->inputs()) {
                pending.push_back(input);
            }
        }
    }

    return annotations;
}

}  // namespace ThorImplementation
