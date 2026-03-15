#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace ThorImplementation {
enum class ExprOp : uint16_t {
    INPUT = 3,
    SCALAR_FP,
    ADD,
    SUB,
    MUL,
    DIV,
    POW,
    NEG,
    EXP,
    EXP2,
    EXP10,
    LN,
    LOG2,
    LOG10,
    SQRT,
    MIN,
    MAX,
    REDUCE_SUM,
    REDUCE_PROD,
    REDUCE_MIN,
    REDUCE_MAX,
    REDUCE_AMAX,
    REDUCE_AVG,
    REDUCE_NORM1,
    REDUCE_NORM2,
};

inline bool isCudnnReduceOp(ExprOp op) {
    return op == ExprOp::REDUCE_SUM || op == ExprOp::REDUCE_PROD || op == ExprOp::REDUCE_MIN || op == ExprOp::REDUCE_MAX ||
           op == ExprOp::REDUCE_AMAX || op == ExprOp::REDUCE_AVG || op == ExprOp::REDUCE_NORM1 || op == ExprOp::REDUCE_NORM2;
}

inline cudnnReduceTensorOp_t toCudnnReduceOp(ExprOp op) {
    switch (op) {
        case ExprOp::REDUCE_SUM:
            return CUDNN_REDUCE_TENSOR_ADD;
        case ExprOp::REDUCE_PROD:
            return CUDNN_REDUCE_TENSOR_MUL;
        case ExprOp::REDUCE_MIN:
            return CUDNN_REDUCE_TENSOR_MIN;
        case ExprOp::REDUCE_MAX:
            return CUDNN_REDUCE_TENSOR_MAX;
        case ExprOp::REDUCE_AMAX:
            return CUDNN_REDUCE_TENSOR_AMAX;
        case ExprOp::REDUCE_AVG:
            return CUDNN_REDUCE_TENSOR_AVG;
        case ExprOp::REDUCE_NORM1:
            return CUDNN_REDUCE_TENSOR_NORM1;
        case ExprOp::REDUCE_NORM2:
            return CUDNN_REDUCE_TENSOR_NORM2;
        default:
            throw;
    }
}

struct ExprNode {
    ExprOp op;
    uint32_t lhs = UINT32_MAX;
    uint32_t rhs = UINT32_MAX;  // unused for unary/scalar ops
    uint32_t input_slot = UINT32_MAX;
    double scalar_fp = 0.0;

    // for reduction nodes only
    std::vector<uint64_t> reduction_axes;
    bool keepdim = true;
    TensorDescriptor::DataType compute_dtype;
};

struct NamedInput {
    std::string name;
    uint32_t slot;
};

struct PhysicalExpression {
    std::vector<ExprNode> nodes;
    std::vector<NamedInput> inputs;
    uint32_t output_node = UINT32_MAX;

    uint32_t numInputs() const { return static_cast<uint32_t>(inputs.size()); }

    uint32_t getOrCreateInputSlot(const std::string& name);
};

class Expression {
   public:
    Expression(double value);

    static Expression input(const std::string& name);
    static Expression scalar(double value);

    [[nodiscard]] PhysicalExpression expression() const;

    Expression operator+(const Expression& other) const;
    Expression operator-(const Expression& other) const;
    Expression operator*(const Expression& other) const;
    Expression operator/(const Expression& other) const;

    Expression operator-() const;

    [[nodiscard]] Expression ln() const;
    [[nodiscard]] Expression log2() const;
    [[nodiscard]] Expression log10() const;
    [[nodiscard]] Expression log(double base) const;
    [[nodiscard]] Expression exp() const;
    [[nodiscard]] Expression exp2() const;
    [[nodiscard]] Expression exp10() const;
    [[nodiscard]] Expression sqrt() const;
    [[nodiscard]] Expression pow(const Expression& exponent) const;

    Expression min(const Expression& other) const;
    Expression max(const Expression& other) const;

   private:
    std::shared_ptr<PhysicalExpression> expr;
    uint32_t nodeIndex = UINT32_MAX;

    Expression(std::shared_ptr<PhysicalExpression> expr, uint32_t nodeIndex) : expr(std::move(expr)), nodeIndex(nodeIndex) {}

    static Expression binaryOp(const Expression& lhsExpr, const Expression& rhsExpr, ExprOp op);
    static Expression unaryOp(const Expression& inputExpr, ExprOp op);
};

std::string formatFloatCanonical(double x);
bool isCommutative(ExprOp op);
std::string opName(ExprOp op);
std::string canonicalizeNode(const PhysicalExpression& expr, uint32_t nodeIndex, std::unordered_map<uint32_t, std::string>& memo);
std::string canonicalize(const PhysicalExpression& expr);

}  // namespace ThorImplementation
