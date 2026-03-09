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
enum class ExprOp : uint16_t { INPUT = 3, SCALAR_F32, ADD, SUB, MUL, DIV, POW, NEG, EXP, EXP2, EXP10, LOG, LOG2, LOG10, SQRT };

struct ExprNode {
    ExprOp op;
    uint32_t lhs = UINT32_MAX;
    uint32_t rhs = UINT32_MAX;  // unused for unary/scalar ops
    uint32_t input_index = UINT32_MAX;
    float scalar_f32 = 0.0f;
};

struct PhysicalExpression {
    std::vector<ExprNode> nodes;
    uint32_t output_node;
    uint32_t num_inputs;
};

class Expression {
   public:
    static Expression input(uint32_t inputIndex);
    static Expression scalar(float value);

    [[nodiscard]] PhysicalExpression expression() const;

    Expression operator+(const Expression& other) const;
    Expression operator-(const Expression& other) const;
    Expression operator*(const Expression& other) const;
    Expression operator/(const Expression& other) const;

    Expression operator+(float scalar) const;
    Expression operator-(float scalar) const;
    Expression operator*(float scalar) const;
    Expression operator/(float scalar) const;

    Expression operator-() const;

    [[nodiscard]] Expression log() const;  // natural log
    [[nodiscard]] Expression log2() const;
    [[nodiscard]] Expression log10() const;
    [[nodiscard]] Expression log(float base) const;
    [[nodiscard]] Expression exp() const;
    [[nodiscard]] Expression exp2() const;
    [[nodiscard]] Expression exp10() const;
    [[nodiscard]] Expression sqrt() const;
    [[nodiscard]] Expression pow(float exponent) const;
    [[nodiscard]] Expression pow(const Expression& exponent) const;

   private:
    std::shared_ptr<PhysicalExpression> expr;
    uint32_t nodeIndex = UINT32_MAX;

    Expression(std::shared_ptr<PhysicalExpression> expr, uint32_t nodeIndex) : expr(std::move(expr)), nodeIndex(nodeIndex) {}

    static Expression binaryOp(const Expression& lhsExpr, const Expression& rhsExpr, ExprOp op);
    static Expression unaryOp(const Expression& inputExpr, ExprOp op);
    static Expression powScalarOp(const Expression& baseExpr, float exponent);
};

inline Expression operator+(float lhs, const Expression& rhs) { return Expression::scalar(lhs) + rhs; }
inline Expression operator-(float lhs, const Expression& rhs) { return Expression::scalar(lhs) - rhs; }
inline Expression operator*(float lhs, const Expression& rhs) { return Expression::scalar(lhs) * rhs; }
inline Expression operator/(float lhs, const Expression& rhs) { return Expression::scalar(lhs) / rhs; }
inline Expression pow(float base, const Expression& exponent) { return Expression::scalar(base).pow(exponent); }

std::string formatFloatCanonical(float x);
bool isCommutative(ExprOp op);
std::string opName(ExprOp op);
std::string canonicalizeNode(const PhysicalExpression& expr, uint32_t nodeIndex, std::unordered_map<uint32_t, std::string>& memo);
std::string canonicalize(const PhysicalExpression& expr);

}  // namespace ThorImplementation
