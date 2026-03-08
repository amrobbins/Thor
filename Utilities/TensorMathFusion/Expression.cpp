#include "Utilities/TensorMathFusion/Expression.h"

namespace ThorImplementation {

std::string formatFloatCanonical(float x) {
    std::ostringstream ss;
    ss << std::setprecision(9) << x;
    return ss.str();
}

bool isCommutative(ExprOp op) { return op == ExprOp::ADD || op == ExprOp::MUL; }

std::string opName(ExprOp op) {
    switch (op) {
        case ExprOp::INPUT:
            return "IN";
        case ExprOp::SCALAR_F32:
            return "F32";
        case ExprOp::ADD:
            return "ADD";
        case ExprOp::SUB:
            return "SUB";
        case ExprOp::MUL:
            return "MUL";
        case ExprOp::DIV:
            return "DIV";
        case ExprOp::POW_SCALAR:
            return "POW_SCALAR";
        case ExprOp::NEG:
            return "NEG";
        case ExprOp::EXP:
            return "EXP";
        case ExprOp::LOG:
            return "LOG";
        case ExprOp::SQRT:
            return "SQRT";
        default:
            throw std::runtime_error("Unknown ExprOp");
    }
}

std::string canonicalizeNode(const PhysicalExpression& expr, uint32_t nodeIndex, std::unordered_map<uint32_t, std::string>& memo) {
    auto it = memo.find(nodeIndex);
    if (it != memo.end())
        return it->second;

    const ExprNode& n = expr.nodes[nodeIndex];
    std::string out;

    switch (n.op) {
        case ExprOp::INPUT:
            out = "IN" + std::to_string(n.input_index);
            break;

        case ExprOp::SCALAR_F32:
            out = "F32(" + formatFloatCanonical(n.scalar_f32) + ")";
            break;

        case ExprOp::NEG:
        case ExprOp::EXP:
        case ExprOp::LOG:
        case ExprOp::SQRT:
            out = opName(n.op) + "(" + canonicalizeNode(expr, n.lhs, memo) + ")";
            break;

        case ExprOp::POW_SCALAR:
            out = "POW_SCALAR(" + canonicalizeNode(expr, n.lhs, memo) + "," + formatFloatCanonical(n.scalar_f32) + ")";
            break;

        case ExprOp::ADD:
        case ExprOp::MUL: {
            std::string a = canonicalizeNode(expr, n.lhs, memo);
            std::string b = canonicalizeNode(expr, n.rhs, memo);
            if (a > b)
                std::swap(a, b);
            out = opName(n.op) + "(" + a + "," + b + ")";
            break;
        }

        case ExprOp::SUB:
        case ExprOp::DIV:
            out = opName(n.op) + "(" + canonicalizeNode(expr, n.lhs, memo) + "," + canonicalizeNode(expr, n.rhs, memo) + ")";
            break;

        default:
            throw std::runtime_error("Unsupported ExprOp in canonicalizeNode");
    }

    memo[nodeIndex] = out;
    return out;
}

std::string canonicalize(const PhysicalExpression& expr) {
    std::unordered_map<uint32_t, std::string> memo;
    return canonicalizeNode(expr, expr.output_node, memo);
}

namespace {
uint32_t cloneSubtree(const PhysicalExpression& src,
                      uint32_t srcNodeIndex,
                      PhysicalExpression& dst,
                      std::unordered_map<uint32_t, uint32_t>& oldToNew) {
    auto it = oldToNew.find(srcNodeIndex);
    if (it != oldToNew.end())
        return it->second;

    const ExprNode& srcNode = src.nodes[srcNodeIndex];
    ExprNode newNode = srcNode;

    if (srcNode.op != ExprOp::INPUT && srcNode.op != ExprOp::SCALAR_F32) {
        if (srcNode.lhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing lhs");
        newNode.lhs = cloneSubtree(src, srcNode.lhs, dst, oldToNew);

        if (srcNode.rhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing rhs");
        newNode.rhs = cloneSubtree(src, srcNode.rhs, dst, oldToNew);
    }

    uint32_t newIndex = static_cast<uint32_t>(dst.nodes.size());
    dst.nodes.push_back(newNode);
    oldToNew[srcNodeIndex] = newIndex;
    return newIndex;
}
}  // namespace

Expression Expression::input(uint32_t inputIndex) {
    auto out = std::make_shared<PhysicalExpression>();

    ExprNode node;
    node.op = ExprOp::INPUT;
    node.input_index = inputIndex;

    out->nodes.push_back(node);
    out->output_node = 0;
    out->num_inputs = inputIndex + 1;

    return Expression(out, 0);
}

Expression Expression::scalar(float value) {
    auto out = std::make_shared<PhysicalExpression>();

    ExprNode node;
    node.op = ExprOp::SCALAR_F32;
    node.scalar_f32 = value;

    out->nodes.push_back(node);
    out->output_node = 0;
    out->num_inputs = 0;

    return Expression(out, 0);
}

PhysicalExpression Expression::expression() const {
    if (!expr)
        throw std::runtime_error("Expr has no underlying expression");

    PhysicalExpression out = *expr;
    out.output_node = nodeIndex;
    return out;
}

Expression Expression::binaryOp(const Expression& lhsExpr, const Expression& rhsExpr, ExprOp op) {
    if (!lhsExpr.expr || !rhsExpr.expr)
        throw std::runtime_error("Cannot combine empty expressions");

    auto out = std::make_shared<PhysicalExpression>();
    out->num_inputs = std::max(lhsExpr.expr->num_inputs, rhsExpr.expr->num_inputs);

    std::unordered_map<uint32_t, uint32_t> lhsMap;
    std::unordered_map<uint32_t, uint32_t> rhsMap;

    uint32_t newLhsIndex = cloneSubtree(*lhsExpr.expr, lhsExpr.nodeIndex, *out, lhsMap);
    uint32_t newRhsIndex = cloneSubtree(*rhsExpr.expr, rhsExpr.nodeIndex, *out, rhsMap);

    ExprNode node;
    node.op = op;
    node.lhs = newLhsIndex;
    node.rhs = newRhsIndex;

    uint32_t newIndex = static_cast<uint32_t>(out->nodes.size());
    out->nodes.push_back(node);
    out->output_node = newIndex;

    return Expression(out, newIndex);
}

Expression Expression::operator+(const Expression& other) const { return binaryOp(*this, other, ExprOp::ADD); }

Expression Expression::operator-(const Expression& other) const { return binaryOp(*this, other, ExprOp::SUB); }

Expression Expression::operator*(const Expression& other) const { return binaryOp(*this, other, ExprOp::MUL); }

Expression Expression::operator/(const Expression& other) const { return binaryOp(*this, other, ExprOp::DIV); }

Expression Expression::operator+(float scalarValue) const { return *this + Expression::scalar(scalarValue); }

Expression Expression::operator-(float scalarValue) const { return *this - Expression::scalar(scalarValue); }

Expression Expression::operator*(float scalarValue) const { return *this * Expression::scalar(scalarValue); }

Expression Expression::operator/(float scalarValue) const { return *this / Expression::scalar(scalarValue); }

}  // namespace ThorImplementation
