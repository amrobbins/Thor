#include "Utilities/TensorMathFusion/Expression.h"

namespace ThorImplementation {

std::string formatFloatCanonical(double x) {
    std::ostringstream ss;
    ss << std::setprecision(9) << x;
    return ss.str();
}

bool isCommutative(ExprOp op) { return op == ExprOp::ADD || op == ExprOp::MUL || op == ExprOp::MIN || op == ExprOp::MAX; }

std::string opName(ExprOp op) {
    switch (op) {
        case ExprOp::INPUT:
            return "IN";
        case ExprOp::SCALAR_FP:
            return "F32";
        case ExprOp::SCALAR_INT:
            // FIXME:
            return "F32";
        case ExprOp::ADD:
            return "ADD";
        case ExprOp::SUB:
            return "SUB";
        case ExprOp::MUL:
            return "MUL";
        case ExprOp::DIV:
            return "DIV";
        case ExprOp::POW:
            return "POW";
        case ExprOp::NEG:
            return "NEG";
        case ExprOp::EXP:
            return "EXP";
        case ExprOp::LN:
            return "LOG";
        case ExprOp::SQRT:
            return "SQRT";
        case ExprOp::EXP2:
            return "EXP2";
        case ExprOp::EXP10:
            return "EXP10";
        case ExprOp::LOG2:
            return "LOG2";
        case ExprOp::LOG10:
            return "LOG10";
        case ExprOp::MIN:
            return "MIN";
        case ExprOp::MAX:
            return "MAX";
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

        case ExprOp::SCALAR_FP:
            out = "F32(" + formatFloatCanonical(n.scalar_fp) + ")";
            break;
        case ExprOp::SCALAR_INT:
            // FIXME: don't convert to a float, but this is fp32 only now
            out = "F32(" + formatFloatCanonical(n.scalar_int) + ")";
            break;

        case ExprOp::NEG:
        case ExprOp::EXP:
        case ExprOp::EXP2:
        case ExprOp::EXP10:
        case ExprOp::LN:
        case ExprOp::LOG2:
        case ExprOp::LOG10:
        case ExprOp::SQRT:
            out = opName(n.op) + "(" + canonicalizeNode(expr, n.lhs, memo) + ")";
            break;

        case ExprOp::ADD:
        case ExprOp::SUB:
        case ExprOp::MUL:
        case ExprOp::DIV:
        case ExprOp::POW:
        case ExprOp::MIN:
        case ExprOp::MAX: {
            std::string a = canonicalizeNode(expr, n.lhs, memo);
            std::string b = canonicalizeNode(expr, n.rhs, memo);

            if (isCommutative(n.op) && a > b)
                std::swap(a, b);

            out = opName(n.op) + "(" + a + "," + b + ")";
            break;
        }

        default:
            std::string error_message = "Unsupported ExprOp in canonicalizeNode: " + std::to_string((int)n.op);
            throw std::runtime_error(error_message.c_str());
    }

    memo[nodeIndex] = out;
    return out;
}

std::string canonicalize(const PhysicalExpression& expr) {
    std::unordered_map<uint32_t, std::string> memo;
    return canonicalizeNode(expr, expr.output_node, memo);
}

namespace {

bool isLeafOp(ExprOp op) { return op == ExprOp::INPUT || op == ExprOp::SCALAR_FP || op == ExprOp::SCALAR_INT; }

bool isUnaryOp(ExprOp op) {
    return op == ExprOp::NEG || op == ExprOp::EXP || op == ExprOp::EXP2 || op == ExprOp::EXP10 || op == ExprOp::LN || op == ExprOp::LOG2 ||
           op == ExprOp::LOG10 || op == ExprOp::SQRT;
}

bool isBinaryOp(ExprOp op) {
    return op == ExprOp::ADD || op == ExprOp::SUB || op == ExprOp::MUL || op == ExprOp::DIV || op == ExprOp::POW || op == ExprOp::MIN ||
           op == ExprOp::MAX;
}

uint32_t cloneSubtree(const PhysicalExpression& src,
                      uint32_t srcNodeIndex,
                      PhysicalExpression& dst,
                      std::unordered_map<uint32_t, uint32_t>& oldToNew) {
    auto it = oldToNew.find(srcNodeIndex);
    if (it != oldToNew.end())
        return it->second;

    const ExprNode& srcNode = src.nodes[srcNodeIndex];
    ExprNode newNode = srcNode;

    if (isUnaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing lhs for unary op");
        newNode.lhs = cloneSubtree(src, srcNode.lhs, dst, oldToNew);
        newNode.rhs = UINT32_MAX;
    } else if (isBinaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing lhs for binary op");
        if (srcNode.rhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing rhs for binary op");
        newNode.lhs = cloneSubtree(src, srcNode.lhs, dst, oldToNew);
        newNode.rhs = cloneSubtree(src, srcNode.rhs, dst, oldToNew);
    } else if (isLeafOp(srcNode.op)) {
        // nothing to recurse into
    } else {
        std::string error_message = "Malformed expression: unsupported op in cloneSubtree: " + std::to_string((int)srcNode.op);
        throw std::runtime_error(error_message.c_str());
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

Expression::Expression(double value) {
    expr = std::make_shared<PhysicalExpression>();
    expr->num_inputs = 0;

    ExprNode node{};
    node.op = ExprOp::SCALAR_FP;
    node.scalar_fp = value;

    nodeIndex = static_cast<uint32_t>(expr->nodes.size());
    expr->nodes.push_back(node);
    expr->output_node = nodeIndex;
}

Expression::Expression(int64_t value) {
    expr = std::make_shared<PhysicalExpression>();
    expr->num_inputs = 0;

    ExprNode node{};
    node.op = ExprOp::SCALAR_INT;
    node.scalar_int = value;

    nodeIndex = static_cast<uint32_t>(expr->nodes.size());
    expr->nodes.push_back(node);
    expr->output_node = nodeIndex;
}

Expression Expression::scalar(double value) { return Expression(value); }
Expression Expression::scalar(int64_t value) { return Expression(value); }

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

Expression Expression::unaryOp(const Expression& inputExpr, ExprOp op) {
    if (!inputExpr.expr)
        throw std::runtime_error("Cannot apply unary op to empty expression");

    auto out = std::make_shared<PhysicalExpression>();
    out->num_inputs = inputExpr.expr->num_inputs;

    std::unordered_map<uint32_t, uint32_t> oldToNew;
    uint32_t newLhsIndex = cloneSubtree(*inputExpr.expr, inputExpr.nodeIndex, *out, oldToNew);

    ExprNode node;
    node.op = op;
    node.lhs = newLhsIndex;

    uint32_t newIndex = static_cast<uint32_t>(out->nodes.size());
    out->nodes.push_back(node);
    out->output_node = newIndex;

    return Expression(out, newIndex);
}

Expression Expression::operator+(const Expression& other) const { return binaryOp(*this, other, ExprOp::ADD); }
Expression Expression::operator-(const Expression& other) const { return binaryOp(*this, other, ExprOp::SUB); }
Expression Expression::operator*(const Expression& other) const { return binaryOp(*this, other, ExprOp::MUL); }
Expression Expression::operator/(const Expression& other) const { return binaryOp(*this, other, ExprOp::DIV); }
// Expression Expression::operator+(double scalarValue) const { return *this + Expression::scalar(scalarValue); }
// Expression Expression::operator-(double scalarValue) const { return *this - Expression::scalar(scalarValue); }
// Expression Expression::operator*(double scalarValue) const { return *this * Expression::scalar(scalarValue); }
// Expression Expression::operator/(double scalarValue) const { return *this / Expression::scalar(scalarValue); }
Expression Expression::operator-() const { return unaryOp(*this, ExprOp::NEG); }
Expression Expression::sqrt() const { return unaryOp(*this, ExprOp::SQRT); }
// x_i^y_i  both tensors <- deprecated comment
Expression Expression::pow(const Expression& exponent) const { return binaryOp(*this, exponent, ExprOp::POW); }
// x_i^scalar_exponent
// Expression Expression::pow(double exponent) const { return this->pow(Expression::scalar(exponent)); }

Expression Expression::ln() const { return unaryOp(*this, ExprOp::LN); }
Expression Expression::log2() const { return unaryOp(*this, ExprOp::LOG2); }
Expression Expression::log10() const { return unaryOp(*this, ExprOp::LOG10); }
Expression Expression::log(double base) const {
    if (base <= 0.0f || base == 1.0f) {
        throw std::runtime_error("log base must be positive and not equal to 1, received base = " + std::to_string(base));
    }
    return this->ln() / Expression::scalar(std::log(base));
}

Expression Expression::min(const Expression& other) const { return binaryOp(*this, other, ExprOp::MIN); }
Expression Expression::max(const Expression& other) const { return binaryOp(*this, other, ExprOp::MAX); }

// e^x_i
Expression Expression::exp() const { return unaryOp(*this, ExprOp::EXP); }
// 2^x_i
Expression Expression::exp2() const { return unaryOp(*this, ExprOp::EXP2); }
Expression Expression::exp10() const { return unaryOp(*this, ExprOp::EXP10); }
// Can also use Expression::scalar(s).pow(x) for s^x_i

}  // namespace ThorImplementation
