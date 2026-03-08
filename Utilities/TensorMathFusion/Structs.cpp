#include "Utilities/TensorMathFusion/Structs.h"

namespace ThorImplementation {

void hashCombine(std::size_t& seed, std::size_t value) { seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2); }

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

}  // namespace ThorImplementation
