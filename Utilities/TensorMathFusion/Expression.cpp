#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/EquationCompiler.h"

using DataType = ThorImplementation::TensorDescriptor::DataType;

namespace ThorImplementation {

Stream& Expression::getNextHelperStream(uint32_t gpu_num) {
    static std::vector<std::vector<Stream>> runnerHelperStreams;
    static std::vector<uint32_t> nextHelperStreamIndex;
    constexpr uint32_t MAX_HELPER_STREAMS_PER_GPU = 7;
    if (runnerHelperStreams.empty()) {
        runnerHelperStreams.reserve(MAX_HELPER_STREAMS_PER_GPU);
        nextHelperStreamIndex.reserve(MAX_HELPER_STREAMS_PER_GPU);
        runnerHelperStreams = std::vector<std::vector<Stream>>(MachineEvaluator::instance().getNumGpus(), std::vector<Stream>());
        nextHelperStreamIndex = std::vector<uint32_t>(MachineEvaluator::instance().getNumGpus(), 0);
    }
    uint32_t cur_index = nextHelperStreamIndex[gpu_num];
    if (cur_index == MAX_HELPER_STREAMS_PER_GPU)
        cur_index = 0;
    nextHelperStreamIndex[gpu_num] = cur_index + 1;

    if (cur_index >= runnerHelperStreams[gpu_num].size())
        runnerHelperStreams[gpu_num].emplace_back(gpu_num);

    return runnerHelperStreams[gpu_num][cur_index];
}

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
        case ExprOp::FILL:
            return "FILL";
        case ExprOp::UNSQUEEZE:
            return "UNSQ";
        case ExprOp::SQUEEZE:
            return "SQZ";
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
        case ExprOp::REDUCE_SUM:
            return "RSUM";
        case ExprOp::REDUCE_PROD:
            return "RPROD";
        case ExprOp::REDUCE_MIN:
            return "RMIN";
        case ExprOp::REDUCE_MAX:
            return "RMAX";
        case ExprOp::REDUCE_ARGMIN:
            return "RARGMIN";
        case ExprOp::REDUCE_ARGMAX:
            return "RARGMAX";
        case ExprOp::REDUCE_MIN_BACKWARD:
            return "RMIN_BW";
        case ExprOp::REDUCE_MAX_BACKWARD:
            return "RMAX_BW";
        case ExprOp::REDUCE_AVG:
            return "RAVG";
        case ExprOp::REDUCE_NORM1:
            return "RNORM1";
        case ExprOp::REDUCE_NORM2:
            return "RNORM2";
        default:
            throw std::runtime_error("Unknown ExprOp");
    }
}

static std::string formatUIntVectorCanonical(const std::vector<uint64_t>& values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0)
            ss << ",";
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static std::string formatOptionalDTypeCanonical(const Optional<DataType>& dtype) {
    if (!dtype.isPresent()) {
        return "none";
    }
    return TensorDescriptor::getElementTypeName(dtype.get());
}

static void appendNodeDTypeMetadata(std::string& out, const ExprNode& n) {
    out += ";input=" + formatOptionalDTypeCanonical(n.input_tensor_dtype);
    out += ";out=" + formatOptionalDTypeCanonical(n.output_dtype);
    out += ";compute=" + formatOptionalDTypeCanonical(n.compute_dtype);
    out += ";bwd_out=" + formatOptionalDTypeCanonical(n.backward_output_dtype);
    out += ";bwd_compute=" + formatOptionalDTypeCanonical(n.backward_compute_dtype);
}

static std::string canonicalizeNode(const PhysicalExpression& expr,
                                    uint32_t nodeIndex,
                                    std::vector<std::string>& memo,
                                    std::vector<uint8_t>& memoReady) {
    if (nodeIndex >= expr.nodes.size()) {
        throw std::runtime_error("canonicalizeNode nodeIndex out of range.");
    }

    if (memoReady[nodeIndex]) {
        return memo[nodeIndex];
    }

    const ExprNode& n = expr.nodes[nodeIndex];
    std::string out;

    switch (n.op) {
        case ExprOp::INPUT:
            out = "IN" + std::to_string(n.input_slot);
            break;

        case ExprOp::SCALAR_FP:
            out = "F32(" + formatFloatCanonical(n.scalar_fp) + ")";
            break;

        case ExprOp::FILL:
            out = "FILL(" + formatFloatCanonical(n.scalar_fp) + ";dims=" + formatUIntVectorCanonical(n.fill_dims) + ")";
            break;

        case ExprOp::NEG:
        case ExprOp::EXP:
        case ExprOp::EXP2:
        case ExprOp::EXP10:
        case ExprOp::LN:
        case ExprOp::LOG2:
        case ExprOp::LOG10:
        case ExprOp::SQRT:
            out = opName(n.op) + "(" + canonicalizeNode(expr, n.lhs, memo, memoReady) + ")";
            break;
        case ExprOp::UNSQUEEZE:
            out = opName(n.op) + "(" + canonicalizeNode(expr, n.lhs, memo, memoReady) +
                  ";axes=" + formatUIntVectorCanonical(n.unsqueeze_axes) + ")";
            break;
        case ExprOp::SQUEEZE:
            out = opName(n.op) + "(" + canonicalizeNode(expr, n.lhs, memo, memoReady) +
                  ";axes=" + formatUIntVectorCanonical(n.squeeze_axes) + ")";
            break;

        case ExprOp::REDUCE_SUM:
        case ExprOp::REDUCE_PROD:
        case ExprOp::REDUCE_MIN:
        case ExprOp::REDUCE_MAX:
        case ExprOp::REDUCE_ARGMIN:
        case ExprOp::REDUCE_ARGMAX:
        case ExprOp::REDUCE_AVG:
        case ExprOp::REDUCE_NORM1:
        case ExprOp::REDUCE_NORM2: {
            out = opName(n.op) + "(" + canonicalizeNode(expr, n.lhs, memo, memoReady) +
                  ";axes=" + formatUIntVectorCanonical(n.reduction_axes) + ";squeeze=" + formatUIntVectorCanonical(n.squeeze_axes) + ")";
            break;
        }

        case ExprOp::REDUCE_MIN_BACKWARD:
        case ExprOp::REDUCE_MAX_BACKWARD: {
            std::string a = canonicalizeNode(expr, n.lhs, memo, memoReady);
            std::string b = canonicalizeNode(expr, n.rhs, memo, memoReady);
            out = opName(n.op) + "(" + a + "," + b + ";axes=" + formatUIntVectorCanonical(n.reduction_axes) +
                  ";squeeze=" + formatUIntVectorCanonical(n.squeeze_axes) + ")";
            break;
        }

        case ExprOp::ADD:
        case ExprOp::SUB:
        case ExprOp::MUL:
        case ExprOp::DIV:
        case ExprOp::POW:
        case ExprOp::MIN:
        case ExprOp::MAX: {
            std::string a = canonicalizeNode(expr, n.lhs, memo, memoReady);
            std::string b = canonicalizeNode(expr, n.rhs, memo, memoReady);

            if (isCommutative(n.op) && a > b)
                std::swap(a, b);

            out = opName(n.op) + "(" + a + "," + b + ")";
            break;
        }

        default:
            throw std::runtime_error("Unsupported ExprOp in canonicalizeNode: " + std::to_string((int)n.op));
    }

    appendNodeDTypeMetadata(out, n);
    memo[nodeIndex] = out;
    memoReady[nodeIndex] = 1;
    return memo[nodeIndex];
}

std::string canonicalize(const PhysicalExpression& expr) {
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("canonicalize(PhysicalExpression): output_node out of range.");
    }

    std::vector<std::string> memo(expr.nodes.size());
    std::vector<uint8_t> memoReady(expr.nodes.size(), 0);
    return canonicalizeNode(expr, expr.output_node, memo, memoReady);
}

std::string canonicalize(const PhysicalExecutionStage& stage) {
    std::ostringstream ss;

    ss << "kind=";
    switch (stage.kind) {
        case PhysicalExecutionStage::Kind::FusedKernel:
            ss << "fused";
            break;
        case PhysicalExecutionStage::Kind::Reduction:
            ss << "reduction";
            break;
        case PhysicalExecutionStage::Kind::ArgMinMax:
            ss << "argminmax";
            break;
        case PhysicalExecutionStage::Kind::ReduceMinMaxBackward:
            ss << "reduce_minmax_backward";
            break;
        default:
            throw std::runtime_error("canonicalize(PhysicalExecutionStage): unknown stage kind.");
    }

    ss << ";inputs=[";
    for (size_t i = 0; i < stage.input_value_ids.size(); ++i) {
        if (i > 0)
            ss << ",";
        ss << stage.input_value_ids[i];
    }
    ss << "]";

    std::vector<std::string> memo(stage.expr.nodes.size());
    std::vector<uint8_t> memoReady(stage.expr.nodes.size(), 0);

    ss << ";outputs=[";
    for (size_t i = 0; i < stage.outputs.size(); ++i) {
        if (i > 0)
            ss << ",";

        const CompiledStageOutput& out = stage.outputs[i];
        ss << "{local_node_idx=" << out.local_node_idx << ";expr=";

        if (out.local_node_idx == UINT32_MAX) {
            ss << "NONE";
        } else {
            ss << canonicalizeNode(stage.expr, out.local_node_idx, memo, memoReady);
        }

        ss << "}";
    }
    ss << "]";

    return ss.str();
}

bool Expression::isLeafOp(const ExprOp op) {
    switch (op) {
        case ExprOp::INPUT:
        case ExprOp::SCALAR_FP:
        case ExprOp::FILL:
            return true;
        default:
            return false;
    }
}

bool Expression::isUnaryOp(const ExprOp op) {
    switch (op) {
        case ExprOp::NEG:
        case ExprOp::EXP:
        case ExprOp::EXP2:
        case ExprOp::EXP10:
        case ExprOp::LN:
        case ExprOp::LOG2:
        case ExprOp::LOG10:
        case ExprOp::SQRT:
        case ExprOp::UNSQUEEZE:
        case ExprOp::SQUEEZE:
        case ExprOp::REDUCE_SUM:
        case ExprOp::REDUCE_PROD:
        case ExprOp::REDUCE_MIN:
        case ExprOp::REDUCE_MAX:
        case ExprOp::REDUCE_ARGMIN:
        case ExprOp::REDUCE_ARGMAX:
        case ExprOp::REDUCE_AVG:
        case ExprOp::REDUCE_NORM1:
        case ExprOp::REDUCE_NORM2:
            return true;
        default:
            return false;
    }
}

bool Expression::isBinaryOp(const ExprOp op) {
    switch (op) {
        case ExprOp::ADD:
        case ExprOp::SUB:
        case ExprOp::MUL:
        case ExprOp::DIV:
        case ExprOp::POW:
        case ExprOp::MIN:
        case ExprOp::MAX:
        case ExprOp::REDUCE_MIN_BACKWARD:
        case ExprOp::REDUCE_MAX_BACKWARD:
            return true;
        default:
            return false;
    }
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

    if (Expression::isUnaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing lhs for unary op");
        newNode.lhs = cloneSubtree(src, srcNode.lhs, dst, oldToNew);
        newNode.rhs = UINT32_MAX;
    } else if (Expression::isBinaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing lhs for binary op");
        if (srcNode.rhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing rhs for binary op");
        newNode.lhs = cloneSubtree(src, srcNode.lhs, dst, oldToNew);
        newNode.rhs = cloneSubtree(src, srcNode.rhs, dst, oldToNew);
    } else if (Expression::isLeafOp(srcNode.op)) {
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

uint32_t cloneSubtreeWithMergedInputs(const PhysicalExpression& src,
                                      uint32_t srcNodeIndex,
                                      PhysicalExpression& dst,
                                      std::unordered_map<uint32_t, uint32_t>& oldToNew,
                                      std::unordered_map<std::string, uint32_t>& dstInputSlotsByName) {
    auto it = oldToNew.find(srcNodeIndex);
    if (it != oldToNew.end())
        return it->second;

    const ExprNode& srcNode = src.nodes.at(srcNodeIndex);
    ExprNode newNode = srcNode;

    if (srcNode.op == ExprOp::INPUT) {
        if (srcNode.input_slot >= src.inputs.size()) {
            throw std::runtime_error("Input slot out of range while merging expression outputs.");
        }

        const std::string& inputName = src.inputs[srcNode.input_slot].name;
        auto slotIt = dstInputSlotsByName.find(inputName);
        uint32_t mergedSlot;

        if (slotIt != dstInputSlotsByName.end()) {
            mergedSlot = slotIt->second;
        } else {
            mergedSlot = static_cast<uint32_t>(dst.inputs.size());
            dst.inputs.push_back(NamedInput{inputName, mergedSlot});
            dstInputSlotsByName.emplace(inputName, mergedSlot);
        }

        newNode.input_slot = mergedSlot;
        newNode.lhs = UINT32_MAX;
        newNode.rhs = UINT32_MAX;
    } else if (Expression::isUnaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing lhs for unary op while merging outputs.");
        newNode.lhs = cloneSubtreeWithMergedInputs(src, srcNode.lhs, dst, oldToNew, dstInputSlotsByName);
        newNode.rhs = UINT32_MAX;
    } else if (Expression::isBinaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX || srcNode.rhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing child for binary op while merging outputs.");
        newNode.lhs = cloneSubtreeWithMergedInputs(src, srcNode.lhs, dst, oldToNew, dstInputSlotsByName);
        newNode.rhs = cloneSubtreeWithMergedInputs(src, srcNode.rhs, dst, oldToNew, dstInputSlotsByName);
    } else if (srcNode.op == ExprOp::SCALAR_FP) {
        // nothing to recurse into
    } else {
        throw std::runtime_error("Unsupported op while merging expression outputs: " + std::to_string((int)srcNode.op));
    }

    uint32_t newIndex = static_cast<uint32_t>(dst.nodes.size());
    dst.nodes.push_back(std::move(newNode));
    oldToNew[srcNodeIndex] = newIndex;
    return newIndex;
}

}  // namespace

Expression Expression::input(const std::string& name, Optional<DataType> as_type) {
    auto out = std::make_shared<PhysicalExpression>();

    ExprNode node;
    node.op = ExprOp::INPUT;
    node.input_slot = out->getOrCreateInputSlot(name);

    // as_type means the graph value produced by this input defaults to that dtype,
    // even though the actual bound runtime tensor may have a different dtype.
    if (as_type.isPresent()) {
        node.output_dtype = as_type.get();
    }

    out->nodes.push_back(node);
    out->output_node = 0;

    return Expression(out, 0);
}

Expression::Expression(double value) {
    expr = std::make_shared<PhysicalExpression>();

    ExprNode node{};
    node.op = ExprOp::SCALAR_FP;
    node.scalar_fp = value;

    nodeIndex = static_cast<uint32_t>(expr->nodes.size());
    expr->nodes.push_back(node);
    expr->output_node = nodeIndex;
}

Expression Expression::scalar(double value) { return Expression(value); }
// Expression Expression::scalar(int64_t value) { return Expression(value); }

PhysicalExpression Expression::expression() const {
    if (!expr)
        throw std::runtime_error("Expr has no underlying expression");

    PhysicalExpression out = *expr;
    out.output_node = nodeIndex;
    return out;
}

struct MergeInputsResult {
    std::vector<NamedInput> mergedInputs;
    std::vector<uint32_t> lhsSlotRemap;
    std::vector<uint32_t> rhsSlotRemap;
};

static MergeInputsResult mergeInputsByName(const PhysicalExpression& lhs, const PhysicalExpression& rhs) {
    MergeInputsResult result;

    std::unordered_map<std::string, uint32_t> mergedByName;
    mergedByName.reserve(lhs.inputs.size() + rhs.inputs.size());

    auto getOrCreateMergedSlot = [&](const std::string& name) -> uint32_t {
        auto it = mergedByName.find(name);
        if (it != mergedByName.end())
            return it->second;

        const uint32_t slot = static_cast<uint32_t>(result.mergedInputs.size());
        mergedByName.emplace(name, slot);
        result.mergedInputs.push_back(NamedInput{name, slot});
        return slot;
    };

    result.lhsSlotRemap.resize(lhs.inputs.size());
    for (size_t i = 0; i < lhs.inputs.size(); ++i) {
        result.lhsSlotRemap[i] = getOrCreateMergedSlot(lhs.inputs[i].name);
    }

    result.rhsSlotRemap.resize(rhs.inputs.size());
    for (size_t i = 0; i < rhs.inputs.size(); ++i) {
        result.rhsSlotRemap[i] = getOrCreateMergedSlot(rhs.inputs[i].name);
    }

    return result;
}

static void remapClonedInputSlots(const PhysicalExpression& sourceExpr,
                                  const std::unordered_map<uint32_t, uint32_t>& oldToNewNodeMap,
                                  const std::vector<uint32_t>& slotRemap,
                                  PhysicalExpression& outExpr) {
    for (const auto& [oldNodeIndex, newNodeIndex] : oldToNewNodeMap) {
        const ExprNode& oldNode = sourceExpr.nodes.at(oldNodeIndex);
        if (oldNode.op != ExprOp::INPUT)
            continue;

        if (oldNode.input_slot >= slotRemap.size()) {
            throw std::runtime_error("Input slot out of range while remapping cloned expression.");
        }

        ExprNode& newNode = outExpr.nodes.at(newNodeIndex);
        newNode.input_slot = slotRemap[oldNode.input_slot];
    }
}

Expression Expression::binaryOp(const Expression& lhsExpr, const Expression& rhsExpr, ExprOp op) {
    if (!lhsExpr.expr || !rhsExpr.expr)
        throw std::runtime_error("Cannot combine empty expressions");

    auto out = std::make_shared<PhysicalExpression>();

    const MergeInputsResult mergedInputs = mergeInputsByName(*lhsExpr.expr, *rhsExpr.expr);
    out->inputs = mergedInputs.mergedInputs;

    std::unordered_map<uint32_t, uint32_t> lhsMap;
    std::unordered_map<uint32_t, uint32_t> rhsMap;

    uint32_t newLhsIndex = cloneSubtree(*lhsExpr.expr, lhsExpr.nodeIndex, *out, lhsMap);
    uint32_t newRhsIndex = cloneSubtree(*rhsExpr.expr, rhsExpr.nodeIndex, *out, rhsMap);

    remapClonedInputSlots(*lhsExpr.expr, lhsMap, mergedInputs.lhsSlotRemap, *out);
    remapClonedInputSlots(*rhsExpr.expr, rhsMap, mergedInputs.rhsSlotRemap, *out);

    ExprNode node{};
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
    out->inputs = inputExpr.expr->inputs;

    std::unordered_map<uint32_t, uint32_t> oldToNew;
    uint32_t newLhsIndex = cloneSubtree(*inputExpr.expr, inputExpr.nodeIndex, *out, oldToNew);

    ExprNode node{};
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
Expression Expression::operator-() const { return unaryOp(*this, ExprOp::NEG); }
Expression Expression::sqrt() const { return unaryOp(*this, ExprOp::SQRT); }

Expression Expression::unsqueeze(const std::vector<uint64_t>& unsqueeze_axes) const {
    if (!expr)
        throw std::runtime_error("Cannot unsqueeze an empty expression");

    Expression out = unaryOp(*this, ExprOp::UNSQUEEZE);
    std::vector<uint64_t> normalized = unsqueeze_axes;
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
    out.expr->nodes[out.nodeIndex].unsqueeze_axes = std::move(normalized);
    return out;
}

Expression Expression::squeeze(const std::vector<uint64_t>& squeeze_axes) const {
    if (!expr)
        throw std::runtime_error("Cannot squeeze an empty expression");

    Expression out = unaryOp(*this, ExprOp::SQUEEZE);
    std::vector<uint64_t> normalized = squeeze_axes;
    std::sort(normalized.begin(), normalized.end());
    normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
    out.expr->nodes[out.nodeIndex].squeeze_axes = std::move(normalized);
    return out;
}

Expression Expression::pow(const Expression& exponent) const { return binaryOp(*this, exponent, ExprOp::POW); }

// Reductions
DataType validate_reduction_compute_type(Optional<DataType> compute_dtype) {
    if (compute_dtype.isPresent() && compute_dtype.get() != DataType::FP32) {
        throw std::runtime_error("Reductions currently only support compute_dtype = FP32. Received: " +
                                 TensorDescriptor::getElementTypeName(compute_dtype.get()));
    }
    return DataType::FP32;
}

static bool isArgReductionOp(ExprOp op) { return op == ExprOp::REDUCE_ARGMIN || op == ExprOp::REDUCE_ARGMAX; }

Expression Expression::reduction(ExprOp op,
                                 const std::vector<uint64_t>& reduction_axes,
                                 const std::vector<uint64_t>& squeeze_axes,
                                 Optional<DataType> compute_dtype) const {
    Expression out = unaryOp(*this, op);

    out.expr->nodes[out.nodeIndex].reduction_axes = reduction_axes;
    out.expr->nodes[out.nodeIndex].squeeze_axes = squeeze_axes;
    out.expr->nodes[out.nodeIndex].compute_dtype = validate_reduction_compute_type(compute_dtype);
    if (isArgReductionOp(op)) {
        out.expr->nodes[out.nodeIndex].output_dtype = DataType::UINT32;
        out.expr->nodes[out.nodeIndex].backward_output_dtype = DataType::UINT32;
        out.expr->nodes[out.nodeIndex].backward_compute_dtype = DataType::FP32;
    }
    return out;
}

Expression Expression::reduce_sum(const std::vector<uint64_t>& reduction_axes,
                                  const std::vector<uint64_t>& squeeze_axes,
                                  Optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_SUM, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::reduce_prod(const std::vector<uint64_t>& reduction_axes,
                                   const std::vector<uint64_t>& squeeze_axes,
                                   Optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_PROD, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::reduce_min(const std::vector<uint64_t>& reduction_axes,
                                  const std::vector<uint64_t>& squeeze_axes,
                                  Optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_MIN, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::reduce_max(const std::vector<uint64_t>& reduction_axes,
                                  const std::vector<uint64_t>& squeeze_axes,
                                  Optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_MAX, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::argmin(const std::vector<uint64_t>& reduction_axes,
                              const std::vector<uint64_t>& squeeze_axes,
                              Optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_ARGMIN, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::argmax(const std::vector<uint64_t>& reduction_axes,
                              const std::vector<uint64_t>& squeeze_axes,
                              Optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_ARGMAX, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::reduce_mean(const std::vector<uint64_t>& reduction_axes,
                                   const std::vector<uint64_t>& squeeze_axes,
                                   Optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_AVG, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::reduce_norm1(const std::vector<uint64_t>& reduction_axes,
                                    const std::vector<uint64_t>& squeeze_axes,
                                    Optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_NORM1, reduction_axes, squeeze_axes, compute_dtype);
}

Expression Expression::reduce_norm2(const std::vector<uint64_t>& reduction_axes,
                                    const std::vector<uint64_t>& squeeze_axes,
                                    Optional<DataType> compute_dtype) const {
    return reduction(ExprOp::REDUCE_NORM2, reduction_axes, squeeze_axes, compute_dtype);
}

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

uint32_t PhysicalExpression::getOrCreateInputSlot(const std::string& name) {
    for (const NamedInput& input : inputs) {
        if (input.name == name)
            return input.slot;
    }

    const uint32_t slot = static_cast<uint32_t>(inputs.size());
    inputs.push_back(NamedInput{name, slot});
    return slot;
}

Outputs Expression::outputs(const std::vector<std::pair<std::string, Expression>>& named_exprs) {
    if (named_exprs.empty()) {
        throw std::runtime_error("Expression::outputs requires at least one named output.");
    }

    auto merged = std::make_shared<PhysicalExpression>();
    std::vector<NamedOutput> outputs;
    outputs.reserve(named_exprs.size());

    std::unordered_set<std::string> seen_names;
    std::unordered_map<std::string, uint32_t> mergedInputSlotsByName;

    for (const auto& [name, expr] : named_exprs) {
        if (name.empty()) {
            throw std::runtime_error("Output name cannot be empty.");
        }

        if (!seen_names.insert(name).second) {
            throw std::runtime_error("Duplicate output name: " + name);
        }

        if (!expr.expr) {
            throw std::runtime_error("Output expression has no backing PhysicalExpression.");
        }

        if (expr.nodeIndex == UINT32_MAX) {
            throw std::runtime_error("Output expression has invalid node index.");
        }

        if (expr.nodeIndex >= expr.expr->nodes.size()) {
            throw std::runtime_error("Output expression node index is out of range.");
        }

        std::unordered_map<uint32_t, uint32_t> oldToNew;
        uint32_t mergedRoot = cloneSubtreeWithMergedInputs(*expr.expr, expr.nodeIndex, *merged, oldToNew, mergedInputSlotsByName);

        outputs.push_back(NamedOutput{
            .name = name,
            .node_idx = mergedRoot,
        });
    }

    return Outputs(std::move(merged), std::move(outputs));
}

Outputs Expression::outputs(std::initializer_list<std::pair<std::string, Expression>> named_exprs) {
    return outputs(std::vector<std::pair<std::string, Expression>>(named_exprs));
}

}  // namespace ThorImplementation
