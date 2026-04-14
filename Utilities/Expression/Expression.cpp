#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"

#include <functional>

using DataType = ThorImplementation::TensorDescriptor::DataType;

namespace ThorImplementation {

static void validateUserInputName(const std::string& name) {
    if (name.rfind("__arg", 0) == 0) {
        throw std::runtime_error("Input names may not start with reserved prefix '__arg'.");
    }
}

namespace {

class HelperStreamPool {
   public:
    static constexpr uint32_t MAX_HELPER_STREAMS_PER_GPU = 8;
    static_assert((MAX_HELPER_STREAMS_PER_GPU & (MAX_HELPER_STREAMS_PER_GPU - 1)) == 0,
                  "MAX_HELPER_STREAMS_PER_GPU must be a power of two");
    static constexpr uint32_t HELPER_STREAM_MASK = MAX_HELPER_STREAMS_PER_GPU - 1;

    struct PerGpuHelperStreams {
        std::array<Stream, MAX_HELPER_STREAMS_PER_GPU> streams;
        std::atomic<uint32_t> next_index{0};

        explicit PerGpuHelperStreams(int32_t gpu)
            : streams{Stream(gpu), Stream(gpu), Stream(gpu), Stream(gpu), Stream(gpu), Stream(gpu), Stream(gpu), Stream(gpu)} {
            for (Stream& stream : streams) {
                stream.informIsStatic();
            }
        }
    };

    Stream& getNextHelperStream(uint32_t gpu_num) {
        ensureInitialized();
        assert(gpu_num < per_gpu_.size());

        PerGpuHelperStreams& state = *per_gpu_[gpu_num];
        const uint32_t idx = state.next_index.fetch_add(1, std::memory_order_relaxed) & HELPER_STREAM_MASK;
        return state.streams[idx];
    }

   private:
    void ensureInitialized() {
        std::call_once(init_once_, [this]() {
            const uint32_t num_gpus = MachineEvaluator::instance().getNumGpus();
            per_gpu_.reserve(num_gpus);
            for (uint32_t g = 0; g < num_gpus; ++g) {
                per_gpu_.push_back(std::make_unique<PerGpuHelperStreams>(g));
            }
        });
    }

    std::once_flag init_once_;
    std::vector<std::unique_ptr<PerGpuHelperStreams>> per_gpu_;
};

HelperStreamPool helperStreamPool;

}  // namespace

Stream& Expression::getNextHelperStream(uint32_t gpu_num) { return helperStreamPool.getNextHelperStream(gpu_num); }

std::set<std::string> Expression::getInputNames() const {
    if (expr == nullptr)
        return {};
    return expr->getInputNames();
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
        case ExprOp::RUNTIME_SCALAR:
            return "RIN";
        case ExprOp::TENSOR_RUNTIME_SCALAR:
            return "TRIN";
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
        case ExprOp::ABS:
            return "ABS";
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
        case ExprOp::TRANSPOSE:
            return "TRANSPOSE";
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
        case ExprOp::MIN_GRAD_LEFT:
            return "MIN_GL";
        case ExprOp::MIN_GRAD_RIGHT:
            return "MIN_GR";
        case ExprOp::MAX_GRAD_LEFT:
            return "MAX_GL";
        case ExprOp::MAX_GRAD_RIGHT:
            return "MAX_GR";
        case ExprOp::MATMUL:
            return "MATMUL";
        case ExprOp::GEMM:
            return "GEMM";
        case ExprOp::CONV2D:
            return "CONV2D";
        case ExprOp::CONV2D_BACKWARD_DATA:
            return "CONV2D_BWD_DATA";
        case ExprOp::CONV2D_BACKWARD_FILTER:
            return "CONV2D_BWD_FILTER";
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

        case ExprOp::RUNTIME_SCALAR:
            out = "RIN" + std::to_string(n.input_slot);
            break;

        case ExprOp::TENSOR_RUNTIME_SCALAR:
            out = "TRIN" + std::to_string(n.input_slot);
            break;

        case ExprOp::SCALAR_FP:
            out = "F32(" + formatFloatCanonical(n.scalar_fp) + ")";
            break;

        case ExprOp::FILL:
            out = "FILL(" + formatFloatCanonical(n.scalar_fp) + ";dims=" + formatUIntVectorCanonical(n.fill_dims) + ")";
            break;

        case ExprOp::NEG:
        case ExprOp::ABS:
        case ExprOp::EXP:
        case ExprOp::EXP2:
        case ExprOp::EXP10:
        case ExprOp::LN:
        case ExprOp::LOG2:
        case ExprOp::LOG10:
        case ExprOp::SQRT:
        case ExprOp::TRANSPOSE:
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
        case ExprOp::MAX:
        case ExprOp::MIN_GRAD_LEFT:
        case ExprOp::MIN_GRAD_RIGHT:
        case ExprOp::MAX_GRAD_LEFT:
        case ExprOp::MAX_GRAD_RIGHT:
        case ExprOp::MATMUL:
        case ExprOp::CONV2D:
        case ExprOp::CONV2D_BACKWARD_DATA:
        case ExprOp::CONV2D_BACKWARD_FILTER: {
            std::string a = canonicalizeNode(expr, n.lhs, memo, memoReady);
            std::string b = canonicalizeNode(expr, n.rhs, memo, memoReady);

            if (isCommutative(n.op) && a > b)
                std::swap(a, b);

            out = opName(n.op) + "(" + a + "," + b;
            if (n.op == ExprOp::MATMUL) {
                out += ";tA=" + std::to_string(n.transpose_lhs ? 1 : 0);
                out += ";tB=" + std::to_string(n.transpose_rhs ? 1 : 0);
            } else if (n.op == ExprOp::CONV2D || n.op == ExprOp::CONV2D_BACKWARD_DATA || n.op == ExprOp::CONV2D_BACKWARD_FILTER) {
                out += ";sH=" + std::to_string(n.conv_stride_h);
                out += ";sW=" + std::to_string(n.conv_stride_w);
                out += ";pH=" + std::to_string(n.conv_pad_h);
                out += ";pW=" + std::to_string(n.conv_pad_w);
                if (!n.fill_dims.empty()) {
                    out += ";shape=";
                    for (size_t i = 0; i < n.fill_dims.size(); ++i) {
                        if (i != 0)
                            out += "x";
                        out += std::to_string(n.fill_dims[i]);
                    }
                }
            }
            out += ")";
            break;
        }

        case ExprOp::GEMM: {
            std::string a = canonicalizeNode(expr, n.lhs, memo, memoReady);
            std::string b = canonicalizeNode(expr, n.rhs, memo, memoReady);
            std::string c = canonicalizeNode(expr, n.aux, memo, memoReady);
            auto gemmScaleString = [&](const char* label, uint32_t scale_node, double scale_fp) {
                if (scale_node != UINT32_MAX) {
                    return std::string(";") + label + "Node=" + canonicalizeNode(expr, scale_node, memo, memoReady) + ";" + label +
                           "Scale=" + formatFloatCanonical(scale_fp);
                }
                return std::string(";") + label + "=" + formatFloatCanonical(scale_fp);
            };
            out = opName(n.op) + "(" + a + "," + b + "," + c + gemmScaleString("alpha", n.alpha_node, n.alpha_fp) +
                  gemmScaleString("beta", n.beta_node, n.beta_fp) + ";tA=" + std::to_string(n.transpose_lhs ? 1 : 0) +
                  ";tB=" + std::to_string(n.transpose_rhs ? 1 : 0) + ";tC=" + std::to_string(n.transpose_aux ? 1 : 0) + ")";
            break;
        }

        default:
            throw std::runtime_error("Unsupported ExprOp in canonicalizeNode: " + std::to_string(static_cast<int>(n.op)));
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
        case PhysicalExecutionStage::Kind::Matmul:
            ss << "matmul";
            break;
        case PhysicalExecutionStage::Kind::Convolution:
            ss << "convolution";
            break;
        case PhysicalExecutionStage::Kind::ConvolutionBackward:
            ss << "convolution_backward";
            break;
        case PhysicalExecutionStage::Kind::ReduceMinMaxBackward:
            ss << "reduce_minmax_backward";
            break;
        case PhysicalExecutionStage::Kind::Transpose:
            ss << "transpose";
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
        case ExprOp::RUNTIME_SCALAR:
        case ExprOp::TENSOR_RUNTIME_SCALAR:
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
        case ExprOp::ABS:
        case ExprOp::EXP:
        case ExprOp::EXP2:
        case ExprOp::EXP10:
        case ExprOp::LN:
        case ExprOp::LOG2:
        case ExprOp::LOG10:
        case ExprOp::SQRT:
        case ExprOp::TRANSPOSE:
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
        case ExprOp::MIN_GRAD_LEFT:
        case ExprOp::MIN_GRAD_RIGHT:
        case ExprOp::MAX_GRAD_LEFT:
        case ExprOp::MAX_GRAD_RIGHT:
        case ExprOp::MATMUL:
        case ExprOp::CONV2D:
        case ExprOp::CONV2D_BACKWARD_DATA:
        case ExprOp::CONV2D_BACKWARD_FILTER:
        case ExprOp::REDUCE_MIN_BACKWARD:
        case ExprOp::REDUCE_MAX_BACKWARD:
            return true;
        default:
            return false;
    }
}

bool Expression::isTernaryOp(const ExprOp op) {
    switch (op) {
        case ExprOp::GEMM:
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
        newNode.aux = UINT32_MAX;
    } else if (Expression::isBinaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing lhs for binary op");
        if (srcNode.rhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing rhs for binary op");
        newNode.lhs = cloneSubtree(src, srcNode.lhs, dst, oldToNew);
        newNode.rhs = cloneSubtree(src, srcNode.rhs, dst, oldToNew);
        newNode.aux = UINT32_MAX;
    } else if (Expression::isTernaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX || srcNode.rhs == UINT32_MAX || srcNode.aux == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing child for ternary op");
        newNode.lhs = cloneSubtree(src, srcNode.lhs, dst, oldToNew);
        newNode.rhs = cloneSubtree(src, srcNode.rhs, dst, oldToNew);
        newNode.aux = cloneSubtree(src, srcNode.aux, dst, oldToNew);
        if (srcNode.alpha_node != UINT32_MAX) {
            newNode.alpha_node = cloneSubtree(src, srcNode.alpha_node, dst, oldToNew);
        }
        if (srcNode.beta_node != UINT32_MAX) {
            newNode.beta_node = cloneSubtree(src, srcNode.beta_node, dst, oldToNew);
        }
    } else if (Expression::isLeafOp(srcNode.op)) {
        // nothing to recurse into
    } else {
        std::string error_message = "Malformed expression: unsupported op in cloneSubtree: " + std::to_string(static_cast<int>(srcNode.op));
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

    if (srcNode.op == ExprOp::INPUT || srcNode.op == ExprOp::RUNTIME_SCALAR || srcNode.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
        if (srcNode.input_slot >= src.inputs.size()) {
            throw std::runtime_error("Input slot out of range while merging expression outputs.");
        }

        const std::string& inputName = src.inputs[srcNode.input_slot].name;
        const NamedInput::Kind inputKind = src.inputs[srcNode.input_slot].kind;
        auto slotIt = dstInputSlotsByName.find(inputName);
        uint32_t mergedSlot;

        if (slotIt != dstInputSlotsByName.end()) {
            mergedSlot = slotIt->second;
            if (mergedSlot >= dst.inputs.size()) {
                throw std::runtime_error("Merged input slot out of range while merging expression outputs.");
            }
            if (dst.inputs[mergedSlot].kind != inputKind) {
                throw std::runtime_error("Input kind mismatch while merging expression outputs for input: " + inputName);
            }
        } else {
            mergedSlot = static_cast<uint32_t>(dst.inputs.size());
            dst.inputs.push_back(NamedInput{inputName, mergedSlot, inputKind});
            dstInputSlotsByName.emplace(inputName, mergedSlot);
        }

        newNode.input_slot = mergedSlot;
        newNode.lhs = UINT32_MAX;
        newNode.rhs = UINT32_MAX;
        newNode.aux = UINT32_MAX;
    } else if (Expression::isUnaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing lhs for unary op while merging outputs.");
        newNode.lhs = cloneSubtreeWithMergedInputs(src, srcNode.lhs, dst, oldToNew, dstInputSlotsByName);
        newNode.rhs = UINT32_MAX;
        newNode.aux = UINT32_MAX;
    } else if (Expression::isBinaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX || srcNode.rhs == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing child for binary op while merging outputs.");
        newNode.lhs = cloneSubtreeWithMergedInputs(src, srcNode.lhs, dst, oldToNew, dstInputSlotsByName);
        newNode.rhs = cloneSubtreeWithMergedInputs(src, srcNode.rhs, dst, oldToNew, dstInputSlotsByName);
        newNode.aux = UINT32_MAX;
    } else if (Expression::isTernaryOp(srcNode.op)) {
        if (srcNode.lhs == UINT32_MAX || srcNode.rhs == UINT32_MAX || srcNode.aux == UINT32_MAX)
            throw std::runtime_error("Malformed expression: missing child for ternary op while merging outputs.");
        newNode.lhs = cloneSubtreeWithMergedInputs(src, srcNode.lhs, dst, oldToNew, dstInputSlotsByName);
        newNode.rhs = cloneSubtreeWithMergedInputs(src, srcNode.rhs, dst, oldToNew, dstInputSlotsByName);
        newNode.aux = cloneSubtreeWithMergedInputs(src, srcNode.aux, dst, oldToNew, dstInputSlotsByName);
        if (srcNode.alpha_node != UINT32_MAX) {
            newNode.alpha_node = cloneSubtreeWithMergedInputs(src, srcNode.alpha_node, dst, oldToNew, dstInputSlotsByName);
        }
        if (srcNode.beta_node != UINT32_MAX) {
            newNode.beta_node = cloneSubtreeWithMergedInputs(src, srcNode.beta_node, dst, oldToNew, dstInputSlotsByName);
        }
    } else if (srcNode.op == ExprOp::SCALAR_FP || srcNode.op == ExprOp::FILL) {
        // nothing to recurse into
    } else {
        throw std::runtime_error("Unsupported op while merging expression outputs: " + std::to_string(static_cast<int>(srcNode.op)));
    }

    uint32_t newIndex = static_cast<uint32_t>(dst.nodes.size());
    dst.nodes.push_back(std::move(newNode));
    oldToNew[srcNodeIndex] = newIndex;
    return newIndex;
}

}  // namespace

Expression Expression::input(const std::string& name, Optional<DataType> compute_dtype, Optional<DataType> output_dtype) {
    validateUserInputName(name);
    auto out = std::make_shared<PhysicalExpression>();

    ExprNode node;
    node.op = ExprOp::INPUT;
    node.input_slot = out->getOrCreateInputSlot(name, NamedInput::Kind::Tensor);

    // output_dtype means the graph value produced by this input defaults to that dtype,
    // even though the actual bound runtime tensor may have a different dtype.
    if (output_dtype.isPresent()) {
        node.output_dtype = output_dtype.get();
    }
    if (compute_dtype.isPresent()) {
        node.compute_dtype = compute_dtype.get();
    }

    out->nodes.push_back(node);
    out->output_node = 0;

    return Expression(out, 0);
}

Expression Expression::runtimeScalar(const std::string& name, Optional<DataType> compute_dtype, Optional<DataType> output_dtype) {
    validateUserInputName(name);
    auto out = std::make_shared<PhysicalExpression>();

    ExprNode node;
    node.op = ExprOp::RUNTIME_SCALAR;
    node.input_slot = out->getOrCreateInputSlot(name, NamedInput::Kind::RuntimeScalarFp32);

    if (output_dtype.isPresent()) {
        node.output_dtype = output_dtype.get();
    }
    if (compute_dtype.isPresent()) {
        node.compute_dtype = compute_dtype.get();
    }

    out->nodes.push_back(node);
    out->output_node = 0;

    return Expression(out, 0);
}

Expression Expression::tensorRuntimeScalar(const std::string& name, Optional<DataType> compute_dtype, Optional<DataType> output_dtype) {
    validateUserInputName(name);
    auto out = std::make_shared<PhysicalExpression>();

    ExprNode node;
    node.op = ExprOp::TENSOR_RUNTIME_SCALAR;
    node.input_slot = out->getOrCreateInputSlot(name, NamedInput::Kind::TensorRuntimeScalar);

    if (output_dtype.isPresent()) {
        node.output_dtype = output_dtype.get();
    }
    if (compute_dtype.isPresent()) {
        node.compute_dtype = compute_dtype.get();
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

Expression Expression::constantScalar(double value) { return Expression(value); }
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

    auto getOrCreateMergedSlot = [&](const std::string& name, NamedInput::Kind kind) -> uint32_t {
        auto it = mergedByName.find(name);
        if (it != mergedByName.end()) {
            if (it->second >= result.mergedInputs.size()) {
                throw std::runtime_error("Merged input slot out of range while combining expressions.");
            }
            if (result.mergedInputs[it->second].kind != kind) {
                throw std::runtime_error("Input kind mismatch while combining expressions for input: " + name);
            }
            return it->second;
        }

        const uint32_t slot = static_cast<uint32_t>(result.mergedInputs.size());
        mergedByName.emplace(name, slot);
        result.mergedInputs.push_back(NamedInput{name, slot, kind});
        return slot;
    };

    result.lhsSlotRemap.resize(lhs.inputs.size());
    for (size_t i = 0; i < lhs.inputs.size(); ++i) {
        result.lhsSlotRemap[i] = getOrCreateMergedSlot(lhs.inputs[i].name, lhs.inputs[i].kind);
    }

    result.rhsSlotRemap.resize(rhs.inputs.size());
    for (size_t i = 0; i < rhs.inputs.size(); ++i) {
        result.rhsSlotRemap[i] = getOrCreateMergedSlot(rhs.inputs[i].name, rhs.inputs[i].kind);
    }

    return result;
}

static void remapClonedInputSlots(const PhysicalExpression& sourceExpr,
                                  const std::unordered_map<uint32_t, uint32_t>& oldToNewNodeMap,
                                  const std::vector<uint32_t>& slotRemap,
                                  PhysicalExpression& outExpr) {
    for (const auto& [oldNodeIndex, newNodeIndex] : oldToNewNodeMap) {
        const ExprNode& oldNode = sourceExpr.nodes.at(oldNodeIndex);
        if (oldNode.op != ExprOp::INPUT && oldNode.op != ExprOp::RUNTIME_SCALAR && oldNode.op != ExprOp::TENSOR_RUNTIME_SCALAR)
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

Expression Expression::ternaryOp(const Expression& lhsExpr, const Expression& rhsExpr, const Expression& auxExpr, ExprOp op) {
    if (!lhsExpr.expr || !rhsExpr.expr || !auxExpr.expr)
        throw std::runtime_error("Cannot combine empty expressions");

    auto out = std::make_shared<PhysicalExpression>();

    const MergeInputsResult lhs_rhs_inputs = mergeInputsByName(*lhsExpr.expr, *rhsExpr.expr);

    std::unordered_map<std::string, uint32_t> mergedByName;
    mergedByName.reserve(lhs_rhs_inputs.mergedInputs.size());
    for (const NamedInput& input : lhs_rhs_inputs.mergedInputs) {
        mergedByName.emplace(input.name, input.slot);
    }

    // Reuse the final out expression directly so slot indices stay stable.
    out->inputs = lhs_rhs_inputs.mergedInputs;
    for (const NamedInput& input : auxExpr.expr->inputs) {
        auto it = mergedByName.find(input.name);
        if (it != mergedByName.end()) {
            if (out->inputs[it->second].kind != input.kind) {
                throw std::runtime_error("Input kind mismatch while combining expressions for input: " + input.name);
            }
        } else {
            const uint32_t slot = static_cast<uint32_t>(out->inputs.size());
            out->inputs.push_back(NamedInput{input.name, slot, input.kind});
            mergedByName.emplace(input.name, slot);
        }
    }

    std::unordered_map<uint32_t, uint32_t> lhsMap;
    std::unordered_map<uint32_t, uint32_t> rhsMap;
    std::unordered_map<uint32_t, uint32_t> auxMap;

    uint32_t newLhsIndex = cloneSubtree(*lhsExpr.expr, lhsExpr.nodeIndex, *out, lhsMap);
    uint32_t newRhsIndex = cloneSubtree(*rhsExpr.expr, rhsExpr.nodeIndex, *out, rhsMap);
    uint32_t newAuxIndex = cloneSubtree(*auxExpr.expr, auxExpr.nodeIndex, *out, auxMap);

    std::vector<uint32_t> lhsSlotRemap(lhsExpr.expr->inputs.size());
    for (size_t i = 0; i < lhsExpr.expr->inputs.size(); ++i)
        lhsSlotRemap[i] = mergedByName.at(lhsExpr.expr->inputs[i].name);
    std::vector<uint32_t> rhsSlotRemap(rhsExpr.expr->inputs.size());
    for (size_t i = 0; i < rhsExpr.expr->inputs.size(); ++i)
        rhsSlotRemap[i] = mergedByName.at(rhsExpr.expr->inputs[i].name);
    std::vector<uint32_t> auxSlotRemap(auxExpr.expr->inputs.size());
    for (size_t i = 0; i < auxExpr.expr->inputs.size(); ++i)
        auxSlotRemap[i] = mergedByName.at(auxExpr.expr->inputs[i].name);

    remapClonedInputSlots(*lhsExpr.expr, lhsMap, lhsSlotRemap, *out);
    remapClonedInputSlots(*rhsExpr.expr, rhsMap, rhsSlotRemap, *out);
    remapClonedInputSlots(*auxExpr.expr, auxMap, auxSlotRemap, *out);

    ExprNode node{};
    node.op = op;
    node.lhs = newLhsIndex;
    node.rhs = newRhsIndex;
    node.aux = newAuxIndex;

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
Expression Expression::abs() const { return unaryOp(*this, ExprOp::ABS); }
Expression Expression::sqrt() const { return unaryOp(*this, ExprOp::SQRT); }
Expression Expression::sqrt(const Expression& expr) { return unaryOp(expr, ExprOp::SQRT); }

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

Expression Expression::transpose() const { return unaryOp(*this, ExprOp::TRANSPOSE); }

Expression Expression::pow(const Expression& exponent) const { return binaryOp(*this, exponent, ExprOp::POW); }

Expression Expression::matmul(const Expression& lhs,
                              const Expression& rhs,
                              bool transpose_lhs,
                              bool transpose_rhs,
                              Optional<DataType> compute_dtype,
                              Optional<DataType> output_dtype) {
    Expression out = binaryOp(lhs, rhs, ExprOp::MATMUL);
    ExprNode& node = out.expr->nodes[out.nodeIndex];
    node.transpose_lhs = transpose_lhs;
    node.transpose_rhs = transpose_rhs;
    if (compute_dtype.isPresent()) {
        node.compute_dtype = compute_dtype.get();
    }
    if (output_dtype.isPresent()) {
        node.output_dtype = output_dtype.get();
    }
    return out;
}

static uint32_t cloneSubtreeIntoMergedExpression(const Expression& src_expr,
                                                 PhysicalExpression& dst,
                                                 std::unordered_map<uint32_t, uint32_t>& old_to_new,
                                                 std::unordered_map<std::string, uint32_t>& dst_input_slots_by_name) {
    PhysicalExpression src = src_expr.expression();
    return cloneSubtreeWithMergedInputs(src, src.output_node, dst, old_to_new, dst_input_slots_by_name);
}

uint32_t Expression::encodeLowerableGemmScaleExpression(const Expression& scale_expr,
                                                        PhysicalExpression& dst,
                                                        std::unordered_map<std::string, uint32_t>& dst_input_slots_by_name,
                                                        double& scale_fp) {
    PhysicalExpression scale = scale_expr.expression();

    struct SimpleScaleEncodingResult {
        bool success = false;
        uint32_t dynamic_node = UINT32_MAX;
        double constant_scale = 1.0;
    };

    std::function<SimpleScaleEncodingResult(uint32_t)> try_encode_simple = [&](uint32_t node_idx) -> SimpleScaleEncodingResult {
        const ExprNode& src_node = scale.nodes.at(node_idx);
        if (src_node.op == ExprOp::SCALAR_FP) {
            return SimpleScaleEncodingResult{true, UINT32_MAX, src_node.scalar_fp};
        }
        if (src_node.op == ExprOp::INPUT || src_node.op == ExprOp::RUNTIME_SCALAR || src_node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            std::unordered_map<uint32_t, uint32_t> old_to_new;
            uint32_t cloned =
                cloneSubtreeIntoMergedExpression(Expression(scale_expr.expr, node_idx), dst, old_to_new, dst_input_slots_by_name);
            return SimpleScaleEncodingResult{true, cloned, 1.0};
        }
        if (src_node.op == ExprOp::MUL) {
            const SimpleScaleEncodingResult lhs = try_encode_simple(src_node.lhs);
            const SimpleScaleEncodingResult rhs = try_encode_simple(src_node.rhs);
            if (!lhs.success || !rhs.success) {
                return {};
            }
            if (lhs.dynamic_node != UINT32_MAX && rhs.dynamic_node != UINT32_MAX) {
                return {};
            }
            return SimpleScaleEncodingResult{
                true, lhs.dynamic_node != UINT32_MAX ? lhs.dynamic_node : rhs.dynamic_node, lhs.constant_scale * rhs.constant_scale};
        }
        return {};
    };

    const SimpleScaleEncodingResult simple = try_encode_simple(scale.output_node);
    if (simple.success) {
        scale_fp *= simple.constant_scale;
        return simple.dynamic_node;
    }

    std::unordered_map<uint32_t, uint32_t> old_to_new;
    return cloneSubtreeIntoMergedExpression(scale_expr, dst, old_to_new, dst_input_slots_by_name);
}

Expression Expression::gemm(const Expression& lhs,
                            const Expression& rhs,
                            const Expression& addend,
                            double alpha,
                            double beta,
                            bool transpose_lhs,
                            bool transpose_rhs,
                            bool transpose_addend,
                            Optional<DataType> compute_dtype,
                            Optional<DataType> output_dtype) {
    return gemm(lhs,
                rhs,
                addend,
                Expression::constantScalar(alpha),
                Expression::constantScalar(beta),
                transpose_lhs,
                transpose_rhs,
                transpose_addend,
                compute_dtype,
                output_dtype);
}

Expression Expression::gemm(const Expression& lhs,
                            const Expression& rhs,
                            const Expression& addend,
                            const Expression& alpha,
                            const Expression& beta,
                            bool transpose_lhs,
                            bool transpose_rhs,
                            bool transpose_addend,
                            Optional<DataType> compute_dtype,
                            Optional<DataType> output_dtype) {
    if (!lhs.expr || !rhs.expr || !addend.expr || !alpha.expr || !beta.expr) {
        throw std::runtime_error("Cannot build GEMM from empty expressions.");
    }

    auto out = std::make_shared<PhysicalExpression>();
    std::unordered_map<std::string, uint32_t> merged_by_name;

    auto clone_root = [&](const Expression& src_expr) {
        std::unordered_map<uint32_t, uint32_t> old_to_new;
        return cloneSubtreeIntoMergedExpression(src_expr, *out, old_to_new, merged_by_name);
    };

    const uint32_t new_lhs_index = clone_root(lhs);
    const uint32_t new_rhs_index = clone_root(rhs);
    const uint32_t new_aux_index = clone_root(addend);

    ExprNode node{};
    node.op = ExprOp::GEMM;
    node.lhs = new_lhs_index;
    node.rhs = new_rhs_index;
    node.aux = new_aux_index;
    node.alpha_fp = 1.0;
    node.beta_fp = 1.0;
    node.alpha_node = encodeLowerableGemmScaleExpression(alpha, *out, merged_by_name, node.alpha_fp);
    node.beta_node = encodeLowerableGemmScaleExpression(beta, *out, merged_by_name, node.beta_fp);
    node.transpose_lhs = transpose_lhs;
    node.transpose_rhs = transpose_rhs;
    node.transpose_aux = transpose_addend;
    if (compute_dtype.isPresent()) {
        node.compute_dtype = compute_dtype.get();
    }
    if (output_dtype.isPresent()) {
        node.output_dtype = output_dtype.get();
    }

    const uint32_t new_index = static_cast<uint32_t>(out->nodes.size());
    out->nodes.push_back(node);
    out->output_node = new_index;
    return Expression(out, new_index);
}

Expression Expression::conv2d(const Expression& input,
                              const Expression& filter,
                              int32_t stride_h,
                              int32_t stride_w,
                              int32_t pad_h,
                              int32_t pad_w,
                              Optional<DataType> compute_dtype,
                              Optional<DataType> output_dtype) {
    if (stride_h <= 0 || stride_w <= 0) {
        throw std::runtime_error("conv2d stride must be positive.");
    }
    if (pad_h < 0 || pad_w < 0) {
        throw std::runtime_error("conv2d padding must be non-negative.");
    }

    Expression out = binaryOp(input, filter, ExprOp::CONV2D);
    ExprNode& node = out.expr->nodes[out.nodeIndex];
    node.conv_stride_h = stride_h;
    node.conv_stride_w = stride_w;
    node.conv_pad_h = pad_h;
    node.conv_pad_w = pad_w;
    if (compute_dtype.isPresent()) {
        node.compute_dtype = compute_dtype.get();
    }
    if (output_dtype.isPresent()) {
        node.output_dtype = output_dtype.get();
    }
    return out;
}

// Reductions
DataType validate_reduction_compute_type(Optional<DataType> compute_dtype) {
    const DataType requested_compute_dtype = compute_dtype.isPresent() ? compute_dtype.get() : DataType::FP32;
    return toSupportedComputeDType(ExprOp::REDUCE_SUM, requested_compute_dtype);
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

Expression Expression::withDTypes(Optional<DataType> compute_dtype, Optional<DataType> output_dtype) const {
    if (!expr)
        throw std::runtime_error("Cannot override dtypes on an empty expression");
    if (nodeIndex >= expr->nodes.size())
        throw std::runtime_error("Cannot override dtypes on an invalid expression node");

    auto out = std::make_shared<PhysicalExpression>();
    out->inputs = expr->inputs;

    std::unordered_map<uint32_t, uint32_t> oldToNew;
    uint32_t newRootIndex = cloneSubtree(*expr, nodeIndex, *out, oldToNew);
    out->output_node = newRootIndex;

    ExprNode& root = out->nodes[newRootIndex];
    if (compute_dtype.isPresent()) {
        root.compute_dtype = compute_dtype.get();
    }
    if (output_dtype.isPresent()) {
        root.output_dtype = output_dtype.get();
    }

    return Expression(out, newRootIndex);
}

Expression Expression::withComputeDType(DataType compute_dtype) const { return withDTypes(compute_dtype, Optional<DataType>::empty()); }

Expression Expression::withOutputDType(DataType output_dtype) const { return withDTypes(Optional<DataType>::empty(), output_dtype); }

Expression Expression::ln() const { return unaryOp(*this, ExprOp::LN); }
Expression Expression::log2() const { return unaryOp(*this, ExprOp::LOG2); }
Expression Expression::log10() const { return unaryOp(*this, ExprOp::LOG10); }
Expression Expression::log(double base) const {
    if (base <= 0.0f || base == 1.0f) {
        throw std::runtime_error("log base must be positive and not equal to 1, received base = " + std::to_string(base));
    }
    return this->ln() / Expression::constantScalar(std::log(base));
}

Expression Expression::min(const Expression& other) const { return binaryOp(*this, other, ExprOp::MIN); }
Expression Expression::max(const Expression& other) const { return binaryOp(*this, other, ExprOp::MAX); }

// e^x_i
Expression Expression::exp() const { return unaryOp(*this, ExprOp::EXP); }
// 2^x_i
Expression Expression::exp2() const { return unaryOp(*this, ExprOp::EXP2); }
Expression Expression::exp10() const { return unaryOp(*this, ExprOp::EXP10); }
// Can also use Expression::scalar(s).pow(x) for s^x_i

uint32_t PhysicalExpression::getOrCreateInputSlot(const std::string& name, NamedInput::Kind kind) {
    for (const NamedInput& input : inputs) {
        if (input.name == name) {
            if (input.kind != kind) {
                throw std::runtime_error("Input kind mismatch for input: " + name);
            }
            return input.slot;
        }
    }

    const uint32_t slot = static_cast<uint32_t>(inputs.size());
    inputs.push_back(NamedInput{name, slot, kind});
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
