#include "Utilities/TensorMathFusion/CudaSourceEmitter.h"

using namespace std;

using DataType = ThorImplementation::TensorDescriptor::DataType;

namespace ThorImplementation {

static std::string emitScalarFpLiteral(double x, DataType dtype) {
    auto formatFloating = [](double v, int precision) -> std::string {
        std::ostringstream oss;
        oss << std::setprecision(precision) << std::defaultfloat << v;
        std::string s = oss.str();

        // Make sure whole numbers still look like floating-point literals.
        if (s.find('.') == std::string::npos && s.find('e') == std::string::npos && s.find('E') == std::string::npos) {
            s += ".0";
        }

        return s;
    };

    switch (dtype) {
        case DataType::FP8_E4M3:
            return "__nv_cvt_float_to_fp8(" + formatFloating(x, 9) + "f, __NV_SATFINITE, __NV_E4M3)";
        case DataType::FP8_E5M2:
            return "__nv_cvt_float_to_fp8(" + formatFloating(x, 9) + "f, __NV_SATFINITE, __NV_E5M2)";
        case DataType::BF16:
            return "__float2bfloat16(" + formatFloating(x, 9) + "f)";
        case DataType::FP16:
            return "__float2half_rn(" + formatFloating(x, 9) + "f)";
        case DataType::FP32: {
            return formatFloating(x, 9) + "f";
        }
        case DataType::FP64: {
            return formatFloating(x, 17);
        }
        default:
            throw std::runtime_error("emitScalarLiteral: unsupported dtype");
    }
}

string CudaSourceEmitter::emit(const PhysicalExpression& expr, const string& kernel_name, const bool broadcast_support) {
    ostringstream ss;

    if (broadcast_support) {
        ss << "struct BroadcastInputInfo {\n";
        ss << "  unsigned long long strides[10];\n";
        ss << "};\n\n";

        ss << "struct BroadcastInfo {\n";
        ss << "  unsigned int rank;\n";
        ss << "  unsigned int _pad;\n";
        ss << "  unsigned long long numel;\n";
        ss << "  unsigned long long output_strides[10];\n";
        const uint32_t emitted_num_inputs = std::max<uint32_t>(expr.num_inputs, 1);
        ss << "  BroadcastInputInfo inputs[" << emitted_num_inputs << "];\n";
        ss << "};\n\n";
    }

    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";

    for (uint32_t i = 0; i < expr.num_inputs; ++i) {
        ss << "const float* in" << i << ", ";
    }

    if (broadcast_support) {
        ss << "float* out, const BroadcastInfo* broadcast) {\n";
    } else {
        ss << "float* out, unsigned long long numel) {\n";
    }

    ss << "  unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n";

    if (broadcast_support) {
        ss << "  if (idx >= broadcast->numel) return;\n\n";

        ss << "  unsigned long long coord[10];\n";
        ss << "  #pragma unroll\n";
        ss << "  for (unsigned int axis = 0; axis < 10; ++axis) {\n";
        ss << "    coord[axis] = 0ULL;\n";
        ss << "  }\n\n";

        ss << "  unsigned long long remaining = idx;\n";
        ss << "  #pragma unroll\n";
        ss << "  for (unsigned int axis = 0; axis < 10; ++axis) {\n";
        ss << "    if (axis < broadcast->rank) {\n";
        ss << "      const unsigned long long stride = broadcast->output_strides[axis];\n";
        ss << "      coord[axis] = remaining / stride;\n";
        ss << "      remaining = remaining % stride;\n";
        ss << "    }\n";
        ss << "  }\n\n";

        for (uint32_t inputIdx = 0; inputIdx < expr.num_inputs; ++inputIdx) {
            ss << "  unsigned long long in" << inputIdx << "_offset = 0ULL;\n";
            ss << "  #pragma unroll\n";
            ss << "  for (unsigned int axis = 0; axis < 10; ++axis) {\n";
            ss << "    if (axis < broadcast->rank) {\n";
            ss << "      in" << inputIdx << "_offset += coord[axis] * broadcast->inputs[" << inputIdx << "].strides[axis];\n";
            ss << "    }\n";
            ss << "  }\n";
        }

        ss << "\n";
    } else {
        ss << "  if (idx >= numel) return;\n\n";
    }

    for (uint32_t i = 0; i < expr.nodes.size(); ++i) {
        const auto& n = expr.nodes[i];
        switch (n.op) {
            case ExprOp::INPUT:
                if (broadcast_support) {
                    ss << "  float t" << i << " = in" << n.input_index << "[in" << n.input_index << "_offset];\n";
                } else {
                    ss << "  float t" << i << " = in" << n.input_index << "[idx];\n";
                }
                break;
            case ExprOp::SCALAR_FP:
                ss << "  float t" << i << " = " << emitScalarFpLiteral(n.scalar_fp, DataType::FP32) << ";\n";
                break;
            case ExprOp::ADD:
                ss << "  float t" << i << " = " << ref(n.lhs) << " + " << ref(n.rhs) << ";\n";
                break;
            case ExprOp::SUB:
                ss << "  float t" << i << " = " << ref(n.lhs) << " - " << ref(n.rhs) << ";\n";
                break;
            case ExprOp::MUL:
                ss << "  float t" << i << " = " << ref(n.lhs) << " * " << ref(n.rhs) << ";\n";
                break;
            case ExprOp::DIV:
                ss << "  float t" << i << " = " << ref(n.lhs) << " / " << ref(n.rhs) << ";\n";
                break;
            case ExprOp::NEG:
                ss << "  float t" << i << " = -" << ref(n.lhs) << ";\n";
                break;
            case ExprOp::EXP:
                ss << "  float t" << i << " = expf(" << ref(n.lhs) << ");\n";
                break;
            case ExprOp::EXP2:
                ss << "  float t" << i << " = exp2f(" << ref(n.lhs) << ");\n";
                break;
            case ExprOp::EXP10:
                ss << "  float t" << i << " = exp10f(" << ref(n.lhs) << ");\n";
                break;
            case ExprOp::LN:
                ss << "  float t" << i << " = logf(" << ref(n.lhs) << ");\n";
                break;
            case ExprOp::LOG2:
                ss << "  float t" << i << " = log2f(" << ref(n.lhs) << ");\n";
                break;
            case ExprOp::LOG10:
                ss << "  float t" << i << " = log10f(" << ref(n.lhs) << ");\n";
                break;
            case ExprOp::SQRT:
                ss << "  float t" << i << " = sqrtf(" << ref(n.lhs) << ");\n";
                break;
            case ExprOp::POW:
                ss << "  float t" << i << " = powf(" << ref(n.lhs) << ", " << ref(n.rhs) << ");\n";
                break;
            case ExprOp::MIN:
                ss << "  float t" << i << " = fminf(" << ref(n.lhs) << ", " << ref(n.rhs) << ");\n";
                break;
            case ExprOp::MAX:
                ss << "  float t" << i << " = fmaxf(" << ref(n.lhs) << ", " << ref(n.rhs) << ");\n";
                break;
            default:
                throw runtime_error("Unsupported op in emitter: " + to_string((int32_t)n.op) + "\n" + ss.str());
        }
    }

    ss << "\n  out[idx] = " << ref(expr.output_node) << ";\n";
    ss << "}\n";

    // printf("%s\n", ss.str().c_str());
    return ss.str();
}

string CudaSourceEmitter::ref(uint32_t idx) {
    // temporaries are tN
    return "t" + to_string(idx);
}

}  // namespace ThorImplementation
