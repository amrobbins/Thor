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

string CudaSourceEmitter::emit(const PhysicalExpression& expr, const string& kernel_name) {
    ostringstream ss;
    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";

    for (uint32_t i = 0; i < expr.num_inputs; ++i) {
        ss << "const float* in" << i << ", ";
    }
    ss << "float* out, unsigned long long numel) {\n";
    ss << "  unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "  if (idx >= numel) return;\n\n";

    for (uint32_t i = 0; i < expr.nodes.size(); ++i) {
        const auto& n = expr.nodes[i];
        switch (n.op) {
            case ExprOp::INPUT:
                ss << "  float t" << i << " = in" << n.input_index << "[idx];\n";
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
    printf("%s\n", ss.str().c_str());
    return ss.str();
}

string CudaSourceEmitter::ref(uint32_t idx) {
    // temporaries are tN
    return "t" + to_string(idx);
}

}  // namespace ThorImplementation
