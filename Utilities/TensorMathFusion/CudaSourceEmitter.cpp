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

    if (dtype == DataType::FP64)
        return formatFloating(x, 17);
    return formatFloating(x, 9) + "f";
}

string CudaSourceEmitter::emit(const PhysicalExpression& expr, DataType dtype, const string& kernel_name, const bool broadcast_support) {
    ostringstream ss;

    if (dtype != DataType::FP32 && dtype != DataType::FP16 && dtype != DataType::BF16 && dtype != DataType::FP8_E4M3 &&
        dtype != DataType::FP8_E5M2) {
        throw runtime_error("Unsupported data type in fusion emitter: " + TensorDescriptor::getElementTypeName(dtype) +
                            " - Only floating point types of max precision fp32 are supported.");
    }

    string fp_type;
    if (dtype == DataType::FP32) {
        fp_type = "float";
    } else if (dtype == DataType::FP16) {
        fp_type = "half";
        ss << "#include <cuda_fp16.h>\n";
    } else {  // if (dtype == DataType::BF16) {
        fp_type = "__nv_bfloat16";
        ss << "#include <cuda_bf16.h>\n";
    }

    string inout_dtype = fp_type;
    if (dtype == DataType::FP8_E4M3) {
        inout_dtype = "__nv_fp8_e4m3";
        ss << "#include <cuda_fp8.h>\n";
    } else if (dtype == DataType::FP8_E5M2) {
        inout_dtype = "__nv_fp8_e5m2";
        ss << "#include <cuda_fp8.h>\n";
    }

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
        ss << "const " << inout_dtype << "* in" << i << ", ";
    }

    if (broadcast_support) {
        ss << inout_dtype << "* out, const BroadcastInfo* broadcast) {\n";
    } else {
        ss << inout_dtype << "* out, unsigned long long numel) {\n";
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
                    ss << "  " << fp_type << " t" << i << " = " << fp_type << "(in" << n.input_index << "[in" << n.input_index
                       << "_offset]);\n";
                } else {
                    ss << "  " << fp_type << " t" << i << " = " << fp_type << "(in" << n.input_index << "[idx]);\n";
                }
                break;
            case ExprOp::SCALAR_FP:
                ss << "  " << fp_type << " t" << i << "(" << emitScalarFpLiteral(n.scalar_fp, dtype) << ");\n";
                break;
            case ExprOp::ADD:
                ss << "  " << fp_type << " t" << i << " = " << ref(n.lhs) << " + " << ref(n.rhs) << ";\n";
                break;
            case ExprOp::SUB:
                ss << "  " << fp_type << " t" << i << " = " << ref(n.lhs) << " - " << ref(n.rhs) << ";\n";
                break;
            case ExprOp::MUL:
                ss << "  " << fp_type << " t" << i << " = " << ref(n.lhs) << " * " << ref(n.rhs) << ";\n";
                break;
            case ExprOp::DIV:
                ss << "  " << fp_type << " t" << i << " = " << ref(n.lhs) << " / " << ref(n.rhs) << ";\n";
                break;
            case ExprOp::NEG:
                ss << "  " << fp_type << " t" << i << " = -" << ref(n.lhs) << ";\n";
                break;
            case ExprOp::EXP:
                ss << "  " << fp_type << " t" << i << " = expf(" << ref(n.lhs) << ");\n";
                break;
            case ExprOp::EXP2:
                ss << "  " << fp_type << " t" << i << " = exp2f(" << ref(n.lhs) << ");\n";
                break;
            case ExprOp::EXP10:
                ss << "  " << fp_type << " t" << i << " = exp10f(" << ref(n.lhs) << ");\n";
                break;
            case ExprOp::LN:
                ss << "  " << fp_type << " t" << i << " = logf(" << ref(n.lhs) << ");\n";
                break;
            case ExprOp::LOG2:
                ss << "  " << fp_type << " t" << i << " = log2f(" << ref(n.lhs) << ");\n";
                break;
            case ExprOp::LOG10:
                ss << "  " << fp_type << " t" << i << " = log10f(" << ref(n.lhs) << ");\n";
                break;
            case ExprOp::SQRT:
                ss << "  " << fp_type << " t" << i << " = sqrtf(" << ref(n.lhs) << ");\n";
                break;
            case ExprOp::POW:
                ss << "  " << fp_type << " t" << i << " = powf(" << ref(n.lhs) << ", " << ref(n.rhs) << ");\n";
                break;
            case ExprOp::MIN:
                ss << "  " << fp_type << " t" << i << " = fminf(" << ref(n.lhs) << ", " << ref(n.rhs) << ");\n";
                break;
            case ExprOp::MAX:
                ss << "  " << fp_type << " t" << i << " = fmaxf(" << ref(n.lhs) << ", " << ref(n.rhs) << ");\n";
                break;
            default:
                throw runtime_error("Unsupported op in emitter: " + to_string((int32_t)n.op) + "\n" + ss.str());
        }
    }

    if (fp_type == inout_dtype)
        ss << "\n  out[idx] = " << ref(expr.output_node) << ";\n";
    else
        ss << "\n  out[idx] = " << inout_dtype << "(" << ref(expr.output_node) << ");\n";
    ss << "}\n";

    // printf("%s\n", ss.str().c_str());
    return ss.str();
}

string CudaSourceEmitter::ref(uint32_t idx) {
    // temporaries are tN
    return "t" + to_string(idx);
}

}  // namespace ThorImplementation
