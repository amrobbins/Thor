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
    if (dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2) {
        if (broadcast_support == false) {
            return emitVector2Flat(expr, dtype, kernel_name);
        } else {
            return emitVector2Broadcast(expr, dtype, kernel_name);
        }
    }

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
        ss << R"DEVICE(
            struct BroadcastInfoHeader {
              unsigned int rank;
              unsigned int num_inputs;
              unsigned long long numel;
            };

            __device__ __forceinline__
            const unsigned long long* broadcast_output_strides(const BroadcastInfoHeader* broadcast) {
              return reinterpret_cast<const unsigned long long*>(
                  reinterpret_cast<const char*>(broadcast) + sizeof(BroadcastInfoHeader));
            }

            __device__ __forceinline__
            const unsigned long long* broadcast_input_strides(const BroadcastInfoHeader* broadcast) {
              return broadcast_output_strides(broadcast) + broadcast->rank;
            }

        )DEVICE";
    }

    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";

    for (uint32_t i = 0; i < expr.numInputs(); ++i) {
        ss << "const " << inout_dtype << "* in" << i << ", ";
    }

    if (broadcast_support) {
        ss << inout_dtype << "* out, const BroadcastInfoHeader* broadcast) {\n";
    } else {
        ss << inout_dtype << "* out, unsigned long long numel) {\n";
    }

    ss << "  unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n";

    if (broadcast_support) {
        ss << R"DEVICE(
          if (idx >= broadcast->numel) return;

          const unsigned int rank = broadcast->rank;
          const unsigned long long* output_strides = broadcast_output_strides(broadcast);
          const unsigned long long* input_strides = broadcast_input_strides(broadcast);

        )DEVICE";

        for (uint32_t inputIdx = 0; inputIdx < expr.numInputs(); ++inputIdx) {
            ss << "  unsigned long long in" << inputIdx << "_offset = 0ULL;\n";
        }

        ss << R"DEVICE(
          unsigned long long remaining = idx;
          for (unsigned int axis = 0; axis < rank; ++axis) {
            const unsigned long long stride = output_strides[axis];
            const unsigned long long c = remaining / stride;
            remaining %= stride;
        )DEVICE";

        for (uint32_t inputIdx = 0; inputIdx < expr.numInputs(); ++inputIdx) {
            if (inputIdx == 0)
                ss << "    in" << inputIdx << "_offset += c * input_strides[axis];\n";
            else if (inputIdx == 1)
                ss << "    in" << inputIdx << "_offset += c * input_strides[rank + axis];\n";
            else
                ss << "    in" << inputIdx << "_offset += c * input_strides[" << inputIdx << " * rank + axis];\n";
        }

        ss << R"DEVICE(
          }

        )DEVICE";
    } else {
        ss << "  if (idx >= numel) return;\n\n";
    }

    for (uint32_t i = 0; i < expr.nodes.size(); ++i) {
        const auto& n = expr.nodes[i];
        switch (n.op) {
            case ExprOp::INPUT:
                if (broadcast_support) {
                    ss << "  " << fp_type << " t" << i << "(in" << n.input_slot << "[in" << n.input_slot << "_offset]);\n";
                } else {
                    ss << "  " << fp_type << " t" << i << "(in" << n.input_slot << "[idx]);\n";
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

static string vector_compute_conversion(string storage_dtype_vector, string variable) {
    if (storage_dtype_vector == "half2")
        return variable;
    else if (storage_dtype_vector == "__nv_bfloat162")
        return variable;
    else if (storage_dtype_vector == "__nv_fp8x2_e4m3")
        return "static_cast<__half2>(" + variable + ")";
    else if (storage_dtype_vector == "__nv_fp8x2_e5m2")
        return "static_cast<__half2>(" + variable + ")";
    throw runtime_error("Unsupported vector storage dtype in vector_compute_conversion: " + storage_dtype_vector);
}

static string vector_storage_conversion(string storage_dtype_vector, string variable) {
    if (storage_dtype_vector == "half2")
        return variable;
    else if (storage_dtype_vector == "__nv_bfloat162")
        return variable;
    else if (storage_dtype_vector == "__nv_fp8x2_e4m3")
        return "__nv_fp8x2_e4m3(" + variable + ")";
    else if (storage_dtype_vector == "__nv_fp8x2_e5m2")
        return "__nv_fp8x2_e5m2(" + variable + ")";
    throw runtime_error("Unsupported vector storage dtype in vector_compute_conversion: " + storage_dtype_vector);
}

static string emitVector2ScalarLiteral(double x, DataType dtype) {
    const string lit = emitScalarFpLiteral(x, dtype);

    if (dtype == DataType::FP16 || dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2) {
        return "__halves2half2(__float2half_rn(" + lit + "), __float2half_rn(" + lit + "))";
    } else if (dtype == DataType::BF16) {
        return "__floats2bfloat162_rn(" + lit + ", " + lit + ")";
    }
    throw runtime_error("Unsupported dtype in emitVector2ScalarLiteral.");
}

static string emitVector2Add(const string& a, const string& b) { return "__hadd2(" + a + ", " + b + ")"; }
static string emitVector2Sub(const string& a, const string& b) { return "__hsub2(" + a + ", " + b + ")"; }
static string emitVector2Mul(const string& a, const string& b) { return "__hmul2(" + a + ", " + b + ")"; }
static string emitVector2Div(const string& a, const string& b) { return "__h2div(" + a + ", " + b + ")"; }
static string emitVector2Neg(const string& x, DataType dtype) { return "__hneg2(" + x + ")"; }
static string emitVector2Exp(const string& x, DataType dtype) { return "h2exp(" + x + ")"; }
static string emitVector2Exp2(const string& x, DataType dtype) { return "h2exp2(" + x + ")"; }
static string emitVector2Exp10(const string& x, DataType dtype) { return "h2exp10(" + x + ")"; }
static string emitVector2Ln(const string& x, DataType dtype) { return "h2log(" + x + ")"; }
static string emitVector2Log2(const string& x, DataType dtype) { return "h2log2(" + x + ")"; }
static string emitVector2Log10(const string& x, DataType dtype) { return "h2log10(" + x + ")"; }
static string emitVector2Sqrt(const string& x, DataType dtype) { return "h2sqrt(" + x + ")"; }
static string emitVector2Pow(const string& a, const string& b, DataType dtype) {
    if (dtype == DataType::BF16)
        return "__floats2bfloat162_rn( powf(" + a + ".x, " + b + ".x), powf(" + a + ".y, " + b + ".y) )";
    else
        return "__floats2half2_rn( powf(" + a + ".x, " + b + ".x), powf(" + a + ".y, " + b + ".y) )";
}
static string emitVector2Min(const string& a, const string& b) { return "__hmin2(" + a + ", " + b + ")"; }
static string emitVector2Max(const string& a, const string& b) { return "__hmax2(" + a + ", " + b + ")"; }

string CudaSourceEmitter::emitVector2Flat(const PhysicalExpression& expr, DataType dtype, const string& kernel_name) {
    ostringstream ss;

    if (dtype != DataType::FP16 && dtype != DataType::BF16 && dtype != DataType::FP8_E4M3 && dtype != DataType::FP8_E5M2) {
        throw runtime_error("Unsupported data type in vectorized fusion emitter: " + TensorDescriptor::getElementTypeName(dtype) +
                            " - Only floating point types of max precision fp16 are supported.");
    }

    string compute_dtype;
    string compute_dtype_vector;
    if (dtype == DataType::BF16) {
        compute_dtype = "__nv_bfloat16";
        compute_dtype_vector = "__nv_bfloat162";
        ss << "#include <cuda_bf16.h>\n";
    } else {
        compute_dtype = "half";
        compute_dtype_vector = "half2";
        ss << "#include <cuda_fp16.h>\n";
    }

    string storage_dtype_vector = compute_dtype_vector;
    if (dtype == DataType::FP8_E4M3) {
        storage_dtype_vector = "__nv_fp8x2_e4m3";
        ss << "#include <cuda_fp8.h>\n";
    } else if (dtype == DataType::FP8_E5M2) {
        storage_dtype_vector = "__nv_fp8x2_e5m2";
        ss << "#include <cuda_fp8.h>\n";
    }

    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";

    for (uint32_t i = 0; i < expr.numInputs(); ++i) {
        ss << "const " << storage_dtype_vector << "* in" << i << ", ";
    }

    ss << storage_dtype_vector << "* out, unsigned long long numel) {\n";

    ss << "  unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "  if (idx >= (numel >> 1)) return;\n\n";

    for (uint32_t i = 0; i < expr.nodes.size(); ++i) {
        const auto& n = expr.nodes[i];
        switch (n.op) {
            case ExprOp::INPUT: {
                string variable = "in" + to_string(n.input_slot) + "[idx]";
                ss << "  " << compute_dtype_vector << " t" << i << " = " << vector_compute_conversion(storage_dtype_vector, variable)
                   << ";\n";
                break;
            }
            case ExprOp::SCALAR_FP:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2ScalarLiteral(n.scalar_fp, dtype) << ";\n";
                break;
            case ExprOp::ADD:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Add(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            case ExprOp::SUB:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Sub(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            case ExprOp::MUL:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Mul(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            case ExprOp::DIV:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Div(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            case ExprOp::NEG:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Neg(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::EXP:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Exp(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::EXP2:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Exp2(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::EXP10:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Exp10(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::LN:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Ln(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::LOG2:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Log2(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::LOG10:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Log10(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::SQRT:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Sqrt(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::POW:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Pow(ref(n.lhs), ref(n.rhs), dtype) << ";\n";
                break;
            case ExprOp::MIN:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Min(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            case ExprOp::MAX:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Max(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            default:
                throw runtime_error("Unsupported op in vectorized emitter: " + to_string((int32_t)n.op) + "\n" + ss.str());
        }
    }

    if (compute_dtype_vector == storage_dtype_vector)
        ss << "\n  out[idx] = " << ref(expr.output_node) << ";\n";
    else
        ss << "\n  out[idx] = " << vector_storage_conversion(storage_dtype_vector, ref(expr.output_node)) << ";\n";
    ss << "}\n";

    // printf("%s\n", ss.str().c_str());
    return ss.str();
}

static string emitVector2BroadcastPackLoad(const string& storage_dtype, const string& variable0, const string& variable1) {
    if (storage_dtype == "half") {
        return "__halves2half2(" + variable0 + ", " + variable1 + ")";
    } else if (storage_dtype == "__nv_bfloat16") {
        return "__halves2bfloat162(" + variable0 + ", " + variable1 + ")";
    } else if (storage_dtype == "__nv_fp8_e4m3") {
        return "__halves2half2(static_cast<half>(" + variable0 + "), static_cast<half>(" + variable1 + "))";
    } else if (storage_dtype == "__nv_fp8_e5m2") {
        return "__halves2half2(static_cast<half>(" + variable0 + "), static_cast<half>(" + variable1 + "))";
    }

    throw runtime_error("Unsupported scalar storage dtype in emitVector2BroadcastPackLoad: " + storage_dtype);
}

string CudaSourceEmitter::emitVector2Broadcast(const PhysicalExpression& expr, DataType dtype, const string& kernel_name) {
    ostringstream ss;

    if (dtype != DataType::FP16 && dtype != DataType::BF16 && dtype != DataType::FP8_E4M3 && dtype != DataType::FP8_E5M2) {
        throw runtime_error("Unsupported data type in vectorized broadcast fusion emitter: " + TensorDescriptor::getElementTypeName(dtype) +
                            " - Only floating point types of max precision fp16 are supported.");
    }

    string compute_dtype;
    string compute_dtype_vector;
    string storage_dtype;
    string storage_dtype_vector;

    if (dtype == DataType::BF16) {
        compute_dtype = "__nv_bfloat16";
        compute_dtype_vector = "__nv_bfloat162";
        storage_dtype = "__nv_bfloat16";
        storage_dtype_vector = "__nv_bfloat162";
        ss << "#include <cuda_bf16.h>\n";
    } else if (dtype == DataType::FP16) {
        compute_dtype = "half";
        compute_dtype_vector = "half2";
        storage_dtype = "half";
        storage_dtype_vector = "half2";
        ss << "#include <cuda_fp16.h>\n";
    } else if (dtype == DataType::FP8_E4M3) {
        compute_dtype = "half";
        compute_dtype_vector = "half2";
        storage_dtype = "__nv_fp8_e4m3";
        storage_dtype_vector = "__nv_fp8x2_e4m3";
        ss << "#include <cuda_fp16.h>\n";
        ss << "#include <cuda_fp8.h>\n";
    } else {  // FP8_E5M2
        compute_dtype = "half";
        compute_dtype_vector = "half2";
        storage_dtype = "__nv_fp8_e5m2";
        storage_dtype_vector = "__nv_fp8x2_e5m2";
        ss << "#include <cuda_fp16.h>\n";
        ss << "#include <cuda_fp8.h>\n";
    }

    ss << R"DEVICE(
struct BroadcastInfoHeader {
  unsigned int rank;
  unsigned int num_inputs;
  unsigned long long numel;
};

__device__ __forceinline__
const unsigned long long* broadcast_output_strides(const BroadcastInfoHeader* broadcast) {
  return reinterpret_cast<const unsigned long long*>(
      reinterpret_cast<const char*>(broadcast) + sizeof(BroadcastInfoHeader));
}

__device__ __forceinline__
const unsigned long long* broadcast_input_strides(const BroadcastInfoHeader* broadcast) {
  return broadcast_output_strides(broadcast) + broadcast->rank;
}
)DEVICE";

    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";

    for (uint32_t i = 0; i < expr.numInputs(); ++i) {
        ss << "const " << storage_dtype << "* in" << i << ", ";
    }

    ss << storage_dtype_vector << "* out, const BroadcastInfoHeader* broadcast) {\n";

    ss << "  unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "  unsigned long long idx0 = idx << 1;\n";
    ss << "  if (idx0 >= broadcast->numel) return;\n";
    ss << "  unsigned long long idx1 = idx0 + 1;\n\n";

    ss << "  const unsigned int rank = broadcast->rank;\n";
    ss << "  const unsigned long long* output_strides = broadcast_output_strides(broadcast);\n";
    ss << "  const unsigned long long* input_strides = broadcast_input_strides(broadcast);\n\n";

    for (uint32_t inputIdx = 0; inputIdx < expr.numInputs(); ++inputIdx) {
        ss << "  unsigned long long in" << inputIdx << "_offset0 = 0ULL;\n";
        ss << "  unsigned long long in" << inputIdx << "_offset1 = 0ULL;\n";
    }
    ss << "\n";

    ss << "  {\n";
    ss << "    unsigned long long remaining = idx0;\n";
    ss << "    for (unsigned int axis = 0; axis < rank; ++axis) {\n";
    ss << "      const unsigned long long stride = output_strides[axis];\n";
    ss << "      const unsigned long long c = remaining / stride;\n";
    ss << "      remaining %= stride;\n";
    for (uint32_t inputIdx = 0; inputIdx < expr.numInputs(); ++inputIdx) {
        if (inputIdx == 0)
            ss << "      in" << inputIdx << "_offset0 += c * input_strides[axis];\n";
        else if (inputIdx == 1)
            ss << "      in" << inputIdx << "_offset0 += c * input_strides[rank + axis];\n";
        else
            ss << "      in" << inputIdx << "_offset0 += c * input_strides[" << inputIdx << " * rank + axis];\n";
    }
    ss << "    }\n";
    ss << "  }\n\n";

    ss << "  const bool have_second_lane = (idx1 < broadcast->numel);\n";
    ss << "  if (have_second_lane) {\n";
    ss << "    unsigned long long remaining = idx1;\n";
    ss << "    for (unsigned int axis = 0; axis < rank; ++axis) {\n";
    ss << "      const unsigned long long stride = output_strides[axis];\n";
    ss << "      const unsigned long long c = remaining / stride;\n";
    ss << "      remaining %= stride;\n";
    for (uint32_t inputIdx = 0; inputIdx < expr.numInputs(); ++inputIdx) {
        if (inputIdx == 0)
            ss << "      in" << inputIdx << "_offset1 += c * input_strides[axis];\n";
        else if (inputIdx == 1)
            ss << "      in" << inputIdx << "_offset1 += c * input_strides[rank + axis];\n";
        else
            ss << "      in" << inputIdx << "_offset1 += c * input_strides[" << inputIdx << " * rank + axis];\n";
    }
    ss << "    }\n";
    ss << "  } else {\n";
    for (uint32_t inputIdx = 0; inputIdx < expr.numInputs(); ++inputIdx) {
        ss << "    in" << inputIdx << "_offset1 = in" << inputIdx << "_offset0;\n";
    }
    ss << "  }\n\n";

    for (uint32_t i = 0; i < expr.nodes.size(); ++i) {
        const auto& n = expr.nodes[i];
        switch (n.op) {
            case ExprOp::INPUT: {
                const string var0 = "in" + to_string(n.input_slot) + "[in" + to_string(n.input_slot) + "_offset0]";
                const string var1 = "in" + to_string(n.input_slot) + "[in" + to_string(n.input_slot) + "_offset1]";
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2BroadcastPackLoad(storage_dtype, var0, var1)
                   << ";\n";
                break;
            }
            case ExprOp::SCALAR_FP:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2ScalarLiteral(n.scalar_fp, dtype) << ";\n";
                break;
            case ExprOp::ADD:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Add(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            case ExprOp::SUB:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Sub(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            case ExprOp::MUL:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Mul(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            case ExprOp::DIV:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Div(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            case ExprOp::NEG:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Neg(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::EXP:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Exp(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::EXP2:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Exp2(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::EXP10:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Exp10(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::LN:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Ln(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::LOG2:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Log2(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::LOG10:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Log10(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::SQRT:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Sqrt(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::POW:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Pow(ref(n.lhs), ref(n.rhs), dtype) << ";\n";
                break;
            case ExprOp::MIN:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Min(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            case ExprOp::MAX:
                ss << "  " << compute_dtype_vector << " t" << i << " = " << emitVector2Max(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            default:
                throw runtime_error("Unsupported op in vectorized broadcast emitter: " + to_string((int32_t)n.op) + "\n" + ss.str());
        }
    }

    ss << "\n";
    if (storage_dtype_vector == "half2" || storage_dtype_vector == "__nv_bfloat162" || storage_dtype_vector == "__nv_fp8x2_e4m3" ||
        storage_dtype_vector == "__nv_fp8x2_e5m2") {
        ss << "  out[idx] = " << vector_storage_conversion(storage_dtype_vector, ref(expr.output_node)) << ";\n";
    } else {
        throw runtime_error("Unsupported vector output storage type in emitVector2Broadcast.");
    }

    ss << "}\n";

    // printf("%s\n", ss.str().c_str());
    return ss.str();
}

}  // namespace ThorImplementation
