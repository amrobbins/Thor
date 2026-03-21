#include "Utilities/TensorMathFusion/CudaSourceEmitter.h"
#include "Utilities/TensorMathFusion/EquationCompiler.h"

using namespace std;

using DataType = ThorImplementation::TensorDescriptor::DataType;

namespace ThorImplementation {

constexpr bool PRINT_KERNELS = false;

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

static bool isPowerOfTwo(uint64_t x) { return x != 0 && (x & (x - 1)) == 0; }

static uint32_t log2Exact(uint64_t x) {
    uint32_t s = 0;
    while (x > 1) {
        x >>= 1;
        ++s;
    }
    return s;
}

static std::string emitVector2BroadcastNativeLoad(const std::string& storage_dtype_vector,
                                                  const std::string& base_ptr,
                                                  const std::string& scalar_offset0) {
    const std::string vec_expr = "reinterpret_cast<const " + storage_dtype_vector + "*>(" + base_ptr + " + " + scalar_offset0 + ")[0]";
    return vector_compute_conversion(storage_dtype_vector, vec_expr);
}

static void emitSpecializedBroadcastOffsetMath(std::ostringstream& ss,
                                               const SpecializedBroadcastGroup& group,
                                               const std::vector<size_t>& used_input_indices,
                                               const std::string& idx_expr,
                                               const std::string& offset_suffix,
                                               const std::string& indent) {
    if (used_input_indices.empty()) {
        return;
    }

    for (size_t axis_i = 0; axis_i < group.active_axes.size(); ++axis_i) {
        const SpecializedBroadcastAxis& axis = group.active_axes[axis_i];

        const std::string coord = "c" + (offset_suffix.empty() ? std::string() : offset_suffix) + "_" + std::to_string(axis_i);

        std::string base_expr;
        if (axis.output_stride == 1) {
            base_expr = idx_expr;
        } else if (isPowerOfTwo(axis.output_stride)) {
            base_expr = "(" + idx_expr + " >> " + std::to_string(log2Exact(axis.output_stride)) + ")";
        } else {
            base_expr = "(" + idx_expr + " / " + std::to_string(axis.output_stride) + "ULL)";
        }

        std::string coord_expr;
        if (axis.dim == 1) {
            coord_expr = "0ULL";
        } else if (isPowerOfTwo(axis.dim)) {
            coord_expr = "(" + base_expr + " & " + std::to_string(axis.dim - 1) + "ULL)";
        } else {
            coord_expr = "(" + base_expr + " % " + std::to_string(axis.dim) + "ULL)";
        }

        ss << indent << "const unsigned long long " << coord << " = " << coord_expr << ";\n";

        for (size_t used_i : used_input_indices) {
            const uint64_t stride = axis.input_strides[used_i];
            if (stride == 0) {
                continue;
            }

            const uint32_t input_slot = group.used_input_slots[used_i];

            if (stride == 1) {
                ss << indent << "in" << input_slot << "_offset" << offset_suffix << " += " << coord << ";\n";
            } else if (isPowerOfTwo(stride)) {
                ss << indent << "in" << input_slot << "_offset" << offset_suffix << " += (" << coord << " << "
                   << std::to_string(log2Exact(stride)) << ");\n";
            } else {
                ss << indent << "in" << input_slot << "_offset" << offset_suffix << " += " << coord << " * " << stride << "ULL;\n";
            }
        }
    }
}

static void collectRequiredNodes(const PhysicalExpression& expr, uint32_t node_idx, std::unordered_set<uint32_t>& required) {
    if (!required.insert(node_idx).second) {
        return;
    }

    const ExprNode& node = expr.nodes[node_idx];
    if (Expression::isLeafOp(node.op)) {
        return;
    }

    collectRequiredNodes(expr, node.lhs, required);
    if (Expression::isBinaryOp(node.op)) {
        collectRequiredNodes(expr, node.rhs, required);
    }
}

static std::vector<uint32_t> orderedRequiredNodesForGroup(const CompiledExecutionStage& stage,
                                                          const std::vector<uint32_t>& output_indices) {
    std::unordered_set<uint32_t> required;
    for (uint32_t out_idx : output_indices) {
        if (out_idx >= stage.outputs.size()) {
            throw std::runtime_error("orderedRequiredNodesForGroup output index out of range.");
        }
        collectRequiredNodes(stage.expr, stage.outputs[out_idx].local_node_idx, required);
    }

    std::vector<uint32_t> ordered;
    ordered.reserve(required.size());
    for (uint32_t i = 0; i < stage.expr.nodes.size(); ++i) {
        if (required.contains(i)) {
            ordered.push_back(i);
        }
    }
    return ordered;
}

static std::vector<uint32_t> usedInputSlotsForNodes(const PhysicalExpression& expr, const std::vector<uint32_t>& node_indices) {
    std::unordered_set<uint32_t> slots;
    for (uint32_t node_idx : node_indices) {
        const ExprNode& node = expr.nodes[node_idx];
        if (node.op == ExprOp::INPUT) {
            slots.insert(node.input_slot);
        }
    }

    std::vector<uint32_t> ordered(slots.begin(), slots.end());
    std::sort(ordered.begin(), ordered.end());
    return ordered;
}

void CudaSourceEmitter::emitScalarNode(std::ostringstream& ss,
                                       const PhysicalExpression& expr,
                                       uint32_t node_idx,
                                       DataType dtype,
                                       const std::string& compute_type,
                                       bool broadcast_support) {
    const auto& n = expr.nodes[node_idx];

    switch (n.op) {
        case ExprOp::INPUT:
            if (broadcast_support) {
                ss << "    " << compute_type << " t" << node_idx << "(in" << n.input_slot << "[in" << n.input_slot << "_offset]);\n";
            } else {
                ss << "    " << compute_type << " t" << node_idx << "(in" << n.input_slot << "[idx]);\n";
            }
            break;
        case ExprOp::SCALAR_FP:
            ss << "    " << compute_type << " t" << node_idx << "(" << emitScalarFpLiteral(n.scalar_fp, dtype) << ");\n";
            break;
        case ExprOp::ADD:
            ss << "    " << compute_type << " t" << node_idx << " = " << CudaSourceEmitter::ref(n.lhs) << " + "
               << CudaSourceEmitter::ref(n.rhs) << ";\n";
            break;
        case ExprOp::SUB:
            ss << "    " << compute_type << " t" << node_idx << " = " << CudaSourceEmitter::ref(n.lhs) << " - "
               << CudaSourceEmitter::ref(n.rhs) << ";\n";
            break;
        case ExprOp::MUL:
            ss << "    " << compute_type << " t" << node_idx << " = " << CudaSourceEmitter::ref(n.lhs) << " * "
               << CudaSourceEmitter::ref(n.rhs) << ";\n";
            break;
        case ExprOp::DIV:
            ss << "    " << compute_type << " t" << node_idx << " = " << CudaSourceEmitter::ref(n.lhs) << " / "
               << CudaSourceEmitter::ref(n.rhs) << ";\n";
            break;
        case ExprOp::NEG:
            ss << "    " << compute_type << " t" << node_idx << " = -" << CudaSourceEmitter::ref(n.lhs) << ";\n";
            break;
        case ExprOp::EXP:
            ss << "    " << compute_type << " t" << node_idx << " = expf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
            break;
        case ExprOp::EXP2:
            ss << "    " << compute_type << " t" << node_idx << " = exp2f(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
            break;
        case ExprOp::EXP10:
            ss << "    " << compute_type << " t" << node_idx << " = exp10f(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
            break;
        case ExprOp::LN:
            ss << "    " << compute_type << " t" << node_idx << " = logf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
            break;
        case ExprOp::LOG2:
            ss << "    " << compute_type << " t" << node_idx << " = log2f(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
            break;
        case ExprOp::LOG10:
            ss << "    " << compute_type << " t" << node_idx << " = log10f(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
            break;
        case ExprOp::SQRT:
            ss << "    " << compute_type << " t" << node_idx << " = sqrtf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
            break;
        case ExprOp::POW:
            ss << "    " << compute_type << " t" << node_idx << " = powf(" << CudaSourceEmitter::ref(n.lhs) << ", "
               << CudaSourceEmitter::ref(n.rhs) << ");\n";
            break;
        case ExprOp::MIN:
            ss << "    " << compute_type << " t" << node_idx << " = fminf(" << CudaSourceEmitter::ref(n.lhs) << ", "
               << CudaSourceEmitter::ref(n.rhs) << ");\n";
            break;
        case ExprOp::MAX:
            ss << "    " << compute_type << " t" << node_idx << " = fmaxf(" << CudaSourceEmitter::ref(n.lhs) << ", "
               << CudaSourceEmitter::ref(n.rhs) << ");\n";
            break;
        default:
            throw runtime_error("Unsupported op in fused stage emitter: " + to_string((int32_t)n.op));
    }
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
            remaining -= c * stride;
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

    if (PRINT_KERNELS) {
        printf("%s\n", ss.str().c_str());
    }

    return ss.str();
}

string CudaSourceEmitter::ref(uint32_t idx) {
    // temporaries are tN
    return "t" + to_string(idx);
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

    if (PRINT_KERNELS) {
        printf("%s\n", ss.str().c_str());
    }

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
    ss << "      remaining -= c * stride;\n";
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
    ss << "      remaining -= c * stride;\n";
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

    if (PRINT_KERNELS) {
        printf("%s\n", ss.str().c_str());
    }

    return ss.str();
}

static bool stageUsesBroadcast(const PhysicalExecutionStage& stage) {
    for (const CompiledStageOutput& output : stage.outputs) {
        uint32_t node_idx = output.local_node_idx;
        if (node_idx >= stage.expr.nodes.size()) {
            throw runtime_error("Stage output local_node_idx out of range.");
        }

        // For now, use the conservative rule:
        // if any INPUT exists, we still let launch/runtime decide metadata.
        // This helper is just a placeholder if you later want per-stage specialization.
    }

    return false;
}

static string scalarComputeType(DataType dtype, std::ostringstream& ss) {
    if (dtype == DataType::FP32) {
        return "float";
    }
    if (dtype == DataType::FP16) {
        ss << "#include <cuda_fp16.h>\n";
        return "half";
    }
    if (dtype == DataType::BF16) {
        ss << "#include <cuda_bf16.h>\n";
        return "__nv_bfloat16";
    }

    throw runtime_error("Unsupported scalar compute dtype in fused stage emitter: " + TensorDescriptor::getElementTypeName(dtype));
}

static string scalarStorageType(DataType dtype, std::ostringstream& ss) {
    if (dtype == DataType::FP32) {
        return "float";
    }
    if (dtype == DataType::FP16) {
        ss << "#include <cuda_fp16.h>\n";
        return "half";
    }
    if (dtype == DataType::BF16) {
        ss << "#include <cuda_bf16.h>\n";
        return "__nv_bfloat16";
    }
    if (dtype == DataType::FP8_E4M3) {
        ss << "#include <cuda_fp16.h>\n";
        ss << "#include <cuda_fp8.h>\n";
        return "__nv_fp8_e4m3";
    }
    if (dtype == DataType::FP8_E5M2) {
        ss << "#include <cuda_fp16.h>\n";
        ss << "#include <cuda_fp8.h>\n";
        return "__nv_fp8_e5m2";
    }

    throw runtime_error("Unsupported scalar storage dtype in fused stage emitter: " + TensorDescriptor::getElementTypeName(dtype));
}

static string storeScalarExpr(const string& storage_type, const string& compute_type, const string& value_expr) {
    if (storage_type == compute_type) {
        return value_expr;
    }
    return storage_type + "(" + value_expr + ")";
}

static vector<vector<const CompiledStageOutput*>> groupOutputsByNumelDescending(const PhysicalExecutionStage& stage) {
    // For 1.0, we assume runtime/planner has already arranged that outputs in a fused stage
    // are legal to execute in one kernel, and that emitter/runtime can determine numel per output.
    //
    // Here we use a temporary grouping rule:
    // each output is its own group in declared order.
    //
    // Replace this later once stage/output metadata includes per-output numel.
    vector<vector<const CompiledStageOutput*>> groups;
    groups.reserve(stage.outputs.size());
    for (const CompiledStageOutput& output : stage.outputs) {
        groups.push_back({&output});
    }
    return groups;
}

string CudaSourceEmitter::emit(const PhysicalExecutionStage& stage,
                               DataType dtype,
                               const string& kernel_name,
                               const bool broadcast_support) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel) {
        throw runtime_error("CudaSourceEmitter::emit(stage, ...) called on non-fused stage.");
    }

    if (stage.outputs.empty()) {
        throw runtime_error("Fused stage has no outputs.");
    }

    const bool vector2_path =
        dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2;

    ostringstream ss;

    if (!vector2_path) {
        const string compute_type = scalarComputeType(dtype, ss);
        const string storage_type = scalarStorageType(dtype, ss);

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

        for (uint32_t i = 0; i < stage.expr.numInputs(); ++i) {
            ss << "const " << storage_type << "* in" << i << ", ";
        }

        for (uint32_t i = 0; i < stage.outputs.size(); ++i) {
            ss << storage_type << "* out" << i << ", ";
        }

        if (broadcast_support) {
            ss << "const BroadcastInfoHeader* broadcast) {\n";
        } else {
            ss << "unsigned long long numel) {\n";
        }

        ss << "  unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n";

        if (broadcast_support) {
            ss << R"DEVICE(
  if (idx >= broadcast->numel) return;

  const unsigned int rank = broadcast->rank;
  const unsigned long long* output_strides = broadcast_output_strides(broadcast);
  const unsigned long long* input_strides = broadcast_input_strides(broadcast);

)DEVICE";

            for (uint32_t inputIdx = 0; inputIdx < stage.expr.numInputs(); ++inputIdx) {
                ss << "  unsigned long long in" << inputIdx << "_offset = 0ULL;\n";
            }

            ss << R"DEVICE(
  unsigned long long remaining = idx;
  for (unsigned int axis = 0; axis < rank; ++axis) {
    const unsigned long long stride = output_strides[axis];
    const unsigned long long c = remaining / stride;
    remaining -= c * stride;
)DEVICE";

            for (uint32_t inputIdx = 0; inputIdx < stage.expr.numInputs(); ++inputIdx) {
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

        for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
            emitScalarNode(ss, stage.expr, node_idx, dtype, compute_type, broadcast_support);
        }

        ss << "\n";
        for (uint32_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
            const CompiledStageOutput& output = stage.outputs[out_idx];

            if (output.local_node_idx >= stage.expr.nodes.size()) {
                throw runtime_error("Stage output local_node_idx out of range.");
            }

            ss << "  out" << out_idx << "[idx] = " << storeScalarExpr(storage_type, compute_type, ref(output.local_node_idx)) << ";\n";
        }

        ss << "}\n";

        if (PRINT_KERNELS) {
            printf("%s\n", ss.str().c_str());
        }

        return ss.str();
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
    } else {
        compute_dtype = "half";
        compute_dtype_vector = "half2";
        storage_dtype = "__nv_fp8_e5m2";
        storage_dtype_vector = "__nv_fp8x2_e5m2";
        ss << "#include <cuda_fp16.h>\n";
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

    if (broadcast_support) {
        for (uint32_t i = 0; i < stage.expr.numInputs(); ++i) {
            ss << "const " << storage_dtype << "* in" << i << ", ";
        }
    } else {
        for (uint32_t i = 0; i < stage.expr.numInputs(); ++i) {
            ss << "const " << storage_dtype_vector << "* in" << i << ", ";
        }
    }

    for (uint32_t i = 0; i < stage.outputs.size(); ++i) {
        ss << storage_dtype_vector << "* out" << i << ", ";
    }

    if (broadcast_support) {
        ss << "const BroadcastInfoHeader* broadcast) {\n";
    } else {
        ss << "unsigned long long numel) {\n";
    }

    ss << "  unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n";

    if (!broadcast_support) {
        ss << "  if (idx >= (numel >> 1)) return;\n\n";
    } else {
        ss << "  unsigned long long idx0 = idx << 1;\n";
        ss << "  if (idx0 >= broadcast->numel) return;\n";
        ss << "  unsigned long long idx1 = idx0 + 1;\n\n";

        ss << "  const unsigned int rank = broadcast->rank;\n";
        ss << "  const unsigned long long* output_strides = broadcast_output_strides(broadcast);\n";
        ss << "  const unsigned long long* input_strides = broadcast_input_strides(broadcast);\n\n";

        for (uint32_t inputIdx = 0; inputIdx < stage.expr.numInputs(); ++inputIdx) {
            ss << "  unsigned long long in" << inputIdx << "_offset0 = 0ULL;\n";
            ss << "  unsigned long long in" << inputIdx << "_offset1 = 0ULL;\n";
        }
        ss << "\n";

        ss << "  {\n";
        ss << "    unsigned long long remaining = idx0;\n";
        ss << "    for (unsigned int axis = 0; axis < rank; ++axis) {\n";
        ss << "      const unsigned long long stride = output_strides[axis];\n";
        ss << "      const unsigned long long c = remaining / stride;\n";
        ss << "      remaining -= c * stride;\n";
        for (uint32_t inputIdx = 0; inputIdx < stage.expr.numInputs(); ++inputIdx) {
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
        ss << "      remaining -= c * stride;\n";
        for (uint32_t inputIdx = 0; inputIdx < stage.expr.numInputs(); ++inputIdx) {
            if (inputIdx == 0)
                ss << "      in" << inputIdx << "_offset1 += c * input_strides[axis];\n";
            else if (inputIdx == 1)
                ss << "      in" << inputIdx << "_offset1 += c * input_strides[rank + axis];\n";
            else
                ss << "      in" << inputIdx << "_offset1 += c * input_strides[" << inputIdx << " * rank + axis];\n";
        }
        ss << "    }\n";
        ss << "  } else {\n";
        for (uint32_t inputIdx = 0; inputIdx < stage.expr.numInputs(); ++inputIdx) {
            ss << "    in" << inputIdx << "_offset1 = in" << inputIdx << "_offset0;\n";
        }
        ss << "  }\n\n";
    }

    for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
        const auto& n = stage.expr.nodes[node_idx];
        switch (n.op) {
            case ExprOp::INPUT: {
                if (broadcast_support) {
                    const string var0 = "in" + to_string(n.input_slot) + "[in" + to_string(n.input_slot) + "_offset0]";
                    const string var1 = "in" + to_string(n.input_slot) + "[in" + to_string(n.input_slot) + "_offset1]";
                    ss << "  " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2BroadcastPackLoad(storage_dtype, var0, var1) << ";\n";
                } else {
                    const string variable = "in" + to_string(n.input_slot) + "[idx]";
                    ss << "  " << compute_dtype_vector << " t" << node_idx << " = "
                       << vector_compute_conversion(storage_dtype_vector, variable) << ";\n";
                }
                break;
            }
            case ExprOp::SCALAR_FP:
                ss << "  " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2ScalarLiteral(n.scalar_fp, dtype) << ";\n";
                break;
            case ExprOp::ADD:
                ss << "  " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Add(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            case ExprOp::SUB:
                ss << "  " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Sub(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            case ExprOp::MUL:
                ss << "  " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Mul(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            case ExprOp::DIV:
                ss << "  " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Div(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            case ExprOp::NEG:
                ss << "  " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Neg(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::EXP:
                ss << "  " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Exp(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::EXP2:
                ss << "  " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Exp2(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::EXP10:
                ss << "  " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Exp10(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::LN:
                ss << "  " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Ln(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::LOG2:
                ss << "  " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Log2(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::LOG10:
                ss << "  " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Log10(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::SQRT:
                ss << "  " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Sqrt(ref(n.lhs), dtype) << ";\n";
                break;
            case ExprOp::POW:
                ss << "  " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Pow(ref(n.lhs), ref(n.rhs), dtype) << ";\n";
                break;
            case ExprOp::MIN:
                ss << "  " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Min(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            case ExprOp::MAX:
                ss << "  " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Max(ref(n.lhs), ref(n.rhs)) << ";\n";
                break;
            default:
                throw runtime_error("Unsupported op in multi-output vector emitter: " + to_string((int32_t)n.op));
        }
    }

    ss << "\n";
    for (uint32_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
        const CompiledStageOutput& output = stage.outputs[out_idx];

        if (output.local_node_idx >= stage.expr.nodes.size()) {
            throw runtime_error("Stage output local_node_idx out of range.");
        }

        ss << "  out" << out_idx << "[idx] = " << vector_storage_conversion(storage_dtype_vector, ref(output.local_node_idx)) << ";\n";
    }

    ss << "}\n";

    if (PRINT_KERNELS) {
        printf("%s\n", ss.str().c_str());
    }

    return ss.str();
}

std::string CudaSourceEmitter::emitGroupedBroadcast(const CompiledExecutionStage& stage,
                                                    const std::vector<std::vector<uint32_t>>& output_groups,
                                                    DataType dtype,
                                                    const std::string& kernel_name) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        throw std::runtime_error("emitGroupedBroadcast called on non-fused stage.");
    }

    if (output_groups.empty()) {
        throw std::runtime_error("emitGroupedBroadcast requires at least one output group.");
    }

    if (dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2) {
        throw std::runtime_error("First cut: emitGroupedBroadcast is scalar-path only. Mirror the same structure into vector2 next.");
    }

    std::ostringstream ss;

    const std::string compute_type = scalarComputeType(dtype, ss);
    const std::string storage_type = scalarStorageType(dtype, ss);

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

    for (uint32_t i = 0; i < stage.expr.numInputs(); ++i) {
        ss << "const " << storage_type << "* in" << i << ", ";
    }
    for (uint32_t i = 0; i < stage.outputs.size(); ++i) {
        ss << storage_type << "* out" << i << ", ";
    }
    for (uint32_t g = 0; g < output_groups.size(); ++g) {
        ss << "const BroadcastInfoHeader* broadcast" << g;
        if (g + 1 < output_groups.size()) {
            ss << ", ";
        }
    }
    ss << ") {\n";

    ss << "  unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n\n";

    for (uint32_t g = 0; g < output_groups.size(); ++g) {
        const std::vector<uint32_t> required_nodes = orderedRequiredNodesForGroup(stage, output_groups[g]);
        const std::vector<uint32_t> used_inputs = usedInputSlotsForNodes(stage.expr, required_nodes);

        ss << "  if (idx < broadcast" << g << "->numel) {\n";
        ss << "    const unsigned int rank = broadcast" << g << "->rank;\n";
        ss << "    const unsigned long long* output_strides = broadcast_output_strides(broadcast" << g << ");\n";
        ss << "    const unsigned long long* input_strides = broadcast_input_strides(broadcast" << g << ");\n";

        for (uint32_t input_slot : used_inputs) {
            ss << "    unsigned long long in" << input_slot << "_offset = 0ULL;\n";
        }

        ss << R"DEVICE(
    unsigned long long remaining = idx;
    for (unsigned int axis = 0; axis < rank; ++axis) {
      const unsigned long long stride = output_strides[axis];
      const unsigned long long c = remaining / stride;
      remaining -= c * stride;
)DEVICE";

        for (uint32_t input_slot : used_inputs) {
            if (input_slot == 0) {
                ss << "      in" << input_slot << "_offset += c * input_strides[axis];\n";
            } else if (input_slot == 1) {
                ss << "      in" << input_slot << "_offset += c * input_strides[rank + axis];\n";
            } else {
                ss << "      in" << input_slot << "_offset += c * input_strides[" << input_slot << " * rank + axis];\n";
            }
        }

        ss << "    }\n\n";

        for (uint32_t node_idx : required_nodes) {
            emitScalarNode(ss, stage.expr, node_idx, dtype, compute_type, /*broadcast_support=*/true);
        }

        ss << "\n";
        for (uint32_t out_idx : output_groups[g]) {
            const CompiledStageOutput& output = stage.outputs[out_idx];
            ss << "    out" << out_idx << "[idx] = " << storeScalarExpr(storage_type, compute_type, ref(output.local_node_idx)) << ";\n";
        }

        ss << "  }\n\n";
    }

    ss << "}\n";

    if (PRINT_KERNELS) {
        printf("%s\n", ss.str().c_str());
    }

    return ss.str();
}

std::string CudaSourceEmitter::emitSpecializedBroadcast(const CompiledExecutionStage& stage,
                                                        const std::vector<SpecializedBroadcastGroup>& groups,
                                                        DataType dtype,
                                                        const std::string& kernel_name) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        throw std::runtime_error("emitSpecializedBroadcast called on non-fused stage.");
    }
    if (groups.empty()) {
        throw std::runtime_error("emitSpecializedBroadcast requires at least one group.");
    }

    const bool vector2_path =
        dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2;

    std::ostringstream ss;

    if (!vector2_path) {
        const std::string compute_type = scalarComputeType(dtype, ss);
        const std::string storage_type = scalarStorageType(dtype, ss);

        ss << "extern \"C\" __global__\n";
        ss << "void " << kernel_name << "(";

        for (uint32_t i = 0; i < stage.expr.numInputs(); ++i) {
            ss << "const " << storage_type << "* in" << i << ", ";
        }
        for (uint32_t i = 0; i < stage.outputs.size(); ++i) {
            ss << storage_type << "* out" << i;
            if (i < stage.outputs.size() - 1)
                ss << ", ";
        }

        ss << ") {\n";

        ss << "  unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n\n";

        for (uint32_t g = 0; g < groups.size(); ++g) {
            const SpecializedBroadcastGroup& group = groups[g];
            const std::vector<uint32_t> required_nodes = orderedRequiredNodesForGroup(stage, group.output_indices);

            std::vector<size_t> all_used_indices(group.used_input_slots.size());
            std::iota(all_used_indices.begin(), all_used_indices.end(), 0);

            ss << "  if (idx < " << group.numel << "ULL) {\n";

            for (uint32_t input_slot : group.used_input_slots) {
                ss << "    unsigned long long in" << input_slot << "_offset = 0ULL;\n";
            }

            if (!group.active_axes.empty()) {
                ss << "\n";
                emitSpecializedBroadcastOffsetMath(ss, group, all_used_indices, "idx", "", "    ");
                ss << "\n";
            }

            for (uint32_t node_idx : required_nodes) {
                emitScalarNode(ss, stage.expr, node_idx, dtype, compute_type, /*broadcast_support=*/true);
            }

            ss << "\n";
            for (uint32_t out_idx : group.output_indices) {
                const CompiledStageOutput& output = stage.outputs[out_idx];
                ss << "    out" << out_idx << "[idx] = " << storeScalarExpr(storage_type, compute_type, ref(output.local_node_idx))
                   << ";\n";
            }

            ss << "  }\n\n";
        }

        ss << "}\n";

        if (PRINT_KERNELS) {
            printf("%s\n", ss.str().c_str());
        }

        return ss.str();
    }

    std::string compute_dtype;
    std::string compute_dtype_vector;
    std::string storage_dtype;
    std::string storage_dtype_vector;

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
    } else {
        compute_dtype = "half";
        compute_dtype_vector = "half2";
        storage_dtype = "__nv_fp8_e5m2";
        storage_dtype_vector = "__nv_fp8x2_e5m2";
        ss << "#include <cuda_fp16.h>\n";
        ss << "#include <cuda_fp8.h>\n";
    }

    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";

    for (uint32_t i = 0; i < stage.expr.numInputs(); ++i) {
        ss << "const " << storage_dtype << "* in" << i << ", ";
    }
    for (uint32_t i = 0; i < stage.outputs.size(); ++i) {
        ss << storage_dtype_vector << "* out" << i;
        if (i < stage.outputs.size() - 1)
            ss << ", ";
    }

    ss << ") {\n";

    ss << "  unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;\n\n";

    for (uint32_t g = 0; g < groups.size(); ++g) {
        const SpecializedBroadcastGroup& group = groups[g];
        const std::vector<uint32_t> required_nodes = orderedRequiredNodesForGroup(stage, group.output_indices);
        const uint64_t packed_numel = (group.numel + 1ULL) >> 1;

        std::vector<size_t> all_used_indices;
        std::vector<size_t> scalar_pack_indices;
        std::unordered_map<uint32_t, SpecializedInputLoadKind> input_load_kind_by_slot;

        all_used_indices.reserve(group.used_input_slots.size());
        scalar_pack_indices.reserve(group.used_input_slots.size());

        for (size_t used_i = 0; used_i < group.used_input_slots.size(); ++used_i) {
            all_used_indices.push_back(used_i);
            input_load_kind_by_slot.emplace(group.used_input_slots[used_i], group.used_input_load_kinds[used_i]);

            if (group.used_input_load_kinds[used_i] == SpecializedInputLoadKind::ScalarPack) {
                scalar_pack_indices.push_back(used_i);
            }
        }

        const bool any_scalar_pack = !scalar_pack_indices.empty();

        ss << "  if (idx < " << packed_numel << "ULL) {\n";
        ss << "    const unsigned long long idx0 = idx << 1;\n";

        if (any_scalar_pack) {
            ss << "    const unsigned long long idx1 = idx0 + 1ULL;\n";
        }

        for (uint32_t input_slot : group.used_input_slots) {
            ss << "    unsigned long long in" << input_slot << "_offset0 = 0ULL;\n";
        }
        for (size_t used_i : scalar_pack_indices) {
            const uint32_t input_slot = group.used_input_slots[used_i];
            ss << "    unsigned long long in" << input_slot << "_offset1 = 0ULL;\n";
        }

        ss << "\n";
        if (!group.active_axes.empty()) {
            emitSpecializedBroadcastOffsetMath(ss, group, all_used_indices, "idx0", "0", "    ");
            ss << "\n";

            if (any_scalar_pack) {
                emitSpecializedBroadcastOffsetMath(ss, group, scalar_pack_indices, "idx1", "1", "    ");
                ss << "\n";
            }
        }

        for (uint32_t node_idx : required_nodes) {
            const auto& n = stage.expr.nodes[node_idx];
            switch (n.op) {
                case ExprOp::INPUT: {
                    const uint32_t slot = n.input_slot;
                    const auto kind_it = input_load_kind_by_slot.find(slot);
                    if (kind_it == input_load_kind_by_slot.end()) {
                        throw std::runtime_error("Missing input load kind for specialized vector broadcast input.");
                    }

                    if (kind_it->second == SpecializedInputLoadKind::NativeVector) {
                        const std::string base = "in" + std::to_string(slot);
                        const std::string offset0 = "in" + std::to_string(slot) + "_offset0";
                        ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                           << emitVector2BroadcastNativeLoad(storage_dtype_vector, base, offset0) << ";\n";
                    } else {
                        const std::string var0 = "in" + std::to_string(slot) + "[in" + std::to_string(slot) + "_offset0]";
                        const std::string var1 = "in" + std::to_string(slot) + "[in" + std::to_string(slot) + "_offset1]";
                        ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                           << emitVector2BroadcastPackLoad(storage_dtype, var0, var1) << ";\n";
                    }
                    break;
                }

                case ExprOp::SCALAR_FP:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2ScalarLiteral(n.scalar_fp, dtype)
                       << ";\n";
                    break;
                case ExprOp::ADD:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Add(ref(n.lhs), ref(n.rhs)) << ";\n";
                    break;
                case ExprOp::SUB:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Sub(ref(n.lhs), ref(n.rhs)) << ";\n";
                    break;
                case ExprOp::MUL:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Mul(ref(n.lhs), ref(n.rhs)) << ";\n";
                    break;
                case ExprOp::DIV:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Div(ref(n.lhs), ref(n.rhs)) << ";\n";
                    break;
                case ExprOp::NEG:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Neg(ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::EXP:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Exp(ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::EXP2:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Exp2(ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::EXP10:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Exp10(ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::LN:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Ln(ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::LOG2:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Log2(ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::LOG10:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Log10(ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::SQRT:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Sqrt(ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::POW:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Pow(ref(n.lhs), ref(n.rhs), dtype)
                       << ";\n";
                    break;
                case ExprOp::MIN:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Min(ref(n.lhs), ref(n.rhs)) << ";\n";
                    break;
                case ExprOp::MAX:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Max(ref(n.lhs), ref(n.rhs)) << ";\n";
                    break;
                default:
                    throw std::runtime_error("Unsupported op in specialized vector broadcast emitter.");
            }
        }

        ss << "\n";
        for (uint32_t out_idx : group.output_indices) {
            const CompiledStageOutput& output = stage.outputs[out_idx];
            ss << "    out" << out_idx << "[idx] = " << vector_storage_conversion(storage_dtype_vector, ref(output.local_node_idx))
               << ";\n";
        }

        ss << "  }\n\n";
    }

    ss << "}\n";

    if (PRINT_KERNELS) {
        printf("%s\n", ss.str().c_str());
    }

    return ss.str();
}

}  // namespace ThorImplementation
