#include "Utilities/Expression/SparseRowUpdate.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/CudaDriver/CudaGraphDynamicGrid.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/FusedEquation.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace ThorImplementation {
namespace {

using DataType = TensorDescriptor::DataType;

std::string dtypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

bool isSupportedSparseRowValueDType(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

bool isSupportedSparseRowDType(DataType dtype) {
    return dtype == DataType::UINT16 || dtype == DataType::UINT32 || dtype == DataType::UINT64;
}

bool rowDTypeCanRepresentVocabularySize(DataType dtype, uint64_t vocabularySize) {
    switch (dtype) {
        case DataType::UINT16:
            return vocabularySize <= static_cast<uint64_t>(std::numeric_limits<uint16_t>::max());
        case DataType::UINT32:
            return vocabularySize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
        case DataType::UINT64:
            return true;
        default:
            return false;
    }
}

std::string scalarStorageType(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return "float";
        case DataType::FP16:
            return "__half";
        case DataType::BF16:
            return "__nv_bfloat16";
        case DataType::UINT16:
            return "unsigned short";
        case DataType::UINT32:
            return "unsigned int";
        case DataType::UINT64:
            return "unsigned long long";
        default:
            throw std::runtime_error("Unsupported sparse row update dtype: " + dtypeName(dtype));
    }
}

std::string emitScalarFpLiteral(double x) {
    std::ostringstream ss;
    ss << std::setprecision(9) << std::defaultfloat << x;
    std::string out = ss.str();
    if (out.find('.') == std::string::npos && out.find('e') == std::string::npos && out.find('E') == std::string::npos) {
        out += ".0";
    }
    out += "f";
    return out;
}

std::string ref(uint32_t nodeIdx) { return "t" + std::to_string(nodeIdx); }

std::string unaryFunctionName(ExprOp op) {
    switch (op) {
        case ExprOp::ABS:
            return "fabsf";
        case ExprOp::EXP:
            return "expf";
        case ExprOp::EXPM1:
            return "expm1f";
        case ExprOp::EXP2:
            return "exp2f";
        case ExprOp::EXP10:
            return "exp10f";
        case ExprOp::LN:
            return "logf";
        case ExprOp::LOG1P:
            return "log1pf";
        case ExprOp::LOG2:
            return "log2f";
        case ExprOp::LOG10:
            return "log10f";
        case ExprOp::SQRT:
            return "sqrtf";
        case ExprOp::TANH:
            return "tanhf";
        case ExprOp::NORMCDF:
            return "normcdff";
        default:
            throw std::runtime_error("Unsupported sparse row unary expression op.");
    }
}

bool isUnaryOpSupported(ExprOp op) {
    switch (op) {
        case ExprOp::NEG:
        case ExprOp::ABS:
        case ExprOp::EXP:
        case ExprOp::EXPM1:
        case ExprOp::EXP2:
        case ExprOp::EXP10:
        case ExprOp::LN:
        case ExprOp::LOG1P:
        case ExprOp::LOG2:
        case ExprOp::LOG10:
        case ExprOp::SQRT:
        case ExprOp::TANH:
        case ExprOp::NORMCDF:
            return true;
        default:
            return false;
    }
}

bool isBinaryOpSupported(ExprOp op) {
    switch (op) {
        case ExprOp::ADD:
        case ExprOp::SUB:
        case ExprOp::MUL:
        case ExprOp::DIV:
        case ExprOp::POW:
        case ExprOp::MIN:
        case ExprOp::MAX:
            return true;
        default:
            return false;
    }
}

std::string binaryExpr(ExprOp op, const std::string& lhs, const std::string& rhs) {
    switch (op) {
        case ExprOp::ADD:
            return "(" + lhs + " + " + rhs + ")";
        case ExprOp::SUB:
            return "(" + lhs + " - " + rhs + ")";
        case ExprOp::MUL:
            return "(" + lhs + " * " + rhs + ")";
        case ExprOp::DIV:
            return "(" + lhs + " / " + rhs + ")";
        case ExprOp::POW:
            return "powf(" + lhs + ", " + rhs + ")";
        case ExprOp::MIN:
            return "fminf(" + lhs + ", " + rhs + ")";
        case ExprOp::MAX:
            return "fmaxf(" + lhs + ", " + rhs + ")";
        default:
            throw std::runtime_error("Unsupported sparse row binary expression op.");
    }
}

std::string loadFloatExpr(const std::string& ptrName, const std::string& offsetExpr) {
    return "thor_sparse_load_float(" + ptrName + ", " + offsetExpr + ")";
}

std::string outputStoreStmt(const std::string& ptrName, const std::string& offsetExpr, const std::string& valueExpr) {
    return "thor_sparse_store_float(" + ptrName + ", " + offsetExpr + ", " + valueExpr + ");";
}

std::string vectorLoadExpr(const std::string& ptrName, const std::string& offsetExpr) {
    return "thor_sparse_load_float4(" + ptrName + ", " + offsetExpr + ")";
}

std::string vectorOutputStoreStmt(const std::string& ptrName, const std::string& offsetExpr, const std::string& valueExpr) {
    return "thor_sparse_store_float4(" + ptrName + ", " + offsetExpr + ", " + valueExpr + ");";
}

std::string splatFloat4(const std::string& valueExpr) {
    return "make_float4(" + valueExpr + ", " + valueExpr + ", " + valueExpr + ", " + valueExpr + ")";
}

std::string unaryVectorFunctionName(ExprOp op) {
    switch (op) {
        case ExprOp::ABS:
            return "fabsf";
        case ExprOp::EXP:
            return "expf";
        case ExprOp::EXPM1:
            return "expm1f";
        case ExprOp::EXP2:
            return "exp2f";
        case ExprOp::EXP10:
            return "exp10f";
        case ExprOp::LN:
            return "logf";
        case ExprOp::LOG1P:
            return "log1pf";
        case ExprOp::LOG2:
            return "log2f";
        case ExprOp::LOG10:
            return "log10f";
        case ExprOp::SQRT:
            return "sqrtf";
        case ExprOp::TANH:
            return "tanhf";
        case ExprOp::NORMCDF:
            return "normcdff";
        default:
            throw std::runtime_error("Unsupported sparse row vector unary expression op.");
    }
}

std::string vectorUnaryExpr(ExprOp op, const std::string& value) {
    if (op == ExprOp::NEG) {
        return "make_float4(-" + value + ".x, -" + value + ".y, -" + value + ".z, -" + value + ".w)";
    }
    const std::string fn = unaryVectorFunctionName(op);
    return "make_float4(" + fn + "(" + value + ".x), " + fn + "(" + value + ".y), " + fn + "(" + value + ".z), " + fn + "(" + value +
           ".w))";
}

std::string vectorBinaryExpr(ExprOp op, const std::string& lhs, const std::string& rhs) {
    auto componentwise = [&](const char* symbol) {
        return std::string("make_float4((") + lhs + ".x " + symbol + " " + rhs + ".x), (" + lhs + ".y " + symbol + " " + rhs + ".y), (" +
               lhs + ".z " + symbol + " " + rhs + ".z), (" + lhs + ".w " + symbol + " " + rhs + ".w))";
    };
    switch (op) {
        case ExprOp::ADD:
            return componentwise("+");
        case ExprOp::SUB:
            return componentwise("-");
        case ExprOp::MUL:
            return componentwise("*");
        case ExprOp::DIV:
            return componentwise("/");
        case ExprOp::POW:
            return "make_float4(powf(" + lhs + ".x, " + rhs + ".x), powf(" + lhs + ".y, " + rhs + ".y), powf(" + lhs + ".z, " + rhs +
                   ".z), powf(" + lhs + ".w, " + rhs + ".w))";
        case ExprOp::MIN:
            return "make_float4(fminf(" + lhs + ".x, " + rhs + ".x), fminf(" + lhs + ".y, " + rhs + ".y), fminf(" + lhs + ".z, " + rhs +
                   ".z), fminf(" + lhs + ".w, " + rhs + ".w))";
        case ExprOp::MAX:
            return "make_float4(fmaxf(" + lhs + ".x, " + rhs + ".x), fmaxf(" + lhs + ".y, " + rhs + ".y), fmaxf(" + lhs + ".z, " + rhs +
                   ".z), fmaxf(" + lhs + ".w, " + rhs + ".w))";
        default:
            throw std::runtime_error("Unsupported sparse row vector binary expression op.");
    }
}

void validateDenseContiguous(const Tensor& tensor, const std::string& label) {
    if (tensor.hasCustomStrides() || !tensor.isDenseContiguous()) {
        throw std::invalid_argument("Sparse row update " + label + " tensor must be dense contiguous.");
    }
}

void validateSameGpuPlacement(const Tensor& tensor, const TensorPlacement& placement, const std::string& label) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument("Sparse row update " + label + " tensor is not initialized.");
    }
    if (tensor.getPlacement() != placement) {
        throw std::invalid_argument("Sparse row update " + label + " tensor must live on the same GPU placement as rows.");
    }
}

uint64_t checkedProduct(uint64_t a, uint64_t b, const std::string& label) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::overflow_error("Sparse row update " + label + " exceeds uint64_t range.");
    }
    return a * b;
}

std::vector<DataType> rootInputDTypes(const PhysicalExpression& expr,
                                      const std::unordered_map<std::string, SparseRowUpdateTensorBinding>& inputs) {
    std::vector<DataType> dtypes(expr.inputs.size(), DataType::FP32);
    for (const NamedInput& input : expr.inputs) {
        if (input.slot >= dtypes.size()) {
            throw std::runtime_error("Sparse row update expression input slot is out of range.");
        }
        switch (input.kind) {
            case NamedInput::Kind::Tensor: {
                auto it = inputs.find(input.name);
                if (it == inputs.end()) {
                    throw std::invalid_argument("Sparse row update missing tensor input binding for expression input '" + input.name +
                                                "'.");
                }
                dtypes[input.slot] = it->second.tensor.getDataType();
                break;
            }
            case NamedInput::Kind::RuntimeScalarFp32:
                dtypes[input.slot] = DataType::FP32;
                break;
            case NamedInput::Kind::TensorRuntimeScalar:
                throw std::invalid_argument("Sparse row update does not support tensor-backed runtime scalar inputs yet.");
        }
    }
    return dtypes;
}

std::string emitVector4KernelSource(const PhysicalOutputs& outputs,
                                    const std::vector<SparseRowUpdatePlan::RuntimeInputSlot>& inputSlots,
                                    const std::vector<SparseRowUpdatePlan::RuntimeOutputSlot>& outputSlots,
                                    DataType rowDataType,
                                    uint64_t capacity,
                                    uint64_t vocabularySize,
                                    uint64_t embeddingDim,
                                    const std::string& kernelName) {
    if (!outputs.expr) {
        throw std::runtime_error("Sparse row update requires a non-null expression.");
    }
    if (outputs.outputs.empty()) {
        throw std::runtime_error("Sparse row update requires at least one output.");
    }
    if (embeddingDim == 0 || embeddingDim % 4 != 0) {
        throw std::runtime_error("Sparse row vector4 update requires embeddingDim to be divisible by 4.");
    }

    const std::string rowType = scalarStorageType(rowDataType);
    std::ostringstream ss;
    ss << "#include <cuda_fp16.h>\n";
    ss << "#include <cuda_bf16.h>\n";
    ss << "#include <math_functions.h>\n\n";
    ss << "__device__ __forceinline__ float4 thor_sparse_load_float4(const float* p, unsigned long long i) { return "
          "*reinterpret_cast<const float4*>(p + i); }\n";
    ss << "__device__ __forceinline__ float4 thor_sparse_load_float4(const __half* p, unsigned long long i) {\n";
    ss << "  const __half2 lo = *reinterpret_cast<const __half2*>(p + i);\n";
    ss << "  const __half2 hi = *reinterpret_cast<const __half2*>(p + i + 2ULL);\n";
    ss << "  const float2 a = __half22float2(lo);\n";
    ss << "  const float2 b = __half22float2(hi);\n";
    ss << "  return make_float4(a.x, a.y, b.x, b.y);\n";
    ss << "}\n";
    ss << "__device__ __forceinline__ float4 thor_sparse_load_float4(const __nv_bfloat16* p, unsigned long long i) {\n";
    ss << "  return make_float4(__bfloat162float(p[i]), __bfloat162float(p[i + 1ULL]), __bfloat162float(p[i + 2ULL]), __bfloat162float(p[i "
          "+ 3ULL]));\n";
    ss << "}\n";
    ss << "__device__ __forceinline__ void thor_sparse_store_float4(float* p, unsigned long long i, float4 v) { "
          "*reinterpret_cast<float4*>(p + i) = v; }\n";
    ss << "__device__ __forceinline__ void thor_sparse_store_float4(__half* p, unsigned long long i, float4 v) {\n";
    ss << "  *reinterpret_cast<__half2*>(p + i) = __floats2half2_rn(v.x, v.y);\n";
    ss << "  *reinterpret_cast<__half2*>(p + i + 2ULL) = __floats2half2_rn(v.z, v.w);\n";
    ss << "}\n";
    ss << "__device__ __forceinline__ void thor_sparse_store_float4(__nv_bfloat16* p, unsigned long long i, float4 v) {\n";
    ss << "  p[i] = __float2bfloat16(v.x);\n";
    ss << "  p[i + 1ULL] = __float2bfloat16(v.y);\n";
    ss << "  p[i + 2ULL] = __float2bfloat16(v.z);\n";
    ss << "  p[i + 3ULL] = __float2bfloat16(v.w);\n";
    ss << "}\n\n";

    ss << "extern \"C\" __global__\n";
    ss << "void " << kernelName << "(const " << rowType << "* rows, const " << rowType << "* num_rows";
    for (size_t slotIdx = 0; slotIdx < inputSlots.size(); ++slotIdx) {
        const auto& slot = inputSlots[slotIdx];
        if (slot.inputKind == NamedInput::Kind::Tensor) {
            ss << ", const " << scalarStorageType(slot.dtype) << "* in" << slotIdx;
        } else if (slot.inputKind == NamedInput::Kind::RuntimeScalarFp32) {
            ss << ", float in" << slotIdx;
        } else {
            throw std::runtime_error("Sparse row vector4 update source emitter encountered unsupported input kind.");
        }
    }
    for (size_t i = 0; i < outputSlots.size(); ++i) {
        ss << ", " << scalarStorageType(outputSlots[i].dtype) << "* out" << i;
    }
    ss << ") {\n";
    ss << "  const unsigned long long idx = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n";
    ss << "  const unsigned long long active_rows = static_cast<unsigned long long>(num_rows[0]);\n";
    ss << "  if (active_rows > " << capacity << "ULL) { asm(\"trap;\"); return; }\n";
    ss << "  const unsigned long long embedding_dim = " << embeddingDim << "ULL;\n";
    ss << "  const unsigned long long vectors_per_row = " << (embeddingDim / 4ULL) << "ULL;\n";
    ss << "  const unsigned long long total = active_rows * vectors_per_row;\n";
    ss << "  if (idx >= total) return;\n";
    ss << "  const unsigned long long logical_row = idx / vectors_per_row;\n";
    ss << "  const unsigned long long vector_idx = idx - logical_row * vectors_per_row;\n";
    ss << "  const unsigned long long dim = vector_idx * 4ULL;\n";
    ss << "  const unsigned long long row = static_cast<unsigned long long>(rows[logical_row]);\n";
    ss << "  if (row >= " << vocabularySize << "ULL) { asm(\"trap;\"); return; }\n";
    ss << "  const unsigned long long logical_offset = logical_row * embedding_dim + dim;\n";
    ss << "  const unsigned long long indexed_offset = row * embedding_dim + dim;\n\n";

    std::unordered_map<uint32_t, const SparseRowUpdatePlan::RuntimeInputSlot*> slotByInputSlot;
    for (const auto& namedInput : outputs.expr->inputs) {
        if (namedInput.slot >= inputSlots.size()) {
            throw std::runtime_error("Sparse row vector4 update input slot map is inconsistent.");
        }
        slotByInputSlot[namedInput.slot] = &inputSlots[namedInput.slot];
    }

    for (uint32_t nodeIdx = 0; nodeIdx < outputs.expr->nodes.size(); ++nodeIdx) {
        const ExprNode& node = outputs.expr->nodes[nodeIdx];
        switch (node.op) {
            case ExprOp::INPUT: {
                auto it = slotByInputSlot.find(node.input_slot);
                if (it == slotByInputSlot.end()) {
                    throw std::runtime_error("Sparse row vector4 update INPUT node references an unknown input slot.");
                }
                const auto& slot = *it->second;
                if (slot.inputKind != NamedInput::Kind::Tensor) {
                    throw std::runtime_error("Sparse row vector4 update INPUT node was not bound to a tensor input.");
                }
                const std::string offset = slot.tensorKind == SparseRowUpdateTensorKind::IndexedRows ? "indexed_offset" : "logical_offset";
                ss << "  const float4 " << ref(nodeIdx) << " = " << vectorLoadExpr("in" + std::to_string(node.input_slot), offset) << ";\n";
                break;
            }
            case ExprOp::RUNTIME_SCALAR: {
                auto it = slotByInputSlot.find(node.input_slot);
                if (it == slotByInputSlot.end()) {
                    throw std::runtime_error("Sparse row vector4 update RUNTIME_SCALAR node references an unknown input slot.");
                }
                const auto& slot = *it->second;
                if (slot.inputKind != NamedInput::Kind::RuntimeScalarFp32) {
                    throw std::runtime_error("Sparse row vector4 update runtime scalar node was not bound to a runtime scalar input.");
                }
                ss << "  const float4 " << ref(nodeIdx) << " = " << splatFloat4("in" + std::to_string(node.input_slot)) << ";\n";
                break;
            }
            case ExprOp::SCALAR_FP:
                ss << "  const float4 " << ref(nodeIdx) << " = " << splatFloat4(emitScalarFpLiteral(node.scalar_fp)) << ";\n";
                break;
            case ExprOp::NEG:
                ss << "  const float4 " << ref(nodeIdx) << " = " << vectorUnaryExpr(node.op, ref(node.lhs)) << ";\n";
                break;
            default:
                if (isUnaryOpSupported(node.op)) {
                    ss << "  const float4 " << ref(nodeIdx) << " = " << vectorUnaryExpr(node.op, ref(node.lhs)) << ";\n";
                } else if (isBinaryOpSupported(node.op)) {
                    ss << "  const float4 " << ref(nodeIdx) << " = " << vectorBinaryExpr(node.op, ref(node.lhs), ref(node.rhs)) << ";\n";
                } else {
                    throw std::runtime_error("Sparse row vector4 update expression uses unsupported op " +
                                             std::to_string(static_cast<int>(node.op)) +
                                             ". Sparse optimizer updates currently support scalar pointwise expression math only.");
                }
                break;
        }
    }

    for (size_t outputIdx = 0; outputIdx < outputs.outputs.size(); ++outputIdx) {
        const NamedOutput& out = outputs.outputs[outputIdx];
        if (out.node_idx >= outputs.expr->nodes.size()) {
            throw std::runtime_error("Sparse row vector4 update output node index is out of range.");
        }
        ss << "  " << vectorOutputStoreStmt("out" + std::to_string(outputIdx), "indexed_offset", ref(out.node_idx)) << "\n";
    }

    ss << "}\n";
    return ss.str();
}

std::string emitKernelSource(const PhysicalOutputs& outputs,
                             const std::vector<SparseRowUpdatePlan::RuntimeInputSlot>& inputSlots,
                             const std::vector<SparseRowUpdatePlan::RuntimeOutputSlot>& outputSlots,
                             DataType rowDataType,
                             uint64_t capacity,
                             uint64_t vocabularySize,
                             uint64_t embeddingDim,
                             const std::string& kernelName) {
    if (embeddingDim != 0 && embeddingDim % 4 == 0) {
        return emitVector4KernelSource(outputs, inputSlots, outputSlots, rowDataType, capacity, vocabularySize, embeddingDim, kernelName);
    }
    if (!outputs.expr) {
        throw std::runtime_error("Sparse row update requires a non-null expression.");
    }
    if (outputs.outputs.empty()) {
        throw std::runtime_error("Sparse row update requires at least one output.");
    }

    const std::string rowType = scalarStorageType(rowDataType);
    std::ostringstream ss;
    ss << "#include <cuda_fp16.h>\n";
    ss << "#include <cuda_bf16.h>\n";
    ss << "#include <math_functions.h>\n\n";
    ss << "__device__ __forceinline__ float thor_sparse_load_float(const float* p, unsigned long long i) { return p[i]; }\n";
    ss << "__device__ __forceinline__ float thor_sparse_load_float(const __half* p, unsigned long long i) { return __half2float(p[i]); }\n";
    ss << "__device__ __forceinline__ float thor_sparse_load_float(const __nv_bfloat16* p, unsigned long long i) { return "
          "__bfloat162float(p[i]); }\n";
    ss << "__device__ __forceinline__ void thor_sparse_store_float(float* p, unsigned long long i, float v) { p[i] = v; }\n";
    ss << "__device__ __forceinline__ void thor_sparse_store_float(__half* p, unsigned long long i, float v) { p[i] = __float2half_rn(v); "
          "}\n";
    ss << "__device__ __forceinline__ void thor_sparse_store_float(__nv_bfloat16* p, unsigned long long i, float v) { p[i] = "
          "__float2bfloat16(v); }\n\n";

    ss << "extern \"C\" __global__\n";
    ss << "void " << kernelName << "(const " << rowType << "* rows, const " << rowType << "* num_rows";
    for (size_t slotIdx = 0; slotIdx < inputSlots.size(); ++slotIdx) {
        const auto& slot = inputSlots[slotIdx];
        if (slot.inputKind == NamedInput::Kind::Tensor) {
            ss << ", const " << scalarStorageType(slot.dtype) << "* in" << slotIdx;
        } else if (slot.inputKind == NamedInput::Kind::RuntimeScalarFp32) {
            ss << ", float in" << slotIdx;
        } else {
            throw std::runtime_error("Sparse row update source emitter encountered unsupported input kind.");
        }
    }
    for (size_t i = 0; i < outputSlots.size(); ++i) {
        ss << ", " << scalarStorageType(outputSlots[i].dtype) << "* out" << i;
    }
    ss << ") {\n";
    ss << "  const unsigned long long idx = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n";
    ss << "  const unsigned long long active_rows = static_cast<unsigned long long>(num_rows[0]);\n";
    ss << "  if (active_rows > " << capacity << "ULL) { asm(\"trap;\"); return; }\n";
    ss << "  const unsigned long long embedding_dim = " << embeddingDim << "ULL;\n";
    ss << "  const unsigned long long total = active_rows * embedding_dim;\n";
    ss << "  if (idx >= total) return;\n";
    ss << "  const unsigned long long logical_row = idx / embedding_dim;\n";
    ss << "  const unsigned long long dim = idx - logical_row * embedding_dim;\n";
    ss << "  const unsigned long long row = static_cast<unsigned long long>(rows[logical_row]);\n";
    ss << "  if (row >= " << vocabularySize << "ULL) { asm(\"trap;\"); return; }\n";
    ss << "  const unsigned long long logical_offset = logical_row * embedding_dim + dim;\n";
    ss << "  const unsigned long long indexed_offset = row * embedding_dim + dim;\n\n";

    std::unordered_map<uint32_t, const SparseRowUpdatePlan::RuntimeInputSlot*> slotByInputSlot;
    for (const auto& namedInput : outputs.expr->inputs) {
        if (namedInput.slot >= inputSlots.size()) {
            throw std::runtime_error("Sparse row update input slot map is inconsistent.");
        }
        slotByInputSlot[namedInput.slot] = &inputSlots[namedInput.slot];
    }

    for (uint32_t nodeIdx = 0; nodeIdx < outputs.expr->nodes.size(); ++nodeIdx) {
        const ExprNode& node = outputs.expr->nodes[nodeIdx];
        switch (node.op) {
            case ExprOp::INPUT: {
                auto it = slotByInputSlot.find(node.input_slot);
                if (it == slotByInputSlot.end()) {
                    throw std::runtime_error("Sparse row update INPUT node references an unknown input slot.");
                }
                const auto& slot = *it->second;
                if (slot.inputKind != NamedInput::Kind::Tensor) {
                    throw std::runtime_error("Sparse row update INPUT node was not bound to a tensor input.");
                }
                const std::string offset = slot.tensorKind == SparseRowUpdateTensorKind::IndexedRows ? "indexed_offset" : "logical_offset";
                ss << "  const float " << ref(nodeIdx) << " = " << loadFloatExpr("in" + std::to_string(node.input_slot), offset) << ";\n";
                break;
            }
            case ExprOp::RUNTIME_SCALAR: {
                auto it = slotByInputSlot.find(node.input_slot);
                if (it == slotByInputSlot.end()) {
                    throw std::runtime_error("Sparse row update RUNTIME_SCALAR node references an unknown input slot.");
                }
                const auto& slot = *it->second;
                if (slot.inputKind != NamedInput::Kind::RuntimeScalarFp32) {
                    throw std::runtime_error("Sparse row update runtime scalar node was not bound to a runtime scalar input.");
                }
                ss << "  const float " << ref(nodeIdx) << " = in" << node.input_slot << ";\n";
                break;
            }
            case ExprOp::SCALAR_FP:
                ss << "  const float " << ref(nodeIdx) << " = " << emitScalarFpLiteral(node.scalar_fp) << ";\n";
                break;
            case ExprOp::NEG:
                ss << "  const float " << ref(nodeIdx) << " = -" << ref(node.lhs) << ";\n";
                break;
            default:
                if (isUnaryOpSupported(node.op)) {
                    ss << "  const float " << ref(nodeIdx) << " = " << unaryFunctionName(node.op) << "(" << ref(node.lhs) << ");\n";
                } else if (isBinaryOpSupported(node.op)) {
                    ss << "  const float " << ref(nodeIdx) << " = " << binaryExpr(node.op, ref(node.lhs), ref(node.rhs)) << ";\n";
                } else {
                    throw std::runtime_error("Sparse row update expression uses unsupported op " +
                                             std::to_string(static_cast<int>(node.op)) +
                                             ". Sparse optimizer updates currently support scalar pointwise expression math only.");
                }
                break;
        }
    }

    for (size_t outputIdx = 0; outputIdx < outputs.outputs.size(); ++outputIdx) {
        const NamedOutput& out = outputs.outputs[outputIdx];
        if (out.node_idx >= outputs.expr->nodes.size()) {
            throw std::runtime_error("Sparse row update output node index is out of range.");
        }
        ss << "  " << outputStoreStmt("out" + std::to_string(outputIdx), "indexed_offset", ref(out.node_idx)) << "\n";
    }

    ss << "}\n";
    return ss.str();
}

std::string safeKernelSuffix(uint64_t value) { return std::to_string(value); }

uint32_t sparseUpdateBlockSize(uint64_t maxElements) {
    return static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(maxElements, 1ULL), 256ULL));
}

uint32_t sparseUpdateMaxGrid(uint64_t maxElements, uint32_t blockSize) {
    const uint64_t blocks64 = (maxElements + static_cast<uint64_t>(blockSize) - 1ULL) / static_cast<uint64_t>(blockSize);
    if (blocks64 == 0 || blocks64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("Sparse row update launch grid exceeds uint32_t range.");
    }
    return static_cast<uint32_t>(blocks64);
}

struct SparseRowUpdateKernelArgs {
    std::vector<const void*> tensorInputPtrs;
    std::vector<void*> tensorOutputPtrs;
    std::vector<float> scalarValues;
    std::vector<void*> args;
    const void* rowsPtr = nullptr;
    const void* numRowsPtr = nullptr;
};

SparseRowUpdateKernelArgs buildSparseRowUpdateKernelArgs(const Tensor& rows,
                                                         const Tensor& numRows,
                                                         const std::vector<SparseRowUpdatePlan::RuntimeInputSlot>& inputSlots,
                                                         const std::vector<SparseRowUpdatePlan::RuntimeOutputSlot>& outputSlots,
                                                         const std::unordered_map<std::string, float>& runtimeScalars) {
    SparseRowUpdateKernelArgs out;
    out.tensorInputPtrs.reserve(inputSlots.size());
    out.tensorOutputPtrs.reserve(outputSlots.size());
    out.scalarValues.reserve(inputSlots.size());

    out.rowsPtr = rows.getMemPtr();
    out.numRowsPtr = numRows.getMemPtr();

    for (const SparseRowUpdatePlan::RuntimeInputSlot& slot : inputSlots) {
        if (slot.inputKind == NamedInput::Kind::Tensor) {
            out.tensorInputPtrs.push_back(slot.tensor.getMemPtr());
        } else if (slot.inputKind == NamedInput::Kind::RuntimeScalarFp32) {
            auto it = runtimeScalars.find(slot.name);
            if (it == runtimeScalars.end()) {
                throw std::invalid_argument("Sparse row update missing runtime scalar '" + slot.name + "'.");
            }
            out.scalarValues.push_back(it->second);
        } else {
            throw std::runtime_error("Sparse row update encountered unsupported runtime input kind.");
        }
    }
    for (const SparseRowUpdatePlan::RuntimeOutputSlot& slot : outputSlots) {
        out.tensorOutputPtrs.push_back(const_cast<void*>(static_cast<const void*>(slot.tensor.getMemPtr())));
    }

    out.args.reserve(2 + inputSlots.size() + outputSlots.size());
    out.args.push_back((void*)&out.rowsPtr);
    out.args.push_back((void*)&out.numRowsPtr);

    size_t tensorInputIndex = 0;
    size_t scalarIndex = 0;
    for (const SparseRowUpdatePlan::RuntimeInputSlot& slot : inputSlots) {
        if (slot.inputKind == NamedInput::Kind::Tensor) {
            out.args.push_back((void*)&out.tensorInputPtrs[tensorInputIndex++]);
        } else {
            out.args.push_back((void*)&out.scalarValues[scalarIndex++]);
        }
    }
    for (void*& ptr : out.tensorOutputPtrs) {
        out.args.push_back((void*)&ptr);
    }

    return out;
}

struct SparseRowUpdateFusionBuild {
    PhysicalOutputs outputs;
    uint64_t capacity = 0;
    uint64_t vocabularySize = 0;
    uint64_t embeddingDim = 0;
    DataType rowDataType = DataType::UINT64;
    std::vector<SparseRowUpdatePlan::RuntimeInputSlot> inputSlots;
    std::vector<SparseRowUpdatePlan::RuntimeOutputSlot> outputSlots;
};

SparseRowUpdateFusionBuild buildSparseRowUpdateFusionBuild(
    PhysicalOutputs outputs,
    const Tensor& rows,
    const Tensor& numRows,
    const std::unordered_map<std::string, SparseRowUpdateTensorBinding>& inputs,
    const std::unordered_map<std::string, Tensor>& indexedOutputs,
    const std::unordered_map<std::string, std::string>& localDenseLogicalInputExpressions) {
    if (!outputs.expr) {
        throw std::invalid_argument("Sparse row update fusion source requires expression outputs.");
    }
    if (outputs.outputs.empty()) {
        throw std::invalid_argument("Sparse row update fusion source requires at least one output.");
    }
    if (!rows.isInitialized() || !numRows.isInitialized()) {
        throw std::invalid_argument("Sparse row update fusion source rows and numRows tensors must be initialized.");
    }
    if (rows.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("Sparse row update fusion source rows tensor must live on GPU.");
    }
    const TensorPlacement placement = rows.getPlacement();
    validateSameGpuPlacement(numRows, placement, "numRows");
    validateDenseContiguous(rows, "rows");
    validateDenseContiguous(numRows, "numRows");
    if (!isSupportedSparseRowDType(rows.getDataType()) || numRows.getDataType() != rows.getDataType()) {
        throw std::invalid_argument(
            "Sparse row update fusion source rows and numRows tensors must both be uint16, uint32, or uint64 with matching dtype.");
    }
    if (numRows.getDimensions() != std::vector<uint64_t>{1}) {
        throw std::invalid_argument("Sparse row update fusion source numRows tensor must have shape [1].");
    }
    const std::vector<uint64_t> rowsDims = rows.getDimensions();
    if (rowsDims.size() != 1 || rowsDims[0] == 0) {
        throw std::invalid_argument("Sparse row update fusion source rows tensor must have non-empty shape [row_storage_capacity].");
    }
    const uint64_t rowsStorageCapacity = rowsDims[0];
    uint64_t capacity = 0;

    std::vector<DataType> inputDTypes = rootInputDTypes(*outputs.expr, inputs);
    resolveOutputsDTypesInPlace(outputs, inputDTypes);

    std::vector<SparseRowUpdatePlan::RuntimeInputSlot> inputSlots(outputs.expr->inputs.size());
    uint64_t vocabularySize = 0;
    uint64_t embeddingDim = 0;

    for (const NamedInput& input : outputs.expr->inputs) {
        if (input.slot >= inputSlots.size()) {
            throw std::runtime_error("Sparse row update fusion source expression input slot out of range.");
        }
        SparseRowUpdatePlan::RuntimeInputSlot slot;
        slot.name = input.name;
        slot.inputKind = input.kind;
        if (input.kind == NamedInput::Kind::Tensor) {
            auto it = inputs.find(input.name);
            if (it == inputs.end()) {
                throw std::invalid_argument("Sparse row update fusion source missing tensor input binding for '" + input.name + "'.");
            }
            slot.tensor = it->second.tensor;
            slot.tensorKind = it->second.kind;
            slot.dtype = slot.tensor.getDataType();
            validateSameGpuPlacement(slot.tensor, placement, "input '" + input.name + "'");
            validateDenseContiguous(slot.tensor, "input '" + input.name + "'");
            if (!isSupportedSparseRowValueDType(slot.dtype)) {
                throw std::invalid_argument("Sparse row update fusion source input '" + input.name + "' has unsupported dtype " +
                                            dtypeName(slot.dtype) + ". Supported dtypes are fp16, bf16, and fp32.");
            }
            const std::vector<uint64_t> dims = slot.tensor.getDimensions();
            if (dims.size() != 2 || dims[0] == 0 || dims[1] == 0) {
                throw std::invalid_argument("Sparse row update fusion source input '" + input.name + "' must have non-empty rank-2 shape.");
            }
            if (slot.tensorKind == SparseRowUpdateTensorKind::DenseLogicalRows) {
                if (capacity == 0) {
                    capacity = dims[0];
                } else if (dims[0] != capacity) {
                    throw std::invalid_argument("Sparse row update fusion source dense logical inputs must have matching first dimensions.");
                }
                if (rowsStorageCapacity < capacity) {
                    throw std::invalid_argument(
                        "Sparse row update fusion source rows tensor storage is smaller than the dense logical row capacity.");
                }
            } else {
                if (capacity != 0 && dims[0] < capacity) {
                    throw std::invalid_argument("Sparse row update fusion source indexed input '" + input.name +
                                                "' vocabulary dimension cannot be smaller than dense logical row capacity.");
                }
                if (vocabularySize == 0) {
                    vocabularySize = dims[0];
                } else if (vocabularySize != dims[0]) {
                    throw std::invalid_argument("Sparse row update fusion source indexed tensors must have the same vocabulary dimension.");
                }
            }
            if (embeddingDim == 0) {
                embeddingDim = dims[1];
            } else if (embeddingDim != dims[1]) {
                throw std::invalid_argument("Sparse row update fusion source tensors must have the same embedding dimension.");
            }
        } else if (input.kind == NamedInput::Kind::RuntimeScalarFp32) {
            slot.dtype = DataType::FP32;
        } else {
            throw std::invalid_argument("Sparse row update fusion source does not support tensor-backed runtime scalar inputs yet.");
        }
        inputSlots[input.slot] = std::move(slot);
    }
    for (size_t slotIdx = 0; slotIdx < inputSlots.size(); ++slotIdx) {
        if (inputSlots[slotIdx].name.empty()) {
            throw std::runtime_error("Sparse row update fusion source expression inputs must use contiguous input slots.");
        }
    }

    for (const auto& [name, expr] : localDenseLogicalInputExpressions) {
        (void)expr;
        auto inputIt = inputs.find(name);
        if (inputIt == inputs.end()) {
            throw std::invalid_argument("Sparse row update fusion source local input '" + name + "' is not bound as an expression input.");
        }
        if (inputIt->second.kind != SparseRowUpdateTensorKind::DenseLogicalRows) {
            throw std::invalid_argument("Sparse row update fusion source local input '" + name + "' must be a dense logical row input.");
        }
    }

    std::vector<SparseRowUpdatePlan::RuntimeOutputSlot> outputSlots;
    outputSlots.reserve(outputs.outputs.size());
    for (const NamedOutput& namedOutput : outputs.outputs) {
        auto outIt = indexedOutputs.find(namedOutput.name);
        if (outIt == indexedOutputs.end()) {
            throw std::invalid_argument("Sparse row update fusion source missing indexed output binding for '" + namedOutput.name + "'.");
        }
        Tensor tensor = outIt->second;
        validateSameGpuPlacement(tensor, placement, "output '" + namedOutput.name + "'");
        validateDenseContiguous(tensor, "output '" + namedOutput.name + "'");
        if (!isSupportedSparseRowValueDType(tensor.getDataType())) {
            throw std::invalid_argument("Sparse row update fusion source output '" + namedOutput.name + "' has unsupported dtype " +
                                        dtypeName(tensor.getDataType()) + ". Supported dtypes are fp16, bf16, and fp32.");
        }
        const std::vector<uint64_t> dims = tensor.getDimensions();
        if (dims.size() != 2 || dims[0] == 0 || dims[1] == 0) {
            throw std::invalid_argument("Sparse row update fusion source indexed output '" + namedOutput.name +
                                        "' must have shape [vocabulary, D].");
        }
        if (vocabularySize == 0) {
            vocabularySize = dims[0];
        } else if (vocabularySize != dims[0]) {
            throw std::invalid_argument(
                "Sparse row update fusion source indexed outputs must have the same vocabulary dimension as indexed inputs.");
        }
        if (embeddingDim == 0) {
            embeddingDim = dims[1];
        } else if (embeddingDim != dims[1]) {
            throw std::invalid_argument("Sparse row update fusion source indexed outputs must have the same embedding dimension as inputs.");
        }
        if (namedOutput.node_idx >= outputs.expr->nodes.size() || !outputs.expr->nodes[namedOutput.node_idx].output_dtype.has_value()) {
            throw std::runtime_error("Sparse row update fusion source output node is missing resolved dtype.");
        }
        const DataType expressionOutputDType = outputs.expr->nodes[namedOutput.node_idx].output_dtype.value();
        if (expressionOutputDType != tensor.getDataType()) {
            throw std::invalid_argument("Sparse row update fusion source expression output '" + namedOutput.name + "' dtype " +
                                        dtypeName(expressionOutputDType) + " does not match bound output tensor dtype " +
                                        dtypeName(tensor.getDataType()) + ".");
        }
        outputSlots.push_back(SparseRowUpdatePlan::RuntimeOutputSlot{namedOutput.name, tensor, tensor.getDataType()});
    }

    if (capacity == 0) {
        capacity = rowsStorageCapacity;
    }
    if (rowsStorageCapacity < capacity) {
        throw std::invalid_argument("Sparse row update fusion source rows tensor storage is smaller than the logical sparse-row capacity.");
    }
    if (vocabularySize != 0 && vocabularySize < capacity) {
        throw std::invalid_argument(
            "Sparse row update fusion source indexed tensor vocabulary dimension cannot be smaller than logical sparse-row capacity.");
    }
    if (vocabularySize == 0 || embeddingDim == 0) {
        throw std::invalid_argument("Sparse row update fusion source requires at least one indexed input or output tensor to infer [vocabulary, D].");
    }
    if (embeddingDim % 4ULL != 0ULL) {
        throw std::invalid_argument("Sparse row update fusion source currently requires embedding_dim divisible by 4 for float4 lanes.");
    }
    if (!rowDTypeCanRepresentVocabularySize(rows.getDataType(), vocabularySize)) {
        throw std::invalid_argument("Sparse row update fusion source row dtype " + dtypeName(rows.getDataType()) +
                                    " cannot represent vocabulary_size as the invalid-row sentinel.");
    }
    checkedProduct(capacity, embeddingDim, "fusion source launch domain");

    SparseRowUpdateFusionBuild build;
    build.outputs = std::move(outputs);
    build.capacity = capacity;
    build.vocabularySize = vocabularySize;
    build.embeddingDim = embeddingDim;
    build.rowDataType = rows.getDataType();
    build.inputSlots = std::move(inputSlots);
    build.outputSlots = std::move(outputSlots);
    return build;
}

std::string emitSparseRowUpdateFusionHelperSource() {
    std::ostringstream ss;
    ss << "__device__ __forceinline__ float4 thor_sparse_load_float4(const float* p, unsigned long long i) { return "
          "*reinterpret_cast<const float4*>(p + i); }\n";
    ss << "__device__ __forceinline__ float4 thor_sparse_load_float4(const __half* p, unsigned long long i) {\n";
    ss << "  const __half2 lo = *reinterpret_cast<const __half2*>(p + i);\n";
    ss << "  const __half2 hi = *reinterpret_cast<const __half2*>(p + i + 2ULL);\n";
    ss << "  const float2 a = __half22float2(lo);\n";
    ss << "  const float2 b = __half22float2(hi);\n";
    ss << "  return make_float4(a.x, a.y, b.x, b.y);\n";
    ss << "}\n";
    ss << "__device__ __forceinline__ float4 thor_sparse_load_float4(const __nv_bfloat16* p, unsigned long long i) {\n";
    ss << "  return make_float4(__bfloat162float(p[i]), __bfloat162float(p[i + 1ULL]), __bfloat162float(p[i + 2ULL]), __bfloat162float(p[i "
          "+ 3ULL]));\n";
    ss << "}\n";
    ss << "__device__ __forceinline__ void thor_sparse_store_float4(float* p, unsigned long long i, float4 v) { "
          "*reinterpret_cast<float4*>(p + i) = v; }\n";
    ss << "__device__ __forceinline__ void thor_sparse_store_float4(__half* p, unsigned long long i, float4 v) {\n";
    ss << "  *reinterpret_cast<__half2*>(p + i) = __floats2half2_rn(v.x, v.y);\n";
    ss << "  *reinterpret_cast<__half2*>(p + i + 2ULL) = __floats2half2_rn(v.z, v.w);\n";
    ss << "}\n";
    ss << "__device__ __forceinline__ void thor_sparse_store_float4(__nv_bfloat16* p, unsigned long long i, float4 v) {\n";
    ss << "  p[i] = __float2bfloat16(v.x);\n";
    ss << "  p[i + 1ULL] = __float2bfloat16(v.y);\n";
    ss << "  p[i + 2ULL] = __float2bfloat16(v.z);\n";
    ss << "  p[i + 3ULL] = __float2bfloat16(v.w);\n";
    ss << "}\n\n";
    return ss.str();
}

SparseRowUpdateFusionSource emitVector4SparseRowUpdateFusionSource(
    const PhysicalOutputs& outputs,
    const std::vector<SparseRowUpdatePlan::RuntimeInputSlot>& inputSlots,
    const std::vector<SparseRowUpdatePlan::RuntimeOutputSlot>& outputSlots,
    const std::unordered_map<std::string, std::string>& localDenseLogicalInputExpressions) {
    SparseRowUpdateFusionSource source;
    source.helperSource = emitSparseRowUpdateFusionHelperSource();
    source.outputSlots = outputSlots;

    std::ostringstream parameters;
    for (size_t slotIdx = 0; slotIdx < inputSlots.size(); ++slotIdx) {
        const SparseRowUpdatePlan::RuntimeInputSlot& slot = inputSlots[slotIdx];
        const bool localDenseLogicalInput = slot.inputKind == NamedInput::Kind::Tensor &&
                                           slot.tensorKind == SparseRowUpdateTensorKind::DenseLogicalRows &&
                                           localDenseLogicalInputExpressions.contains(slot.name);
        if (localDenseLogicalInput) {
            continue;
        }
        if (slot.inputKind == NamedInput::Kind::Tensor) {
            parameters << ", const " << scalarStorageType(slot.dtype) << "* __restrict__ sru_in" << slotIdx;
            source.kernelInputSlots.push_back(slot);
        } else if (slot.inputKind == NamedInput::Kind::RuntimeScalarFp32) {
            parameters << ", float sru_in" << slotIdx;
            source.kernelInputSlots.push_back(slot);
        } else {
            throw std::runtime_error("Sparse row update fusion source encountered unsupported input kind.");
        }
    }
    for (size_t outputIdx = 0; outputIdx < outputSlots.size(); ++outputIdx) {
        parameters << ", " << scalarStorageType(outputSlots[outputIdx].dtype) << "* __restrict__ sru_out" << outputIdx;
    }
    source.parameterSource = parameters.str();

    std::unordered_map<uint32_t, const SparseRowUpdatePlan::RuntimeInputSlot*> slotByInputSlot;
    for (const auto& namedInput : outputs.expr->inputs) {
        if (namedInput.slot >= inputSlots.size()) {
            throw std::runtime_error("Sparse row update fusion source input slot map is inconsistent.");
        }
        slotByInputSlot[namedInput.slot] = &inputSlots[namedInput.slot];
    }

    std::ostringstream body;
    body << "  const unsigned long long sru_logical_offset = sru_logical_row * embeddingDim + sru_vector_index * 4ULL;\n";
    body << "  const unsigned long long sru_indexed_offset = sru_indexed_row * embeddingDim + sru_vector_index * 4ULL;\n";
    for (uint32_t nodeIdx = 0; nodeIdx < outputs.expr->nodes.size(); ++nodeIdx) {
        const ExprNode& node = outputs.expr->nodes[nodeIdx];
        switch (node.op) {
            case ExprOp::INPUT: {
                auto it = slotByInputSlot.find(node.input_slot);
                if (it == slotByInputSlot.end()) {
                    throw std::runtime_error("Sparse row update fusion source INPUT node references an unknown input slot.");
                }
                const auto& slot = *it->second;
                if (slot.inputKind != NamedInput::Kind::Tensor) {
                    throw std::runtime_error("Sparse row update fusion source INPUT node was not bound to a tensor input.");
                }
                auto localIt = localDenseLogicalInputExpressions.find(slot.name);
                if (localIt != localDenseLogicalInputExpressions.end()) {
                    if (slot.tensorKind != SparseRowUpdateTensorKind::DenseLogicalRows) {
                        throw std::runtime_error("Sparse row update fusion source local tensor input must be dense logical rows.");
                    }
                    body << "  const float4 " << ref(nodeIdx) << " = " << localIt->second << ";\n";
                    break;
                }
                const std::string offset = slot.tensorKind == SparseRowUpdateTensorKind::IndexedRows ? "sru_indexed_offset" : "sru_logical_offset";
                body << "  const float4 " << ref(nodeIdx) << " = "
                     << vectorLoadExpr("sru_in" + std::to_string(node.input_slot), offset) << ";\n";
                break;
            }
            case ExprOp::RUNTIME_SCALAR: {
                auto it = slotByInputSlot.find(node.input_slot);
                if (it == slotByInputSlot.end()) {
                    throw std::runtime_error("Sparse row update fusion source RUNTIME_SCALAR node references an unknown input slot.");
                }
                const auto& slot = *it->second;
                if (slot.inputKind != NamedInput::Kind::RuntimeScalarFp32) {
                    throw std::runtime_error("Sparse row update fusion source runtime scalar node was not bound to a runtime scalar input.");
                }
                body << "  const float4 " << ref(nodeIdx) << " = " << splatFloat4("sru_in" + std::to_string(node.input_slot)) << ";\n";
                break;
            }
            case ExprOp::SCALAR_FP:
                body << "  const float4 " << ref(nodeIdx) << " = " << splatFloat4(emitScalarFpLiteral(node.scalar_fp)) << ";\n";
                break;
            case ExprOp::NEG:
                body << "  const float4 " << ref(nodeIdx) << " = " << vectorUnaryExpr(node.op, ref(node.lhs)) << ";\n";
                break;
            default:
                if (isUnaryOpSupported(node.op)) {
                    body << "  const float4 " << ref(nodeIdx) << " = " << vectorUnaryExpr(node.op, ref(node.lhs)) << ";\n";
                } else if (isBinaryOpSupported(node.op)) {
                    body << "  const float4 " << ref(nodeIdx) << " = " << vectorBinaryExpr(node.op, ref(node.lhs), ref(node.rhs)) << ";\n";
                } else {
                    throw std::runtime_error("Sparse row update fusion source expression uses unsupported op " +
                                             std::to_string(static_cast<int>(node.op)) +
                                             ". Sparse optimizer fusion currently supports scalar pointwise expression math only.");
                }
                break;
        }
    }

    for (size_t outputIdx = 0; outputIdx < outputs.outputs.size(); ++outputIdx) {
        const NamedOutput& out = outputs.outputs[outputIdx];
        if (out.node_idx >= outputs.expr->nodes.size()) {
            throw std::runtime_error("Sparse row update fusion source output node index is out of range.");
        }
        body << "  " << vectorOutputStoreStmt("sru_out" + std::to_string(outputIdx), "sru_indexed_offset", ref(out.node_idx)) << "\n";
    }
    source.bodySource = body.str();
    return source;
}

}  // namespace

SparseRowUpdatePlan::SparseRowUpdatePlan(SparseRowUpdatePlan&& other) noexcept { *this = std::move(other); }

SparseRowUpdatePlan& SparseRowUpdatePlan::operator=(SparseRowUpdatePlan&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (module != nullptr) {
        try {
            CU_CHECK(cuModuleUnload(module));
        } catch (...) {
        }
    }
    module = other.module;
    kernel = other.kernel;
    kernelName = std::move(other.kernelName);
    deviceNum = other.deviceNum;
    capacity = other.capacity;
    vocabularySize = other.vocabularySize;
    embeddingDim = other.embeddingDim;
    valuesPerThread = other.valuesPerThread;
    rowDataType = other.rowDataType;
    rows = other.rows;
    numRows = other.numRows;
    inputSlots = std::move(other.inputSlots);
    outputSlots = std::move(other.outputSlots);
    indexedOutputsByName = std::move(other.indexedOutputsByName);
    other.module = nullptr;
    other.kernel = nullptr;
    return *this;
}

SparseRowUpdatePlan::~SparseRowUpdatePlan() {
    if (module != nullptr) {
        try {
            CU_CHECK(cuModuleUnload(module));
        } catch (...) {
        }
    }
}

SparseRowUpdateFusionSource SparseRowUpdatePlan::emitFusionSource(
    PhysicalOutputs outputs,
    const Tensor& rows,
    const Tensor& numRows,
    const std::unordered_map<std::string, SparseRowUpdateTensorBinding>& inputs,
    const std::unordered_map<std::string, Tensor>& indexedOutputs,
    const std::unordered_map<std::string, std::string>& localDenseLogicalInputExpressions) {
    SparseRowUpdateFusionBuild build = buildSparseRowUpdateFusionBuild(std::move(outputs),
                                                                       rows,
                                                                       numRows,
                                                                       inputs,
                                                                       indexedOutputs,
                                                                       localDenseLogicalInputExpressions);
    return emitVector4SparseRowUpdateFusionSource(build.outputs,
                                                  build.inputSlots,
                                                  build.outputSlots,
                                                  localDenseLogicalInputExpressions);
}

std::unique_ptr<SparseRowUpdatePlan> SparseRowUpdatePlan::compile(
    PhysicalOutputs outputs,
    const Tensor& rows,
    const Tensor& numRows,
    const std::unordered_map<std::string, SparseRowUpdateTensorBinding>& inputs,
    const std::unordered_map<std::string, Tensor>& indexedOutputs,
    int deviceNum,
    bool useFastMath) {
    if (!outputs.expr) {
        throw std::invalid_argument("Sparse row update compile requires expression outputs.");
    }
    if (outputs.outputs.empty()) {
        throw std::invalid_argument("Sparse row update compile requires at least one output.");
    }
    if (!rows.isInitialized() || !numRows.isInitialized()) {
        throw std::invalid_argument("Sparse row update rows and numRows tensors must be initialized.");
    }
    if (rows.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("Sparse row update rows tensor must live on GPU.");
    }
    const TensorPlacement placement = rows.getPlacement();
    validateSameGpuPlacement(numRows, placement, "numRows");
    validateDenseContiguous(rows, "rows");
    validateDenseContiguous(numRows, "numRows");
    if (!isSupportedSparseRowDType(rows.getDataType()) || numRows.getDataType() != rows.getDataType()) {
        throw std::invalid_argument(
            "Sparse row update rows and numRows tensors must both be uint16, uint32, or uint64 with matching dtype.");
    }
    if (numRows.getDimensions() != std::vector<uint64_t>{1}) {
        throw std::invalid_argument("Sparse row update numRows tensor must have shape [1].");
    }
    const std::vector<uint64_t> rowsDims = rows.getDimensions();
    if (rowsDims.size() != 1 || rowsDims[0] == 0) {
        throw std::invalid_argument("Sparse row update rows tensor must have non-empty shape [row_storage_capacity].");
    }
    const uint64_t rowsStorageCapacity = rowsDims[0];
    uint64_t capacity = 0;

    std::vector<DataType> inputDTypes = rootInputDTypes(*outputs.expr, inputs);
    resolveOutputsDTypesInPlace(outputs, inputDTypes);

    std::vector<RuntimeInputSlot> inputSlots(outputs.expr->inputs.size());
    uint64_t vocabularySize = 0;
    uint64_t embeddingDim = 0;

    for (const NamedInput& input : outputs.expr->inputs) {
        if (input.slot >= inputSlots.size()) {
            throw std::runtime_error("Sparse row update expression input slot out of range.");
        }
        RuntimeInputSlot slot;
        slot.name = input.name;
        slot.inputKind = input.kind;
        if (input.kind == NamedInput::Kind::Tensor) {
            auto it = inputs.find(input.name);
            if (it == inputs.end()) {
                throw std::invalid_argument("Sparse row update missing tensor input binding for '" + input.name + "'.");
            }
            slot.tensor = it->second.tensor;
            slot.tensorKind = it->second.kind;
            slot.dtype = slot.tensor.getDataType();
            validateSameGpuPlacement(slot.tensor, placement, "input '" + input.name + "'");
            validateDenseContiguous(slot.tensor, "input '" + input.name + "'");
            if (!isSupportedSparseRowValueDType(slot.dtype)) {
                throw std::invalid_argument("Sparse row update input '" + input.name + "' has unsupported dtype " + dtypeName(slot.dtype) +
                                            ". Supported dtypes are fp16, bf16, and fp32.");
            }
            const std::vector<uint64_t> dims = slot.tensor.getDimensions();
            if (dims.size() != 2 || dims[0] == 0 || dims[1] == 0) {
                throw std::invalid_argument("Sparse row update input '" + input.name + "' must have non-empty rank-2 shape.");
            }
            if (slot.tensorKind == SparseRowUpdateTensorKind::DenseLogicalRows) {
                if (capacity == 0) {
                    capacity = dims[0];
                } else if (dims[0] != capacity) {
                    throw std::invalid_argument("Sparse row update dense logical inputs must have matching first dimensions.");
                }
                if (rowsStorageCapacity < capacity) {
                    throw std::invalid_argument("Sparse row update rows tensor storage is smaller than the dense logical row capacity.");
                }
            } else {
                if (capacity != 0 && dims[0] < capacity) {
                    throw std::invalid_argument("Sparse row update indexed input '" + input.name +
                                                "' vocabulary dimension cannot be smaller than dense logical row capacity.");
                }
                if (vocabularySize == 0) {
                    vocabularySize = dims[0];
                } else if (vocabularySize != dims[0]) {
                    throw std::invalid_argument("Sparse row update indexed tensors must have the same vocabulary dimension.");
                }
            }
            if (embeddingDim == 0) {
                embeddingDim = dims[1];
            } else if (embeddingDim != dims[1]) {
                throw std::invalid_argument("Sparse row update tensors must have the same embedding dimension.");
            }
        } else if (input.kind == NamedInput::Kind::RuntimeScalarFp32) {
            slot.dtype = DataType::FP32;
        } else {
            throw std::invalid_argument("Sparse row update does not support tensor-backed runtime scalar inputs yet.");
        }
        inputSlots[input.slot] = std::move(slot);
    }
    for (size_t slotIdx = 0; slotIdx < inputSlots.size(); ++slotIdx) {
        if (inputSlots[slotIdx].name.empty()) {
            throw std::runtime_error("Sparse row update expression inputs must use contiguous input slots.");
        }
    }

    std::vector<RuntimeOutputSlot> outputSlots;
    outputSlots.reserve(outputs.outputs.size());
    std::unordered_map<std::string, Tensor> finalOutputs;
    for (const NamedOutput& namedOutput : outputs.outputs) {
        auto outIt = indexedOutputs.find(namedOutput.name);
        if (outIt == indexedOutputs.end()) {
            throw std::invalid_argument("Sparse row update missing indexed output binding for '" + namedOutput.name + "'.");
        }
        Tensor tensor = outIt->second;
        validateSameGpuPlacement(tensor, placement, "output '" + namedOutput.name + "'");
        validateDenseContiguous(tensor, "output '" + namedOutput.name + "'");
        if (!isSupportedSparseRowValueDType(tensor.getDataType())) {
            throw std::invalid_argument("Sparse row update output '" + namedOutput.name + "' has unsupported dtype " +
                                        dtypeName(tensor.getDataType()) + ". Supported dtypes are fp16, bf16, and fp32.");
        }
        const std::vector<uint64_t> dims = tensor.getDimensions();
        if (dims.size() != 2 || dims[0] == 0 || dims[1] == 0) {
            throw std::invalid_argument("Sparse row update indexed output '" + namedOutput.name + "' must have shape [vocabulary, D].");
        }
        if (vocabularySize == 0) {
            vocabularySize = dims[0];
        } else if (vocabularySize != dims[0]) {
            throw std::invalid_argument("Sparse row update indexed outputs must have the same vocabulary dimension as indexed inputs.");
        }
        if (embeddingDim == 0) {
            embeddingDim = dims[1];
        } else if (embeddingDim != dims[1]) {
            throw std::invalid_argument("Sparse row update indexed outputs must have the same embedding dimension as inputs.");
        }
        if (namedOutput.node_idx >= outputs.expr->nodes.size() || !outputs.expr->nodes[namedOutput.node_idx].output_dtype.has_value()) {
            throw std::runtime_error("Sparse row update output node is missing resolved dtype.");
        }
        const DataType expressionOutputDType = outputs.expr->nodes[namedOutput.node_idx].output_dtype.value();
        if (expressionOutputDType != tensor.getDataType()) {
            throw std::invalid_argument("Sparse row update expression output '" + namedOutput.name + "' dtype " +
                                        dtypeName(expressionOutputDType) + " does not match bound output tensor dtype " +
                                        dtypeName(tensor.getDataType()) + ".");
        }
        outputSlots.push_back(RuntimeOutputSlot{namedOutput.name, tensor, tensor.getDataType()});
        finalOutputs[namedOutput.name] = tensor;
    }

    if (capacity == 0) {
        capacity = rowsStorageCapacity;
    }
    if (rowsStorageCapacity < capacity) {
        throw std::invalid_argument("Sparse row update rows tensor storage is smaller than the logical sparse-row capacity.");
    }
    if (vocabularySize != 0 && vocabularySize < capacity) {
        throw std::invalid_argument(
            "Sparse row update indexed tensor vocabulary dimension cannot be smaller than logical sparse-row capacity.");
    }

    if (vocabularySize == 0 || embeddingDim == 0) {
        throw std::invalid_argument("Sparse row update requires at least one indexed input or output tensor to infer [vocabulary, D].");
    }
    if (!rowDTypeCanRepresentVocabularySize(rows.getDataType(), vocabularySize)) {
        throw std::invalid_argument("Sparse row update row dtype " + dtypeName(rows.getDataType()) +
                                    " cannot represent vocabulary_size as the invalid-row sentinel.");
    }
    checkedProduct(capacity, embeddingDim, "launch domain");

    const bool useVector4Update = (embeddingDim != 0 && embeddingDim % 4 == 0);
    const std::string kernelName = "thor_sparse_row_update_v" + safeKernelSuffix(vocabularySize) + "_d" + safeKernelSuffix(embeddingDim) +
                                   (useVector4Update ? "_vec4" : "") + "_r" + safeKernelSuffix(static_cast<uint64_t>(rows.getDataType()));
    const std::string src =
        emitKernelSource(outputs, inputSlots, outputSlots, rows.getDataType(), capacity, vocabularySize, embeddingDim, kernelName);

    ScopedGpu scoped(deviceNum);
    CUDA_CHECK(cudaFree(nullptr));
    EquationSignature sig = FusedEquation::buildSignature(static_cast<uint32_t>(inputSlots.size()), deviceNum, useFastMath);
    std::vector<char> lto = EquationCompiler::compileToLtoIr(src, kernelName, sig);
    std::vector<char> cubin = EquationCompiler::linkToCubin(lto, sig);

    auto plan = std::make_unique<SparseRowUpdatePlan>();
    plan->kernelName = kernelName;
    plan->deviceNum = deviceNum;
    plan->capacity = capacity;
    plan->vocabularySize = vocabularySize;
    plan->embeddingDim = embeddingDim;
    plan->valuesPerThread = (embeddingDim != 0 && embeddingDim % 4 == 0) ? 4U : 1U;
    plan->rowDataType = rows.getDataType();
    plan->rows = rows;
    plan->numRows = numRows;
    plan->inputSlots = std::move(inputSlots);
    plan->outputSlots = std::move(outputSlots);
    plan->indexedOutputsByName = std::move(finalOutputs);

    CU_CHECK(cuModuleLoadData(&plan->module, cubin.data()));
    CU_CHECK(cuModuleGetFunction(&plan->kernel, plan->module, kernelName.c_str()));
    return plan;
}

void SparseRowUpdatePlan::run(const std::unordered_map<std::string, float>& runtimeScalars, Stream stream) const {
    if (kernel == nullptr) {
        throw std::runtime_error("Sparse row update plan has not been compiled.");
    }
    THOR_THROW_IF_FALSE(rows.isInitialized());
    THOR_THROW_IF_FALSE(numRows.isInitialized());

    const uint64_t vectorsPerRow = (embeddingDim + static_cast<uint64_t>(valuesPerThread) - 1ULL) / static_cast<uint64_t>(valuesPerThread);
    const uint64_t maxElements = checkedProduct(capacity, vectorsPerRow, "max launch elements");
    if (maxElements == 0) {
        return;
    }
    const uint32_t block = sparseUpdateBlockSize(maxElements);
    const uint32_t grid = sparseUpdateMaxGrid(maxElements, block);

    SparseRowUpdateKernelArgs kernelArgs = buildSparseRowUpdateKernelArgs(rows, numRows, inputSlots, outputSlots, runtimeScalars);

    ScopedGpu scoped(deviceNum);
    CU_CHECK(cuLaunchKernel(kernel, grid, 1, 1, block, 1, 1, 0, stream, kernelArgs.args.data(), nullptr));
}

void SparseRowUpdatePlan::capture(CudaGraphCaptureBuilder& builder,
                                  CapturedSparseRowUpdate& captured,
                                  const std::unordered_map<std::string, float>& runtimeScalars) const {
    if (kernel == nullptr) {
        throw std::runtime_error("Sparse row update plan has not been compiled.");
    }
    THOR_THROW_IF_FALSE(rows.isInitialized());
    THOR_THROW_IF_FALSE(numRows.isInitialized());
    if (!captured.targetNodeHandle.isInitialized()) {
        throw std::invalid_argument(
            "Sparse row update graph capture requires a preallocated target-node handle. Allocate CapturedSparseRowUpdate before stream "
            "capture begins.");
    }
    if (captured.targetNodeHandle.getGpuNum() != deviceNum) {
        throw std::invalid_argument("Sparse row update graph capture target-node handle must live on the compiled plan GPU.");
    }
    if (builder.stream().getGpuNum() != deviceNum) {
        throw std::invalid_argument("Sparse row update graph capture stream must be on the compiled plan GPU.");
    }

    const uint64_t vectorsPerRow = (embeddingDim + static_cast<uint64_t>(valuesPerThread) - 1ULL) / static_cast<uint64_t>(valuesPerThread);
    const uint64_t maxElements = checkedProduct(capacity, vectorsPerRow, "max graph launch elements");
    if (maxElements == 0) {
        throw std::runtime_error("Sparse row update graph capture requires a non-empty launch domain.");
    }
    const uint32_t block = sparseUpdateBlockSize(maxElements);
    const uint32_t maxGrid = sparseUpdateMaxGrid(maxElements, block);

    launchUpdateDeviceGrid1DFromScalar(DynamicGrid1DFromScalarDescriptor{&captured.targetNodeHandle,
                                                                         numRows,
                                                                         vectorsPerRow,
                                                                         block,
                                                                         /*minGridDimX=*/1,
                                                                         maxGrid},
                                       builder.stream());

    SparseRowUpdateKernelArgs kernelArgs = buildSparseRowUpdateKernelArgs(rows, numRows, inputSlots, outputSlots, runtimeScalars);

    ScopedGpu scoped(deviceNum);
    captured.targetNode = builder.captureDeviceUpdatableKernel(
        CudaGraphKernelLaunch{kernel, dim3(1, 1, 1), dim3(block, 1, 1), 0, kernelArgs.args.data(), nullptr});
}

std::vector<std::string> SparseRowUpdatePlan::outputNames() const {
    std::vector<std::string> names;
    names.reserve(outputSlots.size());
    for (const RuntimeOutputSlot& output : outputSlots) {
        names.push_back(output.name);
    }
    return names;
}

}  // namespace ThorImplementation
