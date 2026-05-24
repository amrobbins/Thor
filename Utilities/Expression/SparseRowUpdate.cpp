#include "Utilities/Expression/SparseRowUpdate.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/CudaDriver/CudaGraphDynamicGrid.h"
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

bool isSupportedSparseRowDType(DataType dtype) { return dtype == DataType::UINT16 || dtype == DataType::UINT32 || dtype == DataType::UINT64; }

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
                    throw std::invalid_argument("Sparse row update missing tensor input binding for expression input '" + input.name + "'.");
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

std::string emitKernelSource(const PhysicalOutputs& outputs,
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

    const std::string rowType = scalarStorageType(rowDataType);
    std::ostringstream ss;
    ss << "#include <cuda_fp16.h>\n";
    ss << "#include <cuda_bf16.h>\n";
    ss << "#include <math_functions.h>\n\n";
    ss << "__device__ __forceinline__ float thor_sparse_load_float(const float* p, unsigned long long i) { return p[i]; }\n";
    ss << "__device__ __forceinline__ float thor_sparse_load_float(const __half* p, unsigned long long i) { return __half2float(p[i]); }\n";
    ss << "__device__ __forceinline__ float thor_sparse_load_float(const __nv_bfloat16* p, unsigned long long i) { return __bfloat162float(p[i]); }\n";
    ss << "__device__ __forceinline__ void thor_sparse_store_float(float* p, unsigned long long i, float v) { p[i] = v; }\n";
    ss << "__device__ __forceinline__ void thor_sparse_store_float(__half* p, unsigned long long i, float v) { p[i] = __float2half_rn(v); }\n";
    ss << "__device__ __forceinline__ void thor_sparse_store_float(__nv_bfloat16* p, unsigned long long i, float v) { p[i] = __float2bfloat16(v); }\n\n";

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
                    throw std::runtime_error("Sparse row update expression uses unsupported op " + std::to_string(static_cast<int>(node.op)) +
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

SparseRowUpdateKernelArgs buildSparseRowUpdateKernelArgs(
    const Tensor& rows,
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
        throw std::invalid_argument("Sparse row update rows and numRows tensors must both be uint16, uint32, or uint64 with matching dtype.");
    }
    if (numRows.getDimensions() != std::vector<uint64_t>{1}) {
        throw std::invalid_argument("Sparse row update numRows tensor must have shape [1].");
    }
    const std::vector<uint64_t> rowsDims = rows.getDimensions();
    if (rowsDims.size() != 1 || rowsDims[0] == 0) {
        throw std::invalid_argument("Sparse row update rows tensor must have non-empty shape [capacity].");
    }
    const uint64_t capacity = rowsDims[0];

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
            if (dims.size() != 2 || dims[1] == 0) {
                throw std::invalid_argument("Sparse row update input '" + input.name + "' must have rank-2 shape.");
            }
            if (slot.tensorKind == SparseRowUpdateTensorKind::DenseLogicalRows) {
                if (dims[0] != capacity) {
                    throw std::invalid_argument("Sparse row update dense logical input '" + input.name + "' first dimension must equal rows capacity.");
                }
            } else {
                if (dims[0] < capacity) {
                    throw std::invalid_argument("Sparse row update indexed input '" + input.name + "' vocabulary dimension cannot be smaller than capacity.");
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

    if (vocabularySize == 0 || embeddingDim == 0) {
        throw std::invalid_argument("Sparse row update requires at least one indexed input or output tensor to infer [vocabulary, D].");
    }
    if (!rowDTypeCanRepresentVocabularySize(rows.getDataType(), vocabularySize)) {
        throw std::invalid_argument("Sparse row update row dtype " + dtypeName(rows.getDataType()) +
                                    " cannot represent vocabulary_size as the invalid-row sentinel.");
    }
    checkedProduct(capacity, embeddingDim, "launch domain");

    const std::string kernelName = "thor_sparse_row_update_v" + safeKernelSuffix(vocabularySize) + "_d" + safeKernelSuffix(embeddingDim) +
                                   "_r" + safeKernelSuffix(static_cast<uint64_t>(rows.getDataType()));
    const std::string src = emitKernelSource(outputs, inputSlots, outputSlots, rows.getDataType(), capacity, vocabularySize, embeddingDim, kernelName);

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

    const uint64_t maxElements = checkedProduct(capacity, embeddingDim, "max launch elements");
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
        throw std::invalid_argument("Sparse row update graph capture requires a preallocated target-node handle. Allocate CapturedSparseRowUpdate before stream capture begins.");
    }
    if (captured.targetNodeHandle.getGpuNum() != deviceNum) {
        throw std::invalid_argument("Sparse row update graph capture target-node handle must live on the compiled plan GPU.");
    }
    if (builder.stream().getGpuNum() != deviceNum) {
        throw std::invalid_argument("Sparse row update graph capture stream must be on the compiled plan GPU.");
    }

    const uint64_t maxElements = checkedProduct(capacity, embeddingDim, "max graph launch elements");
    if (maxElements == 0) {
        throw std::runtime_error("Sparse row update graph capture requires a non-empty launch domain.");
    }
    const uint32_t block = sparseUpdateBlockSize(maxElements);
    const uint32_t maxGrid = sparseUpdateMaxGrid(maxElements, block);

    launchUpdateDeviceGrid1DFromScalar(DynamicGrid1DFromScalarDescriptor{&captured.targetNodeHandle,
                                                                         numRows,
                                                                         embeddingDim,
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
