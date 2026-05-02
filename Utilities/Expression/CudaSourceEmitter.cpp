#include "Utilities/Expression/CudaSourceEmitter.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"

#include <algorithm>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

using DataType = ThorImplementation::TensorDescriptor::DataType;

namespace ThorImplementation {

static std::string emitScalarFpLiteral(double x) {
    auto formatFloating = [](double v, int precision) -> std::string {
        std::ostringstream oss;
        oss << std::setprecision(precision) << std::defaultfloat << v;
        std::string s = oss.str();

        if (s.find('.') == std::string::npos && s.find('e') == std::string::npos && s.find('E') == std::string::npos) {
            s += ".0";
        }

        return s;
    };

    return formatFloating(x, 9) + "f";
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

static bool fitsInUInt32(uint64_t x) { return x <= std::numeric_limits<uint32_t>::max(); }

static std::string emittedIndexType(bool use_uint32_index_math) { return use_uint32_index_math ? "unsigned int" : "unsigned long long"; }

static std::string scalarStorageType(DataType dtype);
static std::string chunkScalarTypeForBytes(uint32_t bytes);

static std::string emitFlatThreadIndexExpr(bool use_uint32_index_math) {
    if (use_uint32_index_math) {
        return "blockIdx.x * blockDim.x + threadIdx.x";
    }
    return "static_cast<unsigned long long>(blockIdx.x) * static_cast<unsigned long long>(blockDim.x) + "
           "static_cast<unsigned long long>(threadIdx.x)";
}

static std::string float4LaneComponent(uint32_t lane) {
    switch (lane) {
        case 0:
            return "x";
        case 1:
            return "y";
        case 2:
            return "z";
        case 3:
            return "w";
        default:
            throw runtime_error("float4 lane out of range.");
    }
}

static std::string emitChunkLaneReadExpr(const std::string& chunk_expr,
                                         const std::string& chunk_type,
                                         DataType scalar_dtype,
                                         const std::string& lane_expr,
                                         const std::string& chunk_data_expr = "") {
    if (chunk_type == "float4" && scalar_dtype == DataType::FP32) {
        const uint32_t lane = static_cast<uint32_t>(std::stoul(lane_expr));
        return chunk_expr + "." + float4LaneComponent(lane);
    }
    if (!chunk_data_expr.empty()) {
        return chunk_data_expr + "[" + lane_expr + "]";
    }
    return "reinterpret_cast<const " + scalarStorageType(scalar_dtype) + "*>(&" + chunk_expr + ")[" + lane_expr + "]";
}

static std::string emitChunkLaneWriteStmt(const std::string& chunk_expr,
                                          const std::string& chunk_type,
                                          DataType scalar_dtype,
                                          uint32_t lane,
                                          const std::string& value_expr,
                                          const std::string& indent,
                                          const std::string& chunk_data_expr = "") {
    if (chunk_type == "float4" && scalar_dtype == DataType::FP32) {
        return indent + chunk_expr + "." + float4LaneComponent(lane) + " = " + value_expr + ";\n";
    }
    if (!chunk_data_expr.empty()) {
        return indent + chunk_data_expr + "[" + std::to_string(lane) + "] = " + value_expr + ";\n";
    }
    return indent + "reinterpret_cast<" + scalarStorageType(scalar_dtype) + "*>(&" + chunk_expr + ")[" + std::to_string(lane) +
           "] = " + value_expr + ";\n";
}

static std::string emitUnsignedLiteral(uint64_t value, bool use_uint32_index_math) {
    if (use_uint32_index_math && fitsInUInt32(value)) {
        return std::to_string(static_cast<uint32_t>(value)) + "U";
    }
    return std::to_string(value) + "ULL";
}

static bool groupSupportsUInt32IndexMath(const SpecializedBroadcastGroup& group) {
    if (!fitsInUInt32(group.numel) || !fitsInUInt32((group.numel + 1ULL) >> 1)) {
        return false;
    }

    for (const SpecializedBroadcastAxis& axis : group.active_axes) {
        if (!fitsInUInt32(axis.dim) || !fitsInUInt32(axis.output_stride)) {
            return false;
        }
        for (uint64_t stride : axis.input_strides) {
            if (!fitsInUInt32(stride)) {
                return false;
            }
        }
    }

    return true;
}

static bool groupsSupportUInt32IndexMath(const std::vector<SpecializedBroadcastGroup>& groups) {
    return std::all_of(
        groups.begin(), groups.end(), [](const SpecializedBroadcastGroup& group) { return groupSupportsUInt32IndexMath(group); });
}

bool CudaSourceEmitter::specializedBroadcastUsesUInt32IndexMath(const std::vector<SpecializedBroadcastGroup>& groups) {
    return groupsSupportUInt32IndexMath(groups);
}

static void emitSpecializedBroadcastOffsetMath(std::ostringstream& ss,
                                               const SpecializedBroadcastGroup& group,
                                               const std::vector<size_t>& used_input_indices,
                                               const std::string& idx_expr,
                                               const std::string& offset_suffix,
                                               const std::string& indent,
                                               bool use_uint32_index_math) {
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
            base_expr = "(" + idx_expr + " / " + emitUnsignedLiteral(axis.output_stride, use_uint32_index_math) + ")";
        }

        std::string coord_expr;
        if (axis.dim == 1) {
            coord_expr = emitUnsignedLiteral(0, use_uint32_index_math);
        } else if (isPowerOfTwo(axis.dim)) {
            coord_expr = "(" + base_expr + " & " + emitUnsignedLiteral(axis.dim - 1, use_uint32_index_math) + ")";
        } else {
            coord_expr = "(" + base_expr + " % " + emitUnsignedLiteral(axis.dim, use_uint32_index_math) + ")";
        }

        ss << indent << "const " << emittedIndexType(use_uint32_index_math) << " " << coord << " = " << coord_expr << ";\n";

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
                ss << indent << "in" << input_slot << "_offset" << offset_suffix << " += " << coord << " * "
                   << emitUnsignedLiteral(stride, use_uint32_index_math) << ";\n";
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

static DataType requireNodeOutputDType(const ExprNode& node) {
    if (!node.output_dtype.isPresent()) {
        throw runtime_error("Fused stage node is missing resolved output_dtype.");
    }
    return node.output_dtype.get();
}

static DataType requireNodeInputTensorDType(const ExprNode& node) {
    if (!node.input_tensor_dtype.isPresent()) {
        throw runtime_error("Fused stage INPUT node is missing resolved input_tensor_dtype.");
    }
    return node.input_tensor_dtype.get();
}

static DataType requireNodeComputeDType(const ExprNode& node) {
    if (!node.compute_dtype.isPresent()) {
        throw runtime_error("Fused stage node is missing resolved compute_dtype.");
    }
    return node.compute_dtype.get();
}

static std::string scalarStorageType(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return "float";
        case DataType::FP16:
            return "half";
        case DataType::BF16:
            return "__nv_bfloat16";
        case DataType::FP8_E4M3:
            return "__nv_fp8_e4m3";
        case DataType::FP8_E5M2:
            return "__nv_fp8_e5m2";
        default:
            throw runtime_error("Unsupported scalar storage dtype in fused stage emitter: " + TensorDescriptor::getElementTypeName(dtype));
    }
}

static uint32_t scalarStorageTypeSizeBytes(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return 4;
        case DataType::FP16:
        case DataType::BF16:
            return 2;
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return 1;
        default:
            throw runtime_error("Unsupported scalar storage dtype size in transpose emitter: " +
                                TensorDescriptor::getElementTypeName(dtype));
    }
}

static uint32_t transposePackScalars(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
        case DataType::BF16:
            return 2;
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return 4;
        default:
            return 1;
    }
}

static std::string transposePackType(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return "half2";
        case DataType::BF16:
            return "__nv_bfloat162";
        case DataType::FP8_E4M3:
            return "__nv_fp8x4_e4m3";
        case DataType::FP8_E5M2:
            return "__nv_fp8x4_e5m2";
        default:
            throw runtime_error("Unsupported transpose pack dtype: " + TensorDescriptor::getElementTypeName(dtype));
    }
}

static std::string transposeVector2StorageType(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return "float2";
        case DataType::FP16:
            return "half2";
        case DataType::BF16:
            return "__nv_bfloat162";
        case DataType::FP8_E4M3:
            return "__nv_fp8x2_e4m3";
        case DataType::FP8_E5M2:
            return "__nv_fp8x2_e5m2";
        default:
            throw runtime_error("Unsupported transpose vector2 dtype: " + TensorDescriptor::getElementTypeName(dtype));
    }
}

static bool isTwoByteFloatDType(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::BF16; }

static bool isFp8DType(DataType dtype) { return dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2; }

static bool isHalf2ComputeStorageDType(DataType dtype) { return dtype == DataType::FP16 || isFp8DType(dtype); }

static void emitRequiredHeaders(const PhysicalExpression& expr, std::ostringstream& ss) {
    bool need_fp16 = false;
    bool need_bf16 = false;
    bool need_fp8 = false;

    auto note_dtype = [&](DataType dtype) {
        switch (dtype) {
            case DataType::FP16:
                need_fp16 = true;
                break;
            case DataType::BF16:
                need_bf16 = true;
                break;
            case DataType::FP8_E4M3:
            case DataType::FP8_E5M2:
                need_fp16 = true;
                need_fp8 = true;
                break;
            default:
                break;
        }
    };

    for (const ExprNode& node : expr.nodes) {
        if (node.input_tensor_dtype.isPresent()) {
            note_dtype(node.input_tensor_dtype.get());
        }
        if (node.output_dtype.isPresent()) {
            note_dtype(node.output_dtype.get());
        }
        if (node.compute_dtype.isPresent()) {
            note_dtype(node.compute_dtype.get());
        }
    }

    if (need_fp16) {
        ss << "#include <cuda_fp16.h>\n";
    }
    if (need_bf16) {
        ss << "#include <cuda_bf16.h>\n";
    }
    if (need_fp8) {
        ss << "#include <cuda_fp8.h>\n";
    }
}

static std::vector<DataType> collectInputSlotDTypes(const PhysicalExpression& expr) {
    std::vector<DataType> input_dtypes(expr.numInputs(), DataType::FP32);
    std::vector<uint8_t> seen(expr.numInputs(), 0);

    for (const ExprNode& node : expr.nodes) {
        if (node.op != ExprOp::INPUT && node.op != ExprOp::RUNTIME_SCALAR && node.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
            continue;
        }

        if (node.input_slot >= input_dtypes.size()) {
            throw runtime_error("Input slot out of range while collecting fused stage input dtypes.");
        }

        const DataType dtype = requireNodeInputTensorDType(node);
        if (seen[node.input_slot]) {
            if (input_dtypes[node.input_slot] != dtype) {
                throw runtime_error("Inconsistent resolved input_tensor_dtype for fused stage input slot.");
            }
        } else {
            input_dtypes[node.input_slot] = dtype;
            seen[node.input_slot] = 1;
        }
    }

    for (uint32_t slot = 0; slot < input_dtypes.size(); ++slot) {
        if (!seen[slot]) {
            std::ostringstream ss;
            ss << "Unused or unresolved input slot encountered in fused stage: slot=" << slot;
            if (slot < expr.inputs.size()) {
                ss << ", name='" << expr.inputs[slot].name << "'";
                ss << ", kind=";
                switch (expr.inputs[slot].kind) {
                    case NamedInput::Kind::Tensor:
                        ss << "Tensor";
                        break;
                    case NamedInput::Kind::RuntimeScalarFp32:
                        ss << "RuntimeScalar";
                        break;
                    case NamedInput::Kind::TensorRuntimeScalar:
                        ss << "TensorRuntimeScalar";
                        break;
                }
            }
            ss << ". This usually means the fused stage declared an input in expr.inputs "
                  "that no INPUT/RUNTIME_SCALAR/TENSOR_RUNTIME_SCALAR node actually referenced.";
            throw runtime_error(ss.str());
        }
    }

    return input_dtypes;
}

static std::vector<DataType> collectOutputDTypes(const PhysicalExecutionStage& stage) {
    std::vector<DataType> output_dtypes;
    output_dtypes.reserve(stage.outputs.size());

    for (const CompiledStageOutput& output : stage.outputs) {
        if (output.local_node_idx >= stage.expr.nodes.size()) {
            throw runtime_error("Stage output local_node_idx out of range.");
        }
        output_dtypes.push_back(requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]));
    }

    return output_dtypes;
}

static std::vector<DataType> collectOutputDTypes(const CompiledExecutionStage& stage) {
    std::vector<DataType> output_dtypes;
    output_dtypes.reserve(stage.outputs.size());

    for (const CompiledStageOutput& output : stage.outputs) {
        if (output.local_node_idx >= stage.expr.nodes.size()) {
            throw runtime_error("Stage output local_node_idx out of range.");
        }
        output_dtypes.push_back(requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]));
    }

    return output_dtypes;
}

static bool stageHasTransposedMaterializedOutput(const std::vector<CompiledStageOutput>& outputs) {
    return std::any_of(outputs.begin(), outputs.end(), [](const CompiledStageOutput& output) {
        return output.materialized_layout == MaterializedTensorLayout::Transposed;
    });
}

static const CompiledStageOutput& requireSingleTransposedMaterializedOutput(const PhysicalExecutionStage& stage) {
    if (stage.outputs.size() != 1) {
        throw runtime_error("Transposed fused materialization currently supports exactly one output.");
    }
    const CompiledStageOutput& output = stage.outputs[0];
    if (output.materialized_layout != MaterializedTensorLayout::Transposed) {
        throw runtime_error("Expected a transposed materialized fused output.");
    }
    if (output.local_node_idx >= stage.expr.nodes.size()) {
        throw runtime_error("Transposed fused output local_node_idx out of range.");
    }
    return output;
}

static const CompiledStageOutput& requireSingleTransposedMaterializedOutput(const CompiledExecutionStage& stage) {
    if (stage.outputs.size() != 1) {
        throw runtime_error("Transposed fused materialization currently supports exactly one output.");
    }
    const CompiledStageOutput& output = stage.outputs[0];
    if (output.materialized_layout != MaterializedTensorLayout::Transposed) {
        throw runtime_error("Expected a transposed materialized fused output.");
    }
    if (output.local_node_idx >= stage.expr.nodes.size()) {
        throw runtime_error("Transposed fused output local_node_idx out of range.");
    }
    return output;
}

static Optional<DataType> getVectorizedStageStorageDTypeImpl(const PhysicalExpression& expr,
                                                             const std::vector<DataType>& input_dtypes,
                                                             const std::vector<DataType>& output_dtypes) {
    if (input_dtypes.empty() || output_dtypes.empty()) {
        return Optional<DataType>::empty();
    }

    Optional<DataType> maybe_stage_dtype = Optional<DataType>::empty();
    for (uint32_t slot = 0; slot < expr.inputs.size(); ++slot) {
        if (expr.inputs[slot].kind != NamedInput::Kind::Tensor) {
            continue;
        }

        const DataType dtype = input_dtypes.at(slot);
        if (!maybe_stage_dtype.isPresent()) {
            maybe_stage_dtype = dtype;
        } else if (maybe_stage_dtype.get() != dtype) {
            return Optional<DataType>::empty();
        }
    }

    if (!maybe_stage_dtype.isPresent()) {
        return Optional<DataType>::empty();
    }

    const DataType stage_dtype = maybe_stage_dtype.get();
    if (stage_dtype != DataType::FP16 && stage_dtype != DataType::BF16 && stage_dtype != DataType::FP8_E4M3 &&
        stage_dtype != DataType::FP8_E5M2) {
        return Optional<DataType>::empty();
    }

    for (DataType dtype : output_dtypes) {
        if (dtype != stage_dtype) {
            return Optional<DataType>::empty();
        }
    }

    const DataType expected_compute_dtype = defaultComputeDType(stage_dtype);

    for (const ExprNode& node : expr.nodes) {
        if (node.op == ExprOp::SCALAR_FP || node.op == ExprOp::RUNTIME_SCALAR || node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            continue;
        }

        if (node.op == ExprOp::INPUT) {
            if (requireNodeInputTensorDType(node) != stage_dtype) {
                return Optional<DataType>::empty();
            }
        }

        if (requireNodeOutputDType(node) != stage_dtype) {
            return Optional<DataType>::empty();
        }

        if (node.op != ExprOp::INPUT) {
            if (requireNodeComputeDType(node) != expected_compute_dtype) {
                return Optional<DataType>::empty();
            }
        }
    }

    return stage_dtype;
}

Optional<DataType> CudaSourceEmitter::getVectorizedStageStorageDType(const PhysicalExecutionStage& stage) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel) {
        return Optional<DataType>::empty();
    }
    return getVectorizedStageStorageDTypeImpl(stage.expr, collectInputSlotDTypes(stage.expr), collectOutputDTypes(stage));
}

Optional<DataType> CudaSourceEmitter::getVectorizedStageStorageDType(const CompiledExecutionStage& stage) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        return Optional<DataType>::empty();
    }
    return getVectorizedStageStorageDTypeImpl(stage.expr, collectInputSlotDTypes(stage.expr), collectOutputDTypes(stage));
}

static Optional<DataType> getSingleTensorInputStorageDType(const PhysicalExpression& expr, const std::vector<DataType>& input_dtypes) {
    Optional<DataType> maybe_tensor_dtype = Optional<DataType>::empty();

    for (uint32_t slot = 0; slot < expr.inputs.size(); ++slot) {
        if (expr.inputs[slot].kind != NamedInput::Kind::Tensor) {
            continue;
        }

        const DataType dtype = input_dtypes.at(slot);
        if (!maybe_tensor_dtype.isPresent()) {
            maybe_tensor_dtype = dtype;
        } else if (maybe_tensor_dtype.get() != dtype) {
            return Optional<DataType>::empty();
        }
    }

    return maybe_tensor_dtype;
}

static bool supportsMixedTwoByteFloat2TransposedVectorization(const PhysicalExpression& expr, DataType input_dtype, DataType output_dtype) {
    if (input_dtype == output_dtype || !isTwoByteFloatDType(input_dtype) || !isTwoByteFloatDType(output_dtype)) {
        return false;
    }

    for (const ExprNode& node : expr.nodes) {
        switch (node.op) {
            case ExprOp::INPUT:
                if (requireNodeInputTensorDType(node) != input_dtype) {
                    return false;
                }
                break;
            case ExprOp::SCALAR_FP:
            case ExprOp::FILL:
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
                break;
            default:
                if (requireNodeComputeDType(node) != DataType::FP32) {
                    return false;
                }
                break;
        }
    }

    return true;
}

static bool supportsMixedFp8TransposedVectorization(const PhysicalExpression& expr, DataType input_dtype, DataType output_dtype) {
    if (input_dtype == output_dtype || !isFp8DType(input_dtype) || !isFp8DType(output_dtype)) {
        return false;
    }

    for (const ExprNode& node : expr.nodes) {
        switch (node.op) {
            case ExprOp::INPUT:
                if (requireNodeInputTensorDType(node) != input_dtype) {
                    return false;
                }
                break;
            case ExprOp::SCALAR_FP:
            case ExprOp::FILL:
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
                break;
            default:
                if (requireNodeComputeDType(node) != DataType::FP16) {
                    return false;
                }
                break;
        }
    }

    return true;
}

static bool supportsFloat2TransposedVectorization(const PhysicalExpression& expr, DataType input_dtype, DataType output_dtype) {
    if (transposePackScalars(output_dtype) <= 1 || input_dtype == output_dtype) {
        return false;
    }

    // This path is for cross-width/cross-family cases that should compute the fused
    // producer in fp32 pairs before packing the low-precision transposed output.
    if (output_dtype == DataType::FP32) {
        return false;
    }

    for (const ExprNode& node : expr.nodes) {
        switch (node.op) {
            case ExprOp::INPUT:
                if (requireNodeInputTensorDType(node) != input_dtype) {
                    return false;
                }
                break;
            case ExprOp::SCALAR_FP:
            case ExprOp::FILL:
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
                break;
            default:
                if (requireNodeComputeDType(node) != DataType::FP32) {
                    return false;
                }
                break;
        }
    }

    return true;
}

static bool supportsHalf2TransposedVectorization(const PhysicalExpression& expr, DataType input_dtype, DataType output_dtype) {
    if (input_dtype == output_dtype || !isHalf2ComputeStorageDType(input_dtype) || !isHalf2ComputeStorageDType(output_dtype)) {
        return false;
    }
    if (transposePackScalars(output_dtype) <= 1) {
        return false;
    }

    for (const ExprNode& node : expr.nodes) {
        switch (node.op) {
            case ExprOp::INPUT:
                if (requireNodeInputTensorDType(node) != input_dtype) {
                    return false;
                }
                break;
            case ExprOp::SCALAR_FP:
            case ExprOp::FILL:
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
                break;
            default:
                if (requireNodeComputeDType(node) != DataType::FP16) {
                    return false;
                }
                break;
        }
    }

    return true;
}

static bool supportsFp8ToBf16Float2TransposedVectorization(const PhysicalExpression& expr, DataType input_dtype, DataType output_dtype) {
    if (!isFp8DType(input_dtype) || output_dtype != DataType::BF16) {
        return false;
    }

    for (const ExprNode& node : expr.nodes) {
        switch (node.op) {
            case ExprOp::INPUT:
                if (requireNodeInputTensorDType(node) != input_dtype) {
                    return false;
                }
                break;
            case ExprOp::SCALAR_FP:
            case ExprOp::FILL:
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
                break;
            default: {
                const DataType compute_dtype = requireNodeComputeDType(node);
                if (compute_dtype != DataType::FP32 && compute_dtype != DataType::BF16) {
                    return false;
                }
                break;
            }
        }
    }

    return true;
}

static uint32_t dataTypeStorageBytes(DataType dtype) {
    switch (dtype) {
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return 1;
        case DataType::FP16:
        case DataType::BF16:
            return 2;
        case DataType::FP32:
            return 4;
        default:
            throw runtime_error("Unsupported dtype in dataTypeStorageBytes.");
    }
}

static uint32_t flatScalarElementsPerThreadImpl(const std::vector<DataType>& input_dtypes, const std::vector<DataType>& output_dtypes) {
    uint32_t max_storage_bytes = 1;

    for (DataType dtype : input_dtypes) {
        max_storage_bytes = std::max(max_storage_bytes, dataTypeStorageBytes(dtype));
    }
    for (DataType dtype : output_dtypes) {
        max_storage_bytes = std::max(max_storage_bytes, dataTypeStorageBytes(dtype));
    }

    return std::max<uint32_t>(1, 16u / max_storage_bytes);
}

uint32_t CudaSourceEmitter::flatElementsPerThread(const PhysicalExecutionStage& stage) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel) {
        return 1;
    }
    if (stageHasTransposedMaterializedOutput(stage.outputs)) {
        return 1;
    }

    const Optional<DataType> vectorized_dtype = getVectorizedStageStorageDType(stage);
    if (vectorized_dtype.isPresent()) {
        switch (vectorized_dtype.get()) {
            case DataType::FP16:
            case DataType::BF16:
                return 8;
            case DataType::FP8_E4M3:
            case DataType::FP8_E5M2:
                return 16;
            default:
                break;
        }
    }

    return flatScalarElementsPerThreadImpl(collectInputSlotDTypes(stage.expr), collectOutputDTypes(stage));
}

uint32_t CudaSourceEmitter::tiledTransposePackScalars(const PhysicalExecutionStage& stage) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel || !stageHasTransposedMaterializedOutput(stage.outputs)) {
        return 1;
    }

    const CompiledStageOutput& output = requireSingleTransposedMaterializedOutput(stage);
    const DataType output_dtype = requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]);
    return transposePackScalars(output_dtype);
}

uint32_t CudaSourceEmitter::tiledTransposePackScalars(const CompiledExecutionStage& stage) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel || !stageHasTransposedMaterializedOutput(stage.outputs)) {
        return 1;
    }

    const CompiledStageOutput& output = requireSingleTransposedMaterializedOutput(stage);
    const DataType output_dtype = requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]);
    return transposePackScalars(output_dtype);
}

static std::string castScalarExpr(const std::string& expr, DataType src_dtype, DataType dst_dtype) {
    if (src_dtype == dst_dtype) {
        return expr;
    }

    switch (dst_dtype) {
        case DataType::FP32:
            switch (src_dtype) {
                case DataType::FP32:
                    return expr;
                case DataType::FP16:
                case DataType::BF16:
                    return "float(" + expr + ")";
                case DataType::FP8_E4M3:
                case DataType::FP8_E5M2:
                    return "float(half(" + expr + "))";
                default:
                    break;
            }
            break;

        case DataType::FP16:
            switch (src_dtype) {
                case DataType::FP32:
                    return "half(" + expr + ")";
                case DataType::FP16:
                    return expr;
                case DataType::BF16:
                    return "half(float(" + expr + "))";
                case DataType::FP8_E4M3:
                case DataType::FP8_E5M2:
                    return "half(" + expr + ")";
                default:
                    break;
            }
            break;

        case DataType::BF16:
            switch (src_dtype) {
                case DataType::FP32:
                    return "__nv_bfloat16(" + expr + ")";
                case DataType::FP16:
                    return "__nv_bfloat16(float(" + expr + "))";
                case DataType::BF16:
                    return expr;
                case DataType::FP8_E4M3:
                case DataType::FP8_E5M2:
                    return "__nv_bfloat16(float(half(" + expr + ")))";
                default:
                    break;
            }
            break;

        case DataType::FP8_E4M3:
            switch (src_dtype) {
                case DataType::FP32:
                    return "__nv_fp8_e4m3(half(" + expr + "))";
                case DataType::FP16:
                    return "__nv_fp8_e4m3(" + expr + ")";
                case DataType::BF16:
                    return "__nv_fp8_e4m3(half(float(" + expr + ")))";
                case DataType::FP8_E4M3:
                    return expr;
                case DataType::FP8_E5M2:
                    return "__nv_fp8_e4m3(half(" + expr + "))";
                default:
                    break;
            }
            break;

        case DataType::FP8_E5M2:
            switch (src_dtype) {
                case DataType::FP32:
                    return "__nv_fp8_e5m2(half(" + expr + "))";
                case DataType::FP16:
                    return "__nv_fp8_e5m2(" + expr + ")";
                case DataType::BF16:
                    return "__nv_fp8_e5m2(half(float(" + expr + ")))";
                case DataType::FP8_E4M3:
                    return "__nv_fp8_e5m2(half(" + expr + "))";
                case DataType::FP8_E5M2:
                    return expr;
                default:
                    break;
            }
            break;

        default:
            break;
    }

    throw runtime_error("Unsupported scalar cast in fused stage emitter from " + TensorDescriptor::getElementTypeName(src_dtype) + " to " +
                        TensorDescriptor::getElementTypeName(dst_dtype));
}

static std::string toFloatExpr(const std::string& expr, DataType src_dtype) { return castScalarExpr(expr, src_dtype, DataType::FP32); }

static std::string emitUnaryComputeExpr(ExprOp op, const std::string& x, DataType compute_dtype) {
    const std::string x_f = toFloatExpr(x, compute_dtype);

    switch (op) {
        case ExprOp::NEG:
            if (compute_dtype == DataType::FP32 || compute_dtype == DataType::FP16 || compute_dtype == DataType::BF16) {
                return "(-" + x + ")";
            }
            return castScalarExpr("(-" + x_f + ")", DataType::FP32, compute_dtype);

        case ExprOp::ABS:
            return castScalarExpr("fabsf(" + x_f + ")", DataType::FP32, compute_dtype);

        case ExprOp::EXP:
            return castScalarExpr("expf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::EXP2:
            return castScalarExpr("exp2f(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::EXP10:
            return castScalarExpr("exp10f(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::LN:
            return castScalarExpr("logf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::LOG2:
            return castScalarExpr("log2f(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::LOG10:
            return castScalarExpr("log10f(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::SQRT:
            return castScalarExpr("sqrtf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::UNSQUEEZE:
        case ExprOp::SQUEEZE:
            return x;

        default:
            throw runtime_error("Unsupported unary op in fused stage emitter.");
    }
}

static std::string emitBinaryComputeExpr(ExprOp op, const std::string& a, const std::string& b, DataType compute_dtype) {
    const std::string a_f = toFloatExpr(a, compute_dtype);
    const std::string b_f = toFloatExpr(b, compute_dtype);

    switch (op) {
        case ExprOp::ADD:
            if (compute_dtype == DataType::FP32 || compute_dtype == DataType::FP16 || compute_dtype == DataType::BF16) {
                return "(" + a + " + " + b + ")";
            }
            return castScalarExpr("(" + a_f + " + " + b_f + ")", DataType::FP32, compute_dtype);

        case ExprOp::SUB:
            if (compute_dtype == DataType::FP32 || compute_dtype == DataType::FP16 || compute_dtype == DataType::BF16) {
                return "(" + a + " - " + b + ")";
            }
            return castScalarExpr("(" + a_f + " - " + b_f + ")", DataType::FP32, compute_dtype);

        case ExprOp::MUL:
            if (compute_dtype == DataType::FP32 || compute_dtype == DataType::FP16 || compute_dtype == DataType::BF16) {
                return "(" + a + " * " + b + ")";
            }
            return castScalarExpr("(" + a_f + " * " + b_f + ")", DataType::FP32, compute_dtype);

        case ExprOp::DIV:
            if (compute_dtype == DataType::FP32 || compute_dtype == DataType::FP16 || compute_dtype == DataType::BF16) {
                return "(" + a + " / " + b + ")";
            }
            return castScalarExpr("(" + a_f + " / " + b_f + ")", DataType::FP32, compute_dtype);

        case ExprOp::POW:
            return castScalarExpr("powf(" + a_f + ", " + b_f + ")", DataType::FP32, compute_dtype);

        case ExprOp::MIN:
            return castScalarExpr("fminf(" + a_f + ", " + b_f + ")", DataType::FP32, compute_dtype);

        case ExprOp::MAX:
            return castScalarExpr("fmaxf(" + a_f + ", " + b_f + ")", DataType::FP32, compute_dtype);

        case ExprOp::MIN_GRAD_LEFT:
            return castScalarExpr(
                "((" + a_f + " < " + b_f + ") ? 1.0f : ((" + a_f + " > " + b_f + ") ? 0.0f : 0.5f))", DataType::FP32, compute_dtype);

        case ExprOp::MIN_GRAD_RIGHT:
            return castScalarExpr(
                "((" + a_f + " > " + b_f + ") ? 1.0f : ((" + a_f + " < " + b_f + ") ? 0.0f : 0.5f))", DataType::FP32, compute_dtype);

        case ExprOp::MAX_GRAD_LEFT:
            return castScalarExpr(
                "((" + a_f + " > " + b_f + ") ? 1.0f : ((" + a_f + " < " + b_f + ") ? 0.0f : 0.5f))", DataType::FP32, compute_dtype);

        case ExprOp::MAX_GRAD_RIGHT:
            return castScalarExpr(
                "((" + a_f + " < " + b_f + ") ? 1.0f : ((" + a_f + " > " + b_f + ") ? 0.0f : 0.5f))", DataType::FP32, compute_dtype);

        default:
            throw runtime_error("Unsupported binary op in fused stage emitter.");
    }
}

static bool tryGetEmitterConstantValue(const PhysicalExpression& expr, uint32_t node_idx, double& value) {
    if (node_idx >= expr.nodes.size()) {
        throw runtime_error("Node index out of range in constant emitter query.");
    }

    const ExprNode& n = expr.nodes[node_idx];
    switch (n.op) {
        case ExprOp::SCALAR_FP:
        case ExprOp::FILL:
            value = n.scalar_fp;
            return true;
        case ExprOp::UNSQUEEZE:
        case ExprOp::SQUEEZE:
            return tryGetEmitterConstantValue(expr, n.lhs, value);
        case ExprOp::NEG:
            if (tryGetEmitterConstantValue(expr, n.lhs, value)) {
                value = -value;
                return true;
            }
            return false;
        case ExprOp::ADD:
        case ExprOp::SUB:
        case ExprOp::MUL:
        case ExprOp::DIV: {
            double lhs = 0.0;
            double rhs = 0.0;
            if (!tryGetEmitterConstantValue(expr, n.lhs, lhs) || !tryGetEmitterConstantValue(expr, n.rhs, rhs)) {
                return false;
            }
            switch (n.op) {
                case ExprOp::ADD:
                    value = lhs + rhs;
                    return true;
                case ExprOp::SUB:
                    value = lhs - rhs;
                    return true;
                case ExprOp::MUL:
                    value = lhs * rhs;
                    return true;
                case ExprOp::DIV:
                    value = lhs / rhs;
                    return true;
                default:
                    return false;
            }
        }
        default:
            return false;
    }
}

static bool isEmitterConstantZero(const PhysicalExpression& expr, uint32_t node_idx) {
    double value = 0.0;
    return tryGetEmitterConstantValue(expr, node_idx, value) && value == 0.0;
}

static bool isEmitterConstantOne(const PhysicalExpression& expr, uint32_t node_idx) {
    double value = 0.0;
    return tryGetEmitterConstantValue(expr, node_idx, value) && value == 1.0;
}

static bool tryGetEmitterAliasSource(const PhysicalExpression& expr, uint32_t node_idx, uint32_t& source_idx) {
    if (node_idx >= expr.nodes.size()) {
        throw runtime_error("Node index out of range in emitter alias query.");
    }

    const ExprNode& n = expr.nodes[node_idx];
    switch (n.op) {
        case ExprOp::UNSQUEEZE:
        case ExprOp::SQUEEZE:
            source_idx = n.lhs;
            return true;
        case ExprOp::ADD:
            if (isEmitterConstantZero(expr, n.lhs)) {
                source_idx = n.rhs;
                return true;
            }
            if (isEmitterConstantZero(expr, n.rhs)) {
                source_idx = n.lhs;
                return true;
            }
            return false;
        case ExprOp::SUB:
            if (isEmitterConstantZero(expr, n.rhs)) {
                source_idx = n.lhs;
                return true;
            }
            return false;
        case ExprOp::MUL:
            if (isEmitterConstantOne(expr, n.lhs)) {
                source_idx = n.rhs;
                return true;
            }
            if (isEmitterConstantOne(expr, n.rhs)) {
                source_idx = n.lhs;
                return true;
            }
            return false;
        case ExprOp::DIV:
            if (isEmitterConstantOne(expr, n.rhs)) {
                source_idx = n.lhs;
                return true;
            }
            return false;
        default:
            return false;
    }
}

static DataType emittedScalarNodeValueDType(const ExprNode& node) {
    if (Expression::isLeafOp(node.op)) {
        return requireNodeOutputDType(node);
    }
    return requireNodeComputeDType(node);
}

static std::string emitResolvedScalarValueExpr(const PhysicalExpression& expr, uint32_t node_idx, DataType target_dtype) {
    if (node_idx >= expr.nodes.size()) {
        throw runtime_error("Node index out of range in resolved scalar emitter query.");
    }

    double constant_value = 0.0;
    if (tryGetEmitterConstantValue(expr, node_idx, constant_value)) {
        return castScalarExpr(emitScalarFpLiteral(constant_value), DataType::FP32, target_dtype);
    }

    uint32_t source_idx = UINT32_MAX;
    if (tryGetEmitterAliasSource(expr, node_idx, source_idx)) {
        return emitResolvedScalarValueExpr(expr, source_idx, target_dtype);
    }

    const DataType source_dtype = emittedScalarNodeValueDType(expr.nodes[node_idx]);
    return castScalarExpr(CudaSourceEmitter::ref(node_idx), source_dtype, target_dtype);
}

static bool shouldEmitScalarNodeDefinition(const PhysicalExpression& expr, uint32_t node_idx) {
    double constant_value = 0.0;
    if (tryGetEmitterConstantValue(expr, node_idx, constant_value)) {
        return false;
    }

    uint32_t source_idx = UINT32_MAX;
    if (tryGetEmitterAliasSource(expr, node_idx, source_idx)) {
        return false;
    }

    return true;
}

static void emitScalarAliasNode(
    std::ostringstream& ss, const PhysicalExpression& expr, uint32_t node_idx, uint32_t source_node_idx, const std::string& indent) {
    const DataType emitted_dtype = emittedScalarNodeValueDType(expr.nodes[node_idx]);
    const std::string output_type = scalarStorageType(emitted_dtype);
    ss << indent << "const " << output_type << " t" << node_idx << " = "
       << emitResolvedScalarValueExpr(expr, source_node_idx, emitted_dtype) << ";\n";
}

static void emitScalarNode(
    std::ostringstream& ss, const PhysicalExpression& expr, uint32_t node_idx, bool broadcast_support, const std::string& indent) {
    const ExprNode& n = expr.nodes[node_idx];
    const DataType emitted_dtype = emittedScalarNodeValueDType(n);
    const std::string output_type = scalarStorageType(emitted_dtype);

    double folded_constant = 0.0;
    if (n.op != ExprOp::INPUT && n.op != ExprOp::RUNTIME_SCALAR && n.op != ExprOp::TENSOR_RUNTIME_SCALAR &&
        tryGetEmitterConstantValue(expr, node_idx, folded_constant)) {
        const std::string literal = emitScalarFpLiteral(folded_constant);
        ss << indent << "const " << output_type << " t" << node_idx << " = " << castScalarExpr(literal, DataType::FP32, emitted_dtype)
           << ";\n";
        return;
    }

    switch (n.op) {
        case ExprOp::INPUT: {
            const std::string idx_expr = broadcast_support ? ("in" + std::to_string(n.input_slot) + "_offset") : "idx";
            const DataType input_tensor_dtype = requireNodeInputTensorDType(n);
            ss << indent << "const " << output_type << " t" << node_idx << " = "
               << castScalarExpr("in" + std::to_string(n.input_slot) + "[" + idx_expr + "]", input_tensor_dtype, emitted_dtype) << ";\n";
            return;
        }

        case ExprOp::RUNTIME_SCALAR: {
            const DataType input_tensor_dtype = requireNodeInputTensorDType(n);
            ss << indent << "const " << output_type << " t" << node_idx << " = "
               << castScalarExpr("in" + std::to_string(n.input_slot), input_tensor_dtype, emitted_dtype) << ";\n";
            return;
        }

        case ExprOp::TENSOR_RUNTIME_SCALAR: {
            const DataType input_tensor_dtype = requireNodeInputTensorDType(n);
            const std::string input_storage_type = scalarStorageType(input_tensor_dtype);
            const std::string input_expr = "reinterpret_cast<const " + input_storage_type + "*>(in" + std::to_string(n.input_slot) + ")[0]";
            ss << indent << "const " << output_type << " t" << node_idx << " = "
               << castScalarExpr(input_expr, input_tensor_dtype, emitted_dtype) << ";\n";
            return;
        }

        case ExprOp::SCALAR_FP: {
            const std::string literal = emitScalarFpLiteral(n.scalar_fp);
            ss << indent << "const " << output_type << " t" << node_idx << " = " << castScalarExpr(literal, DataType::FP32, emitted_dtype)
               << ";\n";
            return;
        }

        case ExprOp::FILL: {
            const std::string literal = emitScalarFpLiteral(n.scalar_fp);
            ss << indent << "const " << output_type << " t" << node_idx << " = " << castScalarExpr(literal, DataType::FP32, emitted_dtype)
               << ";\n";
            return;
        }

        default:
            break;
    }

    const DataType compute_dtype = requireNodeComputeDType(n);

    auto child_value = [&](uint32_t child_idx) -> std::string {
        if (child_idx >= expr.nodes.size()) {
            throw runtime_error("Child node index out of range in fused stage emitter.");
        }
        return emitResolvedScalarValueExpr(expr, child_idx, compute_dtype);
    };

    if (n.op == ExprOp::UNSQUEEZE || n.op == ExprOp::SQUEEZE) {
        emitScalarAliasNode(ss, expr, node_idx, n.lhs, indent);
        return;
    }

    if (Expression::isBinaryOp(n.op)) {
        switch (n.op) {
            case ExprOp::ADD:
                if (isEmitterConstantZero(expr, n.lhs)) {
                    emitScalarAliasNode(ss, expr, node_idx, n.rhs, indent);
                    return;
                }
                if (isEmitterConstantZero(expr, n.rhs)) {
                    emitScalarAliasNode(ss, expr, node_idx, n.lhs, indent);
                    return;
                }
                break;
            case ExprOp::SUB:
                if (isEmitterConstantZero(expr, n.rhs)) {
                    emitScalarAliasNode(ss, expr, node_idx, n.lhs, indent);
                    return;
                }
                break;
            case ExprOp::MUL:
                if (isEmitterConstantOne(expr, n.lhs)) {
                    emitScalarAliasNode(ss, expr, node_idx, n.rhs, indent);
                    return;
                }
                if (isEmitterConstantOne(expr, n.rhs)) {
                    emitScalarAliasNode(ss, expr, node_idx, n.lhs, indent);
                    return;
                }
                break;
            case ExprOp::DIV:
                if (isEmitterConstantOne(expr, n.rhs)) {
                    emitScalarAliasNode(ss, expr, node_idx, n.lhs, indent);
                    return;
                }
                break;
            default:
                break;
        }
    }

    std::string compute_expr;
    if (Expression::isBinaryOp(n.op)) {
        compute_expr = emitBinaryComputeExpr(n.op, child_value(n.lhs), child_value(n.rhs), compute_dtype);
    } else if (Expression::isUnaryOp(n.op)) {
        compute_expr = emitUnaryComputeExpr(n.op, child_value(n.lhs), compute_dtype);
    } else {
        throw runtime_error("Unsupported op in fused stage emitter.");
    }

    ss << indent << "const " << output_type << " t" << node_idx << " = " << castScalarExpr(compute_expr, compute_dtype, emitted_dtype)
       << ";\n";
}

static std::string refWithSuffix(uint32_t idx, const std::string& suffix) { return "t" + to_string(idx) + suffix; }

static std::string emitResolvedScalarValueExprSuffixed(const PhysicalExpression& expr,
                                                       uint32_t node_idx,
                                                       DataType target_dtype,
                                                       const std::string& suffix) {
    if (node_idx >= expr.nodes.size()) {
        throw runtime_error("Node index out of range in suffixed resolved scalar emitter query.");
    }

    double constant_value = 0.0;
    if (tryGetEmitterConstantValue(expr, node_idx, constant_value)) {
        return castScalarExpr(emitScalarFpLiteral(constant_value), DataType::FP32, target_dtype);
    }

    uint32_t source_idx = UINT32_MAX;
    if (tryGetEmitterAliasSource(expr, node_idx, source_idx)) {
        return emitResolvedScalarValueExprSuffixed(expr, source_idx, target_dtype, suffix);
    }

    const DataType source_dtype = emittedScalarNodeValueDType(expr.nodes[node_idx]);
    return castScalarExpr(refWithSuffix(node_idx, suffix), source_dtype, target_dtype);
}

static void emitScalarAliasNodeSuffixed(std::ostringstream& ss,
                                        const PhysicalExpression& expr,
                                        uint32_t node_idx,
                                        uint32_t source_node_idx,
                                        const std::string& suffix,
                                        const std::string& indent) {
    const DataType emitted_dtype = emittedScalarNodeValueDType(expr.nodes[node_idx]);
    const std::string output_type = scalarStorageType(emitted_dtype);
    ss << indent << "const " << output_type << " " << refWithSuffix(node_idx, suffix) << " = "
       << emitResolvedScalarValueExprSuffixed(expr, source_node_idx, emitted_dtype, suffix) << ";\n";
}

static void emitScalarNodeSuffixed(std::ostringstream& ss,
                                   const PhysicalExpression& expr,
                                   uint32_t node_idx,
                                   const std::string& idx_expr,
                                   const std::string& suffix,
                                   const std::string& indent,
                                   const std::string& flat_chunk_lane_expr = "",
                                   uint32_t flat_elements_per_thread = 0) {
    const ExprNode& n = expr.nodes[node_idx];
    const DataType emitted_dtype = emittedScalarNodeValueDType(n);
    const std::string output_type = scalarStorageType(emitted_dtype);

    double folded_constant = 0.0;
    if (n.op != ExprOp::INPUT && n.op != ExprOp::RUNTIME_SCALAR && n.op != ExprOp::TENSOR_RUNTIME_SCALAR &&
        tryGetEmitterConstantValue(expr, node_idx, folded_constant)) {
        const std::string literal = emitScalarFpLiteral(folded_constant);
        ss << indent << "const " << output_type << " " << refWithSuffix(node_idx, suffix) << " = "
           << castScalarExpr(literal, DataType::FP32, emitted_dtype) << ";\n";
        return;
    }

    switch (n.op) {
        case ExprOp::INPUT: {
            const DataType input_tensor_dtype = requireNodeInputTensorDType(n);
            const std::string input_expr =
                flat_chunk_lane_expr.empty()
                    ? ("in" + std::to_string(n.input_slot) + "[" + idx_expr + "]")
                    : emitChunkLaneReadExpr("in" + std::to_string(n.input_slot) + "_chunk",
                                            chunkScalarTypeForBytes(dataTypeStorageBytes(input_tensor_dtype) * flat_elements_per_thread),
                                            input_tensor_dtype,
                                            flat_chunk_lane_expr,
                                            "in" + std::to_string(n.input_slot) + "_chunk_data");
            ss << indent << "const " << output_type << " " << refWithSuffix(node_idx, suffix) << " = "
               << castScalarExpr(input_expr, input_tensor_dtype, emitted_dtype) << ";\n";
            return;
        }

        case ExprOp::RUNTIME_SCALAR: {
            const DataType input_tensor_dtype = requireNodeInputTensorDType(n);
            ss << indent << "const " << output_type << " " << refWithSuffix(node_idx, suffix) << " = "
               << castScalarExpr("in" + std::to_string(n.input_slot), input_tensor_dtype, emitted_dtype) << ";\n";
            return;
        }

        case ExprOp::TENSOR_RUNTIME_SCALAR: {
            const DataType input_tensor_dtype = requireNodeInputTensorDType(n);
            const std::string input_storage_type = scalarStorageType(input_tensor_dtype);
            const std::string input_expr = "reinterpret_cast<const " + input_storage_type + "*>(in" + std::to_string(n.input_slot) + ")[0]";
            ss << indent << "const " << output_type << " " << refWithSuffix(node_idx, suffix) << " = "
               << castScalarExpr(input_expr, input_tensor_dtype, emitted_dtype) << ";\n";
            return;
        }

        case ExprOp::SCALAR_FP: {
            const std::string literal = emitScalarFpLiteral(n.scalar_fp);
            ss << indent << "const " << output_type << " " << refWithSuffix(node_idx, suffix) << " = "
               << castScalarExpr(literal, DataType::FP32, emitted_dtype) << ";\n";
            return;
        }

        case ExprOp::FILL: {
            const std::string literal = emitScalarFpLiteral(n.scalar_fp);
            ss << indent << "const " << output_type << " " << refWithSuffix(node_idx, suffix) << " = "
               << castScalarExpr(literal, DataType::FP32, emitted_dtype) << ";\n";
            return;
        }

        default:
            break;
    }

    const DataType compute_dtype = requireNodeComputeDType(n);

    auto child_value = [&](uint32_t child_idx) -> std::string {
        if (child_idx >= expr.nodes.size()) {
            throw runtime_error("Child node index out of range in suffixed fused stage emitter.");
        }
        return emitResolvedScalarValueExprSuffixed(expr, child_idx, compute_dtype, suffix);
    };

    if (n.op == ExprOp::UNSQUEEZE || n.op == ExprOp::SQUEEZE) {
        emitScalarAliasNodeSuffixed(ss, expr, node_idx, n.lhs, suffix, indent);
        return;
    }

    if (Expression::isBinaryOp(n.op)) {
        switch (n.op) {
            case ExprOp::ADD:
                if (isEmitterConstantZero(expr, n.lhs)) {
                    emitScalarAliasNodeSuffixed(ss, expr, node_idx, n.rhs, suffix, indent);
                    return;
                }
                if (isEmitterConstantZero(expr, n.rhs)) {
                    emitScalarAliasNodeSuffixed(ss, expr, node_idx, n.lhs, suffix, indent);
                    return;
                }
                break;
            case ExprOp::SUB:
                if (isEmitterConstantZero(expr, n.rhs)) {
                    emitScalarAliasNodeSuffixed(ss, expr, node_idx, n.lhs, suffix, indent);
                    return;
                }
                break;
            case ExprOp::MUL:
                if (isEmitterConstantOne(expr, n.lhs)) {
                    emitScalarAliasNodeSuffixed(ss, expr, node_idx, n.rhs, suffix, indent);
                    return;
                }
                if (isEmitterConstantOne(expr, n.rhs)) {
                    emitScalarAliasNodeSuffixed(ss, expr, node_idx, n.lhs, suffix, indent);
                    return;
                }
                break;
            case ExprOp::DIV:
                if (isEmitterConstantOne(expr, n.rhs)) {
                    emitScalarAliasNodeSuffixed(ss, expr, node_idx, n.lhs, suffix, indent);
                    return;
                }
                break;
            default:
                break;
        }
    }

    std::string compute_expr;
    if (Expression::isBinaryOp(n.op)) {
        compute_expr = emitBinaryComputeExpr(n.op, child_value(n.lhs), child_value(n.rhs), compute_dtype);
    } else if (Expression::isUnaryOp(n.op)) {
        compute_expr = emitUnaryComputeExpr(n.op, child_value(n.lhs), compute_dtype);
    } else {
        throw runtime_error("Unsupported op in suffixed fused stage emitter. " + to_string((int)n.op));
    }

    ss << indent << "const " << output_type << " " << refWithSuffix(node_idx, suffix) << " = "
       << castScalarExpr(compute_expr, compute_dtype, emitted_dtype) << ";\n";
}

static std::string vector_compute_conversion(const std::string& storage_dtype_vector, const std::string& variable) {
    if (storage_dtype_vector == "half2") {
        return variable;
    } else if (storage_dtype_vector == "__nv_bfloat162") {
        return variable;
    } else if (storage_dtype_vector == "__nv_fp8x2_e4m3") {
        return "static_cast<__half2>(" + variable + ")";
    } else if (storage_dtype_vector == "__nv_fp8x2_e5m2") {
        return "static_cast<__half2>(" + variable + ")";
    }
    throw runtime_error("Unsupported vector storage dtype in vector_compute_conversion: " + storage_dtype_vector);
}

static std::string vector_storage_conversion(const std::string& storage_dtype_vector, const std::string& variable) {
    if (storage_dtype_vector == "half2") {
        return variable;
    } else if (storage_dtype_vector == "__nv_bfloat162") {
        return variable;
    } else if (storage_dtype_vector == "__nv_fp8x2_e4m3") {
        return "__nv_fp8x2_e4m3(" + variable + ")";
    } else if (storage_dtype_vector == "__nv_fp8x2_e5m2") {
        return "__nv_fp8x2_e5m2(" + variable + ")";
    }
    throw runtime_error("Unsupported vector storage dtype in vector_storage_conversion: " + storage_dtype_vector);
}

static std::string emitVector2ScalarLiteral(double x, DataType dtype) {
    const std::string lit = emitScalarFpLiteral(x);

    if (dtype == DataType::FP16 || dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2) {
        return "__halves2half2(__float2half_rn(" + lit + "), __float2half_rn(" + lit + "))";
    } else if (dtype == DataType::BF16) {
        return "__floats2bfloat162_rn(" + lit + ", " + lit + ")";
    }
    throw runtime_error("Unsupported dtype in emitVector2ScalarLiteral.");
}

static std::string float2_compute_conversion(const std::string& storage_dtype_vector, const std::string& variable) {
    if (storage_dtype_vector == "float2") {
        return variable;
    } else if (storage_dtype_vector == "half2") {
        return "__half22float2(" + variable + ")";
    } else if (storage_dtype_vector == "__nv_bfloat162") {
        return "__bfloat1622float2(" + variable + ")";
    } else if (storage_dtype_vector == "__nv_fp8x2_e4m3" || storage_dtype_vector == "__nv_fp8x2_e5m2") {
        return "__half22float2(static_cast<__half2>(" + variable + "))";
    }
    throw runtime_error("Unsupported vector storage dtype in float2_compute_conversion: " + storage_dtype_vector);
}

static std::string float2_storage_conversion(const std::string& storage_dtype_vector, const std::string& variable) {
    if (storage_dtype_vector == "float2") {
        return variable;
    } else if (storage_dtype_vector == "half2") {
        return "__float22half2_rn(" + variable + ")";
    } else if (storage_dtype_vector == "__nv_bfloat162") {
        return "__float22bfloat162_rn(" + variable + ")";
    } else if (storage_dtype_vector == "__nv_fp8x2_e4m3") {
        return "__nv_fp8x2_e4m3(__float22half2_rn(" + variable + "))";
    } else if (storage_dtype_vector == "__nv_fp8x2_e5m2") {
        return "__nv_fp8x2_e5m2(__float22half2_rn(" + variable + "))";
    }
    throw runtime_error("Unsupported vector storage dtype in float2_storage_conversion: " + storage_dtype_vector);
}

static std::string emitFloat2ScalarLiteral(double x) {
    const std::string lit = emitScalarFpLiteral(x);
    return "make_float2(" + lit + ", " + lit + ")";
}

static std::string emitFloat2RuntimeScalarValue(const PhysicalExpression& expr, const ExprNode& node) {
    const DataType input_dtype = requireNodeInputTensorDType(node);
    const DataType output_dtype = requireNodeOutputDType(node);

    std::string scalar_source;
    if (expr.inputs.at(node.input_slot).kind == NamedInput::Kind::TensorRuntimeScalar) {
        const std::string input_storage_type = scalarStorageType(input_dtype);
        scalar_source = "reinterpret_cast<const " + input_storage_type + "*>(in" + std::to_string(node.input_slot) + ")[0]";
    } else {
        scalar_source = "in" + std::to_string(node.input_slot);
    }

    std::string scalar_expr = castScalarExpr(scalar_source, input_dtype, output_dtype);
    scalar_expr = castScalarExpr(scalar_expr, output_dtype, DataType::FP32);
    return "make_float2(" + scalar_expr + ", " + scalar_expr + ")";
}

static std::string emitFloat2Binary(const std::string& op, const std::string& a, const std::string& b) {
    return "make_float2(" + a + ".x " + op + " " + b + ".x, " + a + ".y " + op + " " + b + ".y)";
}

static std::string emitFloat2UnaryCall(const std::string& fn, const std::string& x) {
    return "make_float2(" + fn + "(" + x + ".x), " + fn + "(" + x + ".y))";
}

static std::string emitFloat2MinMaxGradMask(ExprOp op, const std::string& a, const std::string& b) {
    std::string x_expr;
    std::string y_expr;
    switch (op) {
        case ExprOp::MIN_GRAD_LEFT:
            x_expr = "((" + a + ".x < " + b + ".x) ? 1.0f : ((" + a + ".x > " + b + ".x) ? 0.0f : 0.5f))";
            y_expr = "((" + a + ".y < " + b + ".y) ? 1.0f : ((" + a + ".y > " + b + ".y) ? 0.0f : 0.5f))";
            break;
        case ExprOp::MIN_GRAD_RIGHT:
            x_expr = "((" + a + ".x > " + b + ".x) ? 1.0f : ((" + a + ".x < " + b + ".x) ? 0.0f : 0.5f))";
            y_expr = "((" + a + ".y > " + b + ".y) ? 1.0f : ((" + a + ".y < " + b + ".y) ? 0.0f : 0.5f))";
            break;
        case ExprOp::MAX_GRAD_LEFT:
            x_expr = "((" + a + ".x > " + b + ".x) ? 1.0f : ((" + a + ".x < " + b + ".x) ? 0.0f : 0.5f))";
            y_expr = "((" + a + ".y > " + b + ".y) ? 1.0f : ((" + a + ".y < " + b + ".y) ? 0.0f : 0.5f))";
            break;
        case ExprOp::MAX_GRAD_RIGHT:
            x_expr = "((" + a + ".x < " + b + ".x) ? 1.0f : ((" + a + ".x > " + b + ".x) ? 0.0f : 0.5f))";
            y_expr = "((" + a + ".y < " + b + ".y) ? 1.0f : ((" + a + ".y > " + b + ".y) ? 0.0f : 0.5f))";
            break;
        default:
            throw runtime_error("Unsupported min/max grad mask op in float2 vector emitter.");
    }
    return "make_float2(" + x_expr + ", " + y_expr + ")";
}

static std::string emitFloatScalarRuntimeScalarValue(const PhysicalExpression& expr, const ExprNode& node) {
    const DataType input_dtype = requireNodeInputTensorDType(node);
    const DataType output_dtype = requireNodeOutputDType(node);

    std::string scalar_source;
    if (expr.inputs.at(node.input_slot).kind == NamedInput::Kind::TensorRuntimeScalar) {
        const std::string input_storage_type = scalarStorageType(input_dtype);
        scalar_source = "reinterpret_cast<const " + input_storage_type + "*>(in" + std::to_string(node.input_slot) + ")[0]";
    } else {
        scalar_source = "in" + std::to_string(node.input_slot);
    }

    std::string scalar_expr = castScalarExpr(scalar_source, input_dtype, output_dtype);
    return castScalarExpr(scalar_expr, output_dtype, DataType::FP32);
}

static void emitFloatScalarNodeDefinitions(std::ostringstream& ss,
                                           const PhysicalExpression& expr,
                                           const std::string& indent,
                                           const std::function<std::string(uint32_t)>& input_slot_value) {
    for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
        const ExprNode& n = expr.nodes[node_idx];
        switch (n.op) {
            case ExprOp::INPUT: {
                const std::string variable = input_slot_value(n.input_slot);
                const DataType input_dtype = requireNodeInputTensorDType(n);
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = "
                   << castScalarExpr(variable, input_dtype, DataType::FP32) << ";\n";
                break;
            }
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = " << emitFloatScalarRuntimeScalarValue(expr, n)
                   << ";\n";
                break;
            case ExprOp::SCALAR_FP:
            case ExprOp::FILL:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = " << emitScalarFpLiteral(n.scalar_fp) << ";\n";
                break;
            case ExprOp::ADD:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = " << CudaSourceEmitter::ref(n.lhs) << " + "
                   << CudaSourceEmitter::ref(n.rhs) << ";\n";
                break;
            case ExprOp::SUB:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = " << CudaSourceEmitter::ref(n.lhs) << " - "
                   << CudaSourceEmitter::ref(n.rhs) << ";\n";
                break;
            case ExprOp::MUL:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = " << CudaSourceEmitter::ref(n.lhs) << " * "
                   << CudaSourceEmitter::ref(n.rhs) << ";\n";
                break;
            case ExprOp::DIV:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = " << CudaSourceEmitter::ref(n.lhs) << " / "
                   << CudaSourceEmitter::ref(n.rhs) << ";\n";
                break;
            case ExprOp::NEG:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = -" << CudaSourceEmitter::ref(n.lhs) << ";\n";
                break;
            case ExprOp::ABS:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = fabsf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::EXP:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = expf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::EXP2:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = exp2f(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::EXP10:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = exp10f(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::LN:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = logf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::LOG2:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = log2f(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::LOG10:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = log10f(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::SQRT:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = sqrtf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::UNSQUEEZE:
            case ExprOp::SQUEEZE:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = " << CudaSourceEmitter::ref(n.lhs) << ";\n";
                break;
            case ExprOp::POW:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = powf(" << CudaSourceEmitter::ref(n.lhs) << ", "
                   << CudaSourceEmitter::ref(n.rhs) << ");\n";
                break;
            case ExprOp::MIN:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = fminf(" << CudaSourceEmitter::ref(n.lhs) << ", "
                   << CudaSourceEmitter::ref(n.rhs) << ");\n";
                break;
            case ExprOp::MAX:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = fmaxf(" << CudaSourceEmitter::ref(n.lhs) << ", "
                   << CudaSourceEmitter::ref(n.rhs) << ");\n";
                break;
            case ExprOp::MIN_GRAD_LEFT:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = ((" << CudaSourceEmitter::ref(n.lhs) << " < "
                   << CudaSourceEmitter::ref(n.rhs) << ") ? 1.0f : ((" << CudaSourceEmitter::ref(n.lhs) << " > "
                   << CudaSourceEmitter::ref(n.rhs) << ") ? 0.0f : 0.5f));\n";
                break;
            case ExprOp::MIN_GRAD_RIGHT:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = ((" << CudaSourceEmitter::ref(n.lhs) << " > "
                   << CudaSourceEmitter::ref(n.rhs) << ") ? 1.0f : ((" << CudaSourceEmitter::ref(n.lhs) << " < "
                   << CudaSourceEmitter::ref(n.rhs) << ") ? 0.0f : 0.5f));\n";
                break;
            case ExprOp::MAX_GRAD_LEFT:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = ((" << CudaSourceEmitter::ref(n.lhs) << " > "
                   << CudaSourceEmitter::ref(n.rhs) << ") ? 1.0f : ((" << CudaSourceEmitter::ref(n.lhs) << " < "
                   << CudaSourceEmitter::ref(n.rhs) << ") ? 0.0f : 0.5f));\n";
                break;
            case ExprOp::MAX_GRAD_RIGHT:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = ((" << CudaSourceEmitter::ref(n.lhs) << " < "
                   << CudaSourceEmitter::ref(n.rhs) << ") ? 1.0f : ((" << CudaSourceEmitter::ref(n.lhs) << " > "
                   << CudaSourceEmitter::ref(n.rhs) << ") ? 0.0f : 0.5f));\n";
                break;
            default:
                throw runtime_error("Unsupported op in fp32 scalar transpose fallback emitter: " + to_string((int32_t)n.op));
        }
    }
}

static DataType vectorizedComputeScalarDType(DataType dtype) {
    switch (dtype) {
        case DataType::BF16:
            return DataType::BF16;
        case DataType::FP16:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return DataType::FP16;
        default:
            throw runtime_error("Unsupported dtype in vectorizedComputeScalarDType.");
    }
}

static std::string emitVector2DupScalar(const std::string& expr, DataType scalar_dtype) {
    switch (scalar_dtype) {
        case DataType::FP16:
            return "__halves2half2(" + expr + ", " + expr + ")";
        case DataType::BF16:
            return "__halves2bfloat162(" + expr + ", " + expr + ")";
        default:
            throw runtime_error("Unsupported scalar dtype in emitVector2DupScalar.");
    }
}

static std::string emitVector2RuntimeScalarValue(const PhysicalExpression& expr, const ExprNode& node, DataType stage_dtype) {
    const DataType input_dtype = requireNodeInputTensorDType(node);
    const DataType output_dtype = requireNodeOutputDType(node);
    const DataType compute_scalar_dtype = vectorizedComputeScalarDType(stage_dtype);

    std::string scalar_source;
    if (expr.inputs.at(node.input_slot).kind == NamedInput::Kind::TensorRuntimeScalar) {
        const std::string input_storage_type = scalarStorageType(input_dtype);
        scalar_source = "reinterpret_cast<const " + input_storage_type + "*>(in" + std::to_string(node.input_slot) + ")[0]";
    } else {
        scalar_source = "in" + std::to_string(node.input_slot);
    }

    std::string scalar_expr = castScalarExpr(scalar_source, input_dtype, output_dtype);
    scalar_expr = castScalarExpr(scalar_expr, output_dtype, compute_scalar_dtype);
    return emitVector2DupScalar(scalar_expr, compute_scalar_dtype);
}

static std::string emitVector2BroadcastNativeLoad(const std::string& storage_dtype_vector,
                                                  const std::string& base_ptr,
                                                  const std::string& scalar_offset0) {
    const std::string vec_expr = "reinterpret_cast<const " + storage_dtype_vector + "*>(" + base_ptr + " + " + scalar_offset0 + ")[0]";
    return vector_compute_conversion(storage_dtype_vector, vec_expr);
}

static std::string emitVector2Add(const std::string& a, const std::string& b) { return "__hadd2(" + a + ", " + b + ")"; }
static std::string emitVector2Sub(const std::string& a, const std::string& b) { return "__hsub2(" + a + ", " + b + ")"; }
static std::string emitVector2Mul(const std::string& a, const std::string& b) { return "__hmul2(" + a + ", " + b + ")"; }
static std::string emitVector2Div(const std::string& a, const std::string& b) { return "__h2div(" + a + ", " + b + ")"; }
static std::string emitVector2Neg(const std::string& x, DataType dtype) { return "__hneg2(" + x + ")"; }
static std::string emitVector2Abs(const std::string& x, DataType dtype) { return "__hmax2(" + x + ", " + emitVector2Neg(x, dtype) + ")"; }
static std::string emitVector2Exp(const std::string& x, DataType dtype) { return "h2exp(" + x + ")"; }
static std::string emitVector2Exp2(const std::string& x, DataType dtype) { return "h2exp2(" + x + ")"; }
static std::string emitVector2Exp10(const std::string& x, DataType dtype) { return "h2exp10(" + x + ")"; }
static std::string emitVector2Ln(const std::string& x, DataType dtype) { return "h2log(" + x + ")"; }
static std::string emitVector2Log2(const std::string& x, DataType dtype) { return "h2log2(" + x + ")"; }
static std::string emitVector2Log10(const std::string& x, DataType dtype) { return "h2log10(" + x + ")"; }
static std::string emitVector2Sqrt(const std::string& x, DataType dtype) { return "h2sqrt(" + x + ")"; }
static std::string emitVector2Pow(const std::string& a, const std::string& b, DataType dtype) {
    if (dtype == DataType::BF16)
        return "__floats2bfloat162_rn( powf(" + a + ".x, " + b + ".x), powf(" + a + ".y, " + b + ".y) )";
    else
        return "__floats2half2_rn( powf(" + a + ".x, " + b + ".x), powf(" + a + ".y, " + b + ".y) )";
}
static std::string emitVector2Min(const std::string& a, const std::string& b) { return "__hmin2(" + a + ", " + b + ")"; }
static std::string emitVector2Max(const std::string& a, const std::string& b) { return "__hmax2(" + a + ", " + b + ")"; }

static std::string emitVector2MinMaxGradMask(ExprOp op, const std::string& a, const std::string& b, DataType dtype) {
    std::string x_expr;
    std::string y_expr;
    switch (op) {
        case ExprOp::MIN_GRAD_LEFT:
            x_expr = "((" + a + ".x < " + b + ".x) ? 1.0f : ((" + a + ".x > " + b + ".x) ? 0.0f : 0.5f))";
            y_expr = "((" + a + ".y < " + b + ".y) ? 1.0f : ((" + a + ".y > " + b + ".y) ? 0.0f : 0.5f))";
            break;
        case ExprOp::MIN_GRAD_RIGHT:
            x_expr = "((" + a + ".x > " + b + ".x) ? 1.0f : ((" + a + ".x < " + b + ".x) ? 0.0f : 0.5f))";
            y_expr = "((" + a + ".y > " + b + ".y) ? 1.0f : ((" + a + ".y < " + b + ".y) ? 0.0f : 0.5f))";
            break;
        case ExprOp::MAX_GRAD_LEFT:
            x_expr = "((" + a + ".x > " + b + ".x) ? 1.0f : ((" + a + ".x < " + b + ".x) ? 0.0f : 0.5f))";
            y_expr = "((" + a + ".y > " + b + ".y) ? 1.0f : ((" + a + ".y < " + b + ".y) ? 0.0f : 0.5f))";
            break;
        case ExprOp::MAX_GRAD_RIGHT:
            x_expr = "((" + a + ".x < " + b + ".x) ? 1.0f : ((" + a + ".x > " + b + ".x) ? 0.0f : 0.5f))";
            y_expr = "((" + a + ".y < " + b + ".y) ? 1.0f : ((" + a + ".y > " + b + ".y) ? 0.0f : 0.5f))";
            break;
        default:
            throw runtime_error("Unsupported min/max grad mask op in vector emitter.");
    }

    if (dtype == DataType::BF16) {
        return "__floats2bfloat162_rn(" + x_expr + ", " + y_expr + ")";
    }
    return "__floats2half2_rn(" + x_expr + ", " + y_expr + ")";
}

static std::string emitVector2BroadcastPackLoad(const std::string& storage_dtype,
                                                const std::string& variable0,
                                                const std::string& variable1) {
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

static void emitVector2NodeDefinitionsForSuffix(std::ostringstream& ss,
                                                const PhysicalExpression& expr,
                                                DataType dtype,
                                                const std::string& storage_dtype_vector,
                                                const std::string& suffix,
                                                const std::string& indent,
                                                const std::function<std::string(uint32_t)>& input_slot_value,
                                                const std::function<std::string(uint32_t)>& scalar_const_value = {},
                                                bool input_slot_value_is_compute = false) {
    std::string compute_dtype_vector;
    if (dtype == DataType::BF16) {
        compute_dtype_vector = "__nv_bfloat162";
    } else if (dtype == DataType::FP16 || dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2) {
        compute_dtype_vector = "half2";
    } else {
        throw runtime_error("emitVector2NodeDefinitionsForSuffix called with non-vectorizable dtype.");
    }

    for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
        const auto& n = expr.nodes[node_idx];
        switch (n.op) {
            case ExprOp::INPUT: {
                const std::string variable = input_slot_value(n.input_slot);
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = ";
                if (input_slot_value_is_compute) {
                    ss << variable;
                } else {
                    ss << vector_compute_conversion(storage_dtype_vector, variable);
                }
                ss << ";\n";
                break;
            }
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2RuntimeScalarValue(expr, n, dtype) << ";\n";
                break;
            case ExprOp::SCALAR_FP:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << (scalar_const_value ? scalar_const_value(node_idx) : emitVector2ScalarLiteral(n.scalar_fp, dtype)) << ";\n";
                break;
            case ExprOp::FILL:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << (scalar_const_value ? scalar_const_value(node_idx) : emitVector2ScalarLiteral(n.scalar_fp, dtype)) << ";\n";
                break;
            case ExprOp::ADD:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Add(refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                break;
            case ExprOp::SUB:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Sub(refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                break;
            case ExprOp::MUL:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Mul(refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                break;
            case ExprOp::DIV:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Div(refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                break;
            case ExprOp::NEG:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Neg(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::ABS:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Abs(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::EXP:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Exp(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::EXP2:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Exp2(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::EXP10:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Exp10(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::LN:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Ln(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::LOG2:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Log2(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::LOG10:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Log10(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::SQRT:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Sqrt(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::UNSQUEEZE:
            case ExprOp::SQUEEZE:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = " << refWithSuffix(n.lhs, suffix)
                   << ";\n";
                break;
            case ExprOp::POW:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Pow(refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::MIN:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Min(refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                break;
            case ExprOp::MAX:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Max(refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                break;
            case ExprOp::MIN_GRAD_LEFT:
            case ExprOp::MIN_GRAD_RIGHT:
            case ExprOp::MAX_GRAD_LEFT:
            case ExprOp::MAX_GRAD_RIGHT:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2MinMaxGradMask(n.op, refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix), dtype) << ";\n";
                break;
            default:
                throw runtime_error("Unsupported op in vectorized fused transpose emitter: " + to_string((int32_t)n.op));
        }
    }
}

static void emitFloat2NodeDefinitionsForSuffix(std::ostringstream& ss,
                                               const PhysicalExpression& expr,
                                               const std::string& input_storage_dtype_vector,
                                               const std::string& suffix,
                                               const std::string& indent,
                                               const std::function<std::string(uint32_t)>& input_slot_value,
                                               const std::function<std::string(uint32_t)>& scalar_const_value = {}) {
    for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
        const auto& n = expr.nodes[node_idx];
        switch (n.op) {
            case ExprOp::INPUT: {
                const std::string variable = input_slot_value(n.input_slot);
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << float2_compute_conversion(input_storage_dtype_vector, variable) << ";\n";
                break;
            }
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = " << emitFloat2RuntimeScalarValue(expr, n) << ";\n";
                break;
            case ExprOp::SCALAR_FP:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << (scalar_const_value ? scalar_const_value(node_idx) : emitFloat2ScalarLiteral(n.scalar_fp)) << ";\n";
                break;
            case ExprOp::FILL:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << (scalar_const_value ? scalar_const_value(node_idx) : emitFloat2ScalarLiteral(n.scalar_fp)) << ";\n";
                break;
            case ExprOp::ADD:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2Binary("+", refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                break;
            case ExprOp::SUB:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2Binary("-", refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                break;
            case ExprOp::MUL:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2Binary("*", refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                break;
            case ExprOp::DIV:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2Binary("/", refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                break;
            case ExprOp::NEG:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = make_float2(-" << refWithSuffix(n.lhs, suffix)
                   << ".x, -" << refWithSuffix(n.lhs, suffix) << ".y);\n";
                break;
            case ExprOp::ABS:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("fabsf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::EXP:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("expf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::EXP2:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("exp2f", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::EXP10:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = make_float2(exp10f(" << refWithSuffix(n.lhs, suffix)
                   << ".x), exp10f(" << refWithSuffix(n.lhs, suffix) << ".y));\n";
                break;
            case ExprOp::LN:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("logf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::LOG2:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("log2f", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::LOG10:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("log10f", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::SQRT:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("sqrtf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::UNSQUEEZE:
            case ExprOp::SQUEEZE:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = " << refWithSuffix(n.lhs, suffix) << ";\n";
                break;
            case ExprOp::POW:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = make_float2(powf(" << refWithSuffix(n.lhs, suffix)
                   << ".x, " << refWithSuffix(n.rhs, suffix) << ".x), powf(" << refWithSuffix(n.lhs, suffix) << ".y, "
                   << refWithSuffix(n.rhs, suffix) << ".y));\n";
                break;
            case ExprOp::MIN:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = make_float2(fminf(" << refWithSuffix(n.lhs, suffix)
                   << ".x, " << refWithSuffix(n.rhs, suffix) << ".x), fminf(" << refWithSuffix(n.lhs, suffix) << ".y, "
                   << refWithSuffix(n.rhs, suffix) << ".y));\n";
                break;
            case ExprOp::MAX:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = make_float2(fmaxf(" << refWithSuffix(n.lhs, suffix)
                   << ".x, " << refWithSuffix(n.rhs, suffix) << ".x), fmaxf(" << refWithSuffix(n.lhs, suffix) << ".y, "
                   << refWithSuffix(n.rhs, suffix) << ".y));\n";
                break;
            case ExprOp::MIN_GRAD_LEFT:
            case ExprOp::MIN_GRAD_RIGHT:
            case ExprOp::MAX_GRAD_LEFT:
            case ExprOp::MAX_GRAD_RIGHT:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2MinMaxGradMask(n.op, refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                break;
            default:
                throw runtime_error("Unsupported op in float2 fused transpose emitter: " + to_string((int32_t)n.op));
        }
    }
}

static std::string emitVector2Flat(const PhysicalExecutionStage& stage,
                                   DataType dtype,
                                   const std::string& kernel_name,
                                   bool use_uint32_index_math) {
    std::ostringstream ss;

    std::string compute_dtype;
    std::string compute_dtype_vector;
    std::string storage_dtype_vector;
    uint32_t packs_per_thread = 0;

    if (dtype == DataType::BF16) {
        compute_dtype = "__nv_bfloat16";
        compute_dtype_vector = "__nv_bfloat162";
        storage_dtype_vector = "__nv_bfloat162";
        packs_per_thread = 4;
        ss << "#include <cuda_bf16.h>\n";
    } else if (dtype == DataType::FP16) {
        compute_dtype = "half";
        compute_dtype_vector = "half2";
        storage_dtype_vector = "half2";
        packs_per_thread = 4;
        ss << "#include <cuda_fp16.h>\n";
    } else if (dtype == DataType::FP8_E4M3) {
        compute_dtype = "half";
        compute_dtype_vector = "half2";
        storage_dtype_vector = "__nv_fp8x2_e4m3";
        packs_per_thread = 8;
        ss << "#include <cuda_fp16.h>\n";
        ss << "#include <cuda_fp8.h>\n";
    } else if (dtype == DataType::FP8_E5M2) {
        compute_dtype = "half";
        compute_dtype_vector = "half2";
        storage_dtype_vector = "__nv_fp8x2_e5m2";
        packs_per_thread = 8;
        ss << "#include <cuda_fp16.h>\n";
        ss << "#include <cuda_fp8.h>\n";
    } else {
        throw runtime_error("emitVector2Flat called with non-vectorizable dtype.");
    }

    const uint32_t num_inputs = stage.expr.numInputs();
    const std::string index_type = emittedIndexType(use_uint32_index_math);
    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);

    ss << "#include <vector_types.h>\n";
    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";

    for (uint32_t i = 0; i < num_inputs; ++i) {
        if (stage.expr.inputs[i].kind == NamedInput::Kind::RuntimeScalarFp32) {
            ss << scalarStorageType(input_dtypes[i]) << " in" << i;
        } else {
            ss << "const float4* in" << i;
        }
        ss << ", ";
    }

    for (uint32_t i = 0; i < stage.outputs.size(); ++i) {
        ss << "float4* out" << i << ", ";
    }

    ss << index_type << " numel) {\n";
    ss << "  " << index_type << " idx = " << emitFlatThreadIndexExpr(use_uint32_index_math) << ";\n";
    ss << "  const " << index_type << " packed_numel = (numel + " << emitUnsignedLiteral(1, use_uint32_index_math) << ") >> 1;\n";
    ss << "  const " << index_type << " packed_base = idx * "
       << emitUnsignedLiteral(static_cast<uint64_t>(packs_per_thread), use_uint32_index_math) << ";\n";
    ss << "  if (packed_base >= packed_numel) return;\n\n";

    bool emitted_any_tensor_chunk = false;
    for (uint32_t i = 0; i < num_inputs; ++i) {
        if (stage.expr.inputs[i].kind != NamedInput::Kind::Tensor) {
            continue;
        }
        emitted_any_tensor_chunk = true;
        ss << "  const float4 in" << i << "_chunk = in" << i << "[idx];\n";
        ss << "  const " << storage_dtype_vector << "* in" << i << "_chunk_data = reinterpret_cast<const " << storage_dtype_vector
           << "*>(&in" << i << "_chunk);\n";
    }
    if (emitted_any_tensor_chunk) {
        ss << "\n";
    }

    for (uint32_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
        ss << "  float4 out" << out_idx << "_chunk;\n";
        ss << "  " << storage_dtype_vector << "* out" << out_idx << "_chunk_data = reinterpret_cast<" << storage_dtype_vector << "*>(&out"
           << out_idx << "_chunk);\n";
    }
    if (!stage.outputs.empty()) {
        ss << "\n";
    }

    for (uint32_t pack = 0; pack < packs_per_thread; ++pack) {
        const std::string suffix = "_p" + std::to_string(pack);

        for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
            const auto& n = stage.expr.nodes[node_idx];
            switch (n.op) {
                case ExprOp::INPUT: {
                    const std::string variable = "in" + to_string(n.input_slot) + "_chunk_data[" + std::to_string(pack) + "]";
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << vector_compute_conversion(storage_dtype_vector, variable) << ";\n";
                    break;
                }
                case ExprOp::RUNTIME_SCALAR:
                case ExprOp::TENSOR_RUNTIME_SCALAR:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2RuntimeScalarValue(stage.expr, n, dtype) << ";\n";
                    break;
                case ExprOp::SCALAR_FP:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2ScalarLiteral(n.scalar_fp, dtype) << ";\n";
                    break;
                case ExprOp::FILL:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2ScalarLiteral(n.scalar_fp, dtype) << ";\n";
                    break;
                case ExprOp::ADD:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Add(refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                    break;
                case ExprOp::SUB:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Sub(refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                    break;
                case ExprOp::MUL:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Mul(refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                    break;
                case ExprOp::DIV:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Div(refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                    break;
                case ExprOp::NEG:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Neg(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::ABS:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Abs(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::EXP:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Exp(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::EXP2:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Exp2(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::EXP10:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Exp10(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::LN:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Ln(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::LOG2:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Log2(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::LOG10:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Log10(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::SQRT:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Sqrt(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::UNSQUEEZE:
                case ExprOp::SQUEEZE:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = " << refWithSuffix(n.lhs, suffix)
                       << ";\n";
                    break;
                case ExprOp::POW:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Pow(refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::MIN:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Min(refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                    break;
                case ExprOp::MAX:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Max(refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix)) << ";\n";
                    break;
                case ExprOp::MIN_GRAD_LEFT:
                case ExprOp::MIN_GRAD_RIGHT:
                case ExprOp::MAX_GRAD_LEFT:
                case ExprOp::MAX_GRAD_RIGHT:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2MinMaxGradMask(n.op, refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix), dtype) << ";\n";
                    break;
                default:
                    throw runtime_error("Unsupported op in vectorized fused emitter: " + to_string((int32_t)n.op));
            }
        }

        ss << "\n";
        for (uint32_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
            const CompiledStageOutput& output = stage.outputs[out_idx];
            ss << "  out" << out_idx << "_chunk_data[" << pack
               << "] = " << vector_storage_conversion(storage_dtype_vector, refWithSuffix(output.local_node_idx, suffix)) << ";\n";
        }
        ss << "\n";
    }

    for (uint32_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
        ss << "  out" << out_idx << "[idx] = out" << out_idx << "_chunk;\n";
    }

    ss << "}\n";
    return ss.str();
}

static std::string chunkScalarTypeForBytes(uint32_t bytes) {
    switch (bytes) {
        case 4:
            return "unsigned int";
        case 8:
            return "unsigned long long";
        case 16:
            return "float4";
        default:
            throw runtime_error("Unsupported wide flat chunk size: " + to_string(bytes));
    }
}

static std::string emitWideScalarFlat(const PhysicalExecutionStage& stage,
                                      const std::string& kernel_name,
                                      bool use_uint32_index_math,
                                      uint32_t elements_per_thread) {
    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const std::vector<DataType> output_dtypes = collectOutputDTypes(stage);

    std::vector<std::string> input_chunk_types;
    input_chunk_types.reserve(input_dtypes.size());
    for (DataType input_dtype : input_dtypes) {
        input_chunk_types.push_back(chunkScalarTypeForBytes(dataTypeStorageBytes(input_dtype) * elements_per_thread));
    }

    std::vector<std::string> output_chunk_types;
    output_chunk_types.reserve(output_dtypes.size());
    for (DataType output_dtype : output_dtypes) {
        output_chunk_types.push_back(chunkScalarTypeForBytes(dataTypeStorageBytes(output_dtype) * elements_per_thread));
    }

    std::ostringstream ss;
    emitRequiredHeaders(stage.expr, ss);

    ss << "#include <vector_types.h>\n";
    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";

    bool first_arg = true;
    for (uint32_t i = 0; i < input_dtypes.size(); ++i) {
        if (!first_arg) {
            ss << ", ";
        }
        first_arg = false;
        if (stage.expr.inputs[i].kind == NamedInput::Kind::RuntimeScalarFp32) {
            ss << scalarStorageType(input_dtypes[i]) << " in" << i;
        } else {
            ss << "const " << input_chunk_types[i] << "* in" << i;
        }
    }

    for (uint32_t i = 0; i < output_dtypes.size(); ++i) {
        if (!first_arg) {
            ss << ", ";
        }
        first_arg = false;
        ss << output_chunk_types[i] << "* out" << i;
    }

    if (!first_arg) {
        ss << ", ";
    }
    const std::string index_type = emittedIndexType(use_uint32_index_math);
    ss << index_type << " numel) {\n";

    ss << "  " << index_type << " idx = " << emitFlatThreadIndexExpr(use_uint32_index_math) << ";\n";
    ss << "  const " << index_type << " base = idx * "
       << emitUnsignedLiteral(static_cast<uint64_t>(elements_per_thread), use_uint32_index_math) << ";\n";
    ss << "  if (base >= numel) return;\n\n";

    bool emitted_any_tensor_chunk = false;
    for (uint32_t i = 0; i < input_dtypes.size(); ++i) {
        if (stage.expr.inputs[i].kind != NamedInput::Kind::Tensor) {
            continue;
        }
        emitted_any_tensor_chunk = true;
        ss << "  const " << input_chunk_types[i] << " in" << i << "_chunk = in" << i << "[idx];\n";
        if (!(input_chunk_types[i] == "float4" && input_dtypes[i] == DataType::FP32)) {
            ss << "  const " << scalarStorageType(input_dtypes[i]) << "* in" << i << "_chunk_data = reinterpret_cast<const "
               << scalarStorageType(input_dtypes[i]) << "*>(&in" << i << "_chunk);\n";
        }
    }
    if (emitted_any_tensor_chunk) {
        ss << "\n";
    }

    for (uint32_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
        ss << "  " << output_chunk_types[out_idx] << " out" << out_idx << "_chunk;\n";
        if (!(output_chunk_types[out_idx] == "float4" && output_dtypes[out_idx] == DataType::FP32)) {
            ss << "  " << scalarStorageType(output_dtypes[out_idx]) << "* out" << out_idx << "_chunk_data = reinterpret_cast<"
               << scalarStorageType(output_dtypes[out_idx]) << "*>(&out" << out_idx << "_chunk);\n";
        }
    }
    if (!stage.outputs.empty()) {
        ss << "\n";
    }

    for (uint32_t lane = 0; lane < elements_per_thread; ++lane) {
        const std::string suffix = "_l" + std::to_string(lane);
        const std::string lane_idx_expr = "base + " + emitUnsignedLiteral(static_cast<uint64_t>(lane), use_uint32_index_math);

        for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
            if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
                continue;
            }
            emitScalarNodeSuffixed(ss, stage.expr, node_idx, lane_idx_expr, suffix, "  ", std::to_string(lane), elements_per_thread);
        }

        ss << "\n";
        for (uint32_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
            const CompiledStageOutput& output = stage.outputs[out_idx];
            const DataType output_dtype = requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]);
            ss << emitChunkLaneWriteStmt("out" + std::to_string(out_idx) + "_chunk",
                                         output_chunk_types[out_idx],
                                         output_dtype,
                                         lane,
                                         emitResolvedScalarValueExprSuffixed(stage.expr, output.local_node_idx, output_dtype, suffix),
                                         "  ",
                                         "out" + std::to_string(out_idx) + "_chunk_data");
        }
        ss << "\n";
    }

    for (uint32_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
        ss << "  out" << out_idx << "[idx] = out" << out_idx << "_chunk;\n";
    }

    ss << "}\n";
    return ss.str();
}

static std::string emitVector2SpecializedBroadcast(const CompiledExecutionStage& stage,
                                                   const std::vector<SpecializedBroadcastGroup>& groups,
                                                   DataType dtype,
                                                   const std::string& kernel_name) {
    std::ostringstream ss;

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
    } else if (dtype == DataType::FP8_E5M2) {
        compute_dtype = "half";
        compute_dtype_vector = "half2";
        storage_dtype = "__nv_fp8_e5m2";
        storage_dtype_vector = "__nv_fp8x2_e5m2";
        ss << "#include <cuda_fp16.h>\n";
        ss << "#include <cuda_fp8.h>\n";
    } else {
        throw runtime_error("emitVector2SpecializedBroadcast called with non-vectorizable dtype.");
    }

    const bool use_uint32_index_math = groupsSupportUInt32IndexMath(groups);
    const std::string index_type = emittedIndexType(use_uint32_index_math);
    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);

    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";

    for (uint32_t i = 0; i < stage.expr.numInputs(); ++i) {
        if (stage.expr.inputs[i].kind == NamedInput::Kind::RuntimeScalarFp32) {
            ss << scalarStorageType(input_dtypes[i]) << " in" << i;
        } else {
            ss << "const " << storage_dtype << "* in" << i;
        }
        ss << ", ";
    }
    for (uint32_t i = 0; i < stage.outputs.size(); ++i) {
        ss << storage_dtype_vector << "* out" << i;
        if (i + 1 < stage.outputs.size()) {
            ss << ", ";
        }
    }

    ss << ") {\n";
    ss << "  " << index_type << " idx = " << emitFlatThreadIndexExpr(use_uint32_index_math) << ";\n\n";

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

        ss << "  if (idx < " << emitUnsignedLiteral(packed_numel, use_uint32_index_math) << ") {\n";
        ss << "    const " << index_type << " idx0 = idx << 1;\n";

        if (any_scalar_pack) {
            ss << "    const " << index_type << " idx1 = idx0 + " << emitUnsignedLiteral(1, use_uint32_index_math) << ";\n";
        }

        for (uint32_t input_slot : group.used_input_slots) {
            ss << "    " << index_type << " in" << input_slot << "_offset0 = " << emitUnsignedLiteral(0, use_uint32_index_math) << ";\n";
        }
        for (size_t used_i : scalar_pack_indices) {
            const uint32_t input_slot = group.used_input_slots[used_i];
            ss << "    " << index_type << " in" << input_slot << "_offset1 = " << emitUnsignedLiteral(0, use_uint32_index_math) << ";\n";
        }

        ss << "\n";
        if (!group.active_axes.empty()) {
            emitSpecializedBroadcastOffsetMath(ss, group, all_used_indices, "idx0", "0", "    ", use_uint32_index_math);
            ss << "\n";

            if (any_scalar_pack) {
                emitSpecializedBroadcastOffsetMath(ss, group, scalar_pack_indices, "idx1", "1", "    ", use_uint32_index_math);
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

                case ExprOp::RUNTIME_SCALAR:
                case ExprOp::TENSOR_RUNTIME_SCALAR:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2RuntimeScalarValue(stage.expr, n, dtype)
                       << ";\n";
                    break;
                case ExprOp::SCALAR_FP:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2ScalarLiteral(n.scalar_fp, dtype)
                       << ";\n";
                    break;
                case ExprOp::FILL:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2ScalarLiteral(n.scalar_fp, dtype)
                       << ";\n";
                    break;
                case ExprOp::ADD:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Add(CudaSourceEmitter::ref(n.lhs), CudaSourceEmitter::ref(n.rhs)) << ";\n";
                    break;
                case ExprOp::SUB:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Sub(CudaSourceEmitter::ref(n.lhs), CudaSourceEmitter::ref(n.rhs)) << ";\n";
                    break;
                case ExprOp::MUL:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Mul(CudaSourceEmitter::ref(n.lhs), CudaSourceEmitter::ref(n.rhs)) << ";\n";
                    break;
                case ExprOp::DIV:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Div(CudaSourceEmitter::ref(n.lhs), CudaSourceEmitter::ref(n.rhs)) << ";\n";
                    break;
                case ExprOp::NEG:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Neg(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::ABS:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Abs(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::EXP:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Exp(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::EXP2:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Exp2(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::EXP10:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Exp10(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::LN:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Ln(CudaSourceEmitter::ref(n.lhs), dtype)
                       << ";\n";
                    break;
                case ExprOp::LOG2:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Log2(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::LOG10:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Log10(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::SQRT:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Sqrt(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::UNSQUEEZE:
                case ExprOp::SQUEEZE:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << CudaSourceEmitter::ref(n.lhs) << ";\n";
                    break;
                case ExprOp::POW:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Pow(CudaSourceEmitter::ref(n.lhs), CudaSourceEmitter::ref(n.rhs), dtype) << ";\n";
                    break;
                case ExprOp::MIN:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Min(CudaSourceEmitter::ref(n.lhs), CudaSourceEmitter::ref(n.rhs)) << ";\n";
                    break;
                case ExprOp::MAX:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Max(CudaSourceEmitter::ref(n.lhs), CudaSourceEmitter::ref(n.rhs)) << ";\n";
                    break;
                case ExprOp::MIN_GRAD_LEFT:
                case ExprOp::MIN_GRAD_RIGHT:
                case ExprOp::MAX_GRAD_LEFT:
                case ExprOp::MAX_GRAD_RIGHT:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2MinMaxGradMask(n.op, CudaSourceEmitter::ref(n.lhs), CudaSourceEmitter::ref(n.rhs), dtype) << ";\n";
                    break;
                default:
                    throw std::runtime_error("Unsupported op in specialized vector broadcast emitter.");
            }
        }

        ss << "\n";
        for (uint32_t out_idx : group.output_indices) {
            const CompiledStageOutput& output = stage.outputs[out_idx];
            ss << "    out" << out_idx
               << "[idx] = " << vector_storage_conversion(storage_dtype_vector, CudaSourceEmitter::ref(output.local_node_idx)) << ";\n";
        }

        ss << "  }\n\n";
    }

    ss << "}\n";
    return ss.str();
}

static std::string emitTiledTransposeMaterializedFused(const PhysicalExecutionStage& stage,
                                                       const std::string& kernel_name,
                                                       bool use_uint32_index_math) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel) {
        throw runtime_error("emitTiledTransposeMaterializedFused called on non-fused stage.");
    }
    const CompiledStageOutput& output = requireSingleTransposedMaterializedOutput(stage);

    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const DataType output_dtype = requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]);
    const std::string output_type = scalarStorageType(output_dtype);
    const Optional<DataType> maybe_vectorized_dtype = CudaSourceEmitter::getVectorizedStageStorageDType(stage);
    const Optional<DataType> maybe_tensor_input_dtype = getSingleTensorInputStorageDType(stage.expr, input_dtypes);
    const bool emit_packed_low_precision_path = transposePackScalars(output_dtype) > 1;
    const bool emit_homogeneous_packed_vectorized_path =
        maybe_vectorized_dtype.isPresent() && maybe_vectorized_dtype.get() == output_dtype && transposePackScalars(output_dtype) > 1;
    const bool emit_mixed_two_byte_float2_path =
        maybe_tensor_input_dtype.isPresent() &&
        supportsMixedTwoByteFloat2TransposedVectorization(stage.expr, maybe_tensor_input_dtype.get(), output_dtype);
    const bool emit_mixed_fp8_vectorized_path =
        maybe_tensor_input_dtype.isPresent() &&
        supportsMixedFp8TransposedVectorization(stage.expr, maybe_tensor_input_dtype.get(), output_dtype);
    const bool emit_cross_width_float2_path =
        maybe_tensor_input_dtype.isPresent() &&
        supportsFloat2TransposedVectorization(stage.expr, maybe_tensor_input_dtype.get(), output_dtype);
    const bool emit_cross_width_half2_path = maybe_tensor_input_dtype.isPresent() &&
                                             supportsHalf2TransposedVectorization(stage.expr, maybe_tensor_input_dtype.get(), output_dtype);
    const bool emit_fp8_to_bf16_float2_path =
        maybe_tensor_input_dtype.isPresent() &&
        supportsFp8ToBf16Float2TransposedVectorization(stage.expr, maybe_tensor_input_dtype.get(), output_dtype);
    const bool emit_packed_vectorized_path = emit_homogeneous_packed_vectorized_path || emit_mixed_two_byte_float2_path ||
                                             emit_mixed_fp8_vectorized_path || emit_cross_width_float2_path ||
                                             emit_cross_width_half2_path || emit_fp8_to_bf16_float2_path;
    const std::string index_type = emittedIndexType(use_uint32_index_math);
    std::ostringstream ss;
    emitRequiredHeaders(stage.expr, ss);
    ss << "#define TILE_DIM 32\n";
    ss << "#define BLOCK_ROWS 8\n\n";
    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";

    bool first_arg = true;
    for (uint32_t i = 0; i < input_dtypes.size(); ++i) {
        if (!first_arg) {
            ss << ", ";
        }
        first_arg = false;
        if (stage.expr.inputs[i].kind == NamedInput::Kind::RuntimeScalarFp32) {
            ss << scalarStorageType(input_dtypes[i]) << " in" << i;
        } else {
            ss << "const " << scalarStorageType(input_dtypes[i]) << "* __restrict__ in" << i;
        }
    }

    if (!first_arg) {
        ss << ", ";
    }
    ss << output_type << "* __restrict__ out0, " << index_type << " numRows, " << index_type << " numCols) {\n";

    if (emit_packed_low_precision_path) {
        const DataType dtype = output_dtype;
        const bool needs_input_vector_type = emit_homogeneous_packed_vectorized_path || emit_mixed_two_byte_float2_path ||
                                             emit_mixed_fp8_vectorized_path || emit_cross_width_float2_path ||
                                             emit_cross_width_half2_path || emit_fp8_to_bf16_float2_path;
        const DataType vector_input_dtype =
            needs_input_vector_type ? (maybe_tensor_input_dtype.isPresent() ? maybe_tensor_input_dtype.get() : output_dtype) : output_dtype;
        const uint32_t pack_scalars = transposePackScalars(dtype);
        const uint32_t pairs_per_pack = pack_scalars / 2;
        const std::string pack_type = transposePackType(dtype);
        const std::string output_storage_dtype_vector = transposeVector2StorageType(output_dtype);
        const std::string input_storage_dtype_vector =
            needs_input_vector_type ? transposeVector2StorageType(vector_input_dtype) : output_storage_dtype_vector;

        ss << "  constexpr unsigned int PACK_SCALARS = " << pack_scalars << "U;\n";
        ss << "  constexpr unsigned int TILE_COL_SCALARS = TILE_DIM * PACK_SCALARS;\n";
        ss << "  using Pack = " << pack_type << ";\n";
        if (emit_packed_vectorized_path) {
            ss << "  using OutputVec2 = " << output_storage_dtype_vector << ";\n";
            ss << "  using InputVec2 = " << input_storage_dtype_vector << ";\n";
        }
        ss << "  union PackRaw { unsigned int raw; Pack pack; };\n";
        ss << "  __shared__ unsigned int tile[TILE_DIM][TILE_DIM + 1];\n";
        ss << "  const " << index_type << " rowStart = static_cast<" << index_type << ">(blockIdx.y) * TILE_DIM;\n";
        ss << "  const " << index_type << " colStart = static_cast<" << index_type << ">(blockIdx.x) * TILE_COL_SCALARS;\n";
        ss << "  const unsigned int packedCol = threadIdx.x;\n";
        ss << "  const " << index_type << " logicalColBase = colStart + static_cast<" << index_type
           << ">(packedCol) * PACK_SCALARS;\n";
        if (emit_packed_vectorized_path) {
            for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
                const ExprNode& node = stage.expr.nodes[node_idx];
                if (node.op != ExprOp::SCALAR_FP && node.op != ExprOp::FILL) {
                    continue;
                }
                if (emit_mixed_two_byte_float2_path || emit_cross_width_float2_path || emit_fp8_to_bf16_float2_path) {
                    ss << "  const float2 c" << node_idx << " = " << emitFloat2ScalarLiteral(node.scalar_fp) << ";\n";
                } else {
                    ss << "  const " << (vector_input_dtype == DataType::BF16 ? "__nv_bfloat162" : "half2") << " c" << node_idx << " = "
                       << emitVector2ScalarLiteral(node.scalar_fp, vector_input_dtype) << ";\n";
                }
            }
        }
        ss << "\n";

        ss << "  for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {\n";
        ss << "    const " << index_type << " logicalRow = rowStart + threadIdx.y + j;\n";
        ss << "    const " << index_type << " idx_base = logicalRow * numCols + logicalColBase;\n";
        ss << "    const bool inputPackedLoadOk = (logicalRow < numRows) && (logicalColBase + PACK_SCALARS <= numCols) &&\n";
        ss << "                                   ((idx_base % PACK_SCALARS) == " << emitUnsignedLiteral(0, use_uint32_index_math) << ");\n";
        ss << "    Pack output_pack{};\n";
        if (emit_packed_vectorized_path) {
            ss << "    if (inputPackedLoadOk) {\n";
            ss << "      OutputVec2* output_vec2 = reinterpret_cast<OutputVec2*>(&output_pack);\n";
            for (uint32_t pair = 0; pair < pairs_per_pack; ++pair) {
                const std::string suffix = "_tp" + std::to_string(pair);
                ss << "      {\n";
                ss << "        constexpr unsigned int PAIR = " << pair << "U;\n";
                if (emit_mixed_two_byte_float2_path || emit_cross_width_float2_path || emit_fp8_to_bf16_float2_path) {
                    emitFloat2NodeDefinitionsForSuffix(
                        ss,
                        stage.expr,
                        input_storage_dtype_vector,
                        suffix,
                        "        ",
                        [&](uint32_t input_slot) {
                            return "reinterpret_cast<const InputVec2*>(in" + std::to_string(input_slot) + " + idx_base)[PAIR]";
                        },
                        [&](uint32_t scalar_node_idx) { return "c" + std::to_string(scalar_node_idx); });
                    ss << "        output_vec2[PAIR] = "
                       << float2_storage_conversion(output_storage_dtype_vector, refWithSuffix(output.local_node_idx, suffix)) << ";\n";
                } else {
                    emitVector2NodeDefinitionsForSuffix(
                        ss,
                        stage.expr,
                        vector_input_dtype,
                        input_storage_dtype_vector,
                        suffix,
                        "        ",
                        [&](uint32_t input_slot) {
                            return "reinterpret_cast<const InputVec2*>(in" + std::to_string(input_slot) + " + idx_base)[PAIR]";
                        },
                        [&](uint32_t scalar_node_idx) { return "c" + std::to_string(scalar_node_idx); });
                    ss << "        output_vec2[PAIR] = "
                       << vector_storage_conversion(output_storage_dtype_vector, refWithSuffix(output.local_node_idx, suffix)) << ";\n";
                }
                ss << "      }\n";
            }
            ss << "    } else if (logicalRow < numRows) {\n";
        } else {
            ss << "    if (logicalRow < numRows) {\n";
        }
        ss << "      " << output_type << "* output_scalar = reinterpret_cast<" << output_type << "*>(&output_pack);\n";
        ss << "      for (unsigned int lane = 0; lane < PACK_SCALARS; ++lane) {\n";
        ss << "        const " << index_type << " logicalCol = logicalColBase + lane;\n";
        ss << "        if (logicalCol < numCols) {\n";
        ss << "          const " << index_type << " idx = logicalRow * numCols + logicalCol;\n";
        ss << "          {\n";
        if (emit_fp8_to_bf16_float2_path) {
            emitFloatScalarNodeDefinitions(
                ss, stage.expr, "            ", [&](uint32_t input_slot) { return "in" + std::to_string(input_slot) + "[idx]"; });
            ss << "            output_scalar[lane] = __nv_bfloat16(" << CudaSourceEmitter::ref(output.local_node_idx) << ");\n";
        } else {
            for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
                if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
                    continue;
                }
                emitScalarNode(ss, stage.expr, node_idx, /*broadcast_support=*/false, "            ");
            }
            ss << "            output_scalar[lane] = " << emitResolvedScalarValueExpr(stage.expr, output.local_node_idx, output_dtype)
               << ";\n";
        }
        ss << "          }\n";
        ss << "        }\n";
        ss << "      }\n";
        ss << "    }\n";
        ss << "    PackRaw raw{};\n";
        ss << "    raw.pack = output_pack;\n";
        ss << "    tile[threadIdx.y + j][packedCol] = raw.raw;\n";
        ss << "  }\n\n";
        ss << "  __syncthreads();\n\n";
        ss << "  constexpr unsigned int PACKED_OUTPUT_COLS_PER_TILE = TILE_DIM / PACK_SCALARS;\n";
        ss << "  constexpr unsigned int PACKED_STORES_PER_TILE = TILE_COL_SCALARS * PACKED_OUTPUT_COLS_PER_TILE;\n";
        ss << "  const unsigned int threadLinear = threadIdx.y * TILE_DIM + threadIdx.x;\n";
        ss << "  for (unsigned int packedStore = threadLinear; packedStore < PACKED_STORES_PER_TILE; packedStore += TILE_DIM * BLOCK_ROWS) "
              "{\n";
        ss << "    const unsigned int localOutRow = packedStore / PACKED_OUTPUT_COLS_PER_TILE;\n";
        ss << "    const unsigned int localOutPackCol = packedStore - localOutRow * PACKED_OUTPUT_COLS_PER_TILE;\n";
        ss << "    const unsigned int inputPackCol = localOutRow / PACK_SCALARS;\n";
        ss << "    const unsigned int inputColLane = localOutRow - inputPackCol * PACK_SCALARS;\n";
        ss << "    const " << index_type << " outputRow = colStart + localOutRow;\n";
        ss << "    const " << index_type << " outputColBase = rowStart + static_cast<" << index_type
           << ">(localOutPackCol) * PACK_SCALARS;\n";
        ss << "    Pack output_pack{};\n";
        ss << "    " << output_type << "* output_scalar = reinterpret_cast<" << output_type << "*>(&output_pack);\n";
        ss << "    for (unsigned int lane = 0; lane < PACK_SCALARS; ++lane) {\n";
        ss << "      PackRaw input_raw{};\n";
        ss << "      input_raw.raw = tile[localOutPackCol * PACK_SCALARS + lane][inputPackCol];\n";
        ss << "      const " << output_type << "* input_scalar = reinterpret_cast<const " << output_type << "*>(&input_raw.pack);\n";
        ss << "      output_scalar[lane] = input_scalar[inputColLane];\n";
        ss << "    }\n";
        ss << "    const " << index_type << " out_base_idx = outputRow * numRows + outputColBase;\n";
        ss << "    const bool outputPackedStoreOk = (outputRow < numCols) && (outputColBase + PACK_SCALARS <= numRows) &&\n";
        ss << "                                     ((out_base_idx % PACK_SCALARS) == " << emitUnsignedLiteral(0, use_uint32_index_math) << ");\n";
        ss << "    if (outputPackedStoreOk) {\n";
        ss << "      Pack* out_ptr = reinterpret_cast<Pack*>(out0 + out_base_idx);\n";
        ss << "      *out_ptr = output_pack;\n";
        ss << "    } else if (outputRow < numCols) {\n";
        ss << "      for (unsigned int lane = 0; lane < PACK_SCALARS; ++lane) {\n";
        ss << "        const " << index_type << " outputCol = outputColBase + lane;\n";
        ss << "        if (outputCol < numRows) {\n";
        ss << "          out0[outputRow * numRows + outputCol] = output_scalar[lane];\n";
        ss << "        }\n";
        ss << "      }\n";
        ss << "    }\n";
        ss << "  }\n";
        ss << "}\n";
        return ss.str();
    }

    ss << "  __shared__ " << output_type << " tile[TILE_DIM][TILE_DIM + 1];\n";
    ss << "  const " << index_type << " x = static_cast<" << index_type << ">(blockIdx.x) * TILE_DIM + threadIdx.x;\n";
    ss << "  const " << index_type << " y = static_cast<" << index_type << ">(blockIdx.y) * TILE_DIM + threadIdx.y;\n";
    ss << "  for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {\n";
    ss << "    const " << index_type << " logical_row = y + j;\n";
    ss << "    const " << index_type << " logical_col = x;\n";
    ss << "    if (logical_col < numCols && logical_row < numRows) {\n";
    ss << "      const " << index_type << " idx = logical_row * numCols + logical_col;\n";

    for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
        if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
            continue;
        }
        emitScalarNode(ss, stage.expr, node_idx, /*broadcast_support=*/false, "      ");
    }

    ss << "      tile[threadIdx.y + j][threadIdx.x] = " << emitResolvedScalarValueExpr(stage.expr, output.local_node_idx, output_dtype)
       << ";\n";
    ss << "    }\n";
    ss << "  }\n\n";
    ss << "  __syncthreads();\n\n";
    ss << "  const " << index_type << " tx = static_cast<" << index_type << ">(blockIdx.y) * TILE_DIM + threadIdx.x;\n";
    ss << "  const " << index_type << " ty = static_cast<" << index_type << ">(blockIdx.x) * TILE_DIM + threadIdx.y;\n";
    ss << "  for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {\n";
    ss << "    if (tx < numRows && (ty + j) < numCols) {\n";
    ss << "      const " << index_type << " out_idx = (ty + j) * numRows + tx;\n";
    ss << "      out0[out_idx] = tile[threadIdx.x][threadIdx.y + j];\n";
    ss << "    }\n";
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

std::string CudaSourceEmitter::emitFlat(const PhysicalExpression& expr, const std::string& kernel_name, bool use_uint32_index_math) {
    if (expr.output_node >= expr.nodes.size()) {
        throw runtime_error("CudaSourceEmitter::emitFlat(expr, ...) output_node out of range.");
    }

    PhysicalExecutionStage stage{
        .kind = PhysicalExecutionStage::Kind::FusedKernel,
        .expr = expr,
        .input_value_ids = {},
        .outputs =
            {
                CompiledStageOutput{
                    .name = "output",
                    .local_node_idx = expr.output_node,
                    .value_id = 0,
                },
            },
    };

    return emitFlat(stage, kernel_name, use_uint32_index_math);
}

std::string CudaSourceEmitter::emitFlat(const PhysicalExecutionStage& stage, const std::string& kernel_name, bool use_uint32_index_math) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel) {
        throw runtime_error("CudaSourceEmitter::emitFlat(stage, ...) called on non-fused stage.");
    }

    if (stage.outputs.empty()) {
        throw runtime_error("Fused stage has no outputs.");
    }

    if (stageHasTransposedMaterializedOutput(stage.outputs)) {
        return emitTiledTransposeMaterializedFused(stage, kernel_name, use_uint32_index_math);
    }

    Optional<DataType> vectorized_dtype = getVectorizedStageStorageDType(stage);
    if (vectorized_dtype.isPresent()) {
        return emitVector2Flat(stage, vectorized_dtype.get(), kernel_name, use_uint32_index_math);
    }

    const uint32_t elements_per_thread = flatElementsPerThread(stage);
    if (elements_per_thread > 1) {
        return emitWideScalarFlat(stage, kernel_name, use_uint32_index_math, elements_per_thread);
    }

    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const std::vector<DataType> output_dtypes = collectOutputDTypes(stage);

    std::ostringstream ss;
    emitRequiredHeaders(stage.expr, ss);

    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";

    bool first_arg = true;
    for (uint32_t i = 0; i < input_dtypes.size(); ++i) {
        if (!first_arg) {
            ss << ", ";
        }
        first_arg = false;
        if (stage.expr.inputs[i].kind == NamedInput::Kind::RuntimeScalarFp32) {
            ss << scalarStorageType(input_dtypes[i]) << " in" << i;
        } else {
            ss << "const " << scalarStorageType(input_dtypes[i]) << "* in" << i;
        }
    }

    for (uint32_t i = 0; i < output_dtypes.size(); ++i) {
        if (!first_arg) {
            ss << ", ";
        }
        first_arg = false;
        ss << scalarStorageType(output_dtypes[i]) << "* out" << i;
    }

    if (!first_arg) {
        ss << ", ";
    }
    const std::string index_type = emittedIndexType(use_uint32_index_math);
    ss << index_type << " numel) {";

    ss << "  " << index_type << " idx = " << emitFlatThreadIndexExpr(use_uint32_index_math) << ";";
    ss << "  if (idx >= numel) return;";

    for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
        if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
            continue;
        }
        emitScalarNode(ss, stage.expr, node_idx, /*broadcast_support=*/false, "  ");
    }

    ss << "";
    for (uint32_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
        const CompiledStageOutput& output = stage.outputs[out_idx];
        const DataType output_dtype = requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]);
        ss << "  out" << out_idx << "[idx] = " << emitResolvedScalarValueExpr(stage.expr, output.local_node_idx, output_dtype) << ";";
    }

    ss << "}";
    return ss.str();
}

std::string CudaSourceEmitter::ref(uint32_t idx) { return "t" + to_string(idx); }

static std::string emitTiledTransposeMaterializedSpecializedBroadcast(const CompiledExecutionStage& stage,
                                                                      const std::vector<SpecializedBroadcastGroup>& groups,
                                                                      const std::string& kernel_name) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        throw runtime_error("emitTiledTransposeMaterializedSpecializedBroadcast called on non-fused stage.");
    }
    const CompiledStageOutput& output = requireSingleTransposedMaterializedOutput(stage);
    if (groups.size() != 1) {
        throw runtime_error("Transposed broadcast fused materialization expects exactly one broadcast group.");
    }

    const SpecializedBroadcastGroup& group = groups[0];
    if (group.output_indices.size() != 1 || group.output_indices[0] != 0) {
        throw runtime_error("Transposed broadcast fused materialization expects the single stage output in its broadcast group.");
    }
    if (group.output_dims.size() != 2) {
        throw runtime_error("Transposed broadcast fused materialization currently requires a rank-2 logical output.");
    }

    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const DataType output_dtype = requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]);
    const std::string output_type = scalarStorageType(output_dtype);
    const bool emit_packed_low_precision_path = transposePackScalars(output_dtype) > 1;
    const Optional<DataType> maybe_vectorized_dtype = CudaSourceEmitter::getVectorizedStageStorageDType(stage);
    const bool emit_fp8_vectorized_pair_path = emit_packed_low_precision_path && isFp8DType(output_dtype) &&
                                               maybe_vectorized_dtype.isPresent() && maybe_vectorized_dtype.get() == output_dtype;
    const bool use_uint32_index_math = groupSupportsUInt32IndexMath(group);
    const std::string index_type = emittedIndexType(use_uint32_index_math);

    std::vector<size_t> all_used_indices(group.used_input_slots.size());
    std::iota(all_used_indices.begin(), all_used_indices.end(), 0);

    std::vector<size_t> scalar_pack_indices;
    std::unordered_map<uint32_t, SpecializedInputLoadKind> input_load_kind_by_slot;
    scalar_pack_indices.reserve(group.used_input_slots.size());
    for (size_t used_i = 0; used_i < group.used_input_slots.size(); ++used_i) {
        const SpecializedInputLoadKind kind = group.used_input_load_kinds.at(used_i);
        input_load_kind_by_slot.emplace(group.used_input_slots[used_i], kind);
        if (kind == SpecializedInputLoadKind::ScalarPack) {
            scalar_pack_indices.push_back(used_i);
        }
    }

    std::ostringstream ss;
    emitRequiredHeaders(stage.expr, ss);
    ss << "#define TILE_DIM 32\n";
    ss << "#define BLOCK_ROWS 8\n\n";
    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";

    bool first_arg = true;
    for (uint32_t i = 0; i < input_dtypes.size(); ++i) {
        if (!first_arg) {
            ss << ", ";
        }
        first_arg = false;
        if (stage.expr.inputs[i].kind == NamedInput::Kind::RuntimeScalarFp32) {
            ss << scalarStorageType(input_dtypes[i]) << " in" << i;
        } else {
            ss << "const " << scalarStorageType(input_dtypes[i]) << "* __restrict__ in" << i;
        }
    }

    if (!first_arg) {
        ss << ", ";
    }
    ss << output_type << "* __restrict__ out0, " << index_type << " numRows, " << index_type << " numCols) {\n";

    if (emit_packed_low_precision_path) {
        const uint32_t pack_scalars = transposePackScalars(output_dtype);
        const std::string pack_type = transposePackType(output_dtype);

        ss << "  constexpr unsigned int PACK_SCALARS = " << pack_scalars << "U;\n";
        ss << "  constexpr unsigned int TILE_COL_SCALARS = TILE_DIM * PACK_SCALARS;\n";
        ss << "  using Pack = " << pack_type << ";\n";
        if (emit_fp8_vectorized_pair_path) {
            ss << "  using Vec2 = " << transposeVector2StorageType(output_dtype) << ";\n";
        }
        ss << "  union PackRaw { unsigned int raw; Pack pack; };\n";
        ss << "  __shared__ unsigned int tile[TILE_DIM][TILE_DIM + 1];\n";
        ss << "  const " << index_type << " rowStart = static_cast<" << index_type << ">(blockIdx.y) * TILE_DIM;\n";
        ss << "  const " << index_type << " colStart = static_cast<" << index_type << ">(blockIdx.x) * TILE_COL_SCALARS;\n";
        ss << "  const unsigned int packedCol = threadIdx.x;\n";
        ss << "  const " << index_type << " logicalColBase = colStart + static_cast<" << index_type
           << ">(packedCol) * PACK_SCALARS;\n";
        if (emit_fp8_vectorized_pair_path) {
            for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
                const ExprNode& node = stage.expr.nodes[node_idx];
                if (node.op != ExprOp::SCALAR_FP && node.op != ExprOp::FILL) {
                    continue;
                }
                ss << "  const half2 c" << node_idx << " = " << emitVector2ScalarLiteral(node.scalar_fp, output_dtype) << ";\n";
            }
        }
        ss << "\n";

        ss << "  for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {\n";
        ss << "    const " << index_type << " logicalRow = rowStart + threadIdx.y + j;\n";
        ss << "    Pack output_pack{};\n";
        ss << "    if (logicalRow < numRows) {\n";
        if (emit_fp8_vectorized_pair_path) {
            const uint32_t pairs_per_pack = pack_scalars / 2U;
            ss << "      Vec2* output_vec2 = reinterpret_cast<Vec2*>(&output_pack);\n";
            ss << "      bool pair_done[PACK_SCALARS / 2U] = {};\n";
            for (uint32_t pair = 0; pair < pairs_per_pack; ++pair) {
                const std::string suffix = "_btp" + std::to_string(pair);
                const std::string suffix0 = suffix + "_0";
                const std::string suffix1 = suffix + "_1";
                ss << "      {\n";
                ss << "        constexpr unsigned int PAIR = " << pair << "U;\n";
                ss << "        constexpr unsigned int PAIR_BASE_LANE = PAIR * 2U;\n";
                ss << "        const " << index_type << " logicalColPairBase = logicalColBase + PAIR_BASE_LANE;\n";
                ss << "        if (logicalColPairBase + 1U < numCols) {\n";
                ss << "          const " << index_type << " logical_idx" << suffix0 << " = static_cast<" << index_type
                   << ">(logicalRow) * static_cast<" << index_type << ">(numCols) + static_cast<" << index_type
                   << ">(logicalColPairBase);\n";
                for (uint32_t input_slot : group.used_input_slots) {
                    ss << "          " << index_type << " in" << input_slot << "_offset" << suffix0 << " = "
                       << emitUnsignedLiteral(0, use_uint32_index_math) << ";\n";
                }
                for (size_t used_i : scalar_pack_indices) {
                    const uint32_t input_slot = group.used_input_slots[used_i];
                    ss << "          " << index_type << " in" << input_slot << "_offset" << suffix1 << " = "
                       << emitUnsignedLiteral(0, use_uint32_index_math) << ";\n";
                }
                if (!group.active_axes.empty()) {
                    ss << "\n";
                    emitSpecializedBroadcastOffsetMath(
                        ss, group, all_used_indices, "logical_idx" + suffix0, suffix0, "          ", use_uint32_index_math);
                    if (!scalar_pack_indices.empty()) {
                        ss << "          const " << index_type << " logical_idx" << suffix1 << " = logical_idx" << suffix0 << " + "
                           << emitUnsignedLiteral(1, use_uint32_index_math) << ";\n";
                        emitSpecializedBroadcastOffsetMath(
                            ss, group, scalar_pack_indices, "logical_idx" + suffix1, suffix1, "          ", use_uint32_index_math);
                    }
                    ss << "\n";
                }
                emitVector2NodeDefinitionsForSuffix(
                    ss,
                    stage.expr,
                    output_dtype,
                    transposeVector2StorageType(output_dtype),
                    suffix,
                    "          ",
                    [&](uint32_t input_slot) {
                        const auto kind_it = input_load_kind_by_slot.find(input_slot);
                        if (kind_it == input_load_kind_by_slot.end()) {
                            throw std::runtime_error("Missing input load kind for transposed vector broadcast input.");
                        }
                        if (kind_it->second == SpecializedInputLoadKind::NativeVector) {
                            return emitVector2BroadcastNativeLoad(transposeVector2StorageType(output_dtype),
                                                                  "in" + std::to_string(input_slot),
                                                                  "in" + std::to_string(input_slot) + "_offset" + suffix0);
                        }
                        const std::string var0 =
                            "in" + std::to_string(input_slot) + "[in" + std::to_string(input_slot) + "_offset" + suffix0 + "]";
                        const std::string var1 =
                            "in" + std::to_string(input_slot) + "[in" + std::to_string(input_slot) + "_offset" + suffix1 + "]";
                        return emitVector2BroadcastPackLoad(output_type, var0, var1);
                    },
                    [&](uint32_t scalar_node_idx) { return "c" + std::to_string(scalar_node_idx); },
                    true);
                ss << "          output_vec2[PAIR] = "
                   << vector_storage_conversion(transposeVector2StorageType(output_dtype), refWithSuffix(output.local_node_idx, suffix))
                   << ";\n";
                ss << "          pair_done[PAIR] = true;\n";
                ss << "        }\n";
                ss << "      }\n";
            }
        }
        ss << "      " << output_type << "* output_scalar = reinterpret_cast<" << output_type << "*>(&output_pack);\n";
        ss << "      for (unsigned int lane = 0; lane < PACK_SCALARS; ++lane) {\n";
        if (emit_fp8_vectorized_pair_path) {
            ss << "        if (pair_done[lane / 2U]) continue;\n";
        }
        ss << "        const " << index_type << " logicalCol = logicalColBase + lane;\n";
        ss << "        if (logicalCol < numCols) {\n";
        ss << "          const " << index_type << " logical_idx = static_cast<" << index_type << ">(logicalRow) * static_cast<"
           << index_type << ">(numCols) + static_cast<" << index_type << ">(logicalCol);\n";
        for (uint32_t input_slot : group.used_input_slots) {
            ss << "          " << index_type << " in" << input_slot << "_offset = " << emitUnsignedLiteral(0, use_uint32_index_math)
               << ";\n";
        }
        if (!group.active_axes.empty()) {
            ss << "\n";
            emitSpecializedBroadcastOffsetMath(ss, group, all_used_indices, "logical_idx", "", "          ", use_uint32_index_math);
            ss << "\n";
        }
        ss << "          {\n";
        for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
            if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
                continue;
            }
            emitScalarNode(ss, stage.expr, node_idx, /*broadcast_support=*/true, "            ");
        }
        ss << "            output_scalar[lane] = " << emitResolvedScalarValueExpr(stage.expr, output.local_node_idx, output_dtype) << ";\n";
        ss << "          }\n";
        ss << "        }\n";
        ss << "      }\n";
        ss << "    }\n";
        ss << "    PackRaw raw{};\n";
        ss << "    raw.pack = output_pack;\n";
        ss << "    tile[threadIdx.y + j][packedCol] = raw.raw;\n";
        ss << "  }\n\n";
        ss << "  __syncthreads();\n\n";
        ss << "  constexpr unsigned int PACKED_OUTPUT_COLS_PER_TILE = TILE_DIM / PACK_SCALARS;\n";
        ss << "  constexpr unsigned int PACKED_STORES_PER_TILE = TILE_COL_SCALARS * PACKED_OUTPUT_COLS_PER_TILE;\n";
        ss << "  const unsigned int threadLinear = threadIdx.y * TILE_DIM + threadIdx.x;\n";
        ss << "  for (unsigned int packedStore = threadLinear; packedStore < PACKED_STORES_PER_TILE; packedStore += TILE_DIM * BLOCK_ROWS) "
              "{\n";
        ss << "    const unsigned int localOutRow = packedStore / PACKED_OUTPUT_COLS_PER_TILE;\n";
        ss << "    const unsigned int localOutPackCol = packedStore - localOutRow * PACKED_OUTPUT_COLS_PER_TILE;\n";
        ss << "    const unsigned int inputPackCol = localOutRow / PACK_SCALARS;\n";
        ss << "    const unsigned int inputColLane = localOutRow - inputPackCol * PACK_SCALARS;\n";
        ss << "    const " << index_type << " outputRow = colStart + localOutRow;\n";
        ss << "    const " << index_type << " outputColBase = rowStart + static_cast<" << index_type
           << ">(localOutPackCol) * PACK_SCALARS;\n";
        ss << "    Pack output_pack{};\n";
        ss << "    " << output_type << "* output_scalar = reinterpret_cast<" << output_type << "*>(&output_pack);\n";
        ss << "    for (unsigned int lane = 0; lane < PACK_SCALARS; ++lane) {\n";
        ss << "      PackRaw input_raw{};\n";
        ss << "      input_raw.raw = tile[localOutPackCol * PACK_SCALARS + lane][inputPackCol];\n";
        ss << "      const " << output_type << "* input_scalar = reinterpret_cast<const " << output_type << "*>(&input_raw.pack);\n";
        ss << "      output_scalar[lane] = input_scalar[inputColLane];\n";
        ss << "    }\n";
        ss << "    const " << index_type << " out_base_idx = static_cast<" << index_type << ">(outputRow) * static_cast<" << index_type
           << ">(numRows) +\n";
        ss << "                                     static_cast<" << index_type << ">(outputColBase);\n";
        ss << "    const bool outputPackedStoreOk = (outputRow < numCols) && (outputColBase + PACK_SCALARS <= numRows) &&\n";
        ss << "                                     ((out_base_idx % PACK_SCALARS) == " << emitUnsignedLiteral(0, use_uint32_index_math) << ");\n";
        ss << "    if (outputPackedStoreOk) {\n";
        ss << "      Pack* out_ptr = reinterpret_cast<Pack*>(out0 + out_base_idx);\n";
        ss << "      *out_ptr = output_pack;\n";
        ss << "    } else if (outputRow < numCols) {\n";
        ss << "      for (unsigned int lane = 0; lane < PACK_SCALARS; ++lane) {\n";
        ss << "        const " << index_type << " outputCol = outputColBase + lane;\n";
        ss << "        if (outputCol < numRows) {\n";
        ss << "          out0[outputRow * numRows + outputCol] = output_scalar[lane];\n";
        ss << "        }\n";
        ss << "      }\n";
        ss << "    }\n";
        ss << "  }\n";
        ss << "}\n";
        return ss.str();
    }

    ss << "  __shared__ " << output_type << " tile[TILE_DIM][TILE_DIM + 1];\n";
    ss << "  const " << index_type << " x = static_cast<" << index_type << ">(blockIdx.x) * TILE_DIM + threadIdx.x;\n";
    ss << "  const " << index_type << " y = static_cast<" << index_type << ">(blockIdx.y) * TILE_DIM + threadIdx.y;\n";
    ss << "  for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {\n";
    ss << "    const " << index_type << " logical_row = y + j;\n";
    ss << "    const " << index_type << " logical_col = x;\n";
    ss << "    if (logical_col < numCols && logical_row < numRows) {\n";
    ss << "      const " << index_type << " logical_idx = static_cast<" << index_type << ">(logical_row) * static_cast<" << index_type
       << ">(numCols) + static_cast<" << index_type << ">(logical_col);\n";
    for (uint32_t input_slot : group.used_input_slots) {
        ss << "      " << index_type << " in" << input_slot << "_offset = " << emitUnsignedLiteral(0, use_uint32_index_math) << ";\n";
    }
    if (!group.active_axes.empty()) {
        ss << "\n";
        emitSpecializedBroadcastOffsetMath(ss, group, all_used_indices, "logical_idx", "", "      ", use_uint32_index_math);
        ss << "\n";
    }
    for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
        if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
            continue;
        }
        emitScalarNode(ss, stage.expr, node_idx, /*broadcast_support=*/true, "      ");
    }
    ss << "      tile[threadIdx.y + j][threadIdx.x] = " << emitResolvedScalarValueExpr(stage.expr, output.local_node_idx, output_dtype)
       << ";\n";
    ss << "    }\n";
    ss << "  }\n\n";
    ss << "  __syncthreads();\n\n";
    ss << "  const " << index_type << " tx = static_cast<" << index_type << ">(blockIdx.y) * TILE_DIM + threadIdx.x;\n";
    ss << "  const " << index_type << " ty = static_cast<" << index_type << ">(blockIdx.x) * TILE_DIM + threadIdx.y;\n";
    ss << "  for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {\n";
    ss << "    if (tx < numRows && (ty + j) < numCols) {\n";
    ss << "      const " << index_type << " out_idx = static_cast<" << index_type << ">(ty + j) * static_cast<" << index_type
       << ">(numRows) +\n";
    ss << "                                   static_cast<" << index_type << ">(tx);\n";
    ss << "      out0[out_idx] = tile[threadIdx.x][threadIdx.y + j];\n";
    ss << "    }\n";
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

std::string CudaSourceEmitter::emitSpecializedBroadcast(const CompiledExecutionStage& stage,
                                                        const std::vector<SpecializedBroadcastGroup>& groups,
                                                        const std::string& kernel_name) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        throw std::runtime_error("emitSpecializedBroadcast called on non-fused stage.");
    }
    if (groups.empty()) {
        throw std::runtime_error("emitSpecializedBroadcast requires at least one group.");
    }
    if (stageHasTransposedMaterializedOutput(stage.outputs)) {
        return emitTiledTransposeMaterializedSpecializedBroadcast(stage, groups, kernel_name);
    }

    Optional<DataType> vectorized_dtype = getVectorizedStageStorageDType(stage);
    if (vectorized_dtype.isPresent()) {
        return emitVector2SpecializedBroadcast(stage, groups, vectorized_dtype.get(), kernel_name);
    }

    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const std::vector<DataType> output_dtypes = collectOutputDTypes(stage);

    std::ostringstream ss;
    emitRequiredHeaders(stage.expr, ss);

    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";

    bool first_arg = true;
    for (uint32_t i = 0; i < input_dtypes.size(); ++i) {
        if (!first_arg) {
            ss << ", ";
        }
        first_arg = false;
        if (stage.expr.inputs[i].kind == NamedInput::Kind::RuntimeScalarFp32) {
            ss << scalarStorageType(input_dtypes[i]) << " in" << i;
        } else {
            ss << "const " << scalarStorageType(input_dtypes[i]) << "* in" << i;
        }
    }
    for (uint32_t i = 0; i < output_dtypes.size(); ++i) {
        if (!first_arg) {
            ss << ", ";
        }
        first_arg = false;
        ss << scalarStorageType(output_dtypes[i]) << "* out" << i;
    }

    const bool use_uint32_index_math = groupsSupportUInt32IndexMath(groups);
    const std::string index_type = emittedIndexType(use_uint32_index_math);

    ss << ") {\n";
    ss << "  " << index_type << " idx = " << emitFlatThreadIndexExpr(use_uint32_index_math) << ";\n\n";

    for (uint32_t g = 0; g < groups.size(); ++g) {
        const SpecializedBroadcastGroup& group = groups[g];
        const std::vector<uint32_t> required_nodes = orderedRequiredNodesForGroup(stage, group.output_indices);

        std::vector<size_t> all_used_indices(group.used_input_slots.size());
        std::iota(all_used_indices.begin(), all_used_indices.end(), 0);

        ss << "  if (idx < " << emitUnsignedLiteral(group.numel, use_uint32_index_math) << ") {\n";

        for (uint32_t input_slot : group.used_input_slots) {
            ss << "    " << index_type << " in" << input_slot << "_offset = " << emitUnsignedLiteral(0, use_uint32_index_math) << ";\n";
        }

        if (!group.active_axes.empty()) {
            ss << "\n";
            emitSpecializedBroadcastOffsetMath(ss, group, all_used_indices, "idx", "", "    ", use_uint32_index_math);
            ss << "\n";
        }

        for (uint32_t node_idx : required_nodes) {
            if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
                continue;
            }
            emitScalarNode(ss, stage.expr, node_idx, /*broadcast_support=*/true, "    ");
        }

        ss << "\n";
        for (uint32_t out_idx : group.output_indices) {
            const CompiledStageOutput& output = stage.outputs[out_idx];
            const DataType output_dtype = requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]);
            ss << "    out" << out_idx << "[idx] = " << emitResolvedScalarValueExpr(stage.expr, output.local_node_idx, output_dtype)
               << ";\n";
        }

        ss << "  }\n\n";
    }

    ss << "}\n";
    return ss.str();
}

}  // namespace ThorImplementation
