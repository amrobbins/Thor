#include "Utilities/Expression/CudaSourceEmitter.h"
#include <optional>
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <algorithm>
#include <cmath>
#include <array>
#include <cctype>
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

using DataType = ThorImplementation::DataType;

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

static std::string emitFloatArrayLiteral(const std::vector<double>& values) {
    std::ostringstream ss;
    ss << "{";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << emitScalarFpLiteral(values[i]);
    }
    ss << "}";
    return ss.str();
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

// Reads lane `lane_var` (a runtime value in [0, lane_count)) of a register chunk through constant-index
// reads and selects. Indexing chunk_data[lane_var] directly makes the chunk addressable at a runtime offset,
// which forces nvcc to place it in local memory (local_bytes_per_thread > 0 in the census).
static std::string emitChunkLaneSelectExpr(const std::string& chunk_data_expr, const std::string& lane_var, uint32_t lane_count) {
    if (lane_count == 0) {
        throw std::runtime_error("emitChunkLaneSelectExpr requires a non-zero lane count.");
    }
    std::string expr = chunk_data_expr + "[" + std::to_string(lane_count - 1) + "]";
    for (uint32_t lane = lane_count - 1; lane-- > 0;) {
        expr = "(" + lane_var + " == " + std::to_string(lane) + "U ? " + chunk_data_expr + "[" + std::to_string(lane) + "] : " + expr + ")";
    }
    return expr;
}

static bool isUnsignedDecimalLiteral(const std::string& text) {
    return !text.empty() && std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

static std::string emitChunkLaneReadExpr(const std::string& chunk_expr,
                                         const std::string& chunk_type,
                                         DataType scalar_dtype,
                                         const std::string& lane_expr,
                                         const std::string& chunk_data_expr = "",
                                         uint32_t lane_count = 0) {
    if (chunk_type == "float4" && scalar_dtype == DataType::FP32) {
        const uint32_t lane = static_cast<uint32_t>(std::stoul(lane_expr));
        return chunk_expr + "." + float4LaneComponent(lane);
    }
    if (!chunk_data_expr.empty() && lane_count > 1 && !isUnsignedDecimalLiteral(lane_expr)) {
        return emitChunkLaneSelectExpr(chunk_data_expr, lane_expr, lane_count);
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

// Gathers one output pack from PACK_SCALARS transposed tile words. Each 32-bit word holds PACK_SCALARS
// scalars; inputColLane (runtime, thread dependent) picks which one. The lane is extracted with shifts on the
// word: indexing the word's bytes as an array at a runtime lane would force it into local memory.
static void emitPackedTransposeTileLaneGather(std::ostringstream& ss, const std::string& output_type) {
    ss << "    constexpr unsigned int SCALAR_BITS = 32U / PACK_SCALARS;\n";
    ss << "    constexpr unsigned int SCALAR_MASK = (PACK_SCALARS == 1U) ? 0xFFFFFFFFU : ((1U << (SCALAR_BITS % 32U)) - 1U);\n";
    ss << "    const unsigned int inputLaneShift = inputColLane * SCALAR_BITS;\n";
    ss << "    unsigned int output_word = 0U;\n";
    ss << "#pragma unroll\n";
    ss << "    for (unsigned int lane = 0; lane < PACK_SCALARS; ++lane) {\n";
    ss << "      const unsigned int tile_word = tile[localOutPackCol * PACK_SCALARS + lane][inputPackCol];\n";
    ss << "      output_word |= ((tile_word >> inputLaneShift) & SCALAR_MASK) << (lane * SCALAR_BITS);\n";
    ss << "    }\n";
    ss << "    PackRaw output_raw{};\n";
    ss << "    output_raw.raw = output_word;\n";
    ss << "    Pack output_pack = output_raw.pack;\n";
    ss << "    const " << output_type << "* output_scalar = reinterpret_cast<const " << output_type << "*>(&output_pack);\n";
}

static std::string emitUnsignedLiteral(uint64_t value, bool use_uint32_index_math) {
    if (use_uint32_index_math && fitsInUInt32(value)) {
        return std::to_string(static_cast<uint32_t>(value)) + "U";
    }
    return std::to_string(value) + "ULL";
}

// ---------------------------------------------------------------------------
// Packet load/store stages shared by the wide (multi-scalar-per-thread) emitters.
//
// A packet kernel owns `lanes` consecutive scalars per tensor per thread. The
// fast path moves each tensor's packet with one naturally aligned vector access.
// Runtime tensors can be storage aliases whose base address is element aligned
// but not packet aligned, and a packet can extend past the end of a view (the
// allocation's terminal padding covers the end of an allocation, not the end of
// a view). So every packet kernel emits:
//
//   load stage:    vector load when (full packet && aligned), otherwise masked
//                  scalar loads into the same packet register
//   compute stage: emitted exactly once, reading lanes from the packet registers
//   store stage:   vector store when (full packet && aligned), otherwise masked
//                  scalar stores
//
// `packets_aligned` is grid-uniform (it depends only on the base pointers), and
// `full_packet` is false only for the final packet, so the fallback never
// diverges a warp on the ordinary dense path and the expression body is never
// duplicated.
//
// Fused elementwise / broadcast kernel pointers are deliberately NOT __restrict__.
// Caller-provided outputs may alias root inputs: optimizers update weights and
// state in place (see the write-after-read handling in FusedEquation stamping).
// With __restrict__ the compiler may re-load a const input after the store to the
// aliased output (or keep it in the read-only path), reading the updated value.
// Same-index aliasing is otherwise race free because each element is read and
// written by one thread, loads first.
struct PacketTensorAccess {
    std::string chunk_var;     // e.g. in0_chunk
    std::string chunk_type;    // float4 / unsigned long long / unsigned int
    std::string scalar_type;   // storage scalar type of one lane
    std::string pointer_expr;  // e.g. in0 (scalar-typed kernel argument)
    std::string base_expr;     // scalar index of lane 0, e.g. scalar_base or in0_offset_base
};

static std::string emitPacketAlignmentPredicate(const std::vector<std::pair<std::string, uint64_t>>& pointers_and_bytes) {
    if (pointers_and_bytes.empty()) {
        return "true";
    }
    std::string predicate;
    for (const auto& [pointer, bytes] : pointers_and_bytes) {
        if (!predicate.empty()) {
            predicate += " && ";
        }
        predicate += "((reinterpret_cast<unsigned long long>(" + pointer + ") & " + std::to_string(bytes - 1ULL) + "ULL) == 0ULL)";
    }
    return predicate;
}

static void emitPacketLoadStage(std::ostringstream& ss,
                                const std::string& indent,
                                const std::vector<PacketTensorAccess>& inputs,
                                uint32_t lanes,
                                const std::string& vector_predicate,
                                const std::function<std::string(uint32_t)>& lane_valid_expr) {
    if (inputs.empty()) {
        return;
    }
    for (const PacketTensorAccess& in : inputs) {
        ss << indent << in.chunk_type << " " << in.chunk_var << "{};\n";
    }
    ss << indent << "if (" << vector_predicate << ") {\n";
    for (const PacketTensorAccess& in : inputs) {
        ss << indent << "  " << in.chunk_var << " = *reinterpret_cast<const " << in.chunk_type << "*>(" << in.pointer_expr << " + "
           << in.base_expr << ");\n";
    }
    ss << indent << "} else {\n";
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        // An empty lane predicate means every lane of the packet is in range
        // (only the base alignment failed).
        const std::string valid = lane_valid_expr(lane);
        const std::string lane_indent = valid.empty() ? indent + "  " : indent + "    ";
        if (!valid.empty()) ss << indent << "  if (" << valid << ") {\n";
        for (const PacketTensorAccess& in : inputs) {
            ss << lane_indent << "reinterpret_cast<" << in.scalar_type << "*>(&" << in.chunk_var << ")[" << lane << "] = " << in.pointer_expr
               << "[" << in.base_expr << " + " << lane << "];\n";
        }
        if (!valid.empty()) ss << indent << "  }\n";
    }
    ss << indent << "}\n";
}

static void emitPacketStoreStage(std::ostringstream& ss,
                                 const std::string& indent,
                                 const std::vector<PacketTensorAccess>& outputs,
                                 uint32_t lanes,
                                 const std::string& vector_predicate,
                                 const std::function<std::string(uint32_t)>& lane_valid_expr) {
    if (outputs.empty()) {
        return;
    }
    ss << indent << "if (" << vector_predicate << ") {\n";
    for (const PacketTensorAccess& out : outputs) {
        ss << indent << "  *reinterpret_cast<" << out.chunk_type << "*>(" << out.pointer_expr << " + " << out.base_expr
           << ") = " << out.chunk_var << ";\n";
    }
    ss << indent << "} else {\n";
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        const std::string valid = lane_valid_expr(lane);
        const std::string lane_indent = valid.empty() ? indent + "  " : indent + "    ";
        if (!valid.empty()) ss << indent << "  if (" << valid << ") {\n";
        for (const PacketTensorAccess& out : outputs) {
            ss << lane_indent << out.pointer_expr << "[" << out.base_expr << " + " << lane << "] = reinterpret_cast<const "
               << out.scalar_type << "*>(&" << out.chunk_var << ")[" << lane << "];\n";
        }
        if (!valid.empty()) ss << indent << "  }\n";
    }
    ss << indent << "}\n";
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

    // Output-side sizes are not enough: offsets are computed on the source side. A non-dense
    // input view (large row pitch), the source of a strided view, or a gathered tensor can be
    // addressed past 2^32 while the output is small (e.g. a target gather from [32768, 152064]
    // logits into [32768, 1]). Every index the kernel forms must fit.
    const auto span_fits = [](const std::vector<uint64_t>& dims, const std::vector<uint64_t>& strides) {
        if (dims.size() != strides.size()) {
            return false;
        }
        unsigned __int128 max_index = 0;
        for (size_t i = 0; i < dims.size(); ++i) {
            if (dims[i] == 0) {
                return true;
            }
            max_index += static_cast<unsigned __int128>(dims[i] - 1ULL) * strides[i];
        }
        return max_index <= static_cast<unsigned __int128>(std::numeric_limits<uint32_t>::max());
    };
    for (size_t used_i = 0; used_i < group.used_input_slots.size(); ++used_i) {
        unsigned __int128 max_offset = 0;
        for (const SpecializedBroadcastAxis& axis : group.active_axes) {
            if (used_i < axis.input_strides.size() && axis.dim > 0) {
                max_offset += static_cast<unsigned __int128>(axis.dim - 1ULL) * axis.input_strides[used_i];
            }
        }
        if (max_offset > static_cast<unsigned __int128>(std::numeric_limits<uint32_t>::max())) {
            return false;
        }
        if (used_i < group.used_input_visible_dims.size() && used_i < group.used_input_visible_strides.size() &&
            !span_fits(group.used_input_visible_dims[used_i], group.used_input_visible_strides[used_i])) {
            return false;
        }
    }
    for (const std::vector<uint64_t>& dims : group.node_dims) {
        unsigned __int128 numel = 1;
        for (uint64_t dim : dims) {
            numel *= dim;
            if (numel > static_cast<unsigned __int128>(std::numeric_limits<uint32_t>::max())) {
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
    if (node.op == ExprOp::ROPE && node.rope_effective_sequence_length_node != UINT32_MAX) {
        collectRequiredNodes(expr, node.rope_effective_sequence_length_node, required);
    }
    if (node.op == ExprOp::ROPE && node.rope_position_ids_node != UINT32_MAX) {
        collectRequiredNodes(expr, node.rope_position_ids_node, required);
    }
    if ((Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) &&
        node.op != ExprOp::RAGGED_VALUEWISE_EXTENT) {
        collectRequiredNodes(expr, node.rhs, required);
    }
    if (Expression::isTernaryOp(node.op)) {
        collectRequiredNodes(expr, node.aux, required);
    }
}

static void collectRequiredNodesExcludingIndexAwareChildren(const PhysicalExpression& expr,
                                                            uint32_t node_idx,
                                                            std::unordered_set<uint32_t>& required) {
    if (!required.insert(node_idx).second) {
        return;
    }

    const ExprNode& node = expr.nodes[node_idx];
    if (Expression::isLeafOp(node.op) || node.op == ExprOp::STRIDED_VIEW_BACKWARD) {
        return;
    }

    collectRequiredNodesExcludingIndexAwareChildren(expr, node.lhs, required);
    if (node.op == ExprOp::ROPE && node.rope_effective_sequence_length_node != UINT32_MAX) {
        collectRequiredNodesExcludingIndexAwareChildren(expr, node.rope_effective_sequence_length_node, required);
    }
    if ((Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) &&
        node.op != ExprOp::RAGGED_VALUEWISE_EXTENT) {
        collectRequiredNodesExcludingIndexAwareChildren(expr, node.rhs, required);
    }
    if (Expression::isTernaryOp(node.op)) {
        collectRequiredNodesExcludingIndexAwareChildren(expr, node.aux, required);
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
    if (!node.output_dtype.has_value()) {
        throw runtime_error("Fused stage node is missing resolved output_dtype.");
    }
    return node.output_dtype.value();
}

static DataType requireNodeInputTensorDType(const ExprNode& node) {
    if (!node.input_tensor_dtype.has_value()) {
        throw runtime_error("Fused stage INPUT node is missing resolved input_tensor_dtype.");
    }
    return node.input_tensor_dtype.value();
}

static DataType requireNodeComputeDType(const ExprNode& node) {
    if (!node.compute_dtype.has_value()) {
        throw runtime_error("Fused stage node is missing resolved compute_dtype.");
    }
    return node.compute_dtype.value();
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
        case DataType::INT32:
            return "int";
        case DataType::UINT32:
            return "unsigned int";
        case DataType::INT64:
            return "long long";
        case DataType::UINT64:
            return "unsigned long long";
        case DataType::BOOLEAN:
            return "unsigned char";
        case DataType::UINT8:
            return "unsigned char";
        case DataType::INT8:
            return "signed char";
        case DataType::UINT16:
            return "unsigned short";
        case DataType::INT16:
            return "short";
        default:
            throw runtime_error("Unsupported scalar storage dtype in fused stage emitter: " + TensorDescriptor::getElementTypeName(dtype));
    }
}

static uint32_t scalarStorageTypeSizeBytes(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
        case DataType::INT32:
        case DataType::UINT32:
            return 4;
        case DataType::FP16:
        case DataType::BF16:
            return 2;
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return 1;
        case DataType::INT64:
        case DataType::UINT64:
            return 8;
        case DataType::BOOLEAN:
        case DataType::UINT8:
        case DataType::INT8:
            return 1;
        case DataType::UINT16:
        case DataType::INT16:
            return 2;
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

static void emitSharedTransposeWordHelpers(std::ostringstream& ss) {
    // NVRTC bundled headers in CUDA 13.3 do not ship a top-level vector_types.h.
    // NVRTC still provides CUDA vector builtins such as float4, uchar4, and ushort2
    // without an explicit include, so do not emit the legacy header include here.
    ss << R"(template <unsigned int Bytes>
struct thor_transpose_storage_for_bytes;

template <>
struct thor_transpose_storage_for_bytes<1U> {
    using type = unsigned char;
};

template <>
struct thor_transpose_storage_for_bytes<2U> {
    using type = unsigned short;
};

template <>
struct thor_transpose_storage_for_bytes<4U> {
    using type = unsigned int;
};

template <unsigned int SlotBytes>
struct thor_transpose_word_vector_for_slot_bytes;

template <>
struct thor_transpose_word_vector_for_slot_bytes<1U> {
    using type = uchar4;
};

template <>
struct thor_transpose_word_vector_for_slot_bytes<2U> {
    using type = ushort2;
};

template <>
struct thor_transpose_word_vector_for_slot_bytes<4U> {
    using type = unsigned int;
};

template <typename T>
__device__ __forceinline__ typename thor_transpose_storage_for_bytes<sizeof(T)>::type thor_transpose_scalar_bits(T value) {
    static_assert(sizeof(T) == 1U || sizeof(T) == 2U || sizeof(T) == 4U,
                  "transpose shared-memory swizzle supports scalar storage types of 1, 2, or 4 bytes");
    union ScalarCaster {
        T value;
        typename thor_transpose_storage_for_bytes<sizeof(T)>::type bits;
    };
    ScalarCaster caster;
    caster.bits = 0;
    caster.value = value;
    return caster.bits;
}

template <typename T>
__device__ __forceinline__ T thor_transpose_scalar_from_bits(typename thor_transpose_storage_for_bytes<sizeof(T)>::type bits) {
    static_assert(sizeof(T) == 1U || sizeof(T) == 2U || sizeof(T) == 4U,
                  "transpose shared-memory swizzle supports scalar storage types of 1, 2, or 4 bytes");
    union ScalarCaster {
        typename thor_transpose_storage_for_bytes<sizeof(T)>::type bits;
        T value;
    };
    ScalarCaster caster;
    caster.bits = bits;
    return caster.value;
}

template <typename T>
__device__ __forceinline__ unsigned int thor_pack_transpose_word(T value) {
    static_assert(sizeof(T) <= sizeof(unsigned int), "transpose shared-memory swizzle supports scalar storage types up to 32 bits");
    return static_cast<unsigned int>(thor_transpose_scalar_bits(value));
}

template <typename T>
__device__ __forceinline__ T thor_unpack_transpose_word(unsigned int word) {
    static_assert(sizeof(T) <= sizeof(unsigned int), "transpose shared-memory swizzle supports scalar storage types up to 32 bits");
    return thor_transpose_scalar_from_bits<T>(static_cast<typename thor_transpose_storage_for_bytes<sizeof(T)>::type>(word));
}

template <unsigned int SlotBytes>
__device__ __forceinline__ typename thor_transpose_storage_for_bytes<SlotBytes>::type thor_get_transpose_pack_lane_bits(unsigned int word,
                                                                                                                       unsigned int lane) {
    static_assert(SlotBytes == 1U || SlotBytes == 2U || SlotBytes == 4U,
                  "transpose logical swizzle slots must be 1, 2, or 4 bytes");
    union WordCaster {
        unsigned int raw;
        typename thor_transpose_word_vector_for_slot_bytes<SlotBytes>::type lanes;
    };
    WordCaster caster;
    caster.raw = word;
    if constexpr (SlotBytes == 1U) {
        if (lane == 0U) return caster.lanes.x;
        if (lane == 1U) return caster.lanes.y;
        if (lane == 2U) return caster.lanes.z;
        return caster.lanes.w;
    } else if constexpr (SlotBytes == 2U) {
        return lane == 0U ? caster.lanes.x : caster.lanes.y;
    } else {
        return caster.raw;
    }
}

template <unsigned int SlotBytes>
__device__ __forceinline__ unsigned int thor_set_transpose_pack_lane_bits(
    unsigned int word, unsigned int lane, typename thor_transpose_storage_for_bytes<SlotBytes>::type value) {
    static_assert(SlotBytes == 1U || SlotBytes == 2U || SlotBytes == 4U,
                  "transpose logical swizzle slots must be 1, 2, or 4 bytes");
    union WordCaster {
        unsigned int raw;
        typename thor_transpose_word_vector_for_slot_bytes<SlotBytes>::type lanes;
    };
    WordCaster caster;
    caster.raw = word;
    if constexpr (SlotBytes == 1U) {
        if (lane == 0U) {
            caster.lanes.x = value;
        } else if (lane == 1U) {
            caster.lanes.y = value;
        } else if (lane == 2U) {
            caster.lanes.z = value;
        } else {
            caster.lanes.w = value;
        }
    } else if constexpr (SlotBytes == 2U) {
        if (lane == 0U) {
            caster.lanes.x = value;
        } else {
            caster.lanes.y = value;
        }
    } else {
        caster.raw = value;
    }
    return caster.raw;
}

template <unsigned int SlotBytes, typename T>
__device__ __forceinline__ unsigned int thor_set_transpose_pack_lane(unsigned int word, unsigned int lane, T value) {
    static_assert(SlotBytes == 1U || SlotBytes == 2U || SlotBytes == 4U,
                  "transpose logical swizzle slots must be 1, 2, or 4 bytes");
    static_assert(sizeof(T) <= SlotBytes, "transpose logical swizzle lane value must fit in its selected bank slot");
    using SlotStorage = typename thor_transpose_storage_for_bytes<SlotBytes>::type;
    return thor_set_transpose_pack_lane_bits<SlotBytes>(word, lane, static_cast<SlotStorage>(thor_transpose_scalar_bits(value)));
}

template <unsigned int SlotBytes, typename T>
__device__ __forceinline__ T thor_unpack_transpose_pack_lane(unsigned int word, unsigned int lane) {
    static_assert(SlotBytes == 1U || SlotBytes == 2U || SlotBytes == 4U,
                  "transpose logical swizzle slots must be 1, 2, or 4 bytes");
    static_assert(sizeof(T) <= SlotBytes, "transpose logical swizzle lane value must fit in its selected bank slot");
    using ValueStorage = typename thor_transpose_storage_for_bytes<sizeof(T)>::type;
    const auto lane_bits = thor_get_transpose_pack_lane_bits<SlotBytes>(word, lane);
    return thor_transpose_scalar_from_bits<T>(static_cast<ValueStorage>(lane_bits));
}

template <unsigned int InputBytes, unsigned int SlotBytes>
__device__ __forceinline__ unsigned int thor_expand_transpose_dense_input_word(unsigned int raw) {
    static_assert(InputBytes == 1U || InputBytes == 2U || InputBytes == 4U,
                  "transpose packed dense-input loads support 1, 2, or 4 byte inputs");
    static_assert(SlotBytes == 1U || SlotBytes == 2U || SlotBytes == 4U,
                  "transpose packed dense-input slots must be 1, 2, or 4 bytes");
    static_assert(InputBytes <= SlotBytes, "dense-input packed word cannot narrow values while staging the transpose tile");

    if constexpr (InputBytes == SlotBytes) {
        return raw;
    } else if constexpr (InputBytes == 1U && SlotBytes == 2U) {
        union InputCaster {
            unsigned int raw;
            uchar4 lanes;
        } input;
        union OutputCaster {
            unsigned int raw;
            ushort2 lanes;
        } output;
        input.raw = raw;
        output.raw = 0u;
        output.lanes.x = static_cast<unsigned short>(input.lanes.x);
        output.lanes.y = static_cast<unsigned short>(input.lanes.y);
        return output.raw;
    } else if constexpr (InputBytes == 1U && SlotBytes == 4U) {
        union InputCaster {
            unsigned int raw;
            uchar4 lanes;
        } input;
        input.raw = raw;
        return static_cast<unsigned int>(input.lanes.x);
    } else if constexpr (InputBytes == 2U && SlotBytes == 4U) {
        union InputCaster {
            unsigned int raw;
            ushort2 lanes;
        } input;
        input.raw = raw;
        return static_cast<unsigned int>(input.lanes.x);
    } else {
        static_assert(InputBytes == SlotBytes, "Unhandled dense-input packed transpose expansion case");
        return raw;
    }
}

)";
}

static void emitSharedTransposeTileDeclaration(std::ostringstream& ss, const std::string& columns_expr = "TILE_DIM + 1") {
    ss << "  // Shared-memory transpose tiles are always 32-bit bank words. The +1 column\n";
    ss << "  // padding is therefore really +1 4-byte bank slot, not +1 scalar element.\n";
    ss << "  // Narrow scalar values occupy/pack into these words so transposed shared-memory\n";
    ss << "  // reads/writes keep the same bank-friendly layout as the classic 32x33 tile.\n";
    ss << "  __shared__ unsigned int tile[TILE_DIM][" << columns_expr << "];\n";
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

static std::string uint8Vector4LaneMember(uint32_t lane) {
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
            throw runtime_error("Unsupported uint8 vector4 lane: " + std::to_string(lane));
    }
}

static std::string emitFp8PackLaneMemberExpr(const std::string& pack_expr, uint32_t lane, const DataType input_dtype) {
    std::string dtypeString;
    switch (input_dtype) {
        case DataType::FP8_E5M2:
            dtypeString = "__nv_fp8_e5m2";
            break;
        case DataType::FP8_E4M3:
            dtypeString = "__nv_fp8_e4m3";
            break;
        default:
            throw runtime_error("Unsupported fp8 pack lane dtype: " + TensorDescriptor::getElementTypeName(input_dtype));
    }

    switch (lane) {
        case 0:
        case 1:
        case 2:
        case 3:
            return "([](unsigned char raw) { " + dtypeString +
                   " v; "
                   "v.__x = static_cast<__nv_fp8_storage_t>(raw); "
                   "return v; "
                   "})((" +
                   pack_expr + ")." + uint8Vector4LaneMember(lane) + ")";
        default:
            throw runtime_error("Unsupported fp8 pack lane: " + std::to_string(lane));
    }
}

static bool isHalf2ComputeStorageDType(DataType dtype) { return dtype == DataType::FP16 || isFp8DType(dtype); }

static bool expressionUsesOp(const PhysicalExpression& expr, ExprOp op) {
    return std::any_of(expr.nodes.begin(), expr.nodes.end(), [&](const ExprNode& node) { return node.op == op; });
}

static void emitDigammaHelperDefinition(std::ostringstream& ss) {
    ss << R"(__device__ __forceinline__ float thor_digammaf(float x) {
  constexpr float pi = 3.14159265358979323846264338327950288f;

  if (isnan(x)) {
    return x;
  }
  if (isinf(x)) {
    return x > 0.0f ? x : nanf("");
  }

  float result = 0.0f;
  if (x <= 0.0f) {
    const float floored = floorf(x);
    if (x == floored) {
      return nanf("");
    }
    result -= pi / tanf(pi * x);
    x = 1.0f - x;
  }

  while (x < 8.0f) {
    result -= 1.0f / x;
    x += 1.0f;
  }

  const float inv = 1.0f / x;
  const float inv2 = inv * inv;
  const float inv4 = inv2 * inv2;
  const float inv6 = inv4 * inv2;
  const float inv8 = inv4 * inv4;
  const float inv10 = inv8 * inv2;

  result += logf(x) - 0.5f * inv;
  result -= inv2 * (1.0f / 12.0f);
  result += inv4 * (1.0f / 120.0f);
  result -= inv6 * (1.0f / 252.0f);
  result += inv8 * (1.0f / 240.0f);
  result -= inv10 * (1.0f / 132.0f);
  return result;
}

)";
}

static void emitCudaFp16Header(std::ostringstream& ss) {
    // Runtime JIT uses NVRTC's CUDA 13.3 bundled headers, passed via
    // --use-bundled-headers by EquationCompiler.  Use the public CUDA Math API
    // header normally and rely on its official half/half2 aliases.
    ss << "#include <cuda_fp16.h>\n";
}

// thor_half_to_fp8*_satfinite / thor_half2_to_fp8x2*_satfinite convert FP16 -> FP8 directly
// (cvt.rn.satfinite.e4m3x2.f16x2 / e5m2x2 on sm89+): the same rounding as through the exact FP32
// value, without the f16 -> f32 conversions.
static void emitFp8E4M3SatfiniteHelpers(std::ostringstream& ss) {
    ss << R"(
__device__ __forceinline__ __nv_fp8_e4m3 thor_to_fp8_e4m3_satfinite(float value) {
  __nv_fp8_e4m3 result;
  result.__x = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  return result;
}

__device__ __forceinline__ __nv_fp8x2_e4m3 thor_to_fp8x2_e4m3_satfinite(float2 value) {
  __nv_fp8x2_e4m3 result;
  result.__x = __nv_cvt_float2_to_fp8x2(value, __NV_SATFINITE, __NV_E4M3);
  return result;
}

__device__ __forceinline__ __nv_fp8_e4m3 thor_half_to_fp8_e4m3_satfinite(__half value) {
  __nv_fp8_e4m3 result;
  result.__x = __nv_cvt_halfraw_to_fp8(static_cast<__half_raw>(value), __NV_SATFINITE, __NV_E4M3);
  return result;
}

__device__ __forceinline__ __nv_fp8x2_e4m3 thor_half2_to_fp8x2_e4m3_satfinite(__half2 value) {
  __nv_fp8x2_e4m3 result;
  result.__x = __nv_cvt_halfraw2_to_fp8x2(static_cast<__half2_raw>(value), __NV_SATFINITE, __NV_E4M3);
  return result;
}

__device__ __forceinline__ __nv_fp8x2_e4m3 thor_fp8x2_e4m3_from_lanes(__nv_fp8_e4m3 lo, __nv_fp8_e4m3 hi) {
  __nv_fp8x2_e4m3 result;
  result.__x = static_cast<__nv_fp8x2_storage_t>(static_cast<unsigned int>(lo.__x) | (static_cast<unsigned int>(hi.__x) << 8));
  return result;
}

)";
}

static void emitFp8E5M2SatfiniteHelpers(std::ostringstream& ss) {
    ss << R"(
__device__ __forceinline__ __nv_fp8_e5m2 thor_to_fp8_e5m2_satfinite(float value) {
  __nv_fp8_e5m2 result;
  result.__x = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E5M2);
  return result;
}

__device__ __forceinline__ __nv_fp8x2_e5m2 thor_to_fp8x2_e5m2_satfinite(float2 value) {
  __nv_fp8x2_e5m2 result;
  result.__x = __nv_cvt_float2_to_fp8x2(value, __NV_SATFINITE, __NV_E5M2);
  return result;
}

__device__ __forceinline__ __nv_fp8_e5m2 thor_half_to_fp8_e5m2_satfinite(__half value) {
  __nv_fp8_e5m2 result;
  result.__x = __nv_cvt_halfraw_to_fp8(static_cast<__half_raw>(value), __NV_SATFINITE, __NV_E5M2);
  return result;
}

__device__ __forceinline__ __nv_fp8x2_e5m2 thor_half2_to_fp8x2_e5m2_satfinite(__half2 value) {
  __nv_fp8x2_e5m2 result;
  result.__x = __nv_cvt_halfraw2_to_fp8x2(static_cast<__half2_raw>(value), __NV_SATFINITE, __NV_E5M2);
  return result;
}

__device__ __forceinline__ __nv_fp8x2_e5m2 thor_fp8x2_e5m2_from_lanes(__nv_fp8_e5m2 lo, __nv_fp8_e5m2 hi) {
  __nv_fp8x2_e5m2 result;
  result.__x = static_cast<__nv_fp8x2_storage_t>(static_cast<unsigned int>(lo.__x) | (static_cast<unsigned int>(hi.__x) << 8));
  return result;
}

)";
}

static void emitRequiredHeaders(const PhysicalExpression& expr, std::ostringstream& ss) {
    bool need_fp16 = false;
    bool need_bf16 = false;
    bool need_fp8 = false;
    bool need_fp8_e4m3 = false;
    bool need_fp8_e5m2 = false;

    auto note_dtype = [&](DataType dtype) {
        switch (dtype) {
            case DataType::FP16:
                need_fp16 = true;
                break;
            case DataType::BF16:
                need_bf16 = true;
                break;
            case DataType::FP8_E4M3:
                need_fp16 = true;
                need_fp8 = true;
                need_fp8_e4m3 = true;
                break;
            case DataType::FP8_E5M2:
                need_fp16 = true;
                need_fp8 = true;
                need_fp8_e5m2 = true;
                break;
            default:
                break;
        }
    };

    for (const ExprNode& node : expr.nodes) {
        if (node.input_tensor_dtype.has_value()) {
            note_dtype(node.input_tensor_dtype.value());
        }
        if (node.output_dtype.has_value()) {
            note_dtype(node.output_dtype.value());
        }
        if (node.compute_dtype.has_value()) {
            note_dtype(node.compute_dtype.value());
        }
    }

    if (need_fp16) {
        emitCudaFp16Header(ss);
    }
    if (need_bf16) {
        ss << "#include <cuda_bf16.h>\n";
    }
    if (need_fp8) {
        ss << "#include <cuda_fp8.h>\n";
        if (need_fp8_e4m3) {
            emitFp8E4M3SatfiniteHelpers(ss);
        }
        if (need_fp8_e5m2) {
            emitFp8E5M2SatfiniteHelpers(ss);
        }
    }
    // CUDA 13.3 NVRTC bundled headers do not expose the legacy top-level
    // <math_functions.h> header.  NVRTC/libdevice still provide the device
    // math builtins used below (tgammaf, lgammaf, floorf, tanf, logf, nanf,
    // isnan, isinf), so do not emit that legacy include for JIT sources.
    if (expressionUsesOp(expr, ExprOp::DIGAMMA)) {
        emitDigammaHelperDefinition(ss);
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

static bool expressionHasIndexAwareOps(const PhysicalExpression& expr) {
    return std::any_of(expr.nodes.begin(), expr.nodes.end(), [](const ExprNode& node) {
        return node.op == ExprOp::ROPE || node.op == ExprOp::STRIDED_VIEW_BACKWARD || node.op == ExprOp::TRANSPOSE ||
               node.op == ExprOp::TAKE_ALONG_AXIS;
    });
}

static bool supportsVectorizedComputeDTypeForStorage(DataType storage_dtype, DataType compute_dtype) {
    switch (storage_dtype) {
        case DataType::FP16:
            return compute_dtype == DataType::FP16;
        case DataType::BF16:
            return compute_dtype == DataType::BF16;
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            // FP8 storage supports BF16 intermediates by default while retaining
            // the explicit FP16 compute path for callers that request it.
            return compute_dtype == DataType::FP16 || compute_dtype == DataType::BF16;
        default:
            return false;
    }
}

static std::optional<DataType> getVectorizedStageComputeDTypeImpl(const PhysicalExpression& expr, DataType storage_dtype) {
    std::optional<DataType> maybe_compute_dtype = std::nullopt;

    for (const ExprNode& node : expr.nodes) {
        if (node.op == ExprOp::INPUT || node.op == ExprOp::SCALAR_FP || node.op == ExprOp::RUNTIME_SCALAR ||
            node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            continue;
        }

        const DataType compute_dtype = requireNodeComputeDType(node);
        if (!supportsVectorizedComputeDTypeForStorage(storage_dtype, compute_dtype)) {
            return std::nullopt;
        }
        if (!maybe_compute_dtype.has_value()) {
            maybe_compute_dtype = compute_dtype;
        } else if (maybe_compute_dtype.value() != compute_dtype) {
            return std::nullopt;
        }
    }

    if (!maybe_compute_dtype.has_value()) {
        const DataType default_compute_dtype = defaultComputeDType(storage_dtype);
        if (!supportsVectorizedComputeDTypeForStorage(storage_dtype, default_compute_dtype)) {
            return std::nullopt;
        }
        return default_compute_dtype;
    }

    return maybe_compute_dtype;
}

static std::optional<DataType> getVectorizedStageStorageDTypeImpl(
    const PhysicalExpression& expr,
    const std::vector<DataType>& input_dtypes,
    const std::vector<DataType>& output_dtypes,
    const std::unordered_set<uint32_t>* ignored_tensor_input_slots = nullptr) {
    if (input_dtypes.empty() || output_dtypes.empty()) {
        return std::nullopt;
    }
    if (expressionHasIndexAwareOps(expr)) {
        return std::nullopt;
    }

    std::optional<DataType> maybe_stage_dtype = std::nullopt;
    for (uint32_t slot = 0; slot < expr.inputs.size(); ++slot) {
        if (expr.inputs[slot].kind != NamedInput::Kind::Tensor ||
            (ignored_tensor_input_slots != nullptr && ignored_tensor_input_slots->contains(slot))) {
            continue;
        }

        const DataType dtype = input_dtypes.at(slot);
        if (!maybe_stage_dtype.has_value()) {
            maybe_stage_dtype = dtype;
        } else if (maybe_stage_dtype.value() != dtype) {
            return std::nullopt;
        }
    }

    if (!maybe_stage_dtype.has_value()) {
        return std::nullopt;
    }

    const DataType stage_dtype = maybe_stage_dtype.value();
    if (stage_dtype != DataType::FP16 && stage_dtype != DataType::BF16 && stage_dtype != DataType::FP8_E4M3 &&
        stage_dtype != DataType::FP8_E5M2) {
        return std::nullopt;
    }

    for (DataType dtype : output_dtypes) {
        if (dtype != stage_dtype) {
            return std::nullopt;
        }
    }

    if (!getVectorizedStageComputeDTypeImpl(expr, stage_dtype).has_value()) {
        return std::nullopt;
    }

    for (const ExprNode& node : expr.nodes) {
        if (node.op == ExprOp::SCALAR_FP || node.op == ExprOp::RUNTIME_SCALAR || node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            continue;
        }

        if (node.op == ExprOp::INPUT) {
            if (ignored_tensor_input_slots != nullptr && ignored_tensor_input_slots->contains(node.input_slot)) {
                continue;
            }
            if (requireNodeInputTensorDType(node) != stage_dtype) {
                return std::nullopt;
            }
        }

        if (requireNodeOutputDType(node) != stage_dtype) {
            return std::nullopt;
        }

    }

    return stage_dtype;
}

struct RaggedValuewiseExtentEmissionSpec {
    uint32_t partition_input_slot = UINT32_MAX;
    uint32_t partition_node_idx = UINT32_MAX;
    DataType partition_dtype = DataType::UINT32;
    RaggedRuntimeExtentSource source = RaggedRuntimeExtentSource::DEVICE_OFFSETS;
    uint64_t batch_size = 0;
    uint64_t max_active_values = 0;
    uint64_t elements_per_value = 1;
};

static bool nodeDependsOnRaggedExtent(const PhysicalExpression& expr,
                                      uint32_t node_idx,
                                      std::vector<int8_t>& memo) {
    if (node_idx >= expr.nodes.size()) {
        throw runtime_error("ragged valuewise extent dependency walk encountered an invalid node index.");
    }
    if (memo[node_idx] >= 0) {
        return memo[node_idx] != 0;
    }

    const ExprNode& node = expr.nodes[node_idx];
    if (node.op == ExprOp::RAGGED_VALUEWISE_EXTENT) {
        memo[node_idx] = 1;
        return true;
    }

    auto parent_depends = [&](uint32_t parent_idx) {
        return parent_idx != UINT32_MAX && nodeDependsOnRaggedExtent(expr, parent_idx, memo);
    };
    bool depends = parent_depends(node.lhs) || parent_depends(node.rhs) || parent_depends(node.aux) ||
                   parent_depends(node.alpha_node) || parent_depends(node.beta_node) ||
                   parent_depends(node.matmul_epilogue_aux) || parent_depends(node.rope_effective_sequence_length_node) ||
                   parent_depends(node.rope_position_ids_node) || parent_depends(node.attention_seq_len_q_node) ||
                   parent_depends(node.attention_seq_len_kv_node) || parent_depends(node.attention_ragged_offset_q_node) ||
                   parent_depends(node.attention_ragged_offset_kv_node) || parent_depends(node.attention_page_table_k_node) ||
                   parent_depends(node.attention_page_table_v_node) || parent_depends(node.attention_dropout_seed_node) ||
                   parent_depends(node.attention_dropout_offset_node);
    if (!depends && node.op == ExprOp::CUDA_KERNEL_OUTPUT) {
        for (uint32_t input_node : node.cuda_kernel_input_nodes) {
            depends = parent_depends(input_node);
            if (depends) {
                break;
            }
        }
    }
    memo[node_idx] = depends ? 1 : 0;
    return depends;
}

template <typename StageT>
static void validateRaggedExtentStageIsolation(const StageT& stage,
                                               const RaggedValuewiseExtentEmissionSpec& spec) {
    std::vector<int8_t> memo(stage.expr.nodes.size(), -1);
    for (const CompiledStageOutput& output : stage.outputs) {
        if (!nodeDependsOnRaggedExtent(stage.expr, output.local_node_idx, memo)) {
            throw runtime_error(
                "one fused kernel cannot mix ragged-runtime-extent outputs with ordinary dense outputs; materialize them in separate stages.");
        }
    }

    auto is_offsets_input_node = [&](uint32_t node_idx) {
        if (node_idx == UINT32_MAX || node_idx >= stage.expr.nodes.size()) {
            return false;
        }
        const ExprNode& candidate = stage.expr.nodes[node_idx];
        return candidate.op == ExprOp::INPUT && candidate.input_slot == spec.partition_input_slot;
    };

    for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
        if (is_offsets_input_node(node_idx)) {
            continue;
        }
        const ExprNode& node = stage.expr.nodes[node_idx];
        const bool marker_uses_offsets = node.op == ExprOp::RAGGED_VALUEWISE_EXTENT && is_offsets_input_node(node.rhs);
        if (marker_uses_offsets) {
            continue;
        }
        auto references_offsets = [&](uint32_t parent_idx) { return is_offsets_input_node(parent_idx); };
        bool offsets_are_value_data = references_offsets(node.lhs) || references_offsets(node.rhs) || references_offsets(node.aux) ||
                                      references_offsets(node.alpha_node) || references_offsets(node.beta_node) ||
                                      references_offsets(node.matmul_epilogue_aux) ||
                                      references_offsets(node.rope_effective_sequence_length_node) ||
                                      references_offsets(node.rope_position_ids_node) || references_offsets(node.attention_seq_len_q_node) ||
                                      references_offsets(node.attention_seq_len_kv_node) || references_offsets(node.attention_ragged_offset_q_node) ||
                                      references_offsets(node.attention_ragged_offset_kv_node) || references_offsets(node.attention_page_table_k_node) ||
                                      references_offsets(node.attention_page_table_v_node) || references_offsets(node.attention_dropout_seed_node) ||
                                      references_offsets(node.attention_dropout_offset_node);
        if (!offsets_are_value_data && node.op == ExprOp::CUDA_KERNEL_OUTPUT) {
            offsets_are_value_data = std::any_of(node.cuda_kernel_input_nodes.begin(),
                                                 node.cuda_kernel_input_nodes.end(),
                                                 references_offsets);
        }
        if (offsets_are_value_data) {
            throw runtime_error(
                "ragged runtime extent offsets are launch metadata and cannot also participate in fused value computation.");
        }
    }
}

static std::string raggedActiveValueCountLoadIndex(const RaggedValuewiseExtentEmissionSpec& extent) {
    switch (extent.source) {
        case RaggedRuntimeExtentSource::DEVICE_OFFSETS:
            return std::to_string(extent.batch_size) + "ULL";
        case RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT:
            return "0ULL";
        case RaggedRuntimeExtentSource::HOST_EXTENT:
            break;
    }
    throw runtime_error("ragged valuewise extent has no device active-count source.");
}

static bool hasAnyRaggedValuewiseExtentMarker(const PhysicalExpression& expr) {
    return std::any_of(expr.nodes.begin(), expr.nodes.end(), [](const ExprNode& node) {
        return node.op == ExprOp::RAGGED_VALUEWISE_EXTENT;
    });
}

static std::unordered_set<uint32_t> raggedValuewiseExtentPartitionInputSlots(const PhysicalExpression& expr) {
    std::unordered_set<uint32_t> slots;
    for (const ExprNode& node : expr.nodes) {
        if (node.op != ExprOp::RAGGED_VALUEWISE_EXTENT || node.rhs >= expr.nodes.size()) {
            continue;
        }
        const ExprNode& partition = expr.nodes[node.rhs];
        if (partition.op == ExprOp::INPUT && partition.input_slot < expr.inputs.size()) {
            slots.insert(partition.input_slot);
        }
    }
    return slots;
}

static std::optional<RaggedValuewiseExtentEmissionSpec> findRaggedValuewiseExtentSpec(const PhysicalExpression& expr) {
    struct MarkerSpec {
        uint32_t node_idx = UINT32_MAX;
        RaggedValuewiseExtentEmissionSpec spec;
    };

    std::vector<MarkerSpec> markers;
    for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
        const ExprNode& node = expr.nodes[node_idx];
        if (node.op != ExprOp::RAGGED_VALUEWISE_EXTENT ||
            node.ragged_runtime_extent_source == RaggedRuntimeExtentSource::HOST_EXTENT) {
            continue;
        }
        if (node.lhs >= expr.nodes.size() || node.rhs >= expr.nodes.size()) {
            throw runtime_error("ragged valuewise extent node has invalid values or offsets input.");
        }
        const ExprNode& partition_node = expr.nodes[node.rhs];
        if (partition_node.op != ExprOp::INPUT || partition_node.input_slot >= expr.inputs.size() ||
            expr.inputs[partition_node.input_slot].kind != NamedInput::Kind::Tensor) {
            throw runtime_error("ragged valuewise extent partition source must be a direct tensor input.");
        }
        const DataType partition_dtype = requireNodeInputTensorDType(partition_node);
        if (!isCanonicalRowPartitionOffsetDataType(partition_dtype)) {
            throw runtime_error(
                "device ragged valuewise extent source must use the canonical UINT32 or UINT64 row-partition dtype.");
        }
        if (node.ragged_runtime_batch_size == 0 || node.ragged_runtime_max_active_values == 0 ||
            node.ragged_runtime_elements_per_value == 0) {
            throw runtime_error("ragged valuewise extent metadata must be non-zero.");
        }

        RaggedValuewiseExtentEmissionSpec candidate;
        candidate.partition_input_slot = partition_node.input_slot;
        candidate.partition_node_idx = node.rhs;
        candidate.partition_dtype = partition_dtype;
        candidate.source = node.ragged_runtime_extent_source;
        candidate.batch_size = node.ragged_runtime_batch_size;
        candidate.max_active_values = node.ragged_runtime_max_active_values;
        candidate.elements_per_value = node.ragged_runtime_elements_per_value;
        markers.push_back(MarkerSpec{node_idx, candidate});
    }

    if (markers.empty()) {
        return std::nullopt;
    }

    const RaggedValuewiseExtentEmissionSpec partition = markers.front().spec;
    uint64_t launch_elements_per_value = partition.elements_per_value;
    for (const MarkerSpec& marker : markers) {
        const RaggedValuewiseExtentEmissionSpec& candidate = marker.spec;
        if (partition.partition_input_slot != candidate.partition_input_slot || partition.partition_dtype != candidate.partition_dtype ||
            partition.source != candidate.source || partition.batch_size != candidate.batch_size ||
            partition.max_active_values != candidate.max_active_values) {
            throw runtime_error("one fused kernel cannot combine different ragged row partitions.");
        }
        launch_elements_per_value = std::max(launch_elements_per_value, candidate.elements_per_value);
    }

    const bool mixed_widths = std::any_of(markers.begin(), markers.end(), [&](const MarkerSpec& marker) {
        return marker.spec.elements_per_value != launch_elements_per_value;
    });
    if (mixed_widths) {
        // Shape-changing ragged views are represented by ordinary STRIDED_VIEW nodes.
        // Their backward scatter expands a narrow per-row gradient back into the wider
        // source-row domain, then RAGGED_VALUEWISE_EXTENT re-wraps the scatter with that
        // source width.  Permit different widths only for that nested backward pattern.
        // Every consumer path from a narrower marker must pass through a scatter and then
        // through a controlling widest marker before reaching a terminal node.  This keeps
        // the historical rejection for unrelated mixed ragged extents.
        std::vector<std::vector<uint32_t>> consumers(expr.nodes.size());
        auto add_consumer = [&](uint32_t parent, uint32_t consumer) {
            if (parent != UINT32_MAX) {
                if (parent >= expr.nodes.size()) {
                    throw runtime_error("ragged extent consumer graph references an out-of-range parent node.");
                }
                consumers[parent].push_back(consumer);
            }
        };
        for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
            const ExprNode& node = expr.nodes[node_idx];
            add_consumer(node.lhs, node_idx);
            add_consumer(node.rhs, node_idx);
            add_consumer(node.aux, node_idx);
            add_consumer(node.alpha_node, node_idx);
            add_consumer(node.beta_node, node_idx);
            add_consumer(node.matmul_epilogue_aux, node_idx);
            if (node.op == ExprOp::CUDA_KERNEL_OUTPUT) {
                for (uint32_t parent : node.cuda_kernel_input_nodes) add_consumer(parent, node_idx);
            }
        }

        auto marker_width = [&](uint32_t node_idx) -> std::optional<uint64_t> {
            const ExprNode& node = expr.nodes.at(node_idx);
            if (node.op != ExprOp::RAGGED_VALUEWISE_EXTENT) return std::nullopt;
            return node.ragged_runtime_elements_per_value;
        };

        for (const MarkerSpec& marker : markers) {
            if (marker.spec.elements_per_value == launch_elements_per_value) {
                continue;
            }

            std::unordered_map<uint64_t, bool> memo;
            std::function<bool(uint32_t, bool, bool)> all_paths_are_shape_expansion =
                [&](uint32_t node_idx, bool saw_scatter, bool saw_controlling_marker) -> bool {
                const uint64_t state_key = (static_cast<uint64_t>(node_idx) << 2U) |
                                           (static_cast<uint64_t>(saw_scatter) << 1U) |
                                           static_cast<uint64_t>(saw_controlling_marker);
                auto memo_it = memo.find(state_key);
                if (memo_it != memo.end()) return memo_it->second;

                const ExprNode& node = expr.nodes.at(node_idx);
                saw_scatter = saw_scatter || node.op == ExprOp::STRIDED_VIEW_BACKWARD;
                const std::optional<uint64_t> width = marker_width(node_idx);
                saw_controlling_marker = saw_controlling_marker ||
                                         (saw_scatter && width.has_value() && width.value() == launch_elements_per_value);

                if (consumers[node_idx].empty()) {
                    const bool valid_terminal = saw_scatter && saw_controlling_marker;
                    memo.emplace(state_key, valid_terminal);
                    return valid_terminal;
                }
                for (uint32_t consumer : consumers[node_idx]) {
                    if (!all_paths_are_shape_expansion(consumer, saw_scatter, saw_controlling_marker)) {
                        memo.emplace(state_key, false);
                        return false;
                    }
                }
                memo.emplace(state_key, true);
                return true;
            };

            if (!all_paths_are_shape_expansion(marker.node_idx, false, false)) {
                throw runtime_error("one fused kernel cannot combine unrelated ragged runtime widths.");
            }
        }
    }

    RaggedValuewiseExtentEmissionSpec result = partition;
    result.elements_per_value = launch_elements_per_value;
    return result;
}

static uint64_t specializedRaggedGroupElementsPerValue(const RaggedValuewiseExtentEmissionSpec& extent,
                                                       const SpecializedBroadcastGroup& group) {
    if (extent.max_active_values == 0) {
        throw runtime_error("ragged specialized broadcast requires non-zero max_active_values.");
    }
    if ((group.numel % extent.max_active_values) != 0ULL) {
        throw runtime_error(
            "ragged specialized broadcast output capacity must be an integer number of elements per packed value.");
    }
    const uint64_t elements_per_value = group.numel / extent.max_active_values;
    if (elements_per_value == 0) {
        throw runtime_error("ragged specialized broadcast output elements per packed value must be non-zero.");
    }
    return elements_per_value;
}

static std::unordered_set<uint32_t> raggedValuewiseExtentPartitionInputSlots(const PhysicalExpression& expr);

std::optional<DataType> CudaSourceEmitter::getVectorizedStageStorageDType(const PhysicalExecutionStage& stage) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel) {
        return std::nullopt;
    }
    // Device-ragged stages use dedicated packet eligibility.  The ordinary
    // homogeneous check cannot treat the structural partition tensor as value
    // data, and specialized broadcast must additionally prove packet-aligned
    // row boundaries before widening.
    if (hasAnyRaggedValuewiseExtentMarker(stage.expr)) {
        return std::nullopt;
    }
    return getVectorizedStageStorageDTypeImpl(stage.expr, collectInputSlotDTypes(stage.expr), collectOutputDTypes(stage));
}

std::optional<DataType> CudaSourceEmitter::getVectorizedStageStorageDType(const CompiledExecutionStage& stage) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        return std::nullopt;
    }
    if (hasAnyRaggedValuewiseExtentMarker(stage.expr)) {
        return std::nullopt;
    }
    return getVectorizedStageStorageDTypeImpl(stage.expr, collectInputSlotDTypes(stage.expr), collectOutputDTypes(stage));
}

static std::optional<DataType> getRaggedFlatVectorizedStageStorageDType(const PhysicalExecutionStage& stage) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel ||
        !findRaggedValuewiseExtentSpec(stage.expr).has_value() ||
        stageHasTransposedMaterializedOutput(stage.outputs) || expressionHasIndexAwareOps(stage.expr)) {
        return std::nullopt;
    }
    const std::unordered_set<uint32_t> structural_slots =
        raggedValuewiseExtentPartitionInputSlots(stage.expr);
    return getVectorizedStageStorageDTypeImpl(
        stage.expr, collectInputSlotDTypes(stage.expr), collectOutputDTypes(stage), &structural_slots);
}


static uint64_t specializedBroadcastInnermostInputStride(const SpecializedBroadcastGroup& group, size_t used_input_index) {
    if (used_input_index >= group.used_input_slots.size()) {
        throw runtime_error("specialized broadcast used-input index out of range.");
    }
    for (const SpecializedBroadcastAxis& axis : group.active_axes) {
        if (axis.output_stride == 1ULL) {
            if (used_input_index >= axis.input_strides.size()) {
                throw runtime_error("specialized broadcast input stride metadata is incomplete.");
            }
            return axis.input_strides[used_input_index];
        }
    }
    return 0ULL;
}

// Packet ownership for the per-lane ("wide-scalar") packet kernels follows physical
// storage, not arithmetic precision: every tensor moves at most one 16-byte packet
// per thread. Any FP32 storage participant -> 4 scalars; BF16 or FP16 storage -> 8;
// FP8-only storage -> 16. Mixed BF16+FP16 without FP32 remains uncensused (1).
struct WideScalarStorageFamily {
    bool fp8 = false;
    bool bf16 = false;
    bool fp16 = false;
    bool fp32 = false;

    bool accept(DataType dtype) {
        switch (dtype) {
            case DataType::FP8_E4M3:
            case DataType::FP8_E5M2:
                fp8 = true;
                return true;
            case DataType::BF16:
                bf16 = true;
                return true;
            case DataType::FP16:
                fp16 = true;
                return true;
            case DataType::FP32:
                fp32 = true;
                return true;
            default:
                return false;
        }
    }

    uint32_t packetScalars() const {
        if (fp32) return 4u;
        if (bf16 && fp16) return 1u;
        if (bf16 || fp16) return 8u;
        if (fp8) return 16u;
        return 1u;
    }
};

static std::optional<DataType> getWideSpecializedBroadcastVectorizedStageStorageDType(
    const CompiledExecutionStage& stage,
    const std::vector<SpecializedBroadcastGroup>& groups) {
    const bool has_index_mapped_ops = std::any_of(stage.expr.nodes.begin(), stage.expr.nodes.end(), [](const ExprNode& node) {
        return node.op == ExprOp::TAKE_ALONG_AXIS || node.op == ExprOp::STRIDED_VIEW;
    });
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel || groups.empty() ||
        stageHasTransposedMaterializedOutput(stage.outputs) || expressionHasIndexAwareOps(stage.expr) || has_index_mapped_ops) {
        return std::nullopt;
    }

    // Dense stages use the same packet family: a dense group is a ragged group
    // whose active extent is its full capacity.
    const std::optional<RaggedValuewiseExtentEmissionSpec> ragged_extent = findRaggedValuewiseExtentSpec(stage.expr);

    const std::unordered_set<uint32_t> structural_slots = raggedValuewiseExtentPartitionInputSlots(stage.expr);
    const std::optional<DataType> storage_dtype = getVectorizedStageStorageDTypeImpl(
        stage.expr, collectInputSlotDTypes(stage.expr), collectOutputDTypes(stage), &structural_slots);
    if (!storage_dtype.has_value() ||
        (storage_dtype.value() != DataType::BF16 && storage_dtype.value() != DataType::FP16)) {
        return std::nullopt;
    }

    constexpr uint64_t packet_scalars = 8ULL;
    for (const SpecializedBroadcastGroup& group : groups) {
        if (group.output_dims.empty() || group.used_input_slots.size() != group.used_input_load_kinds.size()) {
            return std::nullopt;
        }
        if ((group.output_dims.back() % packet_scalars) != 0ULL || (group.numel % packet_scalars) != 0ULL) {
            return std::nullopt;
        }
        if (ragged_extent.has_value() &&
            (specializedRaggedGroupElementsPerValue(ragged_extent.value(), group) % packet_scalars) != 0ULL) {
            return std::nullopt;
        }

        for (size_t used_i = 0; used_i < group.used_input_slots.size(); ++used_i) {
            if (structural_slots.contains(group.used_input_slots[used_i])) {
                return std::nullopt;
            }
            if (group.used_input_load_kinds[used_i] == SpecializedInputLoadKind::NativeVector &&
                specializedBroadcastInnermostInputStride(group, used_i) != 1ULL) {
                return std::nullopt;
            }
        }
    }

    return storage_dtype;
}

static uint32_t wideScalarSpecializedBroadcastElementsPerThread(
    const CompiledExecutionStage& stage,
    const std::vector<SpecializedBroadcastGroup>& groups) {
    const bool has_index_mapped_ops = std::any_of(stage.expr.nodes.begin(), stage.expr.nodes.end(), [](const ExprNode& node) {
        return node.op == ExprOp::TAKE_ALONG_AXIS || node.op == ExprOp::STRIDED_VIEW;
    });
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel || groups.empty() ||
        stageHasTransposedMaterializedOutput(stage.outputs) || expressionHasIndexAwareOps(stage.expr) || has_index_mapped_ops) {
        return 1u;
    }

    // Dense stages use the same packet family as ragged ones (active extent =
    // full capacity).
    const std::optional<RaggedValuewiseExtentEmissionSpec> ragged_extent = findRaggedValuewiseExtentSpec(stage.expr);

    const std::unordered_set<uint32_t> structural_slots = raggedValuewiseExtentPartitionInputSlots(stage.expr);
    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const std::vector<DataType> output_dtypes = collectOutputDTypes(stage);
    WideScalarStorageFamily family;

    // First classify the physical value-storage family.  Compute dtype is
    // deliberately absent from this decision: the scalar-per-lane wide emitter
    // performs any low-precision -> FP32 promotion after the coalesced packet load.
    for (const SpecializedBroadcastGroup& group : groups) {
        if (group.output_dims.empty() || group.used_input_slots.size() != group.used_input_load_kinds.size()) {
            return 1u;
        }

        for (uint32_t slot : group.used_input_slots) {
            if (slot >= stage.expr.inputs.size() || structural_slots.contains(slot) ||
                stage.expr.inputs[slot].kind != NamedInput::Kind::Tensor || !family.accept(input_dtypes.at(slot))) {
                return 1u;
            }
        }
        for (uint32_t out_idx : group.output_indices) {
            if (out_idx >= output_dtypes.size() || !family.accept(output_dtypes[out_idx])) {
                return 1u;
            }
        }
    }

    const uint32_t full_packet_scalars = family.packetScalars();
    if (full_packet_scalars <= 1u) {
        return 1u;
    }

    // Then prove the broadcast mapping can preserve that packet ownership: the
    // packet may not cross a row/cycle boundary, native operands must be
    // contiguous, and scalar broadcasts must truly be invariant across it.
    // Row widths that are not a full-packet multiple take the widest smaller
    // packet that divides them (FUSED-GATE only ordains full packets).
    const auto packet_fits = [&](uint32_t packet_scalars) {
        for (const SpecializedBroadcastGroup& group : groups) {
            if ((group.output_dims.back() % packet_scalars) != 0ULL || (group.numel % packet_scalars) != 0ULL ||
                (ragged_extent.has_value() &&
                 (specializedRaggedGroupElementsPerValue(ragged_extent.value(), group) % packet_scalars) != 0ULL)) {
                return false;
            }

            for (size_t used_i = 0; used_i < group.used_input_slots.size(); ++used_i) {
                const uint64_t inner_stride = specializedBroadcastInnermostInputStride(group, used_i);
                if (group.used_input_load_kinds[used_i] == SpecializedInputLoadKind::NativeVector) {
                    if (inner_stride != 1ULL) {
                        return false;
                    }
                } else if (inner_stride != 0ULL) {
                    return false;
                }
            }
        }
        return true;
    };
    for (uint32_t packet_scalars = full_packet_scalars; packet_scalars > 1u; packet_scalars /= 2u) {
        if (packet_fits(packet_scalars)) {
            return packet_scalars;
        }
    }
    return 1u;
}

static std::optional<DataType> getSingleTensorInputStorageDType(const PhysicalExpression& expr, const std::vector<DataType>& input_dtypes) {
    std::optional<DataType> maybe_tensor_dtype = std::nullopt;

    for (uint32_t slot = 0; slot < expr.inputs.size(); ++slot) {
        if (expr.inputs[slot].kind != NamedInput::Kind::Tensor) {
            continue;
        }

        const DataType dtype = input_dtypes.at(slot);
        if (!maybe_tensor_dtype.has_value()) {
            maybe_tensor_dtype = dtype;
        } else if (maybe_tensor_dtype.value() != dtype) {
            return std::nullopt;
        }
    }

    return maybe_tensor_dtype;
}

// Ops emitFloat2NodeDefinitionsForSuffix can evaluate. The float2 transpose paths
// must not be selected for an expression containing anything else (for example
// WHERE or a comparison, which also compute in FP32 by default).
static bool isFloat2PathOp(ExprOp op) {
    switch (op) {
        case ExprOp::INPUT:
        case ExprOp::RUNTIME_SCALAR:
        case ExprOp::TENSOR_RUNTIME_SCALAR:
        case ExprOp::SCALAR_FP:
        case ExprOp::FILL:
        case ExprOp::ADD:
        case ExprOp::SUB:
        case ExprOp::MUL:
        case ExprOp::DIV:
        case ExprOp::NEG:
        case ExprOp::ABS:
        case ExprOp::CEIL:
        case ExprOp::FLOOR:
        case ExprOp::ROUND:
        case ExprOp::TRUNC:
        case ExprOp::SIN:
        case ExprOp::COS:
        case ExprOp::TAN:
        case ExprOp::ASIN:
        case ExprOp::ACOS:
        case ExprOp::ATAN:
        case ExprOp::SINH:
        case ExprOp::COSH:
        case ExprOp::ASINH:
        case ExprOp::ACOSH:
        case ExprOp::ATANH:
        case ExprOp::ERF:
        case ExprOp::ERFC:
        case ExprOp::ERFCX:
        case ExprOp::ERFINV:
        case ExprOp::ERFCINV:
        case ExprOp::TGAMMA:
        case ExprOp::LGAMMA:
        case ExprOp::DIGAMMA:
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
        case ExprOp::RESHAPE:
        case ExprOp::UNSQUEEZE:
        case ExprOp::SQUEEZE:
        case ExprOp::POW:
        case ExprOp::MIN:
        case ExprOp::MAX:
        case ExprOp::MIN_GRAD_LEFT:
        case ExprOp::MIN_GRAD_RIGHT:
        case ExprOp::MAX_GRAD_LEFT:
        case ExprOp::MAX_GRAD_RIGHT:
            return true;
        default:
            return false;
    }
}

static bool expressionUsesOnlyFloat2PathOps(const PhysicalExpression& expr) {
    return std::all_of(expr.nodes.begin(), expr.nodes.end(), [](const ExprNode& node) { return isFloat2PathOp(node.op); });
}

static bool supportsMixedTwoByteFloat2TransposedVectorization(const PhysicalExpression& expr, DataType input_dtype, DataType output_dtype) {
    if (expressionHasIndexAwareOps(expr) || !expressionUsesOnlyFloat2PathOps(expr)) {
        return false;
    }

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

static std::optional<DataType> mixedFp8TransposedVectorComputeDType(const PhysicalExpression& expr,
                                                                    DataType input_dtype,
                                                                    DataType output_dtype) {
    if (expressionHasIndexAwareOps(expr)) {
        return std::nullopt;
    }

    if (input_dtype == output_dtype || !isFp8DType(input_dtype) || !isFp8DType(output_dtype)) {
        return std::nullopt;
    }

    std::optional<DataType> maybe_compute_dtype = std::nullopt;
    for (const ExprNode& node : expr.nodes) {
        switch (node.op) {
            case ExprOp::INPUT:
                if (requireNodeInputTensorDType(node) != input_dtype) {
                    return std::nullopt;
                }
                break;
            case ExprOp::SCALAR_FP:
            case ExprOp::FILL:
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
                break;
            default: {
                const DataType compute_dtype = requireNodeComputeDType(node);
                if (compute_dtype != DataType::FP16 && compute_dtype != DataType::BF16) {
                    return std::nullopt;
                }
                if (!maybe_compute_dtype.has_value()) {
                    maybe_compute_dtype = compute_dtype;
                } else if (maybe_compute_dtype.value() != compute_dtype) {
                    return std::nullopt;
                }
                break;
            }
        }
    }

    return maybe_compute_dtype.value_or(defaultComputeDType(output_dtype));
}

static bool supportsFloat2TransposedVectorization(const PhysicalExpression& expr, DataType input_dtype, DataType output_dtype) {
    if (expressionHasIndexAwareOps(expr) || !expressionUsesOnlyFloat2PathOps(expr)) {
        return false;
    }

    if (transposePackScalars(output_dtype) <= 1) {
        return false;
    }

    // This path computes the fused producer in fp32 pairs before packing the
    // low-precision transposed output: cross-width/cross-family storage, and
    // homogeneous low-precision storage computing in FP32 (the default policy).
    // Homogeneous storage with explicit BF16/FP16 compute uses the packed
    // half2/bfloat162 path instead.
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
    if (expressionHasIndexAwareOps(expr)) {
        return false;
    }

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

static bool shouldUseDecoupledLineVectorizedTranspose(const std::optional<DataType>& maybe_tensor_input_dtype, DataType output_dtype) {
    if (!maybe_tensor_input_dtype.has_value()) {
        return false;
    }

    const uint32_t input_pack_scalars = transposePackScalars(maybe_tensor_input_dtype.value());
    const uint32_t output_pack_scalars = transposePackScalars(output_dtype);
    return (input_pack_scalars != output_pack_scalars) && (std::max(input_pack_scalars, output_pack_scalars) > 1U);
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
        case DataType::INT32:
        case DataType::UINT32:
            return 4;
        case DataType::INT64:
        case DataType::UINT64:
            return 8;
        case DataType::BOOLEAN:
        case DataType::UINT8:
        case DataType::INT8:
            return 1;
        case DataType::UINT16:
        case DataType::INT16:
            return 2;
        default:
            throw runtime_error("Unsupported dtype in dataTypeStorageBytes: " + TensorDescriptor::getElementTypeName(dtype));
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

static uint32_t raggedWideScalarFlatElementsPerThread(const PhysicalExecutionStage& stage) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel ||
        !findRaggedValuewiseExtentSpec(stage.expr).has_value() ||
        stageHasTransposedMaterializedOutput(stage.outputs) || expressionHasIndexAwareOps(stage.expr)) {
        return 1u;
    }

    const std::unordered_set<uint32_t> structural_slots =
        raggedValuewiseExtentPartitionInputSlots(stage.expr);
    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const std::vector<DataType> output_dtypes = collectOutputDTypes(stage);

    WideScalarStorageFamily family;
    for (uint32_t slot = 0; slot < stage.expr.inputs.size(); ++slot) {
        if (structural_slots.contains(slot)) {
            continue;
        }
        // Runtime scalars are read once per thread and never packetized, so they do
        // not take part in packet ownership.
        if (stage.expr.inputs[slot].kind != NamedInput::Kind::Tensor) {
            continue;
        }
        if (!family.accept(input_dtypes.at(slot))) {
            return 1u;
        }
    }
    for (DataType dtype : output_dtypes) {
        if (!family.accept(dtype)) {
            return 1u;
        }
    }

    // Homogeneous BF16/FP16 storage keeps the proven eight-scalar / 16-byte packet
    // even when the expression computes in FP32 (the default for low-precision
    // storage): the emitter unpacks each lane after the coalesced load, evaluates it
    // in the resolved compute dtype, then repacks. FP8-only storage owns sixteen
    // scalars; any FP32 storage keeps one float4 per FP32 tensor per thread.
    return family.packetScalars();
}


static bool expressionHasLogicalTransposeOp(const PhysicalExpression& expr);
static bool expressionHasTakeAlongAxisOp(const PhysicalExpression& expr);
static bool expressionHasStridedViewOp(const PhysicalExpression& expr);

static bool fusedGateOrdinaryFloatDType(DataType dtype) {
    return dtype == DataType::BF16 || dtype == DataType::FP16 || dtype == DataType::FP32;
}

static bool fusedGateAllSameDType(const std::vector<DataType>& dtypes, DataType dtype) {
    return !dtypes.empty() && std::all_of(dtypes.begin(), dtypes.end(), [dtype](DataType candidate) {
        return candidate == dtype;
    });
}

std::optional<uint32_t> CudaSourceEmitter::fusedGateRequiredFlatElementsPerThread(
    const PhysicalExecutionStage& stage) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel ||
        stageHasTransposedMaterializedOutput(stage.outputs) || expressionHasIndexAwareOps(stage.expr) ||
        expressionHasStridedViewOp(stage.expr) || expressionHasTakeAlongAxisOp(stage.expr) ||
        expressionHasLogicalTransposeOp(stage.expr)) {
        return std::nullopt;
    }

    const bool is_ragged = hasAnyRaggedValuewiseExtentMarker(stage.expr);
    const std::unordered_set<uint32_t> structural_slots =
        is_ragged ? raggedValuewiseExtentPartitionInputSlots(stage.expr) : std::unordered_set<uint32_t>{};
    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const std::vector<DataType> output_dtypes = collectOutputDTypes(stage);

    std::vector<DataType> value_dtypes;
    value_dtypes.reserve(stage.expr.inputs.size() + output_dtypes.size());
    for (uint32_t slot = 0; slot < stage.expr.inputs.size(); ++slot) {
        if (structural_slots.contains(slot)) {
            continue;
        }
        // FUSED-GATE intentionally covers the ordinary tensor-streaming families
        // established by the census. Runtime scalar and other non-tensor inputs are
        // left outside the invariant until separately benchmarked.
        if (stage.expr.inputs[slot].kind != NamedInput::Kind::Tensor) {
            return std::nullopt;
        }
        const DataType dtype = input_dtypes.at(slot);
        if (!fusedGateOrdinaryFloatDType(dtype)) {
            return std::nullopt;
        }
        value_dtypes.push_back(dtype);
    }
    for (DataType dtype : output_dtypes) {
        if (!fusedGateOrdinaryFloatDType(dtype)) {
            return std::nullopt;
        }
        value_dtypes.push_back(dtype);
    }
    if (value_dtypes.empty()) {
        return std::nullopt;
    }

    // Homogeneous BF16/FP16 flat streaming is ordained at one 16-byte packet
    // (eight logical scalars) per thread for both dense and packed-ragged stages.
    if (fusedGateAllSameDType(value_dtypes, DataType::BF16) ||
        fusedGateAllSameDType(value_dtypes, DataType::FP16)) {
        return 8u;
    }

    // Homogeneous FP32 and every benchmarked BF16/FP16<->FP32 mixed flat family
    // use the proven dense-mixed ownership rule: four logical values per thread,
    // preserving one coalesced float4 transaction for every FP32 tensor.
    if (std::any_of(value_dtypes.begin(), value_dtypes.end(), [](DataType dtype) {
            return dtype == DataType::FP32;
        })) {
        return 4u;
    }

    // Mixed BF16/FP16 without FP32 is intentionally not gated yet; it was not part
    // of the benchmark-supported sequence and should not be constrained by inference.
    return std::nullopt;
}

uint32_t CudaSourceEmitter::flatElementsPerThread(const PhysicalExecutionStage& stage) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel) {
        return 1;
    }
    if (stageHasTransposedMaterializedOutput(stage.outputs)) {
        return 1;
    }
    if (expressionHasIndexAwareOps(stage.expr)) {
        return 1;
    }

    const std::optional<DataType> vectorized_dtype = hasAnyRaggedValuewiseExtentMarker(stage.expr)
                                                        ? getRaggedFlatVectorizedStageStorageDType(stage)
                                                        : getVectorizedStageStorageDType(stage);
    if (vectorized_dtype.has_value()) {
        switch (vectorized_dtype.value()) {
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

    // FUSED-RAGGED-FLAT handles packed low-precision arithmetic above.  The
    // wide-scalar fallback is storage-driven: homogeneous BF16/FP16 storage uses
    // eight logical values/thread even when compute is FP32, while any FP32
    // storage uses the proven four-value mixed geometry.
    if (hasAnyRaggedValuewiseExtentMarker(stage.expr)) {
        return raggedWideScalarFlatElementsPerThread(stage);
    }

    // Only packetized tensors decide the packet width; runtime scalars (for example
    // an FP32 learning rate) are read once per thread.
    const std::vector<DataType> all_input_dtypes = collectInputSlotDTypes(stage.expr);
    std::vector<DataType> tensor_input_dtypes;
    for (uint32_t slot = 0; slot < stage.expr.inputs.size(); ++slot) {
        if (stage.expr.inputs[slot].kind == NamedInput::Kind::Tensor) {
            tensor_input_dtypes.push_back(all_input_dtypes.at(slot));
        }
    }
    return flatScalarElementsPerThreadImpl(tensor_input_dtypes, collectOutputDTypes(stage));
}

uint32_t CudaSourceEmitter::tiledTransposePackScalars(const PhysicalExecutionStage& stage) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel || !stageHasTransposedMaterializedOutput(stage.outputs)) {
        return 1;
    }

    const CompiledStageOutput& output = requireSingleTransposedMaterializedOutput(stage);
    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const DataType output_dtype = requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]);
    const std::optional<DataType> maybe_tensor_input_dtype = getSingleTensorInputStorageDType(stage.expr, input_dtypes);
    if (shouldUseDecoupledLineVectorizedTranspose(maybe_tensor_input_dtype, output_dtype)) {
        return 1;
    }
    return transposePackScalars(output_dtype);
}

uint32_t CudaSourceEmitter::tiledTransposePackScalars(const CompiledExecutionStage& stage) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel || !stageHasTransposedMaterializedOutput(stage.outputs)) {
        return 1;
    }

    const CompiledStageOutput& output = requireSingleTransposedMaterializedOutput(stage);
    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const DataType output_dtype = requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]);
    const std::optional<DataType> maybe_tensor_input_dtype = getSingleTensorInputStorageDType(stage.expr, input_dtypes);
    if (shouldUseDecoupledLineVectorizedTranspose(maybe_tensor_input_dtype, output_dtype)) {
        return 1;
    }
    return transposePackScalars(output_dtype);
}

static std::string castScalarExpr(const std::string& expr, DataType src_dtype, DataType dst_dtype);

static bool isIntegralScalarCastDType(DataType dtype) {
    switch (dtype) {
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::UINT64:
        case DataType::INT64:
            return true;
        default:
            return false;
    }
}

static std::string castToIntegralScalarExpr(const std::string& expr, DataType src_dtype, DataType dst_dtype) {
    const std::string dst_type = scalarStorageType(dst_dtype);
    if (src_dtype == DataType::BOOLEAN) {
        return "static_cast<" + dst_type + ">((" + expr + ") ? 1 : 0)";
    }
    switch (src_dtype) {
        case DataType::FP32:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return "static_cast<" + dst_type + ">(" + castScalarExpr(expr, src_dtype, DataType::FP32) + ")";
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::UINT64:
        case DataType::INT64:
            return "static_cast<" + dst_type + ">(" + expr + ")";
        default:
            break;
    }
    throw runtime_error("Unsupported scalar integral cast in fused stage emitter from " + TensorDescriptor::getElementTypeName(src_dtype) + " to " +
                        TensorDescriptor::getElementTypeName(dst_dtype));
}

static std::string castScalarExpr(const std::string& expr, DataType src_dtype, DataType dst_dtype) {
    if (src_dtype == dst_dtype) {
        return expr;
    }

    if (isIntegralScalarCastDType(dst_dtype)) {
        return castToIntegralScalarExpr(expr, src_dtype, dst_dtype);
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
                    return "float(" + expr + ")";
                case DataType::BOOLEAN:
                    return "(" + expr + " ? 1.0f : 0.0f)";
                case DataType::UINT8:
                case DataType::INT8:
                case DataType::UINT16:
                case DataType::INT16:
                case DataType::UINT32:
                case DataType::INT32:
                case DataType::UINT64:
                case DataType::INT64:
                    return "float(" + expr + ")";
                default:
                    break;
            }
            break;

        case DataType::FP16:
            switch (src_dtype) {
                case DataType::BOOLEAN:
                    return "half((" + expr + ") ? 1.0f : 0.0f)";
                case DataType::FP32:
                    return "half(" + expr + ")";
                case DataType::FP16:
                    return expr;
                case DataType::BF16:
                    return "half(float(" + expr + "))";
                // FP8 -> FP16 is exact: convert directly (cvt.rn.f16x2.e4m3x2 / e5m2x2 on sm89+)
                // instead of FP8 -> FP16 -> FP32 -> FP16.
                case DataType::FP8_E4M3:
                    return "__half(__nv_cvt_fp8_to_halfraw((" + expr + ").__x, __NV_E4M3))";
                case DataType::FP8_E5M2:
                    return "__half(__nv_cvt_fp8_to_halfraw((" + expr + ").__x, __NV_E5M2))";
                case DataType::UINT8:
                case DataType::INT8:
                case DataType::UINT16:
                case DataType::INT16:
                case DataType::UINT32:
                case DataType::INT32:
                case DataType::UINT64:
                case DataType::INT64:
                    // Exact through FP32: any integer FP32 must round (> 2^24) overflows FP16 anyway.
                    return "half(float(" + expr + "))";
                default:
                    break;
            }
            break;

        case DataType::BF16:
            switch (src_dtype) {
                case DataType::BOOLEAN:
                    return "__nv_bfloat16((" + expr + ") ? 1.0f : 0.0f)";
                case DataType::FP32:
                    return "__nv_bfloat16(" + expr + ")";
                case DataType::FP16:
                    return "__nv_bfloat16(float(" + expr + "))";
                case DataType::BF16:
                    return expr;
                case DataType::FP8_E4M3:
                case DataType::FP8_E5M2:
                    return "__nv_bfloat16(float(" + expr + "))";
                case DataType::UINT8:
                case DataType::INT8:
                case DataType::UINT16:
                case DataType::INT16:
                    return "__nv_bfloat16(float(" + expr + "))";  // exact in FP32
                // Wider integers round once, directly: going through FP32 would double-round above 2^24.
                case DataType::INT32:
                    return "__int2bfloat16_rn(" + expr + ")";
                case DataType::UINT32:
                    return "__uint2bfloat16_rn(" + expr + ")";
                case DataType::INT64:
                    return "__ll2bfloat16_rn(" + expr + ")";
                case DataType::UINT64:
                    return "__ull2bfloat16_rn(" + expr + ")";
                default:
                    break;
            }
            break;

        case DataType::FP8_E4M3:
            switch (src_dtype) {
                case DataType::FP32:
                    return "thor_to_fp8_e4m3_satfinite(" + expr + ")";
                case DataType::FP16:
                    return "thor_half_to_fp8_e4m3_satfinite(" + expr + ")";
                case DataType::BF16:
                    return "thor_to_fp8_e4m3_satfinite(float(" + expr + "))";
                case DataType::FP8_E4M3:
                    return expr;
                case DataType::BOOLEAN:
                    return "thor_to_fp8_e4m3_satfinite((" + expr + ") ? 1.0f : 0.0f)";
                case DataType::UINT8:
                case DataType::INT8:
                case DataType::UINT16:
                case DataType::INT16:
                case DataType::UINT32:
                case DataType::INT32:
                case DataType::UINT64:
                case DataType::INT64:
                    // Exact through FP32: integers FP32 must round (> 2^24) saturate FP8 anyway.
                    return "thor_to_fp8_e4m3_satfinite(float(" + expr + "))";
                case DataType::FP8_E5M2:
                    return "thor_to_fp8_e4m3_satfinite(float(" + expr + "))";
                default:
                    break;
            }
            break;

        case DataType::FP8_E5M2:
            switch (src_dtype) {
                case DataType::FP32:
                    return "thor_to_fp8_e5m2_satfinite(" + expr + ")";
                case DataType::FP16:
                    return "thor_half_to_fp8_e5m2_satfinite(" + expr + ")";
                case DataType::BF16:
                    return "thor_to_fp8_e5m2_satfinite(float(" + expr + "))";
                case DataType::FP8_E4M3:
                    return "thor_to_fp8_e5m2_satfinite(float(" + expr + "))";
                case DataType::FP8_E5M2:
                    return expr;
                case DataType::BOOLEAN:
                    return "thor_to_fp8_e5m2_satfinite((" + expr + ") ? 1.0f : 0.0f)";
                case DataType::UINT8:
                case DataType::INT8:
                case DataType::UINT16:
                case DataType::INT16:
                case DataType::UINT32:
                case DataType::INT32:
                case DataType::UINT64:
                case DataType::INT64:
                    // Exact through FP32: integers FP32 must round (> 2^24) saturate FP8 anyway.
                    return "thor_to_fp8_e5m2_satfinite(float(" + expr + "))";
                default:
                    break;
            }
            break;

        case DataType::BOOLEAN:
            switch (src_dtype) {
                case DataType::BOOLEAN:
                    return expr;
                case DataType::FP32:
                case DataType::FP16:
                case DataType::BF16:
                case DataType::FP8_E4M3:
                case DataType::FP8_E5M2:
                case DataType::UINT8:
                case DataType::INT8:
                case DataType::UINT16:
                case DataType::INT16:
                case DataType::UINT32:
                case DataType::INT32:
                case DataType::UINT64:
                case DataType::INT64:
                    return "((" + castScalarExpr(expr, src_dtype, DataType::FP32) + ") != 0.0f)";
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
        case ExprOp::CAST:
        case ExprOp::BROADCAST_TO:
            return x;

        case ExprOp::NEG:
            if (compute_dtype == DataType::FP32 || compute_dtype == DataType::FP16 || compute_dtype == DataType::BF16) {
                return "(-" + x + ")";
            }
            return castScalarExpr("(-" + x_f + ")", DataType::FP32, compute_dtype);

        case ExprOp::ABS:
            return castScalarExpr("fabsf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::CEIL:
            return castScalarExpr("ceilf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::FLOOR:
            return castScalarExpr("floorf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::ROUND:
            return castScalarExpr("roundf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::TRUNC:
            return castScalarExpr("truncf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::SIN:
            return castScalarExpr("sinf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::COS:
            return castScalarExpr("cosf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::TAN:
            return castScalarExpr("tanf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::ASIN:
            return castScalarExpr("asinf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::ACOS:
            return castScalarExpr("acosf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::ATAN:
            return castScalarExpr("atanf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::SINH:
            return castScalarExpr("sinhf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::COSH:
            return castScalarExpr("coshf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::ASINH:
            return castScalarExpr("asinhf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::ACOSH:
            return castScalarExpr("acoshf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::ATANH:
            return castScalarExpr("atanhf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::ERF:
            return castScalarExpr("erff(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::ERFC:
            return castScalarExpr("erfcf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::ERFCX:
            return castScalarExpr("erfcxf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::ERFINV:
            return castScalarExpr("erfinvf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::ERFCINV:
            return castScalarExpr("erfcinvf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::TGAMMA:
            return castScalarExpr("tgammaf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::LGAMMA:
            return castScalarExpr("lgammaf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::DIGAMMA:
            return castScalarExpr("thor_digammaf(" + x_f + ")", DataType::FP32, compute_dtype);

        case ExprOp::EXP:
            return castScalarExpr("expf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::EXPM1: {
            if (compute_dtype == DataType::FP32) {
                return "expm1f(" + x + ")";
            }
            if (compute_dtype == DataType::FP16) {
                return "half(expm1f(float(" + x + ")))";
            }
            return castScalarExpr("expm1f(" + x_f + ")", DataType::FP32, compute_dtype);
        }
        case ExprOp::EXP2:
            return castScalarExpr("exp2f(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::EXP10:
            return castScalarExpr("exp10f(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::LN:
            return castScalarExpr("logf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::LOG1P: {
            if (compute_dtype == DataType::FP32) {
                return "log1pf(" + x + ")";
            }
            if (compute_dtype == DataType::FP16) {
                return "half(log1pf(float(" + x + ")))";
            }
            return castScalarExpr("log1pf(" + x_f + ")", DataType::FP32, compute_dtype);
        }
        case ExprOp::LOG2:
            return castScalarExpr("log2f(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::LOG10:
            return castScalarExpr("log10f(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::SQRT:
            return castScalarExpr("sqrtf(" + x_f + ")", DataType::FP32, compute_dtype);
        case ExprOp::TANH: {
            if (compute_dtype == DataType::FP32) {
                return "tanhf(" + x + ")";
            }
            if (compute_dtype == DataType::FP16) {
                return "htanh(" + x + ")";
            }
            return castScalarExpr("tanhf(" + x_f + ")", DataType::FP32, compute_dtype);
        }
        case ExprOp::NORMCDF: {
            if (compute_dtype == DataType::FP32) {
                return "normcdff(" + x + ")";
            }
            if (compute_dtype == DataType::FP16) {
                return "half(normcdff(float(" + x + ")))";
            }
            return castScalarExpr("normcdff(" + x_f + ")", DataType::FP32, compute_dtype);
        }
        case ExprOp::LOGICAL_NOT:
            if (compute_dtype == DataType::BOOLEAN) {
                return "(!" + x + ")";
            }
            return "!(" + x_f + " != 0.0f)";

        case ExprOp::RESHAPE:
        case ExprOp::STRIDED_VIEW:
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
            if (compute_dtype == DataType::FP32 || compute_dtype == DataType::FP16 || compute_dtype == DataType::BF16 ||
                compute_dtype == DataType::UINT32 || compute_dtype == DataType::UINT64) {
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

        case ExprOp::EQUAL:
            return "(" + a_f + " == " + b_f + ")";
        case ExprOp::NOT_EQUAL:
            return "(" + a_f + " != " + b_f + ")";
        case ExprOp::LESS:
            return "(" + a_f + " < " + b_f + ")";
        case ExprOp::LESS_EQUAL:
            return "(" + a_f + " <= " + b_f + ")";
        case ExprOp::GREATER:
            return "(" + a_f + " > " + b_f + ")";
        case ExprOp::GREATER_EQUAL:
            return "(" + a_f + " >= " + b_f + ")";
        case ExprOp::LOGICAL_AND:
            if (compute_dtype == DataType::BOOLEAN) {
                return "(" + a + " && " + b + ")";
            }
            return "((" + a_f + " != 0.0f) && (" + b_f + " != 0.0f))";
        case ExprOp::LOGICAL_OR:
            if (compute_dtype == DataType::BOOLEAN) {
                return "(" + a + " || " + b + ")";
            }
            return "((" + a_f + " != 0.0f) || (" + b_f + " != 0.0f))";

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

static std::string emitWhereComputeExpr(const std::string& cond, const std::string& true_value, const std::string& false_value) {
    return "(" + cond + " ? " + true_value + " : " + false_value + ")";
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
        case ExprOp::RESHAPE:
        case ExprOp::STRIDED_VIEW:
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
        case ExprOp::RESHAPE:
        case ExprOp::UNSQUEEZE:
        case ExprOp::SQUEEZE:
        case ExprOp::RAGGED_VALUEWISE_EXTENT:
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

static DataType tiledLogicalTransposeFrontierStorageDType(const PhysicalExpression& expr, uint32_t frontier_idx) {
    if (frontier_idx >= expr.nodes.size()) {
        throw runtime_error("Tiled logical-transpose frontier index out of range while selecting storage dtype.");
    }
    const ExprNode& frontier = expr.nodes[frontier_idx];
    if (frontier.op != ExprOp::TRANSPOSE) {
        throw runtime_error("Tiled logical-transpose frontier storage dtype requested for a non-transpose node.");
    }
    return requireNodeOutputDType(frontier);
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

static std::string emitResolvedScalarValueExprSuffixed(const PhysicalExpression& expr,
                                                       uint32_t node_idx,
                                                       DataType target_dtype,
                                                       const std::string& suffix);

static void emitScalarNodeSuffixed(std::ostringstream& ss,
                                   const PhysicalExpression& expr,
                                   uint32_t node_idx,
                                   const std::string& idx_expr,
                                   const std::string& suffix,
                                   const std::string& indent,
                                   const std::string& flat_chunk_lane_expr,
                                   uint32_t flat_elements_per_thread,
                                   const std::function<std::string(uint32_t)>& input_slot_value);

static std::string ropeAxisCoordVar(uint32_t axis) { return "c_" + std::to_string(axis); }
static std::string ropeAxisDimVar(uint32_t axis) { return "out_dim_" + std::to_string(axis); }

static void emitScalarAliasNode(
    std::ostringstream& ss, const PhysicalExpression& expr, uint32_t node_idx, uint32_t source_node_idx, const std::string& indent) {
    const DataType emitted_dtype = emittedScalarNodeValueDType(expr.nodes[node_idx]);
    const std::string output_type = scalarStorageType(emitted_dtype);
    ss << indent << "const " << output_type << " t" << node_idx << " = "
       << emitResolvedScalarValueExpr(expr, source_node_idx, emitted_dtype) << ";\n";
}

static std::unordered_set<uint32_t> collectIndexAwareInputNodesToSkipForFlatOutput(const PhysicalExpression& expr,
                                                                                   const std::vector<CompiledStageOutput>& outputs) {
    std::unordered_set<uint32_t> reachable;
    std::unordered_set<uint32_t> ordinary_required;
    for (const CompiledStageOutput& output : outputs) {
        if (output.local_node_idx >= expr.nodes.size()) {
            throw runtime_error("Fused output local_node_idx out of range while collecting index-aware input skips.");
        }
        collectRequiredNodes(expr, output.local_node_idx, reachable);
        collectRequiredNodesExcludingIndexAwareChildren(expr, output.local_node_idx, ordinary_required);
    }

    std::unordered_set<uint32_t> index_aware_required;
    for (uint32_t node_idx : reachable) {
        const ExprNode& node = expr.nodes.at(node_idx);
        if (node.op == ExprOp::STRIDED_VIEW_BACKWARD) {
            if (node.lhs >= expr.nodes.size()) {
                throw runtime_error("strided_view_backward node has invalid lhs while collecting index-aware input skips.");
            }
            collectRequiredNodes(expr, node.lhs, index_aware_required);
            continue;
        }
        if (node.op == ExprOp::ROPE && node.rope_position_ids_node != UINT32_MAX) {
            if (node.rope_position_ids_node >= expr.nodes.size()) {
                throw runtime_error("RoPE node has invalid position ids while collecting index-aware input skips.");
            }
            collectRequiredNodes(expr, node.rope_position_ids_node, index_aware_required);
        }
    }

    std::unordered_set<uint32_t> skip;
    for (uint32_t node_idx : index_aware_required) {
        if (!ordinary_required.contains(node_idx)) {
            // STRIDED_VIEW_BACKWARD and explicit-position RoPE both consume one
            // dependency in a derived index domain rather than the ordinary
            // output-flat-index domain. Nodes reachable only through one of
            // those structural children must be emitted by that op's
            // index-aware emitter, not by the ordinary flat emitter.
            skip.insert(node_idx);
        }
    }
    return skip;
}

static bool expressionHasRopeOp(const PhysicalExpression& expr) {
    return std::any_of(expr.nodes.begin(), expr.nodes.end(), [](const ExprNode& node) { return node.op == ExprOp::ROPE; });
}

static bool expressionHasLogicalTransposeOp(const PhysicalExpression& expr) {
    return std::any_of(expr.nodes.begin(), expr.nodes.end(), [](const ExprNode& node) { return node.op == ExprOp::TRANSPOSE; });
}

static bool expressionHasTakeAlongAxisOp(const PhysicalExpression& expr) {
    return std::any_of(expr.nodes.begin(), expr.nodes.end(), [](const ExprNode& node) { return node.op == ExprOp::TAKE_ALONG_AXIS; });
}

static bool expressionHasStridedViewOp(const PhysicalExpression& expr) {
    return std::any_of(expr.nodes.begin(), expr.nodes.end(), [](const ExprNode& node) {
        return node.op == ExprOp::STRIDED_VIEW || node.op == ExprOp::STRIDED_VIEW_BACKWARD;
    });
}

static uint64_t productDims(const std::vector<uint64_t>& dims) {
    uint64_t result = 1;
    for (uint64_t dim : dims) {
        result *= dim;
    }
    return result;
}

static std::vector<uint64_t> packedStrides(const std::vector<uint64_t>& dims) {
    std::vector<uint64_t> strides(dims.size(), 1ULL);
    uint64_t running = 1ULL;
    for (size_t i = dims.size(); i-- > 0;) {
        strides[i] = running;
        running *= dims[i];
    }
    return strides;
}

static std::vector<size_t> stridedViewBackwardStorageAxisOrder(const ExprNode& node) {
    if (node.view_dims.empty() || node.view_dims.size() != node.view_strides.size()) {
        throw std::runtime_error("strided_view_backward requires view dimensions and strides with the same non-zero rank.");
    }
    std::vector<size_t> axes(node.view_dims.size());
    std::iota(axes.begin(), axes.end(), 0);
    std::sort(axes.begin(), axes.end(), [&](size_t lhs, size_t rhs) {
        if (node.view_strides[lhs] != node.view_strides[rhs]) {
            return node.view_strides[lhs] > node.view_strides[rhs];
        }
        // A size-one axis consumes no storage span. When two axes have the same
        // stride, decode the expanding axis first so the size-one coordinate is
        // evaluated only after the residual has fallen below that stride.
        const bool lhs_expands = node.view_dims[lhs] > 1;
        const bool rhs_expands = node.view_dims[rhs] > 1;
        if (lhs_expands != rhs_expands) return lhs_expands;
        return lhs < rhs;
    });
    return axes;
}

static std::string nextIndexMappedSuffix(uint32_t node_idx, const std::string& emission_scope, uint32_t& counter) {
    return emission_scope + "_im" + std::to_string(node_idx) + "_" + std::to_string(counter++);
}

static std::string nextIndexMappedSuffix(uint32_t node_idx, uint32_t& counter) {
    return nextIndexMappedSuffix(node_idx, "", counter);
}

static void emitBroadcastOffsetForDomain(std::ostringstream& ss,
                                         const std::vector<uint64_t>& input_dims,
                                         const std::vector<uint64_t>& domain_dims,
                                         const std::string& idx_expr,
                                         const std::string& offset_var,
                                         const std::string& indent,
                                         bool use_uint32_index_math) {
    const std::string index_type = emittedIndexType(use_uint32_index_math);

    if (input_dims.empty() || productDims(input_dims) == 1ULL) {
        ss << indent << "const " << index_type << " " << offset_var << " = " << emitUnsignedLiteral(0, use_uint32_index_math) << ";\n";
        return;
    }

    if (input_dims == domain_dims) {
        ss << indent << "const " << index_type << " " << offset_var << " = " << idx_expr << ";\n";
        return;
    }

    if (input_dims.size() > domain_dims.size()) {
        throw runtime_error("Index-mapped fused transpose input rank exceeds its evaluation domain rank.");
    }

    const std::vector<uint64_t> domain_strides = packedStrides(domain_dims);
    const std::vector<uint64_t> input_strides = packedStrides(input_dims);
    const size_t rank_delta = domain_dims.size() - input_dims.size();

    ss << indent << index_type << " " << offset_var << " = " << emitUnsignedLiteral(0, use_uint32_index_math) << ";\n";
    for (size_t axis = 0; axis < domain_dims.size(); ++axis) {
        if (axis < rank_delta) {
            continue;
        }

        const size_t input_axis = axis - rank_delta;
        const uint64_t input_dim = input_dims[input_axis];
        if (input_dim == 1ULL) {
            continue;
        }
        if (input_dim != domain_dims[axis]) {
            throw runtime_error("Index-mapped fused transpose input shape is not broadcast-compatible with its evaluation domain.");
        }

        const std::string coord = offset_var + "_c" + std::to_string(axis);
        std::string coord_expr;
        if (domain_strides[axis] == 1ULL) {
            coord_expr = idx_expr;
        } else if (isPowerOfTwo(domain_strides[axis])) {
            coord_expr = "(" + idx_expr + " >> " + std::to_string(log2Exact(domain_strides[axis])) + ")";
        } else {
            coord_expr = "(" + idx_expr + " / " + emitUnsignedLiteral(domain_strides[axis], use_uint32_index_math) + ")";
        }

        if (domain_dims[axis] != 1ULL) {
            if (isPowerOfTwo(domain_dims[axis])) {
                coord_expr = "(" + coord_expr + " & " + emitUnsignedLiteral(domain_dims[axis] - 1ULL, use_uint32_index_math) + ")";
            } else {
                coord_expr = "(" + coord_expr + " % " + emitUnsignedLiteral(domain_dims[axis], use_uint32_index_math) + ")";
            }
        } else {
            coord_expr = emitUnsignedLiteral(0, use_uint32_index_math);
        }

        ss << indent << "const " << index_type << " " << coord << " = " << coord_expr << ";\n";
        if (input_strides[input_axis] == 1ULL) {
            ss << indent << offset_var << " += " << coord << ";\n";
        } else if (isPowerOfTwo(input_strides[input_axis])) {
            ss << indent << offset_var << " += (" << coord << " << " << std::to_string(log2Exact(input_strides[input_axis])) << ");\n";
        } else {
            ss << indent << offset_var << " += " << coord << " * " << emitUnsignedLiteral(input_strides[input_axis], use_uint32_index_math)
               << ";\n";
        }
    }
}

static std::string emitTransposeSourceIndexForDomain(std::ostringstream& ss,
                                                     const std::vector<uint64_t>& transposed_dims,
                                                     const std::string& idx_expr,
                                                     const std::string& var_name,
                                                     const std::string& indent,
                                                     bool use_uint32_index_math) {
    if (transposed_dims.size() < 2) {
        throw runtime_error("Index-mapped fused transpose requires rank >= 2.");
    }

    const std::string index_type = emittedIndexType(use_uint32_index_math);
    const uint64_t num_rows = transposed_dims[transposed_dims.size() - 1];
    const uint64_t num_cols = transposed_dims[transposed_dims.size() - 2];
    const uint64_t matrix_elems = num_rows * num_cols;
    const std::string num_rows_lit = emitUnsignedLiteral(num_rows, use_uint32_index_math);
    const std::string num_cols_lit = emitUnsignedLiteral(num_cols, use_uint32_index_math);
    const std::string matrix_elems_lit = emitUnsignedLiteral(matrix_elems, use_uint32_index_math);

    ss << indent << "const " << index_type << " " << var_name << "_matrix = " << idx_expr << " % " << matrix_elems_lit << ";\n";
    ss << indent << "const " << index_type << " " << var_name << "_batch = " << idx_expr << " - " << var_name << "_matrix;\n";
    ss << indent << "const " << index_type << " " << var_name << "_out_row = " << var_name << "_matrix / " << num_rows_lit << ";\n";
    ss << indent << "const " << index_type << " " << var_name << "_out_col = " << var_name << "_matrix % " << num_rows_lit << ";\n";
    ss << indent << "const " << index_type << " " << var_name << " = " << var_name << "_batch + " << var_name << "_out_col * "
       << num_cols_lit << " + " << var_name << "_out_row;\n";
    return var_name;
}


// Decodes a row-major index over `dims` into per-axis coordinates. The index must lie in
// [0, prod(dims)), so the outermost non-unit coordinate is the remaining quotient itself (no
// modulo) and nothing is divided past the outermost coordinate that is needed. Each non-unit
// axis inside that costs one division by a constant (a shift for powers of two); its coordinate
// is the remainder recovered with a multiply-subtract. Size-1 axes decode to the literal 0.
// Only axes marked in `needed` are materialized (all when `needed` is empty); the others are
// returned as empty strings.
static std::vector<std::string> emitRowMajorCoordinateDecode(std::ostringstream& ss,
                                                            const std::vector<uint64_t>& dims,
                                                            const std::string& idx_expr,
                                                            const std::string& prefix,
                                                            const std::string& indent,
                                                            bool use_uint32_index_math,
                                                            const std::vector<bool>& needed = {}) {
    const std::string index_type = emittedIndexType(use_uint32_index_math);
    const size_t rank = dims.size();
    const auto is_needed = [&](size_t axis) { return needed.empty() || needed.at(axis); };
    std::vector<std::string> coords(rank);
    size_t outermost_non_unit = rank;
    size_t outermost_needed = rank;
    for (size_t axis = rank; axis-- > 0;) {
        if (dims[axis] == 0) {
            throw runtime_error("Index-mapped coordinate decode does not support zero-sized dimensions.");
        }
        if (dims[axis] == 1ULL) {
            if (is_needed(axis)) coords[axis] = emitUnsignedLiteral(0, use_uint32_index_math);
            continue;
        }
        outermost_non_unit = axis;
        if (is_needed(axis)) outermost_needed = axis;
    }
    if (outermost_needed == rank) {
        return coords;
    }

    const bool simple_idx = std::all_of(
        idx_expr.begin(), idx_expr.end(), [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; });
    std::string residual = simple_idx ? idx_expr : "(" + idx_expr + ")";
    for (size_t axis = rank; axis-- > outermost_needed;) {
        const uint64_t dim = dims[axis];
        if (dim == 1ULL) {
            continue;
        }
        const std::string coord = prefix + "_c" + std::to_string(axis);
        if (axis == outermost_non_unit) {
            coords[axis] = residual;
            break;
        }
        if (axis == outermost_needed) {
            // Last coordinate needed: its remainder alone, no quotient for further axes.
            const std::string remainder = isPowerOfTwo(dim)
                                              ? residual + " & " + emitUnsignedLiteral(dim - 1ULL, use_uint32_index_math)
                                              : residual + " % " + emitUnsignedLiteral(dim, use_uint32_index_math);
            ss << indent << "const " << index_type << " " << coord << " = " << remainder << ";\n";
            coords[axis] = coord;
            break;
        }
        const std::string quotient = prefix + "_q" + std::to_string(axis);
        if (isPowerOfTwo(dim)) {
            ss << indent << "const " << index_type << " " << quotient << " = " << residual << " >> " << log2Exact(dim) << ";\n";
            if (is_needed(axis)) {
                ss << indent << "const " << index_type << " " << coord << " = " << residual << " & "
                   << emitUnsignedLiteral(dim - 1ULL, use_uint32_index_math) << ";\n";
                coords[axis] = coord;
            }
        } else {
            const std::string dim_lit = emitUnsignedLiteral(dim, use_uint32_index_math);
            ss << indent << "const " << index_type << " " << quotient << " = " << residual << " / " << dim_lit << ";\n";
            if (is_needed(axis)) {
                ss << indent << "const " << index_type << " " << coord << " = " << residual << " - " << quotient << " * " << dim_lit
                   << ";\n";
                coords[axis] = coord;
            }
        }
        residual = quotient;
    }
    return coords;
}

// Sum of coord * stride over the axes with a coordinate and a non-zero stride.
static std::string stridedOffsetExpr(const std::vector<std::string>& coords,
                                     const std::vector<uint64_t>& strides,
                                     uint64_t base,
                                     bool use_uint32_index_math) {
    std::string expr;
    const auto append = [&](const std::string& term) { expr += (expr.empty() ? "" : " + ") + term; };
    if (base != 0ULL) append(emitUnsignedLiteral(base, use_uint32_index_math));
    for (size_t axis = 0; axis < coords.size(); ++axis) {
        const uint64_t stride = strides.at(axis);
        if (coords[axis].empty() || stride == 0ULL || coords[axis] == emitUnsignedLiteral(0, use_uint32_index_math)) {
            continue;
        }
        if (stride == 1ULL) {
            append(coords[axis]);
        } else if (isPowerOfTwo(stride)) {
            append("(" + coords[axis] + " << " + std::to_string(log2Exact(stride)) + ")");
        } else {
            append(coords[axis] + " * " + emitUnsignedLiteral(stride, use_uint32_index_math));
        }
    }
    return expr.empty() ? emitUnsignedLiteral(0, use_uint32_index_math) : expr;
}

static uint64_t normalizedTakeAlongAxisForEmitter(const ExprNode& node, uint64_t rank) {
    if (rank == 0) {
        throw runtime_error("take_along_axis requires a non-scalar tensor.");
    }
    const uint64_t encoded_axis = node.reduction_axes.empty() ? UINT64_MAX : node.reduction_axes.front();
    if (encoded_axis == UINT64_MAX) {
        return rank - 1;
    }
    if (encoded_axis >= rank) {
        throw runtime_error("take_along_axis axis is out of range for the input rank.");
    }
    return encoded_axis;
}

static uint64_t validateTakeAlongAxisShapes(const ExprNode& node,
                                           const std::vector<uint64_t>& input_dims,
                                           const std::vector<uint64_t>& indices_dims) {
    if (input_dims.empty() || indices_dims.empty()) {
        throw runtime_error("take_along_axis requires non-scalar input and indices tensors.");
    }
    if (input_dims.size() != indices_dims.size()) {
        throw runtime_error("take_along_axis input and indices tensors must have the same rank.");
    }

    const uint64_t axis = normalizedTakeAlongAxisForEmitter(node, input_dims.size());
    for (size_t i = 0; i < input_dims.size(); ++i) {
        if (i != axis && input_dims[i] != indices_dims[i]) {
            throw runtime_error("take_along_axis input and indices dimensions must match except along the gather axis.");
        }
        if (input_dims[i] == 0 || indices_dims[i] == 0) {
            throw runtime_error("take_along_axis does not support zero-sized dimensions.");
        }
    }
    return axis;
}

// take_along_axis out-of-range semantics. Indices are unsigned (UINT32/UINT64), so the only
// invalid index is one at or past the gather-axis length; such an element reads 0, as an
// out-of-vocabulary EMBEDDING_LOOKUP row does. The bound is checked on the full-width index,
// before it narrows to the kernel index type (a UINT64 index of 2^32 + 1 must not wrap to 1),
// and the gather coordinate of an out-of-range element is 0, so no lane ever reads outside
// the source. The caller selects the zero with the returned in-range flag.
struct TakeAlongAxisGatherIndex {
    std::string coordinate;
    std::string in_range;
};

static TakeAlongAxisGatherIndex emitTakeAlongAxisGatherIndex(std::ostringstream& ss,
                                                             const std::string& selected_value,
                                                             DataType selected_dtype,
                                                             uint64_t axis_dim,
                                                             const std::string& coordinate_var,
                                                             const std::string& in_range_var,
                                                             const std::string& indent,
                                                             bool use_uint32_index_math) {
    const bool narrow_index = isIntegralScalarCastDType(selected_dtype) && selected_dtype != DataType::UINT64 &&
                              selected_dtype != DataType::INT64;
    const bool compare_in_uint32 = use_uint32_index_math && narrow_index && fitsInUInt32(axis_dim);
    const DataType compare_dtype = compare_in_uint32 ? DataType::UINT32 : DataType::UINT64;
    const std::string index_type = emittedIndexType(use_uint32_index_math);
    const std::string raw_var = coordinate_var + "_raw";
    ss << indent << "const " << scalarStorageType(compare_dtype) << " " << raw_var << " = "
       << castScalarExpr(selected_value, selected_dtype, compare_dtype) << ";\n";
    ss << indent << "const bool " << in_range_var << " = " << raw_var << " < " << emitUnsignedLiteral(axis_dim, compare_in_uint32)
       << ";\n";
    const std::string narrowed = compare_in_uint32 == use_uint32_index_math ? raw_var : "static_cast<" + index_type + ">(" + raw_var + ")";
    ss << indent << "const " << index_type << " " << coordinate_var << " = " << in_range_var << " ? " << narrowed << " : "
       << emitUnsignedLiteral(0, use_uint32_index_math) << ";\n";
    return {coordinate_var, in_range_var};
}

// Logical (dense row-major) index into the take_along_axis input for the element at idx_expr in
// the indices domain: the indices coordinates with the gathered index on the gather axis. Returns
// the in-range flag of the gathered index (see emitTakeAlongAxisGatherIndex).
static std::string emitTakeAlongAxisSourceIndexForDomain(std::ostringstream& ss,
                                                         const ExprNode& node,
                                                         const std::vector<uint64_t>& input_dims,
                                                         const std::vector<uint64_t>& indices_dims,
                                                         const std::string& idx_expr,
                                                         const std::string& selected_index_expr,
                                                         DataType selected_index_dtype,
                                                         const std::string& var_name,
                                                         const std::string& indent,
                                                         bool use_uint32_index_math) {
    const uint64_t axis = validateTakeAlongAxisShapes(node, input_dims, indices_dims);
    const std::string index_type = emittedIndexType(use_uint32_index_math);
    std::vector<bool> needed(indices_dims.size(), true);
    needed[axis] = false;
    std::vector<std::string> coords =
        emitRowMajorCoordinateDecode(ss, indices_dims, idx_expr, var_name, indent, use_uint32_index_math, needed);
    const TakeAlongAxisGatherIndex gather = emitTakeAlongAxisGatherIndex(ss,
                                                                         selected_index_expr,
                                                                         selected_index_dtype,
                                                                         input_dims[axis],
                                                                         var_name + "_gather",
                                                                         var_name + "_in_range",
                                                                         indent,
                                                                         use_uint32_index_math);
    coords[axis] = gather.coordinate;
    ss << indent << "const " << index_type << " " << var_name << " = "
       << stridedOffsetExpr(coords, packedStrides(input_dims), 0, use_uint32_index_math) << ";\n";
    return gather.in_range;
}

static std::optional<size_t> findUsedInputSlotIndex(const SpecializedBroadcastGroup* group, uint32_t input_slot) {
    if (group == nullptr) {
        return std::nullopt;
    }
    for (size_t i = 0; i < group->used_input_slots.size(); ++i) {
        if (group->used_input_slots[i] == input_slot) {
            return i;
        }
    }
    return std::nullopt;
}

static const std::vector<uint64_t>& usedInputVisibleDims(const SpecializedBroadcastGroup& group, size_t used_i) {
    if (used_i >= group.used_input_visible_dims.size() || used_i >= group.used_input_visible_strides.size()) {
        throw runtime_error("Index-aware input storage mapping is missing visible input layout metadata.");
    }
    const std::vector<uint64_t>& dims = group.used_input_visible_dims[used_i];
    if (dims.size() != group.used_input_visible_strides[used_i].size()) {
        throw runtime_error("Index-aware input storage mapping requires visible dimensions and strides with the same rank.");
    }
    return dims;
}

static bool visibleStridesArePacked(const std::vector<uint64_t>& dims, const std::vector<uint64_t>& strides) {
    uint64_t expected_stride = 1ULL;
    for (size_t axis = dims.size(); axis-- > 0;) {
        if (dims[axis] == 1ULL) {
            continue;
        }
        if (strides[axis] != expected_stride) {
            return false;
        }
        expected_stride *= dims[axis];
    }
    return true;
}

// Storage stride of a stage input along each axis of its node dims, or nullopt when the node
// dims and the visible layout do not line up axis for axis (ignoring size-1 axes). An input
// without visible layout metadata is addressed densely.
static std::optional<std::vector<uint64_t>> inputStorageStridesForNodeDims(const SpecializedBroadcastGroup& group,
                                                                           uint32_t input_slot,
                                                                           const std::vector<uint64_t>& node_dims) {
    const std::optional<size_t> used_i = findUsedInputSlotIndex(&group, input_slot);
    if (!used_i.has_value()) {
        return packedStrides(node_dims);
    }
    const std::vector<uint64_t>& dims = usedInputVisibleDims(group, used_i.value());
    const std::vector<uint64_t>& strides = group.used_input_visible_strides[used_i.value()];
    std::vector<uint64_t> node_strides(node_dims.size(), 0ULL);
    size_t visible_axis = 0;
    for (size_t axis = 0; axis < node_dims.size(); ++axis) {
        if (node_dims[axis] == 1ULL) {
            continue;
        }
        while (visible_axis < dims.size() && dims[visible_axis] == 1ULL) ++visible_axis;
        if (visible_axis == dims.size() || dims[visible_axis] != node_dims[axis]) {
            return std::nullopt;
        }
        node_strides[axis] = strides[visible_axis++];
    }
    while (visible_axis < dims.size() && dims[visible_axis] == 1ULL) ++visible_axis;
    if (visible_axis != dims.size()) {
        return std::nullopt;
    }
    return node_strides;
}

static std::string emitInputStorageIndexForLogicalIndex(std::ostringstream& ss,
                                                        const SpecializedBroadcastGroup* group,
                                                        uint32_t input_slot,
                                                        const std::string& idx_expr,
                                                        const std::string& suffix,
                                                        const std::string& indent,
                                                        bool use_uint32_index_math);

// Maps a dense index in a strided view's own domain to the logical index of its source.
// view_dims may contain the Expression copy-dimension marker (0) in the node; callers pass
// the dims resolved by shape inference against the physical input at this placement.
static std::string emitStridedViewSourceIndex(std::ostringstream& ss,
                                              const ExprNode& n,
                                              const std::vector<uint64_t>& view_dims,
                                              const std::string& view_idx_expr,
                                              const std::string& suffix,
                                              const std::string& indent,
                                              bool use_uint32_index_math) {
    if (n.view_dims.empty() || n.view_dims.size() != n.view_strides.size() || view_dims.size() != n.view_dims.size()) {
        throw runtime_error("Index-mapped strided_view emission requires dimensions and strides with the same non-zero rank.");
    }
    for (size_t axis = 0; axis < view_dims.size(); ++axis) {
        if (view_dims[axis] == 0 || n.view_strides[axis] == 0) {
            throw runtime_error("Index-mapped strided_view dimensions and strides must be non-zero.");
        }
    }
    const std::string source_idx = "sv_source_idx" + suffix;
    const std::vector<std::string> coords =
        emitRowMajorCoordinateDecode(ss, view_dims, view_idx_expr, "sv" + suffix, indent, use_uint32_index_math);
    ss << indent << "const " << emittedIndexType(use_uint32_index_math) << " " << source_idx << " = "
       << stridedOffsetExpr(coords, n.view_strides, n.view_element_offset, use_uint32_index_math) << ";\n";
    return source_idx;
}

static std::string emitIndexMappedScalarValue(std::ostringstream& ss,
                                              const PhysicalExpression& expr,
                                              const SpecializedBroadcastGroup& group,
                                              uint32_t node_idx,
                                              const std::string& idx_expr,
                                              const std::vector<uint64_t>& domain_dims,
                                              const std::string& indent,
                                              bool use_uint32_index_math,
                                              const std::string& emission_scope,
                                              uint32_t& counter) {
    if (node_idx >= expr.nodes.size() || node_idx >= group.node_dims.size()) {
        throw runtime_error("Index-mapped fused emitter node index out of range.");
    }

    const ExprNode& n = expr.nodes[node_idx];
    const DataType emitted_dtype = emittedScalarNodeValueDType(n);
    const std::string output_type = scalarStorageType(emitted_dtype);
    // A physical child can be evaluated from multiple index-aware roots in the
    // same generated CUDA scope (for example, sibling strided-view backward
    // scatters). Namespace temporaries by that root as well as by node/counter.
    const std::string suffix = nextIndexMappedSuffix(node_idx, emission_scope, counter);
    const std::string var = "t" + std::to_string(node_idx) + suffix;

    double folded_constant = 0.0;
    if (n.op != ExprOp::INPUT && n.op != ExprOp::RUNTIME_SCALAR && n.op != ExprOp::TENSOR_RUNTIME_SCALAR &&
        tryGetEmitterConstantValue(expr, node_idx, folded_constant)) {
        ss << indent << "const " << output_type << " " << var << " = "
           << castScalarExpr(emitScalarFpLiteral(folded_constant), DataType::FP32, emitted_dtype) << ";\n";
        return var;
    }

    switch (n.op) {
        case ExprOp::INPUT: {
            const DataType input_tensor_dtype = requireNodeInputTensorDType(n);

            // The caller's index is expressed in the logical evaluation domain.
            // First collapse any broadcasted axes into this input node's logical
            // dense index. Then map that logical index through the runtime input's
            // visible storage layout. The second step is essential when a stage
            // input is itself a strided tensor view: indexing in[slot][logical]
            // directly would silently treat that view as dense contiguous.
            const std::string logical_offset_var = "logical_offset" + suffix;
            emitBroadcastOffsetForDomain(
                ss, group.node_dims[node_idx], domain_dims, idx_expr, logical_offset_var, indent, use_uint32_index_math);
            const std::string storage_idx = emitInputStorageIndexForLogicalIndex(
                ss, &group, n.input_slot, logical_offset_var, suffix + "_storage", indent, use_uint32_index_math);
            ss << indent << "const " << output_type << " " << var << " = "
               << castScalarExpr("in" + std::to_string(n.input_slot) + "[" + storage_idx + "]", input_tensor_dtype, emitted_dtype) << ";\n";
            return var;
        }
        case ExprOp::RUNTIME_SCALAR: {
            const DataType input_tensor_dtype = requireNodeInputTensorDType(n);
            ss << indent << "const " << output_type << " " << var << " = "
               << castScalarExpr("in" + std::to_string(n.input_slot), input_tensor_dtype, emitted_dtype) << ";\n";
            return var;
        }
        case ExprOp::TENSOR_RUNTIME_SCALAR: {
            const DataType input_tensor_dtype = requireNodeInputTensorDType(n);
            const std::string input_storage_type = scalarStorageType(input_tensor_dtype);
            const std::string input_expr = "reinterpret_cast<const " + input_storage_type + "*>(in" + std::to_string(n.input_slot) + ")[0]";
            ss << indent << "const " << output_type << " " << var << " = " << castScalarExpr(input_expr, input_tensor_dtype, emitted_dtype)
               << ";\n";
            return var;
        }
        case ExprOp::SCALAR_FP:
        case ExprOp::FILL: {
            ss << indent << "const " << output_type << " " << var << " = "
               << castScalarExpr(emitScalarFpLiteral(n.scalar_fp), DataType::FP32, emitted_dtype) << ";\n";
            return var;
        }
        case ExprOp::ROPE:
            throw runtime_error("Index-mapped fused transpose emission does not support nested RoPE; materialize one side first.");
        default:
            break;
    }

    if (n.op == ExprOp::RAGGED_VALUEWISE_EXTENT) {
        // RAGGED_VALUEWISE_EXTENT changes launch bounds only. Its rhs offsets
        // dependency is structural metadata; scalar value evaluation is exactly
        // the lhs value in the same logical index domain.
        const std::string value = emitIndexMappedScalarValue(
            ss, expr, group, n.lhs, idx_expr, domain_dims, indent, use_uint32_index_math, emission_scope, counter);
        return castScalarExpr(value, emittedScalarNodeValueDType(expr.nodes[n.lhs]), emitted_dtype);
    }

    if (n.op == ExprOp::STRIDED_VIEW_BACKWARD) {
        if (n.view_dims.empty() || n.view_dims.size() != n.view_strides.size() || n.fill_dims.empty()) {
            throw runtime_error("Index-mapped strided_view_backward requires source/view dimensions and strides.");
        }

        std::string eval_idx = idx_expr;
        if (group.node_dims[node_idx] != domain_dims) {
            eval_idx = "svbw_eval_idx" + suffix;
            emitBroadcastOffsetForDomain(
                ss, group.node_dims[node_idx], domain_dims, idx_expr, eval_idx, indent, use_uint32_index_math);
        }

        const std::string in_view = "svbw_in_view" + suffix;
        const std::string residual = "svbw_residual" + suffix;
        const std::string view_linear = "svbw_linear" + suffix;
        ss << indent << "bool " << in_view << " = true;\n";
        ss << indent << emittedIndexType(use_uint32_index_math) << " " << residual << " = static_cast<"
           << emittedIndexType(use_uint32_index_math) << ">(" << eval_idx << ");\n";
        if (n.view_element_offset != 0) {
            ss << indent << "if (" << residual << " < " << emitUnsignedLiteral(n.view_element_offset, use_uint32_index_math)
               << ") { " << in_view << " = false; } else { " << residual << " -= "
               << emitUnsignedLiteral(n.view_element_offset, use_uint32_index_math) << "; }\n";
        }
        const std::vector<size_t> storage_axis_order = stridedViewBackwardStorageAxisOrder(n);
        std::vector<std::string> coords(n.view_dims.size());
        for (size_t axis = 0; axis < n.view_dims.size(); ++axis) {
            const uint64_t dim = n.view_dims[axis];
            const uint64_t stride = n.view_strides[axis];
            if (dim == 0 || stride == 0) {
                throw runtime_error("Index-mapped strided_view_backward dimensions and strides must be non-zero.");
            }
            coords[axis] = "svbw_coord" + suffix + "_" + std::to_string(axis);
            ss << indent << emittedIndexType(use_uint32_index_math) << " " << coords[axis] << " = 0;\n";
        }
        for (size_t axis : storage_axis_order) {
            const uint64_t dim = n.view_dims[axis];
            const uint64_t stride = n.view_strides[axis];
            const std::string& coord = coords[axis];
            ss << indent << "if (" << in_view << ") {\n";
            ss << indent << "  " << coord << " = " << residual << " / "
               << emitUnsignedLiteral(stride, use_uint32_index_math) << ";\n";
            ss << indent << "  if (" << coord << " >= " << emitUnsignedLiteral(dim, use_uint32_index_math) << ") {\n";
            ss << indent << "    " << in_view << " = false;\n";
            ss << indent << "  } else {\n";
            ss << indent << "    " << residual << " -= " << coord << " * " << emitUnsignedLiteral(stride, use_uint32_index_math)
               << ";\n";
            ss << indent << "  }\n";
            ss << indent << "}\n";
        }
        ss << indent << "if (" << residual << " != 0) " << in_view << " = false;\n";
        ss << indent << emittedIndexType(use_uint32_index_math) << " " << view_linear << " = 0;\n";
        for (size_t axis = 0; axis < n.view_dims.size(); ++axis) {
            ss << indent << view_linear << " = " << view_linear << " * "
               << emitUnsignedLiteral(n.view_dims[axis], use_uint32_index_math) << " + " << coords[axis] << ";\n";
        }

        const std::string grad_value = emitIndexMappedScalarValue(ss,
                                                                   expr,
                                                                   group,
                                                                   n.lhs,
                                                                   view_linear,
                                                                   n.view_dims,
                                                                   indent,
                                                                   use_uint32_index_math,
                                                                   emission_scope,
                                                                   counter);
        const DataType grad_dtype = emittedScalarNodeValueDType(expr.nodes[n.lhs]);
        const std::string cast_grad = castScalarExpr(grad_value, grad_dtype, emitted_dtype);
        const std::string zero = castScalarExpr("0.0f", DataType::FP32, emitted_dtype);
        ss << indent << "const " << output_type << " " << var << " = " << in_view << " ? " << cast_grad << " : " << zero
           << ";\n";
        return var;
    }

    if (n.op == ExprOp::TRANSPOSE) {
        const std::string source_idx = "idx" + suffix;
        emitTransposeSourceIndexForDomain(ss, domain_dims, idx_expr, source_idx, indent, use_uint32_index_math);
        const std::string value =
            emitIndexMappedScalarValue(ss, expr, group, n.lhs, source_idx, group.node_dims[n.lhs], indent, use_uint32_index_math, emission_scope, counter);
        return castScalarExpr(value, emittedScalarNodeValueDType(expr.nodes[n.lhs]), emitted_dtype);
    }

    if (n.op == ExprOp::TAKE_ALONG_AXIS) {
        // Decode the element's coordinates in the indices domain once. The gathered element's
        // address is then built straight from them: the input's storage strides on every axis,
        // the gathered index on the gather axis. A non-input source gets the dense logical index
        // instead. The indices tensor is read through the same coordinates when it is a
        // non-dense input; otherwise its index is take_idx itself.
        const std::vector<uint64_t>& input_dims = group.node_dims[n.lhs];
        const std::vector<uint64_t>& indices_dims = group.node_dims[n.rhs];
        const uint64_t axis = validateTakeAlongAxisShapes(n, input_dims, indices_dims);
        const std::string index_type = emittedIndexType(use_uint32_index_math);
        const std::string take_idx = "take_eval_idx" + suffix;
        emitBroadcastOffsetForDomain(ss, group.node_dims[node_idx], domain_dims, idx_expr, take_idx, indent, use_uint32_index_math);

        const ExprNode& source_node = expr.nodes[n.lhs];
        const ExprNode& indices_node = expr.nodes[n.rhs];
        const std::optional<std::vector<uint64_t>> source_storage_strides =
            source_node.op == ExprOp::INPUT ? inputStorageStridesForNodeDims(group, source_node.input_slot, input_dims) : std::nullopt;
        const std::vector<uint64_t> source_strides = source_storage_strides.value_or(packedStrides(input_dims));
        std::optional<std::vector<uint64_t>> indices_strides =
            indices_node.op == ExprOp::INPUT ? inputStorageStridesForNodeDims(group, indices_node.input_slot, indices_dims) : std::nullopt;
        if (indices_strides.has_value() && visibleStridesArePacked(indices_dims, indices_strides.value())) {
            indices_strides.reset();
        }

        std::vector<bool> needed(indices_dims.size(), false);
        for (size_t a = 0; a < indices_dims.size(); ++a) {
            needed[a] = (a != axis && source_strides[a] != 0ULL) || (indices_strides.has_value() && indices_strides.value()[a] != 0ULL);
        }
        std::vector<std::string> coords =
            emitRowMajorCoordinateDecode(ss, indices_dims, take_idx, "take" + suffix, indent, use_uint32_index_math, needed);

        const auto emit_input_load = [&](const ExprNode& input_node, uint32_t input_node_idx, const std::string& storage_idx) {
            const DataType value_dtype = emittedScalarNodeValueDType(input_node);
            const std::string load_var =
                "t" + std::to_string(input_node_idx) + nextIndexMappedSuffix(input_node_idx, emission_scope, counter);
            ss << indent << "const " << scalarStorageType(value_dtype) << " " << load_var << " = "
               << castScalarExpr("in" + std::to_string(input_node.input_slot) + "[" + storage_idx + "]",
                                 requireNodeInputTensorDType(input_node),
                                 value_dtype)
               << ";\n";
            return load_var;
        };

        std::string selected_value;
        if (indices_strides.has_value()) {
            selected_value =
                emit_input_load(indices_node, n.rhs, stridedOffsetExpr(coords, indices_strides.value(), 0, use_uint32_index_math));
        } else {
            selected_value = emitIndexMappedScalarValue(
                ss, expr, group, n.rhs, take_idx, indices_dims, indent, use_uint32_index_math, emission_scope, counter);
        }
        const TakeAlongAxisGatherIndex gather = emitTakeAlongAxisGatherIndex(ss,
                                                                             selected_value,
                                                                             emittedScalarNodeValueDType(indices_node),
                                                                             input_dims[axis],
                                                                             "take_gather_idx" + suffix,
                                                                             "take_in_range" + suffix,
                                                                             indent,
                                                                             use_uint32_index_math);
        coords[axis] = gather.coordinate;

        const std::string source_idx = "take_source_idx" + suffix;
        ss << indent << "const " << index_type << " " << source_idx << " = "
           << stridedOffsetExpr(coords, source_strides, 0, use_uint32_index_math) << ";\n";
        const std::string value =
            source_storage_strides.has_value()
                ? emit_input_load(source_node, n.lhs, source_idx)
                : emitIndexMappedScalarValue(
                      ss, expr, group, n.lhs, source_idx, input_dims, indent, use_uint32_index_math, emission_scope, counter);
        const std::string taken = "take_value" + suffix;
        ss << indent << "const " << scalarStorageType(emitted_dtype) << " " << taken << " = " << gather.in_range << " ? "
           << castScalarExpr(value, emittedScalarNodeValueDType(source_node), emitted_dtype) << " : "
           << castScalarExpr("0.0f", DataType::FP32, emitted_dtype) << ";\n";
        return taken;
    }

    if (n.op == ExprOp::STRIDED_VIEW) {
        // A view broadcast by its parent is first collapsed to its own dense index, as for the
        // shape aliases below; decoding view coordinates from the parent-domain index would
        // read the wrong elements (e.g. a [3,1] view in a [3,4] domain).
        std::string view_idx = idx_expr;
        if (group.node_dims[node_idx] != domain_dims) {
            view_idx = "view_idx" + suffix;
            emitBroadcastOffsetForDomain(ss, group.node_dims[node_idx], domain_dims, idx_expr, view_idx, indent, use_uint32_index_math);
        }
        const std::string source_idx =
            emitStridedViewSourceIndex(ss, n, group.node_dims[node_idx], view_idx, suffix, indent, use_uint32_index_math);
        const std::string value = emitIndexMappedScalarValue(
            ss, expr, group, n.lhs, source_idx, group.node_dims[n.lhs], indent, use_uint32_index_math, emission_scope, counter);
        return castScalarExpr(value, emittedScalarNodeValueDType(expr.nodes[n.lhs]), emitted_dtype);
    }

    if (n.op == ExprOp::RESHAPE || n.op == ExprOp::UNSQUEEZE || n.op == ExprOp::SQUEEZE) {
        // Shape-only aliases preserve linear element order, but the alias itself may
        // be broadcast by a parent expression. First map the caller's logical index
        // into this node's own dense domain, then carry that linear index through
        // the alias to its same-numel child. Without this step an unsqueezed [B,1]
        // child evaluated in a [B,D] broadcast domain would incorrectly pass the
        // [B,D] linear index directly to its [B] input.
        std::string alias_idx = idx_expr;
        if (group.node_dims[node_idx] != domain_dims) {
            alias_idx = "alias_idx" + suffix;
            emitBroadcastOffsetForDomain(
                ss, group.node_dims[node_idx], domain_dims, idx_expr, alias_idx, indent, use_uint32_index_math);
        }
        const std::string value = emitIndexMappedScalarValue(
            ss, expr, group, n.lhs, alias_idx, group.node_dims[n.lhs], indent, use_uint32_index_math, emission_scope, counter);
        return castScalarExpr(value, emittedScalarNodeValueDType(expr.nodes[n.lhs]), emitted_dtype);
    }

    const DataType compute_dtype = requireNodeComputeDType(n);
    auto emit_child = [&](uint32_t child_idx, const std::vector<uint64_t>& child_domain_dims) -> std::string {
        const std::string value =
            emitIndexMappedScalarValue(ss, expr, group, child_idx, idx_expr, child_domain_dims, indent, use_uint32_index_math, emission_scope, counter);
        return castScalarExpr(value, emittedScalarNodeValueDType(expr.nodes[child_idx]), compute_dtype);
    };

    std::string compute_expr;
    if (n.op == ExprOp::WHERE) {
        auto emit_child_as = [&](uint32_t child_idx, const std::vector<uint64_t>& child_domain_dims, DataType target_dtype) -> std::string {
            const std::string value =
                emitIndexMappedScalarValue(ss, expr, group, child_idx, idx_expr, child_domain_dims, indent, use_uint32_index_math, emission_scope, counter);
            return castScalarExpr(value, emittedScalarNodeValueDType(expr.nodes[child_idx]), target_dtype);
        };
        compute_expr = emitWhereComputeExpr(emit_child_as(n.lhs, domain_dims, DataType::BOOLEAN),
                                            emit_child_as(n.rhs, domain_dims, compute_dtype),
                                            emit_child_as(n.aux, domain_dims, compute_dtype));
    } else if (Expression::isBinaryOp(n.op)) {
        compute_expr = emitBinaryComputeExpr(n.op, emit_child(n.lhs, domain_dims), emit_child(n.rhs, domain_dims), compute_dtype);
    } else if (Expression::isUnaryOp(n.op)) {
        compute_expr = emitUnaryComputeExpr(n.op, emit_child(n.lhs, domain_dims), compute_dtype);
    } else {
        throw runtime_error("Unsupported op in index-mapped fused transpose emitter: " + to_string(static_cast<int>(n.op)));
    }

    ss << indent << "const " << output_type << " " << var << " = " << castScalarExpr(compute_expr, compute_dtype, emitted_dtype) << ";\n";
    return var;
}

static void collectReachableLogicalTransposeNodes(const PhysicalExpression& expr,
                                                  uint32_t node_idx,
                                                  std::unordered_set<uint32_t>& visited,
                                                  std::vector<uint32_t>& transpose_nodes) {
    if (node_idx >= expr.nodes.size()) {
        throw runtime_error("Logical-transpose frontier search node index out of range.");
    }
    if (!visited.insert(node_idx).second) {
        return;
    }

    const ExprNode& node = expr.nodes[node_idx];
    if (node.op == ExprOp::TRANSPOSE) {
        transpose_nodes.push_back(node_idx);
    }
    if (Expression::isLeafOp(node.op)) {
        return;
    }

    collectReachableLogicalTransposeNodes(expr, node.lhs, visited, transpose_nodes);
    if (Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) {
        collectReachableLogicalTransposeNodes(expr, node.rhs, visited, transpose_nodes);
    }
    if (Expression::isTernaryOp(node.op)) {
        collectReachableLogicalTransposeNodes(expr, node.aux, visited, transpose_nodes);
    }
}

static bool subgraphContainsLogicalTranspose(const PhysicalExpression& expr, uint32_t node_idx) {
    std::unordered_set<uint32_t> visited;
    std::vector<uint32_t> transpose_nodes;
    collectReachableLogicalTransposeNodes(expr, node_idx, visited, transpose_nodes);
    return !transpose_nodes.empty();
}

static std::optional<std::vector<uint32_t>> tryFindTiledLogicalTransposeConsumerFrontiers(
    const CompiledExecutionStage& stage, const std::vector<SpecializedBroadcastGroup>& groups) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel || stageHasTransposedMaterializedOutput(stage.outputs)) {
        return std::nullopt;
    }
    if (groups.size() != 1 || groups[0].output_indices.empty()) {
        return std::nullopt;
    }

    const SpecializedBroadcastGroup& group = groups[0];
    if (group.node_dims.size() != stage.expr.nodes.size() || group.output_dims.size() < 2) {
        return std::nullopt;
    }
    if (expressionHasRopeOp(stage.expr)) {
        return std::nullopt;
    }

    std::unordered_set<uint32_t> output_index_set;
    output_index_set.reserve(group.output_indices.size());
    for (uint32_t out_idx : group.output_indices) {
        if (out_idx >= stage.outputs.size() || !output_index_set.insert(out_idx).second) {
            return std::nullopt;
        }
    }
    if (output_index_set.size() != stage.outputs.size()) {
        // A tiled logical-transpose launch has one grid/domain. Only select it
        // when every stage output belongs to this one resolved broadcast group.
        return std::nullopt;
    }

    std::vector<uint32_t> transpose_nodes;
    for (uint32_t out_idx : group.output_indices) {
        const CompiledStageOutput& output = stage.outputs[out_idx];
        if (output.local_node_idx >= stage.expr.nodes.size()) {
            throw runtime_error("Tiled logical-transpose consumer output local node out of range.");
        }
        if (stage.expr.nodes[output.local_node_idx].op == ExprOp::TRANSPOSE) {
            // Terminal transposed outputs are handled by the materialized tiled-transpose path.
            return std::nullopt;
        }

        std::unordered_set<uint32_t> visited;
        collectReachableLogicalTransposeNodes(stage.expr, output.local_node_idx, visited, transpose_nodes);
    }
    if (transpose_nodes.empty()) {
        return std::nullopt;
    }
    std::sort(transpose_nodes.begin(), transpose_nodes.end());
    transpose_nodes.erase(std::unique(transpose_nodes.begin(), transpose_nodes.end()), transpose_nodes.end());

    for (uint32_t frontier_idx : transpose_nodes) {
        const ExprNode& frontier = stage.expr.nodes[frontier_idx];
        if (frontier.lhs == UINT32_MAX || frontier.lhs >= stage.expr.nodes.size()) {
            throw runtime_error("Tiled logical-transpose consumer frontier is missing its lhs.");
        }

        // Auto-swizzle frontiers must be true tile boundaries: the producer side
        // is evaluated in the pre-transpose/source domain and may not itself
        // require another logical transpose. Nested transpose chains need a
        // separate simplification/tiled lowering instead of falling through to a
        // silent strided index-mapped kernel.
        if (subgraphContainsLogicalTranspose(stage.expr, frontier.lhs)) {
            return std::nullopt;
        }

        const std::vector<uint64_t>& source_dims = group.node_dims[frontier.lhs];
        const std::vector<uint64_t>& transposed_dims = group.node_dims[frontier_idx];
        if (source_dims.size() < 2 || transposed_dims.size() < 2) {
            return std::nullopt;
        }
        if (transposed_dims != group.output_dims) {
            // The auto-swizzle path intentionally only handles transposed full
            // tensor values consumed in the final output domain. Mixed-domain
            // transpose chains must be simplified or materialized explicitly.
            return std::nullopt;
        }
        if (source_dims.size() != transposed_dims.size()) {
            return std::nullopt;
        }
        for (size_t axis = 0; axis + 2 < source_dims.size(); ++axis) {
            if (source_dims[axis] != transposed_dims[axis]) {
                return std::nullopt;
            }
        }
        if (source_dims[source_dims.size() - 2] != transposed_dims[transposed_dims.size() - 1] ||
            source_dims[source_dims.size() - 1] != transposed_dims[transposed_dims.size() - 2]) {
            return std::nullopt;
        }
    }

    return transpose_nodes;
}

static uint32_t tiledLogicalTransposeConsumerSlotBytes(const CompiledExecutionStage& stage,
                                                       const std::vector<SpecializedBroadcastGroup>& groups) {
    const std::optional<std::vector<uint32_t>> maybe_frontiers = tryFindTiledLogicalTransposeConsumerFrontiers(stage, groups);
    if (!maybe_frontiers.has_value() || maybe_frontiers->empty()) {
        return sizeof(unsigned int);
    }

    uint32_t max_slot_bytes = 1;
    for (uint32_t frontier_idx : maybe_frontiers.value()) {
        if (frontier_idx >= stage.expr.nodes.size()) {
            throw runtime_error("Tiled logical-transpose consumer frontier index out of range while selecting pack width.");
        }
        const DataType frontier_dtype = tiledLogicalTransposeFrontierStorageDType(stage.expr, frontier_idx);
        const uint32_t dtype_bytes = scalarStorageTypeSizeBytes(frontier_dtype);
        if (dtype_bytes > sizeof(unsigned int)) {
            throw runtime_error("Tiled logical-transpose auto-swizzle currently supports frontier scalar storage types up to 32 bits.");
        }
        max_slot_bytes = std::max(max_slot_bytes, dtype_bytes);
    }
    return max_slot_bytes;
}

uint32_t CudaSourceEmitter::tiledLogicalTransposeConsumerSlotBytes(const CompiledExecutionStage& stage,
                                                                   const std::vector<SpecializedBroadcastGroup>& groups) {
    return ::ThorImplementation::tiledLogicalTransposeConsumerSlotBytes(stage, groups);
}

uint32_t CudaSourceEmitter::tiledLogicalTransposeConsumerPackScalars(const CompiledExecutionStage& stage,
                                                                     const std::vector<SpecializedBroadcastGroup>& groups) {
    const uint32_t slot_bytes = CudaSourceEmitter::tiledLogicalTransposeConsumerSlotBytes(stage, groups);
    return std::max<uint32_t>(1U, static_cast<uint32_t>(sizeof(unsigned int)) / slot_bytes);
}

struct TiledLogicalTransposeFrontierValue {
    std::string value_expr;
    DataType dtype;
};

struct TiledLogicalTransposeFrontierVectorValue {
    std::string value_expr;
    DataType scalar_dtype;
};

struct TiledLogicalTransposeDenseInputLoad {
    uint32_t input_slot = UINT32_MAX;
    DataType input_dtype = DataType::FP32;
};

static bool isLogicalTransposeVectorComputeDType(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::BF16; }

static std::string logicalTransposeVectorComputeType(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return "half2";
        case DataType::BF16:
            return "__nv_bfloat162";
        default:
            throw runtime_error("Unsupported logical transpose vector compute dtype: " + TensorDescriptor::getElementTypeName(dtype));
    }
}

static std::string emitLogicalTransposeVector2PackScalarsAsCompute(const std::string& value0,
                                                                   const std::string& value1,
                                                                   DataType value_dtype,
                                                                   DataType vector_compute_dtype) {
    const std::string lane0 = castScalarExpr(value0, value_dtype, vector_compute_dtype);
    const std::string lane1 = castScalarExpr(value1, value_dtype, vector_compute_dtype);

    switch (vector_compute_dtype) {
        case DataType::FP16:
            return "__halves2half2(" + lane0 + ", " + lane1 + ")";
        case DataType::BF16:
            return "__halves2bfloat162(" + lane0 + ", " + lane1 + ")";
        default:
            throw runtime_error("Unsupported logical transpose vector pack dtype: " +
                                TensorDescriptor::getElementTypeName(vector_compute_dtype));
    }
}

static std::string emitLogicalTransposeVector2Lane(const std::string& vector_value, unsigned int lane) {
    if (lane == 0U) {
        return vector_value + ".x";
    }
    if (lane == 1U) {
        return vector_value + ".y";
    }
    throw runtime_error("Vector2 lane extraction supports lane 0 or 1.");
}

static std::optional<uint64_t> productDimsNoOverflow(const std::vector<uint64_t>& dims) {
    uint64_t product = 1;
    for (uint64_t dim : dims) {
        if (dim != 0 && product > std::numeric_limits<uint64_t>::max() / dim) {
            return std::nullopt;
        }
        product *= dim;
    }
    return product;
}

static bool isContiguousMetadataAliasOp(ExprOp op) { return op == ExprOp::RESHAPE || op == ExprOp::UNSQUEEZE || op == ExprOp::SQUEEZE; }

static std::optional<TiledLogicalTransposeDenseInputLoad> tryResolveTiledLogicalTransposeDenseInputLoad(
    const PhysicalExpression& expr,
    const SpecializedBroadcastGroup& group,
    uint32_t source_idx,
    const std::vector<uint64_t>& source_domain_dims,
    DataType frontier_dtype) {
    if (source_idx >= expr.nodes.size() || source_idx >= group.node_dims.size()) {
        throw runtime_error("Tiled logical-transpose dense-input load source index out of range.");
    }

    const std::optional<uint64_t> source_numel = productDimsNoOverflow(source_domain_dims);
    if (!source_numel.has_value()) {
        return std::nullopt;
    }

    uint32_t node_idx = source_idx;
    while (true) {
        if (node_idx >= expr.nodes.size() || node_idx >= group.node_dims.size()) {
            throw runtime_error("Tiled logical-transpose dense-input load alias chain is out of range.");
        }

        const ExprNode& node = expr.nodes[node_idx];
        if (emittedScalarNodeValueDType(node) != frontier_dtype) {
            return std::nullopt;
        }

        if (node.op == ExprOp::INPUT) {
            if (node.input_slot >= expr.inputs.size() || expr.inputs[node.input_slot].kind != NamedInput::Kind::Tensor) {
                return std::nullopt;
            }
            const DataType input_dtype = requireNodeInputTensorDType(node);
            if (input_dtype != frontier_dtype) {
                return std::nullopt;
            }
            const std::optional<uint64_t> input_numel = productDimsNoOverflow(group.node_dims[node_idx]);
            if (!input_numel.has_value() || input_numel.value() != source_numel.value()) {
                return std::nullopt;
            }
            return TiledLogicalTransposeDenseInputLoad{node.input_slot, input_dtype};
        }

        if (!isContiguousMetadataAliasOp(node.op) || node.lhs == UINT32_MAX) {
            return std::nullopt;
        }

        const std::optional<uint64_t> alias_numel = productDimsNoOverflow(group.node_dims[node_idx]);
        const std::optional<uint64_t> child_numel =
            node.lhs < group.node_dims.size() ? productDimsNoOverflow(group.node_dims[node.lhs]) : std::nullopt;
        if (!alias_numel.has_value() || !child_numel.has_value() || alias_numel.value() != child_numel.value()) {
            return std::nullopt;
        }

        node_idx = node.lhs;
    }
}

uint32_t CudaSourceEmitter::tiledLogicalTransposeConsumerDensePackedInputLoadCount(const CompiledExecutionStage& stage,
                                                                                   const std::vector<SpecializedBroadcastGroup>& groups) {
    const std::optional<std::vector<uint32_t>> maybe_frontiers = tryFindTiledLogicalTransposeConsumerFrontiers(stage, groups);
    if (!maybe_frontiers.has_value() || maybe_frontiers->empty()) {
        return 0U;
    }

    const uint32_t logical_transpose_pack_scalars = CudaSourceEmitter::tiledLogicalTransposeConsumerPackScalars(stage, groups);
    if (logical_transpose_pack_scalars <= 1U) {
        return 0U;
    }

    const uint32_t logical_transpose_slot_bytes = CudaSourceEmitter::tiledLogicalTransposeConsumerSlotBytes(stage, groups);
    const SpecializedBroadcastGroup& group = groups[0];
    uint32_t count = 0U;
    for (uint32_t frontier_idx : maybe_frontiers.value()) {
        if (frontier_idx >= stage.expr.nodes.size()) {
            throw runtime_error("Tiled logical-transpose packed-load debug frontier index out of range.");
        }
        const ExprNode& frontier = stage.expr.nodes[frontier_idx];
        if (frontier.lhs == UINT32_MAX || frontier.lhs >= group.node_dims.size()) {
            throw runtime_error("Tiled logical-transpose packed-load debug frontier source out of range.");
        }
        const DataType frontier_dtype = requireNodeOutputDType(frontier);
        const std::optional<TiledLogicalTransposeDenseInputLoad> dense_input_load =
            tryResolveTiledLogicalTransposeDenseInputLoad(stage.expr, group, frontier.lhs, group.node_dims[frontier.lhs], frontier_dtype);
        if (dense_input_load.has_value() && scalarStorageTypeSizeBytes(dense_input_load->input_dtype) <= logical_transpose_slot_bytes) {
            ++count;
        }
    }
    return count;
}

static std::string emitTiledLogicalTransposeConsumerScalarValue(
    std::ostringstream& ss,
    const PhysicalExpression& expr,
    const SpecializedBroadcastGroup& group,
    uint32_t node_idx,
    const std::string& idx_expr,
    const std::vector<uint64_t>& domain_dims,
    const std::unordered_map<uint32_t, TiledLogicalTransposeFrontierValue>& frontier_values,
    const std::string& indent,
    bool use_uint32_index_math,
    uint32_t& counter) {
    if (node_idx >= expr.nodes.size() || node_idx >= group.node_dims.size()) {
        throw runtime_error("Tiled logical-transpose consumer emitter node index out of range.");
    }

    const ExprNode& n = expr.nodes[node_idx];
    const DataType emitted_dtype = emittedScalarNodeValueDType(n);
    const auto frontier_it = frontier_values.find(node_idx);
    if (frontier_it != frontier_values.end()) {
        return castScalarExpr(frontier_it->second.value_expr, frontier_it->second.dtype, emitted_dtype);
    }

    const std::string output_type = scalarStorageType(emitted_dtype);
    const std::string suffix = nextIndexMappedSuffix(node_idx, counter);
    const std::string var = "t" + std::to_string(node_idx) + suffix;

    double folded_constant = 0.0;
    if (n.op != ExprOp::INPUT && n.op != ExprOp::RUNTIME_SCALAR && n.op != ExprOp::TENSOR_RUNTIME_SCALAR &&
        tryGetEmitterConstantValue(expr, node_idx, folded_constant)) {
        ss << indent << "const " << output_type << " " << var << " = "
           << castScalarExpr(emitScalarFpLiteral(folded_constant), DataType::FP32, emitted_dtype) << ";\n";
        return var;
    }

    switch (n.op) {
        case ExprOp::INPUT: {
            const DataType input_tensor_dtype = requireNodeInputTensorDType(n);
            const std::string offset_var = "offset" + suffix;
            emitBroadcastOffsetForDomain(ss, group.node_dims[node_idx], domain_dims, idx_expr, offset_var, indent, use_uint32_index_math);
            // A non-dense runtime view maps its logical index through its visible strides.
            const std::string storage_idx =
                emitInputStorageIndexForLogicalIndex(
                    ss, &group, n.input_slot, offset_var, suffix + "_storage", indent, use_uint32_index_math);
            ss << indent << "const " << output_type << " " << var << " = "
               << castScalarExpr("in" + std::to_string(n.input_slot) + "[" + storage_idx + "]", input_tensor_dtype, emitted_dtype) << ";\n";
            return var;
        }
        case ExprOp::RUNTIME_SCALAR: {
            const DataType input_tensor_dtype = requireNodeInputTensorDType(n);
            ss << indent << "const " << output_type << " " << var << " = "
               << castScalarExpr("in" + std::to_string(n.input_slot), input_tensor_dtype, emitted_dtype) << ";\n";
            return var;
        }
        case ExprOp::TENSOR_RUNTIME_SCALAR: {
            const DataType input_tensor_dtype = requireNodeInputTensorDType(n);
            const std::string input_storage_type = scalarStorageType(input_tensor_dtype);
            const std::string input_expr = "reinterpret_cast<const " + input_storage_type + "*>(in" + std::to_string(n.input_slot) + ")[0]";
            ss << indent << "const " << output_type << " " << var << " = " << castScalarExpr(input_expr, input_tensor_dtype, emitted_dtype)
               << ";\n";
            return var;
        }
        case ExprOp::SCALAR_FP:
        case ExprOp::FILL: {
            ss << indent << "const " << output_type << " " << var << " = "
               << castScalarExpr(emitScalarFpLiteral(n.scalar_fp), DataType::FP32, emitted_dtype) << ";\n";
            return var;
        }
        case ExprOp::ROPE:
            throw runtime_error("Tiled logical-transpose consumer emission does not support RoPE in the same fused stage yet.");
        default:
            break;
    }

    if (n.op == ExprOp::TRANSPOSE) {
        // All logical transpose nodes in this lowered form must be represented
        // by shared-memory frontier values. If one is missing here, selecting
        // the scalar index-mapped fallback would create a dense column read.
        throw runtime_error("Unsupported logical transpose pattern for auto-swizzled fused emission.");
    }

    if (n.op == ExprOp::RESHAPE || n.op == ExprOp::STRIDED_VIEW || n.op == ExprOp::UNSQUEEZE || n.op == ExprOp::SQUEEZE) {
        // Same contract as emitIndexMappedScalarValue: collapse the caller's domain index into
        // this node's own dense index first (the node may be broadcast by its parent), then map
        // it to the child's index: unchanged for shape aliases, through the strides for a view.
        std::string own_idx = idx_expr;
        if (group.node_dims[node_idx] != domain_dims) {
            own_idx = "alias_idx" + suffix;
            emitBroadcastOffsetForDomain(ss, group.node_dims[node_idx], domain_dims, idx_expr, own_idx, indent, use_uint32_index_math);
        }
        const std::string child_idx = n.op == ExprOp::STRIDED_VIEW
                                          ? emitStridedViewSourceIndex(ss, n, group.node_dims[node_idx], own_idx, suffix, indent,
                                                                       use_uint32_index_math)
                                          : own_idx;
        const std::string value = emitTiledLogicalTransposeConsumerScalarValue(
            ss, expr, group, n.lhs, child_idx, group.node_dims[n.lhs], frontier_values, indent, use_uint32_index_math, counter);
        return castScalarExpr(value, emittedScalarNodeValueDType(expr.nodes[n.lhs]), emitted_dtype);
    }

    const DataType compute_dtype = requireNodeComputeDType(n);
    auto emit_child = [&](uint32_t child_idx, const std::vector<uint64_t>& child_domain_dims) -> std::string {
        const std::string value = emitTiledLogicalTransposeConsumerScalarValue(
            ss, expr, group, child_idx, idx_expr, child_domain_dims, frontier_values, indent, use_uint32_index_math, counter);
        return castScalarExpr(value, emittedScalarNodeValueDType(expr.nodes[child_idx]), compute_dtype);
    };

    std::string compute_expr;
    if (n.op == ExprOp::WHERE) {
        auto emit_child_as = [&](uint32_t child_idx, const std::vector<uint64_t>& child_domain_dims, DataType target_dtype) -> std::string {
            const std::string value = emitTiledLogicalTransposeConsumerScalarValue(
                ss, expr, group, child_idx, idx_expr, child_domain_dims, frontier_values, indent, use_uint32_index_math, counter);
            return castScalarExpr(value, emittedScalarNodeValueDType(expr.nodes[child_idx]), target_dtype);
        };
        compute_expr = emitWhereComputeExpr(emit_child_as(n.lhs, domain_dims, DataType::BOOLEAN),
                                            emit_child_as(n.rhs, domain_dims, compute_dtype),
                                            emit_child_as(n.aux, domain_dims, compute_dtype));
    } else if (Expression::isBinaryOp(n.op)) {
        compute_expr = emitBinaryComputeExpr(n.op, emit_child(n.lhs, domain_dims), emit_child(n.rhs, domain_dims), compute_dtype);
    } else if (Expression::isUnaryOp(n.op)) {
        compute_expr = emitUnaryComputeExpr(n.op, emit_child(n.lhs, domain_dims), compute_dtype);
    } else {
        throw runtime_error("Unsupported op in tiled logical-transpose consumer emitter: " + to_string(static_cast<int>(n.op)));
    }

    ss << indent << "const " << output_type << " " << var << " = " << castScalarExpr(compute_expr, compute_dtype, emitted_dtype) << ";\n";
    return var;
}

static std::string stridedViewForwardSourceIndexVar(uint32_t node_idx, const std::string& suffix) {
    return "sv_source_idx_" + std::to_string(node_idx) + suffix;
}

static std::string refWithSuffix(uint32_t idx, const std::string& suffix) { return "t" + to_string(idx) + suffix; }

static std::string stridedViewBackwardSafeSuffix(uint32_t node_idx) { return "_svbw" + std::to_string(node_idx); }

static std::string emitInputStorageIndexForLogicalIndex(std::ostringstream& ss,
                                                        const SpecializedBroadcastGroup* group,
                                                        uint32_t input_slot,
                                                        const std::string& idx_expr,
                                                        const std::string& suffix,
                                                        const std::string& indent,
                                                        bool use_uint32_index_math) {
    const std::optional<size_t> maybe_used_i = findUsedInputSlotIndex(group, input_slot);
    if (!maybe_used_i.has_value()) {
        return idx_expr;
    }

    const size_t used_i = maybe_used_i.value();
    const std::vector<uint64_t>& dims = usedInputVisibleDims(*group, used_i);
    const std::vector<uint64_t>& strides = group->used_input_visible_strides[used_i];
    if (dims.empty()) {
        return "0ULL";
    }
    // Packed visible strides (size-1 axes ignored): the logical index is the storage index.
    if (visibleStridesArePacked(dims, strides)) {
        return idx_expr;
    }

    std::vector<bool> needed(dims.size());
    for (size_t axis = 0; axis < dims.size(); ++axis) needed[axis] = strides[axis] != 0ULL;
    const std::vector<std::string> coords = emitRowMajorCoordinateDecode(
        ss, dims, idx_expr, "input_" + std::to_string(input_slot) + suffix, indent, use_uint32_index_math, needed);
    const std::string storage_idx = "input_storage_idx_" + std::to_string(input_slot) + suffix;
    ss << indent << "const " << emittedIndexType(use_uint32_index_math) << " " << storage_idx << " = "
       << stridedOffsetExpr(coords, strides, 0, use_uint32_index_math) << ";\n";
    return storage_idx;
}

static std::string emitScalarValueAtIndex(std::ostringstream& ss,
                                          const PhysicalExpression& expr,
                                          uint32_t node_idx,
                                          const std::string& idx_expr,
                                          const std::string& suffix,
                                          const std::string& indent,
                                          std::unordered_set<std::string>& emitted,
                                          const SpecializedBroadcastGroup* group = nullptr) {
    if (node_idx >= expr.nodes.size()) {
        throw runtime_error("Node index out of range while emitting index-aware scalar value.");
    }

    const ExprNode& n = expr.nodes[node_idx];
    const DataType emitted_dtype = emittedScalarNodeValueDType(n);
    const std::string output_type = scalarStorageType(emitted_dtype);

    double folded_constant = 0.0;
    if (n.op != ExprOp::INPUT && n.op != ExprOp::RUNTIME_SCALAR && n.op != ExprOp::TENSOR_RUNTIME_SCALAR &&
        tryGetEmitterConstantValue(expr, node_idx, folded_constant)) {
        return castScalarExpr(emitScalarFpLiteral(folded_constant), DataType::FP32, emitted_dtype);
    }

    const std::string key = std::to_string(node_idx) + suffix;
    const std::string var = refWithSuffix(node_idx, suffix);
    if (emitted.contains(key)) {
        return var;
    }

    auto mark_emitted = [&]() { emitted.insert(key); };

    switch (n.op) {
        case ExprOp::INPUT: {
            const DataType input_tensor_dtype = requireNodeInputTensorDType(n);
            const std::string storage_idx = emitInputStorageIndexForLogicalIndex(
                ss, group, n.input_slot, idx_expr, suffix, indent, /*use_uint32_index_math=*/false);
            ss << indent << "const " << output_type << " " << var << " = "
               << castScalarExpr("in" + std::to_string(n.input_slot) + "[" + storage_idx + "]", input_tensor_dtype, emitted_dtype)
               << ";\n";
            mark_emitted();
            return var;
        }
        case ExprOp::RUNTIME_SCALAR: {
            const DataType input_tensor_dtype = requireNodeInputTensorDType(n);
            ss << indent << "const " << output_type << " " << var << " = "
               << castScalarExpr("in" + std::to_string(n.input_slot), input_tensor_dtype, emitted_dtype) << ";\n";
            mark_emitted();
            return var;
        }
        case ExprOp::TENSOR_RUNTIME_SCALAR: {
            const DataType input_tensor_dtype = requireNodeInputTensorDType(n);
            const std::string input_storage_type = scalarStorageType(input_tensor_dtype);
            const std::string input_expr = "reinterpret_cast<const " + input_storage_type + "*>(in" + std::to_string(n.input_slot) + ")[0]";
            ss << indent << "const " << output_type << " " << var << " = " << castScalarExpr(input_expr, input_tensor_dtype, emitted_dtype)
               << ";\n";
            mark_emitted();
            return var;
        }
        case ExprOp::SCALAR_FP: {
            return castScalarExpr(emitScalarFpLiteral(n.scalar_fp), DataType::FP32, emitted_dtype);
        }
        case ExprOp::FILL: {
            return castScalarExpr(emitScalarFpLiteral(n.scalar_fp), DataType::FP32, emitted_dtype);
        }
        default:
            break;
    }

    if (n.op == ExprOp::RAGGED_VALUEWISE_EXTENT) {
        const std::string source_value = emitScalarValueAtIndex(ss, expr, n.lhs, idx_expr, suffix, indent, emitted, group);
        const DataType source_dtype = emittedScalarNodeValueDType(expr.nodes.at(n.lhs));
        ss << indent << "const " << output_type << " " << var << " = "
           << castScalarExpr(source_value, source_dtype, emitted_dtype) << ";\n";
        mark_emitted();
        return var;
    }
    if (n.op == ExprOp::STRIDED_VIEW_BACKWARD) {
        if (n.view_dims.empty() || n.view_dims.size() != n.view_strides.size() || n.fill_dims.empty()) {
            throw runtime_error("Nested strided_view_backward requires source/view dimensions and strides.");
        }
        const std::string in_view = "nested_svbw_in_view_" + std::to_string(node_idx) + suffix;
        const std::string residual = "nested_svbw_residual_" + std::to_string(node_idx) + suffix;
        const std::string view_linear = "nested_svbw_linear_" + std::to_string(node_idx) + suffix;
        ss << indent << "bool " << in_view << " = true;\n";
        ss << indent << "unsigned long long " << residual << " = static_cast<unsigned long long>(" << idx_expr << ");\n";
        if (n.view_element_offset != 0) {
            ss << indent << "if (" << residual << " < " << n.view_element_offset << "ULL) { " << in_view
               << " = false; } else { " << residual << " -= " << n.view_element_offset << "ULL; }\n";
        }
        const std::vector<size_t> storage_axis_order = stridedViewBackwardStorageAxisOrder(n);
        std::vector<std::string> coords(n.view_dims.size());
        for (size_t axis = 0; axis < n.view_dims.size(); ++axis) {
            const uint64_t dim = n.view_dims[axis];
            const uint64_t stride = n.view_strides[axis];
            if (dim == 0 || stride == 0) {
                throw runtime_error("Nested strided_view_backward dimensions and strides must be non-zero.");
            }
            coords[axis] = "nested_svbw_coord_" + std::to_string(node_idx) + "_" + std::to_string(axis) + suffix;
            ss << indent << "unsigned long long " << coords[axis] << " = 0ULL;\n";
        }
        for (size_t axis : storage_axis_order) {
            const uint64_t dim = n.view_dims[axis];
            const uint64_t stride = n.view_strides[axis];
            const std::string& coord = coords[axis];
            ss << indent << "if (" << in_view << ") {\n";
            ss << indent << "  " << coord << " = " << residual << " / " << stride << "ULL;\n";
            ss << indent << "  if (" << coord << " >= " << dim << "ULL) {\n";
            ss << indent << "    " << in_view << " = false;\n";
            ss << indent << "  } else {\n";
            ss << indent << "    " << residual << " -= " << coord << " * " << stride << "ULL;\n";
            ss << indent << "  }\n";
            ss << indent << "}\n";
        }
        ss << indent << "if (" << residual << " != 0ULL) " << in_view << " = false;\n";
        ss << indent << "unsigned long long " << view_linear << " = 0ULL;\n";
        for (size_t axis = 0; axis < n.view_dims.size(); ++axis) {
            ss << indent << view_linear << " = " << view_linear << " * " << n.view_dims[axis] << "ULL + " << coords[axis] << ";\n";
        }
        const std::string nested_suffix = suffix + "_svbw" + std::to_string(node_idx);
        const std::string grad_value = emitScalarValueAtIndex(
            ss, expr, n.lhs, view_linear, nested_suffix, indent, emitted, group);
        const DataType grad_dtype = emittedScalarNodeValueDType(expr.nodes.at(n.lhs));
        const std::string zero = castScalarExpr("0.0f", DataType::FP32, emitted_dtype);
        ss << indent << "const " << output_type << " " << var << " = " << in_view << " ? "
           << castScalarExpr(grad_value, grad_dtype, emitted_dtype) << " : " << zero << ";\n";
        mark_emitted();
        return var;
    }
    if (n.op == ExprOp::ROPE) {
        throw runtime_error("Nested RoPE inside an index-aware scalar source is not supported; materialize the inner RoPE first.");
    }
    if (n.op == ExprOp::TRANSPOSE) {
        throw runtime_error(
            "Nested transpose inside an index-aware scalar source is not supported; materialize the inner transpose first.");
    }

    if (n.op == ExprOp::TAKE_ALONG_AXIS) {
        if (group == nullptr) {
            throw runtime_error("take_along_axis index-aware emission requires specialized broadcast shape metadata.");
        }
        if (n.lhs >= group->node_dims.size() || n.rhs >= group->node_dims.size()) {
            throw runtime_error("take_along_axis child node index out of range for specialized broadcast metadata.");
        }
        const std::string selected_suffix = suffix + "_take_idx" + std::to_string(node_idx);
        const std::string selected_value = emitScalarValueAtIndex(ss, expr, n.rhs, idx_expr, selected_suffix, indent, emitted, group);
        const std::string source_idx = "take_source_idx_" + std::to_string(node_idx) + suffix;
        const std::string in_range = emitTakeAlongAxisSourceIndexForDomain(ss,
                                                                           n,
                                                                           group->node_dims[n.lhs],
                                                                           group->node_dims[n.rhs],
                                                                           idx_expr,
                                                                           selected_value,
                                                                           emittedScalarNodeValueDType(expr.nodes.at(n.rhs)),
                                                                           source_idx,
                                                                           indent,
                                                                           false);
        const std::string nested_suffix = suffix + "_take" + std::to_string(node_idx);
        const std::string source_value = emitScalarValueAtIndex(ss, expr, n.lhs, source_idx, nested_suffix, indent, emitted, group);
        const DataType source_dtype = emittedScalarNodeValueDType(expr.nodes.at(n.lhs));
        ss << indent << "const " << output_type << " " << var << " = " << in_range << " ? "
           << castScalarExpr(source_value, source_dtype, emitted_dtype) << " : " << castScalarExpr("0.0f", DataType::FP32, emitted_dtype)
           << ";\n";
        mark_emitted();
        return var;
    }

    auto child_value_as = [&](uint32_t child_idx, DataType target_dtype) -> std::string {
        const std::string child_value = emitScalarValueAtIndex(ss, expr, child_idx, idx_expr, suffix, indent, emitted, group);
        const DataType child_dtype = emittedScalarNodeValueDType(expr.nodes.at(child_idx));
        return castScalarExpr(child_value, child_dtype, target_dtype);
    };

    if (n.op == ExprOp::STRIDED_VIEW) {
        // Specialized groups carry physical dimensions after copy-dimension
        // markers have been resolved.  The original expression keeps the 0
        // marker for serialization and must not drive address arithmetic.
        const std::vector<uint64_t>& view_dims =
            group != nullptr && node_idx < group->node_dims.size() ? group->node_dims[node_idx] : n.view_dims;
        if (n.view_dims.empty() || n.view_dims.size() != n.view_strides.size() ||
            view_dims.size() != n.view_dims.size()) {
            throw runtime_error("Index-aware strided_view emission requires dimensions and strides with the same non-zero rank.");
        }

        // The flat packet emitters evaluate every lane of a packet, including tail lanes past
        // numel, so this decode keeps the outermost modulo: it keeps those lanes' (discarded)
        // reads inside the source view.
        const std::string source_idx = stridedViewForwardSourceIndexVar(node_idx, suffix);
        const std::string residual_var = "sv_residual_" + std::to_string(node_idx) + suffix;
        ss << indent << "unsigned long long " << residual_var << " = static_cast<unsigned long long>(" << idx_expr << ");\n";
        ss << indent << "unsigned long long " << source_idx << " = " << n.view_element_offset << "ULL;\n";
        for (int64_t axis = static_cast<int64_t>(view_dims.size()) - 1; axis >= 0; --axis) {
            const uint64_t dim = view_dims.at(static_cast<size_t>(axis));
            const uint64_t stride = n.view_strides.at(static_cast<size_t>(axis));
            if (dim == 0 || stride == 0) {
                throw runtime_error("Index-aware strided_view dimensions and strides must be non-zero.");
            }
            const std::string coord_var = "sv_coord_" + std::to_string(node_idx) + "_" + std::to_string(axis) + suffix;
            ss << indent << "const unsigned long long " << coord_var << " = " << residual_var << " % " << dim << "ULL;\n";
            ss << indent << residual_var << " /= " << dim << "ULL;\n";
            ss << indent << source_idx << " += " << coord_var << " * " << stride << "ULL;\n";
        }

        const std::string nested_suffix = suffix + "_sv" + std::to_string(node_idx);
        const std::string source_value = emitScalarValueAtIndex(ss, expr, n.lhs, source_idx, nested_suffix, indent, emitted, group);
        const DataType source_dtype = emittedScalarNodeValueDType(expr.nodes.at(n.lhs));
        ss << indent << "const " << output_type << " " << var << " = " << castScalarExpr(source_value, source_dtype, emitted_dtype)
           << ";\n";
        mark_emitted();
        return var;
    }

    if (n.op == ExprOp::RESHAPE || n.op == ExprOp::UNSQUEEZE || n.op == ExprOp::SQUEEZE) {
        const std::string source_value = emitScalarValueAtIndex(ss, expr, n.lhs, idx_expr, suffix, indent, emitted, group);
        const DataType source_dtype = emittedScalarNodeValueDType(expr.nodes.at(n.lhs));
        ss << indent << "const " << output_type << " " << var << " = " << castScalarExpr(source_value, source_dtype, emitted_dtype)
           << ";\n";
        mark_emitted();
        return var;
    }

    const DataType compute_dtype = requireNodeComputeDType(n);
    std::string compute_expr;
    if (n.op == ExprOp::WHERE) {
        const std::string lhs = child_value_as(n.lhs, DataType::BOOLEAN);
        const std::string rhs = child_value_as(n.rhs, compute_dtype);
        const std::string aux = child_value_as(n.aux, compute_dtype);
        compute_expr = emitWhereComputeExpr(lhs, rhs, aux);
    } else if (Expression::isBinaryOp(n.op)) {
        compute_expr =
            emitBinaryComputeExpr(n.op, child_value_as(n.lhs, compute_dtype), child_value_as(n.rhs, compute_dtype), compute_dtype);
    } else if (Expression::isUnaryOp(n.op)) {
        compute_expr = emitUnaryComputeExpr(n.op, child_value_as(n.lhs, compute_dtype), compute_dtype);
    } else {
        throw runtime_error("Unsupported op in index-aware scalar value emitter: " + to_string(static_cast<int>(n.op)));
    }

    ss << indent << "const " << output_type << " " << var << " = " << castScalarExpr(compute_expr, compute_dtype, emitted_dtype) << ";\n";
    mark_emitted();
    return var;
}

static void emitScalarStridedViewBackwardNode(std::ostringstream& ss,
                                              const PhysicalExpression& expr,
                                              uint32_t node_idx,
                                              const std::string& indent,
                                              const SpecializedBroadcastGroup* group = nullptr) {
    const ExprNode& n = expr.nodes.at(node_idx);
    if (n.lhs >= expr.nodes.size()) {
        throw runtime_error("strided_view_backward node has invalid lhs.");
    }

    const uint32_t grad_source_idx = n.lhs;

    const ExprNode& grad_source = expr.nodes.at(grad_source_idx);
    if (n.view_dims.empty() || n.view_dims.size() != n.view_strides.size()) {
        throw runtime_error("strided_view_backward requires view dimensions and strides with the same non-zero rank.");
    }
    if (n.fill_dims.empty()) {
        throw runtime_error("strided_view_backward requires non-empty source dimensions.");
    }

    const DataType emitted_dtype = emittedScalarNodeValueDType(n);
    const std::string output_type = scalarStorageType(emitted_dtype);

    ss << indent << "bool view_bw_in_view_" << node_idx << " = true;\n";
    ss << indent << "unsigned long long view_bw_residual_" << node_idx << " = static_cast<unsigned long long>(idx);\n";
    if (n.view_element_offset != 0) {
        ss << indent << "if (view_bw_residual_" << node_idx << " < " << n.view_element_offset << "ULL) {\n";
        ss << indent << "  view_bw_in_view_" << node_idx << " = false;\n";
        ss << indent << "} else {\n";
        ss << indent << "  view_bw_residual_" << node_idx << " -= " << n.view_element_offset << "ULL;\n";
        ss << indent << "}\n";
    }
    const std::vector<size_t> storage_axis_order = stridedViewBackwardStorageAxisOrder(n);
    std::vector<std::string> coords(n.view_dims.size());
    for (size_t axis = 0; axis < n.view_dims.size(); ++axis) {
        const uint64_t dim = n.view_dims.at(axis);
        const uint64_t stride = n.view_strides.at(axis);
        if (dim == 0 || stride == 0) {
            throw runtime_error("strided_view_backward dimensions and strides must be non-zero.");
        }
        coords[axis] = "coord_" + std::to_string(node_idx) + "_" + std::to_string(axis);
        ss << indent << "unsigned long long " << coords[axis] << " = 0ULL;\n";
    }
    for (size_t axis : storage_axis_order) {
        const uint64_t dim = n.view_dims.at(axis);
        const uint64_t stride = n.view_strides.at(axis);
        const std::string& coord = coords[axis];
        ss << indent << "if (view_bw_in_view_" << node_idx << ") {\n";
        ss << indent << "  " << coord << " = view_bw_residual_" << node_idx << " / " << stride << "ULL;\n";
        ss << indent << "  if (" << coord << " >= " << dim << "ULL) {\n";
        ss << indent << "    view_bw_in_view_" << node_idx << " = false;\n";
        ss << indent << "  } else {\n";
        ss << indent << "    view_bw_residual_" << node_idx << " -= " << coord << " * " << stride << "ULL;\n";
        ss << indent << "  }\n";
        ss << indent << "}\n";
    }
    ss << indent << "if (view_bw_residual_" << node_idx << " != 0ULL) view_bw_in_view_" << node_idx << " = false;\n";
    ss << indent << "unsigned long long view_bw_linear_" << node_idx << " = 0ULL;\n";
    for (size_t axis = 0; axis < n.view_dims.size(); ++axis) {
        ss << indent << "view_bw_linear_" << node_idx << " = view_bw_linear_" << node_idx << " * " << n.view_dims[axis] << "ULL + "
           << coords[axis] << ";\n";
    }
    std::string grad_value;
    if (group != nullptr) {
        if (group->node_dims.size() != expr.nodes.size()) {
            throw runtime_error("strided_view_backward requires per-node dimensions for broadcast-aware child emission.");
        }
        uint32_t counter = 0;
        grad_value = emitIndexMappedScalarValue(ss,
                                                expr,
                                                *group,
                                                grad_source_idx,
                                                "view_bw_linear_" + std::to_string(node_idx),
                                                n.view_dims,
                                                indent,
                                                /*use_uint32_index_math=*/false,
                                                stridedViewBackwardSafeSuffix(node_idx),
                                                counter);
    } else {
        std::unordered_set<std::string> emitted;
        const std::string suffix = stridedViewBackwardSafeSuffix(node_idx);
        grad_value = emitScalarValueAtIndex(
            ss, expr, grad_source_idx, "view_bw_linear_" + std::to_string(node_idx), suffix, indent, emitted, group);
    }
    const DataType grad_value_dtype = emittedScalarNodeValueDType(grad_source);

    const std::string cast_load = castScalarExpr(grad_value, grad_value_dtype, emitted_dtype);
    const std::string zero = castScalarExpr("0.0f", DataType::FP32, emitted_dtype);
    ss << indent << "const " << output_type << " t" << node_idx << " = view_bw_in_view_" << node_idx << " ? " << cast_load << " : " << zero
       << ";\n";
}

// Emits rope_s / rope_c for one RoPE pair: position (from seq_coord or position ids), frequency
// for the pair index (scaling kinds included), sincos, attention factor and inverse sign. Expects
// unsigned long long rope_pair_index and rope_rotary_dim in scope. Shared by the per-element
// emitter and the pair-per-thread kernel, so both compute identical angles.
static void emitRopeSinCos(std::ostringstream& ss,
                           const PhysicalExpression& expr,
                           uint32_t node_idx,
                           const std::string& seq_coord,
                           const std::string& effective_seq_len_expr,
                           const std::string& indent,
                           const SpecializedBroadcastGroup* group) {
    const ExprNode& n = expr.nodes.at(node_idx);
        if (n.rope_position_ids_node != UINT32_MAX) {
            std::unordered_set<std::string> position_emitted;
            const std::string position_value = emitScalarValueAtIndex(
                ss,
                expr,
                n.rope_position_ids_node,
                seq_coord,
                "_rope_position_" + std::to_string(node_idx),
                indent + "    ",
                position_emitted,
                group);
            const DataType position_dtype = emittedScalarNodeValueDType(expr.nodes.at(n.rope_position_ids_node));
            ss << indent << "    float rope_position = "
               << castScalarExpr(position_value, position_dtype, DataType::FP32) << ";\n";
        } else {
            ss << indent << "    float rope_position = static_cast<float>(" << seq_coord << ") + "
               << emitScalarFpLiteral(static_cast<double>(n.rope_position_offset)) << ";\n";
        }
        if (n.rope_scaling_kind == RotaryScalingKind::Linear) {
            ss << indent << "    rope_position = rope_position / " << emitScalarFpLiteral(n.rope_scaling_factor) << ";\n";
        }
        ss << indent << "    float rope_base = " << emitScalarFpLiteral(n.rope_base) << ";\n";
        ss << indent << "    float rope_freq = 0.0f;\n";
        if (n.rope_scaling_kind == RotaryScalingKind::DynamicNTK) {
            ss << indent << "    const float rope_seq_len = fmaxf(" << effective_seq_len_expr << " + "
               << emitScalarFpLiteral(static_cast<double>(std::max<int64_t>(0, n.rope_position_offset))) << ", 1.0f);\n";
            ss << indent << "    const float rope_original_max = "
               << emitScalarFpLiteral(static_cast<double>(n.rope_original_max_position_embeddings)) << ";\n";
            ss << indent << "    if (rope_seq_len > rope_original_max && rope_rotary_dim > 2ULL) {\n";
            ss << indent << "      const float rope_ntk_ratio = (" << emitScalarFpLiteral(n.rope_scaling_factor)
               << " * rope_seq_len / rope_original_max) - (" << emitScalarFpLiteral(n.rope_scaling_factor) << " - 1.0f);\n";
            ss << indent
               << "      rope_base = rope_base * powf(rope_ntk_ratio, static_cast<float>(rope_rotary_dim) / "
                  "static_cast<float>(rope_rotary_dim - 2ULL));\n";
            ss << indent << "    }\n";
            ss << indent
               << "    rope_freq = powf(rope_base, -2.0f * static_cast<float>(rope_pair_index) / static_cast<float>(rope_rotary_dim));\n";
        } else if (n.rope_scaling_kind == RotaryScalingKind::Yarn) {
            ss << indent
               << "    const float rope_pos_freq = powf(rope_base, 2.0f * static_cast<float>(rope_pair_index) / "
                  "static_cast<float>(rope_rotary_dim));\n";
            ss << indent << "    const float rope_inv_freq_extrap = 1.0f / rope_pos_freq;\n";
            ss << indent << "    const float rope_inv_freq_interp = 1.0f / (" << emitScalarFpLiteral(n.rope_scaling_factor)
               << " * rope_pos_freq);\n";
            ss << indent << "    const float rope_original_max = "
               << emitScalarFpLiteral(static_cast<double>(n.rope_original_max_position_embeddings)) << ";\n";
            ss << indent << "    const float rope_log_base = logf(rope_base);\n";
            ss << indent << "    float rope_yarn_low = (static_cast<float>(rope_rotary_dim) * logf(rope_original_max / ("
               << emitScalarFpLiteral(n.rope_yarn_beta_fast) << " * 6.2831853071795864769f))) / (2.0f * rope_log_base);\n";
            ss << indent << "    float rope_yarn_high = (static_cast<float>(rope_rotary_dim) * logf(rope_original_max / ("
               << emitScalarFpLiteral(n.rope_yarn_beta_slow) << " * 6.2831853071795864769f))) / (2.0f * rope_log_base);\n";
            ss << indent << "    rope_yarn_low = fmaxf(floorf(rope_yarn_low), 0.0f);\n";
            ss << indent << "    rope_yarn_high = fminf(ceilf(rope_yarn_high), static_cast<float>(rope_rotary_dim - 1ULL));\n";
            ss << indent << "    if (rope_yarn_low == rope_yarn_high) { rope_yarn_high += 0.001f; }\n";
            ss << indent
               << "    const float rope_yarn_ramp = fminf(fmaxf((static_cast<float>(rope_pair_index) - rope_yarn_low) / (rope_yarn_high - "
                  "rope_yarn_low), 0.0f), 1.0f);\n";
            ss << indent << "    rope_freq = rope_inv_freq_interp * rope_yarn_ramp + rope_inv_freq_extrap * (1.0f - rope_yarn_ramp);\n";
        } else if (n.rope_scaling_kind == RotaryScalingKind::LongRope) {
            if (n.rope_long_rope_short_factors.empty() || n.rope_long_rope_long_factors.empty()) {
                throw runtime_error("LongRoPE code generation requires non-empty short/long factor lists.");
            }
            ss << indent << "    const float rope_seq_len = fmaxf(" << effective_seq_len_expr << " + "
               << emitScalarFpLiteral(static_cast<double>(std::max<int64_t>(0, n.rope_position_offset))) << ", 1.0f);\n";
            ss << indent << "    const float rope_original_max = "
               << emitScalarFpLiteral(static_cast<double>(n.rope_original_max_position_embeddings)) << ";\n";
            // static: one copy in device memory. A non-static local array indexed by the runtime pair
            // index is rebuilt in local memory by every thread.
            ss << indent << "    static const float rope_short_factors[] = " << emitFloatArrayLiteral(n.rope_long_rope_short_factors) << ";\n";
            ss << indent << "    static const float rope_long_factors[] = " << emitFloatArrayLiteral(n.rope_long_rope_long_factors) << ";\n";
            ss << indent
               << "    const float rope_ext_factor = rope_seq_len > rope_original_max ? rope_long_factors[rope_pair_index] : "
                  "rope_short_factors[rope_pair_index];\n";
            ss << indent
               << "    rope_freq = 1.0f / (rope_ext_factor * powf(rope_base, 2.0f * static_cast<float>(rope_pair_index) / "
                  "static_cast<float>(rope_rotary_dim)));\n";
        } else if (n.rope_scaling_kind == RotaryScalingKind::Llama3) {
            ss << indent
               << "    const float rope_inv_freq = powf(rope_base, -2.0f * static_cast<float>(rope_pair_index) / "
                  "static_cast<float>(rope_rotary_dim));\n";
            ss << indent << "    const float rope_wavelen = 6.2831853071795864769f / rope_inv_freq;\n";
            ss << indent << "    const float rope_original_max = "
               << emitScalarFpLiteral(static_cast<double>(n.rope_original_max_position_embeddings)) << ";\n";
            ss << indent << "    const float rope_low_freq_wavelen = rope_original_max / "
               << emitScalarFpLiteral(n.rope_llama3_low_freq_factor) << ";\n";
            ss << indent << "    const float rope_high_freq_wavelen = rope_original_max / "
               << emitScalarFpLiteral(n.rope_llama3_high_freq_factor) << ";\n";
            ss << indent << "    if (rope_wavelen > rope_low_freq_wavelen) {\n";
            ss << indent << "      rope_freq = rope_inv_freq / " << emitScalarFpLiteral(n.rope_scaling_factor) << ";\n";
            ss << indent << "    } else if (rope_wavelen < rope_high_freq_wavelen) {\n";
            ss << indent << "      rope_freq = rope_inv_freq;\n";
            ss << indent << "    } else {\n";
            ss << indent << "      const float rope_smooth = (rope_original_max / rope_wavelen - "
               << emitScalarFpLiteral(n.rope_llama3_low_freq_factor) << ") / (" << emitScalarFpLiteral(n.rope_llama3_high_freq_factor)
               << " - " << emitScalarFpLiteral(n.rope_llama3_low_freq_factor) << ");\n";
            ss << indent << "      rope_freq = (1.0f - rope_smooth) * (rope_inv_freq / " << emitScalarFpLiteral(n.rope_scaling_factor)
               << ") + rope_smooth * rope_inv_freq;\n";
            ss << indent << "    }\n";
        } else {
            // No frequency scaling (None / Linear): rope_freq = base^(-2i/R) depends only on the pair
            // index and emit-time constants. Read it from a table computed in double and rounded
            // once (within 0.5 ulp; powf is within ~2 ulp) instead of a powf per pair.
            uint64_t rotary_dim = n.rope_rotary_dim;
            if (rotary_dim == 0 && group != nullptr && n.rope_head_dim_axis < group->output_dims.size()) {
                rotary_dim = group->output_dims[n.rope_head_dim_axis];
            }
            if (rotary_dim >= 2 && rotary_dim % 2 == 0 && rotary_dim <= 4096) {
                std::vector<double> inv_freq(rotary_dim / 2);
                for (uint64_t pair = 0; pair < rotary_dim / 2; ++pair) {
                    inv_freq[pair] = static_cast<double>(static_cast<float>(
                        std::pow(n.rope_base, -2.0 * static_cast<double>(pair) / static_cast<double>(rotary_dim))));
                }
                ss << indent << "    static const float rope_inv_freq[] = " << emitFloatArrayLiteral(inv_freq) << ";\n";
                ss << indent << "    rope_freq = rope_inv_freq[rope_pair_index];\n";
            } else {
                ss << indent
                   << "    rope_freq = powf(rope_base, -2.0f * static_cast<float>(rope_pair_index) / static_cast<float>(rope_rotary_dim));\n";
            }
        }
        ss << indent << "    const float rope_theta = rope_position * rope_freq;\n";
        // One range reduction for both: sincosf returns the same values as sinf and cosf.
        ss << indent << "    float rope_s;\n";
        ss << indent << "    float rope_c;\n";
        ss << indent << "    sincosf(rope_theta, &rope_s, &rope_c);\n";
        if (n.rope_attention_factor != 1.0) {
            ss << indent << "    rope_s *= " << emitScalarFpLiteral(n.rope_attention_factor) << ";\n";
            ss << indent << "    rope_c *= " << emitScalarFpLiteral(n.rope_attention_factor) << ";\n";
        }
        if (n.rope_inverse) {
            ss << indent << "    rope_s = -rope_s;\n";
        }
}

static void emitScalarNode(std::ostringstream& ss,
                           const PhysicalExpression& expr,
                           uint32_t node_idx,
                           bool broadcast_support,
                           const std::string& indent,
                           const SpecializedBroadcastGroup* group = nullptr) {
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
        case ExprOp::STRIDED_VIEW_BACKWARD:
            emitScalarStridedViewBackwardNode(ss, expr, node_idx, indent, group);
            return;

        case ExprOp::RAGGED_VALUEWISE_EXTENT:
            emitScalarAliasNode(ss, expr, node_idx, n.lhs, indent);
            return;

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

    if (n.op == ExprOp::ROPE) {
        if (!broadcast_support) {
            throw runtime_error("RoPE fused emission requires specialized index-aware broadcast launch.");
        }
        if (n.rope_base <= 0.0 || n.rope_scaling_factor <= 0.0) {
            throw runtime_error("Invalid RoPE base/scaling factor during code generation.");
        }
        // c_<axis> are output-axis coordinates; they name the RoPE head/sequence axes only when
        // the RoPE result has the output's rank (leading broadcast dims of size 1 are fine).
        if (group != nullptr && node_idx < group->node_dims.size() && !group->node_dims[node_idx].empty() &&
            group->node_dims[node_idx].size() != group->output_dims.size()) {
            throw runtime_error("RoPE fused emission requires the RoPE result rank to match the fused stage output rank.");
        }
        (void)requireNodeComputeDType(n);
        const std::string current_value = emitResolvedScalarValueExpr(expr, n.lhs, DataType::FP32);
        const std::string dim_coord = ropeAxisCoordVar(n.rope_head_dim_axis);
        const std::string seq_coord = ropeAxisCoordVar(n.rope_sequence_axis);
        const std::string head_dim_extent = ropeAxisDimVar(n.rope_head_dim_axis);
        const std::string seq_extent = ropeAxisDimVar(n.rope_sequence_axis);
        const std::string effective_seq_len_expr =
            n.rope_effective_sequence_length_node == UINT32_MAX
                ? std::string("static_cast<float>(") + seq_extent + ")"
                : emitResolvedScalarValueExpr(expr, n.rope_effective_sequence_length_node, DataType::FP32);
        const std::string rotary_dim_expr = n.rope_rotary_dim == 0 ? head_dim_extent : emitUnsignedLiteral(n.rope_rotary_dim, true);
        const std::string suffix = "_rope_peer_" + std::to_string(node_idx);

        ss << indent << output_type << " t" << node_idx << ";\n";
        ss << indent << "{\n";
        ss << indent << "  const unsigned long long rope_dim_coord = static_cast<unsigned long long>(" << dim_coord << ");\n";
        ss << indent << "  const unsigned long long rope_rotary_dim = static_cast<unsigned long long>(" << rotary_dim_expr << ");\n";
        ss << indent << "  if (rope_dim_coord >= rope_rotary_dim) {\n";
        ss << indent << "    t" << node_idx << " = " << castScalarExpr(current_value, DataType::FP32, emitted_dtype) << ";\n";
        ss << indent << "  } else {\n";
        ss << indent << "    const bool rope_interleaved = " << (n.rope_interleaved ? "true" : "false") << ";\n";
        ss << indent << "    const unsigned long long rope_half_dim = rope_rotary_dim / 2ULL;\n";
        ss << indent
           << "    const bool rope_first_lane = rope_interleaved ? ((rope_dim_coord & 1ULL) == 0ULL) : (rope_dim_coord < rope_half_dim);\n";
        ss << indent
           << "    const unsigned long long rope_pair_index = rope_interleaved ? (rope_dim_coord >> 1ULL) : (rope_dim_coord < "
              "rope_half_dim ? rope_dim_coord : rope_dim_coord - rope_half_dim);\n";
        ss << indent << "    const unsigned long long rope_peer_abs_delta = rope_interleaved ? 1ULL : rope_half_dim;\n";
        ss << indent
           << "    const unsigned long long rope_peer_idx = rope_first_lane ? (static_cast<unsigned long long>(idx) + rope_peer_abs_delta) "
              ": (static_cast<unsigned long long>(idx) - rope_peer_abs_delta);\n";

        // The partner lane differs from this lane only in the head-dim coordinate, which is the
        // innermost (stride-1) output axis. Each input's partner element is therefore at that
        // input's own broadcast offset moved by delta * (its innermost stride): the stride is 0
        // for an input broadcast along the head dim and the row pitch of a non-dense view is
        // irrelevant. Reading inputs at the flat output index rope_peer_idx would ignore both.
        const auto peer_input_value = [&](uint32_t input_slot) -> std::string {
            if (group == nullptr) {
                throw runtime_error("RoPE fused emission requires the broadcast group to address partner inputs.");
            }
            const std::optional<size_t> used_i = findUsedInputSlotIndex(group, input_slot);
            if (!used_i.has_value()) {
                throw runtime_error("RoPE partner subgraph references an input outside its broadcast group.");
            }
            const std::string offset = "in" + std::to_string(input_slot) + "_offset";
            const uint64_t inner_stride = specializedBroadcastInnermostInputStride(*group, used_i.value());
            std::string peer_offset = offset;
            if (inner_stride != 0ULL) {
                const std::string step =
                    inner_stride == 1ULL ? std::string("rope_peer_abs_delta") : "rope_peer_abs_delta * " + std::to_string(inner_stride) + "ULL";
                peer_offset = "(rope_first_lane ? (static_cast<unsigned long long>(" + offset + ") + " + step +
                              ") : (static_cast<unsigned long long>(" + offset + ") - " + step + "))";
            }
            return "in" + std::to_string(input_slot) + "[" + peer_offset + "]";
        };

        std::unordered_set<uint32_t> rope_required;
        collectRequiredNodes(expr, n.lhs, rope_required);
        for (uint32_t dep_idx = 0; dep_idx < node_idx; ++dep_idx) {
            if (!rope_required.contains(dep_idx) || !shouldEmitScalarNodeDefinition(expr, dep_idx)) {
                continue;
            }
            emitScalarNodeSuffixed(ss, expr, dep_idx, "rope_peer_idx", suffix, indent + "    ", "", 0, peer_input_value);
        }

        emitRopeSinCos(ss, expr, node_idx, seq_coord, effective_seq_len_expr, indent, group);
        const std::string peer_value = emitResolvedScalarValueExprSuffixed(expr, n.lhs, DataType::FP32, suffix);
        ss << indent << "    const float rope_current = " << current_value << ";\n";
        ss << indent << "    const float rope_peer = " << peer_value << ";\n";
        ss << indent
           << "    const float rope_out = rope_first_lane ? (rope_current * rope_c - rope_peer * rope_s) : (rope_peer * rope_s + "
              "rope_current * rope_c);\n";
        ss << indent << "    t" << node_idx << " = " << castScalarExpr("rope_out", DataType::FP32, emitted_dtype) << ";\n";
        ss << indent << "  }\n";
        ss << indent << "}\n";
        return;
    }

    const DataType compute_dtype = requireNodeComputeDType(n);

    auto child_value = [&](uint32_t child_idx) -> std::string {
        if (child_idx >= expr.nodes.size()) {
            throw runtime_error("Child node index out of range in fused stage emitter.");
        }
        return emitResolvedScalarValueExpr(expr, child_idx, compute_dtype);
    };

    if (n.op == ExprOp::STRIDED_VIEW || n.op == ExprOp::TAKE_ALONG_AXIS) {
        std::unordered_set<std::string> emitted;
        const std::string suffix = (n.op == ExprOp::TAKE_ALONG_AXIS ? "_takenode" : "_svnode") + std::to_string(node_idx);
        const std::string value = emitScalarValueAtIndex(ss, expr, node_idx, "idx", suffix, indent, emitted, group);
        ss << indent << "const " << output_type << " t" << node_idx << " = "
           << castScalarExpr(value, emittedScalarNodeValueDType(n), emitted_dtype) << ";\n";
        return;
    }

    if (n.op == ExprOp::RESHAPE || n.op == ExprOp::UNSQUEEZE || n.op == ExprOp::SQUEEZE) {
        emitScalarAliasNode(ss, expr, node_idx, n.lhs, indent);
        return;
    }

    if (n.op == ExprOp::WHERE) {
        auto child_value_as = [&](uint32_t child_idx, DataType target_dtype) -> std::string {
            if (child_idx >= expr.nodes.size()) {
                throw runtime_error("Child node index out of range in where fused stage emitter.");
            }
            return emitResolvedScalarValueExpr(expr, child_idx, target_dtype);
        };
        const std::string compute_expr = emitWhereComputeExpr(
            child_value_as(n.lhs, DataType::BOOLEAN), child_value_as(n.rhs, compute_dtype), child_value_as(n.aux, compute_dtype));
        ss << indent << "const " << output_type << " t" << node_idx << " = " << castScalarExpr(compute_expr, compute_dtype, emitted_dtype)
           << ";\n";
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
                                   uint32_t flat_elements_per_thread = 0,
                                   const std::function<std::string(uint32_t)>& input_slot_value = {}) {
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
        case ExprOp::STRIDED_VIEW_BACKWARD:
            throw runtime_error("strided_view_backward is index-aware and is only supported by the scalar flat fused emitter.");

        case ExprOp::RAGGED_VALUEWISE_EXTENT:
            emitScalarAliasNodeSuffixed(ss, expr, node_idx, n.lhs, suffix, indent);
            return;

        case ExprOp::INPUT: {
            const DataType input_tensor_dtype = requireNodeInputTensorDType(n);
            std::string input_expr;
            if (input_slot_value) {
                input_expr = input_slot_value(n.input_slot);
            } else {
                input_expr = flat_chunk_lane_expr.empty()
                                 ? ("in" + std::to_string(n.input_slot) + "[" + idx_expr + "]")
                                 : emitChunkLaneReadExpr(
                                       "in" + std::to_string(n.input_slot) + "_chunk",
                                       chunkScalarTypeForBytes(dataTypeStorageBytes(input_tensor_dtype) * flat_elements_per_thread),
                                       input_tensor_dtype,
                                       flat_chunk_lane_expr,
                                       "in" + std::to_string(n.input_slot) + "_chunk_data",
                                       flat_elements_per_thread);
            }
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

    if (n.op == ExprOp::ROPE) {
        throw runtime_error("Nested RoPE inside a suffixed peer expression is not supported; materialize the inner RoPE first.");
    }

    const DataType compute_dtype = requireNodeComputeDType(n);

    auto child_value = [&](uint32_t child_idx) -> std::string {
        if (child_idx >= expr.nodes.size()) {
            throw runtime_error("Child node index out of range in suffixed fused stage emitter.");
        }
        return emitResolvedScalarValueExprSuffixed(expr, child_idx, compute_dtype, suffix);
    };

    if (n.op == ExprOp::STRIDED_VIEW || n.op == ExprOp::TAKE_ALONG_AXIS) {
        std::unordered_set<std::string> emitted;
        const std::string nested_suffix = suffix + (n.op == ExprOp::TAKE_ALONG_AXIS ? "_takenode" : "_svnode") + std::to_string(node_idx);
        const std::string value = emitScalarValueAtIndex(ss, expr, node_idx, idx_expr, nested_suffix, indent, emitted);
        ss << indent << "const " << output_type << " " << refWithSuffix(node_idx, suffix) << " = "
           << castScalarExpr(value, emittedScalarNodeValueDType(n), emitted_dtype) << ";\n";
        return;
    }

    if (n.op == ExprOp::RESHAPE || n.op == ExprOp::UNSQUEEZE || n.op == ExprOp::SQUEEZE) {
        emitScalarAliasNodeSuffixed(ss, expr, node_idx, n.lhs, suffix, indent);
        return;
    }

    if (n.op == ExprOp::WHERE) {
        auto child_value_as = [&](uint32_t child_idx, DataType target_dtype) -> std::string {
            if (child_idx >= expr.nodes.size()) {
                throw runtime_error("Child node index out of range in suffixed where fused stage emitter.");
            }
            return emitResolvedScalarValueExprSuffixed(expr, child_idx, target_dtype, suffix);
        };
        const std::string compute_expr = emitWhereComputeExpr(
            child_value_as(n.lhs, DataType::BOOLEAN), child_value_as(n.rhs, compute_dtype), child_value_as(n.aux, compute_dtype));
        ss << indent << "const " << output_type << " " << refWithSuffix(node_idx, suffix) << " = "
           << castScalarExpr(compute_expr, compute_dtype, emitted_dtype) << ";\n";
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

static std::string vector_compute_conversion(const std::string& storage_dtype_vector,
                                             const std::string& variable,
                                             DataType compute_dtype) {
    if (compute_dtype != DataType::FP16 && compute_dtype != DataType::BF16) {
        throw runtime_error("Unsupported vector compute dtype in vector_compute_conversion: " +
                            TensorDescriptor::getElementTypeName(compute_dtype));
    }

    if (storage_dtype_vector == "half2") {
        if (compute_dtype == DataType::FP16) {
            return variable;
        }
        return "__float22bfloat162_rn(__half22float2(" + variable + "))";
    } else if (storage_dtype_vector == "__nv_bfloat162") {
        if (compute_dtype == DataType::BF16) {
            return variable;
        }
        return "__float22half2_rn(__bfloat1622float2(" + variable + "))";
    } else if (storage_dtype_vector == "__nv_fp8x2_e4m3" || storage_dtype_vector == "__nv_fp8x2_e5m2") {
        if (compute_dtype == DataType::FP16) {
            return "static_cast<__half2>(" + variable + ")";
        }
        // CUDA FP8 x2 types expose an exact float2 conversion.  Widen through
        // float2 and then round once into BF16 instead of routing through half2,
        // which would reintroduce FP16's narrow exponent range before arithmetic.
        return "__float22bfloat162_rn(static_cast<float2>(" + variable + "))";
    }
    throw runtime_error("Unsupported vector storage dtype in vector_compute_conversion: " + storage_dtype_vector);
}

static std::string vector_storage_conversion(const std::string& storage_dtype_vector,
                                             const std::string& variable,
                                             DataType compute_dtype) {
    if (compute_dtype != DataType::FP16 && compute_dtype != DataType::BF16) {
        throw runtime_error("Unsupported vector compute dtype in vector_storage_conversion: " +
                            TensorDescriptor::getElementTypeName(compute_dtype));
    }

    if (storage_dtype_vector == "half2") {
        if (compute_dtype == DataType::FP16) {
            return variable;
        }
        return "__float22half2_rn(__bfloat1622float2(" + variable + "))";
    } else if (storage_dtype_vector == "__nv_bfloat162") {
        if (compute_dtype == DataType::BF16) {
            return variable;
        }
        return "__float22bfloat162_rn(__half22float2(" + variable + "))";
    } else if (storage_dtype_vector == "__nv_fp8x2_e4m3") {
        if (compute_dtype == DataType::BF16) {
            return "thor_to_fp8x2_e4m3_satfinite(__bfloat1622float2(" + variable + "))";
        }
        return "thor_half2_to_fp8x2_e4m3_satfinite(" + variable + ")";
    } else if (storage_dtype_vector == "__nv_fp8x2_e5m2") {
        if (compute_dtype == DataType::BF16) {
            return "thor_to_fp8x2_e5m2_satfinite(__bfloat1622float2(" + variable + "))";
        }
        return "thor_half2_to_fp8x2_e5m2_satfinite(" + variable + ")";
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
        return "static_cast<float2>(" + variable + ")";
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
        return "thor_to_fp8x2_e4m3_satfinite(" + variable + ")";
    } else if (storage_dtype_vector == "__nv_fp8x2_e5m2") {
        return "thor_to_fp8x2_e5m2_satfinite(" + variable + ")";
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

static DataType defaultVectorizedComputeScalarDTypeForStorage(DataType storage_dtype) {
    const DataType compute_dtype = defaultComputeDType(storage_dtype);
    if (compute_dtype == DataType::FP16 || compute_dtype == DataType::BF16) {
        return compute_dtype;
    }
    throw runtime_error("Unsupported storage dtype in defaultVectorizedComputeScalarDTypeForStorage.");
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

static std::string emitVector2RuntimeScalarValue(const PhysicalExpression& expr, const ExprNode& node, DataType compute_scalar_dtype) {
    const DataType input_dtype = requireNodeInputTensorDType(node);
    const DataType output_dtype = requireNodeOutputDType(node);
    if (compute_scalar_dtype != DataType::FP16 && compute_scalar_dtype != DataType::BF16) {
        throw runtime_error("Unsupported vector runtime-scalar compute dtype.");
    }

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

// The raw FP8 pair at base[offset0], base[offset0 + 1]: one 2-byte load when the address is pair
// aligned (element-aligned storage aliases may not be), else the two bytes. The caller converts
// the selected pair once instead of emitting a decode in each branch.
static std::string fp8TransposePairRawExpr(DataType fp8_dtype,
                                           const std::string& pair_type,
                                           const std::string& base,
                                           const std::string& offset0) {
    const std::string from_lanes = fp8_dtype == DataType::FP8_E4M3 ? "thor_fp8x2_e4m3_from_lanes" : "thor_fp8x2_e5m2_from_lanes";
    return "(((reinterpret_cast<unsigned long long>(" + base + " + " + offset0 + ") % sizeof(" + pair_type +
           ")) == 0ULL) ? reinterpret_cast<const " + pair_type + "*>(" + base + " + " + offset0 + ")[0] : " + from_lanes + "(" + base +
           "[" + offset0 + "], " + base + "[" + offset0 + " + 1]))";
}

static std::string emitVector2BroadcastNativeLoad(const std::string& storage_dtype_vector,
                                                  const std::string& base_ptr,
                                                  const std::string& scalar_offset0,
                                                  DataType compute_dtype) {
    const std::string vec_expr = "reinterpret_cast<const " + storage_dtype_vector + "*>(" + base_ptr + " + " + scalar_offset0 + ")[0]";
    return vector_compute_conversion(storage_dtype_vector, vec_expr, compute_dtype);
}

static std::string emitVector2Add(const std::string& a, const std::string& b) { return "__hadd2(" + a + ", " + b + ")"; }
static std::string emitVector2Sub(const std::string& a, const std::string& b) { return "__hsub2(" + a + ", " + b + ")"; }
static std::string emitVector2Mul(const std::string& a, const std::string& b) { return "__hmul2(" + a + ", " + b + ")"; }
static std::string emitVector2Div(const std::string& a, const std::string& b) { return "__h2div(" + a + ", " + b + ")"; }
static std::string emitVector2Neg(const std::string& x, DataType dtype) { return "__hneg2(" + x + ")"; }
static std::string emitVector2Abs(const std::string& x, DataType dtype) { return "__habs2(" + x + ")"; }
static std::string emitVector2Exp(const std::string& x, DataType dtype) { return "h2exp(" + x + ")"; }
static std::string emitVector2Exp2(const std::string& x, DataType dtype) { return "h2exp2(" + x + ")"; }
static std::string emitVector2Exp10(const std::string& x, DataType dtype) { return "h2exp10(" + x + ")"; }
static std::string emitVector2Ln(const std::string& x, DataType dtype) { return "h2log(" + x + ")"; }
static std::string emitVector2Log2(const std::string& x, DataType dtype) { return "h2log2(" + x + ")"; }
static std::string emitVector2Log10(const std::string& x, DataType dtype) { return "h2log10(" + x + ")"; }
static std::string emitVector2Sqrt(const std::string& x, DataType dtype) { return "h2sqrt(" + x + ")"; }
static std::string emitVector2SpecialUnary(const std::string& fn, const std::string& x, DataType dtype) {
    if (dtype == DataType::BF16) {
        return "__floats2bfloat162_rn(" + fn + "(float(" + x + ".x)), " + fn + "(float(" + x + ".y)))";
    }
    return "__floats2half2_rn(" + fn + "(float(half(" + x + ".x))), " + fn + "(float(half(" + x + ".y))))";
}

static std::string emitVector2Half2Tanh(const std::string& x, DataType dtype) {
    if (dtype == DataType::BF16) {
        return "__floats2bfloat162_rn(tanhf(float(" + x + ".x)), tanhf(float(" + x + ".y)))";
    }

    return "h2tanh(" + x + ")";
}

static std::string emitVector2Ceil(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("ceilf", x, dtype); }
static std::string emitVector2Floor(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("floorf", x, dtype); }
static std::string emitVector2Round(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("roundf", x, dtype); }
static std::string emitVector2Trunc(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("truncf", x, dtype); }
static std::string emitVector2Sin(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("sinf", x, dtype); }
static std::string emitVector2Cos(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("cosf", x, dtype); }
static std::string emitVector2Tan(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("tanf", x, dtype); }
static std::string emitVector2Asin(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("asinf", x, dtype); }
static std::string emitVector2Acos(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("acosf", x, dtype); }
static std::string emitVector2Atan(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("atanf", x, dtype); }
static std::string emitVector2Sinh(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("sinhf", x, dtype); }
static std::string emitVector2Cosh(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("coshf", x, dtype); }
static std::string emitVector2Asinh(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("asinhf", x, dtype); }
static std::string emitVector2Acosh(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("acoshf", x, dtype); }
static std::string emitVector2Atanh(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("atanhf", x, dtype); }
static std::string emitVector2Erf(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("erff", x, dtype); }
static std::string emitVector2Erfc(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("erfcf", x, dtype); }
static std::string emitVector2Erfcx(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("erfcxf", x, dtype); }
static std::string emitVector2Erfinv(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("erfinvf", x, dtype); }
static std::string emitVector2Erfcinv(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("erfcinvf", x, dtype); }
static std::string emitVector2Tgamma(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("tgammaf", x, dtype); }
static std::string emitVector2Lgamma(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("lgammaf", x, dtype); }
static std::string emitVector2Digamma(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("thor_digammaf", x, dtype); }
static std::string emitVector2Expm1(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("expm1f", x, dtype); }
static std::string emitVector2Log1p(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("log1pf", x, dtype); }
static std::string emitVector2Tanh(const std::string& x, DataType dtype) { return emitVector2Half2Tanh(x, dtype); }
static std::string emitVector2Normcdf(const std::string& x, DataType dtype) { return emitVector2SpecialUnary("normcdff", x, dtype); }
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
                                                const std::string& variable1,
                                                DataType compute_dtype) {
    if (storage_dtype == "half") {
        if (compute_dtype == DataType::FP16) {
            return "__halves2half2(" + variable0 + ", " + variable1 + ")";
        }
        return "__floats2bfloat162_rn(float(" + variable0 + "), float(" + variable1 + "))";
    } else if (storage_dtype == "__nv_bfloat16") {
        if (compute_dtype == DataType::BF16) {
            return "__halves2bfloat162(" + variable0 + ", " + variable1 + ")";
        }
        return "__floats2half2_rn(float(" + variable0 + "), float(" + variable1 + "))";
    } else if (storage_dtype == "__nv_fp8_e4m3" || storage_dtype == "__nv_fp8_e5m2") {
        if (compute_dtype == DataType::BF16) {
            return "__floats2bfloat162_rn(float(" + variable0 + "), float(" + variable1 + "))";
        }
        return "__halves2half2(static_cast<half>(" + variable0 + "), static_cast<half>(" + variable1 + "))";
    }

    throw runtime_error("Unsupported scalar storage dtype in emitVector2BroadcastPackLoad: " + storage_dtype);
}

static void emitVector2NodeDefinitionsForSuffix(std::ostringstream& ss,
                                                const PhysicalExpression& expr,
                                                DataType compute_dtype,
                                                const std::string& storage_dtype_vector,
                                                const std::string& suffix,
                                                const std::string& indent,
                                                const std::function<std::string(uint32_t)>& input_slot_value,
                                                const std::function<std::string(uint32_t)>& scalar_const_value = {},
                                                bool input_slot_value_is_compute = false,
                                                const std::function<bool(uint32_t)>& should_emit_node = {}) {
    std::string compute_dtype_vector;
    if (compute_dtype == DataType::BF16) {
        compute_dtype_vector = "__nv_bfloat162";
    } else if (compute_dtype == DataType::FP16) {
        compute_dtype_vector = "half2";
    } else {
        throw runtime_error("emitVector2NodeDefinitionsForSuffix called with non-vectorizable dtype.");
    }

    for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
        if (should_emit_node && !should_emit_node(node_idx)) {
            continue;
        }
        const auto& n = expr.nodes[node_idx];
        // Constant-folded and alias nodes (x * 1, x + 0, ...) resolve as in the scalar emitters
        // (emitResolvedScalarValueExpr): the folded literal, or the source's value.
        double folded_constant = 0.0;
        uint32_t alias_source_idx = UINT32_MAX;
        if (n.op != ExprOp::SCALAR_FP && n.op != ExprOp::FILL && tryGetEmitterConstantValue(expr, node_idx, folded_constant)) {
            ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
               << emitVector2ScalarLiteral(folded_constant, compute_dtype) << ";\n";
            continue;
        }
        if (tryGetEmitterAliasSource(expr, node_idx, alias_source_idx)) {
            ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = " << refWithSuffix(alias_source_idx, suffix)
               << ";\n";
            continue;
        }
        switch (n.op) {
            case ExprOp::INPUT: {
                const std::string variable = input_slot_value(n.input_slot);
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = ";
                if (input_slot_value_is_compute) {
                    ss << variable;
                } else {
                    ss << vector_compute_conversion(storage_dtype_vector, variable, compute_dtype);
                }
                ss << ";\n";
                break;
            }
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2RuntimeScalarValue(expr, n, compute_dtype) << ";\n";
                break;
            case ExprOp::SCALAR_FP:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << (scalar_const_value ? scalar_const_value(node_idx) : emitVector2ScalarLiteral(n.scalar_fp, compute_dtype)) << ";\n";
                break;
            case ExprOp::FILL:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << (scalar_const_value ? scalar_const_value(node_idx) : emitVector2ScalarLiteral(n.scalar_fp, compute_dtype)) << ";\n";
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
                   << emitVector2Neg(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::ABS:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Abs(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::CEIL:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Ceil(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::FLOOR:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Floor(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::ROUND:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Round(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::TRUNC:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Trunc(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::SIN:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Sin(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::COS:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Cos(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::TAN:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Tan(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::ASIN:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Asin(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::ACOS:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Acos(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::ATAN:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Atan(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::SINH:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Sinh(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::COSH:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Cosh(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::ASINH:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Asinh(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::ACOSH:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Acosh(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::ATANH:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Atanh(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::ERF:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Erf(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::ERFC:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Erfc(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::ERFCX:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Erfcx(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::ERFINV:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Erfinv(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::ERFCINV:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Erfcinv(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::TGAMMA:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Tgamma(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::LGAMMA:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Lgamma(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::DIGAMMA:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Digamma(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::EXP:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Exp(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::EXPM1:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Expm1(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::EXP2:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Exp2(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::EXP10:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Exp10(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::LN:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Ln(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::LOG1P:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Log1p(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::LOG2:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Log2(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::LOG10:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Log10(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::SQRT:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Sqrt(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::TANH:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Tanh(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::NORMCDF:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Normcdf(refWithSuffix(n.lhs, suffix), compute_dtype) << ";\n";
                break;
            case ExprOp::RAGGED_VALUEWISE_EXTENT:
            case ExprOp::BROADCAST_TO:
            case ExprOp::CAST:
            case ExprOp::RESHAPE:
            case ExprOp::UNSQUEEZE:
            case ExprOp::SQUEEZE:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = " << refWithSuffix(n.lhs, suffix)
                   << ";\n";
                break;
            case ExprOp::POW:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Pow(refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix), compute_dtype) << ";\n";
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
                   << emitVector2MinMaxGradMask(n.op, refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix), compute_dtype) << ";\n";
                break;
            default:
                throw runtime_error("Unsupported op in vectorized fused emitter: " + to_string((int32_t)n.op));
        }
    }
}

static bool isSupportedLogicalTransposeVectorElementwiseOp(ExprOp op) {
    switch (op) {
        case ExprOp::ADD:
        case ExprOp::SUB:
        case ExprOp::MUL:
        case ExprOp::DIV:
        case ExprOp::NEG:
        case ExprOp::ABS:
        case ExprOp::CEIL:
        case ExprOp::FLOOR:
        case ExprOp::ROUND:
        case ExprOp::TRUNC:
        case ExprOp::SIN:
        case ExprOp::COS:
        case ExprOp::TAN:
        case ExprOp::ASIN:
        case ExprOp::ACOS:
        case ExprOp::ATAN:
        case ExprOp::SINH:
        case ExprOp::COSH:
        case ExprOp::ASINH:
        case ExprOp::ACOSH:
        case ExprOp::ATANH:
        case ExprOp::ERF:
        case ExprOp::ERFC:
        case ExprOp::ERFCX:
        case ExprOp::ERFINV:
        case ExprOp::ERFCINV:
        case ExprOp::TGAMMA:
        case ExprOp::LGAMMA:
        case ExprOp::DIGAMMA:
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
        case ExprOp::POW:
        case ExprOp::MIN:
        case ExprOp::MAX:
        case ExprOp::MIN_GRAD_LEFT:
        case ExprOp::MIN_GRAD_RIGHT:
        case ExprOp::MAX_GRAD_LEFT:
        case ExprOp::MAX_GRAD_RIGHT:
            return true;
        default:
            return false;
    }
}

static bool canVectorizeTiledLogicalTransposeConsumerNode(const PhysicalExpression& expr,
                                                          uint32_t node_idx,
                                                          const std::unordered_set<uint32_t>& frontier_indices,
                                                          DataType vector_compute_dtype,
                                                          std::unordered_map<uint32_t, bool>& memo) {
    if (node_idx >= expr.nodes.size()) {
        throw runtime_error("Tiled logical-transpose vector eligibility node index out of range.");
    }

    const auto memo_it = memo.find(node_idx);
    if (memo_it != memo.end()) {
        return memo_it->second;
    }

    const ExprNode& n = expr.nodes[node_idx];
    bool result = false;

    double folded_constant = 0.0;
    if (frontier_indices.find(node_idx) != frontier_indices.end()) {
        result = true;
    } else if (n.op != ExprOp::INPUT && n.op != ExprOp::RUNTIME_SCALAR && n.op != ExprOp::TENSOR_RUNTIME_SCALAR &&
               tryGetEmitterConstantValue(expr, node_idx, folded_constant)) {
        result = true;
    } else {
        switch (n.op) {
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
            case ExprOp::SCALAR_FP:
            case ExprOp::FILL:
                result = true;
                break;

            case ExprOp::RESHAPE:
            case ExprOp::UNSQUEEZE:
            case ExprOp::SQUEEZE:
                result = n.lhs != UINT32_MAX &&
                         canVectorizeTiledLogicalTransposeConsumerNode(expr, n.lhs, frontier_indices, vector_compute_dtype, memo);
                break;

            case ExprOp::INPUT:
            case ExprOp::TRANSPOSE:
            case ExprOp::STRIDED_VIEW:
            case ExprOp::TAKE_ALONG_AXIS:
            case ExprOp::ROPE:
                result = false;
                break;

            default:
                if (!isSupportedLogicalTransposeVectorElementwiseOp(n.op) || requireNodeComputeDType(n) != vector_compute_dtype) {
                    result = false;
                } else if (Expression::isBinaryOp(n.op)) {
                    result = n.lhs != UINT32_MAX && n.rhs != UINT32_MAX &&
                             canVectorizeTiledLogicalTransposeConsumerNode(expr, n.lhs, frontier_indices, vector_compute_dtype, memo) &&
                             canVectorizeTiledLogicalTransposeConsumerNode(expr, n.rhs, frontier_indices, vector_compute_dtype, memo);
                } else if (Expression::isUnaryOp(n.op)) {
                    result = n.lhs != UINT32_MAX &&
                             canVectorizeTiledLogicalTransposeConsumerNode(expr, n.lhs, frontier_indices, vector_compute_dtype, memo);
                } else {
                    result = false;
                }
                break;
        }
    }

    memo.emplace(node_idx, result);
    return result;
}

static std::optional<DataType> preferredTiledLogicalTransposeVectorComputeDType(const PhysicalExpression& expr,
                                                                                uint32_t node_idx,
                                                                                const std::unordered_set<uint32_t>& frontier_indices) {
    if (node_idx >= expr.nodes.size()) {
        throw runtime_error("Tiled logical-transpose vector dtype node index out of range.");
    }

    const ExprNode& n = expr.nodes[node_idx];
    if (frontier_indices.find(node_idx) != frontier_indices.end()) {
        const DataType frontier_dtype = requireNodeOutputDType(n);
        return defaultVectorizedComputeScalarDTypeForStorage(frontier_dtype);
    }

    double folded_constant = 0.0;
    if (n.op != ExprOp::INPUT && n.op != ExprOp::RUNTIME_SCALAR && n.op != ExprOp::TENSOR_RUNTIME_SCALAR &&
        tryGetEmitterConstantValue(expr, node_idx, folded_constant)) {
        return std::nullopt;
    }

    if (n.op == ExprOp::TAKE_ALONG_AXIS) {
        return std::nullopt;
    }

    if (Expression::isBinaryOp(n.op) || Expression::isUnaryOp(n.op)) {
        const DataType compute_dtype = requireNodeComputeDType(n);
        if (!isLogicalTransposeVectorComputeDType(compute_dtype)) {
            return std::nullopt;
        }
        return compute_dtype;
    }

    if (n.op == ExprOp::RESHAPE || n.op == ExprOp::UNSQUEEZE || n.op == ExprOp::SQUEEZE) {
        if (n.lhs == UINT32_MAX) {
            return std::nullopt;
        }
        return preferredTiledLogicalTransposeVectorComputeDType(expr, n.lhs, frontier_indices);
    }

    return std::nullopt;
}

static std::optional<DataType> tiledLogicalTransposeConsumerVectorComputeDType(const PhysicalExpression& expr,
                                                                               uint32_t output_node_idx,
                                                                               const std::vector<uint32_t>& frontier_indices,
                                                                               uint32_t logical_transpose_pack_scalars) {
    if (logical_transpose_pack_scalars < 2U) {
        return std::nullopt;
    }

    const std::unordered_set<uint32_t> frontier_set(frontier_indices.begin(), frontier_indices.end());
    const std::optional<DataType> maybe_vector_compute_dtype =
        preferredTiledLogicalTransposeVectorComputeDType(expr, output_node_idx, frontier_set);
    if (!maybe_vector_compute_dtype.has_value() || !isLogicalTransposeVectorComputeDType(maybe_vector_compute_dtype.value())) {
        return std::nullopt;
    }

    std::unordered_map<uint32_t, bool> memo;
    if (!canVectorizeTiledLogicalTransposeConsumerNode(expr, output_node_idx, frontier_set, maybe_vector_compute_dtype.value(), memo)) {
        return std::nullopt;
    }

    return maybe_vector_compute_dtype.value();
}

uint32_t CudaSourceEmitter::tiledLogicalTransposeConsumerVectorizedOutputCount(const CompiledExecutionStage& stage,
                                                                               const std::vector<SpecializedBroadcastGroup>& groups) {
    const std::optional<std::vector<uint32_t>> maybe_frontiers = tryFindTiledLogicalTransposeConsumerFrontiers(stage, groups);
    if (!maybe_frontiers.has_value() || maybe_frontiers->empty()) {
        return 0U;
    }

    const uint32_t logical_transpose_pack_scalars = CudaSourceEmitter::tiledLogicalTransposeConsumerPackScalars(stage, groups);
    if (logical_transpose_pack_scalars < 2U || groups.empty()) {
        return 0U;
    }

    uint32_t count = 0U;
    const SpecializedBroadcastGroup& group = groups[0];
    for (uint32_t stage_output_idx : group.output_indices) {
        if (stage_output_idx >= stage.outputs.size()) {
            throw runtime_error("Tiled logical-transpose vector debug output index out of range.");
        }
        const CompiledStageOutput& output = stage.outputs[stage_output_idx];
        if (output.local_node_idx >= stage.expr.nodes.size()) {
            throw runtime_error("Tiled logical-transpose vector debug output node index out of range.");
        }
        if (tiledLogicalTransposeConsumerVectorComputeDType(
                stage.expr, output.local_node_idx, maybe_frontiers.value(), logical_transpose_pack_scalars)
                .has_value()) {
            ++count;
        }
    }
    return count;
}

static std::string emitTiledLogicalTransposeConsumerVectorValue(
    std::ostringstream& ss,
    const PhysicalExpression& expr,
    uint32_t node_idx,
    const std::unordered_map<uint32_t, TiledLogicalTransposeFrontierVectorValue>& frontier_values,
    const std::string& indent,
    DataType vector_compute_dtype,
    uint32_t& counter) {
    if (node_idx >= expr.nodes.size()) {
        throw runtime_error("Tiled logical-transpose vector emitter node index out of range.");
    }

    const ExprNode& n = expr.nodes[node_idx];
    const auto frontier_it = frontier_values.find(node_idx);
    if (frontier_it != frontier_values.end()) {
        if (frontier_it->second.scalar_dtype == vector_compute_dtype) {
            return frontier_it->second.value_expr;
        }

        const std::string var = "tv_cast" + std::to_string(node_idx) + nextIndexMappedSuffix(node_idx, counter);
        const std::string vector_type = logicalTransposeVectorComputeType(vector_compute_dtype);
        const std::string lane0 = castScalarExpr(
            emitLogicalTransposeVector2Lane(frontier_it->second.value_expr, 0U), frontier_it->second.scalar_dtype, vector_compute_dtype);
        const std::string lane1 = castScalarExpr(
            emitLogicalTransposeVector2Lane(frontier_it->second.value_expr, 1U), frontier_it->second.scalar_dtype, vector_compute_dtype);
        ss << indent << "const " << vector_type << " " << var << " = ";
        if (vector_compute_dtype == DataType::FP16) {
            ss << "__halves2half2(" << lane0 << ", " << lane1 << ");\n";
        } else if (vector_compute_dtype == DataType::BF16) {
            ss << "__halves2bfloat162(" << lane0 << ", " << lane1 << ");\n";
        } else {
            throw runtime_error("Unsupported logical transpose vector cast dtype.");
        }
        return var;
    }

    const std::string vector_type = logicalTransposeVectorComputeType(vector_compute_dtype);
    const std::string suffix = nextIndexMappedSuffix(node_idx, counter);
    const std::string var = "tv" + std::to_string(node_idx) + suffix;

    double folded_constant = 0.0;
    if (n.op != ExprOp::INPUT && n.op != ExprOp::RUNTIME_SCALAR && n.op != ExprOp::TENSOR_RUNTIME_SCALAR &&
        tryGetEmitterConstantValue(expr, node_idx, folded_constant)) {
        ss << indent << "const " << vector_type << " " << var << " = " << emitVector2ScalarLiteral(folded_constant, vector_compute_dtype)
           << ";\n";
        return var;
    }

    switch (n.op) {
        case ExprOp::RUNTIME_SCALAR:
        case ExprOp::TENSOR_RUNTIME_SCALAR:
            ss << indent << "const " << vector_type << " " << var << " = " << emitVector2RuntimeScalarValue(expr, n, vector_compute_dtype)
               << ";\n";
            return var;

        case ExprOp::SCALAR_FP:
        case ExprOp::FILL:
            ss << indent << "const " << vector_type << " " << var << " = " << emitVector2ScalarLiteral(n.scalar_fp, vector_compute_dtype)
               << ";\n";
            return var;

        case ExprOp::RESHAPE:
        case ExprOp::UNSQUEEZE:
        case ExprOp::SQUEEZE:
            return emitTiledLogicalTransposeConsumerVectorValue(ss, expr, n.lhs, frontier_values, indent, vector_compute_dtype, counter);

        case ExprOp::INPUT:
        case ExprOp::TRANSPOSE:
        case ExprOp::STRIDED_VIEW:
        case ExprOp::ROPE:
            throw runtime_error("Unsupported node in vectorized tiled logical-transpose consumer emitter: " +
                                to_string(static_cast<int>(n.op)));

        default:
            break;
    }

    auto emit_child = [&](uint32_t child_idx) -> std::string {
        return emitTiledLogicalTransposeConsumerVectorValue(ss, expr, child_idx, frontier_values, indent, vector_compute_dtype, counter);
    };

    std::string compute_expr;
    switch (n.op) {
        case ExprOp::ADD:
            compute_expr = emitVector2Add(emit_child(n.lhs), emit_child(n.rhs));
            break;
        case ExprOp::SUB:
            compute_expr = emitVector2Sub(emit_child(n.lhs), emit_child(n.rhs));
            break;
        case ExprOp::MUL:
            compute_expr = emitVector2Mul(emit_child(n.lhs), emit_child(n.rhs));
            break;
        case ExprOp::DIV:
            compute_expr = emitVector2Div(emit_child(n.lhs), emit_child(n.rhs));
            break;
        case ExprOp::NEG:
            compute_expr = emitVector2Neg(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::ABS:
            compute_expr = emitVector2Abs(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::CEIL:
            compute_expr = emitVector2Ceil(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::FLOOR:
            compute_expr = emitVector2Floor(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::ROUND:
            compute_expr = emitVector2Round(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::TRUNC:
            compute_expr = emitVector2Trunc(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::SIN:
            compute_expr = emitVector2Sin(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::COS:
            compute_expr = emitVector2Cos(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::TAN:
            compute_expr = emitVector2Tan(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::ASIN:
            compute_expr = emitVector2Asin(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::ACOS:
            compute_expr = emitVector2Acos(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::ATAN:
            compute_expr = emitVector2Atan(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::SINH:
            compute_expr = emitVector2Sinh(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::COSH:
            compute_expr = emitVector2Cosh(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::ASINH:
            compute_expr = emitVector2Asinh(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::ACOSH:
            compute_expr = emitVector2Acosh(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::ATANH:
            compute_expr = emitVector2Atanh(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::ERF:
            compute_expr = emitVector2Erf(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::ERFC:
            compute_expr = emitVector2Erfc(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::ERFCX:
            compute_expr = emitVector2Erfcx(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::ERFINV:
            compute_expr = emitVector2Erfinv(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::ERFCINV:
            compute_expr = emitVector2Erfcinv(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::TGAMMA:
            compute_expr = emitVector2Tgamma(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::LGAMMA:
            compute_expr = emitVector2Lgamma(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::DIGAMMA:
            compute_expr = emitVector2Digamma(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::EXP:
            compute_expr = emitVector2Exp(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::EXPM1:
            compute_expr = emitVector2Expm1(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::EXP2:
            compute_expr = emitVector2Exp2(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::EXP10:
            compute_expr = emitVector2Exp10(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::LN:
            compute_expr = emitVector2Ln(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::LOG1P:
            compute_expr = emitVector2Log1p(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::LOG2:
            compute_expr = emitVector2Log2(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::LOG10:
            compute_expr = emitVector2Log10(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::SQRT:
            compute_expr = emitVector2Sqrt(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::TANH:
            compute_expr = emitVector2Tanh(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::NORMCDF:
            compute_expr = emitVector2Normcdf(emit_child(n.lhs), vector_compute_dtype);
            break;
        case ExprOp::POW:
            compute_expr = emitVector2Pow(emit_child(n.lhs), emit_child(n.rhs), vector_compute_dtype);
            break;
        case ExprOp::MIN:
            compute_expr = emitVector2Min(emit_child(n.lhs), emit_child(n.rhs));
            break;
        case ExprOp::MAX:
            compute_expr = emitVector2Max(emit_child(n.lhs), emit_child(n.rhs));
            break;
        case ExprOp::MIN_GRAD_LEFT:
        case ExprOp::MIN_GRAD_RIGHT:
        case ExprOp::MAX_GRAD_LEFT:
        case ExprOp::MAX_GRAD_RIGHT:
            compute_expr = emitVector2MinMaxGradMask(n.op, emit_child(n.lhs), emit_child(n.rhs), vector_compute_dtype);
            break;
        default:
            throw runtime_error("Unsupported op in vectorized tiled logical-transpose consumer emitter: " +
                                to_string(static_cast<int>(n.op)));
    }

    ss << indent << "const " << vector_type << " " << var << " = " << compute_expr << ";\n";
    return var;
}

static void emitFloat2NodeDefinitionsForSuffix(std::ostringstream& ss,
                                               const PhysicalExpression& expr,
                                               const std::string& input_storage_dtype_vector,
                                               const std::string& suffix,
                                               const std::string& indent,
                                               const std::function<std::string(uint32_t)>& input_slot_value,
                                               const std::function<std::string(uint32_t)>& scalar_const_value = {},
                                               bool input_slot_value_is_compute = false) {
    for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
        const auto& n = expr.nodes[node_idx];
        // Constant-folded and alias nodes (x * 1, x + 0, ...) resolve as in the scalar emitters
        // (emitResolvedScalarValueExpr): the folded literal, or the source's value.
        double folded_constant = 0.0;
        uint32_t alias_source_idx = UINT32_MAX;
        if (n.op != ExprOp::SCALAR_FP && n.op != ExprOp::FILL && tryGetEmitterConstantValue(expr, node_idx, folded_constant)) {
            ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = " << emitFloat2ScalarLiteral(folded_constant) << ";\n";
            continue;
        }
        if (tryGetEmitterAliasSource(expr, node_idx, alias_source_idx)) {
            ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = " << refWithSuffix(alias_source_idx, suffix) << ";\n";
            continue;
        }
        switch (n.op) {
            case ExprOp::INPUT: {
                const std::string variable = input_slot_value(n.input_slot);
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << (input_slot_value_is_compute ? variable : float2_compute_conversion(input_storage_dtype_vector, variable))
                   << ";\n";
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
            case ExprOp::CEIL:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("ceilf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::FLOOR:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("floorf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::ROUND:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("roundf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::TRUNC:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("truncf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::SIN:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("sinf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::COS:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("cosf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::TAN:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("tanf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::ASIN:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("asinf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::ACOS:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("acosf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::ATAN:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("atanf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::SINH:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("sinhf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::COSH:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("coshf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::ASINH:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("asinhf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::ACOSH:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("acoshf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::ATANH:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("atanhf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::ERF:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("erff", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::ERFC:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("erfcf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::ERFCX:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("erfcxf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::ERFINV:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("erfinvf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::ERFCINV:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("erfcinvf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::TGAMMA:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("tgammaf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::LGAMMA:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("lgammaf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::DIGAMMA:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("thor_digammaf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::EXP:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("expf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::EXPM1:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("expm1f", refWithSuffix(n.lhs, suffix)) << ";\n";
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
            case ExprOp::LOG1P:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("log1pf", refWithSuffix(n.lhs, suffix)) << ";\n";
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
            case ExprOp::TANH:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("tanhf", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::NORMCDF:
                ss << indent << "float2 " << refWithSuffix(node_idx, suffix) << " = "
                   << emitFloat2UnaryCall("normcdff", refWithSuffix(n.lhs, suffix)) << ";\n";
                break;
            case ExprOp::RESHAPE:
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
                                   DataType storage_dtype,
                                   DataType vector_compute_dtype,
                                   const std::string& kernel_name,
                                   bool use_uint32_index_math) {
    std::ostringstream ss;

    std::string compute_dtype_vector;
    std::string storage_dtype_vector;
    uint32_t packs_per_thread = 0;

    if (!supportsVectorizedComputeDTypeForStorage(storage_dtype, vector_compute_dtype)) {
        throw runtime_error("emitVector2Flat called with unsupported storage/compute dtype combination.");
    }

    if (vector_compute_dtype == DataType::BF16) {
        compute_dtype_vector = "__nv_bfloat162";
    } else if (vector_compute_dtype == DataType::FP16) {
        compute_dtype_vector = "half2";
    } else {
        throw runtime_error("emitVector2Flat called with non-vectorizable compute dtype.");
    }

    if (storage_dtype == DataType::BF16) {
        storage_dtype_vector = "__nv_bfloat162";
        packs_per_thread = 4;
        emitCudaFp16Header(ss);
        ss << "#include <cuda_bf16.h>\n";
    } else if (storage_dtype == DataType::FP16) {
        storage_dtype_vector = "half2";
        packs_per_thread = 4;
        emitCudaFp16Header(ss);
    } else if (storage_dtype == DataType::FP8_E4M3) {
        storage_dtype_vector = "__nv_fp8x2_e4m3";
        packs_per_thread = 8;
        emitCudaFp16Header(ss);
        if (vector_compute_dtype == DataType::BF16) {
            ss << "#include <cuda_bf16.h>\n";
        }
        ss << "#include <cuda_fp8.h>\n";
        emitFp8E4M3SatfiniteHelpers(ss);
    } else if (storage_dtype == DataType::FP8_E5M2) {
        storage_dtype_vector = "__nv_fp8x2_e5m2";
        packs_per_thread = 8;
        emitCudaFp16Header(ss);
        if (vector_compute_dtype == DataType::BF16) {
            ss << "#include <cuda_bf16.h>\n";
        }
        ss << "#include <cuda_fp8.h>\n";
        emitFp8E5M2SatfiniteHelpers(ss);
    } else {
        throw runtime_error("emitVector2Flat called with non-vectorizable storage dtype.");
    }

    const uint32_t num_inputs = stage.expr.numInputs();
    const std::string index_type = emittedIndexType(use_uint32_index_math);
    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const std::optional<RaggedValuewiseExtentEmissionSpec> ragged_extent =
        findRaggedValuewiseExtentSpec(stage.expr);
    const std::unordered_set<uint32_t> structural_partition_slots =
        raggedValuewiseExtentPartitionInputSlots(stage.expr);
    const uint32_t packet_scalars = packs_per_thread * 2U;
    const std::string storage_scalar_type = scalarStorageType(storage_dtype);

    // CUDA vector builtins are available to NVRTC without <vector_types.h>;
    // CUDA 13.3 bundled headers do not include that legacy top-level header.
    //
    // The kernel ABI is scalar-typed. A runtime tensor can be a storage alias whose
    // base is element aligned but not 16-byte aligned, and a view's final packet can
    // extend past the view into live neighbouring storage; see emitPacketLoadStage.
    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";

    for (uint32_t i = 0; i < num_inputs; ++i) {
        if (stage.expr.inputs[i].kind == NamedInput::Kind::RuntimeScalarFp32) {
            ss << scalarStorageType(input_dtypes[i]) << " in" << i;
        } else {
            ss << "const " << scalarStorageType(input_dtypes[i]) << "* in" << i;
        }
        ss << ", ";
    }

    for (uint32_t i = 0; i < stage.outputs.size(); ++i) {
        ss << storage_scalar_type << "* out" << i << ", ";
    }

    ss << index_type << " numel) {\n";

    std::vector<PacketTensorAccess> packet_inputs;
    std::vector<PacketTensorAccess> packet_outputs;
    std::vector<std::pair<std::string, uint64_t>> aligned_pointers;
    for (uint32_t i = 0; i < num_inputs; ++i) {
        if (stage.expr.inputs[i].kind != NamedInput::Kind::Tensor || structural_partition_slots.contains(i)) {
            continue;
        }
        packet_inputs.push_back(PacketTensorAccess{.chunk_var = "in" + std::to_string(i) + "_chunk",
                                                   .chunk_type = "float4",
                                                   .scalar_type = storage_scalar_type,
                                                   .pointer_expr = "in" + std::to_string(i),
                                                   .base_expr = "scalar_base"});
        aligned_pointers.emplace_back("in" + std::to_string(i), 16ULL);
    }
    for (uint32_t i = 0; i < stage.outputs.size(); ++i) {
        packet_outputs.push_back(PacketTensorAccess{.chunk_var = "out" + std::to_string(i) + "_chunk",
                                                    .chunk_type = "float4",
                                                    .scalar_type = storage_scalar_type,
                                                    .pointer_expr = "out" + std::to_string(i),
                                                    .base_expr = "scalar_base"});
        aligned_pointers.emplace_back("out" + std::to_string(i), 16ULL);
    }

    std::string indent = "  ";
    ss << "  const bool packets_aligned = " << emitPacketAlignmentPredicate(aligned_pointers) << ";\n";
    if (ragged_extent.has_value()) {
        const RaggedValuewiseExtentEmissionSpec& extent = ragged_extent.value();
        ss << "  const unsigned long long active_values_raw = static_cast<unsigned long long>(in"
           << extent.partition_input_slot << "[" << raggedActiveValueCountLoadIndex(extent) << "]);\n";
        ss << "  const unsigned long long active_values = active_values_raw < " << extent.max_active_values
           << "ULL ? active_values_raw : " << extent.max_active_values << "ULL;\n";
        ss << "  const unsigned long long runtime_numel_u64 = active_values * " << extent.elements_per_value << "ULL;\n";
        ss << "  const " << index_type << " active_numel = static_cast<" << index_type
           << ">(runtime_numel_u64 < static_cast<unsigned long long>(numel) ? runtime_numel_u64 : static_cast<unsigned long long>(numel));\n";
        ss << "  const " << index_type << " packet_count = (active_numel + "
           << emitUnsignedLiteral(static_cast<uint64_t>(packet_scalars - 1U), use_uint32_index_math) << ") / "
           << emitUnsignedLiteral(static_cast<uint64_t>(packet_scalars), use_uint32_index_math) << ";\n";
        ss << "  const " << index_type << " grid_stride = static_cast<" << index_type
           << ">(blockDim.x) * static_cast<" << index_type << ">(gridDim.x);\n";
        ss << "  for (" << index_type << " idx = " << emitFlatThreadIndexExpr(use_uint32_index_math)
           << "; idx < packet_count; idx += grid_stride) {\n";
        indent = "    ";
    } else {
        ss << "  const " << index_type << " active_numel = numel;\n";
        ss << "  const " << index_type << " idx = " << emitFlatThreadIndexExpr(use_uint32_index_math) << ";\n";
    }
    ss << indent << "const " << index_type << " scalar_base = idx * "
       << emitUnsignedLiteral(static_cast<uint64_t>(packet_scalars), use_uint32_index_math) << ";\n";
    if (!ragged_extent.has_value()) {
        ss << indent << "if (scalar_base >= active_numel) return;\n";
    }
    ss << indent << "const bool vector_packet = packets_aligned && scalar_base + "
       << emitUnsignedLiteral(static_cast<uint64_t>(packet_scalars), use_uint32_index_math) << " <= active_numel;\n";
    const auto lane_valid = [&](uint32_t lane) {
        return "scalar_base + " + emitUnsignedLiteral(lane, use_uint32_index_math) + " < active_numel";
    };
    emitPacketLoadStage(ss, indent, packet_inputs, packet_scalars, "vector_packet", lane_valid);
    for (const PacketTensorAccess& in : packet_inputs) {
        ss << indent << "const " << storage_dtype_vector << "* " << in.chunk_var << "_data = reinterpret_cast<const " << storage_dtype_vector
           << "*>(&" << in.chunk_var << ");\n";
    }
    for (const PacketTensorAccess& out : packet_outputs) {
        ss << indent << "float4 " << out.chunk_var << ";\n";
        ss << indent << storage_dtype_vector << "* " << out.chunk_var << "_data = reinterpret_cast<" << storage_dtype_vector << "*>(&"
           << out.chunk_var << ");\n";
    }
    ss << "\n";

    for (uint32_t pack = 0; pack < packs_per_thread; ++pack) {
        const std::string suffix = "_p" + std::to_string(pack);

        for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
            const auto& n = stage.expr.nodes[node_idx];
            switch (n.op) {
                case ExprOp::INPUT: {
                    if (structural_partition_slots.contains(n.input_slot)) {
                        // Structural partition inputs are consumed only by the
                        // runtime-extent prologue and do not produce a value lane.
                        break;
                    }
                    const std::string variable = "in" + to_string(n.input_slot) + "_chunk_data[" + std::to_string(pack) + "]";
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << vector_compute_conversion(storage_dtype_vector, variable, vector_compute_dtype) << ";\n";
                    break;
                }
                case ExprOp::RAGGED_VALUEWISE_EXTENT:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << refWithSuffix(n.lhs, suffix) << ";\n";
                    break;
                case ExprOp::RUNTIME_SCALAR:
                case ExprOp::TENSOR_RUNTIME_SCALAR:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2RuntimeScalarValue(stage.expr, n, vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::SCALAR_FP:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2ScalarLiteral(n.scalar_fp, vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::FILL:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2ScalarLiteral(n.scalar_fp, vector_compute_dtype) << ";\n";
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
                       << emitVector2Neg(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ABS:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Abs(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::CEIL:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Ceil(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::FLOOR:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Floor(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ROUND:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Round(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::TRUNC:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Trunc(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::SIN:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Sin(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::COS:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Cos(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::TAN:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Tan(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ASIN:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Asin(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ACOS:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Acos(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ATAN:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Atan(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::SINH:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Sinh(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::COSH:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Cosh(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ASINH:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Asinh(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ACOSH:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Acosh(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ATANH:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Atanh(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ERF:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Erf(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ERFC:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Erfc(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ERFCX:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Erfcx(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ERFINV:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Erfinv(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ERFCINV:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Erfcinv(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::TGAMMA:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Tgamma(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::LGAMMA:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Lgamma(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::DIGAMMA:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Digamma(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::EXP:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Exp(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::EXPM1:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Expm1(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::EXP2:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Exp2(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::EXP10:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Exp10(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::LN:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Ln(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::LOG1P:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Log1p(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::LOG2:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Log2(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::LOG10:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Log10(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::SQRT:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Sqrt(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::TANH:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Tanh(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::NORMCDF:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Normcdf(refWithSuffix(n.lhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::BROADCAST_TO:
                    [[fallthrough]];
                case ExprOp::CAST:
                    // getVectorizedStageStorageDTypeImpl only selects this packed path
                    // when all tensor inputs, node outputs, and materialized outputs
                    // share dtype.  CAST is therefore a same-dtype no-op here.
                    [[fallthrough]];
                case ExprOp::RESHAPE:
                case ExprOp::UNSQUEEZE:
                case ExprOp::SQUEEZE:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = " << refWithSuffix(n.lhs, suffix)
                       << ";\n";
                    break;
                case ExprOp::POW:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Pow(refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix), vector_compute_dtype) << ";\n";
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
                       << emitVector2MinMaxGradMask(n.op, refWithSuffix(n.lhs, suffix), refWithSuffix(n.rhs, suffix), vector_compute_dtype) << ";\n";
                    break;
                default:
                    throw runtime_error("Unsupported op in vectorized fused emitter: " + to_string((int32_t)n.op));
            }
        }

        ss << "\n";
        for (uint32_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
            const CompiledStageOutput& output = stage.outputs[out_idx];
            ss << "  out" << out_idx << "_chunk_data[" << pack
               << "] = " << vector_storage_conversion(storage_dtype_vector, refWithSuffix(output.local_node_idx, suffix), vector_compute_dtype) << ";\n";
        }
        ss << "\n";
    }

    emitPacketStoreStage(ss, indent, packet_outputs, packet_scalars, "vector_packet", lane_valid);
    if (ragged_extent.has_value()) {
        ss << "  }\n";
    }

    ss << "}\n";
    return ss.str();
}

static std::string chunkScalarTypeForBytes(uint32_t bytes) {
    switch (bytes) {
        case 2:
            return "unsigned short";
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

// FP8 lanes in the wide-scalar packet kernels convert two lanes per instruction.
// On sm89+ cvt.rn.f16x2.e4m3x2 / e5m2x2 decodes and cvt.rn.satfinite.e4m3x2.f32 /
// e5m2x2 encodes a pair; the per-lane form spends a full conversion on each value
// plus a byte extract/insert. Decoded inputs are exact FP32 values, so the
// per-lane body reads them as FP32 (INPUT nodes retyped in a copy of the
// expression) and every consumer cast produces the same value as before.
// Encoding a pair is the same rounding as encoding each lane (sm80 emulates both
// the same way).
struct Fp8LanePairing {
    PhysicalExpression lane_expr;
    std::unordered_set<uint32_t> decoded_slots;
    std::unordered_set<uint32_t> half_decoded_slots;  // decoded to half2: every consumer computes in FP16
    std::vector<bool> paired_outputs;          // indexed by stage output index
    std::vector<uint32_t> paired_value_nodes;  // node whose FP32 value is encoded, per paired output

    [[nodiscard]] bool active() const {
        return !decoded_slots.empty() || std::find(paired_outputs.begin(), paired_outputs.end(), true) != paired_outputs.end();
    }
};

static uint32_t resolveEmitterAliasChain(const PhysicalExpression& expr, uint32_t node_idx) {
    uint32_t source_idx = UINT32_MAX;
    while (tryGetEmitterAliasSource(expr, node_idx, source_idx)) {
        node_idx = source_idx;
    }
    return node_idx;
}

// The tensor input slot a stage output passes through unchanged (alias ops only: x * 1, x + 0,
// reshape, ...) when that input is stored in the output's dtype. The scalar emitters copy the stored
// value for such an output, so paired lanes copy the stored pair as well: decoding and re-encoding is
// not the identity for FP8 (satfinite turns +-Inf into +-max finite and canonicalises NaN).
static std::optional<uint32_t> passthroughInputSlotForOutput(const PhysicalExpression& expr, uint32_t output_node_idx) {
    const ExprNode& source = expr.nodes.at(resolveEmitterAliasChain(expr, output_node_idx));
    if (source.op != ExprOp::INPUT || requireNodeOutputDType(source) != requireNodeOutputDType(expr.nodes.at(output_node_idx))) {
        return std::nullopt;
    }
    return source.input_slot;
}

static std::string fp8x2StorageType(DataType dtype) {
    if (dtype == DataType::FP8_E4M3) return "__nv_fp8x2_e4m3";
    if (dtype == DataType::FP8_E5M2) return "__nv_fp8x2_e5m2";
    throw runtime_error("fp8x2StorageType requires an FP8 dtype.");
}

static std::string fp8x2SatfiniteEncoder(DataType dtype) {
    if (dtype == DataType::FP8_E4M3) return "thor_to_fp8x2_e4m3_satfinite";
    if (dtype == DataType::FP8_E5M2) return "thor_to_fp8x2_e5m2_satfinite";
    throw runtime_error("fp8x2SatfiniteEncoder requires an FP8 dtype.");
}

// candidate_slots: FP8 tensor inputs held in a packet chunk (one lane per scalar).
// output_indices: stage outputs written by this kernel (or group).
static Fp8LanePairing planFp8LanePairing(const PhysicalExpression& expr,
                                         const std::vector<CompiledStageOutput>& outputs,
                                         const std::vector<uint32_t>& output_indices,
                                         const std::vector<uint32_t>& candidate_slots,
                                         uint32_t lanes) {
    Fp8LanePairing plan{.lane_expr = expr,
                        .decoded_slots = {},
                        .half_decoded_slots = {},
                        .paired_outputs = std::vector<bool>(outputs.size(), false),
                        .paired_value_nodes = std::vector<uint32_t>(outputs.size(), UINT32_MAX)};
    if (lanes < 2 || lanes % 2 != 0) {
        return plan;
    }
    // An output that is an FP8 input passed straight through keeps its byte copy.
    std::unordered_set<uint32_t> passthrough_slots;
    for (uint32_t out_idx : output_indices) {
        const ExprNode& source = expr.nodes.at(resolveEmitterAliasChain(expr, outputs.at(out_idx).local_node_idx));
        if (source.op == ExprOp::INPUT) {
            passthrough_slots.insert(source.input_slot);
        }
    }
    for (uint32_t slot : candidate_slots) {
        if (!passthrough_slots.contains(slot)) {
            plan.decoded_slots.insert(slot);
        }
    }
    // FP8 -> FP16 is exact and one conversion per pair on sm89+ (cvt.rn.f16x2.e4m3x2): an input whose
    // every consumer computes in FP16 decodes straight to half2 instead of through float2 and back.
    std::unordered_map<uint32_t, bool> all_consumers_fp16;
    for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
        const ExprNode& node = expr.nodes[node_idx];
        if (Expression::isLeafOp(node.op) || !shouldEmitScalarNodeDefinition(expr, node_idx)) {
            continue;
        }
        for (const uint32_t operand : {node.lhs, node.rhs, node.aux}) {
            if (operand >= expr.nodes.size()) {
                continue;
            }
            const ExprNode& source = expr.nodes[resolveEmitterAliasChain(expr, operand)];
            if (source.op != ExprOp::INPUT || !plan.decoded_slots.contains(source.input_slot)) {
                continue;
            }
            const bool fp16 = node.op != ExprOp::WHERE && node.compute_dtype == DataType::FP16;
            const auto [it, inserted] = all_consumers_fp16.emplace(source.input_slot, fp16);
            if (!inserted) {
                it->second = it->second && fp16;
            }
        }
    }
    for (const auto& [slot, fp16] : all_consumers_fp16) {
        if (fp16) {
            plan.half_decoded_slots.insert(slot);
        }
    }
    for (ExprNode& node : plan.lane_expr.nodes) {
        if (node.op == ExprOp::INPUT && plan.decoded_slots.contains(node.input_slot)) {
            if (plan.half_decoded_slots.contains(node.input_slot)) {
                // The lane reads an already-decoded __half: no FP8 -> FP16 cast on top.
                node.input_tensor_dtype = DataType::FP16;
                node.output_dtype = DataType::FP16;
            } else {
                node.output_dtype = DataType::FP32;
            }
        }
    }
    for (uint32_t out_idx : output_indices) {
        const uint32_t out_node = outputs.at(out_idx).local_node_idx;
        if (!isFp8DType(requireNodeOutputDType(expr.nodes.at(out_node)))) {
            continue;
        }
        const uint32_t source_idx = resolveEmitterAliasChain(plan.lane_expr, out_node);
        double constant_value = 0.0;
        const bool constant = tryGetEmitterConstantValue(plan.lane_expr, source_idx, constant_value);
        const ExprNode& source = plan.lane_expr.nodes.at(source_idx);
        if (constant || !isFp8DType(emittedScalarNodeValueDType(source))) {
            plan.paired_outputs[out_idx] = true;
            plan.paired_value_nodes[out_idx] = out_node;
        } else if (source.op == ExprOp::CAST && source.lhs < plan.lane_expr.nodes.size() &&
                   emittedScalarNodeValueDType(source) == requireNodeOutputDType(expr.nodes.at(out_node))) {
            // An explicit cast to this FP8 type (quantize): encode the cast's input as a pair.
            // satfinite(float(src)) is what the per-lane cast computes for every source type.
            plan.paired_outputs[out_idx] = true;
            plan.paired_value_nodes[out_idx] = source.lhs;
        }
    }
    return plan;
}

static std::string fp8DecodedPairVar(const Fp8LanePairing& plan, uint32_t slot, uint32_t pair) {
    return "in" + std::to_string(slot) + (plan.half_decoded_slots.contains(slot) ? "_h" : "_f") + std::to_string(pair);
}

static std::string fp8DecodedLaneExpr(const Fp8LanePairing& plan, uint32_t slot, uint32_t lane) {
    return fp8DecodedPairVar(plan, slot, lane / 2) + ((lane % 2 == 0) ? ".x" : ".y");
}

static void emitFp8PairedInputDecode(std::ostringstream& ss,
                                     const std::string& indent,
                                     const Fp8LanePairing& plan,
                                     const std::vector<DataType>& input_dtypes,
                                     const std::function<std::string(uint32_t)>& chunk_var_for_slot,
                                     uint32_t lanes) {
    std::vector<uint32_t> slots(plan.decoded_slots.begin(), plan.decoded_slots.end());
    std::sort(slots.begin(), slots.end());
    for (uint32_t slot : slots) {
        const std::string pair_type = fp8x2StorageType(input_dtypes.at(slot));
        for (uint32_t pair = 0; pair < lanes / 2; ++pair) {
            const std::string decoded_type = plan.half_decoded_slots.contains(slot) ? "__half2" : "float2";
            ss << indent << "const " << decoded_type << " " << fp8DecodedPairVar(plan, slot, pair) << " = static_cast<" << decoded_type
               << ">(reinterpret_cast<const "
               << pair_type << "*>(&" << chunk_var_for_slot(slot) << ")[" << pair << "]);\n";
        }
    }
}

// Emitted after the odd lane of each pair; the even lane's values are still in scope.
static void emitFp8PairedOutputStore(std::ostringstream& ss,
                                     const std::string& indent,
                                     const Fp8LanePairing& plan,
                                     const CompiledStageOutput& output,
                                     DataType output_dtype,
                                     const std::string& chunk_var,
                                     uint32_t odd_lane,
                                     const std::string& even_suffix,
                                     const std::string& odd_suffix,
                                     uint32_t out_idx) {
    (void)output;
    const uint32_t value_node = plan.paired_value_nodes.at(out_idx);
    ss << indent << "reinterpret_cast<" << fp8x2StorageType(output_dtype) << "*>(&" << chunk_var << ")[" << odd_lane / 2
       << "] = " << fp8x2SatfiniteEncoder(output_dtype) << "(make_float2("
       << emitResolvedScalarValueExprSuffixed(plan.lane_expr, value_node, DataType::FP32, even_suffix) << ", "
       << emitResolvedScalarValueExprSuffixed(plan.lane_expr, value_node, DataType::FP32, odd_suffix) << "));\n";
}

static std::string emitWideScalarFlat(const PhysicalExecutionStage& stage,
                                      const std::string& kernel_name,
                                      bool use_uint32_index_math,
                                      uint32_t elements_per_thread) {
    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const std::vector<DataType> output_dtypes = collectOutputDTypes(stage);
    const std::optional<RaggedValuewiseExtentEmissionSpec> ragged_extent = findRaggedValuewiseExtentSpec(stage.expr);
    const std::unordered_set<uint32_t> structural_slots =
        raggedValuewiseExtentPartitionInputSlots(stage.expr);
    const std::string index_type = emittedIndexType(use_uint32_index_math);

    // Packet layout per tensor: each thread owns `elements_per_thread` consecutive
    // scalars of every tensor, moved with one chunk-sized access on the fast path.
    std::vector<std::string> input_chunk_types(input_dtypes.size());
    std::vector<PacketTensorAccess> packet_inputs;
    std::vector<PacketTensorAccess> packet_outputs;
    std::vector<std::pair<std::string, uint64_t>> aligned_pointers;
    for (uint32_t i = 0; i < input_dtypes.size(); ++i) {
        if (stage.expr.inputs[i].kind != NamedInput::Kind::Tensor || structural_slots.contains(i)) {
            continue;
        }
        const uint32_t chunk_bytes = dataTypeStorageBytes(input_dtypes[i]) * elements_per_thread;
        input_chunk_types[i] = chunkScalarTypeForBytes(chunk_bytes);
        packet_inputs.push_back(PacketTensorAccess{.chunk_var = "in" + std::to_string(i) + "_chunk",
                                                   .chunk_type = input_chunk_types[i],
                                                   .scalar_type = scalarStorageType(input_dtypes[i]),
                                                   .pointer_expr = "in" + std::to_string(i),
                                                   .base_expr = "scalar_base"});
        aligned_pointers.emplace_back("in" + std::to_string(i), chunk_bytes);
    }
    std::vector<std::string> output_chunk_types(output_dtypes.size());
    for (uint32_t i = 0; i < output_dtypes.size(); ++i) {
        const uint32_t chunk_bytes = dataTypeStorageBytes(output_dtypes[i]) * elements_per_thread;
        output_chunk_types[i] = chunkScalarTypeForBytes(chunk_bytes);
        packet_outputs.push_back(PacketTensorAccess{.chunk_var = "out" + std::to_string(i) + "_chunk",
                                                    .chunk_type = output_chunk_types[i],
                                                    .scalar_type = scalarStorageType(output_dtypes[i]),
                                                    .pointer_expr = "out" + std::to_string(i),
                                                    .base_expr = "scalar_base"});
        aligned_pointers.emplace_back("out" + std::to_string(i), chunk_bytes);
    }
    const auto is_fp32_float4 = [](const PacketTensorAccess& t) { return t.chunk_type == "float4" && t.scalar_type == "float"; };

    std::ostringstream ss;
    emitRequiredHeaders(stage.expr, ss);

    // The kernel ABI is scalar-typed: runtime tensors may be storage aliases whose
    // element offset does not preserve chunk alignment. See emitPacketLoadStage.
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
    ss << index_type << " numel) {\n";

    std::string indent = "  ";
    ss << "  const bool packets_aligned = " << emitPacketAlignmentPredicate(aligned_pointers) << ";\n";
    if (ragged_extent.has_value()) {
        const RaggedValuewiseExtentEmissionSpec& extent = ragged_extent.value();
        ss << "  const unsigned long long active_values_raw = static_cast<unsigned long long>(in" << extent.partition_input_slot << "["
           << raggedActiveValueCountLoadIndex(extent) << "]);\n";
        ss << "  const unsigned long long active_values = active_values_raw < " << extent.max_active_values
           << "ULL ? active_values_raw : " << extent.max_active_values << "ULL;\n";
        ss << "  const unsigned long long runtime_numel_u64 = active_values * " << extent.elements_per_value << "ULL;\n";
        ss << "  const " << index_type << " active_numel = static_cast<" << index_type
           << ">(runtime_numel_u64 < static_cast<unsigned long long>(numel) ? runtime_numel_u64 : static_cast<unsigned long long>(numel));\n";
        ss << "  const " << index_type << " packet_count = (active_numel + "
           << emitUnsignedLiteral(static_cast<uint64_t>(elements_per_thread - 1u), use_uint32_index_math) << ") / "
           << emitUnsignedLiteral(static_cast<uint64_t>(elements_per_thread), use_uint32_index_math) << ";\n";
        ss << "  const " << index_type << " grid_stride = static_cast<" << index_type
           << ">(blockDim.x) * static_cast<" << index_type << ">(gridDim.x);\n";
        ss << "  for (" << index_type << " idx = " << emitFlatThreadIndexExpr(use_uint32_index_math)
           << "; idx < packet_count; idx += grid_stride) {\n";
        indent = "    ";
    } else {
        ss << "  const " << index_type << " active_numel = numel;\n";
        ss << "  const " << index_type << " idx = " << emitFlatThreadIndexExpr(use_uint32_index_math) << ";\n";
    }
    ss << indent << "const " << index_type << " scalar_base = idx * "
       << emitUnsignedLiteral(static_cast<uint64_t>(elements_per_thread), use_uint32_index_math) << ";\n";
    if (!ragged_extent.has_value()) {
        ss << indent << "if (scalar_base >= active_numel) return;\n";
    }
    ss << indent << "const bool vector_packet = packets_aligned && scalar_base + "
       << emitUnsignedLiteral(static_cast<uint64_t>(elements_per_thread), use_uint32_index_math) << " <= active_numel;\n";
    const auto lane_valid = [&](uint32_t lane) {
        return "scalar_base + " + emitUnsignedLiteral(lane, use_uint32_index_math) + " < active_numel";
    };

    emitPacketLoadStage(ss, indent, packet_inputs, elements_per_thread, "vector_packet", lane_valid);
    for (const PacketTensorAccess& in : packet_inputs) {
        if (!is_fp32_float4(in)) {
            ss << indent << "const " << in.scalar_type << "* " << in.chunk_var << "_data = reinterpret_cast<const " << in.scalar_type
               << "*>(&" << in.chunk_var << ");\n";
        }
    }
    for (const PacketTensorAccess& out : packet_outputs) {
        ss << indent << out.chunk_type << " " << out.chunk_var << ";\n";
        if (!is_fp32_float4(out)) {
            ss << indent << out.scalar_type << "* " << out.chunk_var << "_data = reinterpret_cast<" << out.scalar_type << "*>(&"
               << out.chunk_var << ");\n";
        }
    }
    ss << "\n";

    std::vector<uint32_t> fp8_chunk_slots;
    for (uint32_t i = 0; i < input_dtypes.size(); ++i) {
        if (!input_chunk_types[i].empty() && isFp8DType(input_dtypes[i])) {
            fp8_chunk_slots.push_back(i);
        }
    }
    std::vector<uint32_t> all_output_indices(stage.outputs.size());
    std::iota(all_output_indices.begin(), all_output_indices.end(), 0u);
    const Fp8LanePairing fp8_pairing =
        planFp8LanePairing(stage.expr, stage.outputs, all_output_indices, fp8_chunk_slots, elements_per_thread);
    emitFp8PairedInputDecode(
        ss, indent, fp8_pairing, input_dtypes, [](uint32_t slot) { return "in" + std::to_string(slot) + "_chunk"; }, elements_per_thread);
    if (!fp8_pairing.decoded_slots.empty()) {
        ss << "\n";
    }

    // Compute stage: emitted once. Lanes past the active extent compute on zeroed
    // packet registers and are discarded by the masked store.
    for (uint32_t lane = 0; lane < elements_per_thread; ++lane) {
        const std::string suffix = "_l" + std::to_string(lane);
        const std::string lane_idx_expr = "scalar_base + " + emitUnsignedLiteral(static_cast<uint64_t>(lane), use_uint32_index_math);
        const auto input_value = [&](uint32_t slot) -> std::string {
            if (fp8_pairing.decoded_slots.contains(slot)) {
                return fp8DecodedLaneExpr(fp8_pairing, slot, lane);
            }
            return emitChunkLaneReadExpr("in" + std::to_string(slot) + "_chunk",
                                         input_chunk_types.at(slot),
                                         input_dtypes.at(slot),
                                         std::to_string(lane),
                                         "in" + std::to_string(slot) + "_chunk_data",
                                         elements_per_thread);
        };

        for (uint32_t node_idx = 0; node_idx < fp8_pairing.lane_expr.nodes.size(); ++node_idx) {
            const ExprNode& node = fp8_pairing.lane_expr.nodes[node_idx];
            if ((node.op == ExprOp::INPUT && structural_slots.contains(node.input_slot)) ||
                !shouldEmitScalarNodeDefinition(fp8_pairing.lane_expr, node_idx)) {
                continue;
            }
            if (fp8_pairing.decoded_slots.empty()) {
                emitScalarNodeSuffixed(ss,
                                       fp8_pairing.lane_expr,
                                       node_idx,
                                       lane_idx_expr,
                                       suffix,
                                       indent,
                                       std::to_string(lane),
                                       elements_per_thread);
            } else {
                emitScalarNodeSuffixed(ss, fp8_pairing.lane_expr, node_idx, lane_idx_expr, suffix, indent, "", 0, input_value);
            }
        }

        for (uint32_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
            const CompiledStageOutput& output = stage.outputs[out_idx];
            const DataType output_dtype = requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]);
            if (fp8_pairing.paired_outputs[out_idx]) {
                if (lane % 2 == 1) {
                    emitFp8PairedOutputStore(ss,
                                             indent,
                                             fp8_pairing,
                                             output,
                                             output_dtype,
                                             "out" + std::to_string(out_idx) + "_chunk",
                                             lane,
                                             "_l" + std::to_string(lane - 1),
                                             suffix,
                                             out_idx);
                }
                continue;
            }
            ss << emitChunkLaneWriteStmt("out" + std::to_string(out_idx) + "_chunk",
                                         output_chunk_types[out_idx],
                                         output_dtype,
                                         lane,
                                         emitResolvedScalarValueExprSuffixed(fp8_pairing.lane_expr, output.local_node_idx, output_dtype, suffix),
                                         indent,
                                         "out" + std::to_string(out_idx) + "_chunk_data");
        }
        ss << "\n";
    }

    emitPacketStoreStage(ss, indent, packet_outputs, elements_per_thread, "vector_packet", lane_valid);
    if (ragged_extent.has_value()) {
        ss << "  }\n";
    }
    ss << "}\n";
    return ss.str();
}

// Shared prologue / per-group loop header for the wide (16-byte packet)
// specialized-broadcast emitters. A dense stage is a ragged stage whose active
// extent is the group capacity: its packet count is a compile-time constant and
// the launch covers every packet, so the grid-stride loop runs at most once per
// thread. A device-ragged stage reads its active extent on device and launches
// an occupancy-sized grid.
static void emitWideBroadcastActiveValuesPrologue(std::ostringstream& ss,
                                                  const std::optional<RaggedValuewiseExtentEmissionSpec>& ragged_extent,
                                                  const std::string& index_type,
                                                  bool use_uint32_index_math) {
    if (ragged_extent.has_value()) {
        const RaggedValuewiseExtentEmissionSpec& extent = ragged_extent.value();
        ss << "  const unsigned long long active_values_raw = static_cast<unsigned long long>(in" << extent.partition_input_slot << "["
           << raggedActiveValueCountLoadIndex(extent) << "]);\n";
        ss << "  const unsigned long long active_values = active_values_raw < " << extent.max_active_values
           << "ULL ? active_values_raw : " << extent.max_active_values << "ULL;\n";
    }
    ss << "  const " << index_type << " thread_packet = " << emitFlatThreadIndexExpr(use_uint32_index_math) << ";\n";
    ss << "  const " << index_type << " packet_grid_stride = static_cast<" << index_type
       << ">(blockDim.x) * static_cast<" << index_type << ">(gridDim.x);\n\n";
}

static void emitWideBroadcastGroupPacketLoopHeader(std::ostringstream& ss,
                                                   const std::optional<RaggedValuewiseExtentEmissionSpec>& ragged_extent,
                                                   const SpecializedBroadcastGroup& group,
                                                   uint32_t g,
                                                   uint32_t packet_scalars,
                                                   const std::string& index_type,
                                                   bool use_uint32_index_math) {
    if (ragged_extent.has_value()) {
        const uint64_t group_elements_per_value = specializedRaggedGroupElementsPerValue(ragged_extent.value(), group);
        ss << "  const unsigned long long runtime_numel_g" << g << "_u64 = active_values * " << group_elements_per_value << "ULL;\n";
        ss << "  const " << index_type << " runtime_numel_g" << g << " = static_cast<" << index_type
           << ">(runtime_numel_g" << g << "_u64 < " << emitUnsignedLiteral(group.numel, false)
           << " ? runtime_numel_g" << g << "_u64 : " << emitUnsignedLiteral(group.numel, false) << ");\n";
        ss << "  const " << index_type << " packet_count_g" << g << " = runtime_numel_g" << g << " / "
           << emitUnsignedLiteral(packet_scalars, use_uint32_index_math) << ";\n";
    } else {
        ss << "  const " << index_type << " packet_count_g" << g << " = "
           << emitUnsignedLiteral(group.numel / packet_scalars, use_uint32_index_math) << ";\n";
    }
    ss << "  for (" << index_type << " packet_idx = thread_packet; packet_idx < packet_count_g" << g
       << "; packet_idx += packet_grid_stride) {\n";
    ss << "    const " << index_type << " scalar_base = packet_idx * "
       << emitUnsignedLiteral(packet_scalars, use_uint32_index_math) << ";\n";
}

static std::string emitWideSpecializedBroadcast(const CompiledExecutionStage& stage,
                                                       const std::vector<SpecializedBroadcastGroup>& groups,
                                                       DataType storage_dtype_kind,
                                                       DataType vector_compute_dtype,
                                                       const std::string& kernel_name) {
    constexpr uint32_t packet_scalars = 8U;
    constexpr uint32_t packs_per_thread = packet_scalars / 2U;

    if (storage_dtype_kind != DataType::BF16 && storage_dtype_kind != DataType::FP16) {
        throw runtime_error("emitWideSpecializedBroadcast requires BF16 or FP16 storage.");
    }
    if (!supportsVectorizedComputeDTypeForStorage(storage_dtype_kind, vector_compute_dtype)) {
        throw runtime_error("emitWideSpecializedBroadcast called with unsupported storage/compute dtype combination.");
    }

    // Serves dense stages (active extent = group capacity) and device-ragged
    // stages (active extent read from the row partition on device).
    const std::optional<RaggedValuewiseExtentEmissionSpec> ragged_extent = findRaggedValuewiseExtentSpec(stage.expr);
    const std::unordered_set<uint32_t> structural_slots = raggedValuewiseExtentPartitionInputSlots(stage.expr);
    const bool use_uint32_index_math = groupsSupportUInt32IndexMath(groups);
    const std::string index_type = emittedIndexType(use_uint32_index_math);
    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);

    std::string storage_dtype;
    std::string storage_dtype_vector;
    if (storage_dtype_kind == DataType::BF16) {
        storage_dtype = "__nv_bfloat16";
        storage_dtype_vector = "__nv_bfloat162";
    } else {
        storage_dtype = "half";
        storage_dtype_vector = "half2";
    }

    std::ostringstream ss;
    emitCudaFp16Header(ss);
    if (storage_dtype_kind == DataType::BF16 || vector_compute_dtype == DataType::BF16) {
        ss << "#include <cuda_bf16.h>\n";
    }

    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";
    bool first_arg = true;
    for (uint32_t i = 0; i < stage.expr.numInputs(); ++i) {
        if (!first_arg) ss << ", ";
        first_arg = false;
        if (stage.expr.inputs[i].kind == NamedInput::Kind::RuntimeScalarFp32) {
            ss << scalarStorageType(input_dtypes[i]) << " in" << i;
        } else if (structural_slots.contains(i)) {
            ss << "const " << scalarStorageType(input_dtypes[i]) << "* in" << i;
        } else {
            ss << "const " << storage_dtype << "* in" << i;
        }
    }
    for (uint32_t i = 0; i < stage.outputs.size(); ++i) {
        if (!first_arg) ss << ", ";
        first_arg = false;
        ss << storage_dtype << "* out" << i;
    }
    ss << ") {\n";

    emitWideBroadcastActiveValuesPrologue(ss, ragged_extent, index_type, use_uint32_index_math);

    for (uint32_t g = 0; g < groups.size(); ++g) {
        const SpecializedBroadcastGroup& group = groups[g];
        if ((group.numel % packet_scalars) != 0ULL || (group.output_dims.back() % packet_scalars) != 0ULL ||
            (ragged_extent.has_value() &&
             (specializedRaggedGroupElementsPerValue(ragged_extent.value(), group) % packet_scalars) != 0ULL)) {
            throw runtime_error("wide specialized broadcast requires an eight-scalar aligned row width.");
        }

        const std::vector<uint32_t> required_nodes = orderedRequiredNodesForGroup(stage, group.output_indices);
        const std::unordered_set<uint32_t> required_node_set(required_nodes.begin(), required_nodes.end());
        std::unordered_map<uint32_t, size_t> used_index_by_slot;
        used_index_by_slot.reserve(group.used_input_slots.size());
        for (size_t used_i = 0; used_i < group.used_input_slots.size(); ++used_i) {
            used_index_by_slot.emplace(group.used_input_slots[used_i], used_i);
        }
        std::vector<size_t> all_used_indices(group.used_input_slots.size());
        std::iota(all_used_indices.begin(), all_used_indices.end(), 0);

        emitWideBroadcastGroupPacketLoopHeader(ss, ragged_extent, group, g, packet_scalars, index_type, use_uint32_index_math);

        for (uint32_t input_slot : group.used_input_slots) {
            ss << "    " << index_type << " in" << input_slot << "_offset_base = "
               << emitUnsignedLiteral(0, use_uint32_index_math) << ";\n";
        }
        if (!group.active_axes.empty()) {
            emitSpecializedBroadcastOffsetMath(
                ss, group, all_used_indices, "scalar_base", "_base", "    ", use_uint32_index_math);
        }
        ss << "\n";

        // Packets never straddle the active extent here (row widths are packet
        // multiples), but a native-vector operand or the output can be a storage
        // alias whose packet address is not 16-byte aligned, so the vector access
        // is predicated on the actual addresses of this packet.
        std::vector<PacketTensorAccess> packet_inputs;
        std::vector<PacketTensorAccess> packet_outputs;
        std::vector<std::pair<std::string, uint64_t>> packet_addresses;
        for (size_t used_i = 0; used_i < group.used_input_slots.size(); ++used_i) {
            const uint32_t input_slot = group.used_input_slots[used_i];
            const uint64_t inner_stride = specializedBroadcastInnermostInputStride(group, used_i);
            const SpecializedInputLoadKind load_kind = group.used_input_load_kinds[used_i];
            if (load_kind == SpecializedInputLoadKind::NativeVector) {
                if (inner_stride != 1ULL) {
                    throw runtime_error("native-vector ragged broadcast input lost unit innermost stride.");
                }
                const std::string slot = std::to_string(input_slot);
                packet_inputs.push_back(PacketTensorAccess{.chunk_var = "in" + slot + "_packet",
                                                           .chunk_type = "float4",
                                                           .scalar_type = storage_dtype,
                                                           .pointer_expr = "in" + slot,
                                                           .base_expr = "in" + slot + "_offset_base"});
                packet_addresses.emplace_back("in" + slot + " + in" + slot + "_offset_base", 16ULL);
            } else if (inner_stride == 0ULL) {
                ss << "    const " << storage_dtype << " in" << input_slot << "_packet_scalar = in" << input_slot
                   << "[in" << input_slot << "_offset_base];\n";
            }
        }
        for (uint32_t out_idx : group.output_indices) {
            packet_outputs.push_back(PacketTensorAccess{.chunk_var = "out" + std::to_string(out_idx) + "_packet",
                                                        .chunk_type = "float4",
                                                        .scalar_type = storage_dtype,
                                                        .pointer_expr = "out" + std::to_string(out_idx),
                                                        .base_expr = "scalar_base"});
            packet_addresses.emplace_back("out" + std::to_string(out_idx) + " + scalar_base", 16ULL);
        }
        const auto all_lanes_valid = [](uint32_t) { return std::string{}; };
        ss << "    const bool vector_packet = " << emitPacketAlignmentPredicate(packet_addresses) << ";\n";
        emitPacketLoadStage(ss, "    ", packet_inputs, packet_scalars, "vector_packet", all_lanes_valid);
        for (const PacketTensorAccess& in : packet_inputs) {
            ss << "    const " << storage_dtype_vector << "* " << in.chunk_var << "_data = reinterpret_cast<const " << storage_dtype_vector
               << "*>(&" << in.chunk_var << ");\n";
        }
        for (const PacketTensorAccess& out : packet_outputs) {
            ss << "    float4 " << out.chunk_var << ";\n";
            ss << "    " << storage_dtype_vector << "* " << out.chunk_var << "_data = reinterpret_cast<" << storage_dtype_vector << "*>(&"
               << out.chunk_var << ");\n";
        }
        ss << "\n";

        for (uint32_t pack = 0; pack < packs_per_thread; ++pack) {
            const std::string suffix = "_rbp" + std::to_string(pack);
            emitVector2NodeDefinitionsForSuffix(
                ss,
                stage.expr,
                vector_compute_dtype,
                storage_dtype_vector,
                suffix,
                "    ",
                [&](uint32_t input_slot) {
                    const auto used_it = used_index_by_slot.find(input_slot);
                    if (used_it == used_index_by_slot.end()) {
                        throw runtime_error("wide ragged broadcast node references an input outside its broadcast group.");
                    }
                    const size_t used_i = used_it->second;
                    const uint64_t inner_stride = specializedBroadcastInnermostInputStride(group, used_i);
                    if (group.used_input_load_kinds[used_i] == SpecializedInputLoadKind::NativeVector) {
                        return vector_compute_conversion(storage_dtype_vector,
                                                         "in" + std::to_string(input_slot) + "_packet_data[" + std::to_string(pack) + "]",
                                                         vector_compute_dtype);
                    }
                    if (inner_stride == 0ULL) {
                        const std::string scalar = "in" + std::to_string(input_slot) + "_packet_scalar";
                        return emitVector2BroadcastPackLoad(storage_dtype, scalar, scalar, vector_compute_dtype);
                    }
                    const uint64_t lane0 = static_cast<uint64_t>(pack) * 2ULL;
                    const uint64_t lane1 = lane0 + 1ULL;
                    const std::string base = "in" + std::to_string(input_slot) + "_offset_base";
                    const std::string offset0 = base + " + " + emitUnsignedLiteral(lane0 * inner_stride, use_uint32_index_math);
                    const std::string offset1 = base + " + " + emitUnsignedLiteral(lane1 * inner_stride, use_uint32_index_math);
                    const std::string var0 = "in" + std::to_string(input_slot) + "[" + offset0 + "]";
                    const std::string var1 = "in" + std::to_string(input_slot) + "[" + offset1 + "]";
                    return emitVector2BroadcastPackLoad(storage_dtype, var0, var1, vector_compute_dtype);
                },
                {},
                true,
                [&](uint32_t node_idx) {
                    if (!required_node_set.contains(node_idx)) return false;
                    const ExprNode& node = stage.expr.nodes[node_idx];
                    return !(node.op == ExprOp::INPUT && structural_slots.contains(node.input_slot));
                });

            for (uint32_t out_idx : group.output_indices) {
                const CompiledStageOutput& output = stage.outputs[out_idx];
                ss << "    out" << out_idx << "_packet_data[" << pack << "] = "
                   << vector_storage_conversion(storage_dtype_vector,
                                                refWithSuffix(output.local_node_idx, suffix),
                                                vector_compute_dtype)
                   << ";\n";
            }
            ss << "\n";
        }

        emitPacketStoreStage(ss, "    ", packet_outputs, packet_scalars, "vector_packet", all_lanes_valid);
        ss << "  }\n\n";
    }

    ss << "}\n";
    return ss.str();
}

static std::string emitWideScalarSpecializedBroadcast(const CompiledExecutionStage& stage,
                                                            const std::vector<SpecializedBroadcastGroup>& groups,
                                                            const std::string& kernel_name,
                                                            uint32_t packet_scalars) {
    const uint32_t selected_packet_scalars = wideScalarSpecializedBroadcastElementsPerThread(stage, groups);
    if (selected_packet_scalars <= 1u || selected_packet_scalars != packet_scalars) {
        throw runtime_error("emitWideScalarSpecializedBroadcast called with an ineligible packet width.");
    }

    // Serves dense stages (active extent = group capacity) and device-ragged stages.
    const std::optional<RaggedValuewiseExtentEmissionSpec> ragged_extent = findRaggedValuewiseExtentSpec(stage.expr);

    const std::unordered_set<uint32_t> structural_slots = raggedValuewiseExtentPartitionInputSlots(stage.expr);
    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const std::vector<DataType> output_dtypes = collectOutputDTypes(stage);
    const bool use_uint32_index_math = groupsSupportUInt32IndexMath(groups);
    const std::string index_type = emittedIndexType(use_uint32_index_math);

    std::ostringstream ss;
    emitRequiredHeaders(stage.expr, ss);
    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";

    bool first_arg = true;
    for (uint32_t i = 0; i < stage.expr.numInputs(); ++i) {
        if (!first_arg) ss << ", ";
        first_arg = false;
        if (stage.expr.inputs[i].kind == NamedInput::Kind::RuntimeScalarFp32) {
            ss << scalarStorageType(input_dtypes[i]) << " in" << i;
        } else {
            ss << "const " << scalarStorageType(input_dtypes[i]) << "* in" << i;
        }
    }
    for (uint32_t i = 0; i < stage.outputs.size(); ++i) {
        if (!first_arg) ss << ", ";
        first_arg = false;
        ss << scalarStorageType(output_dtypes[i]) << "* out" << i;
    }
    ss << ") {\n";

    emitWideBroadcastActiveValuesPrologue(ss, ragged_extent, index_type, use_uint32_index_math);

    for (uint32_t g = 0; g < groups.size(); ++g) {
        const SpecializedBroadcastGroup& group = groups[g];
        const std::vector<uint32_t> required_nodes = orderedRequiredNodesForGroup(stage, group.output_indices);
        std::unordered_map<uint32_t, size_t> used_index_by_slot;
        used_index_by_slot.reserve(group.used_input_slots.size());
        for (size_t used_i = 0; used_i < group.used_input_slots.size(); ++used_i) {
            used_index_by_slot.emplace(group.used_input_slots[used_i], used_i);
        }
        std::vector<size_t> all_used_indices(group.used_input_slots.size());
        std::iota(all_used_indices.begin(), all_used_indices.end(), 0);

        emitWideBroadcastGroupPacketLoopHeader(ss, ragged_extent, group, g, packet_scalars, index_type, use_uint32_index_math);

        for (uint32_t input_slot : group.used_input_slots) {
            ss << "    " << index_type << " in" << input_slot << "_offset_base = "
               << emitUnsignedLiteral(0, use_uint32_index_math) << ";\n";
        }
        if (!group.active_axes.empty()) {
            emitSpecializedBroadcastOffsetMath(
                ss, group, all_used_indices, "scalar_base", "_base", "    ", use_uint32_index_math);
        }
        ss << "\n";

        // Packets never straddle the active extent (row widths are packet
        // multiples), but a native-vector operand or the output can be a storage
        // alias whose packet address is not chunk aligned; predicate the vector
        // access on this packet's actual addresses.
        std::unordered_map<uint32_t, std::string> input_chunk_types;
        std::vector<PacketTensorAccess> packet_inputs;
        std::vector<PacketTensorAccess> packet_outputs;
        std::vector<std::pair<std::string, uint64_t>> packet_addresses;
        for (size_t used_i = 0; used_i < group.used_input_slots.size(); ++used_i) {
            const uint32_t input_slot = group.used_input_slots[used_i];
            const DataType input_dtype = input_dtypes.at(input_slot);
            const uint64_t inner_stride = specializedBroadcastInnermostInputStride(group, used_i);
            if (group.used_input_load_kinds[used_i] == SpecializedInputLoadKind::NativeVector) {
                const uint32_t chunk_bytes = dataTypeStorageBytes(input_dtype) * packet_scalars;
                const std::string chunk_type = chunkScalarTypeForBytes(chunk_bytes);
                const std::string slot = std::to_string(input_slot);
                input_chunk_types.emplace(input_slot, chunk_type);
                packet_inputs.push_back(PacketTensorAccess{.chunk_var = "in" + slot + "_packet",
                                                           .chunk_type = chunk_type,
                                                           .scalar_type = scalarStorageType(input_dtype),
                                                           .pointer_expr = "in" + slot,
                                                           .base_expr = "in" + slot + "_offset_base"});
                packet_addresses.emplace_back("in" + slot + " + in" + slot + "_offset_base", chunk_bytes);
            } else if (inner_stride == 0ULL) {
                ss << "    const " << scalarStorageType(input_dtype) << " in" << input_slot << "_packet_scalar = in" << input_slot
                   << "[in" << input_slot << "_offset_base];\n";
            }
        }

        std::vector<std::string> output_chunk_types(stage.outputs.size());
        for (uint32_t out_idx : group.output_indices) {
            const DataType output_dtype = output_dtypes.at(out_idx);
            const uint32_t chunk_bytes = dataTypeStorageBytes(output_dtype) * packet_scalars;
            const std::string chunk_type = chunkScalarTypeForBytes(chunk_bytes);
            output_chunk_types[out_idx] = chunk_type;
            packet_outputs.push_back(PacketTensorAccess{.chunk_var = "out" + std::to_string(out_idx) + "_packet",
                                                        .chunk_type = chunk_type,
                                                        .scalar_type = scalarStorageType(output_dtype),
                                                        .pointer_expr = "out" + std::to_string(out_idx),
                                                        .base_expr = "scalar_base"});
            packet_addresses.emplace_back("out" + std::to_string(out_idx) + " + scalar_base", chunk_bytes);
        }
        const auto all_lanes_valid = [](uint32_t) { return std::string{}; };
        ss << "    const bool vector_packet = " << emitPacketAlignmentPredicate(packet_addresses) << ";\n";
        emitPacketLoadStage(ss, "    ", packet_inputs, packet_scalars, "vector_packet", all_lanes_valid);
        for (size_t i = 0; i < packet_inputs.size(); ++i) {
            const PacketTensorAccess& in = packet_inputs[i];
            if (!(in.chunk_type == "float4" && in.scalar_type == "float")) {
                ss << "    const " << in.scalar_type << "* " << in.chunk_var << "_data = reinterpret_cast<const " << in.scalar_type << "*>(&"
                   << in.chunk_var << ");\n";
            }
        }
        for (const PacketTensorAccess& out : packet_outputs) {
            ss << "    " << out.chunk_type << " " << out.chunk_var << ";\n";
            if (!(out.chunk_type == "float4" && out.scalar_type == "float")) {
                ss << "    " << out.scalar_type << "* " << out.chunk_var << "_data = reinterpret_cast<" << out.scalar_type << "*>(&"
                   << out.chunk_var << ");\n";
            }
        }
        ss << "\n";

        std::vector<uint32_t> fp8_chunk_slots;
        for (size_t used_i = 0; used_i < group.used_input_slots.size(); ++used_i) {
            const uint32_t slot = group.used_input_slots[used_i];
            if (group.used_input_load_kinds[used_i] == SpecializedInputLoadKind::NativeVector && isFp8DType(input_dtypes.at(slot))) {
                fp8_chunk_slots.push_back(slot);
            }
        }
        const Fp8LanePairing fp8_pairing =
            planFp8LanePairing(stage.expr, stage.outputs, group.output_indices, fp8_chunk_slots, packet_scalars);
        emitFp8PairedInputDecode(
            ss, "    ", fp8_pairing, input_dtypes, [](uint32_t slot) { return "in" + std::to_string(slot) + "_packet"; }, packet_scalars);
        if (!fp8_pairing.decoded_slots.empty()) {
            ss << "\n";
        }

        for (uint32_t lane = 0; lane < packet_scalars; ++lane) {
            const std::string suffix = "_rmp_l" + std::to_string(lane);
            const std::string lane_idx = "scalar_base + " + emitUnsignedLiteral(lane, use_uint32_index_math);

            auto input_value = [&](uint32_t input_slot) -> std::string {
                if (fp8_pairing.decoded_slots.contains(input_slot)) {
                    return fp8DecodedLaneExpr(fp8_pairing, input_slot, lane);
                }
                const auto used_it = used_index_by_slot.find(input_slot);
                if (used_it == used_index_by_slot.end()) {
                    throw runtime_error("wide-scalar ragged broadcast node references an input outside its broadcast group.");
                }
                const size_t used_i = used_it->second;
                const DataType input_dtype = input_dtypes.at(input_slot);
                const uint64_t inner_stride = specializedBroadcastInnermostInputStride(group, used_i);
                if (group.used_input_load_kinds[used_i] == SpecializedInputLoadKind::NativeVector) {
                    const std::string& chunk_type = input_chunk_types.at(input_slot);
                    const std::string chunk_data = (chunk_type == "float4" && input_dtype == DataType::FP32)
                                                       ? std::string{}
                                                       : "in" + std::to_string(input_slot) + "_packet_data";
                    return emitChunkLaneReadExpr("in" + std::to_string(input_slot) + "_packet",
                                                 chunk_type,
                                                 input_dtype,
                                                 std::to_string(lane),
                                                 chunk_data);
                }
                if (inner_stride == 0ULL) {
                    return "in" + std::to_string(input_slot) + "_packet_scalar";
                }
                return "in" + std::to_string(input_slot) + "[in" + std::to_string(input_slot) + "_offset_base + " +
                       emitUnsignedLiteral(static_cast<uint64_t>(lane) * inner_stride, use_uint32_index_math) + "]";
            };

            for (uint32_t node_idx : required_nodes) {
                const ExprNode& node = fp8_pairing.lane_expr.nodes[node_idx];
                if ((node.op == ExprOp::INPUT && structural_slots.contains(node.input_slot)) ||
                    !shouldEmitScalarNodeDefinition(fp8_pairing.lane_expr, node_idx)) {
                    continue;
                }
                emitScalarNodeSuffixed(ss, fp8_pairing.lane_expr, node_idx, lane_idx, suffix, "    ", "", 0, input_value);
            }

            for (uint32_t out_idx : group.output_indices) {
                const CompiledStageOutput& output = stage.outputs[out_idx];
                const DataType output_dtype = output_dtypes.at(out_idx);
                if (fp8_pairing.paired_outputs[out_idx]) {
                    if (lane % 2 == 1) {
                        emitFp8PairedOutputStore(ss,
                                                 "    ",
                                                 fp8_pairing,
                                                 output,
                                                 output_dtype,
                                                 "out" + std::to_string(out_idx) + "_packet",
                                                 lane,
                                                 "_rmp_l" + std::to_string(lane - 1),
                                                 suffix,
                                                 out_idx);
                    }
                    continue;
                }
                const std::string& chunk_type = output_chunk_types[out_idx];
                const std::string chunk_data = (chunk_type == "float4" && output_dtype == DataType::FP32)
                                                   ? std::string{}
                                                   : "out" + std::to_string(out_idx) + "_packet_data";
                ss << emitChunkLaneWriteStmt("out" + std::to_string(out_idx) + "_packet",
                                             chunk_type,
                                             output_dtype,
                                             lane,
                                             emitResolvedScalarValueExprSuffixed(fp8_pairing.lane_expr, output.local_node_idx, output_dtype, suffix),
                                             "    ",
                                             chunk_data);
            }
            ss << "\n";
        }

        emitPacketStoreStage(ss, "    ", packet_outputs, packet_scalars, "vector_packet", all_lanes_valid);
        ss << "  }\n\n";
    }

    ss << "}\n";
    return ss.str();
}

static std::string emitVector2SpecializedBroadcast(const CompiledExecutionStage& stage,
                                                   const std::vector<SpecializedBroadcastGroup>& groups,
                                                   DataType storage_dtype_kind,
                                                   DataType vector_compute_dtype,
                                                   const std::string& kernel_name) {
    std::ostringstream ss;

    std::string compute_dtype_vector;
    std::string storage_dtype;
    std::string storage_dtype_vector;

    if (!supportsVectorizedComputeDTypeForStorage(storage_dtype_kind, vector_compute_dtype)) {
        throw runtime_error("emitVector2SpecializedBroadcast called with unsupported storage/compute dtype combination.");
    }

    if (vector_compute_dtype == DataType::BF16) {
        compute_dtype_vector = "__nv_bfloat162";
    } else if (vector_compute_dtype == DataType::FP16) {
        compute_dtype_vector = "half2";
    } else {
        throw runtime_error("emitVector2SpecializedBroadcast called with non-vectorizable compute dtype.");
    }

    if (storage_dtype_kind == DataType::BF16) {
        storage_dtype = "__nv_bfloat16";
        storage_dtype_vector = "__nv_bfloat162";
        emitCudaFp16Header(ss);
        ss << "#include <cuda_bf16.h>\n";
    } else if (storage_dtype_kind == DataType::FP16) {
        storage_dtype = "half";
        storage_dtype_vector = "half2";
        emitCudaFp16Header(ss);
    } else if (storage_dtype_kind == DataType::FP8_E4M3) {
        storage_dtype = "__nv_fp8_e4m3";
        storage_dtype_vector = "__nv_fp8x2_e4m3";
        emitCudaFp16Header(ss);
        if (vector_compute_dtype == DataType::BF16) {
            ss << "#include <cuda_bf16.h>\n";
        }
        ss << "#include <cuda_fp8.h>\n";
        emitFp8E4M3SatfiniteHelpers(ss);
    } else if (storage_dtype_kind == DataType::FP8_E5M2) {
        storage_dtype = "__nv_fp8_e5m2";
        storage_dtype_vector = "__nv_fp8x2_e5m2";
        emitCudaFp16Header(ss);
        if (vector_compute_dtype == DataType::BF16) {
            ss << "#include <cuda_bf16.h>\n";
        }
        ss << "#include <cuda_fp8.h>\n";
        emitFp8E5M2SatfiniteHelpers(ss);
    } else {
        throw runtime_error("emitVector2SpecializedBroadcast called with non-vectorizable storage dtype.");
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
        // Keep the output ABI scalar-typed. Runtime storage aliases can begin at
        // an element-aligned but vector-misaligned offset; the generated store
        // below uses the packed path only when the actual address is aligned.
        ss << storage_dtype << "* out" << i;
        if (i + 1 < stage.outputs.size()) {
            ss << ", ";
        }
    }

    const uint32_t storage_vector_alignment = scalarStorageTypeSizeBytes(storage_dtype_kind) * 2u;

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
                        // A strided storage alias can preserve unit inner stride while
                        // shifting the visible base by one scalar. Such an address is
                        // valid for scalar access but can be misaligned for half2 /
                        // bfloat162 / fp8x2. Preserve the native vector load for the
                        // ordinary aligned case and fall back to two adjacent scalar
                        // loads when the runtime address does not meet vector alignment.
                        ss << "    const bool in" << slot << "_native_aligned = "
                           << "((reinterpret_cast<unsigned long long>(" << base << " + " << offset0 << ") & "
                           << (storage_vector_alignment - 1u) << "ULL) == 0ULL);\n";
                        ss << "    " << compute_dtype_vector << " t" << node_idx << " = in" << slot << "_native_aligned ? "
                           << emitVector2BroadcastNativeLoad(storage_dtype_vector, base, offset0, vector_compute_dtype) << " : "
                           << emitVector2BroadcastPackLoad(storage_dtype,
                                                           base + "[" + offset0 + "]",
                                                           base + "[" + offset0 + " + 1]",
                                                           vector_compute_dtype)
                           << ";\n";
                    } else {
                        const std::string var0 = "in" + std::to_string(slot) + "[in" + std::to_string(slot) + "_offset0]";
                        const std::string var1 = "in" + std::to_string(slot) + "[in" + std::to_string(slot) + "_offset1]";
                        ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                           << emitVector2BroadcastPackLoad(storage_dtype, var0, var1, vector_compute_dtype) << ";\n";
                    }
                    break;
                }

                case ExprOp::RUNTIME_SCALAR:
                case ExprOp::TENSOR_RUNTIME_SCALAR:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2RuntimeScalarValue(stage.expr, n, vector_compute_dtype)
                       << ";\n";
                    break;
                case ExprOp::SCALAR_FP:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2ScalarLiteral(n.scalar_fp, vector_compute_dtype)
                       << ";\n";
                    break;
                case ExprOp::FILL:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2ScalarLiteral(n.scalar_fp, vector_compute_dtype)
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
                       << emitVector2Neg(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ABS:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Abs(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::CEIL:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Ceil(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::FLOOR:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Floor(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ROUND:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Round(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::TRUNC:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Trunc(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::SIN:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Sin(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::COS:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Cos(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::TAN:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Tan(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ASIN:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Asin(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ACOS:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Acos(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ATAN:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Atan(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::SINH:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Sinh(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::COSH:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Cosh(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ASINH:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Asinh(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ACOSH:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Acosh(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ATANH:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Atanh(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ERF:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Erf(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ERFC:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Erfc(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ERFCX:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Erfcx(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ERFINV:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Erfinv(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::ERFCINV:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Erfcinv(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::TGAMMA:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Tgamma(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::LGAMMA:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Lgamma(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::DIGAMMA:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Digamma(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::EXP:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Exp(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::EXPM1:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Expm1(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::EXP2:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Exp2(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::EXP10:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Exp10(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::LN:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << emitVector2Ln(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype)
                       << ";\n";
                    break;
                case ExprOp::LOG1P:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Log1p(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::LOG2:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Log2(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::LOG10:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Log10(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::SQRT:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Sqrt(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::TANH:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Tanh(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::NORMCDF:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Normcdf(CudaSourceEmitter::ref(n.lhs), vector_compute_dtype) << ";\n";
                    break;
                case ExprOp::BROADCAST_TO:
                    [[fallthrough]];
                case ExprOp::CAST:
                    // getVectorizedStageStorageDTypeImpl only selects this packed
                    // specialized-broadcast path when every tensor input, node
                    // output, and materialized output has the same dtype. CAST is
                    // therefore a same-dtype alias here; real storage conversions
                    // are emitted by the scalar/mixed-dtype path instead.
                    [[fallthrough]];
                case ExprOp::RESHAPE:
                case ExprOp::UNSQUEEZE:
                case ExprOp::SQUEEZE:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = " << CudaSourceEmitter::ref(n.lhs) << ";\n";
                    break;
                case ExprOp::POW:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Pow(CudaSourceEmitter::ref(n.lhs), CudaSourceEmitter::ref(n.rhs), vector_compute_dtype) << ";\n";
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
                       << emitVector2MinMaxGradMask(n.op, CudaSourceEmitter::ref(n.lhs), CudaSourceEmitter::ref(n.rhs), vector_compute_dtype) << ";\n";
                    break;
                default:
                    throw std::runtime_error("Unsupported op in specialized vector broadcast emitter.");
            }
        }

        ss << "\n";
        for (uint32_t out_idx : group.output_indices) {
            const CompiledStageOutput& output = stage.outputs[out_idx];
            const std::string packed_value = "out" + std::to_string(out_idx) + "_packed";
            ss << "    const " << storage_dtype_vector << " " << packed_value << " = "
               << vector_storage_conversion(storage_dtype_vector, CudaSourceEmitter::ref(output.local_node_idx), vector_compute_dtype) << ";\n";
            // With an odd element count the final pair owns one valid lane. The
            // packed store would write one element past the output, which is live
            // storage when the output is a view, so that pair takes the masked path.
            const std::string odd_tail_guard =
                (group.numel % 2ULL) != 0ULL
                    ? " && (idx0 + " + emitUnsignedLiteral(1, use_uint32_index_math) + " < " +
                          emitUnsignedLiteral(group.numel, use_uint32_index_math) + ")"
                    : std::string{};
            ss << "    const bool out" << out_idx << "_native_aligned = "
               << "((reinterpret_cast<unsigned long long>(out" << out_idx << " + idx0) & "
               << (storage_vector_alignment - 1u) << "ULL) == 0ULL)" << odd_tail_guard << ";\n";
            ss << "    if (out" << out_idx << "_native_aligned) {\n";
            ss << "      *reinterpret_cast<" << storage_dtype_vector << "*>(out" << out_idx << " + idx0) = "
               << packed_value << ";\n";
            ss << "    } else {\n";
            ss << "      const " << storage_dtype << "* out" << out_idx
               << "_packed_scalars = reinterpret_cast<const " << storage_dtype << "*>(&" << packed_value << ");\n";
            ss << "      out" << out_idx << "[idx0] = out" << out_idx << "_packed_scalars[0];\n";
            ss << "      if (idx0 + " << emitUnsignedLiteral(1, use_uint32_index_math) << " < "
               << emitUnsignedLiteral(group.numel, use_uint32_index_math) << ") {\n";
            ss << "        out" << out_idx << "[idx0 + " << emitUnsignedLiteral(1, use_uint32_index_math)
               << "] = out" << out_idx << "_packed_scalars[1];\n";
            ss << "      }\n";
            ss << "    }\n";
        }

        ss << "  }\n\n";
    }

    ss << "}\n";
    return ss.str();
}


// Address-alignment conjunction for a packed access of `pack_type` at
// in<slot> + <index_expr> for every listed input. The element index being a pack
// multiple is not enough: runtime tensors can be storage aliases whose base is
// only element aligned.
static std::string emitInputsPackAddressAligned(const std::vector<uint32_t>& input_slots,
                                                const std::string& index_expr,
                                                const std::string& pack_type) {
    std::string predicate;
    for (uint32_t slot : input_slots) {
        predicate += " && ((reinterpret_cast<unsigned long long>(in" + std::to_string(slot) + " + " + index_expr + ") % sizeof(" +
                     pack_type + ")) == 0ULL)";
    }
    return predicate;
}

static std::vector<uint32_t> tensorInputSlots(const PhysicalExpression& expr) {
    std::vector<uint32_t> slots;
    for (uint32_t i = 0; i < expr.inputs.size(); ++i) {
        if (expr.inputs[i].kind == NamedInput::Kind::Tensor) slots.push_back(i);
    }
    return slots;
}

// Algebraic strength reductions applied to a copy of the stage expression right
// before emission, so every emitter path (scalar, wide, packed vector, float2,
// broadcast, transpose) benefits without per-path special cases.
//
//   pow(x, 2) -> x * x   powf costs tens of instructions per lane (and the packed
//                        BF16/FP16 path calls it once per lane after a float
//                        round trip); a fused stage has only a few hundred
//                        instructions per 16-byte packet before it stops being
//                        memory bound. x * x is also the correctly rounded square.
//   pow(x, 1) -> x * 1   the multiply by one is folded away by the scalar emitters.
//
// Only compile-time constant exponents are rewritten. The rewrite reuses existing
// nodes, so node order (children before parents) and every dtype annotation are
// unchanged, and vector eligibility is unaffected (MUL is supported wherever POW is).
static bool tryGetEmitterConstantValue(const PhysicalExpression& expr, uint32_t node_idx, double& value);

static bool isStrengthReduciblePow(const PhysicalExpression& expr, const ExprNode& node) {
    if (node.op != ExprOp::POW || node.rhs >= expr.nodes.size()) {
        return false;
    }
    double exponent = 0.0;
    return tryGetEmitterConstantValue(expr, node.rhs, exponent) && (exponent == 2.0 || exponent == 1.0);
}

static bool expressionHasStrengthReductions(const PhysicalExpression& expr) {
    return std::any_of(expr.nodes.begin(), expr.nodes.end(), [&](const ExprNode& node) { return isStrengthReduciblePow(expr, node); });
}

static void applyStrengthReductions(PhysicalExpression& expr) {
    for (ExprNode& node : expr.nodes) {
        if (!isStrengthReduciblePow(expr, node)) {
            continue;
        }
        double exponent = 0.0;
        tryGetEmitterConstantValue(expr, node.rhs, exponent);
        node.op = ExprOp::MUL;
        if (exponent == 2.0) {
            node.rhs = node.lhs;
        }
        // exponent == 1.0 keeps rhs: the constant-one operand.
    }
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
    if (scalarStorageTypeSizeBytes(output_dtype) > sizeof(unsigned int)) {
        throw runtime_error("Tiled transpose shared-memory swizzle currently supports output scalar storage types up to 32 bits.");
    }
    const std::string output_type = scalarStorageType(output_dtype);
    const std::optional<DataType> maybe_vectorized_dtype = CudaSourceEmitter::getVectorizedStageStorageDType(stage);
    const std::optional<DataType> maybe_vectorized_compute_dtype =
        maybe_vectorized_dtype.has_value() ? getVectorizedStageComputeDTypeImpl(stage.expr, maybe_vectorized_dtype.value()) : std::nullopt;
    const std::optional<DataType> maybe_tensor_input_dtype = getSingleTensorInputStorageDType(stage.expr, input_dtypes);
    const bool emit_packed_low_precision_path = transposePackScalars(output_dtype) > 1;
    const bool emit_homogeneous_packed_vectorized_path =
        maybe_vectorized_dtype.has_value() && maybe_vectorized_dtype.value() == output_dtype && transposePackScalars(output_dtype) > 1;
    const bool emit_mixed_two_byte_float2_path =
        maybe_tensor_input_dtype.has_value() &&
        supportsMixedTwoByteFloat2TransposedVectorization(stage.expr, maybe_tensor_input_dtype.value(), output_dtype);
    const std::optional<DataType> maybe_mixed_fp8_vector_compute_dtype =
        maybe_tensor_input_dtype.has_value()
            ? mixedFp8TransposedVectorComputeDType(stage.expr, maybe_tensor_input_dtype.value(), output_dtype)
            : std::nullopt;
    const bool emit_mixed_fp8_vectorized_path = maybe_mixed_fp8_vector_compute_dtype.has_value();
    const bool emit_cross_width_float2_path =
        maybe_tensor_input_dtype.has_value() &&
        supportsFloat2TransposedVectorization(stage.expr, maybe_tensor_input_dtype.value(), output_dtype);
    const bool emit_cross_width_half2_path =
        maybe_tensor_input_dtype.has_value() &&
        supportsHalf2TransposedVectorization(stage.expr, maybe_tensor_input_dtype.value(), output_dtype);
    const bool emit_packed_vectorized_path = emit_homogeneous_packed_vectorized_path || emit_mixed_two_byte_float2_path ||
                                             emit_mixed_fp8_vectorized_path || emit_cross_width_float2_path ||
                                             emit_cross_width_half2_path;
    const bool emit_decoupled_line_vectorized_path = shouldUseDecoupledLineVectorizedTranspose(maybe_tensor_input_dtype, output_dtype);
    const std::string index_type = emittedIndexType(use_uint32_index_math);
    std::ostringstream ss;
    emitRequiredHeaders(stage.expr, ss);
    emitSharedTransposeWordHelpers(ss);
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
    ss << output_type << "* __restrict__ out0, " << index_type << " numRows, " << index_type << " numCols, " << index_type
       << " batchCount) {\n";
    ss << "  const " << index_type << " rowTiles = (numRows + static_cast<" << index_type << ">(TILE_DIM) - 1) / static_cast<" << index_type
       << ">(TILE_DIM);\n";
    ss << "  const " << index_type << " flatBlockY = static_cast<" << index_type << ">(blockIdx.y)"
       << " + static_cast<" << index_type << ">(blockIdx.z) * static_cast<" << index_type << ">(gridDim.y);\n";
    ss << "  const " << index_type << " batchIdx = flatBlockY / rowTiles;\n";
    ss << "  if (batchIdx >= batchCount) return;\n";
    ss << "  const " << index_type << " rowTile = flatBlockY - batchIdx * rowTiles;\n";
    ss << "  const " << index_type << " matrixOffset = batchIdx * numRows * numCols;\n\n";

    if (emit_decoupled_line_vectorized_path) {
        const DataType input_dtype = maybe_tensor_input_dtype.value();
        const uint32_t read_pack_scalars = transposePackScalars(input_dtype);
        const uint32_t write_pack_scalars = transposePackScalars(output_dtype);
        const std::string input_type = scalarStorageType(input_dtype);

        ss << "  constexpr unsigned int READ_PACK_SCALARS = " << read_pack_scalars << "U;\n";
        ss << "  constexpr unsigned int WRITE_PACK_SCALARS = " << write_pack_scalars << "U;\n";
        ss << "  constexpr unsigned int TILE_COL_SCALARS = TILE_DIM;\n";
        if (read_pack_scalars > 1) {
            ss << "  using InputPack = " << (isFp8DType(input_dtype) ? std::string("uchar4") : transposePackType(input_dtype)) << ";\n";
        }
        if (write_pack_scalars > 1) {
            ss << "  using OutputPack = " << transposePackType(output_dtype) << ";\n";
        }
        emitSharedTransposeTileDeclaration(ss, "TILE_COL_SCALARS + 1");
        ss << "  const " << index_type << " rowStart = rowTile * TILE_DIM;\n";
        ss << "  const " << index_type << " colStart = static_cast<" << index_type << ">(blockIdx.x) * TILE_COL_SCALARS;\n";
        ss << "  const unsigned int threadLinear = threadIdx.y * TILE_DIM + threadIdx.x;\n\n";

        ss << "  constexpr unsigned int READ_PACKS_PER_ROW = TILE_COL_SCALARS / READ_PACK_SCALARS;\n";
        ss << "  constexpr unsigned int READ_TASKS_PER_TILE = TILE_DIM * READ_PACKS_PER_ROW;\n";
        ss << "  for (unsigned int readTask = threadLinear; readTask < READ_TASKS_PER_TILE; readTask += TILE_DIM * BLOCK_ROWS) {\n";
        ss << "    const unsigned int localRow = readTask / READ_PACKS_PER_ROW;\n";
        ss << "    const unsigned int localReadPackCol = readTask - localRow * READ_PACKS_PER_ROW;\n";
        ss << "    const " << index_type << " logicalRow = rowStart + static_cast<" << index_type << ">(localRow);\n";
        ss << "    const " << index_type << " logicalColBase = colStart + static_cast<" << index_type
           << ">(localReadPackCol) * READ_PACK_SCALARS;\n";
        ss << "    const " << index_type << " idx_base = matrixOffset + logicalRow * numCols + logicalColBase;\n";

        auto emit_flat_scalar_lanes = [&](const std::string& base_indent, const std::string& idx_expr_base, bool use_loaded_chunks) {
            if (use_loaded_chunks && isFp8DType(input_dtype)) {
                // Decode each loaded FP8 chunk as two fp8x2 pairs (one conversion per pair) and
                // run the lanes on the exact FP32 values. Lanes are unrolled with constant indices,
                // so neither the chunk nor the decoded pairs are ever indexed by a runtime lane.
                const Fp8LanePairing fp8_pairing =
                    planFp8LanePairing(stage.expr, stage.outputs, {0u}, tensorInputSlots(stage.expr), read_pack_scalars);
                emitFp8PairedInputDecode(
                    ss, base_indent, fp8_pairing, input_dtypes, [](uint32_t slot) { return "in" + std::to_string(slot) + "_chunk"; },
                    read_pack_scalars);
                for (uint32_t lane = 0; lane < read_pack_scalars; ++lane) {
                    const std::string suffix = "_dl" + std::to_string(lane);
                    const std::string lane_literal = emitUnsignedLiteral(lane, use_uint32_index_math);
                    ss << base_indent << "{\n";
                    ss << base_indent << "  const unsigned int LANE = " << lane << "U;\n";
                    ss << base_indent << "  const unsigned int LOCAL_COL = localReadPackCol * READ_PACK_SCALARS + LANE;\n";
                    const std::string lane_idx_expr = idx_expr_base + " + " + lane_literal;
                    auto input_value = [&](uint32_t input_slot) -> std::string {
                        if (fp8_pairing.decoded_slots.contains(input_slot)) {
                            return fp8DecodedLaneExpr(fp8_pairing, input_slot, lane);
                        }
                        return emitFp8PackLaneMemberExpr("in" + std::to_string(input_slot) + "_chunk", lane, input_dtype);
                    };
                    for (uint32_t node_idx = 0; node_idx < fp8_pairing.lane_expr.nodes.size(); ++node_idx) {
                        if (!shouldEmitScalarNodeDefinition(fp8_pairing.lane_expr, node_idx)) {
                            continue;
                        }
                        emitScalarNodeSuffixed(
                            ss, fp8_pairing.lane_expr, node_idx, lane_idx_expr, suffix, base_indent + "  ", "", 0, input_value);
                    }
                    ss << base_indent << "  tile[localRow][LOCAL_COL] = thor_pack_transpose_word<" << output_type << ">("
                       << emitResolvedScalarValueExprSuffixed(fp8_pairing.lane_expr, output.local_node_idx, output_dtype, suffix)
                       << ");\n";
                    ss << base_indent << "}\n";
                }
                return;
            }

            if (!use_loaded_chunks && isFp8DType(input_dtype)) {
                for (uint32_t lane = 0; lane < read_pack_scalars; ++lane) {
                    const std::string suffix = "_dls" + std::to_string(lane);
                    const std::string lane_literal = emitUnsignedLiteral(lane, use_uint32_index_math);
                    ss << base_indent << "{\n";
                    ss << base_indent << "  const unsigned int LANE = " << lane << "U;\n";
                    ss << base_indent << "  const unsigned int LOCAL_COL = localReadPackCol * READ_PACK_SCALARS + LANE;\n";
                    ss << base_indent << "  const " << index_type << " logicalCol = logicalColBase + " << lane_literal << ";\n";
                    ss << base_indent << "  if (logicalCol < numCols) {\n";
                    ss << base_indent << "    const " << index_type << " idx = matrixOffset + logicalRow * numCols + logicalCol;\n";
                    for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
                        if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
                            continue;
                        }
                        emitScalarNodeSuffixed(ss, stage.expr, node_idx, "idx", suffix, base_indent + "    ");
                    }
                    ss << base_indent << "    tile[localRow][LOCAL_COL] = thor_pack_transpose_word<" << output_type << ">("
                       << emitResolvedScalarValueExprSuffixed(stage.expr, output.local_node_idx, output_dtype, suffix) << ");\n";
                    ss << base_indent << "  }\n";
                    ss << base_indent << "}\n";
                }
                return;
            }

            const std::string suffix = use_loaded_chunks ? "_dl" : "_dls";
            ss << base_indent << "for (unsigned int laneIter = 0; laneIter < READ_PACK_SCALARS; ++laneIter) {\n";
            ss << base_indent << "  const unsigned int LANE = ((localReadPackCol + laneIter) & (READ_PACK_SCALARS - 1U));\n";
            ss << base_indent << "  const unsigned int LOCAL_COL = localReadPackCol * READ_PACK_SCALARS + LANE;\n";
            if (!use_loaded_chunks) {
                ss << base_indent << "  const " << index_type << " logicalCol = logicalColBase + static_cast<" << index_type
                   << ">(LANE);\n";
                ss << base_indent << "  if (logicalCol < numCols) {\n";
                ss << base_indent << "    const " << index_type << " idx = matrixOffset + logicalRow * numCols + logicalCol;\n";
                for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
                    if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
                        continue;
                    }
                    emitScalarNodeSuffixed(ss, stage.expr, node_idx, "idx", suffix, base_indent + "    ");
                }
                ss << base_indent << "    tile[localRow][LOCAL_COL] = thor_pack_transpose_word<" << output_type << ">("
                   << emitResolvedScalarValueExprSuffixed(stage.expr, output.local_node_idx, output_dtype, suffix) << ");\n";
                ss << base_indent << "  }\n";
            } else {
                const std::string lane_idx_expr = idx_expr_base + " + static_cast<" + index_type + ">(LANE)";
                for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
                    if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
                        continue;
                    }
                    emitScalarNodeSuffixed(ss, stage.expr, node_idx, lane_idx_expr, suffix, base_indent + "  ", "LANE", read_pack_scalars);
                }

                ss << base_indent << "  tile[localRow][LOCAL_COL] = thor_pack_transpose_word<" << output_type << ">("
                   << emitResolvedScalarValueExprSuffixed(stage.expr, output.local_node_idx, output_dtype, suffix) << ");\n";
            }
            ss << base_indent << "}\n";
        };

        if (read_pack_scalars > 1) {
            ss << "    const bool inputPackedLoadOk = (logicalRow < numRows) && (logicalColBase + READ_PACK_SCALARS <= numCols) &&\n";
            ss << "                                   ((idx_base % READ_PACK_SCALARS) == " << emitUnsignedLiteral(0, use_uint32_index_math)
               << ")" << emitInputsPackAddressAligned(tensorInputSlots(stage.expr), "idx_base", "InputPack") << ";\n";
            ss << "    if (inputPackedLoadOk) {\n";
            for (uint32_t i = 0; i < input_dtypes.size(); ++i) {
                if (stage.expr.inputs[i].kind != NamedInput::Kind::Tensor) {
                    continue;
                }
                ss << "      const InputPack in" << i << "_chunk = reinterpret_cast<const InputPack*>(in" << i << " + idx_base)[0];\n";
                if (!isFp8DType(input_dtype)) {
                    ss << "      const " << input_type << "* in" << i << "_chunk_data = reinterpret_cast<const " << input_type << "*>(&in"
                       << i << "_chunk);\n";
                }
            }
            emit_flat_scalar_lanes("      ", "idx_base", true);
            ss << "    } else if (logicalRow < numRows) {\n";
            emit_flat_scalar_lanes("      ", "idx_base", false);
            ss << "    }\n";
        } else {
            ss << "    if (logicalRow < numRows) {\n";
            emit_flat_scalar_lanes("      ", "idx_base", false);
            ss << "    }\n";
        }
        ss << "  }\n\n";
        ss << "  __syncthreads();\n\n";

        ss << "  constexpr unsigned int WRITE_PACKS_PER_ROW = TILE_DIM / WRITE_PACK_SCALARS;\n";
        ss << "  constexpr unsigned int WRITE_TASKS_PER_TILE = TILE_COL_SCALARS * WRITE_PACKS_PER_ROW;\n";
        ss << "  for (unsigned int writeTask = threadLinear; writeTask < WRITE_TASKS_PER_TILE; writeTask += TILE_DIM * BLOCK_ROWS) {\n";
        ss << "    const unsigned int localOutRow = writeTask / WRITE_PACKS_PER_ROW;\n";
        ss << "    const unsigned int localOutPackCol = writeTask - localOutRow * WRITE_PACKS_PER_ROW;\n";
        ss << "    const " << index_type << " outputRow = colStart + static_cast<" << index_type << ">(localOutRow);\n";
        ss << "    const " << index_type << " outputColBase = rowStart + static_cast<" << index_type
           << ">(localOutPackCol) * WRITE_PACK_SCALARS;\n";
        if (write_pack_scalars > 1) {
            ss << "    OutputPack output_pack{};\n";
            ss << "    " << output_type << "* output_pack_data = reinterpret_cast<" << output_type << "*>(&output_pack);\n";
            ss << "    for (unsigned int lane = 0; lane < WRITE_PACK_SCALARS; ++lane) {\n";
            ss << "      const " << index_type << " outputCol = outputColBase + static_cast<" << index_type << ">(lane);\n";
            ss << "      if (outputRow < numCols && outputCol < numRows) {\n";
            ss << "        output_pack_data[lane] = thor_unpack_transpose_word<" << output_type
               << ">(tile[localOutPackCol * WRITE_PACK_SCALARS + lane][localOutRow]);\n";
            ss << "      }\n";
            ss << "    }\n";
            ss << "    const " << index_type << " out_base_idx = matrixOffset + outputRow * numRows + outputColBase;\n";
            ss << "    const bool outputPackedStoreOk = (outputRow < numCols) && (outputColBase + WRITE_PACK_SCALARS <= numRows) &&\n";
            ss << "                                     ((out_base_idx % WRITE_PACK_SCALARS) == "
               << emitUnsignedLiteral(0, use_uint32_index_math)
               << ") && ((reinterpret_cast<unsigned long long>(out0 + out_base_idx) % sizeof(OutputPack)) == 0ULL);\n";
            ss << "    if (outputPackedStoreOk) {\n";
            ss << "      OutputPack* out_ptr = reinterpret_cast<OutputPack*>(out0 + out_base_idx);\n";
            ss << "      *out_ptr = output_pack;\n";
            ss << "    } else if (outputRow < numCols) {\n";
            ss << "      for (unsigned int lane = 0; lane < WRITE_PACK_SCALARS; ++lane) {\n";
            ss << "        const " << index_type << " outputCol = outputColBase + static_cast<" << index_type << ">(lane);\n";
            ss << "        if (outputCol < numRows) {\n";
            ss << "          out0[matrixOffset + outputRow * numRows + outputCol] = output_pack_data[lane];\n";
            ss << "        }\n";
            ss << "      }\n";
            ss << "    }\n";
        } else {
            ss << "    if (outputRow < numCols && outputColBase < numRows) {\n";
            ss << "      out0[matrixOffset + outputRow * numRows + outputColBase] = thor_unpack_transpose_word<" << output_type
               << ">(tile[localOutPackCol][localOutRow]);\n";
            ss << "    }\n";
        }
        ss << "  }\n";
        ss << "}\n";
        return ss.str();
    }

    if (emit_packed_low_precision_path) {
        const DataType dtype = output_dtype;
        const bool needs_input_vector_type = emit_homogeneous_packed_vectorized_path || emit_mixed_two_byte_float2_path ||
                                             emit_mixed_fp8_vectorized_path || emit_cross_width_float2_path ||
                                             emit_cross_width_half2_path;
        const DataType vector_input_dtype = needs_input_vector_type
                                                ? (maybe_tensor_input_dtype.has_value() ? maybe_tensor_input_dtype.value() : output_dtype)
                                                : output_dtype;
        const DataType vector_compute_dtype =
            emit_homogeneous_packed_vectorized_path
                ? maybe_vectorized_compute_dtype.value()
                : (emit_mixed_fp8_vectorized_path ? maybe_mixed_fp8_vector_compute_dtype.value()
                                                  : (vector_input_dtype == DataType::BF16 ? DataType::BF16 : DataType::FP16));
        const uint32_t pack_scalars = transposePackScalars(dtype);
        const uint32_t pairs_per_pack = pack_scalars / 2;
        const std::string pack_type = transposePackType(dtype);
        const std::string output_storage_dtype_vector = transposeVector2StorageType(output_dtype);
        const std::string input_storage_dtype_vector =
            needs_input_vector_type ? transposeVector2StorageType(vector_input_dtype) : output_storage_dtype_vector;

        // FP8 inputs: the thread's 4-lane input chunk is one 4-byte load (two pairs) when its actual
        // address is 4-byte aligned; runtime tensors can be element-offset views, so the address is checked.
        const bool load_fp8_input_chunk = emit_packed_vectorized_path && maybe_tensor_input_dtype.has_value() &&
                                          isFp8DType(maybe_tensor_input_dtype.value()) && pack_scalars == 4U;
        ss << "  constexpr unsigned int PACK_SCALARS = " << pack_scalars << "U;\n";
        ss << "  constexpr unsigned int TILE_COL_SCALARS = TILE_DIM * PACK_SCALARS;\n";
        ss << "  using Pack = " << pack_type << ";\n";
        if (emit_packed_vectorized_path) {
            ss << "  using OutputVec2 = " << output_storage_dtype_vector << ";\n";
            ss << "  using InputVec2 = " << input_storage_dtype_vector << ";\n";
        }
        if (load_fp8_input_chunk) {
            ss << "  using InputPack = unsigned int;\n";
        }
        // Pair PAIR of input slot `slot` on the vectorized path.
        const auto input_pair_expr = [&](uint32_t slot, const std::string& vec2_type) -> std::string {
            if (load_fp8_input_chunk) {
                return "reinterpret_cast<const " + vec2_type + "*>(&in" + std::to_string(slot) + "_chunk)[PAIR]";
            }
            return "reinterpret_cast<const " + vec2_type + "*>(in" + std::to_string(slot) + " + idx_base)[PAIR]";
        };
        ss << "  union PackRaw { unsigned int raw; Pack pack; };\n";
        emitSharedTransposeTileDeclaration(ss);
        ss << "  const " << index_type << " rowStart = rowTile * TILE_DIM;\n";
        ss << "  const " << index_type << " colStart = static_cast<" << index_type << ">(blockIdx.x) * TILE_COL_SCALARS;\n";
        ss << "  const unsigned int packedCol = threadIdx.x;\n";
        ss << "  const " << index_type << " logicalColBase = colStart + static_cast<" << index_type << ">(packedCol) * PACK_SCALARS;\n";
        if (emit_packed_vectorized_path) {
            for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
                const ExprNode& node = stage.expr.nodes[node_idx];
                if (node.op != ExprOp::SCALAR_FP && node.op != ExprOp::FILL) {
                    continue;
                }
                if (emit_mixed_two_byte_float2_path || emit_cross_width_float2_path) {
                    ss << "  const float2 c" << node_idx << " = " << emitFloat2ScalarLiteral(node.scalar_fp) << ";\n";
                } else {
                    ss << "  const " << (vector_compute_dtype == DataType::BF16 ? "__nv_bfloat162" : "half2") << " c" << node_idx << " = "
                       << emitVector2ScalarLiteral(node.scalar_fp, vector_compute_dtype) << ";\n";
                }
            }
        }
        ss << "\n";

        ss << "  for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {\n";
        ss << "    const " << index_type << " logicalRow = rowStart + threadIdx.y + j;\n";
        ss << "    const " << index_type << " idx_base = matrixOffset + logicalRow * numCols + logicalColBase;\n";
        ss << "    const bool inputPackedLoadOk = (logicalRow < numRows) && (logicalColBase + PACK_SCALARS <= numCols) &&\n";
        ss << "                                   ((idx_base % PACK_SCALARS) == " << emitUnsignedLiteral(0, use_uint32_index_math) << ")"
           << (emit_packed_vectorized_path
                   ? emitInputsPackAddressAligned(tensorInputSlots(stage.expr), "idx_base", load_fp8_input_chunk ? "InputPack" : "InputVec2")
                   : std::string{})
           << ";\n";
        ss << "    Pack output_pack{};\n";
        if (emit_packed_vectorized_path) {
            const std::optional<uint32_t> passthrough_slot = passthroughInputSlotForOutput(stage.expr, output.local_node_idx);
            ss << "    if (inputPackedLoadOk) {\n";
            if (load_fp8_input_chunk) {
                for (uint32_t slot : tensorInputSlots(stage.expr)) {
                    ss << "      const InputPack in" << slot << "_chunk = *reinterpret_cast<const InputPack*>(in" << slot << " + idx_base);\n";
                }
            }
            ss << "      OutputVec2* output_vec2 = reinterpret_cast<OutputVec2*>(&output_pack);\n";
            for (uint32_t pair = 0; pair < pairs_per_pack; ++pair) {
                const std::string suffix = "_tp" + std::to_string(pair);
                ss << "      {\n";
                ss << "        constexpr unsigned int PAIR = " << pair << "U;\n";
                if (passthrough_slot.has_value()) {
                    ss << "        output_vec2[PAIR] = " << input_pair_expr(passthrough_slot.value(), "OutputVec2") << ";\n";
                } else if (emit_mixed_two_byte_float2_path || emit_cross_width_float2_path) {
                    emitFloat2NodeDefinitionsForSuffix(
                        ss,
                        stage.expr,
                        input_storage_dtype_vector,
                        suffix,
                        "        ",
                        [&](uint32_t input_slot) { return input_pair_expr(input_slot, "InputVec2"); },
                        [&](uint32_t scalar_node_idx) { return "c" + std::to_string(scalar_node_idx); });
                    ss << "        output_vec2[PAIR] = "
                       << float2_storage_conversion(output_storage_dtype_vector, refWithSuffix(output.local_node_idx, suffix)) << ";\n";
                } else {
                    emitVector2NodeDefinitionsForSuffix(
                        ss,
                        stage.expr,
                        vector_compute_dtype,
                        input_storage_dtype_vector,
                        suffix,
                        "        ",
                        [&](uint32_t input_slot) { return input_pair_expr(input_slot, "InputVec2"); },
                        [&](uint32_t scalar_node_idx) { return "c" + std::to_string(scalar_node_idx); });
                    ss << "        output_vec2[PAIR] = "
                       << vector_storage_conversion(output_storage_dtype_vector, refWithSuffix(output.local_node_idx, suffix), vector_compute_dtype)
                       << ";\n";
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
        ss << "          const " << index_type << " idx = matrixOffset + logicalRow * numCols + logicalCol;\n";
        ss << "          {\n";
        for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
            if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
                continue;
            }
            emitScalarNode(ss, stage.expr, node_idx, /*broadcast_support=*/false, "            ");
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
        emitPackedTransposeTileLaneGather(ss, output_type);
        ss << "    const " << index_type << " out_base_idx = matrixOffset + outputRow * numRows + outputColBase;\n";
        ss << "    const bool outputPackedStoreOk = (outputRow < numCols) && (outputColBase + PACK_SCALARS <= numRows) &&\n";
        ss << "                                     ((out_base_idx % PACK_SCALARS) == " << emitUnsignedLiteral(0, use_uint32_index_math)
           << ") && ((reinterpret_cast<unsigned long long>(out0 + out_base_idx) % sizeof(Pack)) == 0ULL);\n";
        ss << "    if (outputPackedStoreOk) {\n";
        ss << "      Pack* out_ptr = reinterpret_cast<Pack*>(out0 + out_base_idx);\n";
        ss << "      *out_ptr = output_pack;\n";
        ss << "    } else if (outputRow < numCols) {\n";
        ss << "      for (unsigned int lane = 0; lane < PACK_SCALARS; ++lane) {\n";
        ss << "        const " << index_type << " outputCol = outputColBase + lane;\n";
        ss << "        if (outputCol < numRows) {\n";
        ss << "          out0[matrixOffset + outputRow * numRows + outputCol] = output_scalar[lane];\n";
        ss << "        }\n";
        ss << "      }\n";
        ss << "    }\n";
        ss << "  }\n";
        ss << "}\n";
        return ss.str();
    }

    emitSharedTransposeTileDeclaration(ss);
    ss << "  const " << index_type << " x = static_cast<" << index_type << ">(blockIdx.x) * TILE_DIM + threadIdx.x;\n";
    ss << "  const " << index_type << " y = rowTile * TILE_DIM + threadIdx.y;\n";
    ss << "  for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {\n";
    ss << "    const " << index_type << " logical_row = y + j;\n";
    ss << "    const " << index_type << " logical_col = x;\n";
    ss << "    if (logical_col < numCols && logical_row < numRows) {\n";
    ss << "      const " << index_type << " idx = matrixOffset + logical_row * numCols + logical_col;\n";

    for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
        if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
            continue;
        }
        emitScalarNode(ss, stage.expr, node_idx, /*broadcast_support=*/false, "      ");
    }

    ss << "      tile[threadIdx.y + j][threadIdx.x] = thor_pack_transpose_word<" << output_type << ">("
       << emitResolvedScalarValueExpr(stage.expr, output.local_node_idx, output_dtype) << ");\n";
    ss << "    }\n";
    ss << "  }\n\n";
    ss << "  __syncthreads();\n\n";
    ss << "  const " << index_type << " tx = rowTile * TILE_DIM + threadIdx.x;\n";
    ss << "  const " << index_type << " ty = static_cast<" << index_type << ">(blockIdx.x) * TILE_DIM + threadIdx.y;\n";
    ss << "  for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {\n";
    ss << "    if (tx < numRows && (ty + j) < numCols) {\n";
    ss << "      const " << index_type << " out_idx = matrixOffset + (ty + j) * numRows + tx;\n";
    ss << "      out0[out_idx] = thor_unpack_transpose_word<" << output_type << ">(tile[threadIdx.x][threadIdx.y + j]);\n";
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

static std::string emitFlatImpl(const PhysicalExecutionStage& stage, const std::string& kernel_name, bool use_uint32_index_math);

std::string CudaSourceEmitter::emitFlat(const PhysicalExecutionStage& stage, const std::string& kernel_name, bool use_uint32_index_math) {
    if (expressionHasStrengthReductions(stage.expr)) {
        PhysicalExecutionStage reduced = stage;
        applyStrengthReductions(reduced.expr);
        return emitFlatImpl(reduced, kernel_name, use_uint32_index_math);
    }
    return emitFlatImpl(stage, kernel_name, use_uint32_index_math);
}

static std::string emitFlatImpl(const PhysicalExecutionStage& stage, const std::string& kernel_name, bool use_uint32_index_math) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel) {
        throw runtime_error("CudaSourceEmitter::emitFlat(stage, ...) called on non-fused stage.");
    }

    if (stage.outputs.empty()) {
        throw runtime_error("Fused stage has no outputs.");
    }

    const std::optional<RaggedValuewiseExtentEmissionSpec> ragged_extent = findRaggedValuewiseExtentSpec(stage.expr);
    if (ragged_extent.has_value()) {
        validateRaggedExtentStageIsolation(stage, ragged_extent.value());
    }
    if (ragged_extent.has_value() && stageHasTransposedMaterializedOutput(stage.outputs)) {
        throw runtime_error("ragged runtime extent is not supported with transposed materialized outputs.");
    }

    if (stageHasTransposedMaterializedOutput(stage.outputs)) {
        return emitTiledTransposeMaterializedFused(stage, kernel_name, use_uint32_index_math);
    }

    std::optional<DataType> vectorized_dtype = ragged_extent.has_value()
                                                   ? getRaggedFlatVectorizedStageStorageDType(stage)
                                                   : CudaSourceEmitter::getVectorizedStageStorageDType(stage);
    if (vectorized_dtype.has_value()) {
        const std::optional<DataType> vector_compute_dtype = getVectorizedStageComputeDTypeImpl(stage.expr, vectorized_dtype.value());
        if (!vector_compute_dtype.has_value()) {
            throw runtime_error("Vectorized fused stage lost its compute dtype after eligibility selection.");
        }
        return emitVector2Flat(stage, vectorized_dtype.value(), vector_compute_dtype.value(), kernel_name, use_uint32_index_math);
    }

    const uint32_t elements_per_thread = CudaSourceEmitter::flatElementsPerThread(stage);
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

    std::string scalar_indent = "  ";
    if (ragged_extent.has_value()) {
        const auto& extent = ragged_extent.value();
        ss << "  const unsigned long long active_values_raw = static_cast<unsigned long long>(in" << extent.partition_input_slot << "["
           << raggedActiveValueCountLoadIndex(extent) << "]);\n";
        ss << "  const unsigned long long active_values = active_values_raw < " << extent.max_active_values
           << "ULL ? active_values_raw : " << extent.max_active_values << "ULL;\n";
        ss << "  const unsigned long long runtime_numel_u64 = active_values * " << extent.elements_per_value << "ULL;\n";
        ss << "  const " << index_type << " runtime_numel = static_cast<" << index_type
           << ">(runtime_numel_u64 < static_cast<unsigned long long>(numel) ? runtime_numel_u64 : static_cast<unsigned long long>(numel));\n";
        ss << "  " << index_type << " idx = " << emitFlatThreadIndexExpr(use_uint32_index_math) << ";\n";
        ss << "  const " << index_type << " grid_stride = static_cast<" << index_type
           << ">(blockDim.x) * static_cast<" << index_type << ">(gridDim.x);\n";
        ss << "  for (; idx < runtime_numel; idx += grid_stride) {\n";
        scalar_indent = "    ";
    } else {
        ss << "  " << index_type << " idx = " << emitFlatThreadIndexExpr(use_uint32_index_math) << ";";
        ss << "  if (idx >= numel) return;";
    }

    const std::unordered_set<uint32_t> structuralPartitionInputSlots =
        raggedValuewiseExtentPartitionInputSlots(stage.expr);
    const std::unordered_set<uint32_t> index_aware_skip_nodes =
        collectIndexAwareInputNodesToSkipForFlatOutput(stage.expr, stage.outputs);
    for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
        const ExprNode& node = stage.expr.nodes[node_idx];
        const bool isStructuralPartitionInput =
            node.op == ExprOp::INPUT && structuralPartitionInputSlots.contains(node.input_slot);
        if (isStructuralPartitionInput || index_aware_skip_nodes.contains(node_idx) ||
            !shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
            continue;
        }
        emitScalarNode(ss, stage.expr, node_idx, /*broadcast_support=*/false, scalar_indent);
    }

    for (uint32_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
        const CompiledStageOutput& output = stage.outputs[out_idx];
        const DataType output_dtype = requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]);
        ss << scalar_indent << "out" << out_idx << "[idx] = "
           << emitResolvedScalarValueExpr(stage.expr, output.local_node_idx, output_dtype) << ";";
    }
    if (ragged_extent.has_value()) {
        ss << "  }\n";
    }

    ss << "}";
    return ss.str();
}

std::string CudaSourceEmitter::emitPaddedRaggedPointwise(
    const PhysicalExecutionStage& original_stage,
    const std::vector<PaddedRaggedPointwiseInputAccess>& input_access,
    uint64_t batch_size,
    uint64_t channels,
    const std::string& kernel_name) {
    PhysicalExecutionStage reduced_storage;
    const bool reduce = expressionHasStrengthReductions(original_stage.expr);
    if (reduce) {
        reduced_storage = original_stage;
        applyStrengthReductions(reduced_storage.expr);
    }
    const PhysicalExecutionStage& stage = reduce ? reduced_storage : original_stage;
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel) {
        throw runtime_error("emitPaddedRaggedPointwise called on non-fused stage.");
    }
    if (stage.outputs.empty()) {
        throw runtime_error("Padded-ragged pointwise stage requires at least one output.");
    }
    if (input_access.size() != stage.expr.inputs.size()) {
        throw runtime_error("Padded-ragged pointwise input-access metadata does not match expression inputs.");
    }
    if (batch_size == 0 || channels == 0 || batch_size > UINT32_MAX || channels > UINT32_MAX ||
        batch_size > static_cast<uint64_t>(UINT32_MAX) / channels) {
        throw runtime_error("Padded-ragged pointwise B*C geometry must fit uint32_t device indexing.");
    }

    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const std::vector<DataType> output_dtypes = collectOutputDTypes(stage);
    std::ostringstream ss;
    emitRequiredHeaders(stage.expr, ss);

    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";
    bool first_arg = true;
    for (uint32_t i = 0; i < input_dtypes.size(); ++i) {
        if (!first_arg) ss << ", ";
        first_arg = false;
        if (stage.expr.inputs[i].kind == NamedInput::Kind::RuntimeScalarFp32) {
            ss << scalarStorageType(input_dtypes[i]) << " in" << i;
        } else {
            ss << "const " << scalarStorageType(input_dtypes[i]) << "* in" << i;
        }
    }
    for (uint32_t i = 0; i < output_dtypes.size(); ++i) {
        if (!first_arg) ss << ", ";
        first_arg = false;
        ss << scalarStorageType(output_dtypes[i]) << "* out" << i;
    }
    if (!first_arg) ss << ", ";
    ss << "unsigned int numel) {\n";
    ss << "  const unsigned int idx = static_cast<unsigned int>(blockIdx.x) * static_cast<unsigned int>(blockDim.x) + "
          "static_cast<unsigned int>(threadIdx.x);\n";
    ss << "  if (idx >= numel || numel == 0U) return;\n";
    ss << "  constexpr unsigned int PADDED_BATCH = " << static_cast<uint32_t>(batch_size) << "U;\n";
    ss << "  constexpr unsigned int PADDED_CHANNELS = " << static_cast<uint32_t>(channels) << "U;\n";
    ss << "  constexpr unsigned int PADDED_ROW_CHANNELS = PADDED_BATCH * PADDED_CHANNELS;\n";
    ss << "  const unsigned int width = numel / PADDED_ROW_CHANNELS;\n";
    ss << "  if (width == 0U || width * PADDED_ROW_CHANNELS != numel) return;\n";
    ss << "  const unsigned int channel = (idx / width) % PADDED_CHANNELS;\n";

    auto input_value = [&](uint32_t input_slot) -> std::string {
        if (input_slot >= input_access.size()) {
            throw runtime_error("Padded-ragged pointwise input slot is out of range.");
        }
        switch (input_access[input_slot]) {
            case PaddedRaggedPointwiseInputAccess::PaddedValue:
                return "in" + std::to_string(input_slot) + "[idx]";
            case PaddedRaggedPointwiseInputAccess::ChannelBroadcast:
                return "in" + std::to_string(input_slot) + "[channel]";
            case PaddedRaggedPointwiseInputAccess::ScalarTensor:
                return "in" + std::to_string(input_slot) + "[0]";
            case PaddedRaggedPointwiseInputAccess::MetadataTensor:
                return "in" + std::to_string(input_slot) + "[0]";
        }
        throw runtime_error("Unknown padded-ragged pointwise input access mode.");
    };

    for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
        const ExprNode& node = stage.expr.nodes[node_idx];
        if (node.op == ExprOp::INPUT && node.input_slot < input_access.size() &&
            input_access[node.input_slot] == PaddedRaggedPointwiseInputAccess::MetadataTensor) {
            continue;
        }
        if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
            continue;
        }
        emitScalarNodeSuffixed(ss, stage.expr, node_idx, "idx", "", "  ", "", 0, input_value);
    }

    for (uint32_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
        const CompiledStageOutput& output = stage.outputs[out_idx];
        if (output.local_node_idx >= stage.expr.nodes.size()) {
            throw runtime_error("Padded-ragged pointwise output node is out of range.");
        }
        const DataType output_dtype = requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]);
        ss << "  out" << out_idx << "[idx] = "
           << emitResolvedScalarValueExprSuffixed(stage.expr, output.local_node_idx, output_dtype, "") << ";\n";
    }
    ss << "}\n";
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
    if (group.output_dims.size() < 2) {
        throw runtime_error("Transposed broadcast fused materialization requires a rank >= 2 logical output.");
    }

    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const DataType output_dtype = requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]);
    if (scalarStorageTypeSizeBytes(output_dtype) > sizeof(unsigned int)) {
        throw runtime_error("Tiled transpose shared-memory swizzle currently supports output scalar storage types up to 32 bits.");
    }
    const std::string output_type = scalarStorageType(output_dtype);
    const bool emit_packed_low_precision_path = transposePackScalars(output_dtype) > 1;
    const std::optional<DataType> maybe_tensor_input_dtype = getSingleTensorInputStorageDType(stage.expr, input_dtypes);
    const bool emit_decoupled_line_vectorized_path = shouldUseDecoupledLineVectorizedTranspose(maybe_tensor_input_dtype, output_dtype);
    const std::optional<DataType> maybe_vectorized_dtype = CudaSourceEmitter::getVectorizedStageStorageDType(stage);
    const std::optional<DataType> maybe_vectorized_compute_dtype =
        maybe_vectorized_dtype.has_value() ? getVectorizedStageComputeDTypeImpl(stage.expr, maybe_vectorized_dtype.value()) : std::nullopt;
    const bool emit_fp8_vector_compute_pair_path = emit_packed_low_precision_path && isFp8DType(output_dtype) &&
                                                   maybe_vectorized_dtype.has_value() && maybe_vectorized_dtype.value() == output_dtype &&
                                                   maybe_vectorized_compute_dtype.has_value();
    // FP8 storage computing in FP32 (the default policy): same paired loads/stores,
    // float2 arithmetic.
    const bool emit_fp8_float2_pair_path = [&] {
        if (!emit_packed_low_precision_path || !isFp8DType(output_dtype) || emit_fp8_vector_compute_pair_path ||
            !expressionUsesOnlyFloat2PathOps(stage.expr)) {
            return false;
        }
        for (const ExprNode& node : stage.expr.nodes) {
            if (node.op == ExprOp::INPUT) {
                if (requireNodeInputTensorDType(node) != output_dtype) return false;
            } else if (node.op != ExprOp::SCALAR_FP && node.op != ExprOp::FILL && node.op != ExprOp::RUNTIME_SCALAR &&
                       node.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
                if (requireNodeComputeDType(node) != DataType::FP32 || requireNodeOutputDType(node) != output_dtype) return false;
            }
        }
        return true;
    }();
    const bool emit_fp8_vectorized_pair_path = emit_fp8_vector_compute_pair_path || emit_fp8_float2_pair_path;
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
    emitSharedTransposeWordHelpers(ss);
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
    ss << output_type << "* __restrict__ out0, " << index_type << " numRows, " << index_type << " numCols, " << index_type
       << " batchCount) {\n";
    ss << "  const " << index_type << " rowTiles = (numRows + static_cast<" << index_type << ">(TILE_DIM) - 1) / static_cast<" << index_type
       << ">(TILE_DIM);\n";
    ss << "  const " << index_type << " flatBlockY = static_cast<" << index_type << ">(blockIdx.y)"
       << " + static_cast<" << index_type << ">(blockIdx.z) * static_cast<" << index_type << ">(gridDim.y);\n";
    ss << "  const " << index_type << " batchIdx = flatBlockY / rowTiles;\n";
    ss << "  if (batchIdx >= batchCount) return;\n";
    ss << "  const " << index_type << " rowTile = flatBlockY - batchIdx * rowTiles;\n";
    ss << "  const " << index_type << " matrixOffset = batchIdx * numRows * numCols;\n\n";

    if (emit_decoupled_line_vectorized_path) {
        const DataType input_dtype = maybe_tensor_input_dtype.value();
        const uint32_t read_pack_scalars = transposePackScalars(input_dtype);
        const uint32_t write_pack_scalars = transposePackScalars(output_dtype);
        const std::string input_type = scalarStorageType(input_dtype);

        ss << "  constexpr unsigned int READ_PACK_SCALARS = " << read_pack_scalars << "U;\n";
        ss << "  constexpr unsigned int WRITE_PACK_SCALARS = " << write_pack_scalars << "U;\n";
        ss << "  constexpr unsigned int TILE_COL_SCALARS = TILE_DIM;\n";
        if (read_pack_scalars > 1) {
            ss << "  using InputPack = " << (isFp8DType(input_dtype) ? std::string("uchar4") : transposePackType(input_dtype)) << ";\n";
        }
        if (write_pack_scalars > 1) {
            ss << "  using OutputPack = " << transposePackType(output_dtype) << ";\n";
        }
        emitSharedTransposeTileDeclaration(ss, "TILE_COL_SCALARS + 1");
        ss << "  const " << index_type << " rowStart = rowTile * TILE_DIM;\n";
        ss << "  const " << index_type << " colStart = static_cast<" << index_type << ">(blockIdx.x) * TILE_COL_SCALARS;\n";
        ss << "  const unsigned int threadLinear = threadIdx.y * TILE_DIM + threadIdx.x;\n\n";

        ss << "  constexpr unsigned int READ_PACKS_PER_ROW = TILE_COL_SCALARS / READ_PACK_SCALARS;\n";
        ss << "  constexpr unsigned int READ_TASKS_PER_TILE = TILE_DIM * READ_PACKS_PER_ROW;\n";
        ss << "  for (unsigned int readTask = threadLinear; readTask < READ_TASKS_PER_TILE; readTask += TILE_DIM * BLOCK_ROWS) {\n";
        ss << "    const unsigned int localRow = readTask / READ_PACKS_PER_ROW;\n";
        ss << "    const unsigned int localReadPackCol = readTask - localRow * READ_PACKS_PER_ROW;\n";
        ss << "    const " << index_type << " logicalRow = rowStart + static_cast<" << index_type << ">(localRow);\n";
        ss << "    const " << index_type << " logicalColBase = colStart + static_cast<" << index_type
           << ">(localReadPackCol) * READ_PACK_SCALARS;\n";
        ss << "    const " << index_type << " logical_idx_base = matrixOffset + static_cast<" << index_type
           << ">(logicalRow) * static_cast<" << index_type << ">(numCols) + static_cast<" << index_type << ">(logicalColBase);\n";

        auto emit_broadcast_offsets = [&](const std::vector<size_t>& used_indices,
                                          const std::string& logical_idx_expr,
                                          const std::string& suffix,
                                          const std::string& indent) {
            for (size_t used_i : used_indices) {
                const uint32_t input_slot = group.used_input_slots[used_i];
                ss << indent << index_type << " in" << input_slot << "_offset" << suffix << " = "
                   << emitUnsignedLiteral(0, use_uint32_index_math) << ";\n";
            }
            if (!group.active_axes.empty()) {
                ss << "\n";
                emitSpecializedBroadcastOffsetMath(ss, group, used_indices, logical_idx_expr, suffix, indent, use_uint32_index_math);
                ss << "\n";
            }
        };

        std::vector<size_t> native_vector_indices;
        native_vector_indices.reserve(group.used_input_slots.size());
        for (size_t used_i = 0; used_i < group.used_input_slots.size(); ++used_i) {
            if (group.used_input_load_kinds.at(used_i) == SpecializedInputLoadKind::NativeVector) {
                native_vector_indices.push_back(used_i);
            }
        }

        if (read_pack_scalars > 1) {
            emit_broadcast_offsets(all_used_indices, "logical_idx_base", "_base", "    ");
            ss << "    bool inputPackedLoadOk = (logicalRow < numRows) && (logicalColBase + READ_PACK_SCALARS <= numCols);\n";
            for (size_t used_i : native_vector_indices) {
                const uint32_t input_slot = group.used_input_slots[used_i];
                ss << "    inputPackedLoadOk = inputPackedLoadOk && ((in" << input_slot
                   << "_offset_base % READ_PACK_SCALARS) == " << emitUnsignedLiteral(0, use_uint32_index_math)
                   << ") && ((reinterpret_cast<unsigned long long>(in" << input_slot << " + in" << input_slot
                   << "_offset_base) % sizeof(InputPack)) == 0ULL);\n";
            }
            ss << "    if (inputPackedLoadOk) {\n";
            for (size_t used_i : native_vector_indices) {
                const uint32_t input_slot = group.used_input_slots[used_i];
                ss << "      const InputPack in" << input_slot << "_chunk = reinterpret_cast<const InputPack*>(in" << input_slot << " + in"
                   << input_slot << "_offset_base)[0];\n";
                if (!isFp8DType(input_dtype)) {
                    ss << "      const " << input_type << "* in" << input_slot << "_chunk_data = reinterpret_cast<const " << input_type
                       << "*>(&in" << input_slot << "_chunk);\n";
                }
            }
            if (isFp8DType(input_dtype)) {
                // Native-vector FP8 chunks decode as two fp8x2 pairs; constant lane indices only.
                std::vector<uint32_t> native_fp8_slots;
                for (size_t used_i : native_vector_indices) {
                    native_fp8_slots.push_back(group.used_input_slots[used_i]);
                }
                const Fp8LanePairing fp8_pairing =
                    planFp8LanePairing(stage.expr, stage.outputs, {0u}, native_fp8_slots, read_pack_scalars);
                emitFp8PairedInputDecode(
                    ss, "      ", fp8_pairing, input_dtypes, [](uint32_t slot) { return "in" + std::to_string(slot) + "_chunk"; },
                    read_pack_scalars);
                for (uint32_t lane = 0; lane < read_pack_scalars; ++lane) {
                    const std::string suffix = "_dbl" + std::to_string(lane);
                    const std::string lane_literal = emitUnsignedLiteral(lane, use_uint32_index_math);
                    ss << "      {\n";
                    ss << "        const unsigned int LANE = " << lane << "U;\n";
                    ss << "        const unsigned int LOCAL_COL = localReadPackCol * READ_PACK_SCALARS + LANE;\n";
                    ss << "        const " << index_type << " logical_idx = logical_idx_base + " << lane_literal << ";\n";
                    emit_broadcast_offsets(scalar_pack_indices, "logical_idx", "", "        ");
                    auto input_value = [&](uint32_t input_slot) -> std::string {
                        const auto kind_it = input_load_kind_by_slot.find(input_slot);
                        if (kind_it == input_load_kind_by_slot.end()) {
                            throw std::runtime_error("Missing input load kind for decoupled transposed broadcast input.");
                        }
                        if (fp8_pairing.decoded_slots.contains(input_slot)) {
                            return fp8DecodedLaneExpr(fp8_pairing, input_slot, lane);
                        }
                        if (kind_it->second == SpecializedInputLoadKind::NativeVector) {
                            return emitFp8PackLaneMemberExpr("in" + std::to_string(input_slot) + "_chunk", lane, input_dtype);
                        }
                        return "in" + std::to_string(input_slot) + "[in" + std::to_string(input_slot) + "_offset]";
                    };
                    for (uint32_t node_idx = 0; node_idx < fp8_pairing.lane_expr.nodes.size(); ++node_idx) {
                        if (!shouldEmitScalarNodeDefinition(fp8_pairing.lane_expr, node_idx)) {
                            continue;
                        }
                        emitScalarNodeSuffixed(ss, fp8_pairing.lane_expr, node_idx, "logical_idx", suffix, "        ", "", 0, input_value);
                    }
                    ss << "        tile[localRow][LOCAL_COL] = thor_pack_transpose_word<" << output_type << ">("
                       << emitResolvedScalarValueExprSuffixed(fp8_pairing.lane_expr, output.local_node_idx, output_dtype, suffix)
                       << ");\n";
                    ss << "      }\n";
                }
            } else {
                ss << "      for (unsigned int laneIter = 0; laneIter < READ_PACK_SCALARS; ++laneIter) {\n";
                ss << "        const unsigned int LANE = ((localReadPackCol + laneIter) & (READ_PACK_SCALARS - 1U));\n";
                ss << "        const unsigned int LOCAL_COL = localReadPackCol * READ_PACK_SCALARS + LANE;\n";
                ss << "        const " << index_type << " logical_idx = logical_idx_base + static_cast<" << index_type << ">(LANE);\n";
                emit_broadcast_offsets(scalar_pack_indices, "logical_idx", "", "        ");
                auto input_value = [&](uint32_t input_slot) -> std::string {
                    const auto kind_it = input_load_kind_by_slot.find(input_slot);
                    if (kind_it == input_load_kind_by_slot.end()) {
                        throw std::runtime_error("Missing input load kind for decoupled transposed broadcast input.");
                    }
                    if (kind_it->second == SpecializedInputLoadKind::NativeVector) {
                        return emitChunkLaneSelectExpr("in" + std::to_string(input_slot) + "_chunk_data", "LANE", read_pack_scalars);
                    }
                    return "in" + std::to_string(input_slot) + "[in" + std::to_string(input_slot) + "_offset]";
                };
                for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
                    if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
                        continue;
                    }
                    emitScalarNodeSuffixed(ss, stage.expr, node_idx, "logical_idx", "_dbl", "        ", "", 0, input_value);
                }
                ss << "        tile[localRow][LOCAL_COL] = thor_pack_transpose_word<" << output_type << ">("
                   << emitResolvedScalarValueExprSuffixed(stage.expr, output.local_node_idx, output_dtype, "_dbl") << ");\n";
                ss << "      }\n";
            }
            ss << "    } else if (logicalRow < numRows) {\n";
        } else {
            ss << "    if (logicalRow < numRows) {\n";
        }

        ss << "      for (unsigned int laneIter = 0; laneIter < READ_PACK_SCALARS; ++laneIter) {\n";
        ss << "        const unsigned int LANE = ((localReadPackCol + laneIter) & (READ_PACK_SCALARS - 1U));\n";
        ss << "        const unsigned int LOCAL_COL = localReadPackCol * READ_PACK_SCALARS + LANE;\n";
        ss << "        const " << index_type << " logicalCol = logicalColBase + static_cast<" << index_type << ">(LANE);\n";
        ss << "        if (logicalCol < numCols) {\n";
        ss << "          const " << index_type << " logical_idx = matrixOffset + static_cast<" << index_type
           << ">(logicalRow) * static_cast<" << index_type << ">(numCols) + static_cast<" << index_type << ">(logicalCol);\n";
        emit_broadcast_offsets(all_used_indices, "logical_idx", "", "          ");
        auto input_value = [&](uint32_t input_slot) -> std::string {
            return "in" + std::to_string(input_slot) + "[in" + std::to_string(input_slot) + "_offset]";
        };
        for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
            if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
                continue;
            }
            emitScalarNodeSuffixed(ss, stage.expr, node_idx, "logical_idx", "_dbs", "          ", "", 0, input_value);
        }
        ss << "          tile[localRow][LOCAL_COL] = thor_pack_transpose_word<" << output_type << ">("
           << emitResolvedScalarValueExprSuffixed(stage.expr, output.local_node_idx, output_dtype, "_dbs") << ");\n";
        ss << "        }\n";
        ss << "      }\n";
        ss << "    }\n";
        ss << "  }\n\n";
        ss << "  __syncthreads();\n\n";

        ss << "  constexpr unsigned int WRITE_PACKS_PER_ROW = TILE_DIM / WRITE_PACK_SCALARS;\n";
        ss << "  constexpr unsigned int WRITE_TASKS_PER_TILE = TILE_COL_SCALARS * WRITE_PACKS_PER_ROW;\n";
        ss << "  for (unsigned int writeTask = threadLinear; writeTask < WRITE_TASKS_PER_TILE; writeTask += TILE_DIM * BLOCK_ROWS) {\n";
        ss << "    const unsigned int localOutRow = writeTask / WRITE_PACKS_PER_ROW;\n";
        ss << "    const unsigned int localOutPackCol = writeTask - localOutRow * WRITE_PACKS_PER_ROW;\n";
        ss << "    const " << index_type << " outputRow = colStart + static_cast<" << index_type << ">(localOutRow);\n";
        ss << "    const " << index_type << " outputColBase = rowStart + static_cast<" << index_type
           << ">(localOutPackCol) * WRITE_PACK_SCALARS;\n";
        if (write_pack_scalars > 1) {
            ss << "    OutputPack output_pack{};\n";
            ss << "    " << output_type << "* output_pack_data = reinterpret_cast<" << output_type << "*>(&output_pack);\n";
            ss << "    for (unsigned int lane = 0; lane < WRITE_PACK_SCALARS; ++lane) {\n";
            ss << "      const " << index_type << " outputCol = outputColBase + static_cast<" << index_type << ">(lane);\n";
            ss << "      if (outputRow < numCols && outputCol < numRows) {\n";
            ss << "        output_pack_data[lane] = thor_unpack_transpose_word<" << output_type
               << ">(tile[localOutPackCol * WRITE_PACK_SCALARS + lane][localOutRow]);\n";
            ss << "      }\n";
            ss << "    }\n";
            ss << "    const " << index_type << " out_base_idx = matrixOffset + outputRow * numRows + outputColBase;\n";
            ss << "    const bool outputPackedStoreOk = (outputRow < numCols) && (outputColBase + WRITE_PACK_SCALARS <= numRows) &&\n";
            ss << "                                     ((out_base_idx % WRITE_PACK_SCALARS) == "
               << emitUnsignedLiteral(0, use_uint32_index_math)
               << ") && ((reinterpret_cast<unsigned long long>(out0 + out_base_idx) % sizeof(OutputPack)) == 0ULL);\n";
            ss << "    if (outputPackedStoreOk) {\n";
            ss << "      OutputPack* out_ptr = reinterpret_cast<OutputPack*>(out0 + out_base_idx);\n";
            ss << "      *out_ptr = output_pack;\n";
            ss << "    } else if (outputRow < numCols) {\n";
            ss << "      for (unsigned int lane = 0; lane < WRITE_PACK_SCALARS; ++lane) {\n";
            ss << "        const " << index_type << " outputCol = outputColBase + static_cast<" << index_type << ">(lane);\n";
            ss << "        if (outputCol < numRows) {\n";
            ss << "          out0[matrixOffset + outputRow * numRows + outputCol] = output_pack_data[lane];\n";
            ss << "        }\n";
            ss << "      }\n";
            ss << "    }\n";
        } else {
            ss << "    if (outputRow < numCols && outputColBase < numRows) {\n";
            ss << "      out0[matrixOffset + outputRow * numRows + outputColBase] = thor_unpack_transpose_word<" << output_type
               << ">(tile[localOutPackCol][localOutRow]);\n";
            ss << "    }\n";
        }
        ss << "  }\n";
        ss << "}\n";
        return ss.str();
    }

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
        emitSharedTransposeTileDeclaration(ss);
        ss << "  const " << index_type << " rowStart = rowTile * TILE_DIM;\n";
        ss << "  const " << index_type << " colStart = static_cast<" << index_type << ">(blockIdx.x) * TILE_COL_SCALARS;\n";
        ss << "  const unsigned int packedCol = threadIdx.x;\n";
        ss << "  const " << index_type << " logicalColBase = colStart + static_cast<" << index_type << ">(packedCol) * PACK_SCALARS;\n";
        if (emit_fp8_vector_compute_pair_path) {
            for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
                const ExprNode& node = stage.expr.nodes[node_idx];
                if (node.op != ExprOp::SCALAR_FP && node.op != ExprOp::FILL) {
                    continue;
                }
                const DataType vector_compute_dtype = maybe_vectorized_compute_dtype.value();
                ss << "  const " << (vector_compute_dtype == DataType::BF16 ? "__nv_bfloat162" : "half2") << " c" << node_idx << " = "
                   << emitVector2ScalarLiteral(node.scalar_fp, vector_compute_dtype) << ";\n";
            }
        }
        ss << "\n";

        ss << "  for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {\n";
        ss << "    const " << index_type << " logicalRow = rowStart + threadIdx.y + j;\n";
        ss << "    Pack output_pack{};\n";
        ss << "    if (logicalRow < numRows) {\n";
        if (emit_fp8_vectorized_pair_path) {
            const uint32_t pairs_per_pack = pack_scalars / 2U;
            const std::optional<uint32_t> passthrough_slot = passthroughInputSlotForOutput(stage.expr, output.local_node_idx);
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
                ss << "          const " << index_type << " logical_idx" << suffix0 << " = matrixOffset + static_cast<" << index_type
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
                const std::string vec2_type = transposeVector2StorageType(output_dtype);
                if (passthrough_slot.has_value()) {
                    const uint32_t input_slot = passthrough_slot.value();
                    const auto kind_it = input_load_kind_by_slot.find(input_slot);
                    if (kind_it == input_load_kind_by_slot.end()) {
                        throw std::runtime_error("Missing input load kind for transposed broadcast passthrough input.");
                    }
                    const std::string base = "in" + std::to_string(input_slot);
                    const std::string from_lanes =
                        output_dtype == DataType::FP8_E4M3 ? "thor_fp8x2_e4m3_from_lanes" : "thor_fp8x2_e5m2_from_lanes";
                    ss << "          output_vec2[PAIR] = "
                       << (kind_it->second == SpecializedInputLoadKind::NativeVector
                               ? fp8TransposePairRawExpr(output_dtype, vec2_type, base, base + "_offset" + suffix0)
                               : from_lanes + "(" + base + "[" + base + "_offset" + suffix0 + "], " + base + "[" + base + "_offset" +
                                     suffix1 + "])")
                       << ";\n";
                } else if (emit_fp8_float2_pair_path) {
                    emitFloat2NodeDefinitionsForSuffix(
                        ss,
                        stage.expr,
                        vec2_type,
                        suffix,
                        "          ",
                        [&](uint32_t input_slot) {
                            const auto kind_it = input_load_kind_by_slot.find(input_slot);
                            if (kind_it == input_load_kind_by_slot.end()) {
                                throw std::runtime_error("Missing input load kind for transposed float2 broadcast input.");
                            }
                            const std::string base = "in" + std::to_string(input_slot);
                            if (kind_it->second == SpecializedInputLoadKind::NativeVector) {
                                // Element-aligned storage aliases may not be pair aligned.
                                const std::string offset0 = base + "_offset" + suffix0;
                                return "static_cast<float2>(" + fp8TransposePairRawExpr(output_dtype, vec2_type, base, offset0) + ")";
                            }
                            return "make_float2(float(" + base + "[" + base + "_offset" + suffix0 + "]), float(" + base + "[" + base +
                                   "_offset" + suffix1 + "]))";
                        },
                        {},
                        /*input_slot_value_is_compute=*/true);
                    ss << "          output_vec2[PAIR] = "
                       << float2_storage_conversion(vec2_type, refWithSuffix(output.local_node_idx, suffix)) << ";\n";
                } else {
                    const DataType vector_compute_dtype = maybe_vectorized_compute_dtype.value();
                    emitVector2NodeDefinitionsForSuffix(
                        ss,
                        stage.expr,
                        vector_compute_dtype,
                        transposeVector2StorageType(output_dtype),
                        suffix,
                        "          ",
                        [&](uint32_t input_slot) {
                            const auto kind_it = input_load_kind_by_slot.find(input_slot);
                            if (kind_it == input_load_kind_by_slot.end()) {
                                throw std::runtime_error("Missing input load kind for transposed vector broadcast input.");
                            }
                            if (kind_it->second == SpecializedInputLoadKind::NativeVector) {
                                // Element-aligned storage aliases may not be pair aligned.
                                const std::string base = "in" + std::to_string(input_slot);
                                const std::string offset0 = base + "_offset" + suffix0;
                                const std::string pair_type = transposeVector2StorageType(output_dtype);
                                return vector_compute_conversion(
                                    pair_type, fp8TransposePairRawExpr(output_dtype, pair_type, base, offset0), vector_compute_dtype);
                            }
                            const std::string var0 =
                                "in" + std::to_string(input_slot) + "[in" + std::to_string(input_slot) + "_offset" + suffix0 + "]";
                            const std::string var1 =
                                "in" + std::to_string(input_slot) + "[in" + std::to_string(input_slot) + "_offset" + suffix1 + "]";
                            return emitVector2BroadcastPackLoad(output_type, var0, var1, vector_compute_dtype);
                        },
                        [&](uint32_t scalar_node_idx) { return "c" + std::to_string(scalar_node_idx); },
                        true);
                    ss << "          output_vec2[PAIR] = "
                       << vector_storage_conversion(
                              transposeVector2StorageType(output_dtype), refWithSuffix(output.local_node_idx, suffix), vector_compute_dtype)
                       << ";\n";
                }
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
        ss << "          const " << index_type << " logical_idx = matrixOffset + static_cast<" << index_type
           << ">(logicalRow) * static_cast<" << index_type << ">(numCols) + static_cast<" << index_type << ">(logicalCol);\n";
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
            emitScalarNode(ss, stage.expr, node_idx, /*broadcast_support=*/true, "            ", &group);
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
        ss << "  for (unsigned int packedStore = threadLinear; packedStore < PACKED_STORES_PER_TILE; packedStore += TILE_DIM * "
              "BLOCK_ROWS) "
              "{\n";
        ss << "    const unsigned int localOutRow = packedStore / PACKED_OUTPUT_COLS_PER_TILE;\n";
        ss << "    const unsigned int localOutPackCol = packedStore - localOutRow * PACKED_OUTPUT_COLS_PER_TILE;\n";
        ss << "    const unsigned int inputPackCol = localOutRow / PACK_SCALARS;\n";
        ss << "    const unsigned int inputColLane = localOutRow - inputPackCol * PACK_SCALARS;\n";
        ss << "    const " << index_type << " outputRow = colStart + localOutRow;\n";
        ss << "    const " << index_type << " outputColBase = rowStart + static_cast<" << index_type
           << ">(localOutPackCol) * PACK_SCALARS;\n";
        emitPackedTransposeTileLaneGather(ss, output_type);
        ss << "    const " << index_type << " out_base_idx = matrixOffset + static_cast<" << index_type << ">(outputRow) * static_cast<"
           << index_type << ">(numRows) +\n";
        ss << "                                     static_cast<" << index_type << ">(outputColBase);\n";
        ss << "    const bool outputPackedStoreOk = (outputRow < numCols) && (outputColBase + PACK_SCALARS <= numRows) &&\n";
        ss << "                                     ((out_base_idx % PACK_SCALARS) == " << emitUnsignedLiteral(0, use_uint32_index_math)
           << ") && ((reinterpret_cast<unsigned long long>(out0 + out_base_idx) % sizeof(Pack)) == 0ULL);\n";
        ss << "    if (outputPackedStoreOk) {\n";
        ss << "      Pack* out_ptr = reinterpret_cast<Pack*>(out0 + out_base_idx);\n";
        ss << "      *out_ptr = output_pack;\n";
        ss << "    } else if (outputRow < numCols) {\n";
        ss << "      for (unsigned int lane = 0; lane < PACK_SCALARS; ++lane) {\n";
        ss << "        const " << index_type << " outputCol = outputColBase + lane;\n";
        ss << "        if (outputCol < numRows) {\n";
        ss << "          out0[matrixOffset + outputRow * numRows + outputCol] = output_scalar[lane];\n";
        ss << "        }\n";
        ss << "      }\n";
        ss << "    }\n";
        ss << "  }\n";
        ss << "}\n";
        return ss.str();
    }

    emitSharedTransposeTileDeclaration(ss);
    ss << "  const " << index_type << " x = static_cast<" << index_type << ">(blockIdx.x) * TILE_DIM + threadIdx.x;\n";
    ss << "  const " << index_type << " y = rowTile * TILE_DIM + threadIdx.y;\n";
    ss << "  for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {\n";
    ss << "    const " << index_type << " logical_row = y + j;\n";
    ss << "    const " << index_type << " logical_col = x;\n";
    ss << "    if (logical_col < numCols && logical_row < numRows) {\n";
    ss << "      const " << index_type << " logical_idx = matrixOffset + static_cast<" << index_type << ">(logical_row) * static_cast<"
       << index_type << ">(numCols) + static_cast<" << index_type << ">(logical_col);\n";
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
        emitScalarNode(ss, stage.expr, node_idx, /*broadcast_support=*/true, "      ", &group);
    }
    ss << "      tile[threadIdx.y + j][threadIdx.x] = thor_pack_transpose_word<" << output_type << ">("
       << emitResolvedScalarValueExpr(stage.expr, output.local_node_idx, output_dtype) << ");\n";
    ss << "    }\n";
    ss << "  }\n\n";
    ss << "  __syncthreads();\n\n";
    ss << "  const " << index_type << " tx = rowTile * TILE_DIM + threadIdx.x;\n";
    ss << "  const " << index_type << " ty = static_cast<" << index_type << ">(blockIdx.x) * TILE_DIM + threadIdx.y;\n";
    ss << "  for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {\n";
    ss << "    if (tx < numRows && (ty + j) < numCols) {\n";
    ss << "      const " << index_type << " out_idx = matrixOffset + static_cast<" << index_type << ">(ty + j) * static_cast<" << index_type
       << ">(numRows) +\n";
    ss << "                                   static_cast<" << index_type << ">(tx);\n";
    ss << "      out0[out_idx] = thor_unpack_transpose_word<" << output_type << ">(tile[threadIdx.x][threadIdx.y + j]);\n";
    ss << "    }\n";
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

static std::string emitTiledLogicalTransposeConsumerSpecializedBroadcast(const CompiledExecutionStage& stage,
                                                                         const std::vector<SpecializedBroadcastGroup>& groups,
                                                                         const std::string& kernel_name) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        throw runtime_error("emitTiledLogicalTransposeConsumerSpecializedBroadcast called on non-fused stage.");
    }
    const std::optional<std::vector<uint32_t>> maybe_frontiers = tryFindTiledLogicalTransposeConsumerFrontiers(stage, groups);
    if (!maybe_frontiers.has_value() || maybe_frontiers->empty()) {
        throw runtime_error("Tiled logical-transpose consumer emitter was selected without supported auto-swizzle frontiers.");
    }

    const std::vector<uint32_t>& frontier_indices = maybe_frontiers.value();
    const SpecializedBroadcastGroup& group = groups[0];

    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const std::vector<DataType> output_dtypes = collectOutputDTypes(stage);
    const bool use_uint32_index_math = groupSupportsUInt32IndexMath(group);
    const std::string index_type = emittedIndexType(use_uint32_index_math);
    const uint32_t logical_transpose_slot_bytes = tiledLogicalTransposeConsumerSlotBytes(stage, groups);
    const uint32_t logical_transpose_pack_scalars = CudaSourceEmitter::tiledLogicalTransposeConsumerPackScalars(stage, groups);

    std::ostringstream ss;
    emitRequiredHeaders(stage.expr, ss);
    emitSharedTransposeWordHelpers(ss);
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

    for (uint32_t i = 0; i < output_dtypes.size(); ++i) {
        if (!first_arg) {
            ss << ", ";
        }
        first_arg = false;
        ss << scalarStorageType(output_dtypes[i]) << "* __restrict__ out" << i;
    }

    if (!first_arg) {
        ss << ", ";
    }
    ss << index_type << " numRows, " << index_type << " numCols, " << index_type << " batchCount) {\n";
    ss << "  static constexpr unsigned int LOGICAL_TRANSPOSE_SLOT_BYTES = " << logical_transpose_slot_bytes << "U;\n";
    ss << "  static constexpr unsigned int LOGICAL_TRANSPOSE_PACK_SCALARS = " << logical_transpose_pack_scalars << "U;\n";
    ss << "  static constexpr unsigned int LOGICAL_TRANSPOSE_TILE_COL_SCALARS = TILE_DIM * LOGICAL_TRANSPOSE_PACK_SCALARS;\n";
    ss << "  const " << index_type << " rowTiles = (numRows + static_cast<" << index_type << ">(TILE_DIM) - 1) / static_cast<" << index_type
       << ">(TILE_DIM);\n";
    ss << "  const " << index_type << " flatBlockY = static_cast<" << index_type << ">(blockIdx.y)"
       << " + static_cast<" << index_type << ">(blockIdx.z) * static_cast<" << index_type << ">(gridDim.y);\n";
    ss << "  const " << index_type << " batchIdx = flatBlockY / rowTiles;\n";
    ss << "  if (batchIdx >= batchCount) return;\n";
    ss << "  const " << index_type << " rowTile = flatBlockY - batchIdx * rowTiles;\n";
    ss << "  const " << index_type << " matrixOffset = batchIdx * numRows * numCols;\n";

    for (uint32_t frontier_idx : frontier_indices) {
        const DataType frontier_dtype = tiledLogicalTransposeFrontierStorageDType(stage.expr, frontier_idx);
        const uint32_t frontier_bytes = scalarStorageTypeSizeBytes(frontier_dtype);
        if (frontier_bytes > sizeof(unsigned int)) {
            throw runtime_error("Tiled logical-transpose auto-swizzle currently supports frontier scalar storage types up to 32 bits.");
        }
        if (frontier_bytes > logical_transpose_slot_bytes) {
            throw runtime_error("Tiled logical-transpose auto-swizzle selected a slot size smaller than a frontier scalar.");
        }
    }
    emitSharedTransposeTileDeclaration(ss);
    ss << "  static constexpr unsigned int FRONTIER_VALUE_SLOTS = TILE_DIM / BLOCK_ROWS;\n";
    for (uint32_t frontier_idx : frontier_indices) {
        const DataType frontier_dtype = tiledLogicalTransposeFrontierStorageDType(stage.expr, frontier_idx);
        ss << "  " << scalarStorageType(frontier_dtype) << " transposed_value_" << frontier_idx
           << "[FRONTIER_VALUE_SLOTS][LOGICAL_TRANSPOSE_PACK_SCALARS];\n";
    }
    ss << "\n";

    ss << "  const " << index_type << " rowStart = rowTile * TILE_DIM;\n";
    ss << "  const " << index_type << " colStart = static_cast<" << index_type << ">(blockIdx.x) * static_cast<" << index_type
       << ">(LOGICAL_TRANSPOSE_TILE_COL_SCALARS);\n";
    ss << "  const " << index_type << " y = rowStart + threadIdx.y;\n";
    ss << "  const " << index_type << " tx = rowStart + threadIdx.x;\n\n";

    for (size_t frontier_position = 0; frontier_position < frontier_indices.size(); ++frontier_position) {
        const uint32_t frontier_idx = frontier_indices[frontier_position];
        const ExprNode& frontier = stage.expr.nodes[frontier_idx];
        const uint32_t frontier_source_idx = frontier.lhs;
        const DataType frontier_dtype = requireNodeOutputDType(frontier);
        const std::string frontier_type = scalarStorageType(frontier_dtype);

        const std::optional<TiledLogicalTransposeDenseInputLoad> dense_input_load = tryResolveTiledLogicalTransposeDenseInputLoad(
            stage.expr, group, frontier_source_idx, group.node_dims[frontier_source_idx], frontier_dtype);
        const bool can_emit_dense_packed_input_load =
            dense_input_load.has_value() && logical_transpose_pack_scalars > 1U &&
            scalarStorageTypeSizeBytes(dense_input_load->input_dtype) <= logical_transpose_slot_bytes;

        auto emit_scalar_lane_loads = [&](const std::string& indent) {
            ss << indent << "for (unsigned int lane = 0; lane < LOGICAL_TRANSPOSE_PACK_SCALARS; ++lane) {\n";
            ss << indent << "  const " << index_type << " logical_col = logical_col_base + static_cast<" << index_type << ">(lane);\n";
            ss << indent << "  if (logical_col < numCols) {\n";
            ss << indent << "    const " << index_type << " source_idx = matrixOffset + logical_row * numCols + logical_col;\n";
            uint32_t counter = 0;
            const std::string source_value = emitIndexMappedScalarValue(ss,
                                                                        stage.expr,
                                                                        group,
                                                                        frontier_source_idx,
                                                                        "source_idx",
                                                                        group.node_dims[frontier_source_idx],
                                                                        indent + "    ",
                                                                        use_uint32_index_math,
                                                                        "_ltf" + std::to_string(frontier_idx),
                                                                        counter);
            ss << indent << "    packed_word = thor_set_transpose_pack_lane<LOGICAL_TRANSPOSE_SLOT_BYTES, " << frontier_type << ">"
               << "(packed_word, lane, "
               << castScalarExpr(source_value, emittedScalarNodeValueDType(stage.expr.nodes[frontier_source_idx]), frontier_dtype)
               << ");\n";
            ss << indent << "  }\n";
            ss << indent << "}\n";
        };

        ss << "  {\n";
        ss << "    for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {\n";
        ss << "      const " << index_type << " logical_row = y + j;\n";
        ss << "      const " << index_type << " logical_col_base = colStart + static_cast<" << index_type
           << ">(threadIdx.x * LOGICAL_TRANSPOSE_PACK_SCALARS);\n";
        ss << "      if (logical_row < numRows) {\n";
        ss << "        unsigned int packed_word = 0u;\n";
        if (can_emit_dense_packed_input_load) {
            const uint32_t input_bytes = scalarStorageTypeSizeBytes(dense_input_load->input_dtype);
            const uint32_t raw_load_bytes = input_bytes * logical_transpose_pack_scalars;
            if (raw_load_bytes != 2U && raw_load_bytes != 4U) {
                throw runtime_error("Tiled logical-transpose dense packed input load expected a 16-bit or 32-bit raw load.");
            }

            ss << "        const " << index_type << " source_idx_base = matrixOffset + logical_row * numCols + logical_col_base;\n";
            ss << "        const bool inputPackedLoadOk = (logical_col_base + static_cast<" << index_type
               << ">(LOGICAL_TRANSPOSE_PACK_SCALARS) <= numCols) &&\n";
            ss << "                                       ((source_idx_base % static_cast<" << index_type
               << ">(LOGICAL_TRANSPOSE_PACK_SCALARS)) == static_cast<" << index_type << ">(0)) &&\n";
            ss << "                                       ((reinterpret_cast<unsigned long long>(in" << dense_input_load->input_slot
               << " + source_idx_base) % " << raw_load_bytes << "ULL) == 0ULL);\n";
            ss << "        if (inputPackedLoadOk) {\n";
            ss << "          // Direct dense frontier: one owner thread performs one packed global load, then expands\n";
            ss << "          // it into the uint32_t shared-memory bank word used by the logical transpose tile.\n";
            if (raw_load_bytes == 4U) {
                ss << "          const unsigned int raw_input_word = reinterpret_cast<const unsigned int*>(in"
                   << dense_input_load->input_slot << " + source_idx_base)[0];\n";
            } else {
                ss << "          const unsigned int raw_input_word = static_cast<unsigned int>(reinterpret_cast<const unsigned short*>(in"
                   << dense_input_load->input_slot << " + source_idx_base)[0]);\n";
            }
            ss << "          packed_word = thor_expand_transpose_dense_input_word<" << input_bytes << "U, LOGICAL_TRANSPOSE_SLOT_BYTES>"
               << "(raw_input_word);\n";
            ss << "        } else {\n";
            emit_scalar_lane_loads("          ");
            ss << "        }\n";
        } else {
            emit_scalar_lane_loads("        ");
        }
        ss << "        tile[threadIdx.y + j][threadIdx.x] = packed_word;\n";
        ss << "      }\n";
        ss << "    }\n";
        ss << "  }\n";
        ss << "  __syncthreads();\n";
        ss << "  {\n";
        ss << "    for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {\n";
        ss << "      const unsigned int localPackedOutputRow = threadIdx.y + j;\n";
        ss << "      const " << index_type << " output_row_base = colStart + static_cast<" << index_type
           << ">(localPackedOutputRow * LOGICAL_TRANSPOSE_PACK_SCALARS);\n";
        ss << "      if (tx < numRows) {\n";
        ss << "        const unsigned int packed_word = tile[threadIdx.x][localPackedOutputRow];\n";
        ss << "        for (unsigned int lane = 0; lane < LOGICAL_TRANSPOSE_PACK_SCALARS; ++lane) {\n";
        ss << "          const " << index_type << " output_row = output_row_base + static_cast<" << index_type << ">(lane);\n";
        ss << "          if (output_row < numCols) {\n";
        ss << "            transposed_value_" << frontier_idx << "[j / BLOCK_ROWS][lane] = thor_unpack_transpose_pack_lane<"
           << "LOGICAL_TRANSPOSE_SLOT_BYTES, " << frontier_type << ">(packed_word, lane);\n";
        ss << "          }\n";
        ss << "        }\n";
        ss << "      }\n";
        ss << "    }\n";
        ss << "  }\n";
        if (frontier_position + 1 < frontier_indices.size()) {
            ss << "  __syncthreads();\n";
        }
        ss << "\n";
    }

    ss << "  for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {\n";
    ss << "    const unsigned int localPackedOutputRow = threadIdx.y + j;\n";
    ss << "    const " << index_type << " output_row_base = colStart + static_cast<" << index_type
       << ">(localPackedOutputRow * LOGICAL_TRANSPOSE_PACK_SCALARS);\n";
    ss << "    if (tx < numRows) {\n";

    for (uint32_t stage_output_idx : group.output_indices) {
        if (stage_output_idx >= stage.outputs.size()) {
            throw runtime_error("Tiled logical-transpose consumer output index out of range while emitting stores.");
        }
        const CompiledStageOutput& output = stage.outputs[stage_output_idx];
        if (output.local_node_idx >= stage.expr.nodes.size()) {
            throw runtime_error("Tiled logical-transpose consumer output local node out of range while emitting stores.");
        }
        const DataType output_dtype = requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]);
        const std::optional<DataType> logical_transpose_vector_compute_dtype = tiledLogicalTransposeConsumerVectorComputeDType(
            stage.expr, output.local_node_idx, frontier_indices, logical_transpose_pack_scalars);

        ss << "      {\n";
        ss << "        // Stage output " << stage_output_idx << ": evaluate from the shared logical-transpose frontiers.\n";
        if (logical_transpose_vector_compute_dtype.has_value()) {
            const DataType vector_compute_dtype = logical_transpose_vector_compute_dtype.value();
            ss << "        // Vectorized downstream lane math: each owner thread evaluates two logical lanes\n";
            ss << "        // with half2/bfloat162 once the packed transpose frontier has been unloaded.\n";
            ss << "        for (unsigned int lane_pair = 0; lane_pair < LOGICAL_TRANSPOSE_PACK_SCALARS; lane_pair += 2U) {\n";
            ss << "          const " << index_type << " output_row0 = output_row_base + static_cast<" << index_type << ">(lane_pair);\n";
            ss << "          if (output_row0 + static_cast<" << index_type << ">(1) < numCols) {\n";
            ss << "            const " << index_type << " out_idx0 = matrixOffset + output_row0 * numRows + tx;\n";
            ss << "            const " << index_type << " out_idx1 = out_idx0 + numRows;\n";

            std::unordered_map<uint32_t, TiledLogicalTransposeFrontierVectorValue> frontier_vector_values;
            for (uint32_t frontier_idx : frontier_indices) {
                const DataType frontier_dtype = tiledLogicalTransposeFrontierStorageDType(stage.expr, frontier_idx);
                const std::string value0 = "transposed_value_" + std::to_string(frontier_idx) + "[j / BLOCK_ROWS][lane_pair]";
                const std::string value1 = "transposed_value_" + std::to_string(frontier_idx) + "[j / BLOCK_ROWS][lane_pair + 1U]";
                const std::string vector_value =
                    emitLogicalTransposeVector2PackScalarsAsCompute(value0, value1, frontier_dtype, vector_compute_dtype);
                frontier_vector_values.emplace(frontier_idx, TiledLogicalTransposeFrontierVectorValue{vector_value, vector_compute_dtype});
            }

            {
                uint32_t counter = 0;
                const std::string vector_value = emitTiledLogicalTransposeConsumerVectorValue(
                    ss, stage.expr, output.local_node_idx, frontier_vector_values, "            ", vector_compute_dtype, counter);
                ss << "            out" << stage_output_idx << "[out_idx0] = "
                   << castScalarExpr(emitLogicalTransposeVector2Lane(vector_value, 0U), vector_compute_dtype, output_dtype) << ";\n";
                ss << "            out" << stage_output_idx << "[out_idx1] = "
                   << castScalarExpr(emitLogicalTransposeVector2Lane(vector_value, 1U), vector_compute_dtype, output_dtype) << ";\n";
            }
            ss << "          } else if (output_row0 < numCols) {\n";
            ss << "            const unsigned int lane = lane_pair;\n";
            ss << "            const " << index_type << " out_idx = matrixOffset + output_row0 * numRows + tx;\n";

            std::unordered_map<uint32_t, TiledLogicalTransposeFrontierValue> scalar_tail_frontier_values;
            for (uint32_t frontier_idx : frontier_indices) {
                const DataType frontier_dtype = tiledLogicalTransposeFrontierStorageDType(stage.expr, frontier_idx);
                const std::string value_name = "transposed_value_" + std::to_string(frontier_idx) + "[j / BLOCK_ROWS][lane]";
                scalar_tail_frontier_values.emplace(frontier_idx, TiledLogicalTransposeFrontierValue{value_name, frontier_dtype});
            }

            {
                uint32_t counter = 0;
                const std::string value = emitTiledLogicalTransposeConsumerScalarValue(ss,
                                                                                       stage.expr,
                                                                                       group,
                                                                                       output.local_node_idx,
                                                                                       "out_idx",
                                                                                       group.output_dims,
                                                                                       scalar_tail_frontier_values,
                                                                                       "            ",
                                                                                       use_uint32_index_math,
                                                                                       counter);
                ss << "            out" << stage_output_idx << "[out_idx] = "
                   << castScalarExpr(value, emittedScalarNodeValueDType(stage.expr.nodes[output.local_node_idx]), output_dtype) << ";\n";
            }
            ss << "          }\n";
            ss << "        }\n";
        } else {
            ss << "        for (unsigned int lane = 0; lane < LOGICAL_TRANSPOSE_PACK_SCALARS; ++lane) {\n";
            ss << "          const " << index_type << " output_row = output_row_base + static_cast<" << index_type << ">(lane);\n";
            ss << "          if (output_row < numCols) {\n";
            ss << "            const " << index_type << " out_idx = matrixOffset + output_row * numRows + tx;\n";

            std::unordered_map<uint32_t, TiledLogicalTransposeFrontierValue> frontier_values;
            for (uint32_t frontier_idx : frontier_indices) {
                const DataType frontier_dtype = tiledLogicalTransposeFrontierStorageDType(stage.expr, frontier_idx);
                const std::string value_name = "transposed_value_" + std::to_string(frontier_idx) + "[j / BLOCK_ROWS][lane]";
                frontier_values.emplace(frontier_idx, TiledLogicalTransposeFrontierValue{value_name, frontier_dtype});
            }

            {
                uint32_t counter = 0;
                const std::string value = emitTiledLogicalTransposeConsumerScalarValue(ss,
                                                                                       stage.expr,
                                                                                       group,
                                                                                       output.local_node_idx,
                                                                                       "out_idx",
                                                                                       group.output_dims,
                                                                                       frontier_values,
                                                                                       "            ",
                                                                                       use_uint32_index_math,
                                                                                       counter);
                ss << "            out" << stage_output_idx << "[out_idx] = "
                   << castScalarExpr(value, emittedScalarNodeValueDType(stage.expr.nodes[output.local_node_idx]), output_dtype) << ";\n";
            }
            ss << "          }\n";
            ss << "        }\n";
        }
        ss << "      }\n";
    }

    ss << "    }\n";
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

[[maybe_unused]] static std::string emitIndexMappedSpecializedBroadcast(const CompiledExecutionStage& stage,
                                                                        const std::vector<SpecializedBroadcastGroup>& groups,
                                                                        const std::string& kernel_name) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        throw runtime_error("emitIndexMappedSpecializedBroadcast called on non-fused stage.");
    }
    if (expressionHasRopeOp(stage.expr)) {
        throw runtime_error("Index-mapped fused transpose emission does not support RoPE in the same fused stage yet.");
    }

    const std::optional<RaggedValuewiseExtentEmissionSpec> ragged_extent = findRaggedValuewiseExtentSpec(stage.expr);
    if (ragged_extent.has_value()) {
        validateRaggedExtentStageIsolation(stage, ragged_extent.value());
    }

    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const std::vector<DataType> output_dtypes = collectOutputDTypes(stage);
    const bool use_uint32_index_math = groupsSupportUInt32IndexMath(groups);
    const std::string index_type = emittedIndexType(use_uint32_index_math);

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
            ss << "const " << scalarStorageType(input_dtypes[i]) << "* __restrict__ in" << i;
        }
    }

    for (uint32_t i = 0; i < output_dtypes.size(); ++i) {
        if (!first_arg) {
            ss << ", ";
        }
        first_arg = false;
        ss << scalarStorageType(output_dtypes[i]) << "* __restrict__ out" << i;
    }

    ss << ") {\n";
    if (ragged_extent.has_value()) {
        const RaggedValuewiseExtentEmissionSpec& extent = ragged_extent.value();
        ss << "  const unsigned long long active_values_raw = static_cast<unsigned long long>(in" << extent.partition_input_slot << "["
           << raggedActiveValueCountLoadIndex(extent) << "]);\n";
        ss << "  const unsigned long long active_values = active_values_raw < " << extent.max_active_values
           << "ULL ? active_values_raw : " << extent.max_active_values << "ULL;\n";
        ss << "  const " << index_type << " thread_idx = " << emitFlatThreadIndexExpr(use_uint32_index_math) << ";\n";
        ss << "  const " << index_type << " grid_stride = static_cast<" << index_type
           << ">(blockDim.x) * static_cast<" << index_type << ">(gridDim.x);\n\n";
    } else {
        ss << "  " << index_type << " idx = " << emitFlatThreadIndexExpr(use_uint32_index_math) << ";\n\n";
    }

    for (uint32_t g = 0; g < groups.size(); ++g) {
        const SpecializedBroadcastGroup& group = groups[g];
        if (group.node_dims.size() != stage.expr.nodes.size()) {
            throw runtime_error("Index-mapped specialized broadcast group is missing per-node dimensions.");
        }

        std::string indent = "    ";
        if (ragged_extent.has_value()) {
            const uint64_t group_elements_per_value = specializedRaggedGroupElementsPerValue(ragged_extent.value(), group);
            ss << "  const unsigned long long runtime_numel_g" << g << "_u64 = active_values * " << group_elements_per_value << "ULL;\n";
            ss << "  const " << index_type << " runtime_numel_g" << g << " = static_cast<" << index_type
               << ">(runtime_numel_g" << g << "_u64 < " << emitUnsignedLiteral(group.numel, false)
               << " ? runtime_numel_g" << g << "_u64 : " << emitUnsignedLiteral(group.numel, false) << ");\n";
            ss << "  for (" << index_type << " idx = thread_idx; idx < runtime_numel_g" << g << "; idx += grid_stride) {\n";
        } else {
            ss << "  if (idx < " << emitUnsignedLiteral(group.numel, use_uint32_index_math) << ") {\n";
        }

        for (uint32_t out_idx : group.output_indices) {
            if (out_idx >= stage.outputs.size()) {
                throw runtime_error("Index-mapped specialized broadcast output index out of range.");
            }
            const CompiledStageOutput& output = stage.outputs[out_idx];
            if (output.local_node_idx >= stage.expr.nodes.size()) {
                throw runtime_error("Index-mapped specialized broadcast output local node out of range.");
            }
            const DataType output_dtype = requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]);
            uint32_t counter = 0;
            const std::string value = emitIndexMappedScalarValue(ss,
                                                                  stage.expr,
                                                                  group,
                                                                  output.local_node_idx,
                                                                  "idx",
                                                                  group.output_dims,
                                                                  indent,
                                                                  use_uint32_index_math,
                                                                  "_g" + std::to_string(g) + "_o" + std::to_string(out_idx),
                                                                  counter);
            ss << indent << "out" << out_idx
               << "[idx] = " << castScalarExpr(value, emittedScalarNodeValueDType(stage.expr.nodes[output.local_node_idx]), output_dtype)
               << ";\n";
        }
        ss << "  }\n\n";
    }

    ss << "}\n";
    return ss.str();
}

bool CudaSourceEmitter::specializedBroadcastUsesTiledLogicalTransposeConsumerLaunch(const CompiledExecutionStage& stage,
                                                                                    const std::vector<SpecializedBroadcastGroup>& groups) {
    return tryFindTiledLogicalTransposeConsumerFrontiers(stage, groups).has_value();
}


std::optional<uint32_t> CudaSourceEmitter::fusedGateRequiredSpecializedBroadcastElementsPerThread(
    const CompiledExecutionStage& stage,
    const std::vector<SpecializedBroadcastGroup>& groups) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel || groups.empty() ||
        stageHasTransposedMaterializedOutput(stage.outputs) || expressionHasIndexAwareOps(stage.expr) ||
        expressionHasStridedViewOp(stage.expr) || expressionHasTakeAlongAxisOp(stage.expr) ||
        expressionHasLogicalTransposeOp(stage.expr)) {
        return std::nullopt;
    }

    const std::optional<RaggedValuewiseExtentEmissionSpec> ragged_extent = findRaggedValuewiseExtentSpec(stage.expr);
    const bool is_ragged = ragged_extent.has_value();
    const std::unordered_set<uint32_t> structural_slots =
        is_ragged ? raggedValuewiseExtentPartitionInputSlots(stage.expr) : std::unordered_set<uint32_t>{};
    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(stage.expr);
    const std::vector<DataType> output_dtypes = collectOutputDTypes(stage);

    std::vector<DataType> value_dtypes;
    for (const SpecializedBroadcastGroup& group : groups) {
        if (group.output_dims.empty() || group.used_input_slots.size() != group.used_input_load_kinds.size()) {
            return std::nullopt;
        }
        for (uint32_t slot : group.used_input_slots) {
            if (slot >= stage.expr.inputs.size() || structural_slots.contains(slot) ||
                stage.expr.inputs[slot].kind != NamedInput::Kind::Tensor) {
                return std::nullopt;
            }
            const DataType dtype = input_dtypes.at(slot);
            if (!fusedGateOrdinaryFloatDType(dtype)) {
                return std::nullopt;
            }
            value_dtypes.push_back(dtype);
        }
        for (uint32_t out_idx : group.output_indices) {
            if (out_idx >= output_dtypes.size() || !fusedGateOrdinaryFloatDType(output_dtypes[out_idx])) {
                return std::nullopt;
            }
            value_dtypes.push_back(output_dtypes[out_idx]);
        }
    }
    if (value_dtypes.empty()) {
        return std::nullopt;
    }

    // Dense and device-ragged specialized broadcast share one packet family:
    // homogeneous BF16/FP16 storage owns one 16-byte packet (eight scalars) per
    // tensor per thread, and any FP32 storage participant uses four scalars so
    // each FP32 tensor still moves one float4 per thread.
    uint32_t required_elements_per_thread = 0u;
    if (fusedGateAllSameDType(value_dtypes, DataType::BF16) || fusedGateAllSameDType(value_dtypes, DataType::FP16)) {
        required_elements_per_thread = 8u;
    } else if (std::any_of(value_dtypes.begin(), value_dtypes.end(), [](DataType dtype) { return dtype == DataType::FP32; })) {
        required_elements_per_thread = 4u;
    } else {
        // As with flat FUSED-GATE, mixed BF16/FP16 without FP32 remains
        // intentionally outside the benchmark-backed invariant.
        return std::nullopt;
    }

    for (const SpecializedBroadcastGroup& group : groups) {
        const uint64_t packet_scalars = required_elements_per_thread;
        if ((group.output_dims.back() % packet_scalars) != 0ULL ||
            (group.numel % packet_scalars) != 0ULL) {
            return std::nullopt;
        }
        if (is_ragged &&
            (specializedRaggedGroupElementsPerValue(ragged_extent.value(), group) % packet_scalars) != 0ULL) {
            return std::nullopt;
        }

        for (size_t used_i = 0; used_i < group.used_input_slots.size(); ++used_i) {
            const uint64_t inner_stride = specializedBroadcastInnermostInputStride(group, used_i);
            if (group.used_input_load_kinds[used_i] == SpecializedInputLoadKind::NativeVector) {
                if (inner_stride != 1ULL) {
                    return std::nullopt;
                }
            } else if (inner_stride != 0ULL) {
                // Gate only the packet mappings proven by the census: contiguous
                // native-vector operands and true scalar broadcasts reused across
                // the packet. Row/column/outer forms that lower to these patterns
                // are included; exotic scalar-pack coordinate mappings are not.
                return std::nullopt;
            }
        }
    }

    return required_elements_per_thread;
}

// The wide-scalar packet width that dispatch actually uses. When only a two-lane
// packet fits and the stage already computes in packed BF16/FP16 (explicit
// request), the vector2 broadcast emitter owns the same two lanes with paired
// arithmetic, so it is preferred.
static uint32_t dispatchedWideScalarSpecializedBroadcastElementsPerThread(const CompiledExecutionStage& stage,
                                                                          const std::vector<SpecializedBroadcastGroup>& groups) {
    const uint32_t packet_scalars = wideScalarSpecializedBroadcastElementsPerThread(stage, groups);
    if (packet_scalars <= 2u && CudaSourceEmitter::getVectorizedStageStorageDType(stage).has_value()) {
        return 1u;
    }
    return packet_scalars;
}

// ---------------------------------------------------------------------------
// RoPE, one thread per rotation pair.
//
// The per-element RoPE path makes every output element compute its own frequency (powf) and
// sincos and evaluate its input subgraph twice (own value and partner). Both lanes of a pair share
// the pair index and the position, so a thread that owns the pair computes the angle once, reads
// each input element once, and writes both lanes. The per-lane formulas are the per-element ones,
// so results are unchanged.
//
// Eligible: one broadcast group, no ragged extent, exactly one ROPE node whose result spans the
// group domain (head dim innermost, stride-1 output axis), no other index-aware ops, and no dynamic
// effective-sequence-length input. Everything else keeps the per-element path.
static std::optional<uint32_t> ropePairPerThreadNode(const CompiledExecutionStage& stage,
                                                    const std::vector<SpecializedBroadcastGroup>& groups) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel || groups.size() != 1 ||
        stageHasTransposedMaterializedOutput(stage.outputs) || findRaggedValuewiseExtentSpec(stage.expr).has_value()) {
        return std::nullopt;
    }
    std::optional<uint32_t> rope_node;
    for (uint32_t i = 0; i < stage.expr.nodes.size(); ++i) {
        const ExprOp op = stage.expr.nodes[i].op;
        if (op == ExprOp::STRIDED_VIEW || op == ExprOp::STRIDED_VIEW_BACKWARD || op == ExprOp::TRANSPOSE ||
            op == ExprOp::TAKE_ALONG_AXIS) {
            return std::nullopt;
        }
        if (op == ExprOp::ROPE) {
            if (rope_node.has_value()) {
                return std::nullopt;
            }
            rope_node = i;
        }
    }
    if (!rope_node.has_value()) {
        return std::nullopt;
    }
    const ExprNode& n = stage.expr.nodes[rope_node.value()];
    const SpecializedBroadcastGroup& group = groups.front();
    const size_t rank = group.output_dims.size();
    if (n.rope_effective_sequence_length_node != UINT32_MAX || rank < 2 || n.rope_head_dim_axis + 1 != rank ||
        n.rope_sequence_axis >= rank || group.node_dims.size() != stage.expr.nodes.size() ||
        group.node_dims[rope_node.value()] != group.output_dims || group.active_axes.size() != rank ||
        group.active_axes.back().output_stride != 1ULL || group.active_axes.back().dim != group.output_dims.back()) {
        return std::nullopt;
    }
    const uint64_t head_dim = group.output_dims.back();
    const uint64_t rotary_dim = n.rope_rotary_dim == 0 ? head_dim : n.rope_rotary_dim;
    if (head_dim < 2 || head_dim % 2 != 0 || rotary_dim < 2 || rotary_dim % 2 != 0 || rotary_dim > head_dim) {
        return std::nullopt;
    }
    return rope_node;
}

static std::string emitRopePairSpecializedBroadcast(const CompiledExecutionStage& stage,
                                                    const SpecializedBroadcastGroup& group,
                                                    uint32_t rope_idx,
                                                    const std::string& kernel_name) {
    const PhysicalExpression& expr = stage.expr;
    const ExprNode& rope = expr.nodes.at(rope_idx);
    const std::vector<DataType> input_dtypes = collectInputSlotDTypes(expr);
    const std::vector<DataType> output_dtypes = collectOutputDTypes(stage);
    const bool u32 = groupSupportsUInt32IndexMath(group);
    const std::string index_type = emittedIndexType(u32);
    const size_t rank = group.output_dims.size();
    const uint64_t head_dim = group.output_dims.back();
    const uint64_t rotary_dim = rope.rope_rotary_dim == 0 ? head_dim : rope.rope_rotary_dim;
    const uint64_t pairs_per_row = head_dim / 2;
    const auto lit = [&](uint64_t v) { return emitUnsignedLiteral(v, u32); };

    std::ostringstream ss;
    emitRequiredHeaders(expr, ss);
    ss << "extern \"C\" __global__\n";
    ss << "void " << kernel_name << "(";
    bool first_arg = true;
    for (uint32_t i = 0; i < input_dtypes.size(); ++i) {
        ss << (first_arg ? "" : ", ");
        first_arg = false;
        if (expr.inputs[i].kind == NamedInput::Kind::RuntimeScalarFp32) {
            ss << scalarStorageType(input_dtypes[i]) << " in" << i;
        } else {
            ss << "const " << scalarStorageType(input_dtypes[i]) << "* in" << i;
        }
    }
    for (uint32_t i = 0; i < output_dtypes.size(); ++i) {
        ss << (first_arg ? "" : ", ");
        first_arg = false;
        ss << scalarStorageType(output_dtypes[i]) << "* out" << i;
    }
    ss << ") {\n";
    const std::string indent = "    ";
    ss << "  const " << index_type << " pair_idx = " << emitFlatThreadIndexExpr(u32) << ";\n";
    ss << "  if (pair_idx < " << lit(group.numel / 2) << ") {\n";
    for (size_t axis = 0; axis < rank; ++axis) {
        ss << indent << "const " << index_type << " out_dim_" << axis << " = " << lit(group.output_dims[axis]) << ";\n";
    }
    const std::string row_expr = isPowerOfTwo(pairs_per_row)
                                     ? "(pair_idx >> " + std::to_string(log2Exact(pairs_per_row)) + ")"
                                     : "(pair_idx / " + lit(pairs_per_row) + ")";
    ss << indent << "const " << index_type << " rope_row = " << row_expr << ";\n";
    ss << indent << "const " << index_type << " rope_slot = pair_idx - rope_row * " << lit(pairs_per_row) << ";\n";
    ss << indent << "const unsigned long long rope_rotary_dim = " << rotary_dim << "ULL;\n";
    ss << indent << "const unsigned long long rope_pair_index = static_cast<unsigned long long>(rope_slot);\n";
    // Lane 0 is the pair's first lane (rotate-half: d, interleaved: 2p); lane 1 its partner.
    const std::string rotated_d0 = rope.rope_interleaved ? "(rope_slot << 1)" : "rope_slot";
    const std::string rotated_d1 =
        rope.rope_interleaved ? "((rope_slot << 1) + " + lit(1) + ")" : "(rope_slot + " + lit(rotary_dim / 2) + ")";
    if (rotary_dim == head_dim) {
        ss << indent << "const bool rope_rotated = true;\n";
        ss << indent << "const " << index_type << " rope_d0 = " << rotated_d0 << ";\n";
        ss << indent << "const " << index_type << " rope_d1 = " << rotated_d1 << ";\n";
    } else {
        // Partial rotary: the trailing head_dim - rotary_dim elements pass through, two per slot.
        ss << indent << "const bool rope_rotated = rope_slot < " << lit(rotary_dim / 2) << ";\n";
        ss << indent << "const " << index_type << " rope_d0 = rope_rotated ? " << rotated_d0 << " : (" << lit(rotary_dim)
           << " + ((rope_slot - " << lit(rotary_dim / 2) << ") << 1));\n";
        ss << indent << "const " << index_type << " rope_d1 = rope_rotated ? " << rotated_d1 << " : (rope_d0 + " << lit(1) << ");\n";
    }
    ss << indent << "const " << index_type << " e0 = rope_row * " << lit(head_dim) << " + rope_d0;\n";
    ss << indent << "const " << index_type << " e1 = rope_row * " << lit(head_dim) << " + rope_d1;\n";

    // Offsets: full decode for lane 0; lane 1 differs only along the innermost axis.
    std::vector<size_t> all_used(group.used_input_slots.size());
    std::iota(all_used.begin(), all_used.end(), 0);
    for (uint32_t slot : group.used_input_slots) {
        ss << indent << index_type << " in" << slot << "_offset_e0 = " << lit(0) << ";\n";
    }
    emitSpecializedBroadcastOffsetMath(ss, group, all_used, "e0", "_e0", indent, u32);
    for (size_t used_i = 0; used_i < group.used_input_slots.size(); ++used_i) {
        const uint32_t slot = group.used_input_slots[used_i];
        const uint64_t inner = specializedBroadcastInnermostInputStride(group, used_i);
        ss << indent << "const " << index_type << " in" << slot << "_offset_e1 = in" << slot << "_offset_e0";
        if (inner == 1ULL) {
            ss << " + (rope_d1 - rope_d0)";
        } else if (inner != 0ULL) {
            ss << " + (rope_d1 - rope_d0) * " << lit(inner);
        }
        ss << ";\n";
    }

    // Nodes reachable only through RoPE position ids are evaluated by the angle emitter.
    std::vector<CompiledStageOutput> group_outputs;
    for (uint32_t out_idx : group.output_indices) group_outputs.push_back(stage.outputs.at(out_idx));
    const std::unordered_set<uint32_t> skip = collectIndexAwareInputNodesToSkipForFlatOutput(expr, group_outputs);
    const std::vector<uint32_t> required_nodes = orderedRequiredNodesForGroup(stage, group.output_indices);
    const std::array<std::string, 2> lanes{"_e0", "_e1"};
    for (uint32_t node_idx : required_nodes) {
        if (skip.contains(node_idx) || !shouldEmitScalarNodeDefinition(expr, node_idx)) {
            continue;
        }
        if (node_idx != rope_idx) {
            for (size_t k = 0; k < 2; ++k) {
                const std::string& suffix = lanes[k];
                emitScalarNodeSuffixed(ss, expr, node_idx, "e" + std::to_string(k), suffix, indent, "", 0, [&](uint32_t slot) {
                    return "in" + std::to_string(slot) + "[in" + std::to_string(slot) + "_offset" + suffix + "]";
                });
            }
            continue;
        }
        const DataType emitted_dtype = emittedScalarNodeValueDType(rope);
        const std::string output_type = scalarStorageType(emitted_dtype);
        ss << indent << "const float rope_v0 = " << emitResolvedScalarValueExprSuffixed(expr, rope.lhs, DataType::FP32, "_e0") << ";\n";
        ss << indent << "const float rope_v1 = " << emitResolvedScalarValueExprSuffixed(expr, rope.lhs, DataType::FP32, "_e1") << ";\n";
        ss << indent << "float rope_out0 = rope_v0;\n";
        ss << indent << "float rope_out1 = rope_v1;\n";
        ss << indent << "if (rope_rotated) {\n";
        // emitRopeSinCos writes its lines at indent + 4.
        emitRopeSinCos(ss, expr, rope_idx, "c_e0_" + std::to_string(rope.rope_sequence_axis),
                       "static_cast<float>(out_dim_" + std::to_string(rope.rope_sequence_axis) + ")", indent, &group);
        // Same per-lane formulas as the per-element path (first lane: cur*c - peer*s; second: peer*s + cur*c).
        ss << indent << "    rope_out0 = rope_v0 * rope_c - rope_v1 * rope_s;\n";
        ss << indent << "    rope_out1 = rope_v0 * rope_s + rope_v1 * rope_c;\n";
        ss << indent << "}\n";
        ss << indent << "const " << output_type << " t" << rope_idx << "_e0 = " << castScalarExpr("rope_out0", DataType::FP32, emitted_dtype)
           << ";\n";
        ss << indent << "const " << output_type << " t" << rope_idx << "_e1 = " << castScalarExpr("rope_out1", DataType::FP32, emitted_dtype)
           << ";\n";
    }
    for (uint32_t out_idx : group.output_indices) {
        const CompiledStageOutput& output = stage.outputs[out_idx];
        const DataType output_dtype = requireNodeOutputDType(expr.nodes[output.local_node_idx]);
        for (size_t k = 0; k < 2; ++k) {
            ss << indent << "out" << out_idx << "[e" << k << "] = "
               << emitResolvedScalarValueExprSuffixed(expr, output.local_node_idx, output_dtype, lanes[k]) << ";\n";
        }
    }
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

uint32_t CudaSourceEmitter::specializedBroadcastElementsPerThread(
    const CompiledExecutionStage& stage,
    const std::vector<SpecializedBroadcastGroup>& groups) {
    if (ropePairPerThreadNode(stage, groups).has_value()) {
        return 2u;  // one thread per RoPE pair; see emitRopePairSpecializedBroadcast
    }
    if (getWideSpecializedBroadcastVectorizedStageStorageDType(stage, groups).has_value()) {
        return 8u;
    }
    const uint32_t wide_scalar_elements_per_thread = dispatchedWideScalarSpecializedBroadcastElementsPerThread(stage, groups);
    if (wide_scalar_elements_per_thread > 1u) {
        return wide_scalar_elements_per_thread;
    }
    // Shapes the 16-byte packet family cannot own (row width not a packet
    // multiple, strided broadcast operands, FP8 storage) keep the vector2 /
    // scalar broadcast emitters.
    return getVectorizedStageStorageDType(stage).has_value() ? 2u : 1u;
}

static std::string emitSpecializedBroadcastImpl(const CompiledExecutionStage& stage,
                                                const std::vector<SpecializedBroadcastGroup>& groups,
                                                const std::string& kernel_name);

std::string CudaSourceEmitter::emitSpecializedBroadcast(const CompiledExecutionStage& stage,
                                                        const std::vector<SpecializedBroadcastGroup>& groups,
                                                        const std::string& kernel_name) {
    if (stage.kind == CompiledExecutionStage::Kind::FusedKernel && expressionHasStrengthReductions(stage.expr)) {
        CompiledExecutionStage reduced = stage;
        applyStrengthReductions(reduced.expr);
        return emitSpecializedBroadcastImpl(reduced, groups, kernel_name);
    }
    return emitSpecializedBroadcastImpl(stage, groups, kernel_name);
}

static std::string emitSpecializedBroadcastImpl(const CompiledExecutionStage& stage,
                                                const std::vector<SpecializedBroadcastGroup>& groups,
                                                const std::string& kernel_name) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        throw std::runtime_error("emitSpecializedBroadcast called on non-fused stage.");
    }
    if (groups.empty()) {
        throw std::runtime_error("emitSpecializedBroadcast requires at least one group.");
    }

    const std::optional<RaggedValuewiseExtentEmissionSpec> ragged_extent = findRaggedValuewiseExtentSpec(stage.expr);
    if (ragged_extent.has_value()) {
        validateRaggedExtentStageIsolation(stage, ragged_extent.value());
    }

    if (stageHasTransposedMaterializedOutput(stage.outputs)) {
        if (ragged_extent.has_value()) {
            throw std::runtime_error("ragged runtime extent is not supported with transposed materialized specialized broadcast outputs.");
        }
        if (expressionHasIndexAwareOps(stage.expr)) {
            throw std::runtime_error("RoPE/index-aware fused stages do not currently support transposed materialized outputs.");
        }
        return emitTiledTransposeMaterializedSpecializedBroadcast(stage, groups, kernel_name);
    }

    if (expressionHasLogicalTransposeOp(stage.expr)) {
        if (ragged_extent.has_value()) {
            throw std::runtime_error("ragged runtime extent is not supported with logical-transpose specialized broadcast stages.");
        }
        if (CudaSourceEmitter::specializedBroadcastUsesTiledLogicalTransposeConsumerLaunch(stage, groups)) {
            return emitTiledLogicalTransposeConsumerSpecializedBroadcast(stage, groups, kernel_name);
        }
        throw std::runtime_error(
            "Logical transpose inside a fused broadcast stage would require dense column-strided global access, "
            "but this pattern is not currently supported by the auto-swizzle emitter. Materialize the transpose boundary "
            "or simplify the transpose chain before fusing.");
    }

    if (expressionHasTakeAlongAxisOp(stage.expr) || expressionHasStridedViewOp(stage.expr)) {
        return emitIndexMappedSpecializedBroadcast(stage, groups, kernel_name);
    }

    if (const std::optional<uint32_t> rope_node = ropePairPerThreadNode(stage, groups); rope_node.has_value()) {
        return emitRopePairSpecializedBroadcast(stage, groups.front(), rope_node.value(), kernel_name);
    }

    const std::optional<DataType> ragged_vectorized_dtype =
        getWideSpecializedBroadcastVectorizedStageStorageDType(stage, groups);
    if (ragged_vectorized_dtype.has_value()) {
        const std::optional<DataType> vector_compute_dtype =
            getVectorizedStageComputeDTypeImpl(stage.expr, ragged_vectorized_dtype.value());
        if (!vector_compute_dtype.has_value()) {
            throw runtime_error("Wide ragged specialized-broadcast stage lost its compute dtype after eligibility selection.");
        }
        return emitWideSpecializedBroadcast(
            stage, groups, ragged_vectorized_dtype.value(), vector_compute_dtype.value(), kernel_name);
    }

    const uint32_t wide_scalar_elements_per_thread = dispatchedWideScalarSpecializedBroadcastElementsPerThread(stage, groups);
    if (wide_scalar_elements_per_thread > 1u) {
        return emitWideScalarSpecializedBroadcast(stage, groups, kernel_name, wide_scalar_elements_per_thread);
    }

    std::optional<DataType> vectorized_dtype = CudaSourceEmitter::getVectorizedStageStorageDType(stage);
    if (vectorized_dtype.has_value()) {
        const std::optional<DataType> vector_compute_dtype = getVectorizedStageComputeDTypeImpl(stage.expr, vectorized_dtype.value());
        if (!vector_compute_dtype.has_value()) {
            throw runtime_error("Vectorized specialized-broadcast stage lost its compute dtype after eligibility selection.");
        }
        return emitVector2SpecializedBroadcast(stage, groups, vectorized_dtype.value(), vector_compute_dtype.value(), kernel_name);
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
    if (ragged_extent.has_value()) {
        const RaggedValuewiseExtentEmissionSpec& extent = ragged_extent.value();
        ss << "  const unsigned long long active_values_raw = static_cast<unsigned long long>(in" << extent.partition_input_slot << "["
           << raggedActiveValueCountLoadIndex(extent) << "]);\n";
        ss << "  const unsigned long long active_values = active_values_raw < " << extent.max_active_values
           << "ULL ? active_values_raw : " << extent.max_active_values << "ULL;\n";
        ss << "  const " << index_type << " thread_idx = " << emitFlatThreadIndexExpr(use_uint32_index_math) << ";\n";
        ss << "  const " << index_type << " grid_stride = static_cast<" << index_type
           << ">(blockDim.x) * static_cast<" << index_type << ">(gridDim.x);\n\n";
    } else {
        ss << "  " << index_type << " idx = " << emitFlatThreadIndexExpr(use_uint32_index_math) << ";\n\n";
    }

    for (uint32_t g = 0; g < groups.size(); ++g) {
        const SpecializedBroadcastGroup& group = groups[g];
        const std::vector<uint32_t> required_nodes = orderedRequiredNodesForGroup(stage, group.output_indices);

        std::vector<size_t> all_used_indices(group.used_input_slots.size());
        std::iota(all_used_indices.begin(), all_used_indices.end(), 0);

        const std::string indent = "    ";
        if (ragged_extent.has_value()) {
            const uint64_t group_elements_per_value = specializedRaggedGroupElementsPerValue(ragged_extent.value(), group);
            ss << "  const unsigned long long runtime_numel_g" << g << "_u64 = active_values * " << group_elements_per_value << "ULL;\n";
            ss << "  const " << index_type << " runtime_numel_g" << g << " = static_cast<" << index_type
               << ">(runtime_numel_g" << g << "_u64 < " << emitUnsignedLiteral(group.numel, false)
               << " ? runtime_numel_g" << g << "_u64 : " << emitUnsignedLiteral(group.numel, false) << ");\n";
            ss << "  for (" << index_type << " idx = thread_idx; idx < runtime_numel_g" << g << "; idx += grid_stride) {\n";
        } else {
            ss << "  if (idx < " << emitUnsignedLiteral(group.numel, use_uint32_index_math) << ") {\n";
        }

        for (size_t axis = 0; axis < group.output_dims.size(); ++axis) {
            ss << indent << "const " << index_type << " out_dim_" << axis << " = "
               << emitUnsignedLiteral(group.output_dims[axis], use_uint32_index_math) << ";\n";
        }

        for (uint32_t input_slot : group.used_input_slots) {
            ss << indent << index_type << " in" << input_slot << "_offset = " << emitUnsignedLiteral(0, use_uint32_index_math) << ";\n";
        }

        if (!group.active_axes.empty()) {
            ss << "\n";
            emitSpecializedBroadcastOffsetMath(ss, group, all_used_indices, "idx", "", indent, use_uint32_index_math);
            ss << "\n";
        }

        for (uint32_t node_idx : required_nodes) {
            const ExprNode& node = stage.expr.nodes.at(node_idx);
            const bool is_extent_offsets_input = ragged_extent.has_value() && node.op == ExprOp::INPUT &&
                                                 node.input_slot == ragged_extent->partition_input_slot;
            if (is_extent_offsets_input || !shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
                continue;
            }
            emitScalarNode(ss, stage.expr, node_idx, /*broadcast_support=*/true, indent, &group);
        }

        ss << "\n";
        for (uint32_t out_idx : group.output_indices) {
            const CompiledStageOutput& output = stage.outputs[out_idx];
            const DataType output_dtype = requireNodeOutputDType(stage.expr.nodes[output.local_node_idx]);
            ss << indent << "out" << out_idx << "[idx] = "
               << emitResolvedScalarValueExpr(stage.expr, output.local_node_idx, output_dtype) << ";\n";
        }

        ss << "  }\n\n";
    }

    ss << "}\n";
    return ss.str();
}

}  // namespace ThorImplementation
