#include "Utilities/Expression/CudaSourceEmitter.h"
#include <optional>
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
    if (Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) {
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
    if (Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) {
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
    ss << R"(#include <vector_types.h>

template <unsigned int Bytes>
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
                // BF16 special-function lowering casts through FP16 for CUDA half/half2 intrinsics.
                need_fp16 = true;
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
        ss << "#include <cuda_fp16.h>\n";
    }
    if (need_bf16) {
        ss << "#include <cuda_bf16.h>\n";
    }
    if (need_fp8) {
        ss << "#include <cuda_fp8.h>\n";
    }
    const bool needs_gamma_math_header =
        expressionUsesOp(expr, ExprOp::TGAMMA) || expressionUsesOp(expr, ExprOp::LGAMMA) || expressionUsesOp(expr, ExprOp::DIGAMMA);
    if (needs_gamma_math_header) {
        ss << "#include <math_functions.h>\n";
    }
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

static std::optional<DataType> getVectorizedStageStorageDTypeImpl(const PhysicalExpression& expr,
                                                                  const std::vector<DataType>& input_dtypes,
                                                                  const std::vector<DataType>& output_dtypes) {
    if (input_dtypes.empty() || output_dtypes.empty()) {
        return std::nullopt;
    }
    if (expressionHasIndexAwareOps(expr)) {
        return std::nullopt;
    }

    std::optional<DataType> maybe_stage_dtype = std::nullopt;
    for (uint32_t slot = 0; slot < expr.inputs.size(); ++slot) {
        if (expr.inputs[slot].kind != NamedInput::Kind::Tensor) {
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

    const DataType expected_compute_dtype = defaultComputeDType(stage_dtype);

    for (const ExprNode& node : expr.nodes) {
        if (node.op == ExprOp::SCALAR_FP || node.op == ExprOp::RUNTIME_SCALAR || node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
            continue;
        }

        if (node.op == ExprOp::INPUT) {
            if (requireNodeInputTensorDType(node) != stage_dtype) {
                return std::nullopt;
            }
        }

        if (requireNodeOutputDType(node) != stage_dtype) {
            return std::nullopt;
        }

        if (node.op != ExprOp::INPUT) {
            if (requireNodeComputeDType(node) != expected_compute_dtype) {
                return std::nullopt;
            }
        }
    }

    return stage_dtype;
}

std::optional<DataType> CudaSourceEmitter::getVectorizedStageStorageDType(const PhysicalExecutionStage& stage) {
    if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel) {
        return std::nullopt;
    }
    return getVectorizedStageStorageDTypeImpl(stage.expr, collectInputSlotDTypes(stage.expr), collectOutputDTypes(stage));
}

std::optional<DataType> CudaSourceEmitter::getVectorizedStageStorageDType(const CompiledExecutionStage& stage) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        return std::nullopt;
    }
    return getVectorizedStageStorageDTypeImpl(stage.expr, collectInputSlotDTypes(stage.expr), collectOutputDTypes(stage));
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

static bool supportsMixedTwoByteFloat2TransposedVectorization(const PhysicalExpression& expr, DataType input_dtype, DataType output_dtype) {
    if (expressionHasIndexAwareOps(expr)) {
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

static bool supportsMixedFp8TransposedVectorization(const PhysicalExpression& expr, DataType input_dtype, DataType output_dtype) {
    if (expressionHasIndexAwareOps(expr)) {
        return false;
    }

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
    if (expressionHasIndexAwareOps(expr)) {
        return false;
    }

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

static bool supportsFp8ToBf16Float2TransposedVectorization(const PhysicalExpression& expr, DataType input_dtype, DataType output_dtype) {
    if (expressionHasIndexAwareOps(expr)) {
        return false;
    }

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

    const std::optional<DataType> vectorized_dtype = getVectorizedStageStorageDType(stage);
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

    return flatScalarElementsPerThreadImpl(collectInputSlotDTypes(stage.expr), collectOutputDTypes(stage));
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
                    return "float(half(" + expr + "))";
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
                case DataType::FP8_E4M3:
                case DataType::FP8_E5M2:
                    return "half(" + expr + ")";
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
            const std::string x_h = castScalarExpr(x, compute_dtype, DataType::FP16);
            return castScalarExpr("half(expm1f(float(" + x_h + ")))", DataType::FP16, compute_dtype);
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
            const std::string x_h = castScalarExpr(x, compute_dtype, DataType::FP16);
            return castScalarExpr("half(log1pf(float(" + x_h + ")))", DataType::FP16, compute_dtype);
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
            const std::string x_h = castScalarExpr(x, compute_dtype, DataType::FP16);
            return castScalarExpr("htanh(" + x_h + ")", DataType::FP16, compute_dtype);
        }
        case ExprOp::NORMCDF: {
            if (compute_dtype == DataType::FP32) {
                return "normcdff(" + x + ")";
            }
            const std::string x_h = castScalarExpr(x, compute_dtype, DataType::FP16);
            return castScalarExpr("half(normcdff(float(" + x_h + ")))", DataType::FP16, compute_dtype);
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
        if (node.op != ExprOp::STRIDED_VIEW_BACKWARD) {
            continue;
        }
        if (node.lhs >= expr.nodes.size()) {
            throw runtime_error("strided_view_backward node has invalid lhs while collecting index-aware input skips.");
        }
        collectRequiredNodes(expr, node.lhs, index_aware_required);
    }

    std::unordered_set<uint32_t> skip;
    for (uint32_t node_idx : index_aware_required) {
        if (!ordinary_required.contains(node_idx)) {
            // STRIDED_VIEW_BACKWARD derives a view-gradient index from the dense
            // source-gradient output idx and evaluates its gradient source in
            // that view domain. Nodes that are only reachable through that
            // index-aware child must be emitted by the strided-view-backward
            // emitter with the derived index, not by the ordinary flat emitter
            // with the output idx.
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

static std::string nextIndexMappedSuffix(uint32_t node_idx, uint32_t& counter) {
    return "_im" + std::to_string(node_idx) + "_" + std::to_string(counter++);
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

static std::string emitTakeAlongAxisSourceIndexForDomain(std::ostringstream& ss,
                                                         const ExprNode& node,
                                                         const std::vector<uint64_t>& input_dims,
                                                         const std::vector<uint64_t>& indices_dims,
                                                         const std::string& idx_expr,
                                                         const std::string& selected_index_expr,
                                                         const std::string& var_name,
                                                         const std::string& indent,
                                                         bool /*use_uint32_index_math*/) {
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

    const auto input_strides = packedStrides(input_dims);
    ss << indent << "unsigned long long " << var_name << "_residual = static_cast<unsigned long long>(" << idx_expr << ");\n";
    ss << indent << "unsigned long long " << var_name << " = 0ULL;\n";

    for (int64_t i = static_cast<int64_t>(indices_dims.size()) - 1; i >= 0; --i) {
        const size_t dim_index = static_cast<size_t>(i);
        const uint64_t dim = indices_dims[dim_index];
        const uint64_t stride = input_strides[dim_index];
        const std::string coord_var = var_name + "_coord_" + std::to_string(dim_index);
        if (dim_index == axis) {
            ss << indent << "const unsigned long long " << coord_var << " = static_cast<unsigned long long>(" << selected_index_expr << ");\n";
            ss << indent << var_name << "_residual /= " << dim << "ULL;\n";
        } else {
            ss << indent << "const unsigned long long " << coord_var << " = " << var_name << "_residual % " << dim << "ULL;\n";
            ss << indent << var_name << "_residual /= " << dim << "ULL;\n";
        }

        if (stride == 1ULL) {
            ss << indent << var_name << " += " << coord_var << ";\n";
        } else if (isPowerOfTwo(stride)) {
            ss << indent << var_name << " += (" << coord_var << " << " << std::to_string(log2Exact(stride)) << ");\n";
        } else if (stride != 0ULL) {
            ss << indent << var_name << " += " << coord_var << " * " << stride << "ULL;\n";
        }
    }

    return var_name;
}

static std::string emitIndexMappedScalarValue(std::ostringstream& ss,
                                              const PhysicalExpression& expr,
                                              const SpecializedBroadcastGroup& group,
                                              uint32_t node_idx,
                                              const std::string& idx_expr,
                                              const std::vector<uint64_t>& domain_dims,
                                              const std::string& indent,
                                              bool use_uint32_index_math,
                                              uint32_t& counter) {
    if (node_idx >= expr.nodes.size() || node_idx >= group.node_dims.size()) {
        throw runtime_error("Index-mapped fused emitter node index out of range.");
    }

    const ExprNode& n = expr.nodes[node_idx];
    const DataType emitted_dtype = emittedScalarNodeValueDType(n);
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
            ss << indent << "const " << output_type << " " << var << " = "
               << castScalarExpr("in" + std::to_string(n.input_slot) + "[" + offset_var + "]", input_tensor_dtype, emitted_dtype) << ";\n";
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

    if (n.op == ExprOp::TRANSPOSE) {
        const std::string source_idx = "idx" + suffix;
        emitTransposeSourceIndexForDomain(ss, domain_dims, idx_expr, source_idx, indent, use_uint32_index_math);
        const std::string value =
            emitIndexMappedScalarValue(ss, expr, group, n.lhs, source_idx, group.node_dims[n.lhs], indent, use_uint32_index_math, counter);
        return castScalarExpr(value, emittedScalarNodeValueDType(expr.nodes[n.lhs]), emitted_dtype);
    }

    if (n.op == ExprOp::TAKE_ALONG_AXIS) {
        const std::string take_idx = "take_eval_idx" + suffix;
        emitBroadcastOffsetForDomain(ss, group.node_dims[node_idx], domain_dims, idx_expr, take_idx, indent, use_uint32_index_math);
        const std::string selected_value =
            emitIndexMappedScalarValue(ss, expr, group, n.rhs, take_idx, group.node_dims[n.rhs], indent, use_uint32_index_math, counter);
        const std::string source_idx = "take_source_idx" + suffix;
        emitTakeAlongAxisSourceIndexForDomain(
            ss, n, group.node_dims[n.lhs], group.node_dims[n.rhs], take_idx, selected_value, source_idx, indent, use_uint32_index_math);
        const std::string value = emitIndexMappedScalarValue(
            ss, expr, group, n.lhs, source_idx, group.node_dims[n.lhs], indent, use_uint32_index_math, counter);
        return castScalarExpr(value, emittedScalarNodeValueDType(expr.nodes[n.lhs]), emitted_dtype);
    }

    if (n.op == ExprOp::STRIDED_VIEW) {
        if (n.view_dims.empty() || n.view_dims.size() != n.view_strides.size()) {
            throw runtime_error("Index-mapped strided_view emission requires dimensions and strides with the same non-zero rank.");
        }

        const std::string source_idx = "sv_source_idx" + suffix;
        const std::string residual_var = "sv_residual" + suffix;
        ss << indent << emittedIndexType(use_uint32_index_math) << " " << residual_var << " = static_cast<"
           << emittedIndexType(use_uint32_index_math) << ">(" << idx_expr << ");\n";
        ss << indent << emittedIndexType(use_uint32_index_math) << " " << source_idx << " = "
           << emitUnsignedLiteral(n.view_element_offset, use_uint32_index_math) << ";\n";
        for (int64_t axis = static_cast<int64_t>(n.view_dims.size()) - 1; axis >= 0; --axis) {
            const uint64_t dim = n.view_dims.at(static_cast<size_t>(axis));
            const uint64_t stride = n.view_strides.at(static_cast<size_t>(axis));
            if (dim == 0 || stride == 0) {
                throw runtime_error("Index-mapped strided_view dimensions and strides must be non-zero.");
            }
            const std::string coord_var = "sv_coord" + suffix + "_" + std::to_string(axis);
            ss << indent << "const " << emittedIndexType(use_uint32_index_math) << " " << coord_var << " = " << residual_var
               << " % " << emitUnsignedLiteral(dim, use_uint32_index_math) << ";\n";
            ss << indent << residual_var << " /= " << emitUnsignedLiteral(dim, use_uint32_index_math) << ";\n";
            if (stride == 1ULL) {
                ss << indent << source_idx << " += " << coord_var << ";\n";
            } else if (isPowerOfTwo(stride)) {
                ss << indent << source_idx << " += (" << coord_var << " << " << std::to_string(log2Exact(stride)) << ");\n";
            } else {
                ss << indent << source_idx << " += " << coord_var << " * " << emitUnsignedLiteral(stride, use_uint32_index_math)
                   << ";\n";
            }
        }

        const std::string value = emitIndexMappedScalarValue(
            ss, expr, group, n.lhs, source_idx, group.node_dims[n.lhs], indent, use_uint32_index_math, counter);
        return castScalarExpr(value, emittedScalarNodeValueDType(expr.nodes[n.lhs]), emitted_dtype);
    }

    if (n.op == ExprOp::RESHAPE || n.op == ExprOp::UNSQUEEZE || n.op == ExprOp::SQUEEZE) {
        const std::string value =
            emitIndexMappedScalarValue(ss, expr, group, n.lhs, idx_expr, group.node_dims[n.lhs], indent, use_uint32_index_math, counter);
        return castScalarExpr(value, emittedScalarNodeValueDType(expr.nodes[n.lhs]), emitted_dtype);
    }

    const DataType compute_dtype = requireNodeComputeDType(n);
    auto emit_child = [&](uint32_t child_idx, const std::vector<uint64_t>& child_domain_dims) -> std::string {
        const std::string value =
            emitIndexMappedScalarValue(ss, expr, group, child_idx, idx_expr, child_domain_dims, indent, use_uint32_index_math, counter);
        return castScalarExpr(value, emittedScalarNodeValueDType(expr.nodes[child_idx]), compute_dtype);
    };

    std::string compute_expr;
    if (n.op == ExprOp::WHERE) {
        auto emit_child_as = [&](uint32_t child_idx, const std::vector<uint64_t>& child_domain_dims, DataType target_dtype) -> std::string {
            const std::string value =
                emitIndexMappedScalarValue(ss, expr, group, child_idx, idx_expr, child_domain_dims, indent, use_uint32_index_math, counter);
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
            ss << indent << "const " << output_type << " " << var << " = "
               << castScalarExpr("in" + std::to_string(n.input_slot) + "[" + offset_var + "]", input_tensor_dtype, emitted_dtype) << ";\n";
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
        const std::string value = emitTiledLogicalTransposeConsumerScalarValue(
            ss, expr, group, n.lhs, idx_expr, group.node_dims[n.lhs], frontier_values, indent, use_uint32_index_math, counter);
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

static std::string emitInputStorageIndexForLogicalIndex(std::ostringstream& ss,
                                                        const SpecializedBroadcastGroup* group,
                                                        uint32_t input_slot,
                                                        const std::string& idx_expr,
                                                        const std::string& suffix,
                                                        const std::string& indent) {
    const std::optional<size_t> maybe_used_i = findUsedInputSlotIndex(group, input_slot);
    if (!maybe_used_i.has_value()) {
        return idx_expr;
    }

    const size_t used_i = maybe_used_i.value();
    if (used_i >= group->used_input_visible_dims.size() || used_i >= group->used_input_visible_strides.size()) {
        throw runtime_error("Index-aware input storage mapping is missing visible input layout metadata.");
    }

    const std::vector<uint64_t>& dims = group->used_input_visible_dims[used_i];
    const std::vector<uint64_t>& strides = group->used_input_visible_strides[used_i];
    if (dims.size() != strides.size()) {
        throw runtime_error("Index-aware input storage mapping requires visible dimensions and strides with the same rank.");
    }
    if (dims.empty()) {
        return "0ULL";
    }

    const std::string residual_var = "input_residual_" + std::to_string(input_slot) + suffix;
    const std::string storage_idx = "input_storage_idx_" + std::to_string(input_slot) + suffix;
    ss << indent << "unsigned long long " << residual_var << " = static_cast<unsigned long long>(" << idx_expr << ");\n";
    ss << indent << "unsigned long long " << storage_idx << " = 0ULL;\n";
    for (int64_t axis = static_cast<int64_t>(dims.size()) - 1; axis >= 0; --axis) {
        const uint64_t dim = dims.at(static_cast<size_t>(axis));
        const uint64_t stride = strides.at(static_cast<size_t>(axis));
        if (dim == 0) {
            throw runtime_error("Index-aware input storage mapping does not support zero-sized dimensions.");
        }
        const std::string coord_var = "input_coord_" + std::to_string(input_slot) + "_" + std::to_string(axis) + suffix;
        ss << indent << "const unsigned long long " << coord_var << " = " << residual_var << " % " << dim << "ULL;\n";
        ss << indent << residual_var << " /= " << dim << "ULL;\n";
        if (stride == 0ULL) {
            continue;
        }
        if (stride == 1ULL) {
            ss << indent << storage_idx << " += " << coord_var << ";\n";
        } else if (isPowerOfTwo(stride)) {
            ss << indent << storage_idx << " += (" << coord_var << " << " << std::to_string(log2Exact(stride)) << ");\n";
        } else {
            ss << indent << storage_idx << " += " << coord_var << " * " << stride << "ULL;\n";
        }
    }
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
            const std::string storage_idx = emitInputStorageIndexForLogicalIndex(ss, group, n.input_slot, idx_expr, suffix, indent);
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

    if (n.op == ExprOp::STRIDED_VIEW_BACKWARD) {
        throw runtime_error("Nested strided_view_backward inside an index-aware scalar source is not supported.");
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
        emitTakeAlongAxisSourceIndexForDomain(
            ss, n, group->node_dims[n.lhs], group->node_dims[n.rhs], idx_expr, selected_value, source_idx, indent, true);
        const std::string nested_suffix = suffix + "_take" + std::to_string(node_idx);
        const std::string source_value = emitScalarValueAtIndex(ss, expr, n.lhs, source_idx, nested_suffix, indent, emitted, group);
        const DataType source_dtype = emittedScalarNodeValueDType(expr.nodes.at(n.lhs));
        ss << indent << "const " << output_type << " " << var << " = " << castScalarExpr(source_value, source_dtype, emitted_dtype) << ";\n";
        mark_emitted();
        return var;
    }

    auto child_value_as = [&](uint32_t child_idx, DataType target_dtype) -> std::string {
        const std::string child_value = emitScalarValueAtIndex(ss, expr, child_idx, idx_expr, suffix, indent, emitted, group);
        const DataType child_dtype = emittedScalarNodeValueDType(expr.nodes.at(child_idx));
        return castScalarExpr(child_value, child_dtype, target_dtype);
    };

    if (n.op == ExprOp::STRIDED_VIEW) {
        if (n.view_dims.empty() || n.view_dims.size() != n.view_strides.size()) {
            throw runtime_error("Index-aware strided_view emission requires dimensions and strides with the same non-zero rank.");
        }

        const std::string source_idx = stridedViewForwardSourceIndexVar(node_idx, suffix);
        const std::string residual_var = "sv_residual_" + std::to_string(node_idx) + suffix;
        ss << indent << "unsigned long long " << residual_var << " = static_cast<unsigned long long>(" << idx_expr << ");\n";
        ss << indent << "unsigned long long " << source_idx << " = " << n.view_element_offset << "ULL;\n";
        for (int64_t axis = static_cast<int64_t>(n.view_dims.size()) - 1; axis >= 0; --axis) {
            const uint64_t dim = n.view_dims.at(static_cast<size_t>(axis));
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
    ss << indent << "unsigned long long view_bw_linear_" << node_idx << " = 0ULL;\n";
    for (size_t axis = 0; axis < n.view_dims.size(); ++axis) {
        const uint64_t dim = n.view_dims.at(axis);
        const uint64_t stride = n.view_strides.at(axis);
        if (dim == 0 || stride == 0) {
            throw runtime_error("strided_view_backward dimensions and strides must be non-zero.");
        }
        ss << indent << "if (view_bw_in_view_" << node_idx << ") {\n";
        ss << indent << "  const unsigned long long coord_" << node_idx << "_" << axis << " = view_bw_residual_" << node_idx << " / "
           << stride << "ULL;\n";
        ss << indent << "  if (coord_" << node_idx << "_" << axis << " >= " << dim << "ULL) {\n";
        ss << indent << "    view_bw_in_view_" << node_idx << " = false;\n";
        ss << indent << "  } else {\n";
        ss << indent << "    view_bw_residual_" << node_idx << " -= coord_" << node_idx << "_" << axis << " * " << stride << "ULL;\n";
        ss << indent << "    view_bw_linear_" << node_idx << " = view_bw_linear_" << node_idx << " * " << dim << "ULL + coord_" << node_idx
           << "_" << axis << ";\n";
        ss << indent << "  }\n";
        ss << indent << "}\n";
    }
    ss << indent << "if (view_bw_residual_" << node_idx << " != 0ULL) view_bw_in_view_" << node_idx << " = false;\n";
    std::unordered_set<std::string> emitted;
    const std::string suffix = stridedViewBackwardSafeSuffix(node_idx);
    const std::string grad_value = emitScalarValueAtIndex(
        ss, expr, grad_source_idx, "view_bw_linear_" + std::to_string(node_idx), suffix, indent, emitted, group);
    const DataType grad_value_dtype = emittedScalarNodeValueDType(grad_source);

    const std::string cast_load = castScalarExpr(grad_value, grad_value_dtype, emitted_dtype);
    const std::string zero = castScalarExpr("0.0f", DataType::FP32, emitted_dtype);
    ss << indent << "const " << output_type << " t" << node_idx << " = view_bw_in_view_" << node_idx << " ? " << cast_load << " : " << zero
       << ";\n";
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
        (void)requireNodeComputeDType(n);
        const std::string current_value = emitResolvedScalarValueExpr(expr, n.lhs, DataType::FP32);
        const std::string dim_coord = ropeAxisCoordVar(n.rope_head_dim_axis);
        const std::string seq_coord = ropeAxisCoordVar(n.rope_sequence_axis);
        const std::string head_dim_extent = ropeAxisDimVar(n.rope_head_dim_axis);
        const std::string seq_extent = ropeAxisDimVar(n.rope_sequence_axis);
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

        std::unordered_set<uint32_t> rope_required;
        collectRequiredNodes(expr, n.lhs, rope_required);
        for (uint32_t dep_idx = 0; dep_idx < node_idx; ++dep_idx) {
            if (!rope_required.contains(dep_idx) || !shouldEmitScalarNodeDefinition(expr, dep_idx)) {
                continue;
            }
            emitScalarNodeSuffixed(ss, expr, dep_idx, "rope_peer_idx", suffix, indent + "    ", "", 0, {});
        }

        ss << indent << "    float rope_position = static_cast<float>(" << seq_coord << ") + "
           << emitScalarFpLiteral(static_cast<double>(n.rope_position_offset)) << ";\n";
        if (n.rope_scaling_kind == RotaryScalingKind::Linear) {
            ss << indent << "    rope_position = rope_position / " << emitScalarFpLiteral(n.rope_scaling_factor) << ";\n";
        }
        ss << indent << "    float rope_base = " << emitScalarFpLiteral(n.rope_base) << ";\n";
        ss << indent << "    float rope_freq = 0.0f;\n";
        if (n.rope_scaling_kind == RotaryScalingKind::DynamicNTK) {
            ss << indent << "    const float rope_seq_len = fmaxf(static_cast<float>(" << seq_extent << ") + "
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
            ss << indent << "    const float rope_seq_len = fmaxf(static_cast<float>(" << seq_extent << ") + "
               << emitScalarFpLiteral(static_cast<double>(std::max<int64_t>(0, n.rope_position_offset))) << ", 1.0f);\n";
            ss << indent << "    const float rope_original_max = "
               << emitScalarFpLiteral(static_cast<double>(n.rope_original_max_position_embeddings)) << ";\n";
            ss << indent << "    const float rope_short_factors[] = " << emitFloatArrayLiteral(n.rope_long_rope_short_factors) << ";\n";
            ss << indent << "    const float rope_long_factors[] = " << emitFloatArrayLiteral(n.rope_long_rope_long_factors) << ";\n";
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
            ss << indent
               << "    rope_freq = powf(rope_base, -2.0f * static_cast<float>(rope_pair_index) / static_cast<float>(rope_rotary_dim));\n";
        }
        ss << indent << "    const float rope_theta = rope_position * rope_freq;\n";
        ss << indent << "    float rope_s = sinf(rope_theta);\n";
        ss << indent << "    float rope_c = cosf(rope_theta);\n";
        if (n.rope_attention_factor != 1.0) {
            ss << indent << "    rope_s *= " << emitScalarFpLiteral(n.rope_attention_factor) << ";\n";
            ss << indent << "    rope_c *= " << emitScalarFpLiteral(n.rope_attention_factor) << ";\n";
        }
        if (n.rope_inverse) {
            ss << indent << "    rope_s = -rope_s;\n";
        }
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
                                       "in" + std::to_string(n.input_slot) + "_chunk_data");
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
            case ExprOp::CEIL:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = ceilf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::FLOOR:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = floorf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::ROUND:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = roundf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::TRUNC:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = truncf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::SIN:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = sinf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::COS:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = cosf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::TAN:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = tanf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::ASIN:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = asinf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::ACOS:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = acosf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::ATAN:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = atanf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::SINH:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = sinhf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::COSH:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = coshf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::ASINH:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = asinhf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::ACOSH:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = acoshf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::ATANH:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = atanhf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::ERF:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = erff(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::ERFC:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = erfcf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::ERFCX:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = erfcxf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::ERFINV:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = erfinvf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::ERFCINV:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = erfcinvf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::TGAMMA:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = tgammaf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::LGAMMA:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = lgammaf(" << CudaSourceEmitter::ref(n.lhs) << ");\n";
                break;
            case ExprOp::DIGAMMA:
                ss << indent << "float " << CudaSourceEmitter::ref(node_idx) << " = thor_digammaf(" << CudaSourceEmitter::ref(n.lhs)
                   << ");\n";
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
            case ExprOp::RESHAPE:
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
static std::string emitVector2SpecialUnary(const std::string& fn, const std::string& x, DataType dtype) {
    if (dtype == DataType::BF16) {
        return "__floats2bfloat162_rn(" + fn + "(float(half(float(" + x + ".x)))), " + fn + "(float(half(float(" + x + ".y)))))";
    }
    return "__floats2half2_rn(" + fn + "(float(half(" + x + ".x))), " + fn + "(float(half(" + x + ".y))))";
}

static std::string emitVector2Half2Tanh(const std::string& x, DataType dtype) {
    if (dtype == DataType::BF16) {
        const std::string x_half2 = "__float22half2_rn(__bfloat1622float2(" + x + "))";
        return "__float22bfloat162_rn(__half22float2(h2tanh(" + x_half2 + ")))";
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
            case ExprOp::CEIL:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Ceil(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::FLOOR:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Floor(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::ROUND:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Round(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::TRUNC:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Trunc(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::SIN:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Sin(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::COS:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Cos(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::TAN:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Tan(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::ASIN:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Asin(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::ACOS:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Acos(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::ATAN:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Atan(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::SINH:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Sinh(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::COSH:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Cosh(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::ASINH:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Asinh(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::ACOSH:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Acosh(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::ATANH:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Atanh(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::ERF:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Erf(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::ERFC:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Erfc(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::ERFCX:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Erfcx(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::ERFINV:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Erfinv(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::ERFCINV:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Erfcinv(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::TGAMMA:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Tgamma(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::LGAMMA:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Lgamma(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::DIGAMMA:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Digamma(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::EXP:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Exp(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::EXPM1:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Expm1(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
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
            case ExprOp::LOG1P:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Log1p(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
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
            case ExprOp::TANH:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Tanh(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::NORMCDF:
                ss << indent << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                   << emitVector2Normcdf(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                break;
            case ExprOp::RESHAPE:
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
        return vectorizedComputeScalarDType(frontier_dtype);
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
        ss << "#include <cuda_fp16.h>\n";
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
                case ExprOp::CEIL:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Ceil(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::FLOOR:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Floor(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::ROUND:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Round(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::TRUNC:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Trunc(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::SIN:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Sin(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::COS:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Cos(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::TAN:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Tan(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::ASIN:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Asin(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::ACOS:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Acos(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::ATAN:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Atan(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::SINH:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Sinh(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::COSH:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Cosh(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::ASINH:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Asinh(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::ACOSH:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Acosh(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::ATANH:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Atanh(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::ERF:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Erf(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::ERFC:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Erfc(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::ERFCX:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Erfcx(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::ERFINV:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Erfinv(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::ERFCINV:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Erfcinv(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::TGAMMA:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Tgamma(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::LGAMMA:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Lgamma(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::DIGAMMA:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Digamma(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::EXP:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Exp(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::EXPM1:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Expm1(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
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
                case ExprOp::LOG1P:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Log1p(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
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
                case ExprOp::TANH:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Tanh(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::NORMCDF:
                    ss << "  " << compute_dtype_vector << " " << refWithSuffix(node_idx, suffix) << " = "
                       << emitVector2Normcdf(refWithSuffix(n.lhs, suffix), dtype) << ";\n";
                    break;
                case ExprOp::RESHAPE:
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
        ss << "#include <cuda_fp16.h>\n";
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
                case ExprOp::CEIL:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Ceil(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::FLOOR:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Floor(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::ROUND:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Round(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::TRUNC:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Trunc(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::SIN:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Sin(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::COS:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Cos(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::TAN:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Tan(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::ASIN:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Asin(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::ACOS:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Acos(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::ATAN:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Atan(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::SINH:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Sinh(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::COSH:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Cosh(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::ASINH:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Asinh(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::ACOSH:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Acosh(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::ATANH:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Atanh(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::ERF:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Erf(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::ERFC:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Erfc(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::ERFCX:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Erfcx(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::ERFINV:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Erfinv(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::ERFCINV:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Erfcinv(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::TGAMMA:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Tgamma(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::LGAMMA:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Lgamma(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::DIGAMMA:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Digamma(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::EXP:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Exp(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::EXPM1:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Expm1(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
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
                case ExprOp::LOG1P:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Log1p(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
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
                case ExprOp::TANH:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Tanh(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::NORMCDF:
                    ss << "    " << compute_dtype_vector << " t" << node_idx << " = "
                       << emitVector2Normcdf(CudaSourceEmitter::ref(n.lhs), dtype) << ";\n";
                    break;
                case ExprOp::RESHAPE:
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
    if (scalarStorageTypeSizeBytes(output_dtype) > sizeof(unsigned int)) {
        throw runtime_error("Tiled transpose shared-memory swizzle currently supports output scalar storage types up to 32 bits.");
    }
    const std::string output_type = scalarStorageType(output_dtype);
    const std::optional<DataType> maybe_vectorized_dtype = CudaSourceEmitter::getVectorizedStageStorageDType(stage);
    const std::optional<DataType> maybe_tensor_input_dtype = getSingleTensorInputStorageDType(stage.expr, input_dtypes);
    const bool emit_packed_low_precision_path = transposePackScalars(output_dtype) > 1;
    const bool emit_homogeneous_packed_vectorized_path =
        maybe_vectorized_dtype.has_value() && maybe_vectorized_dtype.value() == output_dtype && transposePackScalars(output_dtype) > 1;
    const bool emit_mixed_two_byte_float2_path =
        maybe_tensor_input_dtype.has_value() &&
        supportsMixedTwoByteFloat2TransposedVectorization(stage.expr, maybe_tensor_input_dtype.value(), output_dtype);
    const bool emit_mixed_fp8_vectorized_path =
        maybe_tensor_input_dtype.has_value() &&
        supportsMixedFp8TransposedVectorization(stage.expr, maybe_tensor_input_dtype.value(), output_dtype);
    const bool emit_cross_width_float2_path =
        maybe_tensor_input_dtype.has_value() &&
        supportsFloat2TransposedVectorization(stage.expr, maybe_tensor_input_dtype.value(), output_dtype);
    const bool emit_cross_width_half2_path =
        maybe_tensor_input_dtype.has_value() &&
        supportsHalf2TransposedVectorization(stage.expr, maybe_tensor_input_dtype.value(), output_dtype);
    const bool emit_fp8_to_bf16_float2_path =
        maybe_tensor_input_dtype.has_value() &&
        supportsFp8ToBf16Float2TransposedVectorization(stage.expr, maybe_tensor_input_dtype.value(), output_dtype);
    const bool emit_packed_vectorized_path = emit_homogeneous_packed_vectorized_path || emit_mixed_two_byte_float2_path ||
                                             emit_mixed_fp8_vectorized_path || emit_cross_width_float2_path ||
                                             emit_cross_width_half2_path || emit_fp8_to_bf16_float2_path;
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
    ss << "  const " << index_type << " batchIdx = static_cast<" << index_type << ">(blockIdx.y) / rowTiles;\n";
    ss << "  if (batchIdx >= batchCount) return;\n";
    ss << "  const " << index_type << " rowTile = static_cast<" << index_type << ">(blockIdx.y) - batchIdx * rowTiles;\n";
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
                for (uint32_t lane = 0; lane < read_pack_scalars; ++lane) {
                    const std::string suffix = "_dl" + std::to_string(lane);
                    const std::string lane_literal = emitUnsignedLiteral(lane, use_uint32_index_math);
                    ss << base_indent << "{\n";
                    ss << base_indent << "  const unsigned int LANE = " << lane << "U;\n";
                    ss << base_indent << "  const unsigned int LOCAL_COL = localReadPackCol * READ_PACK_SCALARS + LANE;\n";
                    const std::string lane_idx_expr = idx_expr_base + " + " + lane_literal;
                    auto input_value = [&](uint32_t input_slot) -> std::string {
                        return emitFp8PackLaneMemberExpr("in" + std::to_string(input_slot) + "_chunk", lane, input_dtype);
                    };
                    for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
                        if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
                            continue;
                        }
                        emitScalarNodeSuffixed(ss, stage.expr, node_idx, lane_idx_expr, suffix, base_indent + "  ", "", 0, input_value);
                    }
                    ss << base_indent << "  tile[localRow][LOCAL_COL] = thor_pack_transpose_word<" << output_type << ">("
                       << emitResolvedScalarValueExprSuffixed(stage.expr, output.local_node_idx, output_dtype, suffix) << ");\n";
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
               << ");\n";
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
               << emitUnsignedLiteral(0, use_uint32_index_math) << ");\n";
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
                                             emit_cross_width_half2_path || emit_fp8_to_bf16_float2_path;
        const DataType vector_input_dtype = needs_input_vector_type
                                                ? (maybe_tensor_input_dtype.has_value() ? maybe_tensor_input_dtype.value() : output_dtype)
                                                : output_dtype;
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
        ss << "    const " << index_type << " idx_base = matrixOffset + logicalRow * numCols + logicalColBase;\n";
        ss << "    const bool inputPackedLoadOk = (logicalRow < numRows) && (logicalColBase + PACK_SCALARS <= numCols) &&\n";
        ss << "                                   ((idx_base % PACK_SCALARS) == " << emitUnsignedLiteral(0, use_uint32_index_math)
           << ");\n";
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
        ss << "          const " << index_type << " idx = matrixOffset + logicalRow * numCols + logicalCol;\n";
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
        ss << "    const " << index_type << " out_base_idx = matrixOffset + outputRow * numRows + outputColBase;\n";
        ss << "    const bool outputPackedStoreOk = (outputRow < numCols) && (outputColBase + PACK_SCALARS <= numRows) &&\n";
        ss << "                                     ((out_base_idx % PACK_SCALARS) == " << emitUnsignedLiteral(0, use_uint32_index_math)
           << ");\n";
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

    std::optional<DataType> vectorized_dtype = getVectorizedStageStorageDType(stage);
    if (vectorized_dtype.has_value()) {
        return emitVector2Flat(stage, vectorized_dtype.value(), kernel_name, use_uint32_index_math);
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

    const std::unordered_set<uint32_t> index_aware_skip_nodes = collectIndexAwareInputNodesToSkipForFlatOutput(stage.expr, stage.outputs);
    for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
        if (index_aware_skip_nodes.contains(node_idx) || !shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
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
    const bool emit_fp8_vectorized_pair_path = emit_packed_low_precision_path && isFp8DType(output_dtype) &&
                                               maybe_vectorized_dtype.has_value() && maybe_vectorized_dtype.value() == output_dtype;
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
    ss << "  const " << index_type << " batchIdx = static_cast<" << index_type << ">(blockIdx.y) / rowTiles;\n";
    ss << "  if (batchIdx >= batchCount) return;\n";
    ss << "  const " << index_type << " rowTile = static_cast<" << index_type << ">(blockIdx.y) - batchIdx * rowTiles;\n";
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
                   << "_offset_base % READ_PACK_SCALARS) == " << emitUnsignedLiteral(0, use_uint32_index_math) << ");\n";
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
                        if (kind_it->second == SpecializedInputLoadKind::NativeVector) {
                            return emitFp8PackLaneMemberExpr("in" + std::to_string(input_slot) + "_chunk", lane, input_dtype);
                        }
                        return "in" + std::to_string(input_slot) + "[in" + std::to_string(input_slot) + "_offset]";
                    };
                    for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
                        if (!shouldEmitScalarNodeDefinition(stage.expr, node_idx)) {
                            continue;
                        }
                        emitScalarNodeSuffixed(ss, stage.expr, node_idx, "logical_idx", suffix, "        ", "", 0, input_value);
                    }
                    ss << "        tile[localRow][LOCAL_COL] = thor_pack_transpose_word<" << output_type << ">("
                       << emitResolvedScalarValueExprSuffixed(stage.expr, output.local_node_idx, output_dtype, suffix) << ");\n";
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
                        return "in" + std::to_string(input_slot) + "_chunk_data[LANE]";
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
               << emitUnsignedLiteral(0, use_uint32_index_math) << ");\n";
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
        ss << "    Pack output_pack{};\n";
        ss << "    " << output_type << "* output_scalar = reinterpret_cast<" << output_type << "*>(&output_pack);\n";
        ss << "    for (unsigned int lane = 0; lane < PACK_SCALARS; ++lane) {\n";
        ss << "      PackRaw input_raw{};\n";
        ss << "      input_raw.raw = tile[localOutPackCol * PACK_SCALARS + lane][inputPackCol];\n";
        ss << "      const " << output_type << "* input_scalar = reinterpret_cast<const " << output_type << "*>(&input_raw.pack);\n";
        ss << "      output_scalar[lane] = input_scalar[inputColLane];\n";
        ss << "    }\n";
        ss << "    const " << index_type << " out_base_idx = matrixOffset + static_cast<" << index_type << ">(outputRow) * static_cast<"
           << index_type << ">(numRows) +\n";
        ss << "                                     static_cast<" << index_type << ">(outputColBase);\n";
        ss << "    const bool outputPackedStoreOk = (outputRow < numCols) && (outputColBase + PACK_SCALARS <= numRows) &&\n";
        ss << "                                     ((out_base_idx % PACK_SCALARS) == " << emitUnsignedLiteral(0, use_uint32_index_math)
           << ");\n";
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
    ss << "  const " << index_type << " batchIdx = static_cast<" << index_type << ">(blockIdx.y) / rowTiles;\n";
    ss << "  if (batchIdx >= batchCount) return;\n";
    ss << "  const " << index_type << " rowTile = static_cast<" << index_type << ">(blockIdx.y) - batchIdx * rowTiles;\n";
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
               << ">(LOGICAL_TRANSPOSE_PACK_SCALARS)) == static_cast<" << index_type << ">(0));\n";
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
    ss << "  " << index_type << " idx = " << emitFlatThreadIndexExpr(use_uint32_index_math) << ";\n\n";

    for (uint32_t g = 0; g < groups.size(); ++g) {
        const SpecializedBroadcastGroup& group = groups[g];
        if (group.node_dims.size() != stage.expr.nodes.size()) {
            throw runtime_error("Index-mapped specialized broadcast group is missing per-node dimensions.");
        }

        ss << "  if (idx < " << emitUnsignedLiteral(group.numel, use_uint32_index_math) << ") {\n";
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
            const std::string value = emitIndexMappedScalarValue(
                ss, stage.expr, group, output.local_node_idx, "idx", group.output_dims, "    ", use_uint32_index_math, counter);
            ss << "    out" << out_idx
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
        if (expressionHasIndexAwareOps(stage.expr)) {
            throw std::runtime_error("RoPE/index-aware fused stages do not currently support transposed materialized outputs.");
        }
        return emitTiledTransposeMaterializedSpecializedBroadcast(stage, groups, kernel_name);
    }

    if (expressionHasLogicalTransposeOp(stage.expr)) {
        if (CudaSourceEmitter::specializedBroadcastUsesTiledLogicalTransposeConsumerLaunch(stage, groups)) {
            return emitTiledLogicalTransposeConsumerSpecializedBroadcast(stage, groups, kernel_name);
        }
        throw std::runtime_error(
            "Logical transpose inside a fused broadcast stage would require dense column-strided global access, "
            "but this pattern is not currently supported by the auto-swizzle emitter. Materialize the transpose boundary "
            "or simplify the transpose chain before fusing.");
    }

    if (expressionHasTakeAlongAxisOp(stage.expr)) {
        return emitIndexMappedSpecializedBroadcast(stage, groups, kernel_name);
    }

    std::optional<DataType> vectorized_dtype = getVectorizedStageStorageDType(stage);
    if (vectorized_dtype.has_value()) {
        return emitVector2SpecializedBroadcast(stage, groups, vectorized_dtype.value(), kernel_name);
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
        for (size_t axis = 0; axis < group.output_dims.size(); ++axis) {
            ss << "    const " << index_type << " out_dim_" << axis << " = "
               << emitUnsignedLiteral(group.output_dims[axis], use_uint32_index_math) << ";\n";
        }

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
            emitScalarNode(ss, stage.expr, node_idx, /*broadcast_support=*/true, "    ", &group);
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
