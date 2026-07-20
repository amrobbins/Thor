#include "Utilities/Expression/FusedEquation.h"
#include <optional>
#include <array>
#include <algorithm>

#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Expression/CudaSourceEmitter.h"
#include "Utilities/Expression/CudaKernelExpression.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/StampedEquation.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"

#include <cuda_runtime.h>

#include <functional>
#include <limits>
#include <stdexcept>
#include <sstream>
#include "DeepLearning/Implementation/ThorError.h"

using namespace std;
using DataType = ThorImplementation::DataType;

namespace ThorImplementation {

static bool runtimeInputIsTensor(const RuntimeInputValue& value) { return std::holds_alternative<Tensor>(value); }
static bool runtimeInputIsRuntimeScalar(const RuntimeInputValue& value) { return std::holds_alternative<float>(value); }
static bool runtimeInputIsTensorScalarBinding(const RuntimeInputValue& value) { return std::holds_alternative<TensorScalarBinding>(value); }

static const TensorScalarBinding& runtimeInputTensorScalarBinding(const RuntimeInputValue& value) {
    if (!std::holds_alternative<TensorScalarBinding>(value)) {
        throw std::runtime_error("Expected tensor scalar runtime input.");
    }
    return std::get<TensorScalarBinding>(value);
}

static bool optionalRuntimeInputIsTensorLike(const std::optional<RuntimeInputValue>& value) {
    return value.has_value() &&
           (std::holds_alternative<Tensor>(value.value()) || std::holds_alternative<TensorScalarBinding>(value.value()));
}

static size_t dataTypeSizeBytes(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return 4;
        case DataType::FP16:
            return 2;
        case DataType::BF16:
            return 2;
        case DataType::FP8_E4M3:
            return 1;
        case DataType::FP8_E5M2:
            return 1;
        case DataType::BOOLEAN:
            return 1;
        case DataType::UINT8:
            return 1;
        case DataType::INT8:
            return 1;
        case DataType::UINT16:
            return 2;
        case DataType::INT16:
            return 2;
        case DataType::UINT32:
            return 4;
        case DataType::UINT64:
            return 8;
        case DataType::INT32:
            return 4;
        case DataType::INT64:
            return 8;
        default:
            throw std::runtime_error("Unsupported dtype in dataTypeSizeBytes.");
    }
}

static const Tensor& runtimeInputTensor(const RuntimeInputValue& value) {
    if (!std::holds_alternative<Tensor>(value)) {
        throw std::runtime_error("Expected tensor runtime input.");
    }
    return std::get<Tensor>(value);
}

static Tensor runtimeInputTensorScalarView(const RuntimeInputValue& value, const char* label) {
    if (std::holds_alternative<Tensor>(value)) {
        return std::get<Tensor>(value);
    }
    if (!std::holds_alternative<TensorScalarBinding>(value)) {
        throw std::runtime_error(std::string("Expected tensor or tensor-runtime-scalar runtime input for ") + label + ".");
    }

    const TensorScalarBinding& binding = std::get<TensorScalarBinding>(value);
    if (!binding.buffer.isInitialized()) {
        throw std::runtime_error(std::string("Tensor-runtime-scalar buffer is not initialized for ") + label + ".");
    }
    if (binding.buffer.getDataType() != binding.sourceDType) {
        throw std::runtime_error(std::string("Tensor-runtime-scalar binding for ") + label +
                                 " must use a backing buffer with the same dtype as the scalar source dtype.");
    }

    const size_t elem_bytes = dataTypeSizeBytes(binding.sourceDType);
    if (elem_bytes == 0 || binding.byteOffset % elem_bytes != 0) {
        throw std::runtime_error(std::string("Tensor-runtime-scalar binding byte offset is not element-aligned for ") + label + ".");
    }
    if (binding.byteOffset + elem_bytes > binding.buffer.getArraySizeInBytes()) {
        throw std::runtime_error(std::string("Tensor-runtime-scalar binding exceeds backing buffer size for ") + label + ".");
    }

    return binding.buffer.aliasView({1, 1, 1, 1}, {1, 1, 1, 1}, binding.byteOffset / elem_bytes);
}

static std::vector<uint64_t> runtimeInputDims(const RuntimeInputValue& value) {
    if (std::holds_alternative<Tensor>(value)) {
        return std::get<Tensor>(value).getDimensions();
    }
    if (std::holds_alternative<TensorScalarBinding>(value)) {
        return {1};
    }
    return {1};
}

static std::vector<uint64_t> runtimeInputStridesForShapeKey(const RuntimeInputValue& value) {
    if (std::holds_alternative<Tensor>(value)) {
        return std::get<Tensor>(value).getStridesElements();
    }
    if (std::holds_alternative<TensorScalarBinding>(value)) {
        return {1};
    }
    return {};
}

static bool runtimeInputIsNonDenseTensorView(const RuntimeInputValue& value) {
    return std::holds_alternative<Tensor>(value) && !std::get<Tensor>(value).isDenseContiguous();
}

static bool anyRuntimeInputIsNonDenseTensorView(const std::vector<RuntimeInputValue>& values) {
    return std::any_of(values.begin(), values.end(), [](const RuntimeInputValue& value) {
        return runtimeInputIsNonDenseTensorView(value);
    });
}

struct PackedAttentionBackwardDirectSlice {
    size_t attention_slot = 0;
    uint32_t attention_value_id = UINT32_MAX;
    std::vector<uint64_t> view_dims;
    std::vector<uint64_t> view_strides;
    uint64_t view_element_offset = 0;
    uint64_t start = 0;
    uint64_t end = 0;
};

struct PackedAttentionBackwardDirectOutput {
    size_t attention_stage_idx = 0;
    size_t packing_stage_idx = 0;
    uint32_t packed_value_id = UINT32_MAX;
    std::string packed_output_name;
    DataType packed_dtype = DataType::FP16;
    std::vector<uint64_t> packed_dims;
    std::array<std::optional<PackedAttentionBackwardDirectSlice>, 3> slices;
};

static std::optional<size_t> attentionBackwardQkvSlotForOp(ExprOp op) {
    switch (op) {
        case ExprOp::ATTENTION_BACKWARD_Q:
            return 0;
        case ExprOp::ATTENTION_BACKWARD_K:
            return 1;
        case ExprOp::ATTENTION_BACKWARD_V:
            return 2;
        default:
            return std::nullopt;
    }
}

static bool isStaticZeroNodeForPackedAttentionBackwardDirectOutput(const PhysicalExpression& expr, uint32_t node_idx) {
    if (node_idx >= expr.nodes.size()) {
        return false;
    }
    const ExprNode& node = expr.nodes.at(node_idx);
    if ((node.op == ExprOp::SCALAR_FP || node.op == ExprOp::FILL) && node.scalar_fp == 0.0) {
        return true;
    }
    if (node.op == ExprOp::MUL) {
        return isStaticZeroNodeForPackedAttentionBackwardDirectOutput(expr, node.lhs) ||
               isStaticZeroNodeForPackedAttentionBackwardDirectOutput(expr, node.rhs);
    }
    return false;
}

static bool collectStridedViewBackwardSumLeaves(const PhysicalExpression& expr,
                                                uint32_t node_idx,
                                                std::vector<uint32_t>& leaves) {
    if (node_idx >= expr.nodes.size()) {
        return false;
    }

    const ExprNode& node = expr.nodes.at(node_idx);
    if (node.op == ExprOp::ADD) {
        return collectStridedViewBackwardSumLeaves(expr, node.lhs, leaves) &&
               collectStridedViewBackwardSumLeaves(expr, node.rhs, leaves);
    }
    if (node.op == ExprOp::STRIDED_VIEW_BACKWARD) {
        leaves.push_back(node_idx);
        return true;
    }
    return isStaticZeroNodeForPackedAttentionBackwardDirectOutput(expr, node_idx);
}

static bool classifyPackedRowSliceForDirectAttentionBackward(const ExprNode& node,
                                                              uint64_t& start,
                                                              uint64_t& end) {
    if (node.op != ExprOp::STRIDED_VIEW_BACKWARD || node.fill_dims.size() != 2 || node.view_dims.size() < 2 ||
        node.view_dims.size() != node.view_strides.size()) {
        return false;
    }
    if (node.fill_dims[0] == 0 || node.fill_dims[1] == 0) {
        return false;
    }
    for (uint64_t dim : node.view_dims) {
        if (dim == 0) {
            return false;
        }
    }
    for (uint64_t stride : node.view_strides) {
        if (stride == 0) {
            return false;
        }
    }

    const std::vector<uint64_t>& dims = node.view_dims;
    const std::vector<uint64_t>& strides = node.view_strides;
    for (size_t collapsed_last_axis = 0; collapsed_last_axis + 1 < dims.size(); ++collapsed_last_axis) {
        bool ok = true;

        uint64_t dense_suffix_width = 1;
        for (size_t axis = dims.size(); axis-- > collapsed_last_axis + 1;) {
            if (strides[axis] != dense_suffix_width) {
                ok = false;
                break;
            }
            if (dense_suffix_width > std::numeric_limits<uint64_t>::max() / dims[axis]) {
                return false;
            }
            dense_suffix_width *= dims[axis];
        }
        if (!ok) {
            continue;
        }

        uint64_t expected_prefix_stride = strides[collapsed_last_axis];
        for (size_t axis = collapsed_last_axis; axis-- > 0;) {
            if (expected_prefix_stride > std::numeric_limits<uint64_t>::max() / dims[axis + 1]) {
                return false;
            }
            expected_prefix_stride *= dims[axis + 1];
            if (strides[axis] != expected_prefix_stride) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        uint64_t outer = 1;
        for (size_t axis = 0; axis <= collapsed_last_axis; ++axis) {
            if (outer > std::numeric_limits<uint64_t>::max() / dims[axis]) {
                return false;
            }
            outer *= dims[axis];
        }
        const uint64_t row_width = strides[collapsed_last_axis];
        if (outer != node.fill_dims[0] || row_width != node.fill_dims[1]) {
            continue;
        }
        if (node.view_element_offset > row_width || dense_suffix_width > row_width - node.view_element_offset) {
            continue;
        }

        start = node.view_element_offset;
        end = node.view_element_offset + dense_suffix_width;
        return true;
    }

    return false;
}

static std::optional<PackedAttentionBackwardDirectOutput> tryBuildPackedAttentionBackwardDirectOutput(
    const std::vector<CompiledExecutionStage>& stages,
    const std::unordered_map<uint32_t, uint32_t>& consumer_count_by_value_id,
    size_t attention_stage_idx,
    size_t packing_stage_idx) {
    if (attention_stage_idx >= stages.size() || packing_stage_idx >= stages.size()) {
        return std::nullopt;
    }

    const CompiledExecutionStage& attention_stage = stages.at(attention_stage_idx);
    const CompiledExecutionStage& packing_stage = stages.at(packing_stage_idx);
    if (attention_stage.kind != CompiledExecutionStage::Kind::AttentionBackward || !attention_stage.attention_backward ||
        packing_stage.kind != CompiledExecutionStage::Kind::FusedKernel || packing_stage.outputs.size() != 1) {
        return std::nullopt;
    }
    if (packing_stage.outputs.front().local_node_idx >= packing_stage.expr.nodes.size()) {
        return std::nullopt;
    }

    std::unordered_map<uint32_t, size_t> attention_slot_by_value_id;
    std::array<std::optional<uint32_t>, 3> value_id_by_slot;
    std::array<std::optional<ExprOp>, 3> op_by_slot;
    for (const CompiledStageOutput& output : attention_stage.outputs) {
        if (output.local_node_idx >= attention_stage.expr.nodes.size()) {
            return std::nullopt;
        }
        const ExprOp output_op = attention_stage.expr.nodes.at(output.local_node_idx).op;
        const std::optional<size_t> slot = attentionBackwardQkvSlotForOp(output_op);
        if (!slot.has_value()) {
            continue;
        }
        attention_slot_by_value_id.emplace(output.value_id, slot.value());
        value_id_by_slot.at(slot.value()) = output.value_id;
        op_by_slot.at(slot.value()) = output_op;
    }
    for (const auto& value_id : value_id_by_slot) {
        if (!value_id.has_value()) {
            return std::nullopt;
        }
        auto consumer_it = consumer_count_by_value_id.find(value_id.value());
        if (consumer_it == consumer_count_by_value_id.end() || consumer_it->second != 1) {
            return std::nullopt;
        }
    }

    std::vector<uint32_t> leaves;
    if (!collectStridedViewBackwardSumLeaves(packing_stage.expr, packing_stage.outputs.front().local_node_idx, leaves) ||
        leaves.size() != 3) {
        return std::nullopt;
    }

    PackedAttentionBackwardDirectOutput direct;
    direct.attention_stage_idx = attention_stage_idx;
    direct.packing_stage_idx = packing_stage_idx;
    direct.packed_value_id = packing_stage.outputs.front().value_id;
    direct.packed_output_name = packing_stage.outputs.front().name;
    direct.packed_dtype = packing_stage.outputDType(0);

    std::optional<std::vector<uint64_t>> packed_dims;
    for (uint32_t leaf_idx : leaves) {
        const ExprNode& leaf = packing_stage.expr.nodes.at(leaf_idx);
        if (leaf.lhs >= packing_stage.expr.nodes.size()) {
            return std::nullopt;
        }
        const ExprNode& source = packing_stage.expr.nodes.at(leaf.lhs);
        if (source.op != ExprOp::INPUT || source.input_slot >= packing_stage.input_value_ids.size()) {
            return std::nullopt;
        }

        const uint32_t source_value_id = packing_stage.input_value_ids.at(source.input_slot);
        auto slot_it = attention_slot_by_value_id.find(source_value_id);
        if (slot_it == attention_slot_by_value_id.end()) {
            return std::nullopt;
        }
        const size_t slot = slot_it->second;
        if (direct.slices.at(slot).has_value()) {
            return std::nullopt;
        }

        if (!packed_dims.has_value()) {
            packed_dims = leaf.fill_dims;
        } else if (packed_dims.value() != leaf.fill_dims) {
            return std::nullopt;
        }

        if (!op_by_slot.at(slot).has_value()) {
            return std::nullopt;
        }
        const DataType slot_dtype = attention_stage.attention_backward->outputDTypeFor(op_by_slot.at(slot).value());
        if (slot_dtype != direct.packed_dtype) {
            return std::nullopt;
        }

        uint64_t start = 0;
        uint64_t end = 0;
        if (!classifyPackedRowSliceForDirectAttentionBackward(leaf, start, end)) {
            return std::nullopt;
        }

        PackedAttentionBackwardDirectSlice slice;
        slice.attention_slot = slot;
        slice.attention_value_id = source_value_id;
        slice.view_dims = leaf.view_dims;
        slice.view_strides = leaf.view_strides;
        slice.view_element_offset = leaf.view_element_offset;
        slice.start = start;
        slice.end = end;
        direct.slices.at(slot) = std::move(slice);
    }

    if (!packed_dims.has_value() || packed_dims->size() != 2) {
        return std::nullopt;
    }
    direct.packed_dims = packed_dims.value();

    std::vector<PackedAttentionBackwardDirectSlice> sorted_slices;
    sorted_slices.reserve(3);
    for (const auto& slice : direct.slices) {
        if (!slice.has_value()) {
            return std::nullopt;
        }
        sorted_slices.push_back(slice.value());
    }
    std::sort(sorted_slices.begin(), sorted_slices.end(), [](const auto& a, const auto& b) { return a.start < b.start; });
    uint64_t covered_until = 0;
    for (const PackedAttentionBackwardDirectSlice& slice : sorted_slices) {
        if (slice.start != covered_until || slice.end <= slice.start) {
            return std::nullopt;
        }
        covered_until = slice.end;
    }
    if (covered_until != direct.packed_dims.at(1)) {
        return std::nullopt;
    }

    return direct;
}

static std::unordered_map<size_t, PackedAttentionBackwardDirectOutput> findPackedAttentionBackwardDirectOutputs(
    const std::vector<CompiledExecutionStage>& stages,
    const std::vector<CompiledStageOutput>& final_outputs,
    std::unordered_map<size_t, size_t>& attention_stage_by_elided_packing_stage) {
    std::unordered_map<uint32_t, uint32_t> consumer_count_by_value_id;
    for (const CompiledExecutionStage& stage : stages) {
        for (uint32_t value_id : stage.input_value_ids) {
            ++consumer_count_by_value_id[value_id];
        }
    }
    for (const CompiledStageOutput& output : final_outputs) {
        ++consumer_count_by_value_id[output.value_id];
    }

    std::unordered_map<size_t, PackedAttentionBackwardDirectOutput> direct_by_attention_stage;
    for (size_t attention_idx = 0; attention_idx < stages.size(); ++attention_idx) {
        if (stages.at(attention_idx).kind != CompiledExecutionStage::Kind::AttentionBackward) {
            continue;
        }
        for (size_t packing_idx = attention_idx + 1; packing_idx < stages.size(); ++packing_idx) {
            if (attention_stage_by_elided_packing_stage.contains(packing_idx)) {
                continue;
            }
            std::optional<PackedAttentionBackwardDirectOutput> direct =
                tryBuildPackedAttentionBackwardDirectOutput(stages, consumer_count_by_value_id, attention_idx, packing_idx);
            if (!direct.has_value()) {
                continue;
            }
            attention_stage_by_elided_packing_stage.emplace(packing_idx, attention_idx);
            direct_by_attention_stage.emplace(attention_idx, std::move(direct.value()));
            break;
        }
    }
    return direct_by_attention_stage;
}

static constexpr uint64_t EXPRESSION_COPY_DIM = 0;
static constexpr uint64_t EXPRESSION_INFER_DIM = std::numeric_limits<uint64_t>::max();

static uint64_t dimsNumelForRuntimeAlias(const std::vector<uint64_t>& dims, const std::string& what) {
    uint64_t result = 1;
    for (uint64_t dim : dims) {
        if (dim == EXPRESSION_COPY_DIM || dim == EXPRESSION_INFER_DIM) {
            throw std::runtime_error(what + " contains unresolved dynamic dimensions.");
        }
        if (result > std::numeric_limits<uint64_t>::max() / dim) {
            throw std::runtime_error(what + " dimensions overflow uint64_t.");
        }
        result *= dim;
    }
    return result;
}

static std::vector<uint64_t> resolveDynamicAliasDims(const std::vector<uint64_t>& source_dims,
                                                    const std::vector<uint64_t>& requested_dims,
                                                    bool must_preserve_numel,
                                                    const std::string& what) {
    if (requested_dims.empty()) {
        throw std::runtime_error(what + " requires non-empty dimensions.");
    }

    std::vector<uint64_t> resolved = requested_dims;
    std::optional<size_t> infer_index;
    uint64_t known_product = 1;
    for (size_t i = 0; i < resolved.size(); ++i) {
        uint64_t dim = resolved[i];
        if (dim == EXPRESSION_COPY_DIM) {
            if (i >= source_dims.size()) {
                throw std::runtime_error(what + " copy-dimension marker is out of range for source rank.");
            }
            dim = source_dims[i];
            if (dim == 0) {
                throw std::runtime_error(what + " resolved copy dimension must be non-zero.");
            }
            resolved[i] = dim;
        } else if (dim == EXPRESSION_INFER_DIM) {
            if (!must_preserve_numel) {
                throw std::runtime_error(what + " does not support infer-dimension markers.");
            }
            if (infer_index.has_value()) {
                throw std::runtime_error(what + " supports at most one infer-dimension marker.");
            }
            infer_index = i;
            continue;
        } else if (dim == 0) {
            throw std::runtime_error(what + " dimensions must be non-zero after dynamic resolution.");
        }

        if (known_product > std::numeric_limits<uint64_t>::max() / dim) {
            throw std::runtime_error(what + " resolved dimensions overflow uint64_t.");
        }
        known_product *= dim;
    }

    if (must_preserve_numel) {
        const uint64_t source_numel = dimsNumelForRuntimeAlias(source_dims, what + " source");
        if (infer_index.has_value()) {
            if (known_product == 0 || source_numel % known_product != 0) {
                throw std::runtime_error(what + " cannot infer a dimension that preserves the number of elements.");
            }
            resolved[infer_index.value()] = source_numel / known_product;
        } else if (source_numel != known_product) {
            throw std::runtime_error(what + " must preserve the number of elements.");
        }
    }

    return resolved;
}

static std::vector<uint64_t> resolveRuntimeStorageAliasDims(const std::vector<uint64_t>& source_dims, const CompiledValueAlias& alias) {
    if (!alias.strides.empty() && alias.strides.size() != alias.dimensions.size()) {
        throw std::runtime_error("Runtime strided alias dimensions and strides must have the same rank.");
    }
    return resolveDynamicAliasDims(source_dims, alias.dimensions, alias.strides.empty(), "Runtime storage alias");
}

static void applyAvailableValueAliases(const std::vector<CompiledValueAlias>& aliases,
                                       std::unordered_map<uint32_t, std::vector<uint64_t>>& value_dims) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (const CompiledValueAlias& alias : aliases) {
            if (value_dims.contains(alias.value_id)) {
                continue;
            }
            auto source_it = value_dims.find(alias.source_value_id);
            if (source_it == value_dims.end()) {
                continue;
            }
            std::vector<uint64_t> resolved_dims = resolveRuntimeStorageAliasDims(source_it->second, alias);
            value_dims.emplace(alias.value_id, std::move(resolved_dims));
            changed = true;
        }
    }
}

static void applyAvailableValueAliases(const std::vector<CompiledValueAlias>& aliases,
                                       std::unordered_map<uint32_t, DataType>& value_dtypes) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (const CompiledValueAlias& alias : aliases) {
            if (value_dtypes.contains(alias.value_id)) {
                continue;
            }
            auto source_it = value_dtypes.find(alias.source_value_id);
            if (source_it == value_dtypes.end()) {
                continue;
            }
            value_dtypes.emplace(alias.value_id, source_it->second);
            changed = true;
        }
    }
}

static void applyAvailableValueAliases(const std::vector<CompiledValueAlias>& aliases,
                                       std::unordered_map<uint32_t, RuntimeInputValue>& values,
                                       std::unordered_map<uint32_t, uint32_t>* producer_stage_by_value_id = nullptr) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (const CompiledValueAlias& alias : aliases) {
            if (values.contains(alias.value_id)) {
                continue;
            }
            auto source_it = values.find(alias.source_value_id);
            if (source_it == values.end()) {
                continue;
            }
            Tensor tensor = runtimeInputTensor(source_it->second);
            std::vector<uint64_t> resolved_alias_dims = resolveRuntimeStorageAliasDims(tensor.getDimensions(), alias);
            if (alias.strides.empty()) {
                if (!tensor.isDenseContiguous()) {
                    throw std::runtime_error(
                        "Runtime dense reshape alias cannot be applied to a non-dense tensor view; "
                        "materialize the view or use an explicit strided/unsqueeze/squeeze alias.");
                }
                tensor.reshape(resolved_alias_dims);
            } else {
                tensor = tensor.aliasView(resolved_alias_dims, alias.strides, alias.element_offset);
            }
            values.emplace(alias.value_id, tensor);
            if (producer_stage_by_value_id != nullptr) {
                auto producer_it = producer_stage_by_value_id->find(alias.source_value_id);
                if (producer_it != producer_stage_by_value_id->end()) {
                    (*producer_stage_by_value_id)[alias.value_id] = producer_it->second;
                }
            }
            changed = true;
        }
    }
}

static DataType runtimeInputDType(const RuntimeInputValue& value) {
    if (std::holds_alternative<Tensor>(value)) {
        return std::get<Tensor>(value).getDataType();
    }
    if (std::holds_alternative<TensorScalarBinding>(value)) {
        return std::get<TensorScalarBinding>(value).sourceDType;
    }
    return DataType::FP32;
}

static bool resolveLayoutFromDims(const std::vector<std::vector<uint64_t>>& inputs, std::vector<uint64_t>& outputDimensions) {
    if (inputs.empty()) {
        throw std::runtime_error("resolveLayoutFromDims requires at least one input shape.");
    }

    uint64_t maxRank = 0;
    for (const auto& dims : inputs) {
        maxRank = std::max<uint64_t>(maxRank, dims.size());
    }

    outputDimensions.assign(maxRank, 1);

    for (uint64_t axis = 0; axis < maxRank; ++axis) {
        uint64_t resolvedDim = 1;

        for (const auto& dims : inputs) {
            const uint64_t pad = maxRank - dims.size();
            const uint64_t dim = (axis < pad) ? 1ULL : dims[axis - pad];

            if (dim == 1) {
                continue;
            }

            if (resolvedDim == 1) {
                resolvedDim = dim;
            } else if (resolvedDim != dim) {
                throw std::runtime_error("resolveLayoutFromDims found non-broadcast-compatible dimensions.");
            }
        }

        outputDimensions[axis] = resolvedDim;
    }

    bool requiresBroadcast = false;
    for (const auto& dims : inputs) {
        std::vector<uint64_t> padded(maxRank - dims.size(), 1ULL);
        padded.insert(padded.end(), dims.begin(), dims.end());
        if (padded != outputDimensions) {
            requiresBroadcast = true;
            break;
        }
    }

    return requiresBroadcast;
}

static std::vector<uint64_t> applyNormalizedSqueezeDims(const std::vector<uint64_t>& input_dims,
                                                        const std::vector<uint64_t>& squeeze_axes) {
    // Scalar literals are already rank-0. Any squeeze is shape-preserving for stage inference.
    if (input_dims.empty()) {
        return {};
    }

    const std::vector<uint64_t> actual_axes = normalizeSqueezeAxesForInputDims(input_dims, squeeze_axes);

    if (actual_axes.empty()) {
        return input_dims;
    }

    std::vector<uint64_t> output_dims;
    output_dims.reserve(input_dims.size() - actual_axes.size());

    size_t next_remove = 0;
    for (uint64_t axis = 0; axis < input_dims.size(); ++axis) {
        if (next_remove < actual_axes.size() && actual_axes[next_remove] == axis) {
            ++next_remove;
            continue;
        }
        output_dims.push_back(input_dims[axis]);
    }

    return output_dims;
}

static std::vector<uint64_t> applyNormalizedUnsqueezeDims(const std::vector<uint64_t>& input_dims,
                                                          const std::vector<uint64_t>& unsqueeze_axes) {
    // Scalar literals are rank-0 broadcastable leaves in the IR.
    // Unsqueezing a scalar produces an all-ones shape of the requested rank.
    if (input_dims.empty()) {
        std::vector<uint64_t> normalized = unsqueeze_axes;
        std::sort(normalized.begin(), normalized.end());
        normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());

        for (uint64_t i = 0; i < normalized.size(); ++i) {
            if (normalized[i] != i) {
                throw std::runtime_error("unsqueeze axes are invalid for scalar input.");
            }
        }

        return std::vector<uint64_t>(normalized.size(), 1ULL);
    }

    const std::vector<uint64_t> actual_axes = normalizeUnsqueezeAxesForInputDims(input_dims, unsqueeze_axes);

    std::vector<uint64_t> output_dims = input_dims;
    for (uint64_t axis : actual_axes) {
        output_dims.insert(output_dims.begin() + static_cast<std::ptrdiff_t>(axis), 1ULL);
    }
    return output_dims;
}

static const std::optional<TensorPlacement> runtimeInputPlacementOrNull(const RuntimeInputValue& value) {
    if (std::holds_alternative<Tensor>(value)) {
        return std::get<Tensor>(value).getPlacement();
    }
    if (std::holds_alternative<TensorScalarBinding>(value)) {
        return std::get<TensorScalarBinding>(value).buffer.getPlacement();
    }
    return std::nullopt;
}

static bool tryGetLowerableScaleNode(const PhysicalExpression& expr, uint32_t node_idx, double& scale_fp, uint32_t& scale_node_idx) {
    if (node_idx >= expr.nodes.size()) {
        return false;
    }

    const ExprNode& node = expr.nodes[node_idx];
    if (node.op == ExprOp::SCALAR_FP) {
        scale_fp = node.scalar_fp;
        scale_node_idx = UINT32_MAX;
        return true;
    }
    if (node.op == ExprOp::RUNTIME_SCALAR || node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
        scale_fp = 1.0;
        scale_node_idx = node_idx;
        return true;
    }
    return false;
}

static bool expressionHasPotentialGemmLoweringPattern(const PhysicalExpression& expr) {
    bool saw_matmul = false;
    bool saw_gemm_or_add_sub = false;
    bool saw_activation_candidate = false;

    for (const ExprNode& node : expr.nodes) {
        switch (node.op) {
            case ExprOp::MATMUL:
                saw_matmul = true;
                break;

            case ExprOp::GEMM:
            case ExprOp::ADD:
            case ExprOp::SUB:
                saw_gemm_or_add_sub = true;
                break;

            case ExprOp::MAX:
            case ExprOp::MUL:
            case ExprOp::NORMCDF:
                saw_activation_candidate = true;
                break;

            default:
                break;
        }
    }

    return saw_matmul && (saw_gemm_or_add_sub || saw_activation_candidate);
}

static std::vector<uint64_t> inferExpressionMatmulOutputDims(const ExprNode& node,
                                                             const std::vector<uint64_t>& lhs_dims,
                                                             const std::vector<uint64_t>& rhs_dims,
                                                             const std::vector<uint64_t>* aux_dims = nullptr) {
    if (lhs_dims.size() != 2 || rhs_dims.size() != 2) {
        throw std::runtime_error("Matmul/gemm shape inference currently only supports rank-2 tensors.");
    }

    const uint64_t a_rows = node.transpose_lhs ? lhs_dims[1] : lhs_dims[0];
    const uint64_t a_cols = node.transpose_lhs ? lhs_dims[0] : lhs_dims[1];
    const uint64_t b_rows = node.transpose_rhs ? rhs_dims[1] : rhs_dims[0];
    const uint64_t b_cols = node.transpose_rhs ? rhs_dims[0] : rhs_dims[1];

    if (a_cols != b_rows) {
        throw std::runtime_error("Matmul/gemm shape inference found incompatible matrix dimensions.");
    }

    std::vector<uint64_t> out_dims{a_rows, b_cols};
    if (aux_dims) {
        if (aux_dims->size() == 1) {
            if (node.transpose_aux || aux_dims->at(0) != out_dims[1]) {
                throw std::runtime_error("GEMM bias epilogue addend must have shape [output_columns].");
            }
        } else if (aux_dims->size() == 2) {
            const std::vector<uint64_t> expected_aux = node.transpose_aux ? std::vector<uint64_t>{out_dims[1], out_dims[0]} : out_dims;
            if (*aux_dims != expected_aux) {
                throw std::runtime_error("GEMM addend shape inference found incompatible addend dimensions.");
            }
        } else {
            throw std::runtime_error("GEMM addend shape inference currently supports rank-2 addends or rank-1 bias epilogue vectors.");
        }
    }

    return out_dims;
}

static bool isConv3DOp(ExprOp op) {
    return op == ExprOp::CONV3D || op == ExprOp::CONV3D_BACKWARD_DATA || op == ExprOp::CONV3D_BACKWARD_FILTER;
}

static std::vector<uint64_t> inferExpressionConvolutionOutputDims(const ExprNode& node,
                                                                  const std::vector<uint64_t>& input_dims,
                                                                  const std::vector<uint64_t>& filter_dims) {
    const bool is_3d = isConv3DOp(node.op);
    const size_t rank = is_3d ? 5 : 4;
    if (input_dims.size() != rank || filter_dims.size() != rank) {
        throw std::runtime_error(is_3d ? "CONV3D shape inference requires rank-5 NCDHW input/filter tensors."
                                       : "CONV2D shape inference requires rank-4 NCHW input/filter tensors.");
    }

    if (input_dims[1] != filter_dims[1]) {
        throw std::runtime_error("Convolution shape inference found mismatched input/filter channels.");
    }

    std::vector<uint64_t> out_dims{input_dims[0], filter_dims[0]};
    const std::vector<int32_t> strides = is_3d ? std::vector<int32_t>{node.conv_stride_d, node.conv_stride_h, node.conv_stride_w}
                                               : std::vector<int32_t>{node.conv_stride_h, node.conv_stride_w};
    const std::vector<int32_t> pads = is_3d ? std::vector<int32_t>{node.conv_pad_d, node.conv_pad_h, node.conv_pad_w}
                                            : std::vector<int32_t>{node.conv_pad_h, node.conv_pad_w};

    for (size_t i = 0; i < strides.size(); ++i) {
        const size_t dim_idx = 2 + i;
        const int64_t numer = static_cast<int64_t>(input_dims[dim_idx]) + 2LL * pads[i] - static_cast<int64_t>(filter_dims[dim_idx]);
        if (numer < 0) {
            throw std::runtime_error("Convolution shape inference produced negative output extent.");
        }
        out_dims.push_back(static_cast<uint64_t>(numer / strides[i] + 1));
    }

    return out_dims;
}

static std::vector<uint64_t> inferExpressionConvolutionBackwardDataOutputDims(const ExprNode& node,
                                                                              const std::vector<uint64_t>& filter_dims,
                                                                              const std::vector<uint64_t>& grad_output_dims) {
    const bool is_3d = isConv3DOp(node.op);
    const size_t rank = is_3d ? 5 : 4;
    if (filter_dims.size() != rank || grad_output_dims.size() != rank) {
        throw std::runtime_error(is_3d ? "CONV3D_BACKWARD_DATA shape inference requires rank-5 tensors."
                                       : "CONV2D_BACKWARD_DATA shape inference requires rank-4 tensors.");
    }

    const uint64_t k = filter_dims[0];
    const uint64_t c = filter_dims[1];
    const uint64_t n = grad_output_dims[0];
    const uint64_t grad_k = grad_output_dims[1];

    if (k != grad_k) {
        throw std::runtime_error("Convolution backward-data shape inference found mismatched filter/output channels.");
    }
    if (!node.fill_dims.empty()) {
        if (node.fill_dims.size() != rank) {
            throw std::runtime_error("Convolution backward-data explicit output shape rank mismatch.");
        }
        if (node.fill_dims[0] != n || node.fill_dims[1] != c) {
            throw std::runtime_error("Convolution backward-data explicit output shape is incompatible with batch/channels.");
        }
        return node.fill_dims;
    }

    std::vector<uint64_t> out_dims{n, c};
    const std::vector<int32_t> strides = is_3d ? std::vector<int32_t>{node.conv_stride_d, node.conv_stride_h, node.conv_stride_w}
                                               : std::vector<int32_t>{node.conv_stride_h, node.conv_stride_w};
    const std::vector<int32_t> pads = is_3d ? std::vector<int32_t>{node.conv_pad_d, node.conv_pad_h, node.conv_pad_w}
                                            : std::vector<int32_t>{node.conv_pad_h, node.conv_pad_w};
    for (size_t i = 0; i < strides.size(); ++i) {
        const size_t dim_idx = 2 + i;
        const int64_t extent =
            static_cast<int64_t>(grad_output_dims[dim_idx] - 1) * strides[i] - 2LL * pads[i] + static_cast<int64_t>(filter_dims[dim_idx]);
        if (extent <= 0) {
            throw std::runtime_error("Convolution backward-data shape inference produced non-positive output extent.");
        }
        out_dims.push_back(static_cast<uint64_t>(extent));
    }

    return out_dims;
}

static std::vector<uint64_t> inferExpressionConvolutionBackwardFilterOutputDims(const ExprNode& node,
                                                                                const std::vector<uint64_t>& input_dims,
                                                                                const std::vector<uint64_t>& grad_output_dims) {
    const bool is_3d = isConv3DOp(node.op);
    const size_t rank = is_3d ? 5 : 4;
    if (input_dims.size() != rank || grad_output_dims.size() != rank) {
        throw std::runtime_error(is_3d ? "CONV3D_BACKWARD_FILTER shape inference requires rank-5 tensors."
                                       : "CONV2D_BACKWARD_FILTER shape inference requires rank-4 tensors.");
    }

    if (input_dims[0] != grad_output_dims[0]) {
        throw std::runtime_error("Convolution backward-filter shape inference found mismatched batch sizes.");
    }
    const uint64_t c = input_dims[1];
    const uint64_t k = grad_output_dims[1];
    if (!node.fill_dims.empty()) {
        if (node.fill_dims.size() != rank) {
            throw std::runtime_error("Convolution backward-filter explicit output shape rank mismatch.");
        }
        if (node.fill_dims[0] != k || node.fill_dims[1] != c) {
            throw std::runtime_error("Convolution backward-filter explicit output shape is incompatible with channels.");
        }
        return node.fill_dims;
    }

    std::vector<uint64_t> out_dims{k, c};
    const std::vector<int32_t> strides = is_3d ? std::vector<int32_t>{node.conv_stride_d, node.conv_stride_h, node.conv_stride_w}
                                               : std::vector<int32_t>{node.conv_stride_h, node.conv_stride_w};
    const std::vector<int32_t> pads = is_3d ? std::vector<int32_t>{node.conv_pad_d, node.conv_pad_h, node.conv_pad_w}
                                            : std::vector<int32_t>{node.conv_pad_h, node.conv_pad_w};
    for (size_t i = 0; i < strides.size(); ++i) {
        const size_t dim_idx = 2 + i;
        const int64_t extent =
            static_cast<int64_t>(input_dims[dim_idx]) + 2LL * pads[i] - static_cast<int64_t>(grad_output_dims[dim_idx] - 1) * strides[i];
        if (extent <= 0) {
            throw std::runtime_error("Convolution backward-filter shape inference produced non-positive filter extent.");
        }
        out_dims.push_back(static_cast<uint64_t>(extent));
    }

    return out_dims;
}

static std::vector<uint64_t> inferRmsNormOutputDims(const ExprNode& node,
                                                    const std::vector<uint64_t>& input_dims,
                                                    const std::vector<uint64_t>& scale_dims);

static std::vector<uint64_t> inferEmbeddingLookupOutputDims(const std::vector<uint64_t>& index_dims,
                                                             const std::vector<uint64_t>& weights_dims) {
    if (weights_dims.size() != 2 || weights_dims[0] == 0 || weights_dims[1] == 0) {
        throw std::runtime_error("EmbeddingLookup shape inference requires weights shape [vocabulary_size, embedding_dim].");
    }
    std::vector<uint64_t> out = index_dims;
    out.push_back(weights_dims[1]);
    return out;
}

static std::vector<uint64_t> inferTransposeOutputDims(const std::vector<uint64_t>& input_dims) {
    if (input_dims.size() < 2) {
        throw std::runtime_error("Transpose shape inference requires rank >= 2 tensors.");
    }
    std::vector<uint64_t> out_dims = input_dims;
    std::swap(out_dims[out_dims.size() - 2], out_dims[out_dims.size() - 1]);
    return out_dims;
}

static std::vector<uint64_t> resolveReductionAxesForInputRank(const std::vector<uint64_t>& reduction_axes, size_t input_rank) {
    if (!reduction_axes.empty()) {
        return reduction_axes;
    }

    std::vector<uint64_t> axes(input_rank);
    for (size_t i = 0; i < input_rank; ++i) {
        axes[i] = static_cast<uint64_t>(i);
    }
    return axes;
}

static uint64_t normalizedTakeAlongAxis(const ExprNode& node, uint64_t rank) {
    if (rank == 0) {
        throw std::runtime_error("take_along_axis requires rank >= 1.");
    }
    if (node.reduction_axes.empty()) {
        return rank - 1;
    }
    const uint64_t encoded_axis = node.reduction_axes.front();
    if (encoded_axis == UINT64_MAX) {
        return rank - 1;
    }
    if (encoded_axis >= rank) {
        throw std::runtime_error("take_along_axis axis is out of range for the input rank.");
    }
    return encoded_axis;
}

static std::vector<uint64_t> inferTakeAlongAxisOutputDims(const ExprNode& node,
                                                          const std::vector<uint64_t>& input_dims,
                                                          const std::vector<uint64_t>& indices_dims) {
    if (input_dims.empty() || indices_dims.empty()) {
        throw std::runtime_error("take_along_axis requires non-scalar input and indices tensors.");
    }
    if (input_dims.size() != indices_dims.size()) {
        throw std::runtime_error("take_along_axis input and indices must have the same rank.");
    }
    const uint64_t axis = normalizedTakeAlongAxis(node, static_cast<uint64_t>(input_dims.size()));
    for (uint64_t dim = 0; dim < input_dims.size(); ++dim) {
        if (dim == axis) {
            continue;
        }
        if (input_dims[dim] != indices_dims[dim]) {
            throw std::runtime_error("take_along_axis input and indices dimensions must match except on the selected axis.");
        }
    }
    return indices_dims;
}

struct AttentionTensorLogicalDims {
    uint64_t batch = 0;
    uint64_t heads = 0;
    uint64_t sequence_length = 0;
    uint64_t head_dim = 0;
};

static AttentionTensorLogicalDims logicalAttentionDims(const std::vector<uint64_t>& dims,
                                                       AttentionTensorLayout layout,
                                                       const char* tensor_name) {
    if (dims.size() != 4) {
        throw std::runtime_error(std::string("Attention stage requires rank-4 tensor '") + tensor_name + "'.");
    }

    switch (layout) {
        case AttentionTensorLayout::BHSD:
            return {dims.at(0), dims.at(1), dims.at(2), dims.at(3)};
        case AttentionTensorLayout::BSHD:
            return {dims.at(0), dims.at(2), dims.at(1), dims.at(3)};
        default:
            throw std::runtime_error(std::string("Unsupported attention layout for tensor '") + tensor_name + "'.");
    }
}

static bool isAllowedAttentionBiasDims(const std::vector<uint64_t>& dims,
                                           uint64_t batch,
                                           uint64_t heads,
                                           uint64_t query_len,
                                           uint64_t kv_len) {
    return dims.size() == 4 && (dims[0] == 1 || dims[0] == batch) && (dims[1] == 1 || dims[1] == heads) &&
           (dims[2] == query_len || dims[2] == 1) && (dims[3] == kv_len || dims[3] == 1);
}

static std::string attentionBiasShapeDescription(uint64_t batch, uint64_t heads, uint64_t query_len, uint64_t kv_len) {
    return "[1|B,1|Hq,1|Sq,1|Skv] for B/Hq/Sq/Skv=[" + std::to_string(batch) + "," + std::to_string(heads) +
           "," + std::to_string(query_len) + "," + std::to_string(kv_len) + "]";
}

static std::vector<uint64_t> thorAttentionDims(AttentionTensorLayout layout,
                                               uint64_t batch,
                                               uint64_t heads,
                                               uint64_t sequence_length,
                                               uint64_t head_dim) {
    switch (layout) {
        case AttentionTensorLayout::BHSD:
            return {batch, heads, sequence_length, head_dim};
        case AttentionTensorLayout::BSHD:
            return {batch, sequence_length, heads, head_dim};
        default:
            throw std::runtime_error("Unsupported attention output layout.");
    }
}

static uint64_t numelFromDims(const std::vector<uint64_t>& dims);

static std::vector<uint64_t> inferRaggedValuewiseExtentDims(const ExprNode& node,
                                                               const std::vector<uint64_t>& values_dims,
                                                               const std::vector<uint64_t>& offsets_dims) {
    if (node.ragged_runtime_max_active_values == 0 || node.ragged_runtime_elements_per_value == 0) {
        throw std::runtime_error("ragged valuewise extent metadata must be non-zero.");
    }
    if (node.ragged_runtime_batch_size == std::numeric_limits<uint64_t>::max()) {
        throw std::runtime_error("ragged valuewise extent batch size overflows offsets element count.");
    }
    if (offsets_dims != std::vector<uint64_t>{node.ragged_runtime_batch_size + 1}) {
        throw std::runtime_error("ragged valuewise extent offsets must have shape [B + 1].");
    }
    uint64_t expected_numel = node.ragged_runtime_max_active_values;
    if (node.ragged_runtime_elements_per_value > std::numeric_limits<uint64_t>::max() / expected_numel) {
        throw std::runtime_error("ragged valuewise extent maximum element count overflows uint64_t.");
    }
    expected_numel *= node.ragged_runtime_elements_per_value;
    if (numelFromDims(values_dims) != expected_numel) {
        throw std::runtime_error("ragged valuewise extent metadata does not match values capacity.");
    }
    return values_dims;
}

static std::vector<std::vector<uint64_t>> inferExpressionNodeDimsForOptimization(
    const PhysicalExpression& expr, const std::unordered_map<uint32_t, RuntimeInputValue>& root_values) {
    std::vector<std::vector<uint64_t>> node_dims(expr.nodes.size());

    for (size_t i = 0; i < expr.nodes.size(); ++i) {
        const ExprNode& node = expr.nodes[i];

        switch (node.op) {
            case ExprOp::INPUT: {
                auto it = root_values.find(node.input_slot);
                if (it == root_values.end()) {
                    throw std::runtime_error("Missing bound runtime input while inferring expression node dims.");
                }
                node_dims[i] = runtimeInputDims(it->second);
                break;
            }
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
            case ExprOp::SCALAR_FP:
                node_dims[i] = {1};
                break;
            case ExprOp::FILL:
                node_dims[i] = node.fill_dims;
                break;
            case ExprOp::RAGGED_VALUEWISE_EXTENT:
                node_dims[i] = inferRaggedValuewiseExtentDims(node, node_dims.at(node.lhs), node_dims.at(node.rhs));
                break;
            case ExprOp::ADD:
            case ExprOp::SUB:
            case ExprOp::MUL:
            case ExprOp::DIV:
            case ExprOp::POW:
            case ExprOp::EQUAL:
            case ExprOp::NOT_EQUAL:
            case ExprOp::LESS:
            case ExprOp::LESS_EQUAL:
            case ExprOp::GREATER:
            case ExprOp::GREATER_EQUAL:
            case ExprOp::LOGICAL_AND:
            case ExprOp::LOGICAL_OR:
            case ExprOp::MIN:
            case ExprOp::MAX:
            case ExprOp::MIN_GRAD_LEFT:
            case ExprOp::MIN_GRAD_RIGHT:
            case ExprOp::MAX_GRAD_LEFT:
            case ExprOp::MAX_GRAD_RIGHT: {
                std::vector<std::vector<uint64_t>> non_scalar_inputs;
                if (!node_dims[node.lhs].empty()) {
                    non_scalar_inputs.push_back(node_dims[node.lhs]);
                }
                if (!node_dims[node.rhs].empty()) {
                    non_scalar_inputs.push_back(node_dims[node.rhs]);
                }
                if (non_scalar_inputs.empty()) {
                    node_dims[i] = {};
                } else if (non_scalar_inputs.size() == 1) {
                    node_dims[i] = non_scalar_inputs[0];
                } else {
                    std::vector<uint64_t> out_dims;
                    resolveLayoutFromDims(non_scalar_inputs, out_dims);
                    node_dims[i] = std::move(out_dims);
                }
                break;
            }
            case ExprOp::WHERE: {
                std::vector<std::vector<uint64_t>> non_scalar_inputs;
                if (!node_dims[node.lhs].empty()) {
                    non_scalar_inputs.push_back(node_dims[node.lhs]);
                }
                if (!node_dims[node.rhs].empty()) {
                    non_scalar_inputs.push_back(node_dims[node.rhs]);
                }
                if (!node_dims[node.aux].empty()) {
                    non_scalar_inputs.push_back(node_dims[node.aux]);
                }
                if (non_scalar_inputs.empty()) {
                    node_dims[i] = {};
                } else if (non_scalar_inputs.size() == 1) {
                    node_dims[i] = non_scalar_inputs[0];
                } else {
                    std::vector<uint64_t> out_dims;
                    resolveLayoutFromDims(non_scalar_inputs, out_dims);
                    node_dims[i] = std::move(out_dims);
                }
                break;
            }
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
            case ExprOp::CAST:
                node_dims[i] = node_dims[node.lhs];
                break;
            case ExprOp::ROPE: {
                node_dims[i] = node_dims[node.lhs];
                const std::vector<uint64_t>& dims = node_dims[i];
                if (dims.empty()) {
                    throw std::runtime_error("RoPE requires a tensor input.");
                }
                if (node.rope_sequence_axis >= dims.size() || node.rope_head_dim_axis >= dims.size()) {
                    throw std::runtime_error("RoPE sequence_axis/head_dim_axis are out of range for the input rank.");
                }
                if (node.rope_head_dim_axis + 1 != dims.size()) {
                    throw std::runtime_error(
                        "RoPE currently requires head_dim_axis to be the innermost dimension for coalesced pair rotation.");
                }
                const uint64_t head_dim = dims[node.rope_head_dim_axis];
                const uint64_t rotary_dim = node.rope_rotary_dim == 0 ? head_dim : node.rope_rotary_dim;
                if (rotary_dim == 0 || (rotary_dim & 1ULL) != 0ULL || rotary_dim > head_dim) {
                    throw std::runtime_error("RoPE rotary_dim must be even, non-zero, and <= the head dimension.");
                }
                if (node.rope_scaling_kind == RotaryScalingKind::LongRope) {
                    const uint64_t expected = rotary_dim / 2;
                    if (node.rope_long_rope_short_factors.size() != expected ||
                        node.rope_long_rope_long_factors.size() != expected) {
                        throw std::runtime_error("LongRoPE short/long factor lists must have length rotary_dim / 2.");
                    }
                }
                break;
            }
            case ExprOp::RMSNORM:
                node_dims[i] = inferRmsNormOutputDims(node, node_dims[node.lhs], node_dims[node.rhs]);
                break;
            case ExprOp::EMBEDDING_LOOKUP:
                node_dims[i] = inferEmbeddingLookupOutputDims(node_dims[node.lhs], node_dims[node.rhs]);
                break;
            case ExprOp::TAKE_ALONG_AXIS:
                node_dims[i] = inferTakeAlongAxisOutputDims(node, node_dims[node.lhs], node_dims[node.rhs]);
                break;
            case ExprOp::TRANSPOSE:
                node_dims[i] = inferTransposeOutputDims(node_dims[node.lhs]);
                break;
            case ExprOp::RESHAPE:
                node_dims[i] = resolveDynamicAliasDims(node_dims[node.lhs], node.reshape_dims, true, "Expression reshape");
                break;
            case ExprOp::STRIDED_VIEW:
                if (node.view_dims.empty() || node.view_strides.size() != node.view_dims.size()) {
                    throw std::runtime_error("Expression strided_view requires dimensions and strides with the same non-zero rank.");
                }
                node_dims[i] = resolveDynamicAliasDims(node_dims[node.lhs], node.view_dims, false, "Expression strided_view");
                break;
            case ExprOp::STRIDED_VIEW_BACKWARD:
                if (node.fill_dims.empty() || node.view_dims.empty() || node.view_strides.size() != node.view_dims.size()) {
                    throw std::runtime_error(
                        "Expression strided_view_backward requires source dimensions and matching view dimensions/strides.");
                }
                node_dims[i] = node.fill_dims;
                break;
            case ExprOp::UNSQUEEZE:
                node_dims[i] = applyNormalizedUnsqueezeDims(node_dims[node.lhs], node.unsqueeze_axes);
                break;
            case ExprOp::SQUEEZE:
                node_dims[i] = applyNormalizedSqueezeDims(node_dims[node.lhs], node.squeeze_axes);
                break;
            case ExprOp::REDUCE_SUM:
            case ExprOp::REDUCE_PROD:
            case ExprOp::REDUCE_MIN:
            case ExprOp::REDUCE_MAX:
            case ExprOp::REDUCE_ARGMIN:
            case ExprOp::REDUCE_ARGMAX:
            case ExprOp::REDUCE_AVG:
            case ExprOp::REDUCE_NORM1:
            case ExprOp::REDUCE_NORM2: {
                const std::vector<uint64_t>& lhs_dims = node_dims[node.lhs];
                const std::vector<uint64_t> reduction_axes = resolveReductionAxesForInputRank(node.reduction_axes, lhs_dims.size());
                node_dims[i] = StampedEquation::computeReductionOutputDims(lhs_dims, reduction_axes, node.squeeze_axes);
                break;
            }
            case ExprOp::MATMUL:
                node_dims[i] = inferExpressionMatmulOutputDims(node, node_dims[node.lhs], node_dims[node.rhs]);
                break;
            case ExprOp::GEMM:
                node_dims[i] = inferExpressionMatmulOutputDims(node, node_dims[node.lhs], node_dims[node.rhs], &node_dims[node.aux]);
                break;
            case ExprOp::CONV2D:
            case ExprOp::CONV3D:
                node_dims[i] = inferExpressionConvolutionOutputDims(node, node_dims[node.lhs], node_dims[node.rhs]);
                break;
            case ExprOp::CONV2D_BACKWARD_DATA:
            case ExprOp::CONV3D_BACKWARD_DATA:
                node_dims[i] = inferExpressionConvolutionBackwardDataOutputDims(node, node_dims[node.lhs], node_dims[node.rhs]);
                break;
            case ExprOp::CONV2D_BACKWARD_FILTER:
            case ExprOp::CONV3D_BACKWARD_FILTER:
                node_dims[i] = inferExpressionConvolutionBackwardFilterOutputDims(node, node_dims[node.lhs], node_dims[node.rhs]);
                break;
            case ExprOp::REDUCE_MIN_BACKWARD:
            case ExprOp::REDUCE_MAX_BACKWARD:
            case ExprOp::SCAN_MIN_BACKWARD:
            case ExprOp::SCAN_MAX_BACKWARD:
            case ExprOp::SEGMENTED_SCAN_MIN_BACKWARD:
            case ExprOp::SEGMENTED_SCAN_MAX_BACKWARD:
                node_dims[i] = node_dims[node.lhs];
                break;
            case ExprOp::ATTENTION: {
                if (node.lhs == UINT32_MAX || node.rhs == UINT32_MAX || node.aux == UINT32_MAX) {
                    throw std::runtime_error("Attention shape inference for GEMM optimization found a malformed ATTENTION node.");
                }
                const AttentionTensorLogicalDims q_dims = logicalAttentionDims(node_dims[node.lhs], node.attention_q_layout, "q");
                const AttentionTensorLogicalDims k_dims = logicalAttentionDims(node_dims[node.rhs], node.attention_k_layout, "k");
                const AttentionTensorLogicalDims v_dims = logicalAttentionDims(node_dims[node.aux], node.attention_v_layout, "v");
                if (q_dims.batch != k_dims.batch || q_dims.batch != v_dims.batch) {
                    throw std::runtime_error("Attention shape inference for GEMM optimization requires matching q/k/v batch dimensions.");
                }
                if (k_dims.heads != v_dims.heads || q_dims.heads % k_dims.heads != 0) {
                    throw std::runtime_error("Attention shape inference for GEMM optimization found incompatible q/k/v head counts.");
                }
                if (k_dims.sequence_length != v_dims.sequence_length || q_dims.head_dim != k_dims.head_dim) {
                    throw std::runtime_error("Attention shape inference for GEMM optimization found incompatible q/k/v sequence/head dimensions.");
                }
                node_dims[i] = thorAttentionDims(node.attention_o_layout, q_dims.batch, q_dims.heads, q_dims.sequence_length, v_dims.head_dim);
                break;
            }
            case ExprOp::ATTENTION_BACKWARD_Q:
                node_dims[i] = node_dims[node.lhs];
                break;
            case ExprOp::ATTENTION_BACKWARD_K:
                node_dims[i] = node_dims[node.rhs];
                break;
            case ExprOp::ATTENTION_BACKWARD_V:
                node_dims[i] = node_dims[node.aux];
                break;
            case ExprOp::ATTENTION_BACKWARD_BIAS:
                if (!node.attention_use_bias || node.alpha_node == UINT32_MAX) {
                    throw std::runtime_error("Attention dBias shape inference for GEMM optimization requires a bias input.");
                }
                node_dims[i] = node_dims[node.alpha_node];
                break;
            default:
                throw std::runtime_error("inferExpressionNodeDimsForOptimization encountered unknown ExprOp.");
        }
    }

    return node_dims;
}

struct ScaledMatmulPattern {
    uint32_t matmul_node_idx = UINT32_MAX;
    uint32_t lhs_idx = UINT32_MAX;
    uint32_t rhs_idx = UINT32_MAX;
    bool transpose_lhs = false;
    bool transpose_rhs = false;
    double alpha_scale = 1.0;
    uint32_t alpha_node = UINT32_MAX;
    std::optional<DataType> compute_dtype = std::nullopt;
    std::optional<DataType> backward_compute_dtype = std::nullopt;
};

struct ScaledAddendPattern {
    uint32_t node_idx = UINT32_MAX;
    double beta_scale = 1.0;
    uint32_t beta_node = UINT32_MAX;
    bool is_bias_vector = false;
};

struct GemmLoweringPattern {
    uint32_t lhs_idx = UINT32_MAX;
    uint32_t rhs_idx = UINT32_MAX;
    uint32_t addend_idx = UINT32_MAX;
    bool is_bias_epilogue = false;
    std::optional<DataType> matmul_output_dtype = std::nullopt;
    bool transpose_lhs = false;
    bool transpose_rhs = false;
    double alpha_scale = 1.0;
    double beta_scale = 1.0;
    uint32_t alpha_node = UINT32_MAX;
    uint32_t beta_node = UINT32_MAX;
    std::optional<DataType> inherited_compute_dtype = std::nullopt;
    std::optional<DataType> inherited_backward_compute_dtype = std::nullopt;
};

struct MatmulActivationEpiloguePattern {
    uint32_t source_idx = UINT32_MAX;
    MatmulEpilogue epilogue = MatmulEpilogue::Default;
};

static bool isMatmulOp(ExprOp op) {
    return op == ExprOp::MATMUL || op == ExprOp::GEMM;
}

static bool isZeroScalarNode(const PhysicalExpression& expr, uint32_t node_idx) {
    if (node_idx >= expr.nodes.size()) {
        return false;
    }
    const ExprNode& node = expr.nodes[node_idx];
    return node.op == ExprOp::SCALAR_FP && node.scalar_fp == 0.0;
}

static bool canFuseMatmulActivationEpilogue(const PhysicalExpression& expr,
                                            uint32_t node_idx,
                                            const std::vector<std::vector<uint64_t>>& node_dims) {
    if (node_idx >= expr.nodes.size() || node_idx >= node_dims.size()) {
        return false;
    }
    const ExprNode& node = expr.nodes[node_idx];
    if (!isMatmulOp(node.op) || node.matmul_epilogue != MatmulEpilogue::Default ||
        node.matmul_backward_epilogue != MatmulBackwardEpilogue::Default || node.matmul_epilogue_aux != UINT32_MAX ||
        node_dims[node_idx].size() != 2) {
        return false;
    }

    // The current cuBLASLt epilogue wrapper presents Thor row-major matrices through
    // a column-major transposed view. Keep activation epilogue fusion on the same
    // well-tested non-transposed projection path as the existing bias epilogue.
    if (node.transpose_lhs || node.transpose_rhs || node.transpose_aux) {
        return false;
    }
    return true;
}

static bool sameSubexpressionForMatmulEpilogue(const PhysicalExpression& expr,
                                               uint32_t a_idx,
                                               uint32_t b_idx,
                                               uint32_t depth = 0) {
    if (a_idx == b_idx) {
        return true;
    }
    if (depth > 64 || a_idx >= expr.nodes.size() || b_idx >= expr.nodes.size()) {
        return false;
    }

    const ExprNode& a = expr.nodes[a_idx];
    const ExprNode& b = expr.nodes[b_idx];
    if (a.op != b.op) {
        return false;
    }

    if (a.output_dtype != b.output_dtype || a.compute_dtype != b.compute_dtype || a.input_tensor_dtype != b.input_tensor_dtype ||
        a.backward_output_dtype != b.backward_output_dtype || a.backward_compute_dtype != b.backward_compute_dtype) {
        return false;
    }

    switch (a.op) {
        case ExprOp::INPUT:
        case ExprOp::RUNTIME_SCALAR:
        case ExprOp::TENSOR_RUNTIME_SCALAR:
            return a.input_slot == b.input_slot;
        case ExprOp::SCALAR_FP:
            return a.scalar_fp == b.scalar_fp;
        case ExprOp::MATMUL:
            return a.transpose_lhs == b.transpose_lhs && a.transpose_rhs == b.transpose_rhs && a.matmul_epilogue == b.matmul_epilogue &&
                   a.matmul_backward_epilogue == b.matmul_backward_epilogue &&
                   (a.matmul_epilogue_aux == b.matmul_epilogue_aux ||
                    sameSubexpressionForMatmulEpilogue(expr, a.matmul_epilogue_aux, b.matmul_epilogue_aux, depth + 1)) &&
                   sameSubexpressionForMatmulEpilogue(expr, a.lhs, b.lhs, depth + 1) &&
                   sameSubexpressionForMatmulEpilogue(expr, a.rhs, b.rhs, depth + 1);
        case ExprOp::GEMM:
            return a.transpose_lhs == b.transpose_lhs && a.transpose_rhs == b.transpose_rhs && a.transpose_aux == b.transpose_aux &&
                   a.alpha_fp == b.alpha_fp && a.beta_fp == b.beta_fp && a.matmul_epilogue == b.matmul_epilogue &&
                   a.matmul_backward_epilogue == b.matmul_backward_epilogue &&
                   (a.matmul_epilogue_aux == b.matmul_epilogue_aux ||
                    sameSubexpressionForMatmulEpilogue(expr, a.matmul_epilogue_aux, b.matmul_epilogue_aux, depth + 1)) &&
                   sameSubexpressionForMatmulEpilogue(expr, a.lhs, b.lhs, depth + 1) &&
                   sameSubexpressionForMatmulEpilogue(expr, a.rhs, b.rhs, depth + 1) &&
                   sameSubexpressionForMatmulEpilogue(expr, a.aux, b.aux, depth + 1) &&
                   (a.alpha_node == b.alpha_node || sameSubexpressionForMatmulEpilogue(expr, a.alpha_node, b.alpha_node, depth + 1)) &&
                   (a.beta_node == b.beta_node || sameSubexpressionForMatmulEpilogue(expr, a.beta_node, b.beta_node, depth + 1));
        case ExprOp::NORMCDF:
            return sameSubexpressionForMatmulEpilogue(expr, a.lhs, b.lhs, depth + 1);
        case ExprOp::TAKE_ALONG_AXIS:
            return a.reduction_axes == b.reduction_axes &&
                   sameSubexpressionForMatmulEpilogue(expr, a.lhs, b.lhs, depth + 1) &&
                   sameSubexpressionForMatmulEpilogue(expr, a.rhs, b.rhs, depth + 1);
        case ExprOp::ADD:
        case ExprOp::SUB:
        case ExprOp::MUL:
        case ExprOp::DIV:
        case ExprOp::EQUAL:
        case ExprOp::NOT_EQUAL:
        case ExprOp::LESS:
        case ExprOp::LESS_EQUAL:
        case ExprOp::GREATER:
        case ExprOp::GREATER_EQUAL:
        case ExprOp::LOGICAL_AND:
        case ExprOp::LOGICAL_OR:
        case ExprOp::MIN:
        case ExprOp::MAX:
            return sameSubexpressionForMatmulEpilogue(expr, a.lhs, b.lhs, depth + 1) &&
                   sameSubexpressionForMatmulEpilogue(expr, a.rhs, b.rhs, depth + 1);
        case ExprOp::WHERE:
            return sameSubexpressionForMatmulEpilogue(expr, a.lhs, b.lhs, depth + 1) &&
                   sameSubexpressionForMatmulEpilogue(expr, a.rhs, b.rhs, depth + 1) &&
                   sameSubexpressionForMatmulEpilogue(expr, a.aux, b.aux, depth + 1);
        default:
            return false;
    }
}

static bool tryBuildMatmulActivationEpiloguePattern(const PhysicalExpression& expr,
                                                    uint32_t node_idx,
                                                    const std::vector<std::vector<uint64_t>>& node_dims,
                                                    MatmulActivationEpiloguePattern& out) {
    if (node_idx >= expr.nodes.size() || node_idx >= node_dims.size()) {
        return false;
    }

    const ExprNode& node = expr.nodes[node_idx];
    if (node.op == ExprOp::MAX) {
        uint32_t source_idx = UINT32_MAX;
        if (isZeroScalarNode(expr, node.lhs)) {
            source_idx = node.rhs;
        } else if (isZeroScalarNode(expr, node.rhs)) {
            source_idx = node.lhs;
        } else {
            return false;
        }

        if (!canFuseMatmulActivationEpilogue(expr, source_idx, node_dims)) {
            return false;
        }
        out.source_idx = source_idx;
        out.epilogue = MatmulEpilogue::Relu;
        return true;
    }

    if (node.op != ExprOp::MUL) {
        return false;
    }

    auto attempt_gelu = [&](uint32_t source_idx, uint32_t normcdf_idx) -> bool {
        if (normcdf_idx >= expr.nodes.size() || expr.nodes[normcdf_idx].op != ExprOp::NORMCDF) {
            return false;
        }
        if (!canFuseMatmulActivationEpilogue(expr, source_idx, node_dims)) {
            return false;
        }
        if (!sameSubexpressionForMatmulEpilogue(expr, source_idx, expr.nodes[normcdf_idx].lhs)) {
            return false;
        }
        out.source_idx = source_idx;
        out.epilogue = MatmulEpilogue::Gelu;
        return true;
    };

    return attempt_gelu(node.lhs, node.rhs) || attempt_gelu(node.rhs, node.lhs);
}

static void rewriteAsMatmulActivationEpilogue(PhysicalExpression& expr,
                                              uint32_t node_idx,
                                              const MatmulActivationEpiloguePattern& pattern) {
    if (node_idx >= expr.nodes.size() || pattern.source_idx >= expr.nodes.size()) {
        throw std::runtime_error("Matmul activation epilogue rewrite received an out-of-range node index.");
    }

    const ExprNode activation_node = expr.nodes[node_idx];
    ExprNode replacement = expr.nodes[pattern.source_idx];
    replacement.matmul_epilogue = pattern.epilogue;

    if (activation_node.output_dtype.has_value()) {
        replacement.output_dtype = activation_node.output_dtype;
    }
    if (activation_node.compute_dtype.has_value()) {
        replacement.compute_dtype = activation_node.compute_dtype;
    }
    if (activation_node.backward_output_dtype.has_value()) {
        replacement.backward_output_dtype = activation_node.backward_output_dtype;
    }
    if (activation_node.backward_compute_dtype.has_value()) {
        replacement.backward_compute_dtype = activation_node.backward_compute_dtype;
    }

    expr.nodes[node_idx] = std::move(replacement);
}

static bool tryMatchScaledMatmulPattern(const PhysicalExpression& expr, uint32_t node_idx, ScaledMatmulPattern& out) {
    if (node_idx >= expr.nodes.size()) {
        return false;
    }

    const ExprNode& node = expr.nodes[node_idx];
    if (node.op == ExprOp::MATMUL) {
        if (node.matmul_epilogue != MatmulEpilogue::Default) {
            return false;
        }
        out.matmul_node_idx = node_idx;
        out.lhs_idx = node.lhs;
        out.rhs_idx = node.rhs;
        out.transpose_lhs = node.transpose_lhs;
        out.transpose_rhs = node.transpose_rhs;
        out.alpha_scale = 1.0;
        out.alpha_node = UINT32_MAX;
        out.compute_dtype = node.compute_dtype;
        out.backward_compute_dtype = node.backward_compute_dtype;
        return true;
    }

    if (node.op != ExprOp::MUL) {
        return false;
    }

    double scale_fp = 0.0;
    uint32_t scale_node = UINT32_MAX;
    const ExprNode* matmul = nullptr;
    uint32_t matmul_idx = UINT32_MAX;

    if (tryGetLowerableScaleNode(expr, node.lhs, scale_fp, scale_node) && node.rhs < expr.nodes.size() &&
        expr.nodes[node.rhs].op == ExprOp::MATMUL) {
        matmul = &expr.nodes[node.rhs];
        if (matmul->matmul_epilogue != MatmulEpilogue::Default) {
            return false;
        }
        matmul_idx = node.rhs;
    } else if (tryGetLowerableScaleNode(expr, node.rhs, scale_fp, scale_node) && node.lhs < expr.nodes.size() &&
               expr.nodes[node.lhs].op == ExprOp::MATMUL) {
        matmul = &expr.nodes[node.lhs];
        if (matmul->matmul_epilogue != MatmulEpilogue::Default) {
            return false;
        }
        matmul_idx = node.lhs;
    } else {
        return false;
    }

    out.matmul_node_idx = matmul_idx;
    out.lhs_idx = matmul->lhs;
    out.rhs_idx = matmul->rhs;
    out.transpose_lhs = matmul->transpose_lhs;
    out.transpose_rhs = matmul->transpose_rhs;
    out.alpha_scale = scale_fp;
    out.alpha_node = scale_node;
    out.compute_dtype = matmul->compute_dtype;
    out.backward_compute_dtype = matmul->backward_compute_dtype;
    return true;
}

static bool tryMatchScaledAddendPattern(const PhysicalExpression& expr,
                                        uint32_t node_idx,
                                        const std::vector<std::vector<uint64_t>>& node_dims,
                                        const std::vector<uint64_t>& expected_dims,
                                        ScaledAddendPattern& out) {
    if (node_idx >= expr.nodes.size() || node_idx >= node_dims.size()) {
        return false;
    }

    const ExprNode& node = expr.nodes[node_idx];

    auto is_compatible_unscaled_addend = [&](uint32_t candidate_idx) -> bool {
        if (candidate_idx >= node_dims.size() || node_dims[candidate_idx].empty()) {
            return false;
        }
        const std::vector<uint64_t>& dims = node_dims[candidate_idx];
        if (dims == expected_dims) {
            out.is_bias_vector = false;
            return true;
        }
        if (expected_dims.size() == 2 && dims.size() == 1 && dims[0] == expected_dims[1]) {
            out.is_bias_vector = true;
            return true;
        }
        return false;
    };

    // First, prefer matching an explicitly scaled addend like beta * C or C * beta.
    // Otherwise a MUL node whose overall dims match would be incorrectly treated as
    // an unscaled addend with beta = 1.
    if (node.op == ExprOp::MUL) {
        double scale_fp = 0.0;
        uint32_t scale_node = UINT32_MAX;

        if (tryGetLowerableScaleNode(expr, node.lhs, scale_fp, scale_node)) {
            if (is_compatible_unscaled_addend(node.rhs)) {
                out.node_idx = node.rhs;
                out.beta_scale = scale_fp;
                out.beta_node = scale_node;
                return true;
            }
        }

        if (tryGetLowerableScaleNode(expr, node.rhs, scale_fp, scale_node)) {
            if (is_compatible_unscaled_addend(node.lhs)) {
                out.node_idx = node.lhs;
                out.beta_scale = scale_fp;
                out.beta_node = scale_node;
                return true;
            }
        }
    }

    // Fallback: plain unscaled matrix addend C or rank-1 bias vector.
    if (is_compatible_unscaled_addend(node_idx)) {
        out.node_idx = node_idx;
        out.beta_scale = 1.0;
        out.beta_node = UINT32_MAX;
        return true;
    }

    return false;
}

static bool tryBuildGemmLoweringPattern(const PhysicalExpression& expr,
                                        uint32_t node_idx,
                                        const std::vector<std::vector<uint64_t>>& node_dims,
                                        GemmLoweringPattern& out) {
    if (node_idx >= expr.nodes.size() || node_idx >= node_dims.size()) {
        return false;
    }

    const ExprNode& node = expr.nodes[node_idx];
    if (node.op != ExprOp::ADD && node.op != ExprOp::SUB) {
        return false;
    }

    auto attempt = [&](uint32_t matmul_candidate, uint32_t addend_candidate, double matmul_sign, double addend_sign) -> bool {
        ScaledMatmulPattern matmul;
        if (!tryMatchScaledMatmulPattern(expr, matmul_candidate, matmul)) {
            return false;
        }
        if (matmul.matmul_node_idx >= node_dims.size()) {
            return false;
        }
        const std::vector<uint64_t>& matmul_dims = node_dims[matmul.matmul_node_idx];
        if (matmul_dims.size() != 2) {
            return false;
        }

        ScaledAddendPattern addend;
        if (!tryMatchScaledAddendPattern(expr, addend_candidate, node_dims, matmul_dims, addend)) {
            return false;
        }
        if (addend.is_bias_vector) {
            // cuBLASLt bias epilogues add the bias vector directly.  Do not lower scaled
            // or subtracted bias-vector expressions into this pattern until an explicit
            // pre-scaled-bias materialization policy exists.
            if (matmul_sign != 1.0 || addend_sign != 1.0 || addend.beta_scale != 1.0 || addend.beta_node != UINT32_MAX) {
                return false;
            }
            // The current cuBLASLt bias-epilogue wrapper handles the hot row-major projection
            // case.  Leave transposed matmul+bias expressions on the generic path until their
            // exact epilogue contract is added deliberately.
            if (matmul.transpose_lhs || matmul.transpose_rhs) {
                return false;
            }
            // Do not silently turn ``matmul(..., output_dtype=X) + bias_dtype=Y`` into a
            // different materialized output dtype through generic ADD promotion.  Projection
            // bias epilogues are a no-extra-kernel path only when the public add result keeps
            // the matmul output dtype.
            const ExprNode& matmul_node = expr.nodes[matmul.matmul_node_idx];
            if (matmul_node.output_dtype.has_value() && node.output_dtype.has_value() &&
                matmul_node.output_dtype.value() != node.output_dtype.value()) {
                throw std::runtime_error(
                    "GEMM bias epilogue requires the bias tensor dtype to match the matmul output dtype; Thor will not insert an implicit conversion for a cuBLASLt bias epilogue.");
            }
        }

        out.lhs_idx = matmul.lhs_idx;
        out.rhs_idx = matmul.rhs_idx;
        out.addend_idx = addend.node_idx;
        out.is_bias_epilogue = addend.is_bias_vector;
        out.matmul_output_dtype = expr.nodes[matmul.matmul_node_idx].output_dtype;
        out.transpose_lhs = matmul.transpose_lhs;
        out.transpose_rhs = matmul.transpose_rhs;
        out.alpha_scale = matmul_sign * matmul.alpha_scale;
        out.beta_scale = addend_sign * addend.beta_scale;
        out.alpha_node = matmul.alpha_node;
        out.beta_node = addend.beta_node;
        out.inherited_compute_dtype = matmul.compute_dtype;
        out.inherited_backward_compute_dtype = matmul.backward_compute_dtype;
        return true;
    };

    if (node.op == ExprOp::ADD) {
        if (attempt(node.lhs, node.rhs, 1.0, 1.0)) {
            return true;
        }
        if (attempt(node.rhs, node.lhs, 1.0, 1.0)) {
            return true;
        }
        return false;
    }

    if (attempt(node.lhs, node.rhs, 1.0, -1.0)) {
        return true;
    }
    if (attempt(node.rhs, node.lhs, -1.0, 1.0)) {
        return true;
    }
    return false;
}

static void optimizeExpressionGemmPatternsInPlace(PhysicalExpression& expr,
                                                  const std::unordered_map<uint32_t, RuntimeInputValue>& root_values) {
    if (expr.nodes.empty() || !expressionHasPotentialGemmLoweringPattern(expr)) {
        return;
    }

    std::vector<std::vector<uint64_t>> node_dims = inferExpressionNodeDimsForOptimization(expr, root_values);

    for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
        GemmLoweringPattern pattern;
        if (!tryBuildGemmLoweringPattern(expr, node_idx, node_dims, pattern)) {
            continue;
        }

        ExprNode& node = expr.nodes[node_idx];
        node.op = ExprOp::GEMM;
        node.lhs = pattern.lhs_idx;
        node.rhs = pattern.rhs_idx;
        node.aux = pattern.addend_idx;
        node.alpha_fp = pattern.alpha_scale;
        node.beta_fp = pattern.beta_scale;
        node.alpha_node = pattern.alpha_node;
        node.beta_node = pattern.beta_node;
        node.transpose_lhs = pattern.transpose_lhs;
        node.transpose_rhs = pattern.transpose_rhs;
        node.transpose_aux = false;
        if (pattern.is_bias_epilogue && pattern.matmul_output_dtype.has_value()) {
            // A rank-1 bias epilogue is a no-extra-kernel replacement for
            // ``matmul(..., output_dtype=X) + bias``.  Preserve the matmul's
            // explicit materialized dtype instead of letting generic ADD dtype
            // promotion turn the public result into the bias dtype.  Runtime
            // stamping will then fail loudly if the provided bias tensor does
            // not already match this dtype.
            node.output_dtype = pattern.matmul_output_dtype.value();
            if (!node.backward_output_dtype.has_value()) {
                node.backward_output_dtype = pattern.matmul_output_dtype.value();
            }
        }
        node.reduction_axes.clear();
        node.squeeze_axes.clear();
        node.unsqueeze_axes.clear();
        node.fill_dims.clear();
        if (!node.compute_dtype.has_value() && pattern.inherited_compute_dtype.has_value()) {
            node.compute_dtype = pattern.inherited_compute_dtype.value();
        }
        if (!node.backward_compute_dtype.has_value() && pattern.inherited_backward_compute_dtype.has_value()) {
            node.backward_compute_dtype = pattern.inherited_backward_compute_dtype.value();
        }
    }

    for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
        MatmulActivationEpiloguePattern pattern;
        if (!tryBuildMatmulActivationEpiloguePattern(expr, node_idx, node_dims, pattern)) {
            continue;
        }
        rewriteAsMatmulActivationEpilogue(expr, node_idx, pattern);
        node_dims[node_idx] = node_dims[pattern.source_idx];
    }
}

std::unordered_map<std::string, std::vector<uint64_t>> FusedEquation::makeSingleOutputRequestedShapeMap(
    const std::vector<uint64_t>& requestedOutputShape) const {
    std::unordered_map<std::string, std::vector<uint64_t>> requested;

    if (requestedOutputShape.empty()) {
        return requested;
    }

    const auto outputNames = getOutputNames();
    if (outputNames.size() != 1) {
        throw std::runtime_error(
            "Single-output requested-shape stamp overload called on an equation that does not have exactly one output.");
    }

    requested.emplace(outputNames[0], requestedOutputShape);
    return requested;
}

static Tensor allocateCudnnSoftmaxInputAdapterIfNeeded(const Tensor& input,
                                                        DataType expected_input_dtype,
                                                        ExprOp op) {
    if (input.getDataType() == expected_input_dtype) {
        return input;
    }

    if (toSupportedInputDType(op, input.getDataType()) != expected_input_dtype) {
        throw std::runtime_error("Runtime input dtype does not match the compiled cuDNN softmax dtype policy.");
    }

    // Do not enqueue the conversion while stamping. A staged input may be produced by an
    // earlier execution stage and therefore does not contain its runtime value yet. The
    // stamped cuDNN softmax operation owns this adapter tensor and refreshes it immediately before
    // each execution on the caller's run stream.
    TensorDescriptor castDescriptor(expected_input_dtype, input.getDimensions());
    return Tensor(input.getPlacement(), castDescriptor);
}

static bool dimsResolveToSingleElement(const std::vector<uint64_t>& dims) {
    if (dims.empty()) {
        return true;
    }
    uint64_t numel = 1;
    for (uint64_t d : dims) {
        numel *= d;
    }
    return numel == 1;
}

static std::vector<uint64_t> resolveMatmulOutputDimsFromInputs(const CompiledMatmul& compiled_stage,
                                                               const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    const size_t min_expected_inputs = compiled_stage.op == ExprOp::MATMUL ? 2u : 3u;
    if (stage_input_dims.size() < min_expected_inputs) {
        throw std::runtime_error("Matmul/gemm stage expected at least " + std::to_string(min_expected_inputs) + " input shapes.");
    }

    const auto& a_dims = stage_input_dims[0];
    const auto& b_dims = stage_input_dims[1];
    if (a_dims.size() != 2 || b_dims.size() != 2) {
        throw std::runtime_error("Matmul/gemm stage currently only supports rank-2 tensors.");
    }

    const uint64_t a_rows = compiled_stage.transpose_lhs ? a_dims[1] : a_dims[0];
    const uint64_t a_cols = compiled_stage.transpose_lhs ? a_dims[0] : a_dims[1];
    const uint64_t b_rows = compiled_stage.transpose_rhs ? b_dims[1] : b_dims[0];
    const uint64_t b_cols = compiled_stage.transpose_rhs ? b_dims[0] : b_dims[1];

    if (a_cols != b_rows) {
        throw std::runtime_error("Matmul/gemm stage has incompatible matrix dimensions.");
    }

    std::vector<uint64_t> out_dims{a_rows, b_cols};
    if (compiled_stage.op == ExprOp::GEMM) {
        const auto& c_dims = stage_input_dims[2];
        if (c_dims.size() == 1) {
            if (compiled_stage.transpose_aux || c_dims[0] != out_dims[1]) {
                throw std::runtime_error("GEMM bias epilogue addend must have shape [output_columns].");
            }
        } else if (c_dims.size() == 2) {
            std::vector<uint64_t> expected_c = compiled_stage.transpose_aux ? std::vector<uint64_t>{out_dims[1], out_dims[0]} : out_dims;
            if (c_dims != expected_c) {
                throw std::runtime_error("GEMM addend tensor dimensions are incompatible with the matmul output.");
            }
        } else {
            throw std::runtime_error("GEMM addend currently must be rank-2 or a rank-1 bias epilogue vector.");
        }
    }

    auto validate_scalar_input_dims = [&](uint32_t input_slot, const char* label) {
        if (input_slot == UINT32_MAX) {
            return;
        }
        if (input_slot >= stage_input_dims.size()) {
            throw std::runtime_error(std::string("Matmul/gemm ") + label + " scale input slot is out of range.");
        }
        if (!dimsResolveToSingleElement(stage_input_dims[input_slot])) {
            throw std::runtime_error(std::string("Matmul/gemm ") + label + " scale expression must resolve to a single element.");
        }
    };

    validate_scalar_input_dims(compiled_stage.alpha_input_slot, "alpha");
    validate_scalar_input_dims(compiled_stage.beta_input_slot, "beta");
    if (compiled_stage.backward_epilogue != MatmulBackwardEpilogue::Default) {
        if (compiled_stage.epilogue_aux_input_slot == UINT32_MAX || compiled_stage.epilogue_aux_input_slot >= stage_input_dims.size()) {
            throw std::runtime_error("Matmul/gemm backward epilogue aux input slot is out of range.");
        }
        if (stage_input_dims[compiled_stage.epilogue_aux_input_slot] != out_dims) {
            throw std::runtime_error("Matmul/gemm backward epilogue aux dimensions must match the matrix output dimensions.");
        }
    }
    return out_dims;
}

static uint64_t ceilDivPositive(uint64_t numerator, uint64_t denominator) {
    if (numerator == 0 || denominator == 0) {
        throw std::runtime_error("ceilDivPositive requires positive operands.");
    }
    return (numerator + denominator - 1) / denominator;
}

static std::vector<uint64_t> resolveAttentionOutputDimsFromInputs(const CompiledAttention& compiled_stage,
                                                                  const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    if (stage_input_dims.size() < 3) {
        throw std::runtime_error("Attention stage expected q/k/v input shapes.");
    }

    const AttentionTensorLogicalDims q_dims = logicalAttentionDims(stage_input_dims.at(0), compiled_stage.q_layout, "q");
    const AttentionTensorLogicalDims k_dims = logicalAttentionDims(stage_input_dims.at(1), compiled_stage.k_layout, "k");
    const AttentionTensorLogicalDims v_dims = logicalAttentionDims(stage_input_dims.at(2), compiled_stage.v_layout, "v");

    const uint64_t batch = q_dims.batch;
    const uint64_t query_heads = q_dims.heads;
    const uint64_t query_len = q_dims.sequence_length;
    const uint64_t qk_dim = q_dims.head_dim;
    const uint64_t kv_batch_or_blocks = k_dims.batch;
    const uint64_t kv_heads = k_dims.heads;
    const uint64_t kv_len_or_block = k_dims.sequence_length;
    const uint64_t k_dim = k_dims.head_dim;
    const uint64_t v_batch_or_blocks = v_dims.batch;
    const uint64_t v_heads = v_dims.heads;
    const uint64_t v_len_or_block = v_dims.sequence_length;
    const uint64_t v_dim = v_dims.head_dim;

    if (!compiled_stage.use_paged_kv_cache && (batch != kv_batch_or_blocks || batch != v_batch_or_blocks)) {
        throw std::runtime_error("Attention q/k/v batch dimensions must match.");
    }
    if (kv_heads != v_heads) {
        throw std::runtime_error("Attention k/v head counts must match.");
    }
    if (query_heads % kv_heads != 0) {
        throw std::runtime_error("Attention query heads must be an integer multiple of key/value heads for MHA/MQA/GQA.");
    }
    if (!compiled_stage.use_paged_kv_cache && kv_len_or_block != v_len_or_block) {
        throw std::runtime_error("Attention k/v sequence lengths must match.");
    }
    if (qk_dim != k_dim) {
        throw std::runtime_error("Attention q/k head dimensions must match.");
    }
    if (kv_len_or_block == 0 || v_len_or_block == 0 || query_len == 0 || qk_dim == 0 || v_dim == 0) {
        throw std::runtime_error("Attention q/k/v dimensions must be non-zero.");
    }
    if (compiled_stage.use_paged_kv_cache && compiled_stage.paged_kv_max_sequence_length <= 0) {
        throw std::runtime_error("Attention paged KV max sequence length must be positive.");
    }
    const uint64_t kv_len =
        compiled_stage.use_paged_kv_cache ? static_cast<uint64_t>(compiled_stage.paged_kv_max_sequence_length) : kv_len_or_block;
    size_t next_optional_idx = 3;
    if (compiled_stage.use_bias) {
        if (stage_input_dims.size() <= next_optional_idx) {
            throw std::runtime_error("Attention stage with additive bias expected a bias input shape.");
        }
        if (!isAllowedAttentionBiasDims(stage_input_dims.at(next_optional_idx), batch, query_heads, query_len, kv_len)) {
            throw std::runtime_error("Attention additive bias must have shape " +
                                     attentionBiasShapeDescription(batch, query_heads, query_len, kv_len) + ".");
        }
        ++next_optional_idx;
    }
    if (compiled_stage.use_padding_mask) {
        if (stage_input_dims.size() <= next_optional_idx + 1) {
            throw std::runtime_error("Attention stage with padding mask expected q_seq_len and kv_seq_len input shapes.");
        }
        const std::vector<uint64_t> expected_seq_dims{batch};
        if (stage_input_dims.at(next_optional_idx) != expected_seq_dims ||
            stage_input_dims.at(next_optional_idx + 1) != expected_seq_dims) {
            throw std::runtime_error("Attention padding-mask sequence lengths must have shape [B].");
        }
        next_optional_idx += 2;
    }
    if (compiled_stage.use_ragged_offsets) {
        if (qk_dim != v_dim) {
            throw std::runtime_error(
                "Attention ragged offsets require value head_dim to match query/key head_dim because Thor uses shared Q/O and K/V "
                "element offsets.");
        }
        if (stage_input_dims.size() <= next_optional_idx + 1) {
            throw std::runtime_error("Attention stage with ragged offsets expected q_ragged_offsets and kv_ragged_offsets input shapes.");
        }
        const std::vector<uint64_t> expected_offset_dims{batch + 1};
        if (stage_input_dims.at(next_optional_idx) != expected_offset_dims) {
            throw std::runtime_error("Attention ragged q_offsets shape must be [B + 1].");
        }
        if (stage_input_dims.at(next_optional_idx + 1) != expected_offset_dims) {
            throw std::runtime_error("Attention ragged kv_offsets shape must be [B + 1].");
        }
        next_optional_idx += 2;
    }
    if (compiled_stage.use_paged_kv_cache) {
        if (!compiled_stage.use_padding_mask) {
            throw std::runtime_error("Attention paged KV cache requires q_seq_len and kv_seq_len inputs.");
        }
        if (stage_input_dims.size() <= next_optional_idx + 1) {
            throw std::runtime_error("Attention stage with paged KV cache expected page_table_k and page_table_v input shapes.");
        }
        const std::vector<uint64_t> expected_page_table_k_dims{batch, 1, ceilDivPositive(kv_len, kv_len_or_block), 1};
        const std::vector<uint64_t> expected_page_table_v_dims{batch, 1, ceilDivPositive(kv_len, v_len_or_block), 1};
        if (stage_input_dims.at(next_optional_idx) != expected_page_table_k_dims) {
            throw std::runtime_error("Attention paged KV page_table_k shape must be [B,1,ceil(Skv/block_k),1].");
        }
        if (stage_input_dims.at(next_optional_idx + 1) != expected_page_table_v_dims) {
            throw std::runtime_error("Attention paged KV page_table_v shape must be [B,1,ceil(Skv/block_v),1].");
        }
        next_optional_idx += 2;
    }

    return thorAttentionDims(compiled_stage.o_layout, batch, query_heads, query_len, v_dim);
}


static CompiledAttention makeForwardAttentionView(const CompiledAttentionBackward& backward,
                                                  DataType output_dtype,
                                                  std::string debug_suffix = "") {
    CompiledAttention forward;
    forward.q_layout = backward.q_layout;
    forward.k_layout = backward.k_layout;
    forward.v_layout = backward.v_layout;
    forward.o_layout = backward.o_layout;
    forward.mask_kind = backward.mask_kind;
    forward.diagonal_left_bound = backward.diagonal_left_bound;
    forward.diagonal_right_bound = backward.diagonal_right_bound;
    forward.attention_scale = backward.attention_scale;
    forward.use_alibi_mask = backward.use_alibi_mask;
    forward.use_bias = backward.use_bias;
    forward.use_padding_mask = backward.use_padding_mask;
    forward.use_ragged_offsets = backward.use_ragged_offsets;
    forward.use_paged_kv_cache = backward.use_paged_kv_cache;
    forward.paged_kv_max_sequence_length = backward.paged_kv_max_sequence_length;
    forward.dropout_probability = backward.dropout_probability;
    forward.compute_dtype = backward.compute_dtype;
    forward.output_dtype = output_dtype;
    forward.debug_name = debug_suffix.empty() ? backward.debug_name : backward.debug_name + debug_suffix;
    return forward;
}

static std::vector<uint64_t> resolveAttentionBackwardOutputDimsFromInputs(const CompiledAttentionBackward& compiled_stage,
                                                                          const std::vector<std::vector<uint64_t>>& stage_input_dims,
                                                                          ExprOp output_op) {
    if (stage_input_dims.size() < 4) {
        throw std::runtime_error("Attention-backward stage expected q/k/v/dO input shapes.");
    }

    std::vector<std::vector<uint64_t>> forward_input_dims{stage_input_dims.at(0), stage_input_dims.at(1), stage_input_dims.at(2)};
    if (compiled_stage.use_bias && stage_input_dims.size() >= 5) {
        forward_input_dims.push_back(stage_input_dims.at(4));
    }
    size_t next_optional_idx = 4 + (compiled_stage.use_bias ? 1 : 0);
    if (compiled_stage.use_padding_mask) {
        if (stage_input_dims.size() <= next_optional_idx + 1) {
            throw std::runtime_error("Attention-backward stage with padding mask expected q_seq_len and kv_seq_len input shapes.");
        }
        forward_input_dims.push_back(stage_input_dims.at(next_optional_idx));
        forward_input_dims.push_back(stage_input_dims.at(next_optional_idx + 1));
        next_optional_idx += 2;
    }
    if (compiled_stage.use_ragged_offsets) {
        if (stage_input_dims.size() <= next_optional_idx + 1) {
            throw std::runtime_error(
                "Attention-backward stage with ragged offsets expected q_ragged_offsets and kv_ragged_offsets input shapes.");
        }
        forward_input_dims.push_back(stage_input_dims.at(next_optional_idx));
        forward_input_dims.push_back(stage_input_dims.at(next_optional_idx + 1));
        next_optional_idx += 2;
    }
    const std::vector<uint64_t> forward_out_dims =
        resolveAttentionOutputDimsFromInputs(makeForwardAttentionView(compiled_stage, compiled_stage.dQ_dtype), forward_input_dims);

    const auto& dO_dims = stage_input_dims.at(3);
    if (dO_dims != forward_out_dims) {
        throw std::runtime_error("Attention-backward dO shape must match the corresponding forward attention output shape.");
    }

    switch (output_op) {
        case ExprOp::ATTENTION_BACKWARD_Q:
            return stage_input_dims.at(0);
        case ExprOp::ATTENTION_BACKWARD_K:
            return stage_input_dims.at(1);
        case ExprOp::ATTENTION_BACKWARD_V:
            return stage_input_dims.at(2);
        case ExprOp::ATTENTION_BACKWARD_BIAS: {
            if (!compiled_stage.use_bias) {
                throw std::runtime_error("Attention-backward dBias output requested for an unbiased attention stage.");
            }
            const AttentionTensorLogicalDims qLogical = logicalAttentionDims(stage_input_dims.at(0), compiled_stage.q_layout, "q");
            const AttentionTensorLogicalDims kLogical = logicalAttentionDims(stage_input_dims.at(1), compiled_stage.k_layout, "k");
            return {qLogical.batch, qLogical.heads, qLogical.sequence_length, kLogical.sequence_length};
        }
        default:
            throw std::runtime_error("Attention-backward output shape requested for non-attention-backward op.");
    }
}

static std::vector<uint64_t> resolveConvolutionOutputDimsFromInputs(const CompiledConvolution& compiled_stage,
                                                                    const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    if (stage_input_dims.size() < 2) {
        throw std::runtime_error("Convolution stage expected at least two input shapes.");
    }

    ExprNode node{};
    node.op = compiled_stage.is_3d ? ExprOp::CONV3D : ExprOp::CONV2D;
    node.conv_stride_d = compiled_stage.stride_d;
    node.conv_stride_h = compiled_stage.stride_h;
    node.conv_stride_w = compiled_stage.stride_w;
    node.conv_pad_d = compiled_stage.pad_d;
    node.conv_pad_h = compiled_stage.pad_h;
    node.conv_pad_w = compiled_stage.pad_w;
    return inferExpressionConvolutionOutputDims(node, stage_input_dims[0], stage_input_dims[1]);
}

static std::vector<uint64_t> resolveConvolutionBackwardOutputDimsFromInputs(const CompiledConvolutionBackward& compiled_stage,
                                                                            const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    if (stage_input_dims.size() < 2) {
        throw std::runtime_error("Convolution backward stage expected at least two input shapes.");
    }

    ExprNode node{};
    node.op = compiled_stage.op;
    node.conv_stride_d = compiled_stage.stride_d;
    node.conv_stride_h = compiled_stage.stride_h;
    node.conv_stride_w = compiled_stage.stride_w;
    node.conv_pad_d = compiled_stage.pad_d;
    node.conv_pad_h = compiled_stage.pad_h;
    node.conv_pad_w = compiled_stage.pad_w;
    node.fill_dims = compiled_stage.explicit_output_dims;
    if (compiled_stage.op == ExprOp::CONV2D_BACKWARD_DATA || compiled_stage.op == ExprOp::CONV3D_BACKWARD_DATA) {
        return inferExpressionConvolutionBackwardDataOutputDims(node, stage_input_dims[0], stage_input_dims[1]);
    }
    if (compiled_stage.op == ExprOp::CONV2D_BACKWARD_FILTER || compiled_stage.op == ExprOp::CONV3D_BACKWARD_FILTER) {
        return inferExpressionConvolutionBackwardFilterOutputDims(node, stage_input_dims[0], stage_input_dims[1]);
    }
    throw std::runtime_error("resolveConvolutionBackwardOutputDimsFromInputs received unsupported op.");
}

static void collectReachableLocalNodes(const PhysicalExpression& expr, uint32_t node_idx, std::unordered_set<uint32_t>& nodes);

static void mergeParameterFanOverride(FusedEquation::ParameterFanOverrideMap& result, const ParameterFanOverride& hint) {
    auto [it, inserted] = result.emplace(hint.input_name, hint);
    if (!inserted) {
        it->second.input_name = hint.input_name;
        it->second.fan_in = std::max(it->second.fan_in, hint.fan_in);
        it->second.fan_out = std::max(it->second.fan_out, hint.fan_out);
    }
}

static uint64_t computeDimsNumel(const std::vector<uint64_t>& dims) {
    uint64_t numel = 1;
    for (uint64_t dim : dims) {
        numel *= dim;
    }
    return numel;
}

static uint64_t computeNumelPerExample(const std::vector<uint64_t>& dims) {
    uint64_t numel = 1;
    for (size_t i = 1; i < dims.size(); ++i) {
        numel *= dims[i];
    }
    return numel;
}

static void addFusedKernelParameterFanOverrides(const CompiledExecutionStage& stage,
                                                const std::vector<std::vector<uint64_t>>& stage_input_dims,
                                                const std::vector<std::vector<uint64_t>>& stage_output_dims,
                                                const std::unordered_map<uint32_t, std::string>& root_input_name_by_slot,
                                                const std::unordered_set<std::string>& parameter_names,
                                                FusedEquation::ParameterFanOverrideMap& result) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        return;
    }
    if (stage_input_dims.size() != stage.input_value_ids.size()) {
        throw std::runtime_error("Fused-kernel parameter fan override inference stage input count mismatch.");
    }
    if (stage_output_dims.size() != stage.outputs.size()) {
        throw std::runtime_error("Fused-kernel parameter fan override inference stage output count mismatch.");
    }

    std::unordered_map<std::string, uint64_t> param_numel_by_name;
    for (uint32_t node_idx = 0; node_idx < stage.expr.nodes.size(); ++node_idx) {
        const ExprNode& node = stage.expr.nodes[node_idx];
        if (node.op != ExprOp::INPUT) {
            continue;
        }
        if (node.input_slot >= stage.input_value_ids.size()) {
            throw std::runtime_error("Fused-kernel parameter fan override inference input slot out of range.");
        }

        auto name_it = root_input_name_by_slot.find(stage.input_value_ids[node.input_slot]);
        if (name_it == root_input_name_by_slot.end()) {
            continue;
        }
        if (!parameter_names.contains(name_it->second)) {
            continue;
        }

        param_numel_by_name.emplace(name_it->second, computeDimsNumel(stage_input_dims[node.input_slot]));
    }

    std::unordered_map<std::string, uint64_t> max_output_numel_per_example_by_name;
    for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
        const uint64_t output_numel_per_example = computeNumelPerExample(stage_output_dims[output_idx]);

        std::unordered_set<uint32_t> reachable_nodes;
        collectReachableLocalNodes(stage.expr, stage.outputs[output_idx].local_node_idx, reachable_nodes);

        for (uint32_t node_idx : reachable_nodes) {
            if (node_idx >= stage.expr.nodes.size()) {
                throw std::runtime_error("Fused-kernel parameter fan override inference reachable node out of range.");
            }

            const ExprNode& node = stage.expr.nodes[node_idx];
            if (node.op != ExprOp::INPUT) {
                continue;
            }
            if (node.input_slot >= stage.input_value_ids.size()) {
                throw std::runtime_error("Fused-kernel parameter fan override inference reachable input slot out of range.");
            }

            auto name_it = root_input_name_by_slot.find(stage.input_value_ids[node.input_slot]);
            if (name_it == root_input_name_by_slot.end()) {
                continue;
            }
            if (!parameter_names.contains(name_it->second)) {
                continue;
            }

            uint64_t& tracked = max_output_numel_per_example_by_name[name_it->second];
            tracked = std::max(tracked, output_numel_per_example);
        }
    }

    for (const auto& [input_name, output_numel_per_example] : max_output_numel_per_example_by_name) {
        auto param_numel_it = param_numel_by_name.find(input_name);
        if (param_numel_it == param_numel_by_name.end()) {
            continue;
        }

        const uint64_t param_numel = param_numel_it->second;
        const uint64_t fan_out = (param_numel == 0) ? 1 : std::max<uint64_t>(1, output_numel_per_example / param_numel);

        mergeParameterFanOverride(result,
                                  ParameterFanOverride{
                                      .input_name = input_name,
                                      .fan_in = 1,
                                      .fan_out = fan_out,
                                  });
    }
}

static void addMatmulParameterFanOverrides(const CompiledExecutionStage& stage,
                                           const std::vector<std::vector<uint64_t>>& stage_input_dims,
                                           const std::unordered_map<uint32_t, std::string>& root_input_name_by_slot,
                                           const std::unordered_set<std::string>& parameter_names,
                                           FusedEquation::ParameterFanOverrideMap& result) {
    if (stage.kind != CompiledExecutionStage::Kind::Matmul || !stage.matmul) {
        return;
    }
    if (stage_input_dims.size() < 2 || stage.input_value_ids.size() < 2) {
        throw std::runtime_error("Matmul/gemm stage expected at least two inputs for parameter fan override inference.");
    }

    const CompiledMatmul& compiled = *stage.matmul;
    const std::vector<uint64_t>& lhs_dims = stage_input_dims[0];
    const std::vector<uint64_t>& rhs_dims = stage_input_dims[1];
    if (lhs_dims.size() != 2 || rhs_dims.size() != 2) {
        throw std::runtime_error("Matmul/gemm parameter fan override inference currently only supports rank-2 inputs.");
    }

    const uint64_t lhs_rows = compiled.transpose_lhs ? lhs_dims[1] : lhs_dims[0];
    const uint64_t lhs_cols = compiled.transpose_lhs ? lhs_dims[0] : lhs_dims[1];
    const uint64_t rhs_rows = compiled.transpose_rhs ? rhs_dims[1] : rhs_dims[0];
    const uint64_t rhs_cols = compiled.transpose_rhs ? rhs_dims[0] : rhs_dims[1];

    if (lhs_cols != rhs_rows) {
        throw std::runtime_error("Matmul/gemm parameter fan override inference found incompatible matrix dimensions.");
    }

    auto maybe_add = [&](size_t operand_idx, uint64_t fan_in, uint64_t fan_out) {
        if (operand_idx >= stage.input_value_ids.size()) {
            return;
        }

        auto name_it = root_input_name_by_slot.find(stage.input_value_ids[operand_idx]);
        if (name_it == root_input_name_by_slot.end()) {
            return;
        }
        if (!parameter_names.contains(name_it->second)) {
            return;
        }

        mergeParameterFanOverride(result,
                                  ParameterFanOverride{
                                      .input_name = name_it->second,
                                      .fan_in = std::max<uint64_t>(1, fan_in),
                                      .fan_out = std::max<uint64_t>(1, fan_out),
                                  });
    };

    // For effective op(A)[m, k] @ op(B)[k, n], dense-initializer semantics are:
    //  - lhs parameter: fan_in = k, fan_out = m
    //  - rhs parameter: fan_in = k, fan_out = n
    maybe_add(0, lhs_cols, lhs_rows);
    maybe_add(1, rhs_rows, rhs_cols);
}

static void addConvolutionParameterFanOverrides(const CompiledExecutionStage& stage,
                                                const std::vector<std::vector<uint64_t>>& stage_input_dims,
                                                const std::unordered_map<uint32_t, std::string>& root_input_name_by_slot,
                                                const std::unordered_set<std::string>& parameter_names,
                                                FusedEquation::ParameterFanOverrideMap& result) {
    if (stage.kind != CompiledExecutionStage::Kind::Convolution || !stage.convolution) {
        return;
    }
    if (stage_input_dims.size() < 2 || stage.input_value_ids.size() < 2) {
        throw std::runtime_error("Convolution stage expected at least two inputs for parameter fan override inference.");
    }

    const std::vector<uint64_t>& input_dims = stage_input_dims[0];
    const std::vector<uint64_t>& filter_dims = stage_input_dims[1];
    if (input_dims.size() != filter_dims.size() || (input_dims.size() != 4 && input_dims.size() != 5)) {
        throw std::runtime_error(
            "Convolution parameter fan override inference currently only supports rank-4 NCHW/KCRS and rank-5 NCDHW/KCDHW "
            "input/filter tensors.");
    }

    const uint64_t input_channels = input_dims[1];
    const uint64_t filter_out_channels = filter_dims[0];
    const uint64_t filter_in_channels = filter_dims[1];

    uint64_t receptive_field = 1;
    for (size_t dim = 2; dim < filter_dims.size(); ++dim) {
        receptive_field *= filter_dims[dim];
    }

    if (input_channels != filter_in_channels) {
        throw std::runtime_error("Convolution parameter fan override inference found mismatched input/filter channels.");
    }

    auto maybe_add = [&](size_t operand_idx, uint64_t fan_in, uint64_t fan_out) {
        if (operand_idx >= stage.input_value_ids.size()) {
            return;
        }

        auto name_it = root_input_name_by_slot.find(stage.input_value_ids[operand_idx]);
        if (name_it == root_input_name_by_slot.end()) {
            return;
        }
        if (!parameter_names.contains(name_it->second)) {
            return;
        }

        mergeParameterFanOverride(result,
                                  ParameterFanOverride{
                                      .input_name = name_it->second,
                                      .fan_in = std::max<uint64_t>(1, fan_in),
                                      .fan_out = std::max<uint64_t>(1, fan_out),
                                  });
    };

    receptive_field = std::max<uint64_t>(1, receptive_field);

    // For filter[K, C, spatial...], convolution initializer semantics match standard conv kernels:
    //  - filter parameter: fan_in = C * product(spatial dims)
    //  - filter parameter: fan_out = K * product(spatial dims)
    // If someone intentionally makes the activation tensor itself a parameterized stage input,
    // give it the symmetric swapped interpretation.
    maybe_add(0, filter_out_channels * receptive_field, filter_in_channels * receptive_field);
    maybe_add(1, filter_in_channels * receptive_field, filter_out_channels * receptive_field);
}

static RuntimeDTypeKey makeRuntimeDTypeKey(const std::vector<NamedInput>& root_inputs,
                                           const std::unordered_map<uint32_t, RuntimeInputValue>& root_values) {
    RuntimeDTypeKey key;
    key.root_input_dtypes.resize(root_inputs.size());

    for (const NamedInput& input : root_inputs) {
        auto it = root_values.find(input.slot);
        if (it == root_values.end()) {
            throw std::runtime_error("Missing bound runtime input for root input slot " + std::to_string(input.slot) + ".");
        }
        key.root_input_dtypes[input.slot] = runtimeInputDType(it->second);
    }

    return key;
}

static RuntimeShapeKey makeRuntimeShapeKey(const std::vector<NamedInput>& root_inputs,
                                           const std::unordered_map<uint32_t, RuntimeInputValue>& root_values) {
    RuntimeShapeKey key;
    key.dtype_key = makeRuntimeDTypeKey(root_inputs, root_values);
    key.root_input_dims.resize(root_inputs.size());
    key.root_input_strides.resize(root_inputs.size());

    for (const NamedInput& input : root_inputs) {
        auto it = root_values.find(input.slot);
        if (it == root_values.end()) {
            throw std::runtime_error("Missing bound runtime input for root input slot " + std::to_string(input.slot) + ".");
        }
        key.root_input_dims[input.slot] = runtimeInputDims(it->second);
        key.root_input_strides[input.slot] = runtimeInputStridesForShapeKey(it->second);
    }

    return key;
}

static bool accumulatesIntoGradOutputs(const std::optional<BackwardEquationConfig>& backward_config) {
    return backward_config.has_value() && backward_config->accumulate_grad_outputs;
}

static std::unordered_set<std::string> backwardGradOutputNames(const std::optional<BackwardEquationConfig>& backward_config) {
    std::unordered_set<std::string> names;
    if (!backward_config.has_value()) {
        return names;
    }

    names.reserve(backward_config->wrt_names.size());
    for (const std::string& wrt_name : backward_config->wrt_names) {
        names.insert(wrt_name + "_grad");
    }
    return names;
}

static std::unordered_set<std::string> backwardAccumulationOutputNames(const std::optional<BackwardEquationConfig>& backward_config) {
    if (!accumulatesIntoGradOutputs(backward_config)) {
        return {};
    }
    return backwardGradOutputNames(backward_config);
}

static size_t externalRootInputCount(const std::vector<NamedInput>& root_inputs,
                                     const std::optional<BackwardEquationConfig>& backward_config) {
    const size_t accumulation_count = backwardAccumulationOutputNames(backward_config).size();
    if (accumulation_count > root_inputs.size()) {
        throw std::runtime_error("Invalid backward accumulation input accounting.");
    }
    return root_inputs.size() - accumulation_count;
}

static std::vector<const NamedInput*> externalRootTensorInputs(const std::vector<NamedInput>& root_inputs,
                                                               const std::optional<BackwardEquationConfig>& backward_config) {
    std::vector<const NamedInput*> tensor_inputs;
    tensor_inputs.reserve(root_inputs.size());

    const auto accumulation_output_names = backwardAccumulationOutputNames(backward_config);
    for (const NamedInput& input : root_inputs) {
        if (accumulation_output_names.contains(input.name)) {
            continue;
        }
        if (input.kind == NamedInput::Kind::Tensor) {
            tensor_inputs.push_back(&input);
        }
    }

    return tensor_inputs;
}

static size_t externalRootTensorInputCount(const std::vector<NamedInput>& root_inputs,
                                           const std::optional<BackwardEquationConfig>& backward_config) {
    return externalRootTensorInputs(root_inputs, backward_config).size();
}

static std::vector<std::string> inferBackwardWrtNamesFromOutputs(const PhysicalOutputs& backward_outputs) {
    std::vector<std::string> wrt_names;
    wrt_names.reserve(backward_outputs.outputs.size());
    for (const NamedOutput& output : backward_outputs.outputs) {
        constexpr const char* suffix = "_grad";
        constexpr size_t suffix_len = 5;
        if (output.name.size() >= suffix_len && output.name.compare(output.name.size() - suffix_len, suffix_len, suffix) == 0) {
            wrt_names.push_back(output.name.substr(0, output.name.size() - suffix_len));
        } else {
            wrt_names.push_back(output.name);
        }
    }
    return wrt_names;
}

static std::string dimsToString(const std::vector<uint64_t>& dims);
static void verifyRequestedOutputLayout(const std::vector<uint64_t>& outputDimensions, const std::vector<uint64_t>& expectedDimensions);

static std::optional<DataType> preferredBackwardGradBufferDType(const BackwardEquationConfig& backward_config,
                                                                const std::string& wrt_name) {
    if (!backward_config.forward_outputs_template.expr) {
        throw std::runtime_error("Backward grad-buffer dtype lookup requires non-null forward expr.");
    }

    uint32_t slot = UINT32_MAX;
    for (const NamedInput& input : backward_config.forward_outputs_template.expr->inputs) {
        if (input.name == wrt_name) {
            slot = input.slot;
            break;
        }
    }
    if (slot == UINT32_MAX) {
        throw std::runtime_error("Unknown backward grad-buffer input name: " + wrt_name);
    }

    for (const ExprNode& node : backward_config.forward_outputs_template.expr->nodes) {
        if ((node.op == ExprOp::INPUT || node.op == ExprOp::RUNTIME_SCALAR || node.op == ExprOp::TENSOR_RUNTIME_SCALAR) &&
            node.input_slot == slot) {
            if (node.backward_output_dtype.has_value()) {
                return node.backward_output_dtype;
            }
            if (node.output_dtype.has_value()) {
                return node.output_dtype;
            }
            return std::nullopt;
        }
    }

    throw std::runtime_error("No INPUT node found for backward grad-buffer input: " + wrt_name);
}

static std::vector<uint64_t> stripSingletonDimensions(const std::vector<uint64_t>& dims) {
    std::vector<uint64_t> stripped;
    stripped.reserve(dims.size());

    for (uint64_t dim : dims) {
        if (dim != 1)
            stripped.push_back(dim);
    }

    return stripped;
}

static bool outputDimensionsMatchIgnoringSingletons(const std::vector<uint64_t>& actual, const std::vector<uint64_t>& expected) {
    return stripSingletonDimensions(actual) == stripSingletonDimensions(expected);
}

static void validateBackwardAccumulationOutputs(const std::optional<BackwardEquationConfig>& backward_config,
                                                const std::unordered_map<std::string, Tensor>& named_inputs,
                                                const std::unordered_map<std::string, Tensor>& named_outputs,
                                                const std::unordered_map<std::string, std::vector<uint64_t>>& expected_output_shapes) {
    if (!accumulatesIntoGradOutputs(backward_config)) {
        return;
    }

    std::unordered_set<std::string> expected_output_names;
    expected_output_names.reserve(backward_config->wrt_names.size());
    for (const std::string& wrt_name : backward_config->wrt_names) {
        expected_output_names.insert(wrt_name + "_grad");
    }

    if (named_outputs.size() != expected_output_names.size()) {
        throw std::runtime_error("Backward accumulation stamp requires exactly " + std::to_string(expected_output_names.size()) +
                                 " gradient output tensors, but received " + std::to_string(named_outputs.size()) + ".");
    }

    for (const auto& [name, _] : named_outputs) {
        if (!expected_output_names.contains(name)) {
            throw std::runtime_error("Unexpected output tensor supplied for backward accumulation stamp: " + name);
        }
    }

    for (const std::string& wrt_name : backward_config->wrt_names) {
        const std::string grad_output_name = wrt_name + "_grad";

        auto output_it = named_outputs.find(grad_output_name);
        if (output_it == named_outputs.end()) {
            throw std::runtime_error("Missing required gradient accumulator for backward stamp: " + grad_output_name);
        }

        auto input_it = named_inputs.find(wrt_name);
        if (input_it == named_inputs.end()) {
            throw std::runtime_error("Missing forward input required to validate backward accumulator: " + wrt_name);
        }

        const Tensor& accumulator = output_it->second;
        const Tensor& wrt_input = input_it->second;

        std::vector<uint64_t> expected_dims;

        auto expected_it = expected_output_shapes.find(grad_output_name);
        if (expected_it != expected_output_shapes.end() && !expected_it->second.empty()) {
            expected_dims = expected_it->second;
        } else {
            expected_dims = wrt_input.getDimensions();
        }

        if (!outputDimensionsMatchIgnoringSingletons(accumulator.getDimensions(), expected_dims)) {
            throw std::runtime_error("Gradient accumulator tensor dimensions are incompatible for output '" + grad_output_name +
                                     "'. Expected compatible with " + dimsToString(expected_dims) + ", got " +
                                     dimsToString(accumulator.getDimensions()) + ".");
        }

        const std::optional<DataType> preferred_dtype = preferredBackwardGradBufferDType(backward_config.value(), wrt_name);
        const DataType expected_dtype = preferred_dtype.has_value() ? preferred_dtype.value() : wrt_input.getDataType();
        if (accumulator.getDataType() != expected_dtype) {
            throw std::runtime_error("Gradient accumulator tensor dtype mismatch for output '" + grad_output_name + "'.");
        }

        if (accumulator.getPlacement().getMemDevice() != wrt_input.getPlacement().getMemDevice() ||
            accumulator.getPlacement().getDeviceNum() != wrt_input.getPlacement().getDeviceNum()) {
            throw std::runtime_error("Gradient accumulator tensor placement mismatch for output '" + grad_output_name + "'.");
        }
    }
}

static std::unordered_map<std::string, std::vector<uint64_t>> mergeRequestedOutputShapesWithProvidedOutputs(
    const std::unordered_map<std::string, Tensor>& provided_outputs,
    const std::unordered_map<std::string, std::vector<uint64_t>>& requested_output_shapes,
    const std::optional<BackwardEquationConfig>& backward_config = std::nullopt) {
    std::unordered_map<std::string, std::vector<uint64_t>> effective = requested_output_shapes;
    const std::unordered_set<std::string> backward_grad_output_names = backwardGradOutputNames(backward_config);

    for (const auto& [name, output] : provided_outputs) {
        auto requested_it = effective.find(name);
        if (requested_it != effective.end() && !requested_it->second.empty() && !backward_grad_output_names.contains(name)) {
            verifyRequestedOutputLayout(output.getDimensions(), requested_it->second);
        }
        effective[name] = output.getDimensions();
    }

    return effective;
}

static void verifyRequestedOutputLayout(const std::vector<uint64_t>& outputDimensions, const std::vector<uint64_t>& expectedDimensions) {
    if (!outputDimensionsMatchIgnoringSingletons(outputDimensions, expectedDimensions)) {
        throw std::runtime_error("Output tensor dimensions are incompatible with the fused equation result.");
    }
}

static uint64_t product(const std::vector<uint64_t>& dims) {
    uint64_t p = 1;
    for (uint64_t d : dims)
        p *= d;
    return p;
}

static uint64_t maxNumel(const std::vector<std::vector<uint64_t>>& dims_by_output) {
    uint64_t max_numel = 0;
    for (const std::vector<uint64_t>& dims : dims_by_output) {
        max_numel = std::max<uint64_t>(max_numel, product(dims));
    }
    return max_numel;
}

static PhysicalExecutionStage toPhysicalFusedStage(const CompiledExecutionStage& stage) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        throw std::runtime_error("toPhysicalFusedStage called on non-fused stage.");
    }

    return PhysicalExecutionStage{
        .kind = PhysicalExecutionStage::Kind::FusedKernel,
        .expr = stage.expr,
        .input_value_ids = stage.input_value_ids,
        .outputs = stage.outputs,
    };
}

static std::shared_ptr<CompiledEquation> selectFlatCompiledEquation(const CompiledExecutionStage& stage,
                                                                    const EquationSignature& sig,
                                                                    uint64_t max_numel) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        throw std::runtime_error("selectFlatCompiledEquation called on non-fused stage.");
    }
    if (max_numel <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        if (!stage.flat) {
            throw std::runtime_error("Missing default flat fused kernel.");
        }
        return stage.flat;
    }

    return EquationCompiler::compileFusedStage(toPhysicalFusedStage(stage), sig, /*use_uint32_index_math=*/false);
}

static std::vector<uint64_t> computePackedOutputStrides(const std::vector<uint64_t>& outputDimensions) {
    const uint32_t rank = static_cast<uint32_t>(outputDimensions.size());
    std::vector<uint64_t> strides(rank, 1);

    if (rank == 0)
        return strides;

    strides[rank - 1] = 1;
    for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * outputDimensions[static_cast<size_t>(i) + 1];
    }

    return strides;
}

static void collectReferencedLocalInputSlots(const PhysicalExpression& expr, uint32_t node_idx, std::unordered_set<uint32_t>& slots) {
    if (node_idx >= expr.nodes.size()) {
        throw std::runtime_error("collectReferencedLocalInputSlots saw node index out of range.");
    }

    const ExprNode& node = expr.nodes[node_idx];

    if (node.op == ExprOp::INPUT || node.op == ExprOp::RUNTIME_SCALAR || node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
        slots.insert(node.input_slot);
        return;
    }

    if (Expression::isLeafOp(node.op)) {
        return;
    }

    collectReferencedLocalInputSlots(expr, node.lhs, slots);

    if (Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) {
        collectReferencedLocalInputSlots(expr, node.rhs, slots);
    }
    if (Expression::isTernaryOp(node.op)) {
        collectReferencedLocalInputSlots(expr, node.aux, slots);
    }
}

static void collectBroadcastOffsetLocalInputSlots(const PhysicalExpression& expr, uint32_t node_idx, std::unordered_set<uint32_t>& slots) {
    if (node_idx >= expr.nodes.size()) {
        throw std::runtime_error("collectBroadcastOffsetLocalInputSlots saw node index out of range.");
    }

    const ExprNode& node = expr.nodes[node_idx];

    if (node.op == ExprOp::INPUT || node.op == ExprOp::RUNTIME_SCALAR || node.op == ExprOp::TENSOR_RUNTIME_SCALAR) {
        slots.insert(node.input_slot);
        return;
    }

    if (Expression::isLeafOp(node.op)) {
        return;
    }

    if (node.op == ExprOp::STRIDED_VIEW_BACKWARD) {
        // The view-gradient source is evaluated with an explicit view-linear index
        // inside the strided_view_backward emitter. It does not consume ordinary
        // output-domain broadcast offsets, and may itself be a non-dense runtime
        // view alias whose logical rank differs from the source-gradient output.
        return;
    }

    collectBroadcastOffsetLocalInputSlots(expr, node.lhs, slots);

    if (Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) {
        collectBroadcastOffsetLocalInputSlots(expr, node.rhs, slots);
    }
    if (Expression::isTernaryOp(node.op)) {
        collectBroadcastOffsetLocalInputSlots(expr, node.aux, slots);
    }
}

static void collectReachableLocalNodes(const PhysicalExpression& expr, uint32_t node_idx, std::unordered_set<uint32_t>& nodes) {
    if (node_idx >= expr.nodes.size()) {
        throw std::runtime_error("collectReachableLocalNodes saw node index out of range.");
    }

    if (!nodes.insert(node_idx).second) {
        return;
    }

    const ExprNode& node = expr.nodes[node_idx];
    if (Expression::isLeafOp(node.op)) {
        return;
    }

    collectReachableLocalNodes(expr, node.lhs, nodes);

    if (Expression::isBinaryOp(node.op) || Expression::isTernaryOp(node.op)) {
        collectReachableLocalNodes(expr, node.rhs, nodes);
    }
    if (Expression::isTernaryOp(node.op)) {
        collectReachableLocalNodes(expr, node.aux, nodes);
    }
}

// static bool resolveLayoutFromDims(const std::vector<std::vector<uint64_t>>& inputs, std::vector<uint64_t>& outputDimensions);

// static std::vector<uint64_t> applySqueezeDims(const std::vector<uint64_t>& input_dims, const std::vector<uint64_t>& squeeze_axes) {
//     if (squeeze_axes.empty()) {
//         return input_dims;
//     }
//
//     std::vector<uint64_t> normalized = squeeze_axes;
//     std::sort(normalized.begin(), normalized.end());
//     normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
//
//     if (normalized.size() == 1 && normalized[0] == UINT64_MAX) {
//         std::vector<uint64_t> out_dims;
//         out_dims.reserve(input_dims.size());
//         for (uint64_t dim : input_dims) {
//             if (dim != 1) {
//                 out_dims.push_back(dim);
//             }
//         }
//         return out_dims;
//     }
//
//     std::vector<uint64_t> out_dims;
//     out_dims.reserve(input_dims.size());
//     size_t next_axis_i = 0;
//     uint64_t next_axis = normalized.empty() ? UINT64_MAX : normalized[0];
//     for (uint64_t axis = 0; axis < input_dims.size(); ++axis) {
//         if (next_axis_i < normalized.size() && axis == next_axis) {
//             if (input_dims[axis] != 1) {
//                 throw std::runtime_error("squeeze axes must refer to singleton dimensions.");
//             }
//             ++next_axis_i;
//             next_axis = next_axis_i < normalized.size() ? normalized[next_axis_i] : UINT64_MAX;
//             continue;
//         }
//         out_dims.push_back(input_dims[axis]);
//     }
//
//     if (next_axis_i != normalized.size()) {
//         throw std::runtime_error("squeeze axes are invalid for the input rank.");
//     }
//
//     return out_dims;
// }

static std::vector<uint64_t> inferRmsNormOutputDims(uint64_t normalized_feature_count,
                                                    const std::vector<uint64_t>& input_dims,
                                                    const std::vector<uint64_t>& scale_dims) {
    if (input_dims.size() != 2) {
        throw std::runtime_error("RMSNorm expression stage currently expects a rank-2 [outer, normalized_features] input.");
    }
    if (scale_dims.size() != 1) {
        throw std::runtime_error("RMSNorm expression stage expects a rank-1 scale tensor.");
    }
    if (normalized_feature_count == 0) {
        throw std::runtime_error("RMSNorm expression stage has zero normalized feature count.");
    }
    if (input_dims[1] != normalized_feature_count) {
        throw std::runtime_error("RMSNorm expression normalized feature count does not match the input tail dimension.");
    }
    if (scale_dims[0] != normalized_feature_count) {
        throw std::runtime_error("RMSNorm expression scale dimension does not match the normalized feature count.");
    }
    return input_dims;
}

static std::vector<uint64_t> inferRmsNormOutputDims(const ExprNode& node,
                                                    const std::vector<uint64_t>& input_dims,
                                                    const std::vector<uint64_t>& scale_dims) {
    return inferRmsNormOutputDims(node.rms_norm_normalized_feature_count, input_dims, scale_dims);
}

static std::vector<uint64_t> inferRmsNormOutputDims(const CompiledRmsNorm& compiled,
                                                    const std::vector<uint64_t>& input_dims,
                                                    const std::vector<uint64_t>& scale_dims) {
    return inferRmsNormOutputDims(compiled.normalized_feature_count, input_dims, scale_dims);
}

static std::string dimsToString(const std::vector<uint64_t>& dims) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) {
            oss << ", ";
        }
        oss << dims[i];
    }
    oss << "]";
    return oss.str();
}

// static std::vector<uint64_t> applyUnsqueezeDims(const std::vector<uint64_t>& input_dims, const std::vector<uint64_t>& unsqueeze_axes) {
//     std::vector<uint64_t> out_dims;
//     out_dims.reserve(input_dims.size() + unsqueeze_axes.size());
//
//     const uint64_t output_rank = static_cast<uint64_t>(input_dims.size() + unsqueeze_axes.size());
//     size_t input_i = 0;
//     size_t axis_i = 0;
//
//     for (uint64_t out_axis = 0; out_axis < output_rank; ++out_axis) {
//         if (axis_i < unsqueeze_axes.size() && unsqueeze_axes[axis_i] == out_axis) {
//             out_dims.push_back(1);
//             ++axis_i;
//         } else {
//             if (input_i >= input_dims.size()) {
//                 throw std::runtime_error("unsqueeze axes are invalid for the input rank.");
//             }
//             out_dims.push_back(input_dims[input_i++]);
//         }
//     }
//
//     if (input_i != input_dims.size() || axis_i != unsqueeze_axes.size()) {
//         throw std::runtime_error("unsqueeze axes are invalid for the input rank.");
//     }
//
//     return out_dims;
// }

static std::vector<std::vector<uint64_t>> inferFusedStageNodeDims(const PhysicalExpression& expr,
                                                                  const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    std::vector<std::vector<uint64_t>> node_dims(expr.nodes.size());

    for (size_t i = 0; i < expr.nodes.size(); ++i) {
        const ExprNode& node = expr.nodes[i];

        // std::cerr << "[FUSION] visiting node"
        //           << " local_node=" << i << " op=" << static_cast<int>(node.op) << " lhs=" << node.lhs << " rhs=" << node.rhs
        //           << " input_slot=" << node.input_slot << std::endl;

        switch (node.op) {
            case ExprOp::INPUT: {
                if (node.input_slot >= stage_input_dims.size()) {
                    throw std::runtime_error("Stage input slot out of range during fused-stage shape inference.");
                }
                node_dims[i] = stage_input_dims[node.input_slot];
                break;
            }
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
            case ExprOp::SCALAR_FP:
                node_dims[i] = {1};
                break;
            case ExprOp::FILL:
                node_dims[i] = node.fill_dims;
                break;
            case ExprOp::RAGGED_VALUEWISE_EXTENT:
                node_dims[i] = inferRaggedValuewiseExtentDims(node, node_dims.at(node.lhs), node_dims.at(node.rhs));
                break;
            case ExprOp::ADD:
            case ExprOp::SUB:
            case ExprOp::MUL:
            case ExprOp::DIV:
            case ExprOp::POW:
            case ExprOp::EQUAL:
            case ExprOp::NOT_EQUAL:
            case ExprOp::LESS:
            case ExprOp::LESS_EQUAL:
            case ExprOp::GREATER:
            case ExprOp::GREATER_EQUAL:
            case ExprOp::LOGICAL_AND:
            case ExprOp::LOGICAL_OR:
            case ExprOp::MIN:
            case ExprOp::MAX:
            case ExprOp::MIN_GRAD_LEFT:
            case ExprOp::MIN_GRAD_RIGHT:
            case ExprOp::MAX_GRAD_LEFT:
            case ExprOp::MAX_GRAD_RIGHT: {
                std::vector<std::vector<uint64_t>> non_scalar_inputs;
                if (!node_dims[node.lhs].empty())
                    non_scalar_inputs.push_back(node_dims[node.lhs]);
                if (!node_dims[node.rhs].empty())
                    non_scalar_inputs.push_back(node_dims[node.rhs]);
                if (non_scalar_inputs.empty()) {
                    node_dims[i] = {};
                } else if (non_scalar_inputs.size() == 1) {
                    node_dims[i] = non_scalar_inputs[0];
                } else {
                    std::vector<uint64_t> out_dims;
                    resolveLayoutFromDims(non_scalar_inputs, out_dims);
                    node_dims[i] = std::move(out_dims);
                }
                break;
            }
            case ExprOp::WHERE: {
                std::vector<std::vector<uint64_t>> non_scalar_inputs;
                if (!node_dims[node.lhs].empty())
                    non_scalar_inputs.push_back(node_dims[node.lhs]);
                if (!node_dims[node.rhs].empty())
                    non_scalar_inputs.push_back(node_dims[node.rhs]);
                if (!node_dims[node.aux].empty())
                    non_scalar_inputs.push_back(node_dims[node.aux]);
                if (non_scalar_inputs.empty()) {
                    node_dims[i] = {};
                } else if (non_scalar_inputs.size() == 1) {
                    node_dims[i] = non_scalar_inputs[0];
                } else {
                    std::vector<uint64_t> out_dims;
                    resolveLayoutFromDims(non_scalar_inputs, out_dims);
                    node_dims[i] = std::move(out_dims);
                }
                break;
            }
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
            case ExprOp::CAST:
                node_dims[i] = node_dims[node.lhs];
                break;
            case ExprOp::ROPE: {
                node_dims[i] = node_dims[node.lhs];
                const std::vector<uint64_t>& dims = node_dims[i];
                if (dims.empty()) {
                    throw std::runtime_error("RoPE requires a tensor input.");
                }
                if (node.rope_sequence_axis >= dims.size() || node.rope_head_dim_axis >= dims.size()) {
                    throw std::runtime_error("RoPE sequence_axis/head_dim_axis are out of range for the input rank.");
                }
                if (node.rope_head_dim_axis + 1 != dims.size()) {
                    throw std::runtime_error(
                        "RoPE currently requires head_dim_axis to be the innermost dimension for coalesced pair rotation.");
                }
                const uint64_t head_dim = dims[node.rope_head_dim_axis];
                const uint64_t rotary_dim = node.rope_rotary_dim == 0 ? head_dim : node.rope_rotary_dim;
                if (rotary_dim == 0 || (rotary_dim & 1ULL) != 0ULL || rotary_dim > head_dim) {
                    throw std::runtime_error("RoPE rotary_dim must be even, non-zero, and <= the head dimension.");
                }
                break;
            }
            case ExprOp::TAKE_ALONG_AXIS:
                node_dims[i] = inferTakeAlongAxisOutputDims(node, node_dims[node.lhs], node_dims[node.rhs]);
                break;
            case ExprOp::TRANSPOSE:
                node_dims[i] = inferTransposeOutputDims(node_dims[node.lhs]);
                break;
            case ExprOp::RESHAPE:
                node_dims[i] = resolveDynamicAliasDims(node_dims[node.lhs], node.reshape_dims, true, "inferFusedStageNodeDims RESHAPE");
                break;
            case ExprOp::STRIDED_VIEW:
                if (node.view_dims.empty() || node.view_strides.size() != node.view_dims.size()) {
                    throw std::runtime_error("Expression strided_view requires dimensions and strides with the same non-zero rank.");
                }
                node_dims[i] = resolveDynamicAliasDims(node_dims[node.lhs], node.view_dims, false, "Expression strided_view");
                break;
            case ExprOp::STRIDED_VIEW_BACKWARD:
                if (node.fill_dims.empty() || node.view_dims.empty() || node.view_strides.size() != node.view_dims.size()) {
                    throw std::runtime_error(
                        "Expression strided_view_backward requires source dimensions and matching view dimensions/strides.");
                }
                node_dims[i] = node.fill_dims;
                break;
            case ExprOp::UNSQUEEZE: {
                const std::vector<uint64_t>& lhs_dims = node_dims[node.lhs];

                // std::cerr << "[FUSION] infer node UNSQUEEZE begin"
                //           << " local_node=" << i << " lhs=" << node.lhs << " lhs_dims=" << dimsToString(lhs_dims)
                //           << " unsqueeze_axes=" << dimsToString(node.unsqueeze_axes) << std::endl;

                try {
                    node_dims[i] = applyNormalizedUnsqueezeDims(lhs_dims, node.unsqueeze_axes);
                } catch (const std::exception& e) {
                    std::ostringstream oss;
                    oss << "inferFusedStageNodeDims UNSQUEEZE failed"
                        << " local_node=" << i << " lhs=" << node.lhs << " lhs_dims=" << dimsToString(lhs_dims)
                        << " unsqueeze_axes=" << dimsToString(node.unsqueeze_axes) << " error=" << e.what();
                    throw std::runtime_error(oss.str());
                }

                // std::cerr << "[FUSION] infer node UNSQUEEZE end"
                //           << " local_node=" << i << " out_dims=" << dimsToString(node_dims[i]) << std::endl;
                break;
            }

            case ExprOp::SQUEEZE: {
                const std::vector<uint64_t>& lhs_dims = node_dims[node.lhs];

                // std::cerr << "[FUSION] infer node SQUEEZE begin"
                //           << " local_node=" << i << " lhs=" << node.lhs << " lhs_dims=" << dimsToString(lhs_dims)
                //           << " squeeze_axes=" << dimsToString(node.squeeze_axes) << std::endl;

                try {
                    node_dims[i] = applyNormalizedSqueezeDims(lhs_dims, node.squeeze_axes);
                } catch (const std::exception& e) {
                    std::ostringstream oss;
                    oss << "inferFusedStageNodeDims SQUEEZE failed"
                        << " local_node=" << i << " lhs=" << node.lhs << " lhs_dims=" << dimsToString(lhs_dims)
                        << " squeeze_axes=" << dimsToString(node.squeeze_axes) << " error=" << e.what();
                    throw std::runtime_error(oss.str());
                }

                // std::cerr << "[FUSION] infer node SQUEEZE end"
                //           << " local_node=" << i << " out_dims=" << dimsToString(node_dims[i]) << std::endl;
                break;
            }
            case ExprOp::REDUCE_SUM:
            case ExprOp::REDUCE_PROD:
            case ExprOp::REDUCE_MIN:
            case ExprOp::REDUCE_MAX:
            case ExprOp::REDUCE_ARGMIN:
            case ExprOp::REDUCE_ARGMAX:
            case ExprOp::REDUCE_AVG:
            case ExprOp::REDUCE_NORM1:
            case ExprOp::REDUCE_NORM2:
                node_dims[i] = StampedEquation::computeReductionOutputDims(node_dims[node.lhs], node.reduction_axes, node.squeeze_axes);
                break;
            default:
                throw std::runtime_error("inferFusedStageNodeDims encountered unknown ExprOp.");
        }
    }

    return node_dims;
}

static uint64_t checkedAddU64(uint64_t a, uint64_t b, const char* what) {
    unsigned __int128 s = static_cast<unsigned __int128>(a) + static_cast<unsigned __int128>(b);
    if (s > std::numeric_limits<uint64_t>::max()) {
        throw std::runtime_error(std::string("FLOP count overflow in ") + what + ".");
    }
    return static_cast<uint64_t>(s);
}

static uint64_t checkedMulU64(uint64_t a, uint64_t b, const char* what) {
    unsigned __int128 p = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
    if (p > std::numeric_limits<uint64_t>::max()) {
        throw std::runtime_error(std::string("FLOP count overflow in ") + what + ".");
    }
    return static_cast<uint64_t>(p);
}

static uint64_t numelFromDims(const std::vector<uint64_t>& dims) {
    uint64_t n = 1;
    for (uint64_t d : dims) {
        n = checkedMulU64(n, d, "numelFromDims");
    }
    return n;
}

static uint64_t reductionExtent(const std::vector<uint64_t>& input_dims, const std::vector<uint64_t>& reduction_axes) {
    const std::vector<uint64_t> axes = resolveReductionAxesForInputRank(reduction_axes, input_dims.size());
    uint64_t n = 1;
    for (uint64_t axis : axes) {
        if (axis >= input_dims.size()) {
            throw std::runtime_error("Reduction axis out of range while computing FLOPs.");
        }
        n = checkedMulU64(n, input_dims[axis], "reductionExtent");
    }
    return n;
}

static uint64_t reductionSemanticFlops(ExprOp op, uint64_t output_numel, uint64_t reduce_extent) {
    if (output_numel == 0 || reduce_extent == 0) {
        return 0;
    }

    switch (op) {
        case ExprOp::REDUCE_SUM:
        case ExprOp::REDUCE_PROD:
        case ExprOp::REDUCE_MIN:
        case ExprOp::REDUCE_MAX:
        case ExprOp::REDUCE_ARGMIN:
        case ExprOp::REDUCE_ARGMAX:
            return checkedMulU64(output_numel, reduce_extent - 1, "reductionSemanticFlops");

        case ExprOp::REDUCE_AVG:
            return checkedMulU64(output_numel, reduce_extent, "reductionSemanticFlops");

        case ExprOp::REDUCE_NORM1: {
            const uint64_t per_output =
                checkedAddU64(checkedMulU64(2, reduce_extent, "reductionSemanticFlops"), 0, "reductionSemanticFlops") - 1;
            return checkedMulU64(output_numel, per_output, "reductionSemanticFlops");
        }

        case ExprOp::REDUCE_NORM2: {
            const uint64_t per_output = checkedMulU64(2, reduce_extent, "reductionSemanticFlops");
            return checkedMulU64(output_numel, per_output, "reductionSemanticFlops");
        }

        default:
            throw std::runtime_error("Unsupported reduction op while computing FLOPs.");
    }
}

static uint64_t perElementSemanticFlops(ExprOp op) {
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
        case ExprOp::SQRT:
        case ExprOp::MIN:
        case ExprOp::MAX:
        case ExprOp::MIN_GRAD_LEFT:
        case ExprOp::MIN_GRAD_RIGHT:
        case ExprOp::MAX_GRAD_LEFT:
        case ExprOp::MAX_GRAD_RIGHT:
        case ExprOp::EQUAL:
        case ExprOp::NOT_EQUAL:
        case ExprOp::LESS:
        case ExprOp::LESS_EQUAL:
        case ExprOp::GREATER:
        case ExprOp::GREATER_EQUAL:
        case ExprOp::LOGICAL_AND:
        case ExprOp::LOGICAL_OR:
        case ExprOp::LOGICAL_NOT:
        case ExprOp::CAST:
        case ExprOp::WHERE:
            return 1;

        case ExprOp::POW:
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
        case ExprOp::TANH:
        case ExprOp::NORMCDF:
        case ExprOp::ROPE:
            return 10;

        case ExprOp::INPUT:
        case ExprOp::RUNTIME_SCALAR:
        case ExprOp::TENSOR_RUNTIME_SCALAR:
        case ExprOp::SCALAR_FP:
        case ExprOp::FILL:
        case ExprOp::RAGGED_VALUEWISE_EXTENT:
        case ExprOp::RESHAPE:
        case ExprOp::STRIDED_VIEW:
        case ExprOp::STRIDED_VIEW_BACKWARD:
        case ExprOp::UNSQUEEZE:
        case ExprOp::SQUEEZE:
            return 0;

        default:
            throw std::runtime_error("Unsupported per-element op while computing FLOPs.");
    }
}

static std::vector<std::vector<uint64_t>> inferFusedStageNodeDimsForReachable(const PhysicalExpression& expr,
                                                                              const std::vector<std::vector<uint64_t>>& stage_input_dims,
                                                                              const std::unordered_set<uint32_t>& reachable_nodes) {
    std::vector<std::vector<uint64_t>> node_dims(expr.nodes.size());

    for (size_t i = 0; i < expr.nodes.size(); ++i) {
        if (!reachable_nodes.contains(static_cast<uint32_t>(i))) {
            continue;
        }

        const ExprNode& node = expr.nodes[i];

        switch (node.op) {
            case ExprOp::INPUT: {
                if (node.input_slot >= stage_input_dims.size()) {
                    throw std::runtime_error("Stage input slot out of range during fused-stage shape inference.");
                }
                node_dims[i] = stage_input_dims[node.input_slot];
                break;
            }
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
            case ExprOp::SCALAR_FP:
                node_dims[i] = {1};
                break;
            case ExprOp::FILL:
                node_dims[i] = node.fill_dims;
                break;
            case ExprOp::RAGGED_VALUEWISE_EXTENT:
                node_dims[i] = inferRaggedValuewiseExtentDims(node, node_dims.at(node.lhs), node_dims.at(node.rhs));
                break;
            case ExprOp::ADD:
            case ExprOp::SUB:
            case ExprOp::MUL:
            case ExprOp::DIV:
            case ExprOp::POW:
            case ExprOp::EQUAL:
            case ExprOp::NOT_EQUAL:
            case ExprOp::LESS:
            case ExprOp::LESS_EQUAL:
            case ExprOp::GREATER:
            case ExprOp::GREATER_EQUAL:
            case ExprOp::LOGICAL_AND:
            case ExprOp::LOGICAL_OR:
            case ExprOp::MIN:
            case ExprOp::MAX:
            case ExprOp::MIN_GRAD_LEFT:
            case ExprOp::MIN_GRAD_RIGHT:
            case ExprOp::MAX_GRAD_LEFT:
            case ExprOp::MAX_GRAD_RIGHT: {
                std::vector<std::vector<uint64_t>> non_scalar_inputs;
                if (!node_dims[node.lhs].empty())
                    non_scalar_inputs.push_back(node_dims[node.lhs]);
                if (!node_dims[node.rhs].empty())
                    non_scalar_inputs.push_back(node_dims[node.rhs]);

                if (non_scalar_inputs.empty()) {
                    node_dims[i] = {};
                } else if (non_scalar_inputs.size() == 1) {
                    node_dims[i] = non_scalar_inputs[0];
                } else {
                    std::vector<uint64_t> out_dims;
                    try {
                        resolveLayoutFromDims(non_scalar_inputs, out_dims);
                    } catch (const std::exception& e) {
                        std::ostringstream oss;
                        oss << "inferFusedStageNodeDimsForReachable binary-op broadcast failure"
                            << " local_node=" << i << " op=" << static_cast<int>(node.op) << " lhs=" << node.lhs << " rhs=" << node.rhs
                            << " lhs_dims=" << dimsToString(node_dims[node.lhs]) << " rhs_dims=" << dimsToString(node_dims[node.rhs])
                            << " reachable=" << (reachable_nodes.contains(static_cast<uint32_t>(i)) ? "true" : "false")
                            << " error=" << e.what();
                        throw std::runtime_error(oss.str());
                    }
                    node_dims[i] = std::move(out_dims);
                }
                break;
            }
            case ExprOp::WHERE: {
                std::vector<std::vector<uint64_t>> non_scalar_inputs;
                if (!node_dims[node.lhs].empty())
                    non_scalar_inputs.push_back(node_dims[node.lhs]);
                if (!node_dims[node.rhs].empty())
                    non_scalar_inputs.push_back(node_dims[node.rhs]);
                if (!node_dims[node.aux].empty())
                    non_scalar_inputs.push_back(node_dims[node.aux]);

                if (non_scalar_inputs.empty()) {
                    node_dims[i] = {};
                } else if (non_scalar_inputs.size() == 1) {
                    node_dims[i] = non_scalar_inputs[0];
                } else {
                    std::vector<uint64_t> out_dims;
                    try {
                        resolveLayoutFromDims(non_scalar_inputs, out_dims);
                    } catch (const std::exception& e) {
                        std::ostringstream oss;
                        oss << "inferFusedStageNodeDimsForReachable where broadcast failure"
                            << " local_node=" << i << " cond=" << node.lhs << " true=" << node.rhs << " false=" << node.aux
                            << " cond_dims=" << dimsToString(node_dims[node.lhs]) << " true_dims=" << dimsToString(node_dims[node.rhs])
                            << " false_dims=" << dimsToString(node_dims[node.aux])
                            << " reachable=" << (reachable_nodes.contains(static_cast<uint32_t>(i)) ? "true" : "false")
                            << " error=" << e.what();
                        throw std::runtime_error(oss.str());
                    }
                    node_dims[i] = std::move(out_dims);
                }
                break;
            }
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
            case ExprOp::CAST:
                node_dims[i] = node_dims[node.lhs];
                break;
            case ExprOp::ROPE: {
                node_dims[i] = node_dims[node.lhs];
                const std::vector<uint64_t>& dims = node_dims[i];
                if (dims.empty()) {
                    throw std::runtime_error("RoPE requires a tensor input.");
                }
                if (node.rope_sequence_axis >= dims.size() || node.rope_head_dim_axis >= dims.size()) {
                    throw std::runtime_error("RoPE sequence_axis/head_dim_axis are out of range for the input rank.");
                }
                if (node.rope_head_dim_axis + 1 != dims.size()) {
                    throw std::runtime_error(
                        "RoPE currently requires head_dim_axis to be the innermost dimension for coalesced pair rotation.");
                }
                const uint64_t head_dim = dims[node.rope_head_dim_axis];
                const uint64_t rotary_dim = node.rope_rotary_dim == 0 ? head_dim : node.rope_rotary_dim;
                if (rotary_dim == 0 || (rotary_dim & 1ULL) != 0ULL || rotary_dim > head_dim) {
                    throw std::runtime_error("RoPE rotary_dim must be even, non-zero, and <= the head dimension.");
                }
                break;
            }
            case ExprOp::RMSNORM:
                node_dims[i] = inferRmsNormOutputDims(node, node_dims[node.lhs], node_dims[node.rhs]);
                break;
            case ExprOp::RESHAPE:
                node_dims[i] = resolveDynamicAliasDims(node_dims[node.lhs], node.reshape_dims, true, "Fused-stage reshape");
                break;
            case ExprOp::STRIDED_VIEW:
                if (node.view_dims.empty() || node.view_strides.size() != node.view_dims.size()) {
                    throw std::runtime_error("Expression strided_view requires dimensions and strides with the same non-zero rank.");
                }
                node_dims[i] = resolveDynamicAliasDims(node_dims[node.lhs], node.view_dims, false, "Expression strided_view");
                break;
            case ExprOp::STRIDED_VIEW_BACKWARD:
                if (node.fill_dims.empty() || node.view_dims.empty() || node.view_strides.size() != node.view_dims.size()) {
                    throw std::runtime_error(
                        "Expression strided_view_backward requires source dimensions and matching view dimensions/strides.");
                }
                node_dims[i] = node.fill_dims;
                break;
            case ExprOp::UNSQUEEZE: {
                const std::vector<uint64_t>& lhs_dims = node_dims[node.lhs];
                node_dims[i] = applyNormalizedUnsqueezeDims(lhs_dims, node.unsqueeze_axes);
                break;
            }
            case ExprOp::SQUEEZE: {
                const std::vector<uint64_t>& lhs_dims = node_dims[node.lhs];
                node_dims[i] = applyNormalizedSqueezeDims(lhs_dims, node.squeeze_axes);
                break;
            }
            case ExprOp::REDUCE_SUM:
            case ExprOp::REDUCE_PROD:
            case ExprOp::REDUCE_MIN:
            case ExprOp::REDUCE_MAX:
            case ExprOp::REDUCE_ARGMIN:
            case ExprOp::REDUCE_ARGMAX:
            case ExprOp::REDUCE_AVG:
            case ExprOp::REDUCE_NORM1:
            case ExprOp::REDUCE_NORM2: {
                const std::vector<uint64_t>& lhs_dims = node_dims[node.lhs];
                const std::vector<uint64_t> reduction_axes = resolveReductionAxesForInputRank(node.reduction_axes, lhs_dims.size());
                node_dims[i] = StampedEquation::computeReductionOutputDims(lhs_dims, reduction_axes, node.squeeze_axes);
                break;
            }
            case ExprOp::TAKE_ALONG_AXIS:
                node_dims[i] = inferTakeAlongAxisOutputDims(node, node_dims[node.lhs], node_dims[node.rhs]);
                break;
            case ExprOp::TRANSPOSE:
                node_dims[i] = inferTransposeOutputDims(node_dims[node.lhs]);
                break;
            case ExprOp::MATMUL:
            case ExprOp::GEMM:
                throw std::runtime_error("inferFusedStageNodeDimsForReachable encountered unexpected matmul/gemm op in fused stage.");
            case ExprOp::CONV2D:
            case ExprOp::CONV2D_BACKWARD_DATA:
            case ExprOp::CONV2D_BACKWARD_FILTER:
            case ExprOp::CONV3D:
            case ExprOp::CONV3D_BACKWARD_DATA:
            case ExprOp::CONV3D_BACKWARD_FILTER:
                throw std::runtime_error("inferFusedStageNodeDimsForReachable encountered unexpected convolution op in fused stage.");
            default:
                throw std::runtime_error("inferFusedStageNodeDimsForReachable encountered unknown ExprOp.");
        }
    }

    return node_dims;
}

static uint64_t computeFusedStageFlops(const PhysicalExpression& expr,
                                       const std::vector<std::vector<uint64_t>>& stage_input_dims,
                                       const std::vector<CompiledStageOutput>& outputs) {
    std::unordered_set<uint32_t> reachable_nodes;
    for (const CompiledStageOutput& out : outputs) {
        collectReachableLocalNodes(expr, out.local_node_idx, reachable_nodes);
    }

    const std::vector<std::vector<uint64_t>> node_dims = inferFusedStageNodeDimsForReachable(expr, stage_input_dims, reachable_nodes);

    uint64_t total = 0;

    for (size_t i = 0; i < expr.nodes.size(); ++i) {
        const uint32_t node_idx = static_cast<uint32_t>(i);
        if (!reachable_nodes.contains(node_idx)) {
            continue;
        }

        const ExprNode& node = expr.nodes[i];
        switch (node.op) {
            case ExprOp::INPUT:
            case ExprOp::RUNTIME_SCALAR:
            case ExprOp::TENSOR_RUNTIME_SCALAR:
            case ExprOp::SCALAR_FP:
            case ExprOp::FILL:
            case ExprOp::RAGGED_VALUEWISE_EXTENT:
            case ExprOp::RESHAPE:
            case ExprOp::STRIDED_VIEW:
            case ExprOp::STRIDED_VIEW_BACKWARD:
            case ExprOp::UNSQUEEZE:
            case ExprOp::SQUEEZE:
            case ExprOp::TRANSPOSE:
            case ExprOp::TAKE_ALONG_AXIS:
                break;

            case ExprOp::ADD:
            case ExprOp::SUB:
            case ExprOp::MUL:
            case ExprOp::DIV:
            case ExprOp::POW:
            case ExprOp::EQUAL:
            case ExprOp::NOT_EQUAL:
            case ExprOp::LESS:
            case ExprOp::LESS_EQUAL:
            case ExprOp::GREATER:
            case ExprOp::GREATER_EQUAL:
            case ExprOp::LOGICAL_AND:
            case ExprOp::LOGICAL_OR:
            case ExprOp::WHERE:
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
            case ExprOp::LOGICAL_NOT:
            case ExprOp::CAST:
            case ExprOp::ROPE:
            case ExprOp::MIN:
            case ExprOp::MAX:
            case ExprOp::MIN_GRAD_LEFT:
            case ExprOp::MIN_GRAD_RIGHT:
            case ExprOp::MAX_GRAD_LEFT:
            case ExprOp::MAX_GRAD_RIGHT: {
                const uint64_t n = numelFromDims(node_dims[i]);
                total = checkedAddU64(
                    total, checkedMulU64(n, perElementSemanticFlops(node.op), "computeFusedStageFlops"), "computeFusedStageFlops");
                break;
            }

            case ExprOp::REDUCE_SUM:
            case ExprOp::REDUCE_PROD:
            case ExprOp::REDUCE_MIN:
            case ExprOp::REDUCE_MAX:
            case ExprOp::REDUCE_ARGMIN:
            case ExprOp::REDUCE_ARGMAX:
            case ExprOp::REDUCE_AVG:
            case ExprOp::REDUCE_NORM1:
            case ExprOp::REDUCE_NORM2: {
                const uint64_t out_numel = numelFromDims(node_dims[i]);
                const uint64_t red_extent = reductionExtent(node_dims[node.lhs], node.reduction_axes);
                total = checkedAddU64(total, reductionSemanticFlops(node.op, out_numel, red_extent), "computeFusedStageFlops");
                break;
            }

            case ExprOp::CUDA_KERNEL_OUTPUT:
            case ExprOp::SCAN:
            case ExprOp::SEGMENTED_SCAN:
            case ExprOp::SEGMENTED_REDUCE_SUM:
            case ExprOp::SEGMENTED_REDUCE_MIN:
            case ExprOp::SEGMENTED_REDUCE_MAX:
            case ExprOp::EMBEDDING_LOOKUP:
            case ExprOp::SOFTMAX:
            case ExprOp::ATTENTION:
            case ExprOp::ATTENTION_BACKWARD_Q:
            case ExprOp::ATTENTION_BACKWARD_K:
            case ExprOp::ATTENTION_BACKWARD_V:
            case ExprOp::ATTENTION_BACKWARD_BIAS:
            case ExprOp::MATMUL:
            case ExprOp::GEMM:
            case ExprOp::CONV2D:
            case ExprOp::CONV2D_BACKWARD_DATA:
            case ExprOp::CONV2D_BACKWARD_FILTER:
            case ExprOp::CONV3D:
            case ExprOp::CONV3D_BACKWARD_DATA:
            case ExprOp::CONV3D_BACKWARD_FILTER:
            case ExprOp::RMSNORM:
            case ExprOp::REDUCE_MIN_BACKWARD:
            case ExprOp::REDUCE_MAX_BACKWARD:
            case ExprOp::SCAN_MIN_BACKWARD:
            case ExprOp::SCAN_MAX_BACKWARD:
            case ExprOp::SEGMENTED_SCAN_MIN_BACKWARD:
            case ExprOp::SEGMENTED_SCAN_MAX_BACKWARD:
                throw std::runtime_error("Unexpected staged op inside fused kernel while computing FLOPs.");
        }
    }

    return total;
}

static uint64_t computeReductionStageFlops(const CompiledReduction& reduction, const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    if (stage_input_dims.empty()) {
        throw std::runtime_error("Reduction stage missing input dims while computing FLOPs.");
    }

    const std::vector<uint64_t> axes = resolveReductionAxesForInputRank(reduction.reduction_axes, stage_input_dims[0].size());
    const std::vector<uint64_t> out_dims = StampedEquation::computeReductionOutputDims(stage_input_dims[0], axes, reduction.squeeze_axes);

    return reductionSemanticFlops(reduction.op, numelFromDims(out_dims), reductionExtent(stage_input_dims[0], axes));
}

static uint64_t computeArgMinMaxStageFlops(const CompiledArgMinMax& argminmax, const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    if (stage_input_dims.empty()) {
        throw std::runtime_error("ArgMinMax stage missing input dims while computing FLOPs.");
    }

    const std::vector<uint64_t> axes = resolveReductionAxesForInputRank(argminmax.reduction_axes, stage_input_dims[0].size());
    const std::vector<uint64_t> out_dims = StampedEquation::computeReductionOutputDims(stage_input_dims[0], axes, argminmax.squeeze_axes);

    return reductionSemanticFlops(argminmax.op, numelFromDims(out_dims), reductionExtent(stage_input_dims[0], axes));
}

static uint64_t computeMatmulStageFlops(const CompiledMatmul& matmul, const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    const std::vector<uint64_t> out_dims = resolveMatmulOutputDimsFromInputs(matmul, stage_input_dims);
    const uint64_t out_numel = numelFromDims(out_dims);

    const auto& a_dims = stage_input_dims.at(0);
    const uint64_t k = matmul.transpose_lhs ? a_dims.at(0) : a_dims.at(1);

    const bool alpha_dynamic = (matmul.alpha_input_slot != UINT32_MAX);
    const bool beta_dynamic = (matmul.beta_input_slot != UINT32_MAX);

    const bool has_matmul_term = alpha_dynamic || (matmul.alpha != 0.0);
    const bool has_beta_term = (matmul.op == ExprOp::GEMM) && (beta_dynamic || (matmul.beta != 0.0));

    uint64_t total = 0;

    if (has_matmul_term) {
        total = checkedAddU64(total,
                              checkedMulU64(checkedMulU64(out_numel, k, "computeMatmulStageFlops"), 2, "computeMatmulStageFlops"),
                              "computeMatmulStageFlops");

        if (alpha_dynamic || (matmul.alpha != 1.0)) {
            total = checkedAddU64(total, out_numel, "computeMatmulStageFlops");
        }
    }

    if (has_beta_term) {
        if (beta_dynamic || (matmul.beta != 1.0)) {
            total = checkedAddU64(total, out_numel, "computeMatmulStageFlops");
        }
        if (has_matmul_term) {
            total = checkedAddU64(total, out_numel, "computeMatmulStageFlops");
        }
    }

    return total;
}

static uint64_t computeAttentionStageFlops(const CompiledAttention& attention, const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    const std::vector<uint64_t> out_dims = resolveAttentionOutputDimsFromInputs(attention, stage_input_dims);
    const auto& q_dims = stage_input_dims.at(0);
    const auto& k_dims = stage_input_dims.at(1);

    const uint64_t batch = out_dims.at(0);
    const uint64_t query_heads = out_dims.at(1);
    const uint64_t query_len = out_dims.at(2);
    const uint64_t value_dim = out_dims.at(3);
    const uint64_t key_len = k_dims.at(2);
    const uint64_t qk_dim = q_dims.at(3);

    uint64_t scores = checkedMulU64(batch, query_heads, "computeAttentionStageFlops");
    scores = checkedMulU64(scores, query_len, "computeAttentionStageFlops");
    scores = checkedMulU64(scores, key_len, "computeAttentionStageFlops");

    uint64_t qk_flops = checkedMulU64(scores, qk_dim, "computeAttentionStageFlops");
    qk_flops = checkedMulU64(qk_flops, 2, "computeAttentionStageFlops");

    uint64_t pv_flops = checkedMulU64(scores, value_dim, "computeAttentionStageFlops");
    pv_flops = checkedMulU64(pv_flops, 2, "computeAttentionStageFlops");

    // Approximate softmax/masking/scale work separately from the two dense attention products.
    const uint64_t softmax_flops = checkedMulU64(scores, 5, "computeAttentionStageFlops");
    return checkedAddU64(checkedAddU64(qk_flops, pv_flops, "computeAttentionStageFlops"), softmax_flops, "computeAttentionStageFlops");
}

static uint64_t multiplyDimsForFlops(uint64_t value, const std::vector<uint64_t>& dims, size_t begin, const char* where) {
    for (size_t i = begin; i < dims.size(); ++i) {
        value = checkedMulU64(value, dims.at(i), where);
    }
    return value;
}

static uint64_t computeAttentionBackwardStageFlops(const CompiledAttentionBackward& attention_backward,
                                                   const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    std::vector<std::vector<uint64_t>> forward_input_dims{stage_input_dims.at(0), stage_input_dims.at(1), stage_input_dims.at(2)};
    size_t next_optional_idx = 4;  // q, k, v, dO are always present in the backward stage input list.
    if (attention_backward.use_bias) {
        if (stage_input_dims.size() <= next_optional_idx) {
            throw std::runtime_error("AttentionBackward FLOP accounting expected q/k/v/dO/bias input dims for biased attention.");
        }
        forward_input_dims.push_back(stage_input_dims.at(next_optional_idx++));
    }
    if (attention_backward.use_padding_mask) {
        if (stage_input_dims.size() <= next_optional_idx + 1) {
            throw std::runtime_error(
                "AttentionBackward FLOP accounting expected q/k/v/dO plus q/kv sequence length input dims for padding-mask attention.");
        }
        forward_input_dims.push_back(stage_input_dims.at(next_optional_idx++));
        forward_input_dims.push_back(stage_input_dims.at(next_optional_idx++));
    }
    if (attention_backward.use_ragged_offsets) {
        if (stage_input_dims.size() <= next_optional_idx + 1) {
            throw std::runtime_error(
                "AttentionBackward FLOP accounting expected q/k/v/dO plus q/kv ragged offset input dims for ragged attention.");
        }
        forward_input_dims.push_back(stage_input_dims.at(next_optional_idx++));
        forward_input_dims.push_back(stage_input_dims.at(next_optional_idx++));
    }
    if (attention_backward.dropout_probability > 0.0f) {
        if (stage_input_dims.size() <= next_optional_idx + 1) {
            throw std::runtime_error(
                "AttentionBackward FLOP accounting expected q/k/v/dO plus dropout seed/offset input dims for dropout attention.");
        }
        next_optional_idx += 2;
    }
    const uint64_t forward =
        computeAttentionStageFlops(makeForwardAttentionView(attention_backward, attention_backward.dQ_dtype), forward_input_dims);
    // cuDNN backward internally does the reverse softmax path plus three dense-gradient products.
    return checkedMulU64(forward, 4, "computeAttentionBackwardStageFlops");
}

static uint64_t computeConvolutionStageFlops(const CompiledConvolution& convolution,
                                             const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    const std::vector<uint64_t> out_dims = resolveConvolutionOutputDimsFromInputs(convolution, stage_input_dims);
    const auto& filter_dims = stage_input_dims.at(1);

    uint64_t macs = 1;
    macs = multiplyDimsForFlops(macs, out_dims, 0, "computeConvolutionStageFlops");
    macs = checkedMulU64(macs, filter_dims.at(1), "computeConvolutionStageFlops");
    macs = multiplyDimsForFlops(macs, filter_dims, 2, "computeConvolutionStageFlops");

    return checkedMulU64(macs, 2, "computeConvolutionStageFlops");
}

static uint64_t computeConvolutionBackwardStageFlops(const CompiledConvolutionBackward& convolution_backward,
                                                     const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    if (convolution_backward.op == ExprOp::CONV2D_BACKWARD_DATA || convolution_backward.op == ExprOp::CONV3D_BACKWARD_DATA) {
        const auto& filter_dims = stage_input_dims.at(0);
        const auto& grad_dims = stage_input_dims.at(1);

        uint64_t macs = 1;
        macs = multiplyDimsForFlops(macs, grad_dims, 0, "computeConvolutionBackwardStageFlops");
        macs = checkedMulU64(macs, filter_dims.at(1), "computeConvolutionBackwardStageFlops");
        macs = multiplyDimsForFlops(macs, filter_dims, 2, "computeConvolutionBackwardStageFlops");

        return checkedMulU64(macs, 2, "computeConvolutionBackwardStageFlops");
    }

    if (convolution_backward.op == ExprOp::CONV2D_BACKWARD_FILTER || convolution_backward.op == ExprOp::CONV3D_BACKWARD_FILTER) {
        const auto& input_dims = stage_input_dims.at(0);
        const auto& grad_dims = stage_input_dims.at(1);
        const std::vector<uint64_t> filter_dims = resolveConvolutionBackwardOutputDimsFromInputs(convolution_backward, stage_input_dims);

        uint64_t macs = 1;
        macs = checkedMulU64(macs, input_dims.at(0), "computeConvolutionBackwardStageFlops");
        macs = checkedMulU64(macs, grad_dims.at(1), "computeConvolutionBackwardStageFlops");
        macs = multiplyDimsForFlops(macs, grad_dims, 2, "computeConvolutionBackwardStageFlops");
        macs = checkedMulU64(macs, input_dims.at(1), "computeConvolutionBackwardStageFlops");
        macs = multiplyDimsForFlops(macs, filter_dims, 2, "computeConvolutionBackwardStageFlops");

        return checkedMulU64(macs, 2, "computeConvolutionBackwardStageFlops");
    }

    throw std::runtime_error("Unsupported convolution backward op while computing FLOPs.");
}

static uint64_t computeReduceMinMaxBackwardStageFlops(const CompiledReduceMinMaxBackward& backward,
                                                      const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    (void)backward;
    if (stage_input_dims.size() < 2) {
        throw std::runtime_error("ReduceMinMaxBackward stage missing input dims while computing FLOPs.");
    }

    // Reduce-min/max backward compares every original input element with the retained extremum and then
    // conditionally routes the upstream gradient. Count those two per-input-element operations directly;
    // upstream-gradient shaping is no longer represented as a separate helper stage in the optimized
    // backward graph.
    return checkedMulU64(numelFromDims(stage_input_dims[0]), 2, "computeReduceMinMaxBackwardStageFlops");
}

static uint64_t computeStageFlops(const CompiledExecutionStage& stage, const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    switch (stage.kind) {
        case CompiledExecutionStage::Kind::FusedKernel:
            return computeFusedStageFlops(stage.expr, stage_input_dims, stage.outputs);

        case CompiledExecutionStage::Kind::CudaKernel:
            return 0;

        case CompiledExecutionStage::Kind::Reduction:
            if (!stage.reduction)
                throw std::runtime_error("Reduction stage missing payload while computing FLOPs.");
            return computeReductionStageFlops(*stage.reduction, stage_input_dims);

        case CompiledExecutionStage::Kind::ArgMinMax:
            if (!stage.arg_minmax)
                throw std::runtime_error("ArgMinMax stage missing payload while computing FLOPs.");
            return computeArgMinMaxStageFlops(*stage.arg_minmax, stage_input_dims);

        case CompiledExecutionStage::Kind::SegmentedReduction:
            if (!stage.segmented_reduction)
                throw std::runtime_error("SegmentedReduction stage missing payload while computing FLOPs.");
            if (stage_input_dims.empty())
                throw std::runtime_error("SegmentedReduction stage missing input dims while computing FLOPs.");
            return numelFromDims(stage_input_dims[0]);

        case CompiledExecutionStage::Kind::Scan:
            if (stage_input_dims.empty())
                throw std::runtime_error("Scan stage missing input dims while computing FLOPs.");
            return numelFromDims(stage_input_dims[0]);

        case CompiledExecutionStage::Kind::Softmax:
            if (stage_input_dims.empty())
                throw std::runtime_error("Softmax stage missing input dims while computing FLOPs.");
            return numelFromDims(stage_input_dims[0]) * 5;

        case CompiledExecutionStage::Kind::RmsNorm: {
            if (!stage.rms_norm) {
                throw std::runtime_error("RMSNorm stage missing payload while computing FLOPs.");
            }
            if (stage_input_dims.empty()) {
                throw std::runtime_error("RMSNorm stage missing input dims while computing FLOPs.");
            }
            const uint64_t base_flops = checkedMulU64(numelFromDims(stage_input_dims[0]), 6, "computeRmsNormStageFlops");
            if (stage.rms_norm->fused_activation == CudnnRmsNormFusedActivation::SWISH) {
                return checkedAddU64(base_flops, checkedMulU64(numelFromDims(stage_input_dims[0]), 5, "computeRmsNormStageFlops"), "computeRmsNormStageFlops");
            }
            return base_flops;
        }

        case CompiledExecutionStage::Kind::EmbeddingLookup:
            return 0;

        case CompiledExecutionStage::Kind::Matmul:
            if (!stage.matmul)
                throw std::runtime_error("Matmul stage missing payload while computing FLOPs.");
            return computeMatmulStageFlops(*stage.matmul, stage_input_dims);

        case CompiledExecutionStage::Kind::InPlaceRope:
            // RoPE FLOPs are intentionally approximate and are not part of the attention benchmark TFLOP estimate.
            return stage_input_dims.empty() ? 0 : checkedMulU64(numelFromDims(stage_input_dims[0]), 8, "computeInPlaceRopeStageFlops");

        case CompiledExecutionStage::Kind::Attention:
            if (!stage.attention)
                throw std::runtime_error("Attention stage missing payload while computing FLOPs.");
            return computeAttentionStageFlops(*stage.attention, stage_input_dims);

        case CompiledExecutionStage::Kind::AttentionBackward:
            if (!stage.attention_backward)
                throw std::runtime_error("AttentionBackward stage missing payload while computing FLOPs.");
            return computeAttentionBackwardStageFlops(*stage.attention_backward, stage_input_dims);

        case CompiledExecutionStage::Kind::Convolution:
            if (!stage.convolution)
                throw std::runtime_error("Convolution stage missing payload while computing FLOPs.");
            return computeConvolutionStageFlops(*stage.convolution, stage_input_dims);

        case CompiledExecutionStage::Kind::ConvolutionBackward:
            if (!stage.convolution_backward) {
                throw std::runtime_error("ConvolutionBackward stage missing payload while computing FLOPs.");
            }
            return computeConvolutionBackwardStageFlops(*stage.convolution_backward, stage_input_dims);

        case CompiledExecutionStage::Kind::ReduceMinMaxBackward:
            if (!stage.reduce_minmax_backward) {
                throw std::runtime_error("ReduceMinMaxBackward stage missing payload while computing FLOPs.");
            }
            return computeReduceMinMaxBackwardStageFlops(*stage.reduce_minmax_backward, stage_input_dims);

        case CompiledExecutionStage::Kind::ScanMinMaxBackward:
            if (stage_input_dims.empty()) {
                throw std::runtime_error("ScanMinMaxBackward stage missing input dims while computing FLOPs.");
            }
            return checkedMulU64(numelFromDims(stage_input_dims[0]), 4, "computeScanMinMaxBackwardStageFlops");
    }

    throw std::runtime_error("Unknown stage kind while computing FLOPs.");
}

static std::vector<uint64_t> resolveOutputDimsForStageOutput(const CompiledExecutionStage& stage,
                                                             size_t output_idx,
                                                             const std::vector<std::vector<uint64_t>>& stage_input_dims) {
    if (output_idx >= stage.outputs.size()) {
        throw std::runtime_error("resolveOutputDimsForStageOutput output_idx out of range.");
    }

    switch (stage.kind) {
        case CompiledExecutionStage::Kind::Reduction: {
            if (!stage.reduction) {
                throw std::runtime_error("resolveOutputDimsForStageOutput reduction stage missing payload.");
            }
            if (stage_input_dims.empty()) {
                throw std::runtime_error("resolveOutputDimsForStageOutput reduction stage expected at least one input shape.");
            }

            const auto reduction_axes = resolveReductionAxesForInputRank(stage.reduction->reduction_axes, stage_input_dims[0].size());

            return StampedEquation::computeReductionOutputDims(stage_input_dims[0], reduction_axes, stage.reduction->squeeze_axes);
        }

        case CompiledExecutionStage::Kind::ArgMinMax: {
            if (!stage.arg_minmax) {
                throw std::runtime_error("resolveOutputDimsForStageOutput argmin/argmax stage missing payload.");
            }
            if (stage_input_dims.empty()) {
                throw std::runtime_error("resolveOutputDimsForStageOutput argmin/argmax stage expected at least one input shape.");
            }

            const auto reduction_axes = resolveReductionAxesForInputRank(stage.arg_minmax->reduction_axes, stage_input_dims[0].size());

            return StampedEquation::computeReductionOutputDims(stage_input_dims[0], reduction_axes, stage.arg_minmax->squeeze_axes);
        }

        case CompiledExecutionStage::Kind::SegmentedReduction: {
            if (!stage.segmented_reduction) {
                throw std::runtime_error("resolveOutputDimsForStageOutput segmented-reduction stage missing payload.");
            }
            if (stage_input_dims.size() != 2) {
                throw std::runtime_error("resolveOutputDimsForStageOutput segmented-reduction stage expected values and offsets shapes.");
            }
            if (stage_input_dims[0].size() != 1 || stage_input_dims[1].size() != 1 || stage_input_dims[1][0] == 0) {
                throw std::runtime_error("resolveOutputDimsForStageOutput segmented-reduction currently requires rank-1 values and non-empty rank-1 offsets.");
            }
            return std::vector<uint64_t>{stage_input_dims[1][0] - 1};
        }

        case CompiledExecutionStage::Kind::Scan: {
            if (stage_input_dims.empty()) {
                throw std::runtime_error("resolveOutputDimsForStageOutput scan stage expected one input shape.");
            }
            return stage_input_dims[0];
        }

        case CompiledExecutionStage::Kind::Softmax: {
            if (stage_input_dims.empty()) {
                throw std::runtime_error("resolveOutputDimsForStageOutput softmax stage expected one input shape.");
            }
            return stage_input_dims[0];
        }

        case CompiledExecutionStage::Kind::RmsNorm: {
            if (!stage.rms_norm) {
                throw std::runtime_error("resolveOutputDimsForStageOutput RMSNorm stage missing payload.");
            }
            if (stage_input_dims.size() != 2) {
                throw std::runtime_error("resolveOutputDimsForStageOutput RMSNorm stage expected input and scale shapes.");
            }
            return inferRmsNormOutputDims(*stage.rms_norm, stage_input_dims[0], stage_input_dims[1]);
        }

        case CompiledExecutionStage::Kind::EmbeddingLookup: {
            if (!stage.embedding_lookup) {
                throw std::runtime_error("resolveOutputDimsForStageOutput EmbeddingLookup stage missing payload.");
            }
            if (stage_input_dims.size() < 2) {
                throw std::runtime_error("resolveOutputDimsForStageOutput EmbeddingLookup stage expected at least indices and weights shapes.");
            }
            return inferEmbeddingLookupOutputDims(stage_input_dims[0], stage_input_dims[1]);
        }

        case CompiledExecutionStage::Kind::ReduceMinMaxBackward: {
            if (stage_input_dims.empty()) {
                throw std::runtime_error(
                    "resolveOutputDimsForStageOutput reduce-min/max-backward stage expected at least one input shape.");
            }
            return stage_input_dims[0];
        }

        case CompiledExecutionStage::Kind::ScanMinMaxBackward: {
            if (stage_input_dims.empty()) {
                throw std::runtime_error(
                    "resolveOutputDimsForStageOutput scan-min/max-backward stage expected at least one input shape.");
            }
            return stage_input_dims[0];
        }

        case CompiledExecutionStage::Kind::Matmul: {
            if (!stage.matmul) {
                throw std::runtime_error("resolveOutputDimsForStageOutput matmul stage missing payload.");
            }
            const std::vector<uint64_t> matrix_dims = resolveMatmulOutputDimsFromInputs(*stage.matmul, stage_input_dims);
            if (output_idx == 0) {
                return matrix_dims;
            }
            if (stage.matmul->bgrad_output_dtype.has_value()) {
                if (matrix_dims.size() != 2) {
                    throw std::runtime_error("Matmul bias-gradient output requires a rank-2 matrix output.");
                }
                return std::vector<uint64_t>{matrix_dims[1]};
            }
            throw std::runtime_error("Matmul stage requested secondary output but has no bias-gradient epilogue.");
        }

        case CompiledExecutionStage::Kind::InPlaceRope: {
            if (!stage.in_place_rope) {
                throw std::runtime_error("resolveOutputDimsForStageOutput in-place RoPE stage missing payload.");
            }
            if (output_idx >= stage.in_place_rope->tensors.size()) {
                throw std::runtime_error("resolveOutputDimsForStageOutput in-place RoPE output index out of range.");
            }
            return stage.in_place_rope->tensors[output_idx].logical_dims;
        }

        case CompiledExecutionStage::Kind::Attention: {
            if (!stage.attention) {
                throw std::runtime_error("resolveOutputDimsForStageOutput attention stage missing payload.");
            }
            return resolveAttentionOutputDimsFromInputs(*stage.attention, stage_input_dims);
        }

        case CompiledExecutionStage::Kind::AttentionBackward: {
            if (!stage.attention_backward) {
                throw std::runtime_error("resolveOutputDimsForStageOutput attention-backward stage missing payload.");
            }
            if (stage.outputs[output_idx].local_node_idx >= stage.expr.nodes.size()) {
                throw std::runtime_error("resolveOutputDimsForStageOutput attention-backward output node index out of range.");
            }
            return resolveAttentionBackwardOutputDimsFromInputs(
                *stage.attention_backward, stage_input_dims, stage.expr.nodes[stage.outputs[output_idx].local_node_idx].op);
        }

        case CompiledExecutionStage::Kind::Convolution: {
            if (!stage.convolution) {
                throw std::runtime_error("resolveOutputDimsForStageOutput convolution stage missing payload.");
            }
            return resolveConvolutionOutputDimsFromInputs(*stage.convolution, stage_input_dims);
        }

        case CompiledExecutionStage::Kind::ConvolutionBackward: {
            if (!stage.convolution_backward) {
                throw std::runtime_error("resolveOutputDimsForStageOutput convolution-backward stage missing payload.");
            }
            return resolveConvolutionBackwardOutputDimsFromInputs(*stage.convolution_backward, stage_input_dims);
        }

        case CompiledExecutionStage::Kind::CudaKernel: {
            if (!stage.cuda_kernel_expression) {
                throw std::runtime_error("resolveOutputDimsForStageOutput CUDA kernel stage missing expression spec.");
            }
            std::unordered_map<std::string, std::vector<uint64_t>> input_shapes;
            const auto& input_specs = stage.cuda_kernel_expression->inputs();
            if (input_specs.size() != stage_input_dims.size()) {
                throw std::runtime_error("resolveOutputDimsForStageOutput CUDA kernel input shape count mismatch.");
            }
            for (size_t i = 0; i < input_specs.size(); ++i) {
                input_shapes.emplace(input_specs[i].name, stage_input_dims[i]);
            }
            const std::vector<std::vector<uint64_t>> output_shapes =
                stage.cuda_kernel_expression->inferOutputShapesFromInputShapes(input_shapes);
            if (stage.outputs[output_idx].local_node_idx >= stage.expr.nodes.size()) {
                throw std::runtime_error("resolveOutputDimsForStageOutput CUDA kernel output node index out of range.");
            }
            const ExprNode& output_node = stage.expr.nodes[stage.outputs[output_idx].local_node_idx];
            if (output_node.cuda_kernel_output_index >= output_shapes.size()) {
                throw std::runtime_error("resolveOutputDimsForStageOutput CUDA kernel output spec index out of range.");
            }
            return output_shapes[output_node.cuda_kernel_output_index];
        }

        case CompiledExecutionStage::Kind::FusedKernel:
            break;
    }

    const uint32_t local_node_idx = stage.outputs[output_idx].local_node_idx;
    std::unordered_set<uint32_t> reachable_nodes;
    collectReachableLocalNodes(stage.expr, local_node_idx, reachable_nodes);

    const auto safe_infer_node_dims = [&]() {
        try {
            return inferFusedStageNodeDimsForReachable(stage.expr, stage_input_dims, reachable_nodes);
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "resolveOutputDimsForStageOutput failed"
                << " stage_kind=" << static_cast<int>(stage.kind) << " output_idx=" << output_idx
                << " output_name=" << stage.outputs[output_idx].name << " local_node_idx=" << local_node_idx << " error=" << e.what();
            throw std::runtime_error(oss.str());
        }
    };

    const auto node_dims = safe_infer_node_dims();

    if (local_node_idx >= node_dims.size()) {
        throw std::runtime_error("resolveOutputDimsForStageOutput local_node_idx out of range.");
    }

    std::vector<uint64_t> output_dims = node_dims[local_node_idx];
    if (stage.kind == CompiledExecutionStage::Kind::FusedKernel &&
        stage.outputs[output_idx].materialized_layout == MaterializedTensorLayout::Transposed) {
        if (output_dims.size() < 2) {
            throw std::runtime_error("Transposed fused materialization requires a rank >= 2 logical output.");
        }
        std::swap(output_dims[output_dims.size() - 2], output_dims[output_dims.size() - 1]);
    }
    return output_dims;
}

static std::vector<uint64_t> resolveOutputDimsForStageOutput(const CompiledExecutionStage& stage,
                                                             size_t output_idx,
                                                             const std::vector<RuntimeInputValue>& stage_inputs) {
    std::vector<std::vector<uint64_t>> stage_input_dims;
    stage_input_dims.reserve(stage_inputs.size());
    for (const RuntimeInputValue& input : stage_inputs) {
        stage_input_dims.push_back(runtimeInputDims(input));
    }
    return resolveOutputDimsForStageOutput(stage, output_idx, stage_input_dims);
}

static std::vector<uint64_t> logicalDimsForBroadcastComparison(const CompiledExecutionStage& stage,
                                                               size_t output_idx,
                                                               std::vector<uint64_t> physical_output_dims) {
    if (stage.outputs[output_idx].materialized_layout == MaterializedTensorLayout::Transposed) {
        if (physical_output_dims.size() < 2) {
            throw std::runtime_error("Transposed fused materialization requires a rank >= 2 output.");
        }
        std::swap(physical_output_dims[physical_output_dims.size() - 2], physical_output_dims[physical_output_dims.size() - 1]);
    }
    return physical_output_dims;
}

struct ResolvedBroadcastGroup {
    SpecializedBroadcastGroup specialized;
};

static bool expressionHasIndexAwareOps(const PhysicalExpression& expr) {
    return std::any_of(expr.nodes.begin(), expr.nodes.end(), [](const ExprNode& node) {
        return node.op == ExprOp::ROPE || node.op == ExprOp::TRANSPOSE || node.op == ExprOp::TAKE_ALONG_AXIS;
    });
}

static bool stageHasShapeOnlyOps(const CompiledExecutionStage& stage) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        return false;
    }
    for (const ExprNode& node : stage.expr.nodes) {
        if (node.op == ExprOp::RESHAPE || node.op == ExprOp::STRIDED_VIEW || node.op == ExprOp::UNSQUEEZE || node.op == ExprOp::SQUEEZE) {
            return true;
        }
    }
    return false;
}
static TensorPlacement pickStageOutputPlacement(const std::vector<RuntimeInputValue>& stage_inputs,
                                                const std::unordered_map<uint32_t, RuntimeInputValue>& available_values) {
    for (const RuntimeInputValue& input : stage_inputs) {
        std::optional<TensorPlacement> placement = runtimeInputPlacementOrNull(input);
        if (placement.has_value()) {
            return placement.value();
        }
    }
    if (!available_values.empty()) {
        for (const auto& [value_id, value] : available_values) {
            (void)value_id;
            std::optional<TensorPlacement> placement = runtimeInputPlacementOrNull(value);
            if (placement.has_value()) {
                return placement.value();
            }
        }
    }
    throw std::runtime_error("Unable to infer output placement for fused stage with no available tensors.");
}

static void mergeEffectiveInputDimsMaps(std::unordered_map<uint32_t, std::set<std::vector<uint64_t>>>& dst,
                                        const std::unordered_map<uint32_t, std::set<std::vector<uint64_t>>>& src) {
    for (const auto& [slot, dims_set] : src) {
        auto& out = dst[slot];
        out.insert(dims_set.begin(), dims_set.end());
    }
}

static std::unordered_map<std::string, std::vector<uint64_t>> defaultBackwardRequestedOutputShapes(
    const std::optional<BackwardEquationConfig>& backward_config,
    const std::vector<NamedInput>& root_inputs,
    const std::unordered_map<uint32_t, RuntimeInputValue>& root_values,
    const std::unordered_map<std::string, std::vector<uint64_t>>& requested_output_shapes) {
    std::unordered_map<std::string, std::vector<uint64_t>> effective = requested_output_shapes;

    if (!backward_config.has_value()) {
        return effective;
    }

    std::unordered_map<std::string, uint32_t> root_slot_by_name;
    root_slot_by_name.reserve(root_inputs.size());
    for (const NamedInput& input : root_inputs) {
        root_slot_by_name.emplace(input.name, input.slot);
    }

    for (const std::string& wrt_name : backward_config->wrt_names) {
        const std::string output_name = wrt_name + "_grad";
        if (effective.contains(output_name)) {
            continue;
        }

        auto slot_it = root_slot_by_name.find(wrt_name);
        if (slot_it == root_slot_by_name.end()) {
            continue;
        }

        auto value_it = root_values.find(slot_it->second);
        if (value_it == root_values.end()) {
            continue;
        }

        effective.emplace(output_name, runtimeInputDims(value_it->second));
    }

    return effective;
}

static std::vector<uint64_t> mapEffectiveDimsThroughDenseReshapeAlias(const std::vector<uint64_t>& source_dims,
                                                                             const std::vector<uint64_t>& target_dims,
                                                                             const std::vector<uint64_t>& effective_source_dims) {
    if (effective_source_dims == source_dims) {
        return target_dims;
    }

    if (effective_source_dims.size() > source_dims.size()) {
        std::ostringstream oss;
        oss << "Cannot map broadcast shape through dense reshape alias: effective rank exceeds source rank. "
            << "source_dims=" << dimsToString(source_dims) << ", effective_dims=" << dimsToString(effective_source_dims)
            << ", target_dims=" << dimsToString(target_dims);
        throw std::runtime_error(oss.str());
    }

    std::vector<uint64_t> padded_effective_source_dims(source_dims.size(), 1ULL);
    std::copy(effective_source_dims.begin(),
              effective_source_dims.end(),
              padded_effective_source_dims.begin() +
                  static_cast<std::ptrdiff_t>(source_dims.size() - effective_source_dims.size()));
    for (size_t axis = 0; axis < source_dims.size(); ++axis) {
        const uint64_t effective_dim = padded_effective_source_dims[axis];
        if (effective_dim != 1ULL && effective_dim != source_dims[axis]) {
            std::ostringstream oss;
            oss << "Cannot map non-broadcast-compatible effective shape through dense reshape alias. "
                << "source_dims=" << dimsToString(source_dims) << ", effective_dims=" << dimsToString(effective_source_dims)
                << ", padded_effective_dims=" << dimsToString(padded_effective_source_dims)
                << ", target_dims=" << dimsToString(target_dims);
            throw std::runtime_error(oss.str());
        }
    }

    if (computeDimsNumel(source_dims) != computeDimsNumel(target_dims)) {
        std::ostringstream oss;
        oss << "Cannot map broadcast shape through dense reshape alias with mismatched element counts. "
            << "source_dims=" << dimsToString(source_dims) << ", target_dims=" << dimsToString(target_dims);
        throw std::runtime_error(oss.str());
    }

    std::vector<uint64_t> mapped;
    mapped.reserve(target_dims.size());

    size_t src_begin = 0;
    size_t dst_begin = 0;
    while (src_begin < source_dims.size() || dst_begin < target_dims.size()) {
        if (src_begin >= source_dims.size() || dst_begin >= target_dims.size()) {
            std::ostringstream oss;
            oss << "Cannot map broadcast shape through dense reshape alias with unmatched axes. "
                << "source_dims=" << dimsToString(source_dims) << ", effective_dims=" << dimsToString(effective_source_dims)
                << ", target_dims=" << dimsToString(target_dims);
            throw std::runtime_error(oss.str());
        }

        uint64_t src_product = source_dims[src_begin];
        uint64_t dst_product = target_dims[dst_begin];
        size_t src_end = src_begin + 1;
        size_t dst_end = dst_begin + 1;

        while (src_product != dst_product) {
            if (src_product < dst_product) {
                if (src_end >= source_dims.size()) {
                    break;
                }
                if (src_product > std::numeric_limits<uint64_t>::max() / source_dims[src_end]) {
                    throw std::runtime_error("Source reshape partition product overflows uint64_t.");
                }
                src_product *= source_dims[src_end++];
            } else {
                if (dst_end >= target_dims.size()) {
                    break;
                }
                if (dst_product > std::numeric_limits<uint64_t>::max() / target_dims[dst_end]) {
                    throw std::runtime_error("Target reshape partition product overflows uint64_t.");
                }
                dst_product *= target_dims[dst_end++];
            }
        }

        if (src_product != dst_product) {
            std::ostringstream oss;
            oss << "Cannot partition dense reshape alias axes. source_dims=" << dimsToString(source_dims)
                << ", effective_dims=" << dimsToString(effective_source_dims) << ", target_dims=" << dimsToString(target_dims);
            throw std::runtime_error(oss.str());
        }

        bool source_chunk_is_full = true;
        bool source_chunk_is_scalar_broadcast = true;
        for (size_t axis = src_begin; axis < src_end; ++axis) {
            if (padded_effective_source_dims[axis] != source_dims[axis]) {
                source_chunk_is_full = false;
            }
            if (padded_effective_source_dims[axis] != 1ULL) {
                source_chunk_is_scalar_broadcast = false;
            }
        }

        if (source_chunk_is_full) {
            mapped.insert(mapped.end(), target_dims.begin() + static_cast<std::ptrdiff_t>(dst_begin),
                          target_dims.begin() + static_cast<std::ptrdiff_t>(dst_end));
        } else if (source_chunk_is_scalar_broadcast) {
            mapped.insert(mapped.end(), dst_end - dst_begin, 1ULL);
        } else if ((src_end - src_begin) == (dst_end - dst_begin)) {
            bool axes_match = true;
            for (size_t src_axis = src_begin, dst_axis = dst_begin; src_axis < src_end; ++src_axis, ++dst_axis) {
                if (source_dims[src_axis] != target_dims[dst_axis]) {
                    axes_match = false;
                    break;
                }
            }
            if (axes_match) {
                mapped.insert(mapped.end(), padded_effective_source_dims.begin() + static_cast<std::ptrdiff_t>(src_begin),
                              padded_effective_source_dims.begin() + static_cast<std::ptrdiff_t>(src_end));
            } else {
                std::ostringstream oss;
                oss << "Cannot represent partial broadcast through dense reshape alias. source_dims=" << dimsToString(source_dims)
                    << ", effective_dims=" << dimsToString(effective_source_dims) << ", target_dims=" << dimsToString(target_dims);
                throw std::runtime_error(oss.str());
            }
        } else {
            std::ostringstream oss;
            oss << "Cannot represent partial broadcast through dense reshape alias. source_dims=" << dimsToString(source_dims)
                << ", effective_dims=" << dimsToString(effective_source_dims) << ", target_dims=" << dimsToString(target_dims);
            throw std::runtime_error(oss.str());
        }

        src_begin = src_end;
        dst_begin = dst_end;
    }

    if (mapped.size() != target_dims.size()) {
        throw std::runtime_error("Dense reshape broadcast mapping produced wrong rank.");
    }
    return mapped;
}

static std::unordered_map<uint32_t, std::set<std::vector<uint64_t>>> collectEffectiveInputDimsForNode(
    const PhysicalExpression& expr, const std::vector<std::vector<uint64_t>>& node_dims, uint32_t node_idx) {
    if (node_idx >= expr.nodes.size() || node_idx >= node_dims.size()) {
        throw std::runtime_error("collectEffectiveInputDimsForNode node index out of range.");
    }

    const ExprNode& node = expr.nodes[node_idx];
    switch (node.op) {
        case ExprOp::INPUT: {
            return {{node.input_slot, {node_dims[node_idx]}}};
        }
        case ExprOp::RUNTIME_SCALAR:
        case ExprOp::TENSOR_RUNTIME_SCALAR:
        case ExprOp::SCALAR_FP:
            return {};
        case ExprOp::FILL:
            return {};
        case ExprOp::RAGGED_VALUEWISE_EXTENT:
            // Offsets are launch metadata, not a broadcast operand.
            return collectEffectiveInputDimsForNode(expr, node_dims, node.lhs);
        case ExprOp::TRANSPOSE: {
            auto result = collectEffectiveInputDimsForNode(expr, node_dims, node.lhs);
            // A non-terminal transpose is evaluated by the index-aware fused emitter.
            // For broadcast grouping, treat the child tensor inputs as if they are
            // consumed in the transposed node's output domain. The emitter itself
            // remaps the final output index back to the pre-transpose source index
            // before loading those inputs.
            for (auto& [slot, dims_set] : result) {
                dims_set.clear();
                dims_set.insert(node_dims[node_idx]);
            }
            return result;
        }
        case ExprOp::STRIDED_VIEW_BACKWARD: {
            auto result = collectEffectiveInputDimsForNode(expr, node_dims, node.lhs);
            // strided_view_backward is an index-aware gather from the view-gradient input
            // into the source-gradient output domain. The input tensor's physical rank
            // (for packed-QKV this is typically [B,S,H,D]) is not broadcast against the
            // output tensor rank (typically [B*S,QKV]). Treat the referenced input as
            // effectively consumed in the output domain so broadcast planning keeps this
            // as a flat fused kernel instead of trying to build invalid rank-4 -> rank-2
            // broadcast strides.
            for (auto& [slot, dims_set] : result) {
                dims_set.clear();
                dims_set.insert(node_dims[node_idx]);
            }
            return result;
        }
        case ExprOp::RESHAPE: {
            auto result = collectEffectiveInputDimsForNode(expr, node_dims, node.lhs);
            for (auto& [slot, dims_set] : result) {
                std::set<std::vector<uint64_t>> mapped_dims;
                for (const auto& dims : dims_set) {
                    mapped_dims.insert(mapEffectiveDimsThroughDenseReshapeAlias(node_dims[node.lhs], node_dims[node_idx], dims));
                }
                dims_set = std::move(mapped_dims);
            }
            return result;
        }
        case ExprOp::STRIDED_VIEW:
        case ExprOp::UNSQUEEZE:
        case ExprOp::SQUEEZE: {
            auto result = collectEffectiveInputDimsForNode(expr, node_dims, node.lhs);
            for (auto& [slot, dims_set] : result) {
                dims_set.clear();
                dims_set.insert(node_dims[node_idx]);
            }
            return result;
        }
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
        case ExprOp::ROPE:
        case ExprOp::LOGICAL_NOT:
        case ExprOp::CAST:
        case ExprOp::REDUCE_SUM:
        case ExprOp::REDUCE_PROD:
        case ExprOp::REDUCE_MIN:
        case ExprOp::REDUCE_MAX:
        case ExprOp::REDUCE_ARGMIN:
        case ExprOp::REDUCE_ARGMAX:
        case ExprOp::REDUCE_AVG:
        case ExprOp::REDUCE_NORM1:
        case ExprOp::REDUCE_NORM2:
            return collectEffectiveInputDimsForNode(expr, node_dims, node.lhs);
        case ExprOp::TAKE_ALONG_AXIS: {
            auto lhs_map = collectEffectiveInputDimsForNode(expr, node_dims, node.lhs);
            auto rhs_map = collectEffectiveInputDimsForNode(expr, node_dims, node.rhs);
            // take_along_axis is an index-aware gather. Both the values tensor
            // and the indices tensor are consumed in the gathered output domain
            // rather than in the values input's physical shape; otherwise
            // broadcast grouping tries to compare the pre-gather values shape
            // against the output/indices shape and rejects valid gathers.
            for (auto& [slot, dims_set] : lhs_map) {
                dims_set.clear();
                dims_set.insert(node_dims[node_idx]);
            }
            for (auto& [slot, dims_set] : rhs_map) {
                dims_set.clear();
                dims_set.insert(node_dims[node_idx]);
            }
            mergeEffectiveInputDimsMaps(lhs_map, rhs_map);
            return lhs_map;
        }
        case ExprOp::ADD:
        case ExprOp::SUB:
        case ExprOp::MUL:
        case ExprOp::DIV:
        case ExprOp::POW:
        case ExprOp::MIN:
        case ExprOp::MAX:
        case ExprOp::MIN_GRAD_LEFT:
        case ExprOp::MIN_GRAD_RIGHT:
        case ExprOp::MAX_GRAD_LEFT:
        case ExprOp::MAX_GRAD_RIGHT:
        case ExprOp::EQUAL:
        case ExprOp::NOT_EQUAL:
        case ExprOp::LESS:
        case ExprOp::LESS_EQUAL:
        case ExprOp::GREATER:
        case ExprOp::GREATER_EQUAL:
        case ExprOp::LOGICAL_AND:
        case ExprOp::LOGICAL_OR: {
            auto lhs_map = collectEffectiveInputDimsForNode(expr, node_dims, node.lhs);
            auto rhs_map = collectEffectiveInputDimsForNode(expr, node_dims, node.rhs);
            mergeEffectiveInputDimsMaps(lhs_map, rhs_map);
            return lhs_map;
        }
        case ExprOp::WHERE: {
            auto cond_map = collectEffectiveInputDimsForNode(expr, node_dims, node.lhs);
            auto true_map = collectEffectiveInputDimsForNode(expr, node_dims, node.rhs);
            auto false_map = collectEffectiveInputDimsForNode(expr, node_dims, node.aux);
            mergeEffectiveInputDimsMaps(cond_map, true_map);
            mergeEffectiveInputDimsMaps(cond_map, false_map);
            return cond_map;
        }
        default:
            throw std::runtime_error("collectEffectiveInputDimsForNode encountered unknown ExprOp.");
    }
}

static bool fusedStageRequiresBroadcastLaunch(const CompiledExecutionStage& stage,
                                              const std::vector<RuntimeInputValue>& stage_inputs,
                                              const std::unordered_map<std::string, std::vector<uint64_t>>& requested_output_shapes,
                                              bool trust_requested_output_shapes,
                                              std::vector<uint64_t>& resolved_output_dims) {
    resolved_output_dims.clear();

    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        return false;
    }

    if (expressionHasIndexAwareOps(stage.expr)) {
        if (!stage.outputs.empty()) {
            resolved_output_dims = resolveOutputDimsForStageOutput(stage, 0, stage_inputs);
        }
        return true;
    }

    if (stageHasShapeOnlyOps(stage)) {
        if (!stage.outputs.empty()) {
            resolved_output_dims = resolveOutputDimsForStageOutput(stage, 0, stage_inputs);
        }
        return true;
    }

    if (stage_inputs.empty()) {
        if (!stage.outputs.empty()) {
            resolved_output_dims = resolveOutputDimsForStageOutput(stage, 0, stage_inputs);
            auto requested_it = requested_output_shapes.find(stage.outputs[0].name);
            if (requested_it != requested_output_shapes.end() && !requested_it->second.empty()) {
                resolved_output_dims = requested_it->second;
            }
        }
        return false;
    }

    const bool requires_strided_view_broadcast = anyRuntimeInputIsNonDenseTensorView(stage_inputs);

    bool have_common_output_dims = false;
    std::vector<uint64_t> common_output_dims;
    for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
        std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, output_idx, stage_inputs);
        auto requested_it = requested_output_shapes.find(stage.outputs[output_idx].name);
        if (requested_it != requested_output_shapes.end() && !requested_it->second.empty()) {
            if (!trust_requested_output_shapes) {
                verifyRequestedOutputLayout(requested_it->second, output_dims);
            }
            output_dims = requested_it->second;
        }

        output_dims = logicalDimsForBroadcastComparison(stage, output_idx, std::move(output_dims));

        if (!have_common_output_dims) {
            common_output_dims = output_dims;
            have_common_output_dims = true;
        } else if (output_dims != common_output_dims) {
            return true;
        }
    }

    std::vector<std::vector<uint64_t>> stage_input_dims;
    stage_input_dims.reserve(stage_inputs.size());
    for (const RuntimeInputValue& input : stage_inputs) {
        stage_input_dims.push_back(runtimeInputDims(input));
    }

    std::unordered_map<uint32_t, std::set<std::vector<uint64_t>>> effective_dims_by_slot;
    for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
        std::unordered_set<uint32_t> reachable_nodes;
        collectReachableLocalNodes(stage.expr, stage.outputs[output_idx].local_node_idx, reachable_nodes);

        const auto node_dims = inferFusedStageNodeDimsForReachable(stage.expr, stage_input_dims, reachable_nodes);

        auto per_output = collectEffectiveInputDimsForNode(stage.expr, node_dims, stage.outputs[output_idx].local_node_idx);
        mergeEffectiveInputDimsMaps(effective_dims_by_slot, per_output);
    }

    bool requires_broadcast = false;
    for (const auto& [slot, dims_set] : effective_dims_by_slot) {
        for (const auto& dims : dims_set) {
            if (dims.empty()) {
                continue;
            }
            if (dims != common_output_dims) {
                requires_broadcast = true;
                break;
            }
        }
        if (requires_broadcast) {
            break;
        }
    }

    if (have_common_output_dims) {
        resolved_output_dims = common_output_dims;
    }

    return requires_broadcast || requires_strided_view_broadcast;
}

static std::vector<uint64_t> computeInputPackedStridesForBroadcast(const std::vector<uint64_t>& input_dims,
                                                                   const std::vector<uint64_t>& output_dims) {
    if (input_dims.size() > output_dims.size()) {
        throw std::runtime_error("Input rank exceeds broadcast output rank.");
    }

    const size_t rank = output_dims.size();

    std::vector<uint64_t> padded_dims(rank, 1ULL);
    std::copy(input_dims.begin(), input_dims.end(), padded_dims.begin() + (rank - input_dims.size()));

    std::vector<uint64_t> packed_strides(rank, 1ULL);
    if (rank > 0) {
        packed_strides[rank - 1] = 1ULL;
        for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; --i) {
            packed_strides[static_cast<size_t>(i)] = packed_strides[static_cast<size_t>(i) + 1] * padded_dims[static_cast<size_t>(i) + 1];
        }
    }

    std::vector<uint64_t> result(rank, 0ULL);
    for (size_t axis = 0; axis < rank; ++axis) {
        const uint64_t in_dim = padded_dims[axis];
        const uint64_t out_dim = output_dims[axis];

        if (in_dim == out_dim) {
            result[axis] = packed_strides[axis];
        } else if (in_dim == 1ULL) {
            result[axis] = 0ULL;
        } else {
            std::ostringstream oss;
            oss << "Input dimensions are not broadcast-compatible with output dimensions. "
                << "axis=" << axis << ", input_dims=" << dimsToString(input_dims) << ", padded_input_dims=" << dimsToString(padded_dims)
                << ", output_dims=" << dimsToString(output_dims) << ", conflicting_in_dim=" << in_dim
                << ", conflicting_out_dim=" << out_dim;
            throw std::runtime_error(oss.str());
        }
    }

    return result;
}

static std::vector<uint64_t> computeInputStridesForBroadcastFromVisibleLayout(
    const std::vector<uint64_t>& visible_dims,
    const std::vector<uint64_t>& visible_strides,
    const std::vector<uint64_t>& output_dims) {
    if (visible_dims.size() != visible_strides.size()) {
        throw std::runtime_error("Visible tensor dimensions and strides must have the same rank.");
    }
    if (visible_dims.size() > output_dims.size()) {
        throw std::runtime_error("Input rank exceeds broadcast output rank.");
    }

    const size_t rank = output_dims.size();

    std::vector<uint64_t> padded_dims(rank, 1ULL);
    std::vector<uint64_t> padded_strides(rank, 0ULL);
    const size_t axis_offset = rank - visible_dims.size();
    for (size_t i = 0; i < visible_dims.size(); ++i) {
        padded_dims[axis_offset + i] = visible_dims[i];
        padded_strides[axis_offset + i] = visible_strides[i];
    }

    std::vector<uint64_t> result(rank, 0ULL);
    for (size_t axis = 0; axis < rank; ++axis) {
        const uint64_t in_dim = padded_dims[axis];
        const uint64_t out_dim = output_dims[axis];

        if (in_dim == out_dim) {
            result[axis] = padded_strides[axis];
        } else if (in_dim == 1ULL) {
            result[axis] = 0ULL;
        } else {
            std::ostringstream oss;
            oss << "Input dimensions are not broadcast-compatible with output dimensions. "
                << "axis=" << axis << ", input_dims=" << dimsToString(visible_dims)
                << ", padded_input_dims=" << dimsToString(padded_dims) << ", output_dims=" << dimsToString(output_dims)
                << ", conflicting_in_dim=" << in_dim << ", conflicting_out_dim=" << out_dim;
            throw std::runtime_error(oss.str());
        }
    }

    return result;
}

static std::vector<uint64_t> computeRuntimeInputStridesForBroadcast(const RuntimeInputValue& input,
                                                                    const std::vector<uint64_t>& effective_dims,
                                                                    const std::vector<uint64_t>& output_dims) {
    if (!runtimeInputIsNonDenseTensorView(input)) {
        return computeInputPackedStridesForBroadcast(effective_dims, output_dims);
    }

    const Tensor& tensor = runtimeInputTensor(input);
    const std::vector<uint64_t> visible_dims = tensor.getDimensions();
    if (effective_dims != visible_dims) {
        throw std::runtime_error(
            "Fused kernels cannot consume a non-dense tensor view through an additional shape-changing expression yet.");
    }

    return computeInputStridesForBroadcastFromVisibleLayout(visible_dims, tensor.getStridesElements(), output_dims);
}

static std::vector<ResolvedBroadcastGroup> buildResolvedBroadcastGroups(const CompiledExecutionStage& stage,
                                                                        const std::vector<RuntimeInputValue>& stage_inputs) {
    if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
        throw std::runtime_error("buildResolvedBroadcastGroups expects a fused-kernel stage.");
    }
    if (stage_inputs.size() != stage.input_value_ids.size()) {
        throw std::runtime_error("buildResolvedBroadcastGroups stage input count mismatch.");
    }

    std::vector<std::vector<uint64_t>> stage_input_dims;
    stage_input_dims.reserve(stage_inputs.size());
    for (const RuntimeInputValue& input : stage_inputs) {
        stage_input_dims.push_back(runtimeInputDims(input));
    }

    std::map<std::vector<uint64_t>, std::vector<uint32_t>> outputs_by_dims;
    for (uint32_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
        std::vector<uint64_t> physical_output_dims = resolveOutputDimsForStageOutput(stage, out_idx, stage_input_dims);
        std::vector<uint64_t> logical_output_dims = logicalDimsForBroadcastComparison(stage, out_idx, std::move(physical_output_dims));
        outputs_by_dims[logical_output_dims].push_back(out_idx);
    }

    auto effective_dims_maps_equal = [](const std::unordered_map<uint32_t, std::set<std::vector<uint64_t>>>& a,
                                        const std::unordered_map<uint32_t, std::set<std::vector<uint64_t>>>& b) {
        if (a.size() != b.size()) {
            return false;
        }

        for (const auto& [slot, dims_set_a] : a) {
            auto it = b.find(slot);
            if (it == b.end()) {
                return false;
            }
            if (dims_set_a != it->second) {
                return false;
            }
        }

        return true;
    };

    std::vector<ResolvedBroadcastGroup> groups;

    for (const auto& [output_dims, output_indices] : outputs_by_dims) {
        struct OutputInfo {
            uint32_t out_idx;
            std::vector<std::vector<uint64_t>> node_dims;
            std::unordered_map<uint32_t, std::set<std::vector<uint64_t>>> effective_dims_by_slot;
            std::unordered_set<uint32_t> used_slots_set;
            std::unordered_set<uint32_t> broadcast_offset_slots_set;
        };

        std::vector<OutputInfo> infos;
        infos.reserve(output_indices.size());

        for (uint32_t out_idx : output_indices) {
            std::unordered_set<uint32_t> reachable_nodes;
            collectReachableLocalNodes(stage.expr, stage.outputs[out_idx].local_node_idx, reachable_nodes);
            const auto node_dims = inferFusedStageNodeDimsForReachable(stage.expr, stage_input_dims, reachable_nodes);

            OutputInfo info;
            info.out_idx = out_idx;
            info.node_dims = node_dims;
            collectReferencedLocalInputSlots(stage.expr, stage.outputs[out_idx].local_node_idx, info.used_slots_set);
            collectBroadcastOffsetLocalInputSlots(stage.expr, stage.outputs[out_idx].local_node_idx, info.broadcast_offset_slots_set);
            info.effective_dims_by_slot = collectEffectiveInputDimsForNode(stage.expr, node_dims, stage.outputs[out_idx].local_node_idx);

            infos.push_back(std::move(info));
        }

        std::vector<std::vector<uint32_t>> compatible_subgroups;
        std::vector<std::unordered_map<uint32_t, std::set<std::vector<uint64_t>>>> subgroup_effective_dims;
        std::vector<std::unordered_set<uint32_t>> subgroup_used_slots;
        std::vector<std::unordered_set<uint32_t>> subgroup_broadcast_offset_slots;

        for (const auto& info : infos) {
            bool placed = false;
            for (size_t group_i = 0; group_i < compatible_subgroups.size(); ++group_i) {
                if (effective_dims_maps_equal(subgroup_effective_dims[group_i], info.effective_dims_by_slot)) {
                    compatible_subgroups[group_i].push_back(info.out_idx);
                    subgroup_used_slots[group_i].insert(info.used_slots_set.begin(), info.used_slots_set.end());
                    subgroup_broadcast_offset_slots[group_i].insert(info.broadcast_offset_slots_set.begin(), info.broadcast_offset_slots_set.end());
                    placed = true;
                    break;
                }
            }

            if (!placed) {
                compatible_subgroups.push_back({info.out_idx});
                subgroup_effective_dims.push_back(info.effective_dims_by_slot);
                subgroup_used_slots.push_back(info.used_slots_set);
                subgroup_broadcast_offset_slots.push_back(info.broadcast_offset_slots_set);
            }
        }

        for (size_t subgroup_i = 0; subgroup_i < compatible_subgroups.size(); ++subgroup_i) {
            const auto& subgroup_output_indices = compatible_subgroups[subgroup_i];
            const auto& effective_dims_by_slot = subgroup_effective_dims[subgroup_i];
            const auto& used_slots_set = subgroup_used_slots[subgroup_i];
            const auto& broadcast_offset_slots_set = subgroup_broadcast_offset_slots[subgroup_i];

            std::vector<uint32_t> used_input_slots(used_slots_set.begin(), used_slots_set.end());
            std::sort(used_input_slots.begin(), used_input_slots.end());

            SpecializedBroadcastGroup specialized;
            specialized.output_indices = subgroup_output_indices;
            specialized.output_dims = output_dims;
            specialized.numel = product(output_dims);
            specialized.used_input_slots = used_input_slots;
            specialized.node_dims.assign(stage.expr.nodes.size(), {});
            for (const OutputInfo& info : infos) {
                if (std::find(subgroup_output_indices.begin(), subgroup_output_indices.end(), info.out_idx) == subgroup_output_indices.end()) {
                    continue;
                }
                for (size_t node_i = 0; node_i < info.node_dims.size(); ++node_i) {
                    if (info.node_dims[node_i].empty()) {
                        continue;
                    }
                    if (!specialized.node_dims[node_i].empty() && specialized.node_dims[node_i] != info.node_dims[node_i]) {
                        throw std::runtime_error("Broadcast group resolved conflicting local node dimensions for subgroup.");
                    }
                    specialized.node_dims[node_i] = info.node_dims[node_i];
                }
            }

            const std::vector<uint64_t> output_strides = computePackedOutputStrides(output_dims);
            const bool force_all_output_axes = expressionHasIndexAwareOps(stage.expr);

            std::vector<std::vector<uint64_t>> input_strides_by_used;
            input_strides_by_used.reserve(used_input_slots.size());
            specialized.used_input_broadcast_offset_required.reserve(used_input_slots.size());
            specialized.used_input_visible_dims.reserve(used_input_slots.size());
            specialized.used_input_visible_strides.reserve(used_input_slots.size());
            for (uint32_t slot : used_input_slots) {
                if (slot >= stage_inputs.size()) {
                    throw std::runtime_error("Broadcast group input slot out of range.");
                }

                const bool offset_required = broadcast_offset_slots_set.contains(slot);
                specialized.used_input_broadcast_offset_required.push_back(offset_required);
                specialized.used_input_visible_dims.push_back(runtimeInputDims(stage_inputs[slot]));
                specialized.used_input_visible_strides.push_back(runtimeInputStridesForShapeKey(stage_inputs[slot]));

                std::vector<uint64_t> effective_dims = runtimeInputDims(stage_inputs[slot]);
                auto dims_it = effective_dims_by_slot.find(slot);
                if (dims_it != effective_dims_by_slot.end()) {
                    if (dims_it->second.size() > 1) {
                        std::ostringstream oss;
                        oss << "Broadcast group input slot " << slot << " is used with multiple logical shapes in one fused stage. shapes=";
                        bool first = true;
                        for (const auto& dims : dims_it->second) {
                            if (!first) {
                                oss << ", ";
                            }
                            first = false;
                            oss << dimsToString(dims);
                        }
                        throw std::runtime_error(oss.str());
                    }
                    if (!dims_it->second.empty()) {
                        effective_dims = *dims_it->second.begin();
                    }
                }

                if (offset_required) {
                    input_strides_by_used.push_back(
                        computeRuntimeInputStridesForBroadcast(stage_inputs[slot], effective_dims, output_dims));
                } else {
                    input_strides_by_used.emplace_back(output_dims.size(), 0ULL);
                }
            }

            specialized.used_input_load_kinds.assign(used_input_slots.size(), SpecializedInputLoadKind::ScalarPack);
            if (!output_dims.empty() && (output_dims.back() % 2ULL) == 0ULL) {
                const size_t innermost_axis = output_dims.size() - 1ULL;
                for (size_t used_i = 0; used_i < used_input_slots.size(); ++used_i) {
                    // Pair-vector loads are safe when two adjacent output lanes map to two
                    // adjacent input scalars along the innermost logical axis. Requiring an
                    // even innermost output extent avoids pairs crossing a row boundary.
                    if (input_strides_by_used[used_i][innermost_axis] == 1ULL) {
                        specialized.used_input_load_kinds[used_i] = SpecializedInputLoadKind::NativeVector;
                    }
                }
            }

            for (size_t axis = 0; axis < output_dims.size(); ++axis) {
                if (!force_all_output_axes && output_dims[axis] == 1ULL) {
                    continue;
                }

                SpecializedBroadcastAxis axis_desc;
                axis_desc.dim = output_dims[axis];
                axis_desc.output_stride = output_strides[axis];
                axis_desc.input_strides.resize(used_input_slots.size(), 0ULL);

                bool any_nonzero = false;
                for (size_t used_i = 0; used_i < used_input_slots.size(); ++used_i) {
                    axis_desc.input_strides[used_i] = input_strides_by_used[used_i][axis];
                    if (axis_desc.input_strides[used_i] != 0ULL) {
                        any_nonzero = true;
                    }
                }

                if (any_nonzero || force_all_output_axes) {
                    specialized.active_axes.push_back(std::move(axis_desc));
                }
            }

            groups.push_back(ResolvedBroadcastGroup{std::move(specialized)});
        }
    }

    std::sort(groups.begin(), groups.end(), [](const ResolvedBroadcastGroup& a, const ResolvedBroadcastGroup& b) {
        return a.specialized.numel > b.specialized.numel;
    });

    return groups;
}

std::vector<std::string> FusedEquation::getOutputNames() const {
    std::vector<std::string> output_names;
    output_names.reserve(outputs_template.outputs.size());
    for (const NamedOutput& output : outputs_template.outputs) {
        output_names.push_back(output.name);
    }
    return output_names;
}

std::vector<std::string> FusedEquation::filterTensorInputNamesReachableFromOutputs(
    const std::vector<std::string>& input_names, const std::unordered_set<std::string>& output_names) const {
    if (!outputs_template.expr) {
        throw std::runtime_error("FusedEquation reachability query requires non-null PhysicalOutputs.expr.");
    }

    if (output_names.empty() || input_names.empty()) {
        return {};
    }

    const PhysicalExpression& expr = *outputs_template.expr;

    std::unordered_map<std::string, uint32_t> output_node_by_name;
    output_node_by_name.reserve(outputs_template.outputs.size());
    for (const NamedOutput& output : outputs_template.outputs) {
        if (output.node_idx >= expr.nodes.size()) {
            throw std::runtime_error("FusedEquation reachability query found output node out of range: " + output.name);
        }
        output_node_by_name.emplace(output.name, output.node_idx);
    }

    std::vector<uint32_t> stack;
    stack.reserve(output_names.size());
    std::vector<bool> scheduled(expr.nodes.size(), false);
    for (const std::string& output_name : output_names) {
        auto output_it = output_node_by_name.find(output_name);
        if (output_it == output_node_by_name.end()) {
            throw std::runtime_error("FusedEquation reachability query requested unknown output: " + output_name);
        }
        if (!scheduled[output_it->second]) {
            scheduled[output_it->second] = true;
            stack.push_back(output_it->second);
        }
    }

    std::vector<bool> reaches_selected_output(expr.nodes.size(), false);
    auto push_child = [&](uint32_t child) {
        if (child == UINT32_MAX) {
            return;
        }
        if (child >= expr.nodes.size()) {
            throw std::runtime_error("FusedEquation reachability query found child node out of range.");
        }
        if (!scheduled[child]) {
            scheduled[child] = true;
            stack.push_back(child);
        }
    };

    while (!stack.empty()) {
        const uint32_t node_idx = stack.back();
        stack.pop_back();

        if (reaches_selected_output[node_idx]) {
            continue;
        }
        reaches_selected_output[node_idx] = true;

        const ExprNode& node = expr.nodes[node_idx];
        if (node.op == ExprOp::CUDA_KERNEL_OUTPUT) {
            for (uint32_t input_node : node.cuda_kernel_input_nodes) {
                push_child(input_node);
            }
        }
        push_child(node.lhs);
        push_child(node.rhs);
        push_child(node.aux);
        push_child(node.alpha_node);
        push_child(node.beta_node);
    }

    std::unordered_map<std::string, uint32_t> tensor_input_slot_by_name;
    tensor_input_slot_by_name.reserve(expr.inputs.size());
    for (const NamedInput& input : expr.inputs) {
        if (input.kind == NamedInput::Kind::Tensor) {
            tensor_input_slot_by_name.emplace(input.name, input.slot);
        }
    }

    std::vector<std::string> filtered;
    filtered.reserve(input_names.size());
    std::unordered_set<std::string> seen_inputs;
    for (const std::string& input_name : input_names) {
        if (!seen_inputs.insert(input_name).second) {
            throw std::runtime_error("FusedEquation reachability query received duplicate input name: " + input_name);
        }

        auto slot_it = tensor_input_slot_by_name.find(input_name);
        if (slot_it == tensor_input_slot_by_name.end()) {
            throw std::runtime_error("FusedEquation reachability query requested unknown tensor input: " + input_name);
        }

        const uint32_t slot = slot_it->second;
        bool reaches = false;
        for (uint32_t node_idx = 0; node_idx < expr.nodes.size(); ++node_idx) {
            const ExprNode& node = expr.nodes[node_idx];
            if (node.op == ExprOp::INPUT && node.input_slot == slot && reaches_selected_output[node_idx]) {
                reaches = true;
                break;
            }
        }

        if (reaches) {
            filtered.push_back(input_name);
        }
    }

    return filtered;
}

std::vector<uint64_t> FusedEquation::getOutputShape(const Tensor& input) const {
    if (outputs_template.outputs.size() != 1) {
        throw std::runtime_error(
            "FusedEquation::getOutputShape was called for an equation with multiple final outputs. "
            "Use getOutputShapes(...) instead.");
    }

    const auto tensor_inputs = externalRootTensorInputs(root_inputs, backward_config);
    if (tensor_inputs.size() != 1) {
        throw std::runtime_error("FusedEquation::getOutputShape was passed a single tensor input, but this equation requires " +
                                 std::to_string(tensor_inputs.size()) +
                                 " tensor inputs. Pass a dict of name -> Tensor to getOutputShape(...).");
    }

    std::unordered_map<std::string, Tensor> input_map = {
        {tensor_inputs[0]->name, input},
    };

    return getOutputShape(input_map);
}

std::vector<uint64_t> FusedEquation::getOutputShape(const std::unordered_map<std::string, Tensor>& inputs) const {
    if (outputs_template.outputs.size() != 1) {
        throw std::runtime_error(
            "FusedEquation::getOutputShape was called for an equation with multiple final outputs. "
            "Use getOutputShapes(...) instead.");
    }

    if (externalRootTensorInputCount(root_inputs, backward_config) != inputs.size()) {
        throw std::runtime_error("FusedEquation::getOutputShape was passed " + to_string(inputs.size()) +
                                 " tensor inputs, but this equation requires " +
                                 std::to_string(externalRootTensorInputCount(root_inputs, backward_config)) +
                                 " tensor inputs. Pass a dict of name -> Tensor to getOutputShape(...).");
    }

    std::unordered_map<std::string, std::vector<uint64_t>> output_shapes = getOutputShapes(inputs);
    THOR_THROW_IF_FALSE(output_shapes.size() == 1);
    return output_shapes.begin()->second;
}

std::unordered_map<std::string, std::vector<uint64_t>> FusedEquation::getOutputShapes(const Tensor& input) const {
    const auto tensor_inputs = externalRootTensorInputs(root_inputs, backward_config);
    if (tensor_inputs.size() != 1) {
        throw std::runtime_error("FusedEquation::getOutputShapes was passed a single tensor input, but this equation requires " +
                                 std::to_string(tensor_inputs.size()) +
                                 " tensor inputs. Pass a dict of name -> Tensor to getOutputShapes(...).");
    }

    std::unordered_map<std::string, Tensor> input_map = {
        {tensor_inputs[0]->name, input},
    };

    return getOutputShapes(input_map);
}

std::unordered_map<std::string, std::vector<uint64_t>> FusedEquation::getOutputShapes(
    const std::unordered_map<std::string, Tensor>& inputs) const {
    return getOutputShapes(inputs, {});
}

std::unordered_map<std::string, std::vector<uint64_t>> FusedEquation::getOutputShapes(
    const std::unordered_map<std::string, Tensor>& inputs,
    const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs) const {
    std::unordered_map<uint32_t, RuntimeInputValue> root_values = bindRootInputsForCompilation(inputs, {}, tensor_scalar_inputs);
    std::shared_ptr<CompiledOutputs> compiled_outputs = compileForRootValues(root_values);

    if (root_values.empty()) {
        throw std::runtime_error("FusedEquation::getOutputShapes requires at least one bound root input.");
    }

    std::unordered_map<uint32_t, std::vector<uint64_t>> value_dims;
    value_dims.reserve(root_values.size() + compiled_outputs->stages.size());

    for (const auto& [value_id, value] : root_values) {
        value_dims.emplace(value_id, runtimeInputDims(value));
    }
    applyAvailableValueAliases(compiled_outputs->value_aliases, value_dims);

    for (const CompiledExecutionStage& stage : compiled_outputs->stages) {
        std::vector<std::vector<uint64_t>> stage_input_dims;
        stage_input_dims.reserve(stage.input_value_ids.size());

        for (uint32_t value_id : stage.input_value_ids) {
            auto it = value_dims.find(value_id);
            if (it == value_dims.end()) {
                throw std::runtime_error("Missing input shape for staged output-shape inference.");
            }
            stage_input_dims.push_back(it->second);
        }

        if (stage.kind == CompiledExecutionStage::Kind::FusedKernel) {
            for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                value_dims[stage.outputs[output_idx].value_id] = resolveOutputDimsForStageOutput(stage, output_idx, stage_input_dims);
            }
        } else if (stage.kind == CompiledExecutionStage::Kind::CudaKernel) {
            if (!stage.cuda_kernel_expression) {
                throw std::runtime_error("Missing compiled CUDA kernel expression stage.");
            }
            std::unordered_map<std::string, std::vector<uint64_t>> input_shapes;
            const auto& input_specs = stage.cuda_kernel_expression->inputs();
            if (input_specs.size() != stage_input_dims.size()) {
                throw std::runtime_error("CUDA kernel stage input shape count mismatch.");
            }
            for (size_t i = 0; i < input_specs.size(); ++i) {
                input_shapes.emplace(input_specs[i].name, stage_input_dims[i]);
            }
            const std::vector<std::vector<uint64_t>> output_shapes =
                stage.cuda_kernel_expression->inferOutputShapesFromInputShapes(input_shapes);
            for (const CompiledStageOutput& output : stage.outputs) {
                if (output.local_node_idx >= stage.expr.nodes.size()) {
                    throw std::runtime_error("CUDA kernel stage output node index out of range.");
                }
                const ExprNode& output_node = stage.expr.nodes[output.local_node_idx];
                if (output_node.cuda_kernel_output_index >= output_shapes.size()) {
                    throw std::runtime_error("CUDA kernel stage output spec index out of range.");
                }
                value_dims[output.value_id] = output_shapes[output_node.cuda_kernel_output_index];
            }
        } else if (stage.kind == CompiledExecutionStage::Kind::Reduction) {
            if (!stage.reduction) {
                throw std::runtime_error("Missing compiled reduction stage.");
            }

            if (stage.input_value_ids.size() != 1 || stage.outputs.size() != 1) {
                throw std::runtime_error("Reduction stage expected exactly one input and one output.");
            }

            value_dims[stage.outputs[0].value_id] = StampedEquation::computeReductionOutputDims(
                stage_input_dims[0], stage.reduction->reduction_axes, stage.reduction->squeeze_axes);
        } else if (stage.kind == CompiledExecutionStage::Kind::ArgMinMax) {
            if (!stage.arg_minmax) {
                throw std::runtime_error("Missing compiled arg-min/max stage.");
            }

            if (stage.input_value_ids.size() != 1 || stage.outputs.size() != 1) {
                throw std::runtime_error("Arg-min/max stage expected exactly one input and one output.");
            }

            value_dims[stage.outputs[0].value_id] = StampedEquation::computeReductionOutputDims(
                stage_input_dims[0], stage.arg_minmax->reduction_axes, stage.arg_minmax->squeeze_axes);
        } else if (stage.kind == CompiledExecutionStage::Kind::SegmentedReduction) {
            if (!stage.segmented_reduction) {
                throw std::runtime_error("Missing compiled segmented-reduction stage.");
            }
            if (stage.input_value_ids.size() != 2 || stage.outputs.size() != 1) {
                throw std::runtime_error("Segmented-reduction stage expected values, offsets, and one output.");
            }
            if (stage_input_dims[0].size() != 1 || stage_input_dims[1].size() != 1 || stage_input_dims[1][0] == 0) {
                throw std::runtime_error("Segmented-reduction stage currently requires rank-1 values and non-empty rank-1 offsets.");
            }
            value_dims[stage.outputs[0].value_id] = std::vector<uint64_t>{stage_input_dims[1][0] - 1};
        } else if (stage.kind == CompiledExecutionStage::Kind::Scan) {
            if (!stage.scan) {
                throw std::runtime_error("Missing compiled scan stage.");
            }
            const size_t expected_scan_inputs = stage.scan->segmented_by_offsets ? 2 : 1;
            if (stage.input_value_ids.size() != expected_scan_inputs || stage.outputs.empty() || stage.outputs.size() > 2) {
                throw std::runtime_error("Scan stage expected its compiled input count and one or two outputs.");
            }
            for (const CompiledStageOutput& output : stage.outputs) {
                value_dims[output.value_id] = stage_input_dims[0];
            }
        } else if (stage.kind == CompiledExecutionStage::Kind::Softmax) {
            if (!stage.softmax) {
                throw std::runtime_error("Missing compiled softmax stage.");
            }
            if (stage.input_value_ids.size() != 1 || stage.outputs.size() != 1) {
                throw std::runtime_error("Softmax stage expected exactly one input and one output.");
            }
            value_dims[stage.outputs[0].value_id] = stage_input_dims[0];
        } else if (stage.kind == CompiledExecutionStage::Kind::RmsNorm) {
            if (!stage.rms_norm) {
                throw std::runtime_error("Missing compiled RMSNorm stage.");
            }
            if (stage.input_value_ids.size() != 2 || stage.outputs.size() != 1) {
                throw std::runtime_error("RMSNorm stage expected exactly two inputs and one output.");
            }
            value_dims[stage.outputs[0].value_id] = inferRmsNormOutputDims(*stage.rms_norm, stage_input_dims[0], stage_input_dims[1]);
        } else if (stage.kind == CompiledExecutionStage::Kind::ReduceMinMaxBackward) {
            if (!stage.reduce_minmax_backward) {
                throw std::runtime_error("Missing compiled reduce-min/max-backward stage.");
            }

            if (stage.input_value_ids.size() != 2 || stage.outputs.size() != 1) {
                throw std::runtime_error("Reduce-min/max-backward stage expected exactly two inputs and one output.");
            }

            value_dims[stage.outputs[0].value_id] = stage_input_dims[0];
        } else if (stage.kind == CompiledExecutionStage::Kind::Matmul) {
            if (!stage.matmul) {
                throw std::runtime_error("Missing compiled matmul stage.");
            }
            if (stage.outputs.empty() || stage.outputs.size() > 2) {
                throw std::runtime_error("Matmul/gemm stage expected one matrix output and at most one bias-gradient output.");
            }
            const std::vector<uint64_t> matrix_dims = resolveMatmulOutputDimsFromInputs(*stage.matmul, stage_input_dims);
            value_dims[stage.outputs[0].value_id] = matrix_dims;
            if (stage.outputs.size() > 1) {
                if (!stage.matmul->bgrad_output_dtype.has_value()) {
                    throw std::runtime_error("Matmul/gemm stage has a secondary output but no compiled bias-gradient output dtype.");
                }
                if (matrix_dims.size() != 2) {
                    throw std::runtime_error("Matmul/gemm bias-gradient output requires a rank-2 matrix output.");
                }
                value_dims[stage.outputs[1].value_id] = std::vector<uint64_t>{matrix_dims[1]};
            }
        } else if (stage.kind == CompiledExecutionStage::Kind::InPlaceRope) {
            if (!stage.in_place_rope) {
                throw std::runtime_error("Missing compiled in-place RoPE stage.");
            }
            if (stage.outputs.size() != stage.in_place_rope->tensors.size()) {
                throw std::runtime_error("In-place RoPE stage output count mismatch.");
            }
            for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                value_dims[stage.outputs[output_idx].value_id] = stage.in_place_rope->tensors[output_idx].logical_dims;
            }
        } else if (stage.kind == CompiledExecutionStage::Kind::Attention) {
            if (!stage.attention) {
                throw std::runtime_error("Missing compiled attention stage.");
            }
            if (stage.outputs.size() != 1) {
                throw std::runtime_error("Attention stage expected exactly one output.");
            }
            value_dims[stage.outputs[0].value_id] = resolveAttentionOutputDimsFromInputs(*stage.attention, stage_input_dims);
        } else if (stage.kind == CompiledExecutionStage::Kind::AttentionBackward) {
            if (!stage.attention_backward) {
                throw std::runtime_error("Missing compiled attention-backward stage.");
            }
            for (size_t i = 0; i < stage.outputs.size(); ++i) {
                if (stage.outputs[i].local_node_idx >= stage.expr.nodes.size()) {
                    throw std::runtime_error("Attention-backward stage output node index out of range.");
                }
                value_dims[stage.outputs[i].value_id] = resolveAttentionBackwardOutputDimsFromInputs(
                    *stage.attention_backward, stage_input_dims, stage.expr.nodes[stage.outputs[i].local_node_idx].op);
            }
        } else if (stage.kind == CompiledExecutionStage::Kind::Convolution) {
            if (!stage.convolution) {
                throw std::runtime_error("Missing compiled convolution stage.");
            }
            if (stage.outputs.size() != 1) {
                throw std::runtime_error("Convolution stage expected exactly one output.");
            }
            value_dims[stage.outputs[0].value_id] = resolveConvolutionOutputDimsFromInputs(*stage.convolution, stage_input_dims);
        } else if (stage.kind == CompiledExecutionStage::Kind::ConvolutionBackward) {
            if (!stage.convolution_backward) {
                throw std::runtime_error("Missing compiled convolution-backward stage.");
            }
            if (stage.outputs.size() != 1) {
                throw std::runtime_error("Convolution-backward stage expected exactly one output.");
            }
            value_dims[stage.outputs[0].value_id] =
                resolveConvolutionBackwardOutputDimsFromInputs(*stage.convolution_backward, stage_input_dims);
        } else {
            throw std::runtime_error("Unknown execution stage kind in getOutputShapes.");
        }
        applyAvailableValueAliases(compiled_outputs->value_aliases, value_dims);
    }

    applyAvailableValueAliases(compiled_outputs->value_aliases, value_dims);

    std::unordered_map<std::string, std::vector<uint64_t>> final_output_shapes;
    final_output_shapes.reserve(compiled_outputs->final_outputs.size());

    for (const CompiledStageOutput& final_output : compiled_outputs->final_outputs) {
        auto it = value_dims.find(final_output.value_id);
        if (it == value_dims.end()) {
            throw std::runtime_error("Missing final output shape for output: " + final_output.name);
        }
        final_output_shapes.emplace(final_output.name, it->second);
    }

    return final_output_shapes;
}

std::unordered_map<std::string, DataType> FusedEquation::getOutputDataTypes(
    const std::unordered_map<std::string, Tensor>& inputs) const {
    std::unordered_map<uint32_t, RuntimeInputValue> root_values = bindRootInputsForCompilation(inputs);
    std::shared_ptr<CompiledOutputs> compiled_outputs = compileForRootValues(root_values);

    if (root_values.empty()) {
        throw std::runtime_error("FusedEquation::getOutputDataTypes requires at least one bound root input.");
    }

    std::unordered_map<uint32_t, DataType> value_dtypes;
    value_dtypes.reserve(root_values.size() + compiled_outputs->stages.size());

    for (const auto& [value_id, value] : root_values) {
        value_dtypes.emplace(value_id, runtimeInputDType(value));
    }
    applyAvailableValueAliases(compiled_outputs->value_aliases, value_dtypes);

    for (const CompiledExecutionStage& stage : compiled_outputs->stages) {
        for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
            value_dtypes[stage.outputs[output_idx].value_id] = stage.outputDType(output_idx);
        }
        applyAvailableValueAliases(compiled_outputs->value_aliases, value_dtypes);
    }

    applyAvailableValueAliases(compiled_outputs->value_aliases, value_dtypes);

    std::unordered_map<std::string, DataType> final_output_dtypes;
    final_output_dtypes.reserve(compiled_outputs->final_outputs.size());

    for (const CompiledStageOutput& final_output : compiled_outputs->final_outputs) {
        auto it = value_dtypes.find(final_output.value_id);
        if (it == value_dtypes.end()) {
            throw std::runtime_error("Missing final output dtype for output: " + final_output.name);
        }
        final_output_dtypes.emplace(final_output.name, it->second);
    }

    return final_output_dtypes;
}

EquationSignature FusedEquation::buildSignature(uint32_t num_inputs, int device_num, bool use_fast_math) {
    cudaDeviceProp prop{};
    cudaError_t cuda_status = cudaGetDeviceProperties(&prop, device_num);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(cuda_status));
    }

    EquationSignature sig{};
    sig.num_inputs = num_inputs;
    sig.sm_major = prop.major;
    sig.sm_minor = prop.minor;
    sig.device_num = device_num;
    sig.use_fast_math = use_fast_math;
    return sig;
}

PhysicalOutputs FusedEquation::buildShapeSpecializedOutputs(const std::unordered_map<uint32_t, RuntimeInputValue>& root_values) const {
    if (!backward_config.has_value()) {
        return outputs_template;
    }

    // Resolve the forward template dtypes against the actual runtime forward-input dtypes
    // before rebuilding the backward graph. Otherwise preferredGradValueDType(...) sees
    // unresolved INPUT nodes and returns empty, so terminal grad outputs can stay promoted.
    PhysicalOutputs resolved_forward_outputs = backward_config->forward_outputs_template;
    if (!resolved_forward_outputs.expr) {
        throw std::runtime_error("Backward shape specialization requires non-null forward expr.");
    }
    resolved_forward_outputs.expr = std::make_shared<PhysicalExpression>(*backward_config->forward_outputs_template.expr);

    std::vector<DataType> forward_root_input_dtypes(resolved_forward_outputs.expr->numInputs(), DataType::FP32);
    std::vector<bool> have_forward_root_input_dtype(resolved_forward_outputs.expr->numInputs(), false);

    std::unordered_map<std::string, std::vector<uint64_t>> forward_input_dims;
    forward_input_dims.reserve(resolved_forward_outputs.expr->inputs.size());

    for (const NamedInput& forward_input : resolved_forward_outputs.expr->inputs) {
        bool found_name = false;
        for (const NamedInput& root_input : root_inputs) {
            if (root_input.name != forward_input.name) {
                continue;
            }

            auto it = root_values.find(root_input.slot);
            if (it == root_values.end()) {
                throw std::runtime_error("Missing bound runtime input for backward forward-input shape specialization input: " +
                                         forward_input.name);
            }

            forward_input_dims.emplace(forward_input.name, runtimeInputDims(it->second));

            if (forward_input.slot >= forward_root_input_dtypes.size()) {
                throw std::runtime_error("Forward input slot out of range while resolving backward shape-specialized dtypes.");
            }
            forward_root_input_dtypes[forward_input.slot] = runtimeInputDType(it->second);
            have_forward_root_input_dtype[forward_input.slot] = true;

            found_name = true;
            break;
        }

        if (!found_name) {
            throw std::runtime_error("Backward equation root inputs do not contain required forward input: " + forward_input.name);
        }
    }

    for (size_t slot = 0; slot < have_forward_root_input_dtype.size(); ++slot) {
        if (!have_forward_root_input_dtype[slot]) {
            throw std::runtime_error("Missing runtime dtype for forward input slot " + std::to_string(slot) +
                                     " during backward shape specialization.");
        }
    }

    resolveOutputsDTypesInPlace(resolved_forward_outputs, forward_root_input_dtypes);

    // Lower any shape-valid matmul+add/sub patterns to GEMM before building backward.
    // This makes operator-lowered forward expressions follow the same backward path
    // as explicit GEMM expressions, preserving alpha/beta handling consistently.
    optimizeExpressionGemmPatternsInPlace(*resolved_forward_outputs.expr, root_values);

    if (backward_config->upstream_input_names_by_output.has_value()) {
        return buildBackwardOutputs(resolved_forward_outputs,
                                    backward_config->wrt_names,
                                    backward_config->upstream_input_names_by_output.value(),
                                    forward_input_dims,
                                    backward_config->accumulate_grad_outputs);
    }

    return buildBackwardOutputs(
        resolved_forward_outputs, backward_config->wrt_names, std::nullopt, forward_input_dims, backward_config->accumulate_grad_outputs);
}

std::shared_ptr<CompiledOutputs> FusedEquation::compileForInputs(
    const std::unordered_map<std::string, Tensor>& namedInputs,
    const std::unordered_map<std::string, float>& scalarInputs,
    const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs) const {
    std::unordered_map<uint32_t, RuntimeInputValue> root_values =
        bindRootInputsForCompilation(namedInputs, scalarInputs, tensor_scalar_inputs);

    if (root_values.empty()) {
        throw std::runtime_error("FusedEquation::compileForInputs requires at least one bound root input.");
    }

    return compileForRootValues(root_values);
}

std::shared_ptr<PreparedConvenienceRunPlan> FusedEquation::prepareConvenienceRunPlanForInputs(
    const std::unordered_map<std::string, Tensor>& namedInputs,
    const std::unordered_map<std::string, float>& scalarInputs,
    const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs) const {
    std::unordered_map<uint32_t, RuntimeInputValue> root_values =
        bindRootInputsForCompilation(namedInputs, scalarInputs, tensor_scalar_inputs);

    if (root_values.empty()) {
        throw std::runtime_error("FusedEquation::prepareConvenienceRunPlanForInputs requires at least one bound root input.");
    }

    return prepareConvenienceRunPlan(root_values);
}

std::shared_ptr<CompiledOutputs> FusedEquation::compileForRootValues(
    const std::unordered_map<uint32_t, RuntimeInputValue>& root_values) const {
    if (outputs_template.isConditional()) {
        throw std::runtime_error("Graph-level conditional Outputs are stamped through their branch plans and do not expose a flat CompiledOutputs plan.");
    }
    if (root_values.empty()) {
        throw std::runtime_error("FusedEquation::compileForRootValues requires at least one bound root input.");
    }

    const RuntimeDTypeKey dtype_cache_key = makeRuntimeDTypeKey(root_inputs, root_values);
    const bool has_shape_dependent_gemm_lowering =
        outputs_template.expr && expressionHasPotentialGemmLoweringPattern(*outputs_template.expr);

    if (!backward_config.has_value() && !has_shape_dependent_gemm_lowering) {
        std::shared_ptr<CompiledOutputs> cached_compiled_outputs;
        if (compiled_outputs_runtime_cache->tryGet(dtype_cache_key, cached_compiled_outputs)) {
            return cached_compiled_outputs;
        }

        PhysicalOutputs resolved_outputs = outputs_template;
        resolved_outputs.expr = std::make_shared<PhysicalExpression>(*outputs_template.expr);
        resolveOutputsDTypesInPlace(resolved_outputs, dtype_cache_key.root_input_dtypes);

        std::shared_ptr<CompiledOutputs> compiled_outputs = EquationCompiler::compile(resolved_outputs, base_signature, true);
        compiled_outputs_runtime_cache->put(dtype_cache_key, compiled_outputs);
        return compiled_outputs;
    }

    const RuntimeShapeKey shape_cache_key = makeRuntimeShapeKey(root_inputs, root_values);
    std::shared_ptr<CompiledOutputs> cached_compiled_outputs;
    if (compiled_outputs_shape_cache->tryGet(shape_cache_key, cached_compiled_outputs)) {
        return cached_compiled_outputs;
    }

    PhysicalOutputs resolved_outputs = backward_config.has_value() ? buildShapeSpecializedOutputs(root_values) : outputs_template;
    resolved_outputs.expr = std::make_shared<PhysicalExpression>(*resolved_outputs.expr);

    // For backward equations, buildShapeSpecializedOutputs() already rebuilt the backward graph
    // from a shape-specialized, GEMM-lowered forward expression. Running the generic whole-expression
    // GEMM optimization pass again on the full multi-output backward graph can force unrelated output
    // branches through one global shape-inference walk.
    if (!backward_config.has_value()) {
        optimizeExpressionGemmPatternsInPlace(*resolved_outputs.expr, root_values);
    }

    resolveOutputsDTypesInPlace(resolved_outputs, dtype_cache_key.root_input_dtypes);

    std::shared_ptr<CompiledOutputs> compiled_outputs = EquationCompiler::compile(resolved_outputs, base_signature, true);
    compiled_outputs_shape_cache->put(shape_cache_key, compiled_outputs);
    return compiled_outputs;
}

std::shared_ptr<PreparedConvenienceRunPlan> FusedEquation::prepareConvenienceRunPlan(
    const std::unordered_map<uint32_t, RuntimeInputValue>& root_values) const {
    if (outputs_template.isConditional()) {
        throw std::runtime_error("FusedEquation::run convenience path does not support graph-level conditional Outputs; use stamp(...).run().");
    }
    if (root_values.empty()) {
        throw std::runtime_error("FusedEquation::prepareConvenienceRunPlan requires at least one bound root input.");
    }

    const RuntimeShapeKey cache_key = makeRuntimeShapeKey(root_inputs, root_values);

    std::shared_ptr<PreparedConvenienceRunPlan> cached_plan;
    if (convenience_run_plan_cache->tryGet(cache_key, cached_plan)) {
        return cached_plan;
    }

    const std::shared_ptr<CompiledOutputs> compiled_outputs = compileForRootValues(root_values);

    std::unordered_map<uint32_t, RuntimeInputValue> available_values = root_values;
    applyAvailableValueAliases(compiled_outputs->value_aliases, available_values);

    auto plan = std::make_shared<PreparedConvenienceRunPlan>();
    plan->compiled_outputs = compiled_outputs;
    plan->stages.reserve(compiled_outputs->stages.size());

    const auto effectiveRequestedOutputShapes = defaultBackwardRequestedOutputShapes(backward_config, root_inputs, root_values, {});

    std::unordered_set<std::string> expected_output_names;

    for (const CompiledExecutionStage& stage : compiled_outputs->stages) {
        if (stage.kind != CompiledExecutionStage::Kind::FusedKernel) {
            throw std::runtime_error("FusedEquation::run only supports fused-kernel stages, but found stage kind: " +
                                     stage.kindToString(stage.kind) + ". Use stamp(...).run() for staged expressions.");
        }

        applyAvailableValueAliases(compiled_outputs->value_aliases, available_values);

        std::vector<RuntimeInputValue> ordered_inputs;
        ordered_inputs.reserve(stage.input_value_ids.size());
        for (uint32_t value_id : stage.input_value_ids) {
            auto it = available_values.find(value_id);
            if (it == available_values.end()) {
                throw std::runtime_error(
                    "FusedEquation::run encountered a stage that depends on a non-root intermediate tensor. "
                    "Use stamp(...).run() for expressions requiring staged intermediates.");
            }
            ordered_inputs.push_back(it->second);
        }

        PreparedConvenienceRunStage prepared_stage;

        std::vector<uint64_t> resolved_output_dims;
        const bool requires_broadcast = fusedStageRequiresBroadcastLaunch(
            stage, ordered_inputs, effectiveRequestedOutputShapes, backward_config.has_value(), resolved_output_dims);

        if (!requires_broadcast) {
            if (!stage.flat) {
                throw std::runtime_error("FusedEquation::run found a flat fused stage with no compiled kernel.");
            }

            prepared_stage.expected_output_dims.resize(stage.outputs.size());
            for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, output_idx, ordered_inputs);
                auto requested_it = effectiveRequestedOutputShapes.find(stage.outputs[output_idx].name);
                if (requested_it != effectiveRequestedOutputShapes.end() && !requested_it->second.empty()) {
                    if (!(ordered_inputs.empty() && output_dims.empty())) {
                        verifyRequestedOutputLayout(requested_it->second, output_dims);
                    }
                    output_dims = requested_it->second;
                }
                prepared_stage.expected_output_dims[output_idx] = std::move(output_dims);
            }

            prepared_stage.compiled_equation =
                selectFlatCompiledEquation(stage, compiled_outputs->signature, maxNumel(prepared_stage.expected_output_dims));
        } else {
            std::vector<ResolvedBroadcastGroup> groups = buildResolvedBroadcastGroups(stage, ordered_inputs);
            if (groups.empty()) {
                throw std::runtime_error("FusedEquation::run expected at least one broadcast group.");
            }

            std::vector<SpecializedBroadcastGroup> specialized_groups;
            specialized_groups.reserve(groups.size());

            prepared_stage.expected_output_dims.resize(stage.outputs.size());

            for (const ResolvedBroadcastGroup& group : groups) {
                specialized_groups.push_back(group.specialized);

                for (uint32_t output_idx : group.specialized.output_indices) {
                    if (output_idx >= prepared_stage.expected_output_dims.size()) {
                        throw std::runtime_error("Broadcast group output index out of range.");
                    }
                    std::vector<uint64_t> output_dims =
                        stage.outputs[output_idx].materialized_layout == MaterializedTensorLayout::Transposed
                            ? resolveOutputDimsForStageOutput(stage, output_idx, ordered_inputs)
                            : group.specialized.output_dims;
                    auto requested_it = effectiveRequestedOutputShapes.find(stage.outputs[output_idx].name);
                    if (requested_it != effectiveRequestedOutputShapes.end() && !requested_it->second.empty()) {
                        if (!(ordered_inputs.empty() && output_dims.empty())) {
                            verifyRequestedOutputLayout(requested_it->second, output_dims);
                        }
                        output_dims = requested_it->second;
                    }
                    prepared_stage.expected_output_dims[output_idx] = std::move(output_dims);
                }
            }

            prepared_stage.compiled_equation =
                EquationCompiler::compileSpecializedBroadcastStage(stage, compiled_outputs->signature, specialized_groups);
        }

        for (const auto& stage_output : stage.outputs) {
            if (expected_output_names.insert(stage_output.name).second) {
                plan->expected_output_names_in_order.push_back(stage_output.name);
            }
        }

        plan->stages.push_back(std::move(prepared_stage));
    }

    convenience_run_plan_cache->put(cache_key, plan);
    return plan;
}

FusedEquation FusedEquation::compile(const PhysicalOutputs& outputs, int device_num) {
    return compileWithOptions(outputs, device_num, false);
}

FusedEquation FusedEquation::compileWithOptions(const PhysicalOutputs& outputs, int device_num, bool use_fast_math) {
    if (device_num < 0) {
        throw std::runtime_error("FusedEquation::compile requires device_num >= 0.");
    }

    if (!outputs.expr) {
        throw std::runtime_error("FusedEquation::compile requires non-null PhysicalOutputs.expr.");
    }

    if (outputs.outputs.empty()) {
        throw std::runtime_error("FusedEquation::compile requires at least one named output.");
    }

    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(cuda_status));
    }

    if (device_num >= device_count) {
        throw std::runtime_error("FusedEquation::compile device_num is out of range.");
    }

    const EquationSignature base_signature = buildSignature(outputs.expr->numInputs(), device_num, use_fast_math);
    return FusedEquation(outputs, device_num, use_fast_math, base_signature);
}

FusedEquation FusedEquation::compileWithOptions(const PhysicalExpression& expr,
                                                     int device_num,
                                                     bool use_fast_math) {
    if (expr.output_node >= expr.nodes.size()) {
        throw std::runtime_error("FusedEquation::compile PhysicalExpression output_node is out of range.");
    }

    PhysicalOutputs outputs;
    outputs.expr = std::make_shared<PhysicalExpression>(expr);
    outputs.outputs.push_back(NamedOutput{
        .name = "output",
        .node_idx = expr.output_node,
    });

    return compileWithOptions(outputs, device_num, use_fast_math);
}

FusedEquation FusedEquation::compile(const PhysicalExpression& expr, int device_num) {
    return compileWithOptions(expr, device_num, false);
}

FusedEquation FusedEquation::compileBackward(const std::vector<std::string>& wrt_names,
                                             const std::optional<std::string>& upstream_input_name,
                                             bool accumulate_grad_outputs) const {
    PhysicalOutputs backward_outputs =
        buildDeferredShapeBackwardOutputsTemplate(outputs_template, wrt_names, upstream_input_name, accumulate_grad_outputs);
    const EquationSignature backward_signature = buildSignature(backward_outputs.expr->numInputs(), device_num, use_fast_math);
    return FusedEquation(backward_outputs,
                         device_num,
                         use_fast_math,
                         backward_signature,
                         BackwardEquationConfig{
                             .forward_outputs_template = outputs_template,
                             .wrt_names = inferBackwardWrtNamesFromOutputs(backward_outputs),
                             .upstream_input_names_by_output =
                                 upstream_input_name.has_value() ? std::optional<std::unordered_map<std::string, std::string>>(
                                                                       std::unordered_map<std::string, std::string>{
                                                                           {outputs_template.outputs[0].name, upstream_input_name.value()},
                                                                       })
                                                                 : std::nullopt,
                             .accumulate_grad_outputs = accumulate_grad_outputs,
                         });
}

FusedEquation FusedEquation::compileBackward(const std::vector<std::string>& wrt_names,
                                             const std::unordered_map<std::string, std::string>& upstream_input_names_by_output,
                                             bool accumulate_grad_outputs) const {
    PhysicalOutputs backward_outputs =
        buildDeferredShapeBackwardOutputsTemplate(outputs_template, wrt_names, upstream_input_names_by_output, accumulate_grad_outputs);
    const EquationSignature backward_signature = buildSignature(backward_outputs.expr->numInputs(), device_num, use_fast_math);
    return FusedEquation(backward_outputs,
                         device_num,
                         use_fast_math,
                         backward_signature,
                         BackwardEquationConfig{
                             .forward_outputs_template = outputs_template,
                             .wrt_names = inferBackwardWrtNamesFromOutputs(backward_outputs),
                             .upstream_input_names_by_output = upstream_input_names_by_output,
                             .accumulate_grad_outputs = accumulate_grad_outputs,
                         });
}

bool FusedEquation::resolveLayout(std::vector<Tensor>& inputs, std::vector<uint64_t>& outputDimensions) {
    if (inputs.empty()) {
        outputDimensions.clear();
        return false;
    }

    std::vector<std::vector<uint64_t>> originalInputDimensions;
    originalInputDimensions.reserve(inputs.size());
    for (const Tensor& input : inputs) {
        originalInputDimensions.push_back(input.getDimensions());
    }

    uint64_t maxRank = 0;
    for (const Tensor& input : inputs) {
        const std::vector<uint64_t>& dims = input.getDimensions();
        if (dims.empty())
            throw std::runtime_error("Input tensor has 0 dimensions, which is not supported.");
        maxRank = std::max<uint64_t>(maxRank, dims.size());
    }

    for (Tensor& input : inputs) {
        const std::vector<uint64_t>& oldDims = input.getDimensions();
        if (oldDims.size() == maxRank)
            continue;

        std::vector<uint64_t> paddedDims(maxRank - oldDims.size(), 1);
        paddedDims.insert(paddedDims.end(), oldDims.begin(), oldDims.end());
        input.reshape(paddedDims);
    }

    outputDimensions.clear();
    outputDimensions.assign(maxRank, 1);

    for (uint64_t axis = 0; axis < maxRank; ++axis) {
        uint64_t resolvedDim = 1;

        for (const Tensor& input : inputs) {
            const std::vector<uint64_t>& dims = input.getDimensions();
            uint64_t dim = dims[axis];

            if (dim == 1) {
                continue;
            }

            if (resolvedDim == 1) {
                resolvedDim = dim;
            } else if (resolvedDim != dim) {
                std::ostringstream err;
                err << "Input tensors are not broadcast-compatible at axis " << axis << ". "
                    << "Encountered dimension " << resolvedDim << " and dimension " << dim << ". "
                    << "Input shapes: ";
                for (size_t i = 0; i < inputs.size(); ++i) {
                    const std::vector<uint64_t>& inDims = originalInputDimensions[i];
                    err << "[";
                    for (size_t j = 0; j < inDims.size(); ++j) {
                        err << inDims[j];
                        if (j + 1 < inDims.size())
                            err << ", ";
                    }
                    err << "]";
                    if (i + 1 < inputs.size())
                        err << ", ";
                }
                throw std::runtime_error(err.str());
            }
        }

        outputDimensions[axis] = resolvedDim;
    }

    bool requiresBroadcast = false;
    for (const Tensor& input : inputs) {
        if (input.getDimensions() != outputDimensions) {
            requiresBroadcast = true;
            break;
        }
    }

    return requiresBroadcast;
}

std::unordered_map<uint32_t, RuntimeInputValue> FusedEquation::bindRootInputs(
    const std::unordered_map<std::string, Tensor>& namedInputs,
    const std::unordered_map<std::string, float>& scalar_inputs,
    const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
    const std::unordered_map<std::string, Tensor>* namedOutputs) const {
    std::unordered_map<uint32_t, RuntimeInputValue> values;
    values.reserve(root_inputs.size());

    const std::unordered_set<std::string> accumulation_output_names = backwardAccumulationOutputNames(backward_config);

    std::unordered_set<std::string> expected_input_set;
    expected_input_set.reserve(root_inputs.size());
    std::unordered_set<std::string> expected_scalar_input_set;
    expected_scalar_input_set.reserve(root_inputs.size());
    std::unordered_set<std::string> expected_tensor_scalar_input_set;
    expected_tensor_scalar_input_set.reserve(root_inputs.size());

    for (const NamedInput& input : root_inputs) {
        const bool bind_from_outputs = namedOutputs != nullptr && accumulation_output_names.contains(input.name);
        if (bind_from_outputs) {
            auto output_it = namedOutputs->find(input.name);
            if (output_it == namedOutputs->end()) {
                throw std::runtime_error("Missing required gradient output tensor for accumulation: " + input.name);
            }
            values.emplace(input.slot, output_it->second);
            continue;
        }

        if (input.kind == NamedInput::Kind::Tensor) {
            auto input_it = namedInputs.find(input.name);
            if (input_it == namedInputs.end()) {
                throw std::runtime_error("Missing required fused equation input: " + input.name);
            }
            values.emplace(input.slot, input_it->second);
            expected_input_set.insert(input.name);
        } else if (input.kind == NamedInput::Kind::RuntimeScalarFp32) {
            auto scalar_it = scalar_inputs.find(input.name);
            const float scalar_value = scalar_it == scalar_inputs.end() ? 0.0f : scalar_it->second;
            values.emplace(input.slot, scalar_value);
            if (scalar_it != scalar_inputs.end()) {
                expected_scalar_input_set.insert(input.name);
            }
        } else {
            auto scalar_it = tensor_scalar_inputs.find(input.name);
            if (scalar_it == tensor_scalar_inputs.end()) {
                throw std::runtime_error("Missing required fused equation tensor runtime scalar input: " + input.name);
            }
            values.emplace(input.slot, scalar_it->second);
            expected_tensor_scalar_input_set.insert(input.name);
        }
    }

    for (const auto& [name, _] : namedInputs) {
        if (!expected_input_set.contains(name)) {
            throw std::runtime_error("Unexpected input sent to fused equation: " + name);
        }
    }
    for (const auto& [name, _] : scalar_inputs) {
        if (!expected_scalar_input_set.contains(name)) {
            throw std::runtime_error("Unexpected runtime scalar input sent to fused equation: " + name);
        }
    }
    for (const auto& [name, _] : tensor_scalar_inputs) {
        if (!expected_tensor_scalar_input_set.contains(name)) {
            throw std::runtime_error("Unexpected tensor runtime scalar input sent to fused equation: " + name);
        }
    }
    for (const auto& [name, _] : tensor_scalar_inputs) {
        if (!expected_tensor_scalar_input_set.contains(name)) {
            throw std::runtime_error("Unexpected tensor runtime scalar input sent to fused equation: " + name);
        }
    }

    return values;
}

std::unordered_map<uint32_t, RuntimeInputValue> FusedEquation::bindRootInputsForCompilation(
    const std::unordered_map<std::string, Tensor>& namedInputs,
    const std::unordered_map<std::string, float>& scalar_inputs,
    const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
    const std::unordered_map<std::string, std::vector<uint64_t>>& requestedOutputShapes) const {
    if (!accumulatesIntoGradOutputs(backward_config)) {
        return bindRootInputs(namedInputs, scalar_inputs, tensor_scalar_inputs);
    }

    std::unordered_map<uint32_t, RuntimeInputValue> values;
    values.reserve(root_inputs.size());

    std::unordered_set<std::string> expected_input_set;
    expected_input_set.reserve(root_inputs.size());
    std::unordered_set<std::string> expected_scalar_input_set;
    expected_scalar_input_set.reserve(root_inputs.size());
    std::unordered_set<std::string> expected_tensor_scalar_input_set;
    expected_tensor_scalar_input_set.reserve(root_inputs.size());

    std::unordered_map<std::string, uint32_t> root_slot_by_name;
    root_slot_by_name.reserve(root_inputs.size());
    for (const NamedInput& input : root_inputs) {
        root_slot_by_name.emplace(input.name, input.slot);
    }

    const auto accumulation_output_names = backwardAccumulationOutputNames(backward_config);

    for (const NamedInput& input : root_inputs) {
        if (accumulation_output_names.contains(input.name)) {
            continue;
        }

        if (input.kind == NamedInput::Kind::Tensor) {
            auto it = namedInputs.find(input.name);
            if (it == namedInputs.end()) {
                throw std::runtime_error("Missing required fused equation input: " + input.name);
            }
            values.emplace(input.slot, it->second);
            expected_input_set.insert(input.name);
        } else if (input.kind == NamedInput::Kind::RuntimeScalarFp32) {
            auto it = scalar_inputs.find(input.name);
            const float scalar_value = it == scalar_inputs.end() ? 0.0f : it->second;
            values.emplace(input.slot, scalar_value);
            if (it != scalar_inputs.end()) {
                expected_scalar_input_set.insert(input.name);
            }
        } else {
            auto it = tensor_scalar_inputs.find(input.name);
            if (it == tensor_scalar_inputs.end()) {
                throw std::runtime_error("Missing required fused equation tensor runtime scalar input: " + input.name);
            }
            values.emplace(input.slot, it->second);
            expected_tensor_scalar_input_set.insert(input.name);
        }
    }

    for (const auto& [name, _] : namedInputs) {
        if (!expected_input_set.contains(name)) {
            throw std::runtime_error("Unexpected input sent to fused equation: " + name);
        }
    }
    for (const auto& [name, _] : scalar_inputs) {
        if (!expected_scalar_input_set.contains(name)) {
            throw std::runtime_error("Unexpected runtime scalar input sent to fused equation: " + name);
        }
    }
    for (const auto& [name, _] : tensor_scalar_inputs) {
        if (!expected_tensor_scalar_input_set.contains(name)) {
            throw std::runtime_error("Unexpected tensor runtime scalar input sent to fused equation: " + name);
        }
    }

    for (const std::string& wrt_name : backward_config->wrt_names) {
        const std::string grad_output_name = wrt_name + "_grad";
        auto slot_it = root_slot_by_name.find(grad_output_name);
        if (slot_it == root_slot_by_name.end()) {
            throw std::runtime_error("Missing backward accumulation root input for output: " + grad_output_name);
        }

        auto forward_input_it = namedInputs.find(wrt_name);
        if (forward_input_it == namedInputs.end()) {
            throw std::runtime_error("Missing forward input required to infer backward accumulation output: " + wrt_name);
        }

        std::vector<uint64_t> dims = forward_input_it->second.getDimensions();
        auto requested_it = requestedOutputShapes.find(grad_output_name);
        if (requested_it != requestedOutputShapes.end() && !requested_it->second.empty()) {
            dims = requested_it->second;
        }

        const std::optional<DataType> preferred_dtype = preferredBackwardGradBufferDType(backward_config.value(), wrt_name);
        const DataType grad_buffer_dtype = preferred_dtype.has_value() ? preferred_dtype.value() : forward_input_it->second.getDataType();
        TensorDescriptor descriptor(grad_buffer_dtype, dims);
        values.emplace(slot_it->second, Tensor(forward_input_it->second.getPlacement(), descriptor));
    }

    return values;
}

std::shared_ptr<StampedEquation> FusedEquation::stampEquation(const std::shared_ptr<CompiledEquation>& compiledEquation,
                                                              std::vector<RuntimeInputValue>& inputs,
                                                              std::vector<Tensor>& outputs,
                                                              const Stream& stream) const {
    if (!compiledEquation) {
        throw std::runtime_error("Cannot stamp an empty compiled equation.");
    }

    if (outputs.size() != compiledEquation->numOutputs()) {
        throw std::runtime_error("Wrong number of outputs passed to FusedEquation::stampEquation.");
    }

    if (inputs.size() != compiledEquation->numInputs()) {
        throw std::runtime_error("Wrong number of inputs passed to FusedEquation::stampEquation.");
    }

    if (outputs.empty()) {
        throw std::runtime_error("FusedEquation::stampEquation requires at least one output tensor.");
    }

    for (uint64_t i = 0; i < inputs.size(); ++i) {
        if (compiledEquation->input_kinds[i] == NamedInput::Kind::Tensor) {
            if (!runtimeInputIsTensor(inputs[i])) {
                throw std::runtime_error("FusedEquation::stampEquation expected tensor input at slot " + std::to_string(i) + ".");
            }
            const Tensor& input = runtimeInputTensor(inputs[i]);
            if (!input.isInitialized()) {
                throw std::runtime_error("Input tensor is not initialized.");
            }
            if (input.getDescriptor().getDataType() != compiledEquation->input_dtypes[i]) {
                throw std::runtime_error("Input tensor data type mismatch.");
            }
            if (input.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
                throw std::runtime_error("Input tensor GPU mismatch.");
            }
        } else if (compiledEquation->input_kinds[i] == NamedInput::Kind::RuntimeScalarFp32) {
            if (!std::holds_alternative<float>(inputs[i])) {
                throw std::runtime_error("FusedEquation::stampEquation expected runtime scalar input at slot " + std::to_string(i) + ".");
            }
            if (compiledEquation->input_dtypes[i] != DataType::FP32) {
                throw std::runtime_error("Runtime scalar inputs currently require FP32 compiled input dtype.");
            }
        } else {
            if (!runtimeInputIsTensorScalarBinding(inputs[i])) {
                throw std::runtime_error("FusedEquation::stampEquation expected tensor runtime scalar input at slot " + std::to_string(i) +
                                         ".");
            }
            const TensorScalarBinding& binding = runtimeInputTensorScalarBinding(inputs[i]);
            if (!binding.buffer.isInitialized()) {
                throw std::runtime_error("Tensor runtime scalar buffer is not initialized.");
            }
            if (binding.buffer.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
                throw std::runtime_error("Tensor runtime scalar buffer must be on GPU.");
            }
            if (binding.buffer.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
                throw std::runtime_error("Tensor runtime scalar buffer GPU mismatch.");
            }
            if (binding.sourceDType != compiledEquation->input_dtypes[i]) {
                throw std::runtime_error("Tensor runtime scalar source dtype mismatch.");
            }
            const size_t bytes_needed = binding.byteOffset + dataTypeSizeBytes(binding.sourceDType);
            if (bytes_needed > binding.buffer.getArraySizeInBytes()) {
                throw std::runtime_error("Tensor runtime scalar binding exceeds backing buffer size.");
            }
        }
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        const Tensor& output = outputs[i];
        if (!output.isInitialized()) {
            throw std::runtime_error("Output tensor is not initialized.");
        }
        if (output.getDescriptor().getDataType() != compiledEquation->output_dtypes[i]) {
            throw std::runtime_error("Output tensor data type mismatch.");
        }
        if (output.getPlacement().getDeviceNum() != compiledEquation->deviceNum) {
            throw std::runtime_error("Output tensor GPU mismatch.");
        }
    }

    return make_shared<StampedEquation>(compiledEquation, inputs, outputs, stream);
}

std::shared_ptr<StampedReduction> FusedEquation::stampReduction(const std::shared_ptr<CompiledReduction>& compiledReduction,
                                                                Tensor& input,
                                                                const Stream& stream,
                                                                const std::vector<uint64_t>& requested_output_shape) const {
    return stampReduction(compiledReduction, input, std::nullopt, stream, requested_output_shape);
}

std::shared_ptr<StampedReduction> FusedEquation::stampReduction(const std::shared_ptr<CompiledReduction>& compiledReduction,
                                                                Tensor& input,
                                                                const std::optional<Tensor>& preallocatedOutput,
                                                                const Stream& stream,
                                                                const std::vector<uint64_t>& requested_output_shape) const {
    if (!compiledReduction)
        throw std::runtime_error("Tried to stamp reduction on a non-reduction FusedEquation.");

    if (input.getDataType() != compiledReduction->input_dtype) {
        throw std::runtime_error("Runtime reduction input dtype does not match the compiled reduction input dtype.");
    }
    std::shared_ptr<BuiltReduction> built = StampedEquation::buildReduction(compiledReduction, input, stream.getGpuNum());
    if (built->key.result_kind != ReductionResultKind::Value) {
        throw std::runtime_error("Dense value reduction produced an unexpected reduction result kind.");
    }

    std::vector<uint64_t> resolved_output_dimensions =
        StampedEquation::computeReductionOutputDims(input.getDimensions(), built->key.reduction_axes, built->key.squeeze_axes);

    std::vector<uint64_t> output_dimensions = resolved_output_dimensions;
    if (!requested_output_shape.empty()) {
        verifyRequestedOutputLayout(requested_output_shape, resolved_output_dimensions);
        output_dimensions = requested_output_shape;
    }

    Tensor output;
    if (preallocatedOutput.has_value()) {
        output = preallocatedOutput.value();

        if (output.getPlacement() != input.getPlacement()) {
            throw std::runtime_error("Preallocated reduction output tensor placement does not match the reduction input placement.");
        }

        if (output.getDescriptor().getDataType() != compiledReduction->output_dtype) {
            throw std::runtime_error("Preallocated reduction output tensor dtype does not match the compiled reduction output dtype.");
        }

        verifyRequestedOutputLayout(output.getDimensions(), resolved_output_dimensions);

        if (!requested_output_shape.empty() && output.getDimensions() != output_dimensions) {
            throw std::runtime_error("Preallocated reduction output tensor shape does not match the requested output shape.");
        }
    } else {
        TensorDescriptor outputDescriptor(compiledReduction->output_dtype, output_dimensions);
        output = Tensor(input.getPlacement(), outputDescriptor);
    }

    return make_shared<StampedReduction>(std::move(built), input, output, stream);
}

std::shared_ptr<StampedArgMinMax> FusedEquation::stampArgMinMax(const std::shared_ptr<CompiledArgMinMax>& compiledStage,
                                                                Tensor& input,
                                                                const Stream& stream,
                                                                const std::vector<uint64_t>& requested_output_shape) const {
    return stampArgMinMax(compiledStage, input, std::nullopt, stream, requested_output_shape);
}

std::shared_ptr<StampedArgMinMax> FusedEquation::stampArgMinMax(const std::shared_ptr<CompiledArgMinMax>& compiledStage,
                                                                Tensor& input,
                                                                const std::optional<Tensor>& preallocatedOutput,
                                                                const Stream& stream,
                                                                const std::vector<uint64_t>& requested_output_shape) const {
    if (!compiledStage) {
        throw std::runtime_error("stampArgMinMax requires non-null compiled stage.");
    }
    if (input.getDataType() != compiledStage->input_dtype) {
        throw std::runtime_error("Runtime argmin/argmax input dtype does not match the compiled reduction input dtype.");
    }
    std::shared_ptr<BuiltReduction> built = StampedEquation::buildReduction(compiledStage->op,
                                                                            compiledStage->reduction_axes,
                                                                            compiledStage->squeeze_axes,
                                                                            compiledStage->input_dtype,
                                                                            DataType::FP32,
                                                                            compiledStage->compute_dtype,
                                                                            ReductionResultKind::Indices,
                                                                            input,
                                                                            stream.getGpuNum());
    if (built->key.result_kind != ReductionResultKind::Indices) {
        throw std::runtime_error("Dense argmin/argmax produced an unexpected reduction result kind.");
    }

    std::vector<uint64_t> resolved_output_dimensions =
        StampedEquation::computeReductionOutputDims(input.getDimensions(), built->key.reduction_axes, built->key.squeeze_axes);
    std::vector<uint64_t> output_dimensions = resolved_output_dimensions;
    if (!requested_output_shape.empty()) {
        verifyRequestedOutputLayout(requested_output_shape, resolved_output_dimensions);
        output_dimensions = requested_output_shape;
    }

    Tensor output;
    if (preallocatedOutput.has_value()) {
        output = preallocatedOutput.value();

        if (output.getPlacement() != input.getPlacement()) {
            throw std::runtime_error(
                "Preallocated argmin/argmax output tensor placement does not match the argmin/argmax input placement.");
        }

        if (output.getDescriptor().getDataType() != compiledStage->output_dtype) {
            throw std::runtime_error(
                "Preallocated argmin/argmax output tensor dtype does not match the compiled argmin/argmax output dtype.");
        }

        verifyRequestedOutputLayout(output.getDimensions(), resolved_output_dimensions);

        if (!requested_output_shape.empty() && output.getDimensions() != output_dimensions) {
            throw std::runtime_error("Preallocated argmin/argmax output tensor shape does not match the requested output shape.");
        }
    } else {
        TensorDescriptor outputDescriptor(compiledStage->output_dtype, output_dimensions);
        output = Tensor(input.getPlacement(), outputDescriptor);
    }

    return make_shared<StampedArgMinMax>(std::move(built), input, output, stream);
}

std::shared_ptr<StampedSoftmax> FusedEquation::stampSoftmax(const std::shared_ptr<CompiledSoftmax>& compiledStage,
                                                            Tensor& input,
                                                            const std::optional<Tensor>& preallocatedOutput,
                                                            const Stream& stream,
                                                            const std::vector<uint64_t>& requested_output_shape) const {
    if (!compiledStage) {
        throw std::runtime_error("stampSoftmax requires non-null compiled stage.");
    }

    Tensor adaptedInput = allocateCudnnSoftmaxInputAdapterIfNeeded(input, compiledStage->input_dtype, ExprOp::SOFTMAX);

    const std::vector<uint64_t> resolved_output_dimensions = adaptedInput.getDimensions();
    std::vector<uint64_t> output_dimensions = resolved_output_dimensions;
    if (!requested_output_shape.empty()) {
        verifyRequestedOutputLayout(requested_output_shape, resolved_output_dimensions);
        output_dimensions = requested_output_shape;
    }

    Tensor output;
    if (preallocatedOutput.has_value()) {
        output = preallocatedOutput.value();
        if (output.getPlacement() != adaptedInput.getPlacement()) {
            throw std::runtime_error("Preallocated softmax output tensor placement does not match the softmax input placement.");
        }
        if (output.getDescriptor().getDataType() != compiledStage->output_dtype) {
            throw std::runtime_error("Preallocated softmax output tensor dtype does not match the compiled softmax output dtype.");
        }
        verifyRequestedOutputLayout(output.getDimensions(), resolved_output_dimensions);
        if (!requested_output_shape.empty() && output.getDimensions() != output_dimensions) {
            throw std::runtime_error("Preallocated softmax output tensor shape does not match the requested output shape.");
        }
    } else {
        TensorDescriptor outputDescriptor(compiledStage->output_dtype, output_dimensions);
        output = Tensor(adaptedInput.getPlacement(), outputDescriptor);
    }

    std::shared_ptr<BuiltSoftmax> built = StampedEquation::buildSoftmax(compiledStage, adaptedInput, output, stream.getGpuNum());
    return make_shared<StampedSoftmax>(compiledStage, std::move(built), input, adaptedInput, output, stream);
}

std::shared_ptr<StampedRmsNorm> FusedEquation::stampRmsNorm(const std::shared_ptr<CompiledRmsNorm>& compiledStage,
                                                               Tensor& input,
                                                               Tensor& scale,
                                                               const std::optional<Tensor>& preallocatedOutput,
                                                               const Stream& stream) const {
    if (!compiledStage) {
        throw std::runtime_error("stampRmsNorm requires non-null compiled stage.");
    }

    if (input.getDataType() != compiledStage->input_dtype) {
        throw std::runtime_error(
            "RMSNorm input tensor dtype does not match the compiled RMSNorm input dtype. "
            "Thor will not implicitly convert RMSNorm activations during stamping.");
    }
    if (scale.getDataType() != compiledStage->scale_dtype) {
        throw std::runtime_error(
            "RMSNorm scale tensor dtype does not match the compiled RMSNorm scale dtype. "
            "Thor will not implicitly convert RMSNorm parameters during stamping.");
    }

    const Tensor& adaptedInput = input;
    const Tensor& adaptedScale = scale;
    ExprNode rmsNormNode{};
    rmsNormNode.op = ExprOp::RMSNORM;
    rmsNormNode.rms_norm_normalized_feature_count = compiledStage->normalized_feature_count;
    const std::vector<uint64_t> output_dims =
        inferRmsNormOutputDims(rmsNormNode, adaptedInput.getDimensions(), adaptedScale.getDimensions());

    Tensor output;
    if (preallocatedOutput.has_value()) {
        output = preallocatedOutput.value();
        if (output.getPlacement() != adaptedInput.getPlacement()) {
            throw std::runtime_error("Preallocated RMSNorm output tensor placement does not match input placement.");
        }
        if (output.getDescriptor().getDataType() != compiledStage->output_dtype) {
            throw std::runtime_error("Preallocated RMSNorm output tensor dtype does not match compiled output dtype.");
        }
        if (output.getDimensions() != output_dims) {
            throw std::runtime_error("Preallocated RMSNorm output tensor dimensions are incompatible with the RMSNorm output shape.");
        }
    } else {
        TensorDescriptor outputDescriptor(compiledStage->output_dtype, output_dims);
        output = Tensor(adaptedInput.getPlacement(), outputDescriptor);
    }

    return std::make_shared<StampedRmsNorm>(compiledStage, adaptedInput, adaptedScale, output, stream);
}

std::shared_ptr<StampedConvolution> FusedEquation::stampConvolution(const std::shared_ptr<CompiledConvolution>& compiledStage,
                                                                    Tensor& input,
                                                                    Tensor& filter,
                                                                    const std::optional<Tensor>& preallocatedOutput,
                                                                    const Stream& stream) const {
    if (!compiledStage) {
        throw std::runtime_error("stampConvolution requires non-null compiled stage.");
    }
    if (input.getDataType() != compiledStage->input_dtype) {
        throw std::runtime_error(
            "Convolution input tensor dtype does not match the compiled convolution input dtype. "
            "Thor will not implicitly convert convolution operands during stamping.");
    }
    if (filter.getDataType() != compiledStage->filter_dtype) {
        throw std::runtime_error(
            "Convolution filter tensor dtype does not match the compiled convolution filter dtype. "
            "Thor will not implicitly convert convolution parameters during stamping.");
    }

    const std::vector<std::vector<uint64_t>> stage_input_dims = {input.getDimensions(), filter.getDimensions()};
    const std::vector<uint64_t> output_dims = resolveConvolutionOutputDimsFromInputs(*compiledStage, stage_input_dims);

    Tensor output;
    if (preallocatedOutput.has_value()) {
        output = preallocatedOutput.value();
        if (output.getPlacement() != input.getPlacement()) {
            throw std::runtime_error("Preallocated convolution output tensor placement does not match input placement.");
        }
        if (output.getDescriptor().getDataType() != compiledStage->output_dtype) {
            throw std::runtime_error("Preallocated convolution output tensor dtype does not match compiled output dtype.");
        }
        if (output.getDimensions() != output_dims) {
            throw std::runtime_error(
                "Preallocated convolution output tensor dimensions are incompatible with the convolution output shape.");
        }
    } else {
        TensorDescriptor outputDescriptor(compiledStage->output_dtype, output_dims);
        output = Tensor(input.getPlacement(), outputDescriptor);
    }

    std::shared_ptr<BuiltConvolution> built =
        StampedEquation::buildConvolution(compiledStage, input, filter, output, stream, input.getPlacement().getDeviceNum());

    std::optional<Tensor> workspace = std::nullopt;
    if (built->workspace_bytes > 0) {
        workspace = Tensor(input.getPlacement(), TensorDescriptor(DataType::UINT8, {built->workspace_bytes}));
    }

    return std::make_shared<StampedConvolution>(compiledStage, built, input, filter, output, stream, workspace);
}

std::shared_ptr<StampedConvolutionBackward> FusedEquation::stampConvolutionBackward(
    const std::shared_ptr<CompiledConvolutionBackward>& compiledStage,
    Tensor& input,
    Tensor& grad_output,
    const std::optional<Tensor>& preallocatedOutput,
    const Stream& stream) const {
    if (!compiledStage) {
        throw std::runtime_error("stampConvolutionBackward requires non-null compiled stage.");
    }
    if (input.getDataType() != compiledStage->input_dtype) {
        throw std::runtime_error(
            "Convolution-backward input tensor dtype does not match the compiled input dtype. "
            "Thor will not implicitly convert convolution operands during stamping.");
    }
    if (grad_output.getDataType() != compiledStage->grad_output_dtype) {
        throw std::runtime_error(
            "Convolution-backward gradient tensor dtype does not match the compiled gradient dtype. "
            "Thor will not implicitly convert convolution operands during stamping.");
    }

    const std::vector<std::vector<uint64_t>> stage_input_dims = {input.getDimensions(), grad_output.getDimensions()};
    const std::vector<uint64_t> output_dims = resolveConvolutionBackwardOutputDimsFromInputs(*compiledStage, stage_input_dims);

    Tensor output;
    if (preallocatedOutput.has_value()) {
        output = preallocatedOutput.value();
        if (output.getPlacement() != input.getPlacement()) {
            throw std::runtime_error("Preallocated convolution-backward output tensor placement does not match input placement.");
        }
        if (output.getDescriptor().getDataType() != compiledStage->output_dtype) {
            throw std::runtime_error("Preallocated convolution-backward output tensor dtype does not match compiled output dtype.");
        }
        if (output.getDimensions() != output_dims) {
            throw std::runtime_error(
                "Preallocated convolution-backward output tensor dimensions are incompatible with the convolution-backward output shape.");
        }
    } else {
        TensorDescriptor outputDescriptor(compiledStage->output_dtype, output_dims);
        output = Tensor(input.getPlacement(), outputDescriptor);
    }

    std::shared_ptr<BuiltConvolution> built = StampedEquation::buildConvolutionBackward(
        compiledStage, input, grad_output, output, stream, input.getPlacement().getDeviceNum());

    std::optional<Tensor> workspace = std::nullopt;
    if (built->workspace_bytes > 0) {
        workspace = Tensor(input.getPlacement(), TensorDescriptor(DataType::UINT8, {built->workspace_bytes}));
    }

    return std::make_shared<StampedConvolutionBackward>(compiledStage, built, input, grad_output, output, stream, workspace);
}

std::shared_ptr<StampedMatmul> FusedEquation::stampMatmul(const std::shared_ptr<CompiledMatmul>& compiledStage,
                                                          Tensor& lhs,
                                                          Tensor& rhs,
                                                          const std::optional<Tensor>& preallocatedOutput,
                                                          const Stream& stream,
                                                          const std::optional<RuntimeInputValue>& alpha_input,
                                                          const std::optional<RuntimeInputValue>& beta_input,
                                                          const std::optional<std::string>& alpha_runtime_name,
                                                          const std::optional<std::string>& beta_runtime_name,
                                                          const std::optional<Tensor>& epilogue_aux,
                                                          const std::optional<Tensor>& preallocatedBgradOutput) const {
    if (!compiledStage) {
        throw std::runtime_error("stampMatmul requires non-null compiled stage.");
    }
    if (compiledStage->op != ExprOp::MATMUL) {
        throw std::runtime_error("stampMatmul(lhs, rhs, ...) called for a non-matmul stage.");
    }

    if (lhs.getDataType() != compiledStage->lhs_dtype) {
        throw std::runtime_error(
            "Matmul lhs tensor dtype does not match the compiled lhs dtype. "
            "Thor will not implicitly convert matmul operands during stamping.");
    }
    if (rhs.getDataType() != compiledStage->rhs_dtype) {
        throw std::runtime_error(
            "Matmul rhs tensor dtype does not match the compiled rhs dtype. "
            "Thor will not implicitly convert matmul operands or parameter storage during stamping.");
    }

    std::vector<std::vector<uint64_t>> stage_input_dims(2);
    stage_input_dims[0] = lhs.getDimensions();
    stage_input_dims[1] = rhs.getDimensions();

    auto assign_scale_dims = [&](uint32_t slot, const std::optional<RuntimeInputValue>& input, const char* label) {
        if (slot == UINT32_MAX) {
            return;
        }
        if (!input.has_value()) {
            throw std::runtime_error(std::string("Matmul stage missing bound ") + label + " scale input.");
        }
        if (stage_input_dims.size() <= slot) {
            stage_input_dims.resize(slot + 1);
        }
        stage_input_dims[slot] = runtimeInputDims(input.value());
    };

    assign_scale_dims(compiledStage->alpha_input_slot, alpha_input, "alpha");
    assign_scale_dims(compiledStage->beta_input_slot, beta_input, "beta");
    if (compiledStage->epilogue_aux_input_slot != UINT32_MAX) {
        if (!epilogue_aux.has_value()) {
            throw std::runtime_error("Matmul stage missing backward epilogue aux tensor.");
        }
        if (stage_input_dims.size() <= compiledStage->epilogue_aux_input_slot) {
            stage_input_dims.resize(compiledStage->epilogue_aux_input_slot + 1);
        }
        stage_input_dims[compiledStage->epilogue_aux_input_slot] = epilogue_aux->getDimensions();
    }

    const std::vector<uint64_t> output_dims = resolveMatmulOutputDimsFromInputs(*compiledStage, stage_input_dims);

    Tensor output;
    if (preallocatedOutput.has_value()) {
        output = preallocatedOutput.value();
        if (output.getPlacement() != lhs.getPlacement()) {
            throw std::runtime_error("Preallocated matmul output tensor placement does not match the input placement.");
        }
        if (output.getDescriptor().getDataType() != compiledStage->output_dtype) {
            throw std::runtime_error("Preallocated matmul output tensor dtype does not match the compiled output dtype.");
        }
        if (output.getDimensions() != output_dims) {
            throw std::runtime_error("Preallocated matmul output tensor dimensions are incompatible with the matmul output shape.");
        }
    } else {
        TensorDescriptor outputDescriptor(compiledStage->output_dtype, output_dims);
        output = Tensor(lhs.getPlacement(), outputDescriptor);
    }

    std::optional<Tensor> bgrad_output = std::nullopt;
    if (compiledStage->bgrad_output_dtype.has_value()) {
        if (output_dims.size() != 2) {
            throw std::runtime_error("Matmul bias-gradient epilogue requires a rank-2 output matrix.");
        }
        const std::vector<uint64_t> bgrad_dims{output_dims[1]};
        if (preallocatedBgradOutput.has_value()) {
            bgrad_output = preallocatedBgradOutput.value();
            if (bgrad_output->getPlacement() != lhs.getPlacement()) {
                throw std::runtime_error("Preallocated matmul bias-gradient output tensor placement does not match the input placement.");
            }
            if (bgrad_output->getDescriptor().getDataType() != compiledStage->bgrad_output_dtype.value()) {
                throw std::runtime_error("Preallocated matmul bias-gradient output tensor dtype does not match the compiled dtype.");
            }
            if (bgrad_output->getDimensions() != bgrad_dims) {
                throw std::runtime_error("Preallocated matmul bias-gradient output tensor dimensions are incompatible with the output shape.");
            }
        } else {
            bgrad_output = Tensor(lhs.getPlacement(), TensorDescriptor(compiledStage->bgrad_output_dtype.value(), bgrad_dims));
        }
    }

    std::shared_ptr<BuiltMatmul> built = StampedEquation::buildMatmul(compiledStage,
                                                                      lhs,
                                                                      rhs,
                                                                      std::nullopt,
                                                                      output,
                                                                      lhs.getPlacement().getDeviceNum(),
                                                                      epilogue_aux,
                                                                      bgrad_output);

    std::optional<Tensor> workspace = std::nullopt;
    if (built->workspace_bytes > 0) {
        TensorDescriptor workspaceDescriptor(DataType::UINT8, {built->workspace_bytes});
        workspace = Tensor(lhs.getPlacement(), workspaceDescriptor);
    }

    return make_shared<StampedMatmul>(compiledStage,
                                      built,
                                      lhs,
                                      rhs,
                                      std::nullopt,
                                      output,
                                      stream,
                                      workspace,
                                      alpha_input,
                                      beta_input,
                                      alpha_runtime_name,
                                      beta_runtime_name,
                                      std::nullopt,
                                      std::nullopt,
                                      std::nullopt,
                                      std::nullopt,
                                      epilogue_aux,
                                      bgrad_output);
}

std::shared_ptr<StampedMatmul> FusedEquation::stampMatmul(const std::shared_ptr<CompiledMatmul>& compiledStage,
                                                          Tensor& lhs,
                                                          Tensor& rhs,
                                                          Tensor& addend,
                                                          const std::optional<Tensor>& preallocatedOutput,
                                                          const Stream& stream,
                                                          const std::optional<RuntimeInputValue>& alpha_input,
                                                          const std::optional<RuntimeInputValue>& beta_input,
                                                          const std::optional<std::string>& alpha_runtime_name,
                                                          const std::optional<std::string>& beta_runtime_name,
                                                          const std::optional<Tensor>& epilogue_aux,
                                                          const std::optional<Tensor>& preallocatedBgradOutput) const {
    if (!compiledStage) {
        throw std::runtime_error("stampMatmul requires non-null compiled stage.");
    }
    if (compiledStage->op != ExprOp::GEMM) {
        throw std::runtime_error("stampMatmul(lhs, rhs, addend, ...) called for a non-gemm stage.");
    }

    if (lhs.getDataType() != compiledStage->lhs_dtype) {
        throw std::runtime_error(
            "GEMM lhs tensor dtype does not match the compiled lhs dtype. "
            "Thor will not implicitly convert matmul operands during stamping.");
    }
    if (rhs.getDataType() != compiledStage->rhs_dtype) {
        throw std::runtime_error(
            "GEMM rhs tensor dtype does not match the compiled rhs dtype. "
            "Thor will not implicitly convert matmul operands or parameter storage during stamping.");
    }
    const bool use_bias_epilogue = addend.getDimensions().size() == 1;
    if (use_bias_epilogue && addend.getDataType() != compiledStage->output_dtype) {
        throw std::runtime_error(
            "GEMM bias epilogue requires the bias tensor dtype to match the matmul output dtype; Thor will not insert an implicit conversion for a cuBLASLt bias epilogue.");
    }
    if (!use_bias_epilogue && addend.getDataType() != compiledStage->aux_dtype) {
        throw std::runtime_error(
            "GEMM addend tensor dtype does not match the compiled auxiliary dtype. "
            "Thor will not implicitly convert GEMM operands during stamping.");
    }

    std::vector<std::vector<uint64_t>> stage_input_dims(3);
    stage_input_dims[0] = lhs.getDimensions();
    stage_input_dims[1] = rhs.getDimensions();
    stage_input_dims[2] = addend.getDimensions();

    auto assign_scale_dims = [&](uint32_t slot, const std::optional<RuntimeInputValue>& input, const char* label) {
        if (slot == UINT32_MAX) {
            return;
        }
        if (!input.has_value()) {
            throw std::runtime_error(std::string("Matmul stage missing bound ") + label + " scale input.");
        }
        if (stage_input_dims.size() <= slot) {
            stage_input_dims.resize(slot + 1);
        }
        stage_input_dims[slot] = runtimeInputDims(input.value());
    };

    assign_scale_dims(compiledStage->alpha_input_slot, alpha_input, "alpha");
    assign_scale_dims(compiledStage->beta_input_slot, beta_input, "beta");
    if (compiledStage->epilogue_aux_input_slot != UINT32_MAX) {
        if (!epilogue_aux.has_value()) {
            throw std::runtime_error("Matmul stage missing backward epilogue aux tensor.");
        }
        if (stage_input_dims.size() <= compiledStage->epilogue_aux_input_slot) {
            stage_input_dims.resize(compiledStage->epilogue_aux_input_slot + 1);
        }
        stage_input_dims[compiledStage->epilogue_aux_input_slot] = epilogue_aux->getDimensions();
    }

    const std::vector<uint64_t> output_dims = resolveMatmulOutputDimsFromInputs(*compiledStage, stage_input_dims);

    Tensor output;
    if (preallocatedOutput.has_value()) {
        output = preallocatedOutput.value();
        if (output.getPlacement() != lhs.getPlacement()) {
            throw std::runtime_error("Preallocated gemm output tensor placement does not match the input placement.");
        }
        if (output.getDescriptor().getDataType() != compiledStage->output_dtype) {
            throw std::runtime_error("Preallocated gemm output tensor dtype does not match the compiled output dtype.");
        }
        if (output.getDimensions() != output_dims) {
            throw std::runtime_error("Preallocated gemm output tensor dimensions are incompatible with the gemm output shape.");
        }
    } else {
        TensorDescriptor outputDescriptor(compiledStage->output_dtype, output_dims);
        output = Tensor(lhs.getPlacement(), outputDescriptor);
    }

    std::optional<Tensor> bgrad_output = std::nullopt;
    if (compiledStage->bgrad_output_dtype.has_value()) {
        if (output_dims.size() != 2) {
            throw std::runtime_error("GEMM bias-gradient epilogue requires a rank-2 output matrix.");
        }
        const std::vector<uint64_t> bgrad_dims{output_dims[1]};
        if (preallocatedBgradOutput.has_value()) {
            bgrad_output = preallocatedBgradOutput.value();
            if (bgrad_output->getPlacement() != lhs.getPlacement()) {
                throw std::runtime_error("Preallocated GEMM bias-gradient output tensor placement does not match the input placement.");
            }
            if (bgrad_output->getDescriptor().getDataType() != compiledStage->bgrad_output_dtype.value()) {
                throw std::runtime_error("Preallocated GEMM bias-gradient output tensor dtype does not match the compiled dtype.");
            }
            if (bgrad_output->getDimensions() != bgrad_dims) {
                throw std::runtime_error("Preallocated GEMM bias-gradient output tensor dimensions are incompatible with the output shape.");
            }
        } else {
            bgrad_output = Tensor(lhs.getPlacement(), TensorDescriptor(compiledStage->bgrad_output_dtype.value(), bgrad_dims));
        }
    }

    std::shared_ptr<BuiltMatmul> built = StampedEquation::buildMatmul(compiledStage,
                                                                      lhs,
                                                                      rhs,
                                                                      std::optional<Tensor>(addend),
                                                                      output,
                                                                      lhs.getPlacement().getDeviceNum(),
                                                                      epilogue_aux,
                                                                      bgrad_output);

    std::optional<Tensor> workspace = std::nullopt;
    if (built->workspace_bytes > 0) {
        TensorDescriptor workspaceDescriptor(DataType::UINT8, {built->workspace_bytes});
        workspace = Tensor(lhs.getPlacement(), workspaceDescriptor);
    }

    std::optional<Tensor> alpha_device_scratch = std::nullopt;
    std::optional<Tensor> beta_device_scratch = std::nullopt;
    std::optional<Tensor> alpha_host_scratch = std::nullopt;
    std::optional<Tensor> beta_host_scratch = std::nullopt;
    const bool any_tensor_backed_scale = optionalRuntimeInputIsTensorLike(alpha_input) || optionalRuntimeInputIsTensorLike(beta_input);
    if (any_tensor_backed_scale) {
        TensorDescriptor scalarDescriptor(DataType::FP32, {1});
        alpha_device_scratch = Tensor(lhs.getPlacement(), scalarDescriptor);
        beta_device_scratch = Tensor(lhs.getPlacement(), scalarDescriptor);
        const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
        alpha_host_scratch = Tensor(cpuPlacement, scalarDescriptor);
        beta_host_scratch = Tensor(cpuPlacement, scalarDescriptor);
    }

    return make_shared<StampedMatmul>(compiledStage,
                                      built,
                                      lhs,
                                      rhs,
                                      std::optional<Tensor>(addend),
                                      output,
                                      stream,
                                      workspace,
                                      alpha_input,
                                      beta_input,
                                      alpha_runtime_name,
                                      beta_runtime_name,
                                      alpha_device_scratch,
                                      beta_device_scratch,
                                      alpha_host_scratch,
                                      beta_host_scratch,
                                      epilogue_aux,
                                      bgrad_output);
}

std::shared_ptr<StampedAttention> FusedEquation::stampAttention(const std::shared_ptr<CompiledAttention>& compiledStage,
                                                                const Tensor& q,
                                                                const Tensor& k,
                                                                const Tensor& v,
                                                                const std::optional<Tensor>& bias,
                                                                const std::optional<Tensor>& seq_len_q,
                                                                const std::optional<Tensor>& seq_len_kv,
                                                                const std::optional<Tensor>& q_ragged_offsets,
                                                                const std::optional<Tensor>& kv_ragged_offsets,
                                                                const std::optional<Tensor>& page_table_k,
                                                                const std::optional<Tensor>& page_table_v,
                                                                const std::optional<Tensor>& dropout_seed,
                                                                const std::optional<Tensor>& dropout_offset,
                                                                const std::optional<Tensor>& descale_q,
                                                                const std::optional<Tensor>& descale_k,
                                                                const std::optional<Tensor>& descale_v,
                                                                const std::optional<Tensor>& descale_s,
                                                                const std::optional<Tensor>& scale_s,
                                                                const std::optional<Tensor>& scale_o,
                                                                const std::optional<Tensor>& amax_s,
                                                                const std::optional<Tensor>& amax_o,
                                                                std::optional<Tensor> preallocatedOutput,
                                                                const Stream& stream,
                                                                std::shared_ptr<AttentionForwardState> forward_state) const {
    if (!compiledStage) {
        throw std::runtime_error("stampAttention requires non-null compiled stage.");
    }

    if (q.getDataType() != compiledStage->output_dtype || k.getDataType() != compiledStage->output_dtype ||
        v.getDataType() != compiledStage->output_dtype) {
        throw std::runtime_error(
            "Attention q/k/v tensor dtypes must match the compiled attention output dtype exactly. "
            "Thor will not implicitly convert attention operands during stamping.");
    }
    std::optional<Tensor> boundBias = std::nullopt;
    if (compiledStage->use_bias) {
        if (!bias.has_value()) {
            throw std::runtime_error("stampAttention requires an additive bias tensor for this compiled stage.");
        }
        boundBias = bias.value();
    }

    std::vector<std::vector<uint64_t>> stage_input_dims{q.getDimensions(), k.getDimensions(), v.getDimensions()};
    if (compiledStage->use_bias && boundBias.has_value()) {
        stage_input_dims.push_back(boundBias->getDimensions());
    }
    if (compiledStage->use_padding_mask) {
        if (!seq_len_q.has_value() || !seq_len_kv.has_value()) {
            throw std::runtime_error("stampAttention requires q/kv sequence length tensors for padding-mask attention.");
        }
        stage_input_dims.push_back(seq_len_q->getDimensions());
        stage_input_dims.push_back(seq_len_kv->getDimensions());
    }
    if (compiledStage->use_ragged_offsets) {
        if (!q_ragged_offsets.has_value() || !kv_ragged_offsets.has_value()) {
            throw std::runtime_error("stampAttention requires q/kv ragged offset tensors for ragged attention.");
        }
        stage_input_dims.push_back(q_ragged_offsets->getDimensions());
        stage_input_dims.push_back(kv_ragged_offsets->getDimensions());
    }
    if (compiledStage->use_paged_kv_cache) {
        if (!page_table_k.has_value() || !page_table_v.has_value()) {
            throw std::runtime_error("stampAttention requires K/V page-table tensors for paged KV attention.");
        }
        stage_input_dims.push_back(page_table_k->getDimensions());
        stage_input_dims.push_back(page_table_v->getDimensions());
    }
    if (compiledStage->dropout_probability > 0.0f) {
        if (!dropout_seed.has_value() || !dropout_offset.has_value()) {
            throw std::runtime_error("stampAttention requires dropout seed/offset tensors for attention dropout.");
        }
        stage_input_dims.push_back(dropout_seed->getDimensions());
        stage_input_dims.push_back(dropout_offset->getDimensions());
    }
    if (compiledStage->use_fp8_forward_scaling) {
        if (!descale_q.has_value() || !descale_k.has_value() || !descale_v.has_value() || !descale_s.has_value() || !scale_s.has_value() ||
            !scale_o.has_value() || !amax_s.has_value() || !amax_o.has_value()) {
            throw std::runtime_error("stampAttention requires FP8 descale/scale/amax tensors for this compiled stage.");
        }
        stage_input_dims.push_back(descale_q->getDimensions());
        stage_input_dims.push_back(descale_k->getDimensions());
        stage_input_dims.push_back(descale_v->getDimensions());
        stage_input_dims.push_back(descale_s->getDimensions());
        stage_input_dims.push_back(scale_s->getDimensions());
        stage_input_dims.push_back(scale_o->getDimensions());
        stage_input_dims.push_back(amax_s->getDimensions());
        stage_input_dims.push_back(amax_o->getDimensions());
    }
    const std::vector<uint64_t> output_dims = resolveAttentionOutputDimsFromInputs(*compiledStage, stage_input_dims);
    const AttentionTensorLogicalDims qLogical = logicalAttentionDims(q.getDimensions(), compiledStage->q_layout, "q");
    const AttentionTensorLogicalDims kLogical = logicalAttentionDims(k.getDimensions(), compiledStage->k_layout, "k");
    const AttentionTensorLogicalDims vLogical = logicalAttentionDims(v.getDimensions(), compiledStage->v_layout, "v");

    Tensor output;
    if (preallocatedOutput.has_value()) {
        output = preallocatedOutput.value();
        if (output.getPlacement() != q.getPlacement()) {
            throw std::runtime_error("Preallocated attention output tensor placement does not match the input placement.");
        }
        if (output.getDescriptor().getDataType() != compiledStage->output_dtype) {
            throw std::runtime_error("Preallocated attention output tensor dtype does not match the compiled output dtype.");
        }
        if (output.getDimensions() != output_dims) {
            throw std::runtime_error("Preallocated attention output tensor dimensions are incompatible with the attention output shape.");
        }
    } else {
        TensorDescriptor outputDescriptor(compiledStage->output_dtype, output_dims);
        output = Tensor(q.getPlacement(), outputDescriptor);
    }

    if (compiledStage->use_padding_mask) {
        auto validate_seq_len = [&](const std::optional<Tensor>& tensor, const char* label) {
            if (!tensor.has_value()) {
                throw std::runtime_error(std::string("Attention padding-mask missing ") + label + " tensor.");
            }
            if (tensor->getPlacement() != q.getPlacement()) {
                throw std::runtime_error(std::string("Attention padding-mask ") + label + " placement must match q.");
            }
            if (tensor->getDataType() != DataType::INT32) {
                throw std::runtime_error(std::string("Attention padding-mask ") + label + " dtype must be INT32.");
            }
            if (tensor->getDimensions() != std::vector<uint64_t>{qLogical.batch}) {
                throw std::runtime_error(std::string("Attention padding-mask ") + label + " shape must be [B].");
            }
        };
        validate_seq_len(seq_len_q, "q_seq_len");
        validate_seq_len(seq_len_kv, "kv_seq_len");
    }
    if (compiledStage->use_ragged_offsets) {
        auto validate_ragged_offset = [&](const std::optional<Tensor>& tensor, const char* label) {
            if (!tensor.has_value()) {
                throw std::runtime_error(std::string("Attention ragged missing ") + label + " tensor.");
            }
            if (tensor->getPlacement() != q.getPlacement()) {
                throw std::runtime_error(std::string("Attention ragged ") + label + " placement must match q.");
            }
            if (!isCudnnRaggedOffsetDataType(tensor->getDataType())) {
                throw std::runtime_error(std::string("Attention ragged ") + label + " dtype must be INT32.");
            }
            if (tensor->getDimensions() != std::vector<uint64_t>{qLogical.batch + 1}) {
                throw std::runtime_error(std::string("Attention ragged ") + label + " shape must be [B + 1].");
            }
        };
        validate_ragged_offset(q_ragged_offsets, "q_offsets");
        validate_ragged_offset(kv_ragged_offsets, "kv_offsets");
    }
    if (compiledStage->use_paged_kv_cache) {
        auto validate_page_table = [&](const std::optional<Tensor>& tensor, const char* label) {
            if (!tensor.has_value()) {
                throw std::runtime_error(std::string("Attention paged KV missing ") + label + " tensor.");
            }
            if (tensor->getPlacement() != q.getPlacement()) {
                throw std::runtime_error(std::string("Attention paged KV ") + label + " placement must match q.");
            }
            if (tensor->getDataType() != DataType::INT32) {
                throw std::runtime_error(std::string("Attention paged KV ") + label + " dtype must be INT32.");
            }
            if (compiledStage->paged_kv_max_sequence_length <= 0) {
                throw std::runtime_error("Attention paged KV max sequence length must be positive.");
            }
            const uint64_t max_kv = static_cast<uint64_t>(compiledStage->paged_kv_max_sequence_length);
            const uint64_t block_size = std::string(label).find("page_table_v") != std::string::npos ? vLogical.sequence_length
                                                                                                     : kLogical.sequence_length;
            const std::vector<uint64_t> expected{qLogical.batch, 1, ceilDivPositive(max_kv, block_size), 1};
            if (tensor->getDimensions() != expected) {
                throw std::runtime_error(std::string("Attention paged KV ") + label + " shape must be [B,1,ceil(Skv/block),1].");
            }
        };
        validate_page_table(page_table_k, "page_table_k");
        validate_page_table(page_table_v, "page_table_v");
    }
    if (compiledStage->dropout_probability > 0.0f) {
        auto validate_dropout_scalar = [&](const std::optional<Tensor>& tensor, const char* label) {
            if (!tensor.has_value()) {
                throw std::runtime_error(std::string("Attention dropout missing ") + label + " tensor.");
            }
            if (tensor->getPlacement() != q.getPlacement()) {
                throw std::runtime_error(std::string("Attention dropout ") + label + " placement must match q.");
            }
            if (tensor->getDataType() != DataType::INT64) {
                throw std::runtime_error(std::string("Attention dropout ") + label + " dtype must be INT64.");
            }
            if (tensor->getDimensions() != std::vector<uint64_t>{1, 1, 1, 1}) {
                throw std::runtime_error(std::string("Attention dropout ") + label + " shape must be [1,1,1,1].");
            }
        };
        validate_dropout_scalar(dropout_seed, "seed");
        validate_dropout_scalar(dropout_offset, "offset");
    }
    if (compiledStage->use_fp8_forward_scaling) {
        auto validate_fp8_scalar = [&](const std::optional<Tensor>& tensor, const char* label) {
            if (!tensor.has_value()) {
                throw std::runtime_error(std::string("Attention FP8 missing ") + label + " tensor.");
            }
            if (tensor->getPlacement() != q.getPlacement()) {
                throw std::runtime_error(std::string("Attention FP8 ") + label + " placement must match q.");
            }
            if (tensor->getDataType() != DataType::FP32) {
                throw std::runtime_error(std::string("Attention FP8 ") + label + " dtype must be FP32.");
            }
            if (tensor->getDimensions() != std::vector<uint64_t>{1, 1, 1, 1}) {
                throw std::runtime_error(std::string("Attention FP8 ") + label + " shape must be [1,1,1,1].");
            }
        };
        validate_fp8_scalar(descale_q, "descale_q");
        validate_fp8_scalar(descale_k, "descale_k");
        validate_fp8_scalar(descale_v, "descale_v");
        validate_fp8_scalar(descale_s, "descale_s");
        validate_fp8_scalar(scale_s, "scale_s");
        validate_fp8_scalar(scale_o, "scale_o");
        validate_fp8_scalar(amax_s, "amax_s");
        validate_fp8_scalar(amax_o, "amax_o");
    }

    // Build and validate the cuDNN descriptor during stamping so shape/layout errors fail before execution.
    (void)compiledStage->descriptorFor(q, k, v, output);
    if (compiledStage->use_bias && boundBias.has_value()) {
        if (!isAllowedAttentionBiasDims(boundBias->getDimensions(), qLogical.batch, qLogical.heads, qLogical.sequence_length, kLogical.sequence_length)) {
            throw std::runtime_error("Attention additive bias must have shape " +
                                     attentionBiasShapeDescription(qLogical.batch, qLogical.heads, qLogical.sequence_length, kLogical.sequence_length) + ".");
        }
        if (boundBias->getDataType() != compiledStage->compute_dtype) {
            throw std::runtime_error(
                "Attention additive bias dtype must match attention compute dtype; Thor does not insert hidden bias dtype conversions for "
                "cuDNN attention.");
        }
    }

    if (forward_state) {
        forward_state->output = output;
        forward_state->has_valid_stats = false;
    }

    return make_shared<StampedAttention>(compiledStage,
                                         q,
                                         k,
                                         v,
                                         boundBias,
                                         seq_len_q,
                                         seq_len_kv,
                                         q_ragged_offsets,
                                         kv_ragged_offsets,
                                         page_table_k,
                                         page_table_v,
                                         dropout_seed,
                                         dropout_offset,
                                         descale_q,
                                         descale_k,
                                         descale_v,
                                         descale_s,
                                         scale_s,
                                         scale_o,
                                         amax_s,
                                         amax_o,
                                         output,
                                         stream,
                                         std::move(forward_state));
}

std::shared_ptr<StampedAttentionBackward> FusedEquation::stampAttentionBackward(
    const std::shared_ptr<CompiledAttentionBackward>& compiledStage,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& seq_len_q,
    const std::optional<Tensor>& seq_len_kv,
    const std::optional<Tensor>& q_ragged_offsets,
    const std::optional<Tensor>& kv_ragged_offsets,
    const std::optional<Tensor>& dropout_seed,
    const std::optional<Tensor>& dropout_offset,
    const Tensor& dO,
    const std::vector<std::optional<Tensor>>& preallocatedOutputs,
    const Stream& stream,
    std::shared_ptr<AttentionForwardState> saved_forward_state) const {
    if (!compiledStage) {
        throw std::runtime_error("stampAttentionBackward requires non-null compiled stage.");
    }

    if (q.getDataType() != compiledStage->dQ_dtype || k.getDataType() != compiledStage->dK_dtype ||
        v.getDataType() != compiledStage->dV_dtype) {
        throw std::runtime_error(
            "Attention-backward q/k/v tensor dtypes must match the compiled gradient dtypes exactly. "
            "Thor will not implicitly convert attention operands during stamping.");
    }
    if (dO.getDataType() != q.getDataType()) {
        throw std::runtime_error(
            "Attention-backward dO dtype must match q/k/v for the current cuDNN path. "
            "Thor will not implicitly convert attention gradients during stamping.");
    }
    std::optional<Tensor> boundBias = std::nullopt;
    if (compiledStage->use_bias) {
        if (!bias.has_value()) {
            throw std::runtime_error("stampAttentionBackward requires an additive bias tensor for this compiled stage.");
        }
        boundBias = bias.value();
    }

    std::vector<std::vector<uint64_t>> stage_input_dims{
        q.getDimensions(), k.getDimensions(), v.getDimensions(), dO.getDimensions()};
    std::vector<std::vector<uint64_t>> forward_input_dims{stage_input_dims[0], stage_input_dims[1], stage_input_dims[2]};
    if (compiledStage->use_bias) {
        forward_input_dims.push_back(boundBias->getDimensions());
        stage_input_dims.push_back(boundBias->getDimensions());
    }
    if (compiledStage->use_padding_mask) {
        if (!seq_len_q.has_value() || !seq_len_kv.has_value()) {
            throw std::runtime_error("stampAttentionBackward requires q/kv sequence length tensors for padding-mask attention.");
        }
        forward_input_dims.push_back(seq_len_q->getDimensions());
        forward_input_dims.push_back(seq_len_kv->getDimensions());
        stage_input_dims.push_back(seq_len_q->getDimensions());
        stage_input_dims.push_back(seq_len_kv->getDimensions());
    }
    if (compiledStage->use_ragged_offsets) {
        if (!q_ragged_offsets.has_value() || !kv_ragged_offsets.has_value()) {
            throw std::runtime_error("stampAttentionBackward requires q/kv ragged offset tensors for ragged attention.");
        }
        forward_input_dims.push_back(q_ragged_offsets->getDimensions());
        forward_input_dims.push_back(kv_ragged_offsets->getDimensions());
        stage_input_dims.push_back(q_ragged_offsets->getDimensions());
        stage_input_dims.push_back(kv_ragged_offsets->getDimensions());
    }
    if (compiledStage->dropout_probability > 0.0f) {
        if (!dropout_seed.has_value() || !dropout_offset.has_value()) {
            throw std::runtime_error("stampAttentionBackward requires dropout seed/offset tensors for attention dropout.");
        }
        forward_input_dims.push_back(dropout_seed->getDimensions());
        forward_input_dims.push_back(dropout_offset->getDimensions());
        stage_input_dims.push_back(dropout_seed->getDimensions());
        stage_input_dims.push_back(dropout_offset->getDimensions());
    }
    const std::vector<uint64_t> o_dims = resolveAttentionOutputDimsFromInputs(
        makeForwardAttentionView(*compiledStage, dO.getDataType(), ".stats_forward"), forward_input_dims);
    const AttentionTensorLogicalDims qLogical = logicalAttentionDims(q.getDimensions(), compiledStage->q_layout, "q");
    const AttentionTensorLogicalDims kLogical = logicalAttentionDims(k.getDimensions(), compiledStage->k_layout, "k");
    if (dO.getDimensions() != o_dims) {
        throw std::runtime_error("Attention-backward dO dimensions do not match the corresponding forward attention output shape.");
    }

    auto make_or_validate_output = [&](size_t idx, DataType dtype, const std::vector<uint64_t>& dims, const char* label) {
        if (idx < preallocatedOutputs.size() && preallocatedOutputs[idx].has_value()) {
            Tensor out = preallocatedOutputs[idx].value();
            if (out.getPlacement() != q.getPlacement()) {
                throw std::runtime_error(std::string("Preallocated attention-backward ") + label +
                                         " placement does not match q placement.");
            }
            if (out.getDescriptor().getDataType() != dtype) {
                throw std::runtime_error(std::string("Preallocated attention-backward ") + label + " dtype does not match compiled dtype.");
            }
            if (out.getDimensions() != dims) {
                throw std::runtime_error(std::string("Preallocated attention-backward ") + label + " dimensions are incompatible.");
            }
            return out;
        }
        return Tensor(q.getPlacement(), TensorDescriptor(dtype, dims));
    };

    Tensor dQ = make_or_validate_output(0, compiledStage->dQ_dtype, q.getDimensions(), "dQ");
    Tensor dK = make_or_validate_output(1, compiledStage->dK_dtype, k.getDimensions(), "dK");
    Tensor dV = make_or_validate_output(2, compiledStage->dV_dtype, v.getDimensions(), "dV");
    Tensor oScratch(q.getPlacement(), TensorDescriptor(dO.getDataType(), o_dims));
    Tensor stats(q.getPlacement(),
                 TensorDescriptor(DataType::FP32, {qLogical.batch, qLogical.heads, qLogical.sequence_length, 1}));
    const std::vector<uint64_t> denseBiasDims{qLogical.batch, qLogical.heads, qLogical.sequence_length, kLogical.sequence_length};
    std::optional<Tensor> dBiasScratch = std::nullopt;
    if (compiledStage->use_bias) {
        dBiasScratch = make_or_validate_output(3, q.getDataType(), denseBiasDims, "dBias");
    }

    if (compiledStage->use_padding_mask) {
        auto validate_seq_len = [&](const std::optional<Tensor>& tensor, const char* label) {
            if (!tensor.has_value()) {
                throw std::runtime_error(std::string("Attention-backward padding-mask missing ") + label + " tensor.");
            }
            if (tensor->getPlacement() != q.getPlacement()) {
                throw std::runtime_error(std::string("Attention-backward padding-mask ") + label + " placement must match q.");
            }
            if (tensor->getDataType() != DataType::INT32) {
                throw std::runtime_error(std::string("Attention-backward padding-mask ") + label + " dtype must be INT32.");
            }
            if (tensor->getDimensions() != std::vector<uint64_t>{qLogical.batch}) {
                throw std::runtime_error(std::string("Attention-backward padding-mask ") + label + " shape must be [B].");
            }
        };
        validate_seq_len(seq_len_q, "q_seq_len");
        validate_seq_len(seq_len_kv, "kv_seq_len");
    }
    if (compiledStage->use_ragged_offsets) {
        auto validate_ragged_offset = [&](const std::optional<Tensor>& tensor, const char* label) {
            if (!tensor.has_value()) {
                throw std::runtime_error(std::string("Attention-backward ragged missing ") + label + " tensor.");
            }
            if (tensor->getPlacement() != q.getPlacement()) {
                throw std::runtime_error(std::string("Attention-backward ragged ") + label + " placement must match q.");
            }
            if (!isCudnnRaggedOffsetDataType(tensor->getDataType())) {
                throw std::runtime_error(std::string("Attention-backward ragged ") + label + " dtype must be INT32.");
            }
            if (tensor->getDimensions() != std::vector<uint64_t>{qLogical.batch + 1}) {
                throw std::runtime_error(std::string("Attention-backward ragged ") + label + " shape must be [B + 1].");
            }
        };
        validate_ragged_offset(q_ragged_offsets, "q_offsets");
        validate_ragged_offset(kv_ragged_offsets, "kv_offsets");
    }
    if (compiledStage->dropout_probability > 0.0f) {
        auto validate_dropout_scalar = [&](const std::optional<Tensor>& tensor, const char* label) {
            if (!tensor.has_value()) {
                throw std::runtime_error(std::string("Attention-backward dropout missing ") + label + " tensor.");
            }
            if (tensor->getPlacement() != q.getPlacement()) {
                throw std::runtime_error(std::string("Attention-backward dropout ") + label + " placement must match q.");
            }
            if (tensor->getDataType() != DataType::INT64) {
                throw std::runtime_error(std::string("Attention-backward dropout ") + label + " dtype must be INT64.");
            }
            if (tensor->getDimensions() != std::vector<uint64_t>{1, 1, 1, 1}) {
                throw std::runtime_error(std::string("Attention-backward dropout ") + label + " shape must be [1,1,1,1].");
            }
        };
        validate_dropout_scalar(dropout_seed, "seed");
        validate_dropout_scalar(dropout_offset, "offset");
    }

    (void)compiledStage->descriptorFor(q, k, v, oScratch);
    if (compiledStage->use_bias && boundBias.has_value()) {
        if (!isAllowedAttentionBiasDims(boundBias->getDimensions(),
                                        qLogical.batch,
                                        qLogical.heads,
                                        qLogical.sequence_length,
                                        kLogical.sequence_length)) {
            throw std::runtime_error(
                "Attention-backward additive bias must have shape " +
                attentionBiasShapeDescription(qLogical.batch, qLogical.heads, qLogical.sequence_length, kLogical.sequence_length) +
                "; sequence-broadcast bias is materialized to dense by production autodiff before cuDNN backward, and dense "
                "dBias is explicitly reduced afterward.");
        }
        if (boundBias->getDataType() != compiledStage->compute_dtype) {
            throw std::runtime_error(
                "Attention-backward additive bias dtype must match attention compute dtype; Thor does not insert hidden bias dtype "
                "conversions for cuDNN attention.");
        }
    }
    return make_shared<StampedAttentionBackward>(compiledStage,
                                                 q,
                                                 k,
                                                 v,
                                                 boundBias,
                                                 seq_len_q,
                                                 seq_len_kv,
                                                 q_ragged_offsets,
                                                 kv_ragged_offsets,
                                                 dropout_seed,
                                                 dropout_offset,
                                                 dO,
                                                 dQ,
                                                 dK,
                                                 dV,
                                                 oScratch,
                                                 stats,
                                                 dBiasScratch,
                                                 stream,
                                                 std::move(saved_forward_state));
}

std::shared_ptr<StampedReduceMinMaxBackward> FusedEquation::stampReduceMinMaxBackward(
    const std::shared_ptr<CompiledReduceMinMaxBackward>& compiledStage, Tensor& input, Tensor& grad_output, const Stream& stream) const {
    return stampReduceMinMaxBackward(compiledStage, input, grad_output, std::nullopt, stream);
}

std::shared_ptr<StampedReduceMinMaxBackward> FusedEquation::stampReduceMinMaxBackward(
    const std::shared_ptr<CompiledReduceMinMaxBackward>& compiledStage,
    Tensor& input,
    Tensor& grad_output,
    const std::optional<Tensor>& preallocatedOutput,
    const Stream& stream) const {
    if (!compiledStage) {
        throw std::runtime_error("stampReduceMinMaxBackward requires non-null compiled stage.");
    }

    if (input.getDataType() != compiledStage->input_dtype) {
        throw std::runtime_error("Runtime reduce-min/max-backward input dtype does not match the compiled reduction input dtype.");
    }
    if (grad_output.getDataType() != compiledStage->grad_output_dtype) {
        throw std::runtime_error("Grad-output dtype does not match compiled reduce-min/max-backward grad-output dtype.");
    }

    const std::vector<uint64_t> expected_grad_dims = StampedEquation::computeReductionOutputDims(
        input.getDimensions(), compiledStage->reduction_axes, compiledStage->squeeze_axes);
    if (!outputDimensionsMatchIgnoringSingletons(grad_output.getDimensions(), expected_grad_dims)) {
        throw std::runtime_error("Grad-output tensor dimensions are incompatible with reduce-min/max-backward stage.");
    }

    const ExprOp reduce_op = compiledStage->op == ExprOp::REDUCE_MIN_BACKWARD ? ExprOp::REDUCE_MIN : ExprOp::REDUCE_MAX;
    std::shared_ptr<BuiltReduction> built = StampedEquation::buildReduction(reduce_op,
                                                                            compiledStage->reduction_axes,
                                                                            compiledStage->squeeze_axes,
                                                                            compiledStage->input_dtype,
                                                                            DataType::FP32,
                                                                            compiledStage->compute_dtype,
                                                                            ReductionResultKind::Indices,
                                                                            input,
                                                                            stream.getGpuNum());

    if (built->key.result_kind != ReductionResultKind::Indices) {
        throw std::runtime_error("Dense reduce-min/max backward produced an unexpected reduction result kind.");
    }

    const std::vector<uint64_t> unsqueezed_output_dims =
        StampedEquation::computeReductionOutputDims(input.getDimensions(), built->key.reduction_axes, {});

    TensorDescriptor indicesDescriptor(DataType::UINT32, unsqueezed_output_dims);
    Tensor indices(input.getPlacement(), indicesDescriptor);

    Tensor output;
    if (preallocatedOutput.has_value()) {
        output = preallocatedOutput.value();

        if (output.getPlacement() != input.getPlacement()) {
            throw std::runtime_error("Preallocated reduce-min/max-backward output tensor placement does not match the input placement.");
        }
        if (output.getDescriptor().getDataType() != compiledStage->output_dtype) {
            throw std::runtime_error("Preallocated reduce-min/max-backward output tensor dtype does not match the compiled output dtype.");
        }
        if (output.getDimensions() != input.getDimensions()) {
            throw std::runtime_error("Preallocated reduce-min/max-backward output tensor dimensions do not match the input dimensions.");
        }
    } else {
        TensorDescriptor outputDescriptor(compiledStage->output_dtype, input.getDimensions());
        output = Tensor(input.getPlacement(), outputDescriptor);
    }

    return make_shared<StampedReduceMinMaxBackward>(
        std::move(built), input, grad_output, output, indices, stream);
}


std::shared_ptr<StampedScanMinMaxBackward> FusedEquation::stampScanMinMaxBackward(
    const std::shared_ptr<CompiledScanMinMaxBackward>& compiledStage,
    Tensor& input,
    Tensor& grad_output,
    const std::optional<Tensor>& offsets,
    Tensor& output,
    const Stream& stream) const {
    if (!compiledStage) {
        throw std::runtime_error("stampScanMinMaxBackward requires non-null compiled stage.");
    }
    if (compiledStage->value_op != ScanOp::Min && compiledStage->value_op != ScanOp::Max) {
        throw std::runtime_error("stampScanMinMaxBackward supports only min/max scans.");
    }
    if (input.getDataType() != compiledStage->input_dtype) {
        throw std::runtime_error("Scan-min/max-backward input dtype does not match compiled dtype.");
    }
    if (grad_output.getDataType() != compiledStage->grad_output_dtype || output.getDataType() != compiledStage->output_dtype) {
        throw std::runtime_error("Scan-min/max-backward grad/output dtype does not match compiled dtype.");
    }
    if (grad_output.getDimensions() != input.getDimensions() || output.getDimensions() != input.getDimensions()) {
        throw std::runtime_error("Scan-min/max-backward grad/output dimensions must match input dimensions.");
    }
    if (input.getPlacement() != grad_output.getPlacement() || input.getPlacement() != output.getPlacement()) {
        throw std::runtime_error("Scan-min/max-backward tensors must be on the same placement.");
    }
    if (compiledStage->segmented_by_offsets != offsets.has_value()) {
        throw std::runtime_error("Scan-min/max-backward segmented-offset presence does not match compiled descriptor.");
    }
    if (offsets.has_value()) {
        if (!compiledStage->offset_dtype.has_value() || offsets->getDataType() != compiledStage->offset_dtype.value()) {
            throw std::runtime_error("Scan-min/max-backward offsets dtype does not match compiled descriptor.");
        }
        if (offsets->getPlacement() != input.getPlacement()) {
            throw std::runtime_error("Scan-min/max-backward offsets placement must match input placement.");
        }
    }

    TensorDescriptor indicesDescriptor(DataType::UINT32, input.getDimensions());
    Tensor indices(input.getPlacement(), indicesDescriptor);

    const ScanOp arg_op = compiledStage->value_op == ScanOp::Min ? ScanOp::ArgMin : ScanOp::ArgMax;
    auto arg_compiled = std::make_shared<CompiledScan>(arg_op,
                                                       compiledStage->mode,
                                                       compiledStage->axis,
                                                       compiledStage->reverse,
                                                       compiledStage->segmented_by_offsets,
                                                       compiledStage->input_dtype,
                                                       DataType::UINT32,
                                                       compiledStage->offset_dtype);
    auto arg_scan = std::make_shared<StampedScan>(arg_compiled, input, indices, stream, offsets, std::nullopt);
    auto scatter = prepareFlatScatterAdd(grad_output, indices, output);
    return std::make_shared<StampedScanMinMaxBackward>(compiledStage, arg_scan, scatter, input, grad_output, output, indices, stream);
}

StampedExecutionPlan FusedEquation::stampSingleOutput(const std::unordered_map<std::string, Tensor>& inputs,
                                                      const Stream& stream,
                                                      const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
                                                      const std::optional<Tensor>& preallocated_output,
                                                      const std::vector<uint64_t>& requestedOutputShape) const {
    std::unordered_map<std::string, Tensor> preallocated_outputs{};

    const auto outputNames = getOutputNames();
    if (outputNames.size() != 1) {
        throw std::runtime_error("Single-output stamp overload called on an equation that does not have exactly one output.");
    }
    if (outputNames.front() != "output") {
        throw std::runtime_error(
            "Single-output stamp overload requires the sole named output to be \"output\" when a preallocated "
            "output tensor is provided.");
    }

    if (preallocated_output.has_value()) {
        preallocated_outputs["output"] = preallocated_output.value();
    }

    return stamp(inputs, stream, tensor_scalar_inputs, preallocated_outputs, makeSingleOutputRequestedShapeMap(requestedOutputShape));
}

StampedExecutionPlan FusedEquation::stamp(const std::unordered_map<std::string, Tensor>& inputs,
                                          const Stream& stream,
                                          const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
                                          const std::unordered_map<std::string, Tensor>& preallocated_outputs,
                                          const std::unordered_map<std::string, std::vector<uint64_t>>& requestedOutputShapes) const {
    if (outputs_template.isConditional()) {
        if (backward_config.has_value()) {
            throw std::runtime_error("Graph-level conditional Outputs do not currently support compileBackward().");
        }
        const PhysicalConditionalOutputs& conditional = *outputs_template.conditional;

        FusedEquation predicate_equation = FusedEquation::compileWithOptions(conditional.predicate, device_num, use_fast_math);
        FusedEquation then_equation = FusedEquation::compileWithOptions(conditional.then_branch, device_num, use_fast_math);
        FusedEquation else_equation = FusedEquation::compileWithOptions(conditional.else_branch, device_num, use_fast_math);

        auto filter_tensor_inputs = [&](const FusedEquation& equation) {
            std::unordered_map<std::string, Tensor> filtered;
            for (const NamedInput& input : equation.root_inputs) {
                if (input.kind != NamedInput::Kind::Tensor) {
                    continue;
                }
                auto it = inputs.find(input.name);
                if (it != inputs.end()) {
                    filtered.emplace(input.name, it->second);
                }
            }
            return filtered;
        };
        auto filter_tensor_scalar_inputs = [&](const FusedEquation& equation) {
            std::unordered_map<std::string, TensorScalarBinding> filtered;
            for (const NamedInput& input : equation.root_inputs) {
                if (input.kind != NamedInput::Kind::TensorRuntimeScalar) {
                    continue;
                }
                auto it = tensor_scalar_inputs.find(input.name);
                if (it != tensor_scalar_inputs.end()) {
                    filtered.emplace(input.name, it->second);
                }
            }
            return filtered;
        };

        const auto predicate_inputs = filter_tensor_inputs(predicate_equation);
        const auto predicate_tensor_scalars = filter_tensor_scalar_inputs(predicate_equation);
        const auto then_inputs = filter_tensor_inputs(then_equation);
        const auto then_tensor_scalars = filter_tensor_scalar_inputs(then_equation);
        const auto else_inputs = filter_tensor_inputs(else_equation);
        const auto else_tensor_scalars = filter_tensor_scalar_inputs(else_equation);

        auto predicate_plan = std::make_shared<StampedExecutionPlan>(
            predicate_equation.stamp(predicate_inputs, stream, predicate_tensor_scalars, {}, {}));

        auto then_plan = std::make_shared<StampedExecutionPlan>(
            then_equation.stamp(then_inputs, stream, then_tensor_scalars, preallocated_outputs, requestedOutputShapes));

        std::unordered_map<std::string, Tensor> shared_outputs = then_plan->getFinalOutputs();
        for (const auto& [name, tensor] : preallocated_outputs) {
            (void)tensor;
            if (shared_outputs.find(name) == shared_outputs.end()) {
                throw std::runtime_error("Preallocated conditional output tensor was not consumed by the then branch: " + name);
            }
        }

        auto else_plan = std::make_shared<StampedExecutionPlan>(
            else_equation.stamp(else_inputs, stream, else_tensor_scalars, shared_outputs, requestedOutputShapes));

        std::vector<std::string> output_names;
        output_names.reserve(outputs_template.outputs.size());
        for (const NamedOutput& output : outputs_template.outputs) {
            output_names.push_back(output.name);
        }

        auto stamped_conditional = std::make_shared<StampedConditional>(
            std::move(predicate_plan), std::move(then_plan), std::move(else_plan), std::move(output_names), stream);

        std::vector<StampedExecutionStage> stages;
        stages.emplace_back(stamped_conditional);
        return StampedExecutionPlan(std::move(stages), std::move(shared_outputs), stream);
    }

    if (accumulatesIntoGradOutputs(backward_config) && preallocated_outputs.empty()) {
        throw std::runtime_error(
            "Backward equations compiled with accumulate_grad_outputs=true require caller-provided gradient output tensors when stamping.");
    }

    static const std::unordered_map<std::string, float> empty_scalar_inputs;

    const std::unordered_map<std::string, std::vector<uint64_t>> requestedOutputShapesWithOutputs =
        mergeRequestedOutputShapesWithProvidedOutputs(preallocated_outputs, requestedOutputShapes, backward_config);

    std::unordered_map<uint32_t, RuntimeInputValue> compile_root_values =
        accumulatesIntoGradOutputs(backward_config)
            ? bindRootInputs(inputs, empty_scalar_inputs, tensor_scalar_inputs, &preallocated_outputs)
            : bindRootInputsForCompilation(inputs, empty_scalar_inputs, tensor_scalar_inputs, requestedOutputShapesWithOutputs);
    const auto effectiveRequestedOutputShapes =
        defaultBackwardRequestedOutputShapes(backward_config, root_inputs, compile_root_values, requestedOutputShapesWithOutputs);

    if (accumulatesIntoGradOutputs(backward_config)) {
        validateBackwardAccumulationOutputs(backward_config, inputs, preallocated_outputs, effectiveRequestedOutputShapes);
    }

    const std::shared_ptr<CompiledOutputs> compiled_outputs = compileForRootValues(compile_root_values);

    std::unordered_map<uint32_t, RuntimeInputValue> values = accumulatesIntoGradOutputs(backward_config)
                                                                 ? compile_root_values
                                                                 : bindRootInputs(inputs, empty_scalar_inputs, tensor_scalar_inputs, {});

    std::unordered_map<std::string, Tensor> preallocated_final_outputs_by_name = preallocated_outputs;

    std::unordered_map<uint32_t, CompiledValueAlias> alias_by_value_id;
    alias_by_value_id.reserve(compiled_outputs->value_aliases.size());
    for (const CompiledValueAlias& alias : compiled_outputs->value_aliases) {
        alias_by_value_id.emplace(alias.value_id, alias);
    }

    auto ultimateAliasSourceValueId = [&](uint32_t value_id) -> std::optional<uint32_t> {
        uint32_t current = value_id;
        bool traversed_alias = false;
        std::unordered_set<uint32_t> visited;
        while (true) {
            auto alias_it = alias_by_value_id.find(current);
            if (alias_it == alias_by_value_id.end()) {
                return traversed_alias ? std::optional<uint32_t>(current) : std::nullopt;
            }
            if (!visited.insert(current).second) {
                throw std::runtime_error("Cycle detected while resolving reshape alias source value.");
            }
            current = alias_it->second.source_value_id;
            traversed_alias = true;
        }
    };

    std::unordered_map<uint32_t, Tensor> preallocated_outputs_by_source_value_id;
    for (const CompiledStageOutput& final_output : compiled_outputs->final_outputs) {
        auto preallocated_it = preallocated_final_outputs_by_name.find(final_output.name);
        if (preallocated_it == preallocated_final_outputs_by_name.end()) {
            continue;
        }
        std::optional<uint32_t> source_value_id = ultimateAliasSourceValueId(final_output.value_id);
        if (source_value_id.has_value()) {
            auto alias_meta_it = alias_by_value_id.find(final_output.value_id);
            if (alias_meta_it == alias_by_value_id.end() || alias_meta_it->second.strides.empty()) {
                preallocated_outputs_by_source_value_id[source_value_id.value()] = preallocated_it->second;
            }
        }
    }

    auto preallocatedForStageOutput = [&](const CompiledStageOutput& stage_output,
                                          const std::vector<uint64_t>& output_dims) -> std::optional<Tensor> {
        auto named_it = preallocated_final_outputs_by_name.find(stage_output.name);
        if (named_it != preallocated_final_outputs_by_name.end()) {
            return named_it->second;
        }
        auto value_it = preallocated_outputs_by_source_value_id.find(stage_output.value_id);
        if (value_it == preallocated_outputs_by_source_value_id.end()) {
            return std::nullopt;
        }
        Tensor tensor = value_it->second;
        std::vector<uint64_t> resolved_output_dims =
            resolveDynamicAliasDims(tensor.getDimensions(), output_dims, true, "Preallocated stage output alias");
        tensor.reshape(resolved_output_dims);
        return tensor;
    };

    std::vector<StampedExecutionStage> stampedStages;
    stampedStages.reserve(compiled_outputs->stages.size());

    std::unordered_map<uint32_t, uint32_t> producer_stage_by_value_id;
    producer_stage_by_value_id.reserve(compiled_outputs->stages.size() * 2);

    std::unordered_map<uint32_t, std::shared_ptr<AttentionForwardState>> attention_forward_state_by_stage_idx;
    attention_forward_state_by_stage_idx.reserve(compiled_outputs->stages.size());

    std::unordered_map<size_t, size_t> attention_stage_by_elided_packing_stage;
    std::unordered_map<size_t, PackedAttentionBackwardDirectOutput> direct_packed_attention_backward_by_stage =
        findPackedAttentionBackwardDirectOutputs(
            compiled_outputs->stages, compiled_outputs->final_outputs, attention_stage_by_elided_packing_stage);


    auto stageDependsOn = [&](const std::vector<uint32_t>& direct_dependencies, uint32_t candidate_stage_idx) {
        std::unordered_set<uint32_t> visited;
        std::function<bool(uint32_t)> visit = [&](uint32_t stage_idx) -> bool {
            if (stage_idx == candidate_stage_idx) {
                return true;
            }
            if (stage_idx >= stampedStages.size() || !visited.insert(stage_idx).second) {
                return false;
            }
            for (uint32_t dep : stampedStages[stage_idx].dependency_stage_indices) {
                if (visit(dep)) {
                    return true;
                }
            }
            return false;
        };

        for (uint32_t dep : direct_dependencies) {
            if (visit(dep)) {
                return true;
            }
        }
        return false;
    };

    applyAvailableValueAliases(compiled_outputs->value_aliases, values, &producer_stage_by_value_id);

    for (size_t stage_idx = 0; stage_idx < compiled_outputs->stages.size(); ++stage_idx) {
        const CompiledExecutionStage& stage = compiled_outputs->stages.at(stage_idx);
        applyAvailableValueAliases(compiled_outputs->value_aliases, values, &producer_stage_by_value_id);

        std::vector<RuntimeInputValue> stageInputs;
        stageInputs.reserve(stage.input_value_ids.size());

        std::vector<uint32_t> dependency_stage_indices;
        dependency_stage_indices.reserve(stage.input_value_ids.size());

        for (uint32_t value_id : stage.input_value_ids) {
            auto it = values.find(value_id);
            if (it == values.end()) {
                throw std::runtime_error("Missing input value for staged execution plan.");
            }

            stageInputs.push_back(it->second);

            auto producer_it = producer_stage_by_value_id.find(value_id);
            if (producer_it != producer_stage_by_value_id.end()) {
                const uint32_t dep_stage_idx = producer_it->second;
                if (std::find(dependency_stage_indices.begin(), dependency_stage_indices.end(), dep_stage_idx) ==
                    dependency_stage_indices.end()) {
                    dependency_stage_indices.push_back(dep_stage_idx);
                }
            }
        }

        std::vector<std::vector<uint64_t>> stage_input_dims;
        stage_input_dims.reserve(stageInputs.size());
        for (const RuntimeInputValue& input : stageInputs) {
            stage_input_dims.push_back(runtimeInputDims(input));
        }

        auto elided_pack_it = attention_stage_by_elided_packing_stage.find(stage_idx);
        if (elided_pack_it != attention_stage_by_elided_packing_stage.end()) {
            auto direct_it = direct_packed_attention_backward_by_stage.find(elided_pack_it->second);
            if (direct_it == direct_packed_attention_backward_by_stage.end() ||
                !values.contains(direct_it->second.packed_value_id)) {
                throw std::runtime_error("Packed attention-backward output was not produced before eliding the packing stage.");
            }
            applyAvailableValueAliases(compiled_outputs->value_aliases, values, &producer_stage_by_value_id);
            continue;
        }

        const uint64_t stage_flops = computeStageFlops(stage, stage_input_dims);

        switch (stage.kind) {
            case CompiledExecutionStage::Kind::FusedKernel: {
                if (stage.outputs.empty()) {
                    throw std::runtime_error("Fused stage requires at least one output.");
                }
                std::vector<std::vector<uint64_t>> expected_output_dims(stage.outputs.size());
                std::shared_ptr<CompiledEquation> compiledEq;

                std::vector<uint64_t> resolved_output_dims;
                const bool requires_broadcast = fusedStageRequiresBroadcastLaunch(
                    stage, stageInputs, effectiveRequestedOutputShapes, backward_config.has_value(), resolved_output_dims);

                if (!requires_broadcast) {
                    for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                        std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, output_idx, stageInputs);
                        auto requested_it = effectiveRequestedOutputShapes.find(stage.outputs[output_idx].name);
                        if (requested_it != effectiveRequestedOutputShapes.end() && !requested_it->second.empty()) {
                            if (!(stageInputs.empty() && output_dims.empty())) {
                                verifyRequestedOutputLayout(requested_it->second, output_dims);
                            }
                            output_dims = requested_it->second;
                        }
                        expected_output_dims[output_idx] = std::move(output_dims);
                    }
                    compiledEq = selectFlatCompiledEquation(stage, compiled_outputs->signature, maxNumel(expected_output_dims));
                } else {
                    std::vector<ResolvedBroadcastGroup> groups = buildResolvedBroadcastGroups(stage, stageInputs);
                    if (groups.empty()) {
                        throw std::runtime_error("Fused stage expected at least one broadcast group.");
                    }

                    std::vector<SpecializedBroadcastGroup> specialized_groups;
                    specialized_groups.reserve(groups.size());
                    for (const ResolvedBroadcastGroup& group : groups) {
                        specialized_groups.push_back(group.specialized);
                        for (uint32_t output_idx : group.specialized.output_indices) {
                            if (output_idx >= expected_output_dims.size()) {
                                throw std::runtime_error("Broadcast group output index out of range.");
                            }
                            std::vector<uint64_t> output_dims =
                                stage.outputs[output_idx].materialized_layout == MaterializedTensorLayout::Transposed
                                    ? resolveOutputDimsForStageOutput(stage, output_idx, stageInputs)
                                    : group.specialized.output_dims;
                            auto requested_it = effectiveRequestedOutputShapes.find(stage.outputs[output_idx].name);
                            if (requested_it != effectiveRequestedOutputShapes.end() && !requested_it->second.empty()) {
                                if (!(stageInputs.empty() && output_dims.empty())) {
                                    verifyRequestedOutputLayout(requested_it->second, output_dims);
                                }
                                output_dims = requested_it->second;
                            }
                            expected_output_dims[output_idx] = std::move(output_dims);
                        }
                    }

                    compiledEq = EquationCompiler::compileSpecializedBroadcastStage(stage, compiled_outputs->signature, specialized_groups);
                }

                std::vector<Tensor> stageOutputs;
                stageOutputs.reserve(stage.outputs.size());
                const TensorPlacement outputPlacement = pickStageOutputPlacement(stageInputs, values);
                for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                    const CompiledStageOutput& stageOutput = stage.outputs[output_idx];
                    auto preallocated_it = preallocated_final_outputs_by_name.find(stageOutput.name);
                    if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                        const Tensor& outputTensor = preallocated_it->second;

                        if (!outputTensor.isInitialized()) {
                            throw std::runtime_error("Preallocated fused-stage output tensor is not initialized.");
                        }
                        if (outputTensor.getPlacement() != outputPlacement) {
                            throw std::runtime_error(
                                "Preallocated fused-stage output tensor placement does not match the expected output placement.");
                        }
                        if (outputTensor.getDescriptor().getDataType() != compiledEq->output_dtypes.at(output_idx)) {
                            throw std::runtime_error(
                                "Preallocated fused-stage output tensor dtype does not match the compiled output dtype.");
                        }
                        if (outputTensor.getDimensions() != expected_output_dims[output_idx]) {
                            throw std::runtime_error(
                                "Preallocated fused-stage output tensor dimensions are incompatible with the staged output shape.");
                        }

                        stageOutputs.push_back(outputTensor);
                        values[stageOutput.value_id] = outputTensor;
                    } else {
                        TensorDescriptor outputDescriptor(compiledEq->output_dtypes.at(output_idx), expected_output_dims[output_idx]);
                        Tensor outputTensor(outputPlacement, outputDescriptor);
                        stageOutputs.push_back(outputTensor);
                        values[stageOutput.value_id] = outputTensor;
                    }
                    producer_stage_by_value_id[stageOutput.value_id] = static_cast<uint32_t>(stampedStages.size());
                }

                std::shared_ptr<StampedEquation> stampedKernel = stampEquation(compiledEq, stageInputs, stageOutputs, stream);
                stampedStages.emplace_back(stampedKernel, std::move(dependency_stage_indices), stage_flops);
                break;
            }
            case CompiledExecutionStage::Kind::CudaKernel: {
                if (!stage.cuda_kernel_expression || !stage.cuda_kernel) {
                    throw std::runtime_error("CudaKernel compiled stage missing expression or compiled kernel.");
                }
                const auto& input_specs = stage.cuda_kernel_expression->inputs();
                if (input_specs.size() != stageInputs.size()) {
                    throw std::runtime_error("CudaKernel stage input count mismatch while stamping.");
                }

                std::unordered_map<std::string, Tensor> bound_inputs;
                std::unordered_map<std::string, TensorScalarBinding> bound_tensor_scalar_inputs;
                bound_inputs.reserve(input_specs.size());
                bound_tensor_scalar_inputs.reserve(input_specs.size());
                for (size_t i = 0; i < input_specs.size(); ++i) {
                    if (input_specs[i].kind == CudaKernelExpression::TensorParamSpec::Kind::TensorRuntimeScalar) {
                        if (!runtimeInputIsTensorScalarBinding(stageInputs[i])) {
                            throw std::runtime_error("CudaKernel stage expected tensor runtime scalar input: " + input_specs[i].name);
                        }
                        bound_tensor_scalar_inputs.emplace(input_specs[i].name, runtimeInputTensorScalarBinding(stageInputs[i]));
                    } else if (input_specs[i].kind == CudaKernelExpression::TensorParamSpec::Kind::HostRuntimeScalar) {
                        if (!runtimeInputIsRuntimeScalar(stageInputs[i])) {
                            throw std::runtime_error("CudaKernel stage expected host runtime scalar input: " + input_specs[i].name);
                        }
                    } else {
                        if (!runtimeInputIsTensor(stageInputs[i])) {
                            throw std::runtime_error("CudaKernel stage expected tensor input: " + input_specs[i].name);
                        }
                        bound_inputs.emplace(input_specs[i].name, runtimeInputTensor(stageInputs[i]));
                    }
                }

                std::unordered_map<std::string, Tensor> preallocated_outputs;
                std::unordered_map<std::string, std::vector<uint64_t>> requested_output_shapes;
                for (const CompiledStageOutput& stage_output : stage.outputs) {
                    if (stage_output.local_node_idx >= stage.expr.nodes.size()) {
                        throw std::runtime_error("CudaKernel stage output local node index out of range.");
                    }
                    const ExprNode& output_node = stage.expr.nodes[stage_output.local_node_idx];
                    if (output_node.cuda_kernel_output_index >= stage.cuda_kernel_expression->outputs().size()) {
                        throw std::runtime_error("CudaKernel stage output spec index out of range.");
                    }
                    const std::string& output_name = stage.cuda_kernel_expression->outputs()[output_node.cuda_kernel_output_index].name;
                    if (!stage_output.name.empty()) {
                        auto preallocated_it = preallocated_final_outputs_by_name.find(stage_output.name);
                        if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                            preallocated_outputs.emplace(output_name, preallocated_it->second);
                        }
                        auto requested_it = effectiveRequestedOutputShapes.find(stage_output.name);
                        if (requested_it != effectiveRequestedOutputShapes.end()) {
                            requested_output_shapes.emplace(output_name, requested_it->second);
                        }
                    }
                }

                std::unordered_map<std::string, Tensor> resolved_outputs;
                std::shared_ptr<StampedCudaKernel> stampedCudaKernel = stage.cuda_kernel_expression->stampCompiled(
                    stage.cuda_kernel,
                    bound_inputs,
                    preallocated_outputs,
                    requested_output_shapes,
                    stream,
                    resolved_outputs,
                    bound_tensor_scalar_inputs);

                for (const CompiledStageOutput& stage_output : stage.outputs) {
                    const ExprNode& output_node = stage.expr.nodes.at(stage_output.local_node_idx);
                    const std::string& output_name = stage.cuda_kernel_expression->outputs()[output_node.cuda_kernel_output_index].name;
                    auto output_it = resolved_outputs.find(output_name);
                    if (output_it == resolved_outputs.end()) {
                        throw std::runtime_error("CudaKernel stage did not resolve requested output '" + output_name + "'.");
                    }
                    values[stage_output.value_id] = output_it->second;
                    producer_stage_by_value_id[stage_output.value_id] = static_cast<uint32_t>(stage_idx);
                }

                stampedStages.emplace_back(stampedCudaKernel, std::move(dependency_stage_indices), stage_flops);
                break;
            }

            case CompiledExecutionStage::Kind::Reduction: {
                if (!stage.reduction) {
                    throw std::runtime_error("Reduction stage missing compiled reduction payload.");
                }
                if (stageInputs.size() != 1) {
                    throw std::runtime_error("Reduction stage expects exactly one input.");
                }
                Tensor inputTensor = runtimeInputTensor(stageInputs[0]);
                if (stage.outputs.size() != 1) {
                    throw std::runtime_error("Reduction stage expects exactly one output.");
                }
                const CompiledStageOutput& stageOutput = stage.outputs[0];

                std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, 0, stageInputs);
                auto requested_it = effectiveRequestedOutputShapes.find(stageOutput.name);
                if (requested_it != effectiveRequestedOutputShapes.end() && !requested_it->second.empty()) {
                    verifyRequestedOutputLayout(requested_it->second, output_dims);
                    output_dims = requested_it->second;
                }

                auto preallocated_it = preallocated_final_outputs_by_name.find(stageOutput.name);
                Tensor outputTensor;
                if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                    outputTensor = preallocated_it->second;
                } else {
                    TensorDescriptor outputDescriptor(stage.reduction->output_dtype, output_dims);
                    outputTensor = Tensor(inputTensor.getPlacement(), outputDescriptor);
                }

                std::shared_ptr<StampedReduction> stampedReduction =
                    stampReduction(stage.reduction, inputTensor, outputTensor, stream, output_dims);

                values[stageOutput.value_id] = outputTensor;
                producer_stage_by_value_id[stageOutput.value_id] = static_cast<uint32_t>(stampedStages.size());
                stampedStages.emplace_back(stampedReduction, std::move(dependency_stage_indices), stage_flops);
                break;
            }
            case CompiledExecutionStage::Kind::ArgMinMax: {
                if (!stage.arg_minmax) {
                    throw std::runtime_error("Argmin/argmax stage missing compiled payload.");
                }
                if (stageInputs.size() != 1) {
                    throw std::runtime_error("Argmin/argmax stage expects exactly one input.");
                }
                Tensor inputTensor = runtimeInputTensor(stageInputs[0]);
                if (stage.outputs.size() != 1) {
                    throw std::runtime_error("Argmin/argmax stage expects exactly one output.");
                }
                const CompiledStageOutput& stageOutput = stage.outputs[0];

                std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, 0, stageInputs);
                auto requested_it = effectiveRequestedOutputShapes.find(stageOutput.name);
                if (requested_it != effectiveRequestedOutputShapes.end() && !requested_it->second.empty()) {
                    verifyRequestedOutputLayout(requested_it->second, output_dims);
                    output_dims = requested_it->second;
                }

                auto preallocated_it = preallocated_final_outputs_by_name.find(stageOutput.name);
                Tensor outputTensor;
                if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                    outputTensor = preallocated_it->second;
                } else {
                    TensorDescriptor outputDescriptor(stage.arg_minmax->output_dtype, output_dims);
                    outputTensor = Tensor(inputTensor.getPlacement(), outputDescriptor);
                }

                std::shared_ptr<StampedArgMinMax> stampedArgMinMax =
                    stampArgMinMax(stage.arg_minmax, inputTensor, outputTensor, stream, output_dims);

                values[stageOutput.value_id] = outputTensor;
                producer_stage_by_value_id[stageOutput.value_id] = static_cast<uint32_t>(stampedStages.size());
                stampedStages.emplace_back(stampedArgMinMax, std::move(dependency_stage_indices), stage_flops);
                break;
            }
            case CompiledExecutionStage::Kind::SegmentedReduction: {
                if (!stage.segmented_reduction) {
                    throw std::runtime_error("Segmented-reduction stage missing compiled payload.");
                }
                if (stageInputs.size() != 2) {
                    throw std::runtime_error("Segmented-reduction stage expects values and offsets inputs.");
                }
                if (stage.outputs.size() != 1) {
                    throw std::runtime_error("Segmented-reduction stage expects exactly one output.");
                }

                Tensor inputTensor = runtimeInputTensor(stageInputs[0]);
                Tensor offsetsTensor = runtimeInputTensor(stageInputs[1]);
                const CompiledStageOutput& stageOutput = stage.outputs[0];

                std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, 0, stageInputs);
                auto requested_it = effectiveRequestedOutputShapes.find(stageOutput.name);
                if (requested_it != effectiveRequestedOutputShapes.end() && !requested_it->second.empty()) {
                    verifyRequestedOutputLayout(requested_it->second, output_dims);
                    output_dims = requested_it->second;
                }

                auto preallocated_it = preallocated_final_outputs_by_name.find(stageOutput.name);
                Tensor outputTensor;
                if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                    outputTensor = preallocated_it->second;
                } else {
                    TensorDescriptor outputDescriptor(stage.segmented_reduction->output_dtype, output_dims);
                    outputTensor = Tensor(inputTensor.getPlacement(), outputDescriptor);
                }

                std::shared_ptr<StampedSegmentedReduction> stampedSegmentedReduction =
                    std::make_shared<StampedSegmentedReduction>(stage.segmented_reduction, inputTensor, outputTensor, offsetsTensor, stream);

                values[stageOutput.value_id] = outputTensor;
                producer_stage_by_value_id[stageOutput.value_id] = static_cast<uint32_t>(stampedStages.size());
                stampedStages.emplace_back(stampedSegmentedReduction, std::move(dependency_stage_indices), stage_flops);
                break;
            }
            case CompiledExecutionStage::Kind::Scan: {
                if (!stage.scan) {
                    throw std::runtime_error("Scan stage missing compiled payload.");
                }
                const size_t expected_scan_inputs = stage.scan->segmented_by_offsets ? 2 : 1;
                if (stageInputs.size() != expected_scan_inputs) {
                    throw std::runtime_error("Scan stage expects its compiled input count.");
                }
                if (stage.outputs.empty() || stage.outputs.size() > 2) {
                    throw std::runtime_error("Scan stage expects one output, or paired value/index outputs.");
                }

                Tensor inputTensor = runtimeInputTensor(stageInputs[0]);
                auto allocateScanOutput = [&](size_t output_idx) -> Tensor {
                    const CompiledStageOutput& stageOutput = stage.outputs[output_idx];
                    std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, output_idx, stageInputs);
                    auto requested_it = effectiveRequestedOutputShapes.find(stageOutput.name);
                    if (requested_it != effectiveRequestedOutputShapes.end() && !requested_it->second.empty()) {
                        verifyRequestedOutputLayout(requested_it->second, output_dims);
                        output_dims = requested_it->second;
                    }

                    auto preallocated_it = preallocated_final_outputs_by_name.find(stageOutput.name);
                    if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                        return preallocated_it->second;
                    }
                    TensorDescriptor outputDescriptor(stage.outputDType(output_idx), output_dims);
                    return Tensor(inputTensor.getPlacement(), outputDescriptor);
                };

                std::vector<Tensor> outputTensors;
                outputTensors.reserve(stage.outputs.size());
                for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                    outputTensors.push_back(allocateScanOutput(output_idx));
                }

                size_t index_output_idx = 0;
                std::optional<size_t> value_output_idx = std::nullopt;
                if (stage.outputs.size() == 2) {
                    for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                        const ExprNode& outputNode = stage.expr.nodes.at(stage.outputs[output_idx].local_node_idx);
                        if (outputNode.scan_op == ScanOp::ArgMin || outputNode.scan_op == ScanOp::ArgMax) {
                            index_output_idx = output_idx;
                        } else {
                            value_output_idx = output_idx;
                        }
                    }
                    if (!value_output_idx.has_value()) {
                        throw std::runtime_error("Paired scan stage is missing its value output.");
                    }
                }

                std::optional<Tensor> segmentOffsets = std::nullopt;
                if (stage.scan->segmented_by_offsets) {
                    segmentOffsets = runtimeInputTensor(stageInputs[1]);
                }
                std::optional<Tensor> valueOutput =
                    value_output_idx.has_value() ? std::optional<Tensor>(outputTensors[*value_output_idx]) : std::nullopt;
                std::shared_ptr<StampedScan> stampedScan = std::make_shared<StampedScan>(
                    stage.scan, inputTensor, outputTensors[index_output_idx], stream, segmentOffsets, valueOutput);

                const uint32_t producer_stage_idx = static_cast<uint32_t>(stampedStages.size());
                for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                    const CompiledStageOutput& stageOutput = stage.outputs[output_idx];
                    values[stageOutput.value_id] = outputTensors[output_idx];
                    producer_stage_by_value_id[stageOutput.value_id] = producer_stage_idx;
                }
                stampedStages.emplace_back(stampedScan, std::move(dependency_stage_indices), stage_flops);
                break;
            }
            case CompiledExecutionStage::Kind::Softmax: {
                if (!stage.softmax) {
                    throw std::runtime_error("Softmax stage missing compiled payload.");
                }
                if (stageInputs.size() != 1) {
                    throw std::runtime_error("Softmax stage expects exactly one input.");
                }
                if (stage.outputs.size() != 1) {
                    throw std::runtime_error("Softmax stage expects exactly one output.");
                }

                Tensor inputTensor = runtimeInputTensor(stageInputs[0]);
                const CompiledStageOutput& stageOutput = stage.outputs[0];
                std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, 0, stageInputs);
                auto requested_it = effectiveRequestedOutputShapes.find(stageOutput.name);
                if (requested_it != effectiveRequestedOutputShapes.end() && !requested_it->second.empty()) {
                    verifyRequestedOutputLayout(requested_it->second, output_dims);
                    output_dims = requested_it->second;
                }

                auto preallocated_it = preallocated_final_outputs_by_name.find(stageOutput.name);
                std::optional<Tensor> preallocated = std::nullopt;
                if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                    preallocated = preallocated_it->second;
                }

                std::shared_ptr<StampedSoftmax> stampedSoftmax =
                    stampSoftmax(stage.softmax, inputTensor, preallocated, stream, output_dims);
                Tensor outputTensor = stampedSoftmax->getOutputTensor();

                values[stageOutput.value_id] = outputTensor;
                producer_stage_by_value_id[stageOutput.value_id] = static_cast<uint32_t>(stampedStages.size());
                stampedStages.emplace_back(stampedSoftmax, std::move(dependency_stage_indices), stage_flops);
                break;
            }
            case CompiledExecutionStage::Kind::RmsNorm: {
                if (!stage.rms_norm) {
                    throw std::runtime_error("RMSNorm stage missing compiled payload.");
                }
                if (stageInputs.size() != 2) {
                    throw std::runtime_error("RMSNorm stage expects exactly two inputs.");
                }
                if (stage.outputs.size() != 1) {
                    throw std::runtime_error("RMSNorm stage expects exactly one output.");
                }

                Tensor inputTensor = runtimeInputTensor(stageInputs[0]);
                Tensor scaleTensor = runtimeInputTensor(stageInputs[1]);
                const CompiledStageOutput& stageOutput = stage.outputs[0];
                const std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, 0, stageInputs);
                std::optional<Tensor> preallocated = preallocatedForStageOutput(stageOutput, output_dims);

                std::shared_ptr<StampedRmsNorm> stampedRmsNorm =
                    stampRmsNorm(stage.rms_norm, inputTensor, scaleTensor, preallocated, stream);
                Tensor outputTensor = stampedRmsNorm->getOutputTensor();

                values[stageOutput.value_id] = outputTensor;
                producer_stage_by_value_id[stageOutput.value_id] = static_cast<uint32_t>(stampedStages.size());
                stampedStages.emplace_back(stampedRmsNorm, std::move(dependency_stage_indices), stage_flops);
                break;
            }
            case CompiledExecutionStage::Kind::EmbeddingLookup: {
                if (!stage.embedding_lookup) {
                    throw std::runtime_error("EmbeddingLookup stage missing compiled payload.");
                }
                if (stageInputs.size() < 2) {
                    throw std::runtime_error("EmbeddingLookup stage expects at least two inputs: indices and weights.");
                }
                if (stage.outputs.size() != 1) {
                    throw std::runtime_error("EmbeddingLookup stage expects exactly one output.");
                }
                Tensor indicesTensor = runtimeInputTensor(stageInputs[0]);
                Tensor weightsTensor = runtimeInputTensor(stageInputs[1]);
                std::vector<Tensor> epilogueInputs;
                epilogueInputs.reserve(stageInputs.size() > 2 ? stageInputs.size() - 2 : 0);
                for (size_t i = 2; i < stageInputs.size(); ++i) {
                    Tensor epilogueTensor = runtimeInputTensor(stageInputs[i]);
                    epilogueInputs.push_back(epilogueTensor);
                }
                const CompiledStageOutput& stageOutput = stage.outputs[0];
                const std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, 0, stageInputs);
                auto preallocated_it = preallocated_final_outputs_by_name.find(stageOutput.name);
                Tensor outputTensor;
                if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                    outputTensor = preallocated_it->second;
                } else {
                    outputTensor = Tensor(weightsTensor.getPlacement(), TensorDescriptor(stage.embedding_lookup->output_dtype, output_dims));
                }
                for (const Tensor& epilogueTensor : epilogueInputs) {
                    if (epilogueTensor.getDimensions() != output_dims) {
                        throw std::runtime_error("EmbeddingLookup fused epilogue tensor inputs must have exactly the output dimensions.");
                    }
                    if (epilogueTensor.getDataType() != stage.embedding_lookup->output_dtype) {
                        throw std::runtime_error("EmbeddingLookup fused epilogue tensor inputs must match the output dtype.");
                    }
                }
                std::shared_ptr<StampedEmbeddingLookup> stampedEmbeddingLookup =
                    std::make_shared<StampedEmbeddingLookup>(stage.embedding_lookup,
                                                             indicesTensor,
                                                             weightsTensor,
                                                             outputTensor,
                                                             stream,
                                                             std::move(epilogueInputs));
                values[stageOutput.value_id] = outputTensor;
                producer_stage_by_value_id[stageOutput.value_id] = static_cast<uint32_t>(stampedStages.size());
                stampedStages.emplace_back(stampedEmbeddingLookup, std::move(dependency_stage_indices), stage_flops);
                break;
            }
            case CompiledExecutionStage::Kind::Matmul: {
                if (!stage.matmul) {
                    throw std::runtime_error("Matmul/gemm stage missing compiled payload.");
                }
                if (stage.outputs.empty() || stage.outputs.size() > 2) {
                    throw std::runtime_error("Matmul/gemm stage expects one matrix output and at most one bias-gradient output.");
                }
                const CompiledStageOutput& matrixStageOutput = stage.outputs[0];
                const std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, 0, stageInputs);
                std::optional<Tensor> preallocated = preallocatedForStageOutput(matrixStageOutput, output_dims);
                std::optional<Tensor> preallocated_bgrad = std::nullopt;
                std::vector<uint64_t> bgrad_output_dims;
                if (stage.outputs.size() > 1) {
                    bgrad_output_dims = resolveOutputDimsForStageOutput(stage, 1, stageInputs);
                    preallocated_bgrad = preallocatedForStageOutput(stage.outputs[1], bgrad_output_dims);
                }

                std::shared_ptr<StampedMatmul> stampedMatmul;
                std::optional<RuntimeInputValue> alpha_input = std::nullopt;
                std::optional<RuntimeInputValue> beta_input = std::nullopt;
                std::optional<std::string> alpha_runtime_name = std::nullopt;
                std::optional<std::string> beta_runtime_name = std::nullopt;
                std::optional<Tensor> epilogue_aux = std::nullopt;

                auto runtimeScalarNameForStageLocalSlot = [&](uint32_t local_slot) -> std::optional<std::string> {
                    if (local_slot == UINT32_MAX) {
                        return std::nullopt;
                    }
                    if (local_slot >= stage.input_value_ids.size()) {
                        throw std::runtime_error("Matmul stage runtime scalar local slot is out of range.");
                    }

                    const uint32_t root_value_id = stage.input_value_ids[local_slot];
                    if (root_value_id >= root_inputs.size()) {
                        return std::nullopt;
                    }

                    const NamedInput& root_input = root_inputs[root_value_id];
                    if (root_input.kind != NamedInput::Kind::RuntimeScalarFp32) {
                        return std::nullopt;
                    }

                    return root_input.name;
                };

                if (stage.matmul->alpha_input_slot != UINT32_MAX) {
                    if (stage.matmul->alpha_input_slot >= stageInputs.size()) {
                        throw std::runtime_error("Matmul stage alpha runtime scalar slot is out of range.");
                    }
                    alpha_input = stageInputs[stage.matmul->alpha_input_slot];
                    alpha_runtime_name = runtimeScalarNameForStageLocalSlot(stage.matmul->alpha_input_slot);
                }

                if (stage.matmul->beta_input_slot != UINT32_MAX) {
                    if (stage.matmul->beta_input_slot >= stageInputs.size()) {
                        throw std::runtime_error("Matmul stage beta runtime scalar slot is out of range.");
                    }
                    beta_input = stageInputs[stage.matmul->beta_input_slot];
                    beta_runtime_name = runtimeScalarNameForStageLocalSlot(stage.matmul->beta_input_slot);
                }

                if (stage.matmul->epilogue_aux_input_slot != UINT32_MAX) {
                    if (stage.matmul->epilogue_aux_input_slot >= stageInputs.size()) {
                        throw std::runtime_error("Matmul stage backward epilogue aux input slot is out of range.");
                    }
                    epilogue_aux = runtimeInputTensor(stageInputs[stage.matmul->epilogue_aux_input_slot]);
                }

                if (stage.matmul->op == ExprOp::MATMUL) {
                    if (stageInputs.size() < 2) {
                        throw std::runtime_error("Matmul stage expects at least two inputs.");
                    }
                    Tensor lhsTensor = runtimeInputTensor(stageInputs[0]);
                    Tensor rhsTensor = runtimeInputTensor(stageInputs[1]);
                    stampedMatmul = stampMatmul(stage.matmul,
                                                lhsTensor,
                                                rhsTensor,
                                                preallocated,
                                                stream,
                                                alpha_input,
                                                beta_input,
                                                alpha_runtime_name,
                                                beta_runtime_name,
                                                epilogue_aux,
                                                preallocated_bgrad);
                } else {
                    if (stageInputs.size() < 3) {
                        throw std::runtime_error("GEMM stage expects at least three inputs.");
                    }
                    Tensor lhsTensor = runtimeInputTensor(stageInputs[0]);
                    Tensor rhsTensor = runtimeInputTensor(stageInputs[1]);
                    Tensor addendTensor = runtimeInputTensor(stageInputs[2]);
                    stampedMatmul = stampMatmul(stage.matmul,
                                                lhsTensor,
                                                rhsTensor,
                                                addendTensor,
                                                preallocated,
                                                stream,
                                                alpha_input,
                                                beta_input,
                                                alpha_runtime_name,
                                                beta_runtime_name,
                                                epilogue_aux,
                                                preallocated_bgrad);
                }
                Tensor outputTensor = stampedMatmul->getOutputTensor();
                if (outputTensor.getDimensions() != output_dims) {
                    throw std::runtime_error("Stamped matmul/gemm output tensor dimensions are incompatible with the staged output shape.");
                }
                values[matrixStageOutput.value_id] = outputTensor;
                producer_stage_by_value_id[matrixStageOutput.value_id] = static_cast<uint32_t>(stampedStages.size());

                if (stage.outputs.size() > 1) {
                    std::optional<Tensor> bgradTensor = stampedMatmul->getBiasGradientTensor();
                    if (!bgradTensor.has_value()) {
                        throw std::runtime_error("Matmul stage expected a fused bias-gradient output but none was stamped.");
                    }
                    if (bgradTensor->getDimensions() != bgrad_output_dims) {
                        throw std::runtime_error("Stamped matmul/gemm bias-gradient tensor dimensions are incompatible with the staged output shape.");
                    }
                    values[stage.outputs[1].value_id] = bgradTensor.value();
                    producer_stage_by_value_id[stage.outputs[1].value_id] = static_cast<uint32_t>(stampedStages.size());
                }

                stampedStages.emplace_back(stampedMatmul, std::move(dependency_stage_indices), stage_flops);
                break;
            }
            case CompiledExecutionStage::Kind::InPlaceRope: {
                if (!stage.in_place_rope) {
                    throw std::runtime_error("In-place RoPE stage missing compiled payload.");
                }
                if (stageInputs.size() != stage.in_place_rope->tensors.size() || stage.outputs.size() != stage.in_place_rope->tensors.size()) {
                    throw std::runtime_error("In-place RoPE stage input/output count mismatch.");
                }

                std::vector<Tensor> ropeTensors;
                ropeTensors.reserve(stageInputs.size());
                for (size_t i = 0; i < stageInputs.size(); ++i) {
                    Tensor tensor = runtimeInputTensor(stageInputs[i]);
                    if (tensor.getDimensions() != stage.in_place_rope->tensors[i].logical_dims) {
                        tensor.reshape(stage.in_place_rope->tensors[i].logical_dims);
                    }
                    if (tensor.getDataType() != stage.in_place_rope->tensors[i].dtype) {
                        throw std::runtime_error("In-place RoPE tensor dtype mismatch.");
                    }
                    ropeTensors.push_back(tensor);
                }

                auto stampedRope = std::make_shared<StampedInPlaceRope>(stage.in_place_rope, ropeTensors, stream);
                const uint32_t rope_stage_idx = static_cast<uint32_t>(stampedStages.size());
                for (size_t i = 0; i < stage.outputs.size(); ++i) {
                    values[stage.outputs[i].value_id] = stampedRope->outputTensor(i);
                    producer_stage_by_value_id[stage.outputs[i].value_id] = rope_stage_idx;
                }
                stampedStages.emplace_back(stampedRope, std::move(dependency_stage_indices), stage_flops);
                break;
            }
            case CompiledExecutionStage::Kind::Attention: {
                if (!stage.attention) {
                    throw std::runtime_error("Attention stage missing compiled payload.");
                }
                const size_t expected_attention_stage_inputs =
                    3 + (stage.attention->use_bias ? 1 : 0) + (stage.attention->use_padding_mask ? 2 : 0) +
                    (stage.attention->use_ragged_offsets ? 2 : 0) + (stage.attention->use_paged_kv_cache ? 2 : 0) +
                    (stage.attention->dropout_probability > 0.0f ? 2 : 0) + (stage.attention->use_fp8_forward_scaling ? 8 : 0);
                if (stageInputs.size() != expected_attention_stage_inputs) {
                    throw std::runtime_error(
                        "Attention stage input count mismatch for q/k/v plus optional bias, optional q/kv sequence lengths, optional "
                        "ragged offsets, optional paged-KV page tables, optional dropout seed/offset, and optional FP8 scale/descale/amax tensors.");
                }
                if (stage.outputs.size() != 1) {
                    throw std::runtime_error("Attention stage expects exactly one output.");
                }
                Tensor qTensor = runtimeInputTensor(stageInputs[0]);
                Tensor kTensor = runtimeInputTensor(stageInputs[1]);
                Tensor vTensor = runtimeInputTensor(stageInputs[2]);
                size_t next_attention_input = 3;
                std::optional<Tensor> biasTensor = std::nullopt;
                if (stage.attention->use_bias) {
                    biasTensor = runtimeInputTensor(stageInputs[next_attention_input++]);
                }
                std::optional<Tensor> seqLenQTensor = std::nullopt;
                std::optional<Tensor> seqLenKvTensor = std::nullopt;
                if (stage.attention->use_padding_mask) {
                    seqLenQTensor = runtimeInputTensor(stageInputs[next_attention_input++]);
                    seqLenKvTensor = runtimeInputTensor(stageInputs[next_attention_input++]);
                }
                std::optional<Tensor> qRaggedOffsetsTensor = std::nullopt;
                std::optional<Tensor> kvRaggedOffsetsTensor = std::nullopt;
                if (stage.attention->use_ragged_offsets) {
                    qRaggedOffsetsTensor = runtimeInputTensor(stageInputs[next_attention_input++]);
                    kvRaggedOffsetsTensor = runtimeInputTensor(stageInputs[next_attention_input++]);
                }
                std::optional<Tensor> pageTableKTensor = std::nullopt;
                std::optional<Tensor> pageTableVTensor = std::nullopt;
                if (stage.attention->use_paged_kv_cache) {
                    pageTableKTensor = runtimeInputTensor(stageInputs[next_attention_input++]);
                    pageTableVTensor = runtimeInputTensor(stageInputs[next_attention_input++]);
                }
                std::optional<Tensor> dropoutSeedTensor = std::nullopt;
                std::optional<Tensor> dropoutOffsetTensor = std::nullopt;
                if (stage.attention->dropout_probability > 0.0f) {
                    dropoutSeedTensor = runtimeInputTensorScalarView(stageInputs[next_attention_input++], "attention dropout seed");
                    dropoutOffsetTensor = runtimeInputTensorScalarView(stageInputs[next_attention_input++], "attention dropout offset");
                }
                std::optional<Tensor> descaleQTensor = std::nullopt;
                std::optional<Tensor> descaleKTensor = std::nullopt;
                std::optional<Tensor> descaleVTensor = std::nullopt;
                std::optional<Tensor> descaleSTensor = std::nullopt;
                std::optional<Tensor> scaleSTensor = std::nullopt;
                std::optional<Tensor> scaleOTensor = std::nullopt;
                std::optional<Tensor> amaxSTensor = std::nullopt;
                std::optional<Tensor> amaxOTensor = std::nullopt;
                if (stage.attention->use_fp8_forward_scaling) {
                    descaleQTensor = runtimeInputTensor(stageInputs[next_attention_input++]);
                    descaleKTensor = runtimeInputTensor(stageInputs[next_attention_input++]);
                    descaleVTensor = runtimeInputTensor(stageInputs[next_attention_input++]);
                    descaleSTensor = runtimeInputTensor(stageInputs[next_attention_input++]);
                    scaleSTensor = runtimeInputTensor(stageInputs[next_attention_input++]);
                    scaleOTensor = runtimeInputTensor(stageInputs[next_attention_input++]);
                    amaxSTensor = runtimeInputTensor(stageInputs[next_attention_input++]);
                    amaxOTensor = runtimeInputTensor(stageInputs[next_attention_input++]);
                }
                const CompiledStageOutput& stageOutput = stage.outputs[0];
                const std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, 0, stageInputs);
                auto preallocated_it = preallocated_final_outputs_by_name.find(stageOutput.name);
                std::optional<Tensor> preallocated = std::nullopt;
                if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                    preallocated = preallocated_it->second;
                }
                std::shared_ptr<AttentionForwardState> forwardState = std::make_shared<AttentionForwardState>();
                std::shared_ptr<StampedAttention> stampedAttention = stampAttention(stage.attention,
                                                                                    qTensor,
                                                                                    kTensor,
                                                                                    vTensor,
                                                                                    biasTensor,
                                                                                    seqLenQTensor,
                                                                                    seqLenKvTensor,
                                                                                    qRaggedOffsetsTensor,
                                                                                    kvRaggedOffsetsTensor,
                                                                                    pageTableKTensor,
                                                                                    pageTableVTensor,
                                                                                    dropoutSeedTensor,
                                                                                    dropoutOffsetTensor,
                                                                                    descaleQTensor,
                                                                                    descaleKTensor,
                                                                                    descaleVTensor,
                                                                                    descaleSTensor,
                                                                                    scaleSTensor,
                                                                                    scaleOTensor,
                                                                                    amaxSTensor,
                                                                                    amaxOTensor,
                                                                                    preallocated,
                                                                                    stream,
                                                                                    forwardState);
                Tensor outputTensor = stampedAttention->getOutputTensor();
                if (outputTensor.getDimensions() != output_dims) {
                    throw std::runtime_error("Stamped attention output tensor dimensions are incompatible with the staged output shape.");
                }
                values[stageOutput.value_id] = outputTensor;
                const uint32_t attention_stage_idx = static_cast<uint32_t>(stampedStages.size());
                producer_stage_by_value_id[stageOutput.value_id] = attention_stage_idx;
                attention_forward_state_by_stage_idx[attention_stage_idx] = std::move(forwardState);
                stampedStages.emplace_back(stampedAttention, std::move(dependency_stage_indices), stage_flops);
                break;
            }
            case CompiledExecutionStage::Kind::AttentionBackward: {
                if (!stage.attention_backward) {
                    throw std::runtime_error("Attention-backward stage missing compiled payload.");
                }
                const size_t expected_attention_backward_stage_inputs =
                    4 + (stage.attention_backward && stage.attention_backward->use_bias ? 1 : 0) +
                    (stage.attention_backward && stage.attention_backward->use_padding_mask ? 2 : 0) +
                    (stage.attention_backward && stage.attention_backward->use_ragged_offsets ? 2 : 0) +
                    (stage.attention_backward && stage.attention_backward->dropout_probability > 0.0f ? 2 : 0);
                if (stageInputs.size() != expected_attention_backward_stage_inputs) {
                    throw std::runtime_error(
                        "Attention-backward stage input count mismatch for q/k/v/dO plus optional bias, optional q/kv sequence lengths, "
                        "optional ragged offsets, and optional dropout seed/offset.");
                }
                Tensor qTensor = runtimeInputTensor(stageInputs[0]);
                Tensor kTensor = runtimeInputTensor(stageInputs[1]);
                Tensor vTensor = runtimeInputTensor(stageInputs[2]);
                Tensor dOTensor = runtimeInputTensor(stageInputs[3]);
                size_t next_attention_backward_input = 4;
                std::optional<Tensor> biasTensor = std::nullopt;
                if (stage.attention_backward && stage.attention_backward->use_bias) {
                    biasTensor = runtimeInputTensor(stageInputs[next_attention_backward_input++]);
                }
                std::optional<Tensor> seqLenQTensor = std::nullopt;
                std::optional<Tensor> seqLenKvTensor = std::nullopt;
                if (stage.attention_backward && stage.attention_backward->use_padding_mask) {
                    seqLenQTensor = runtimeInputTensor(stageInputs[next_attention_backward_input++]);
                    seqLenKvTensor = runtimeInputTensor(stageInputs[next_attention_backward_input++]);
                }
                std::optional<Tensor> qRaggedOffsetsTensor = std::nullopt;
                std::optional<Tensor> kvRaggedOffsetsTensor = std::nullopt;
                if (stage.attention_backward && stage.attention_backward->use_ragged_offsets) {
                    qRaggedOffsetsTensor = runtimeInputTensor(stageInputs[next_attention_backward_input++]);
                    kvRaggedOffsetsTensor = runtimeInputTensor(stageInputs[next_attention_backward_input++]);
                }
                std::optional<Tensor> dropoutSeedTensor = std::nullopt;
                std::optional<Tensor> dropoutOffsetTensor = std::nullopt;
                if (stage.attention_backward && stage.attention_backward->dropout_probability > 0.0f) {
                    dropoutSeedTensor =
                        runtimeInputTensorScalarView(stageInputs[next_attention_backward_input++], "attention-backward dropout seed");
                    dropoutOffsetTensor =
                        runtimeInputTensorScalarView(stageInputs[next_attention_backward_input++], "attention-backward dropout offset");
                }

                std::vector<std::optional<Tensor>> preallocated(4, std::nullopt);
                for (size_t i = 0; i < stage.outputs.size(); ++i) {
                    if (stage.outputs[i].local_node_idx >= stage.expr.nodes.size()) {
                        throw std::runtime_error("Attention-backward stage output node index out of range while stamping.");
                    }
                    const ExprOp out_op = stage.expr.nodes[stage.outputs[i].local_node_idx].op;
                    size_t slot = out_op == ExprOp::ATTENTION_BACKWARD_Q      ? 0
                                  : out_op == ExprOp::ATTENTION_BACKWARD_K    ? 1
                                  : out_op == ExprOp::ATTENTION_BACKWARD_V    ? 2
                                  : out_op == ExprOp::ATTENTION_BACKWARD_BIAS ? 3
                                                                              : 4;
                    if (slot >= 4) {
                        throw std::runtime_error("Attention-backward stage output op is invalid while stamping.");
                    }
                    auto preallocated_it = preallocated_final_outputs_by_name.find(stage.outputs[i].name);
                    if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                        preallocated[slot] = preallocated_it->second;
                    }
                }

                std::optional<Tensor> directPackedOutput = std::nullopt;
                auto direct_it = direct_packed_attention_backward_by_stage.find(stage_idx);
                if (direct_it != direct_packed_attention_backward_by_stage.end()) {
                    const PackedAttentionBackwardDirectOutput& direct = direct_it->second;
                    std::vector<uint64_t> packed_dims = direct.packed_dims;
                    if (!direct.packed_output_name.empty()) {
                        auto requested_it = effectiveRequestedOutputShapes.find(direct.packed_output_name);
                        if (requested_it != effectiveRequestedOutputShapes.end() && !requested_it->second.empty()) {
                            verifyRequestedOutputLayout(requested_it->second, packed_dims);
                            packed_dims = requested_it->second;
                        }
                    }

                    auto preallocated_it = preallocated_final_outputs_by_name.find(direct.packed_output_name);
                    if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                        Tensor out = preallocated_it->second;
                        if (!out.isInitialized()) {
                            throw std::runtime_error("Preallocated packed attention-backward output tensor is not initialized.");
                        }
                        if (out.getPlacement() != qTensor.getPlacement()) {
                            throw std::runtime_error(
                                "Preallocated packed attention-backward output placement does not match q placement.");
                        }
                        if (out.getDescriptor().getDataType() != direct.packed_dtype) {
                            throw std::runtime_error("Preallocated packed attention-backward output dtype does not match compiled dtype.");
                        }
                        if (out.getDimensions() != packed_dims) {
                            throw std::runtime_error(
                                "Preallocated packed attention-backward output dimensions are incompatible with the packed gradient shape.");
                        }
                        directPackedOutput = out;
                    } else {
                        directPackedOutput = Tensor(qTensor.getPlacement(), TensorDescriptor(direct.packed_dtype, packed_dims));
                    }

                    if (directPackedOutput->getDimensions() != direct.packed_dims) {
                        throw std::runtime_error(
                            "Packed attention-backward output requested shape must match the canonical packed dQKV storage shape.");
                    }

                    for (size_t slot = 0; slot < direct.slices.size(); ++slot) {
                        if (!direct.slices.at(slot).has_value()) {
                            throw std::runtime_error("Packed attention-backward direct-output metadata is missing a Q/K/V slice.");
                        }
                        const PackedAttentionBackwardDirectSlice& slice = direct.slices.at(slot).value();
                        preallocated.at(slot) = directPackedOutput->aliasView(slice.view_dims, slice.view_strides, slice.view_element_offset);
                    }
                }

                std::shared_ptr<AttentionForwardState> savedForwardState = nullptr;
                for (const auto& [candidate_stage_idx, candidate_state] : attention_forward_state_by_stage_idx) {
                    if (!candidate_state || !stageDependsOn(dependency_stage_indices, candidate_stage_idx)) {
                        continue;
                    }
                    const StampedExecutionStage& candidate_stage = stampedStages.at(candidate_stage_idx);
                    if (candidate_stage.kind != StampedExecutionStage::Kind::Attention || !candidate_stage.attention) {
                        continue;
                    }
                    if (candidate_stage.attention->canProvideForwardStateFor(*stage.attention_backward,
                                                                             qTensor,
                                                                             kTensor,
                                                                             vTensor,
                                                                             biasTensor,
                                                                             seqLenQTensor,
                                                                             seqLenKvTensor,
                                                                             qRaggedOffsetsTensor,
                                                                             kvRaggedOffsetsTensor,
                                                                             dropoutSeedTensor,
                                                                             dropoutOffsetTensor,
                                                                             dOTensor)) {
                        if (!candidate_state->stats.isInitialized()) {
                            const std::vector<uint64_t> o_dims = candidate_state->output.getDimensions();
                            TensorDescriptor statsDescriptor(DataType::FP32,
                                                             {o_dims.at(0), o_dims.at(1), o_dims.at(2), 1});
                            candidate_state->stats = Tensor(candidate_state->output.getPlacement(), statsDescriptor);
                        }
                        candidate_state->retain_for_backward = true;
                        savedForwardState = candidate_state;
                        break;
                    }
                }

                std::shared_ptr<StampedAttentionBackward> stampedAttentionBackward = stampAttentionBackward(stage.attention_backward,
                                                                                                            qTensor,
                                                                                                            kTensor,
                                                                                                            vTensor,
                                                                                                            biasTensor,
                                                                                                            seqLenQTensor,
                                                                                                            seqLenKvTensor,
                                                                                                            qRaggedOffsetsTensor,
                                                                                                            kvRaggedOffsetsTensor,
                                                                                                            dropoutSeedTensor,
                                                                                                            dropoutOffsetTensor,
                                                                                                            dOTensor,
                                                                                                            preallocated,
                                                                                                            stream,
                                                                                                            savedForwardState);
                const std::vector<Tensor>& outputTensors = stampedAttentionBackward->getOutputTensors();
                const size_t expected_attention_backward_outputs = stage.attention_backward->use_bias ? 4 : 3;
                if (outputTensors.size() != expected_attention_backward_outputs) {
                    throw std::runtime_error("Stamped attention-backward exposed an unexpected number of output tensors.");
                }
                for (size_t i = 0; i < stage.outputs.size(); ++i) {
                    const ExprOp out_op = stage.expr.nodes[stage.outputs[i].local_node_idx].op;
                    size_t slot = out_op == ExprOp::ATTENTION_BACKWARD_Q      ? 0
                                  : out_op == ExprOp::ATTENTION_BACKWARD_K    ? 1
                                  : out_op == ExprOp::ATTENTION_BACKWARD_V    ? 2
                                  : out_op == ExprOp::ATTENTION_BACKWARD_BIAS ? 3
                                                                              : 4;
                    if (slot >= outputTensors.size()) {
                        throw std::runtime_error("Attention-backward stage output op is invalid while wiring outputs.");
                    }
                    const std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, i, stageInputs);
                    if (outputTensors.at(slot).getDimensions() != output_dims) {
                        throw std::runtime_error("Stamped attention-backward output tensor dimensions are incompatible with staged shape.");
                    }
                    values[stage.outputs[i].value_id] = outputTensors.at(slot);
                    producer_stage_by_value_id[stage.outputs[i].value_id] = static_cast<uint32_t>(stampedStages.size());
                }
                if (directPackedOutput.has_value()) {
                    auto direct_it = direct_packed_attention_backward_by_stage.find(stage_idx);
                    if (direct_it == direct_packed_attention_backward_by_stage.end()) {
                        throw std::runtime_error("Internal error: missing packed attention-backward direct-output metadata.");
                    }
                    values[direct_it->second.packed_value_id] = directPackedOutput.value();
                    producer_stage_by_value_id[direct_it->second.packed_value_id] = static_cast<uint32_t>(stampedStages.size());
                }
                stampedStages.emplace_back(stampedAttentionBackward, std::move(dependency_stage_indices), stage_flops);
                break;
            }
            case CompiledExecutionStage::Kind::Convolution: {
                if (!stage.convolution) {
                    throw std::runtime_error("Convolution stage missing compiled payload.");
                }
                if (stageInputs.size() != 2) {
                    throw std::runtime_error("Convolution stage expects exactly two inputs.");
                }
                if (stage.outputs.size() != 1) {
                    throw std::runtime_error("Convolution stage expects exactly one output.");
                }
                Tensor inputTensor = runtimeInputTensor(stageInputs[0]);
                Tensor filterTensor = runtimeInputTensor(stageInputs[1]);
                const CompiledStageOutput& stageOutput = stage.outputs[0];
                const std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, 0, stageInputs);
                auto preallocated_it = preallocated_final_outputs_by_name.find(stageOutput.name);
                std::optional<Tensor> preallocated = std::nullopt;
                if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                    preallocated = preallocated_it->second;
                }
                std::shared_ptr<StampedConvolution> stampedConvolution =
                    stampConvolution(stage.convolution, inputTensor, filterTensor, preallocated, stream);
                Tensor outputTensor = stampedConvolution->getOutputTensor();
                if (outputTensor.getDimensions() != output_dims) {
                    throw std::runtime_error("Stamped convolution output tensor dimensions are incompatible with the staged output shape.");
                }
                values[stageOutput.value_id] = outputTensor;
                producer_stage_by_value_id[stageOutput.value_id] = static_cast<uint32_t>(stampedStages.size());
                stampedStages.emplace_back(stampedConvolution, std::move(dependency_stage_indices), stage_flops);
                break;
            }
            case CompiledExecutionStage::Kind::ConvolutionBackward: {
                if (!stage.convolution_backward) {
                    throw std::runtime_error("Convolution-backward stage missing compiled payload.");
                }
                if (stageInputs.size() != 2) {
                    throw std::runtime_error("Convolution-backward stage expects exactly two inputs.");
                }
                if (stage.outputs.size() != 1) {
                    throw std::runtime_error("Convolution-backward stage expects exactly one output.");
                }
                Tensor inputTensor = runtimeInputTensor(stageInputs[0]);
                Tensor gradOutputTensor = runtimeInputTensor(stageInputs[1]);
                const CompiledStageOutput& stageOutput = stage.outputs[0];
                const std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, 0, stageInputs);
                auto preallocated_it = preallocated_final_outputs_by_name.find(stageOutput.name);
                std::optional<Tensor> preallocated = std::nullopt;
                if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                    preallocated = preallocated_it->second;
                }
                std::shared_ptr<StampedConvolutionBackward> stampedConvolutionBackward =
                    stampConvolutionBackward(stage.convolution_backward, inputTensor, gradOutputTensor, preallocated, stream);
                Tensor outputTensor = stampedConvolutionBackward->getOutputTensor();
                if (outputTensor.getDimensions() != output_dims) {
                    throw std::runtime_error(
                        "Stamped convolution-backward output tensor dimensions are incompatible with the staged output shape.");
                }
                values[stageOutput.value_id] = outputTensor;
                producer_stage_by_value_id[stageOutput.value_id] = static_cast<uint32_t>(stampedStages.size());
                stampedStages.emplace_back(stampedConvolutionBackward, std::move(dependency_stage_indices), stage_flops);
                break;
            }
            case CompiledExecutionStage::Kind::ReduceMinMaxBackward: {
                if (!stage.reduce_minmax_backward) {
                    throw std::runtime_error("Reduce-min/max-backward stage missing compiled payload.");
                }
                if (stageInputs.size() != 2) {
                    throw std::runtime_error("Reduce-min/max-backward stage expects exactly two inputs.");
                }
                Tensor inputTensor = runtimeInputTensor(stageInputs[0]);
                Tensor gradOutputTensor = runtimeInputTensor(stageInputs[1]);
                if (stage.outputs.size() != 1) {
                    throw std::runtime_error("Reduce-min/max-backward stage expects exactly one output.");
                }
                const CompiledStageOutput& stageOutput = stage.outputs[0];
                const std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, 0, stageInputs);
                auto preallocated_it = preallocated_final_outputs_by_name.find(stageOutput.name);
                Tensor outputTensor;
                if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                    outputTensor = preallocated_it->second;
                } else {
                    TensorDescriptor outputDescriptor(stage.reduce_minmax_backward->output_dtype, output_dims);
                    outputTensor = Tensor(inputTensor.getPlacement(), outputDescriptor);
                }
                std::shared_ptr<StampedReduceMinMaxBackward> stampedReduceMinMaxBackward =
                    stampReduceMinMaxBackward(stage.reduce_minmax_backward, inputTensor, gradOutputTensor, outputTensor, stream);
                values[stageOutput.value_id] = outputTensor;
                producer_stage_by_value_id[stageOutput.value_id] = static_cast<uint32_t>(stampedStages.size());
                stampedStages.emplace_back(stampedReduceMinMaxBackward, std::move(dependency_stage_indices), stage_flops);
                break;
            }
            case CompiledExecutionStage::Kind::ScanMinMaxBackward: {
                if (!stage.scan_minmax_backward) {
                    throw std::runtime_error("Scan-min/max-backward stage missing compiled payload.");
                }
                const size_t expected_inputs = stage.scan_minmax_backward->segmented_by_offsets ? 3 : 2;
                if (stageInputs.size() != expected_inputs) {
                    throw std::runtime_error("Scan-min/max-backward stage expects input, grad, and optional offsets inputs.");
                }
                if (stage.outputs.size() != 1) {
                    throw std::runtime_error("Scan-min/max-backward stage expects exactly one output.");
                }
                Tensor inputTensor = runtimeInputTensor(stageInputs[0]);
                Tensor gradOutputTensor = runtimeInputTensor(stageInputs[1]);
                std::optional<Tensor> offsetsTensor = std::nullopt;
                if (stage.scan_minmax_backward->segmented_by_offsets) {
                    offsetsTensor = runtimeInputTensor(stageInputs[2]);
                }
                const CompiledStageOutput& stageOutput = stage.outputs[0];
                const std::vector<uint64_t> output_dims = resolveOutputDimsForStageOutput(stage, 0, stageInputs);
                auto preallocated_it = preallocated_final_outputs_by_name.find(stageOutput.name);
                Tensor outputTensor;
                if (preallocated_it != preallocated_final_outputs_by_name.end()) {
                    outputTensor = preallocated_it->second;
                } else {
                    TensorDescriptor outputDescriptor(stage.scan_minmax_backward->output_dtype, output_dims);
                    outputTensor = Tensor(inputTensor.getPlacement(), outputDescriptor);
                }
                std::shared_ptr<StampedScanMinMaxBackward> stampedScanMinMaxBackward =
                    stampScanMinMaxBackward(stage.scan_minmax_backward, inputTensor, gradOutputTensor, offsetsTensor, outputTensor, stream);
                values[stageOutput.value_id] = outputTensor;
                producer_stage_by_value_id[stageOutput.value_id] = static_cast<uint32_t>(stampedStages.size());
                stampedStages.emplace_back(stampedScanMinMaxBackward, std::move(dependency_stage_indices), stage_flops);
                break;
            }
        }
        applyAvailableValueAliases(compiled_outputs->value_aliases, values, &producer_stage_by_value_id);
    }

    applyAvailableValueAliases(compiled_outputs->value_aliases, values, &producer_stage_by_value_id);

    std::unordered_map<std::string, Tensor> finalOutputsByName;
    finalOutputsByName.reserve(compiled_outputs->final_outputs.size());
    for (const CompiledStageOutput& final_output : compiled_outputs->final_outputs) {
        auto it = values.find(final_output.value_id);
        if (it == values.end()) {
            throw std::runtime_error("Missing final output tensor for output: " + final_output.name);
        }
        finalOutputsByName.emplace(final_output.name, runtimeInputTensor(it->second));
    }

    return StampedExecutionPlan(std::move(stampedStages), std::move(finalOutputsByName), stream);
}

void FusedEquation::run(const Tensor& input, Tensor& output, Stream& stream) const {
    if (externalRootInputCount(root_inputs, backward_config) != 1) {
        throw std::runtime_error("FusedEquation::run was only passed a single input, but this equation requires " +
                                 std::to_string(externalRootInputCount(root_inputs, backward_config)) +
                                 " inputs. "
                                 "Pass a dict of name -> PhysicalTensor of inputs to run it.");
    }

    const std::string& output_name = outputs_template.outputs[0].name;
    std::unordered_map<std::string, Tensor> input_map = {{root_inputs[0].name, input}};
    std::unordered_map<std::string, Tensor> output_map = {{output_name, output}};

    run(input_map, output_map, stream);
}

void FusedEquation::run(const std::unordered_map<std::string, Tensor>& inputs, Tensor& output, Stream& stream) const {
    static const std::unordered_map<std::string, float> empty_scalar_inputs;
    run(inputs, empty_scalar_inputs, output, stream);
}

void FusedEquation::run(const std::unordered_map<std::string, Tensor>& inputs,
                        const std::unordered_map<std::string, float>& scalar_inputs,
                        Tensor& output,
                        Stream& stream) const {
    if (outputs_template.outputs.size() != 1) {
        throw std::runtime_error(
            "FusedEquation::run was only passed a single output, but this equation has multiple named outputs. "
            "Pass a dict of name -> PhysicalTensor of outputs to run it.");
    }

    const std::string& output_name = outputs_template.outputs[0].name;
    std::unordered_map<std::string, Tensor> output_map = {{output_name, output}};
    run(inputs, scalar_inputs, output_map, stream);
}

void FusedEquation::run(const Tensor& input, std::unordered_map<std::string, Tensor>& outputs, Stream& stream) const {
    if (externalRootInputCount(root_inputs, backward_config) != 1) {
        throw std::runtime_error("FusedEquation::run was only passed a single input, but this equation requires " +
                                 std::to_string(externalRootInputCount(root_inputs, backward_config)) +
                                 " inputs. "
                                 "Pass a dict of name -> PhysicalTensor of inputs to run it.");
    }

    const std::string& input_name = root_inputs[0].name;
    std::unordered_map<std::string, Tensor> input_map = {{input_name, input}};

    run(input_map, outputs, stream);
}

void FusedEquation::run(const std::unordered_map<std::string, Tensor>& inputs,
                        std::unordered_map<std::string, Tensor>& outputs,
                        Stream& stream) const {
    static const std::unordered_map<std::string, float> empty_scalar_inputs;
    run(inputs, empty_scalar_inputs, outputs, stream);
}

void FusedEquation::run(const std::unordered_map<std::string, Tensor>& inputs,
                        const std::unordered_map<std::string, float>& scalar_inputs,
                        std::unordered_map<std::string, Tensor>& outputs,
                        Stream& stream) const {
    const std::unordered_map<uint32_t, RuntimeInputValue> root_values = bindRootInputs(inputs, scalar_inputs, {}, &outputs);
    const std::shared_ptr<PreparedConvenienceRunPlan> prepared_plan = prepareConvenienceRunPlan(root_values);
    const std::shared_ptr<CompiledOutputs>& compiled_outputs = prepared_plan->compiled_outputs;
    std::unordered_map<uint32_t, RuntimeInputValue> available_values = root_values;
    applyAvailableValueAliases(compiled_outputs->value_aliases, available_values);

    if (compiled_outputs->stages.empty()) {
        throw std::runtime_error("Expression has no execution stages.");
    }

    std::unordered_set<std::string> expected_output_names(prepared_plan->expected_output_names_in_order.begin(),
                                                          prepared_plan->expected_output_names_in_order.end());

    for (const std::string& name : prepared_plan->expected_output_names_in_order) {
        auto it = outputs.find(name);
        if (it == outputs.end()) {
            throw std::runtime_error("Missing output tensor '" + name + "' for fused equation run.");
        }
    }

    for (const auto& [name, tensor] : outputs) {
        (void)tensor;
        if (!expected_output_names.contains(name)) {
            std::string expected_names_str;
            for (size_t i = 0; i < prepared_plan->expected_output_names_in_order.size(); ++i) {
                if (i > 0) {
                    expected_names_str += ", ";
                }
                expected_names_str += "'" + prepared_plan->expected_output_names_in_order[i] + "'";
            }
            throw std::runtime_error("Unexpected output tensor '" + name +
                                     "' passed to fused equation run. "
                                     "Expected output names: [" +
                                     expected_names_str + "].");
        }
    }

    int32_t gpu_num = -1;
    for (const auto& [value_id, value] : root_values) {
        (void)value_id;
        std::optional<TensorPlacement> placement = runtimeInputPlacementOrNull(value);
        if (placement.has_value()) {
            gpu_num = placement.value().getDeviceNum();
            break;
        }
    }
    if (gpu_num < 0) {
        if (outputs.empty()) {
            throw std::runtime_error("FusedEquation::run requires at least one tensor input or output.");
        }
        gpu_num = outputs.begin()->second.getPlacement().getDeviceNum();
    }

    for (const auto& [value_id, value] : root_values) {
        (void)value_id;
        std::optional<TensorPlacement> placement = runtimeInputPlacementOrNull(value);
        if (placement.has_value()) {
            if (placement.value().getDeviceNum() != gpu_num) {
                throw std::runtime_error("FusedEquation::run requires all root tensor inputs to be on the same GPU.");
            }
        }
    }
    for (const auto& [name, tensor] : outputs) {
        (void)name;
        if (tensor.getPlacement().getDeviceNum() != gpu_num) {
            throw std::runtime_error("FusedEquation::run requires all outputs to be on the same GPU.");
        }
        if (!tensor.isDenseContiguous()) {
            throw std::runtime_error("FusedEquation::run convenience outputs must be dense contiguous tensors.");
        }
    }

    auto runStageOnStream = [&](const CompiledExecutionStage& stage,
                                const PreparedConvenienceRunStage& prepared_stage,
                                const std::vector<RuntimeInputValue>& orderedInputs,
                                const std::vector<Tensor>& orderedOutputs,
                                Stream& launch_stream) {
        if (orderedOutputs.size() != prepared_stage.expected_output_dims.size()) {
            throw std::runtime_error("Prepared convenience run stage output count mismatch.");
        }

        for (size_t i = 0; i < orderedOutputs.size(); ++i) {
            verifyRequestedOutputLayout(orderedOutputs[i].getDimensions(), prepared_stage.expected_output_dims[i]);
        }

        EquationRunner::run(prepared_stage.compiled_equation, orderedInputs, orderedOutputs, launch_stream);
    };

    std::vector<Stream> helper_streams_used;
    helper_streams_used.reserve(compiled_outputs->stages.size());

    auto rememberHelperStream = [&](Stream& helper_stream) {
        if (std::find(helper_streams_used.begin(), helper_streams_used.end(), helper_stream) == helper_streams_used.end()) {
            helper_streams_used.push_back(helper_stream);
        }
    };

    for (uint32_t stage_num = 0; stage_num < compiled_outputs->stages.size(); ++stage_num) {
        bool use_helper_streams = (stage_num != 0);
        const CompiledExecutionStage& stage = compiled_outputs->stages[stage_num];
        const PreparedConvenienceRunStage& prepared_stage = prepared_plan->stages[stage_num];

        applyAvailableValueAliases(compiled_outputs->value_aliases, available_values);

        std::vector<RuntimeInputValue> orderedInputs;
        orderedInputs.reserve(stage.input_value_ids.size());

        for (uint32_t value_id : stage.input_value_ids) {
            auto it = available_values.find(value_id);
            if (it == available_values.end()) {
                throw std::runtime_error("Missing input value for fused equation run.");
            }
            orderedInputs.push_back(it->second);
        }

        std::vector<Tensor> orderedOutputs;
        orderedOutputs.reserve(stage.outputs.size());

        for (const auto& stage_output : stage.outputs) {
            auto it = outputs.find(stage_output.name);
            if (it == outputs.end()) {
                throw std::runtime_error("Missing output tensor '" + stage_output.name + "' for fused equation run.");
            }
            orderedOutputs.push_back(it->second);
        }

        if (use_helper_streams) {
            Stream& helper_stream = Expression::getNextHelperStream(gpu_num);
            runStageOnStream(stage, prepared_stage, orderedInputs, orderedOutputs, helper_stream);
            rememberHelperStream(helper_stream);
        } else {
            runStageOnStream(stage, prepared_stage, orderedInputs, orderedOutputs, stream);
        }
    }

    for (Stream& helper_stream : helper_streams_used) {
        stream.waitEvent(helper_stream.putEvent());
    }
}

FusedEquation::ParameterFanOverrideMap FusedEquation::getParameterFanOverrides(
    const std::unordered_map<std::string, Tensor>& named_inputs,
    const std::unordered_set<std::string>& parameter_names,
    const std::unordered_map<std::string, TensorScalarBinding>& tensor_scalar_inputs,
    const std::unordered_map<std::string, std::vector<uint64_t>>& requested_output_shapes) const {
    ParameterFanOverrideMap result;

    const auto root_values = bindRootInputsForCompilation(named_inputs, {}, tensor_scalar_inputs, requested_output_shapes);
    const std::shared_ptr<CompiledOutputs> compiled_outputs = compileForRootValues(root_values);

    std::unordered_map<uint32_t, std::string> root_input_name_by_slot;
    root_input_name_by_slot.reserve(root_inputs.size());
    for (const NamedInput& input : root_inputs) {
        if (input.kind == NamedInput::Kind::Tensor) {
            root_input_name_by_slot.emplace(input.slot, input.name);
        }
    }

    std::unordered_map<uint32_t, std::vector<uint64_t>> value_dims;
    value_dims.reserve(root_values.size() + compiled_outputs->stages.size());
    for (const auto& [value_id, value] : root_values) {
        value_dims.emplace(value_id, runtimeInputDims(value));
    }
    applyAvailableValueAliases(compiled_outputs->value_aliases, value_dims);

    for (const CompiledExecutionStage& stage : compiled_outputs->stages) {
        std::vector<std::vector<uint64_t>> stage_input_dims;
        stage_input_dims.reserve(stage.input_value_ids.size());
        for (uint32_t value_id : stage.input_value_ids) {
            auto it = value_dims.find(value_id);
            if (it == value_dims.end()) {
                throw std::runtime_error("Missing input shape for parameter fan override inference.");
            }
            stage_input_dims.push_back(it->second);
        }

        std::vector<std::vector<uint64_t>> resolved_stage_output_dims;
        if (stage.kind == CompiledExecutionStage::Kind::FusedKernel) {
            resolved_stage_output_dims.reserve(stage.outputs.size());
            for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                resolved_stage_output_dims.push_back(resolveOutputDimsForStageOutput(stage, output_idx, stage_input_dims));
            }
            addFusedKernelParameterFanOverrides(
                stage, stage_input_dims, resolved_stage_output_dims, root_input_name_by_slot, parameter_names, result);
        } else if (stage.kind == CompiledExecutionStage::Kind::CudaKernel) {
            // User-defined CUDA kernels do not currently infer initializer fan overrides.
        } else if (stage.kind == CompiledExecutionStage::Kind::Matmul) {
            addMatmulParameterFanOverrides(stage, stage_input_dims, root_input_name_by_slot, parameter_names, result);
        } else if (stage.kind == CompiledExecutionStage::Kind::InPlaceRope) {
            // In-place RoPE has no trainable parameters.
        } else if (stage.kind == CompiledExecutionStage::Kind::Attention) {
            // Attention stages do not currently infer initializer fan overrides for q/k/v inputs.
        } else if (stage.kind == CompiledExecutionStage::Kind::AttentionBackward) {
            // Attention-backward stages do not infer initializer fan overrides.
        } else if (stage.kind == CompiledExecutionStage::Kind::Convolution) {
            addConvolutionParameterFanOverrides(stage, stage_input_dims, root_input_name_by_slot, parameter_names, result);
        } else if (stage.kind == CompiledExecutionStage::Kind::ConvolutionBackward) {
            // No parameter fan overrides are inferred from backward convolution stages.
        }

        for (const ParameterFanOverride& hint : stage.parameter_fan_overrides) {
            if (!parameter_names.contains(hint.input_name)) {
                continue;
            }
            mergeParameterFanOverride(result, hint);
        }

        if (stage.kind == CompiledExecutionStage::Kind::FusedKernel) {
            for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
                value_dims[stage.outputs[output_idx].value_id] = resolved_stage_output_dims[output_idx];
            }
        } else if (stage.kind == CompiledExecutionStage::Kind::CudaKernel) {
            if (!stage.cuda_kernel_expression) {
                throw std::runtime_error("Missing compiled CUDA kernel expression stage.");
            }
            std::unordered_map<std::string, std::vector<uint64_t>> input_shapes;
            const auto& input_specs = stage.cuda_kernel_expression->inputs();
            if (input_specs.size() != stage_input_dims.size()) {
                throw std::runtime_error("CUDA kernel stage input shape count mismatch.");
            }
            for (size_t i = 0; i < input_specs.size(); ++i) {
                input_shapes.emplace(input_specs[i].name, stage_input_dims[i]);
            }
            const std::vector<std::vector<uint64_t>> output_shapes =
                stage.cuda_kernel_expression->inferOutputShapesFromInputShapes(input_shapes);
            for (const CompiledStageOutput& output : stage.outputs) {
                if (output.local_node_idx >= stage.expr.nodes.size()) {
                    throw std::runtime_error("CUDA kernel stage output node index out of range.");
                }
                const ExprNode& output_node = stage.expr.nodes[output.local_node_idx];
                if (output_node.cuda_kernel_output_index >= output_shapes.size()) {
                    throw std::runtime_error("CUDA kernel stage output spec index out of range.");
                }
                value_dims[output.value_id] = output_shapes[output_node.cuda_kernel_output_index];
            }
        } else if (stage.kind == CompiledExecutionStage::Kind::Reduction) {
            if (!stage.reduction) {
                throw std::runtime_error("Missing compiled reduction stage.");
            }
            if (stage.input_value_ids.size() != 1 || stage.outputs.size() != 1) {
                throw std::runtime_error("Reduction stage expected exactly one input and one output.");
            }
            value_dims[stage.outputs[0].value_id] = StampedEquation::computeReductionOutputDims(
                stage_input_dims[0], stage.reduction->reduction_axes, stage.reduction->squeeze_axes);
        } else if (stage.kind == CompiledExecutionStage::Kind::ArgMinMax) {
            if (!stage.arg_minmax) {
                throw std::runtime_error("Missing compiled arg-min/max stage.");
            }
            if (stage.input_value_ids.size() != 1 || stage.outputs.size() != 1) {
                throw std::runtime_error("Arg-min/max stage expected exactly one input and one output.");
            }
            value_dims[stage.outputs[0].value_id] = StampedEquation::computeReductionOutputDims(
                stage_input_dims[0], stage.arg_minmax->reduction_axes, stage.arg_minmax->squeeze_axes);
        } else if (stage.kind == CompiledExecutionStage::Kind::SegmentedReduction) {
            if (!stage.segmented_reduction) {
                throw std::runtime_error("Missing compiled segmented-reduction stage.");
            }
            if (stage.input_value_ids.size() != 2 || stage.outputs.size() != 1) {
                throw std::runtime_error("Segmented-reduction stage expected values, offsets, and one output.");
            }
            if (stage_input_dims[0].size() != 1 || stage_input_dims[1].size() != 1 || stage_input_dims[1][0] == 0) {
                throw std::runtime_error("Segmented-reduction stage currently requires rank-1 values and non-empty rank-1 offsets.");
            }
            value_dims[stage.outputs[0].value_id] = std::vector<uint64_t>{stage_input_dims[1][0] - 1};
        } else if (stage.kind == CompiledExecutionStage::Kind::Scan) {
            if (!stage.scan) {
                throw std::runtime_error("Missing compiled scan stage.");
            }
            const size_t expected_scan_inputs = stage.scan->segmented_by_offsets ? 2 : 1;
            if (stage.input_value_ids.size() != expected_scan_inputs || stage.outputs.empty() || stage.outputs.size() > 2) {
                throw std::runtime_error("Scan stage expected its compiled input count and one or two outputs.");
            }
            for (const CompiledStageOutput& output : stage.outputs) {
                value_dims[output.value_id] = stage_input_dims[0];
            }
        } else if (stage.kind == CompiledExecutionStage::Kind::Softmax) {
            if (!stage.softmax) {
                throw std::runtime_error("Missing compiled softmax stage.");
            }
            if (stage.input_value_ids.size() != 1 || stage.outputs.size() != 1) {
                throw std::runtime_error("Softmax stage expected exactly one input and one output.");
            }
            value_dims[stage.outputs[0].value_id] = stage_input_dims[0];
        } else if (stage.kind == CompiledExecutionStage::Kind::RmsNorm) {
            if (!stage.rms_norm) {
                throw std::runtime_error("Missing compiled RMSNorm stage.");
            }
            if (stage.input_value_ids.size() != 2 || stage.outputs.size() != 1) {
                throw std::runtime_error("RMSNorm stage expected exactly two inputs and one output.");
            }
            value_dims[stage.outputs[0].value_id] = inferRmsNormOutputDims(*stage.rms_norm, stage_input_dims[0], stage_input_dims[1]);
        } else if (stage.kind == CompiledExecutionStage::Kind::ReduceMinMaxBackward) {
            if (!stage.reduce_minmax_backward) {
                throw std::runtime_error("Missing compiled reduce-min/max-backward stage.");
            }
            if (stage.input_value_ids.size() != 2 || stage.outputs.size() != 1) {
                throw std::runtime_error("Reduce-min/max-backward stage expected exactly two inputs and one output.");
            }
            value_dims[stage.outputs[0].value_id] = stage_input_dims[0];
        } else if (stage.kind == CompiledExecutionStage::Kind::ScanMinMaxBackward) {
            if (!stage.scan_minmax_backward) {
                throw std::runtime_error("Missing compiled scan-min/max-backward stage.");
            }
            const size_t expected_inputs = stage.scan_minmax_backward->segmented_by_offsets ? 3 : 2;
            if (stage.input_value_ids.size() != expected_inputs || stage.outputs.size() != 1) {
                throw std::runtime_error("Scan-min/max-backward stage expected input/grad[/offsets] and one output.");
            }
            value_dims[stage.outputs[0].value_id] = stage_input_dims[0];
        } else if (stage.kind == CompiledExecutionStage::Kind::Matmul) {
            if (!stage.matmul) {
                throw std::runtime_error("Missing compiled matmul stage.");
            }
            if (stage.outputs.empty() || stage.outputs.size() > 2) {
                throw std::runtime_error("Matmul/gemm stage expected one matrix output and at most one bias-gradient output.");
            }
            const std::vector<uint64_t> matrix_dims = resolveMatmulOutputDimsFromInputs(*stage.matmul, stage_input_dims);
            value_dims[stage.outputs[0].value_id] = matrix_dims;
            if (stage.outputs.size() > 1) {
                if (!stage.matmul->bgrad_output_dtype.has_value()) {
                    throw std::runtime_error("Matmul/gemm stage has a secondary output but no compiled bias-gradient output dtype.");
                }
                if (matrix_dims.size() != 2) {
                    throw std::runtime_error("Matmul/gemm bias-gradient output requires a rank-2 matrix output.");
                }
                value_dims[stage.outputs[1].value_id] = std::vector<uint64_t>{matrix_dims[1]};
            }
        } else if (stage.kind == CompiledExecutionStage::Kind::InPlaceRope) {
            if (!stage.in_place_rope) {
                throw std::runtime_error("Missing compiled in-place RoPE stage.");
            }
            for (size_t i = 0; i < stage.outputs.size(); ++i) {
                value_dims[stage.outputs[i].value_id] = stage.in_place_rope->tensors[i].logical_dims;
            }
        } else if (stage.kind == CompiledExecutionStage::Kind::Attention) {
            if (!stage.attention) {
                throw std::runtime_error("Missing compiled attention stage.");
            }
            if (stage.outputs.size() != 1) {
                throw std::runtime_error("Attention stage expected exactly one output.");
            }
            value_dims[stage.outputs[0].value_id] = resolveAttentionOutputDimsFromInputs(*stage.attention, stage_input_dims);
        } else if (stage.kind == CompiledExecutionStage::Kind::AttentionBackward) {
            if (!stage.attention_backward) {
                throw std::runtime_error("Missing compiled attention-backward stage.");
            }
            for (size_t i = 0; i < stage.outputs.size(); ++i) {
                if (stage.outputs[i].local_node_idx >= stage.expr.nodes.size()) {
                    throw std::runtime_error("Attention-backward stage output node index out of range.");
                }
                value_dims[stage.outputs[i].value_id] = resolveAttentionBackwardOutputDimsFromInputs(
                    *stage.attention_backward, stage_input_dims, stage.expr.nodes[stage.outputs[i].local_node_idx].op);
            }
        } else if (stage.kind == CompiledExecutionStage::Kind::Convolution) {
            if (!stage.convolution) {
                throw std::runtime_error("Missing compiled convolution stage.");
            }
            if (stage.outputs.size() != 1) {
                throw std::runtime_error("Convolution stage expected exactly one output.");
            }
            value_dims[stage.outputs[0].value_id] = resolveConvolutionOutputDimsFromInputs(*stage.convolution, stage_input_dims);
        } else if (stage.kind == CompiledExecutionStage::Kind::ConvolutionBackward) {
            if (!stage.convolution_backward) {
                throw std::runtime_error("Missing compiled convolution-backward stage.");
            }
            if (stage.outputs.size() != 1) {
                throw std::runtime_error("Convolution-backward stage expected exactly one output.");
            }
            value_dims[stage.outputs[0].value_id] =
                resolveConvolutionBackwardOutputDimsFromInputs(*stage.convolution_backward, stage_input_dims);
        } else {
            throw std::runtime_error("Unknown execution stage kind in getParameterFanOverrides.");
        }
        applyAvailableValueAliases(compiled_outputs->value_aliases, value_dims);
    }

    applyAvailableValueAliases(compiled_outputs->value_aliases, value_dims);

    return result;
}

}  // namespace ThorImplementation
