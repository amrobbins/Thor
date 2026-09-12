#include "Utilities/Expression/LogicalExpression.h"

#include <algorithm>
#include <array>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "Utilities/Expression/CudaKernelExpression.h"

namespace ThorImplementation {
namespace {

bool isInputOp(ExprOp op) {
    return op == ExprOp::INPUT || op == ExprOp::RUNTIME_SCALAR || op == ExprOp::TENSOR_RUNTIME_SCALAR;
}

struct ScalarDependencyDescriptor {
    LogicalDependencyKind kind;
    uint32_t ExprNode::*member;
    const char* field_name;
};

constexpr std::array<ScalarDependencyDescriptor, 24> kScalarDependencyDescriptors{{
    {LogicalDependencyKind::Lhs, &ExprNode::lhs, "lhs"},
    {LogicalDependencyKind::Rhs, &ExprNode::rhs, "rhs"},
    {LogicalDependencyKind::Aux, &ExprNode::aux, "aux"},
    {LogicalDependencyKind::Alpha, &ExprNode::alpha_node, "alpha_node"},
    {LogicalDependencyKind::Beta, &ExprNode::beta_node, "beta_node"},
    {LogicalDependencyKind::MatmulEpilogueAux, &ExprNode::matmul_epilogue_aux, "matmul_epilogue_aux"},
    {LogicalDependencyKind::RopeEffectiveSequenceLength, &ExprNode::rope_effective_sequence_length_node, "rope_effective_sequence_length_node"},
    {LogicalDependencyKind::RopePositionIds, &ExprNode::rope_position_ids_node, "rope_position_ids_node"},
    {LogicalDependencyKind::AttentionSeqLenQ, &ExprNode::attention_seq_len_q_node, "attention_seq_len_q_node"},
    {LogicalDependencyKind::AttentionSeqLenKv, &ExprNode::attention_seq_len_kv_node, "attention_seq_len_kv_node"},
    {LogicalDependencyKind::AttentionRaggedOffsetQ, &ExprNode::attention_ragged_offset_q_node, "attention_ragged_offset_q_node"},
    {LogicalDependencyKind::AttentionRaggedOffsetKv, &ExprNode::attention_ragged_offset_kv_node, "attention_ragged_offset_kv_node"},
    {LogicalDependencyKind::AttentionPageTableK, &ExprNode::attention_page_table_k_node, "attention_page_table_k_node"},
    {LogicalDependencyKind::AttentionPageTableV, &ExprNode::attention_page_table_v_node, "attention_page_table_v_node"},
    {LogicalDependencyKind::AttentionDropoutSeed, &ExprNode::attention_dropout_seed_node, "attention_dropout_seed_node"},
    {LogicalDependencyKind::AttentionDropoutOffset, &ExprNode::attention_dropout_offset_node, "attention_dropout_offset_node"},
    {LogicalDependencyKind::AttentionDescaleQ, &ExprNode::attention_descale_q_node, "attention_descale_q_node"},
    {LogicalDependencyKind::AttentionDescaleK, &ExprNode::attention_descale_k_node, "attention_descale_k_node"},
    {LogicalDependencyKind::AttentionDescaleV, &ExprNode::attention_descale_v_node, "attention_descale_v_node"},
    {LogicalDependencyKind::AttentionDescaleS, &ExprNode::attention_descale_s_node, "attention_descale_s_node"},
    {LogicalDependencyKind::AttentionScaleS, &ExprNode::attention_scale_s_node, "attention_scale_s_node"},
    {LogicalDependencyKind::AttentionScaleO, &ExprNode::attention_scale_o_node, "attention_scale_o_node"},
    {LogicalDependencyKind::AttentionAmaxS, &ExprNode::attention_amax_s_node, "attention_amax_s_node"},
    {LogicalDependencyKind::AttentionAmaxO, &ExprNode::attention_amax_o_node, "attention_amax_o_node"},
}};

consteval bool scalarDependencyKindsAreUniqueAndComplete() {
    std::array<bool, static_cast<size_t>(LogicalDependencyKind::Count)> seen{};
    for (const ScalarDependencyDescriptor& descriptor : kScalarDependencyDescriptors) {
        const size_t index = static_cast<size_t>(descriptor.kind);
        if (index >= seen.size() || seen[index]) {
            return false;
        }
        seen[index] = true;
    }
    return std::all_of(seen.begin(), seen.end(), [](bool value) { return value; });
}

static_assert(kScalarDependencyDescriptors.size() == static_cast<size_t>(LogicalDependencyKind::Count));
static_assert(scalarDependencyKindsAreUniqueAndComplete());

const ScalarDependencyDescriptor* scalarDependencyDescriptor(LogicalDependencyKind kind) {
    const size_t index = static_cast<size_t>(kind);
    if (index >= kScalarDependencyDescriptors.size() || kScalarDependencyDescriptors[index].kind != kind) {
        throw std::runtime_error("Logical dependency descriptor table is incomplete or out of order.");
    }
    return &kScalarDependencyDescriptors[index];
}

void requireNoPhysicalReferences(const ExprNode& node) {
    for (const ScalarDependencyDescriptor& descriptor : kScalarDependencyDescriptors) {
        if (node.*(descriptor.member) != UINT32_MAX) {
            throw std::invalid_argument(std::string("LogicalExpressionNode semantic field '") + descriptor.field_name +
                                        "' must not contain a physical node index.");
        }
    }
    if (node.input_slot != UINT32_MAX) {
        throw std::invalid_argument("LogicalExpressionNode semantic input_slot must be represented by a logical input binding.");
    }
    if (node.cuda_kernel_spec_index != UINT32_MAX) {
        throw std::invalid_argument("LogicalExpressionNode semantic CUDA spec index must be represented by a logical CUDA application.");
    }
    if (node.ragged_runtime_offsets_input_slot != UINT32_MAX) {
        throw std::invalid_argument(
            "LogicalExpressionNode semantic ragged runtime-offset slot must be represented by a logical input binding.");
    }
    if (!node.cuda_kernel_input_nodes.empty()) {
        throw std::invalid_argument("LogicalExpressionNode semantic CUDA input indices must be represented as logical dependencies.");
    }
}

void validateBasicArity(ExprOp op, const std::vector<LogicalDependency>& dependencies) {
    auto has = [&](LogicalDependencyKind kind) {
        return std::any_of(dependencies.begin(), dependencies.end(), [&](const LogicalDependency& dependency) {
            return dependency.kind == kind;
        });
    };

    const bool lhs = has(LogicalDependencyKind::Lhs);
    const bool rhs = has(LogicalDependencyKind::Rhs);
    const bool aux = has(LogicalDependencyKind::Aux);

    if (Expression::isUnaryOp(op)) {
        if (!lhs || rhs || aux) {
            throw std::invalid_argument("Logical unary operation must have exactly its lhs main dependency.");
        }
    } else if (Expression::isBinaryOp(op)) {
        if (!lhs || !rhs || aux) {
            throw std::invalid_argument("Logical binary operation must have lhs/rhs and no aux main dependency.");
        }
    } else if (Expression::isTernaryOp(op)) {
        if (!lhs || !rhs || !aux) {
            throw std::invalid_argument("Logical ternary operation must have lhs/rhs/aux main dependencies.");
        }
    } else if (Expression::isLeafOp(op)) {
        if (lhs || rhs || aux) {
            throw std::invalid_argument("Logical leaf operation cannot have lhs/rhs/aux dependencies.");
        }
    } else if (op == ExprOp::CUDA_KERNEL_OUTPUT) {
        if (lhs || rhs || aux) {
            throw std::invalid_argument("Logical CUDA kernel output uses ordered CUDA input dependencies, not lhs/rhs/aux.");
        }
    } else {
        throw std::invalid_argument("Unsupported ExprOp in logical DAG core.");
    }
}

std::vector<LogicalDependency> normalizeDependencies(std::vector<LogicalDependency> dependencies) {
    for (const LogicalDependency& dependency : dependencies) {
        if (dependency.kind == LogicalDependencyKind::Count) {
            throw std::invalid_argument("Logical dependency kind cannot be the Count sentinel.");
        }
        if (!dependency.node) {
            throw std::invalid_argument("Logical dependency cannot be null.");
        }
        if (dependency.ordinal != 0) {
            throw std::invalid_argument("Logical scalar dependency ordinals must be zero.");
        }
    }

    std::sort(dependencies.begin(), dependencies.end(), [](const LogicalDependency& a, const LogicalDependency& b) {
        return std::tie(a.kind, a.ordinal) < std::tie(b.kind, b.ordinal);
    });
    for (size_t i = 1; i < dependencies.size(); ++i) {
        if (dependencies[i - 1].kind == dependencies[i].kind && dependencies[i - 1].ordinal == dependencies[i].ordinal) {
            throw std::invalid_argument("Logical node contains duplicate dependency roles.");
        }
    }

    return dependencies;
}

ExprOp inputOpForKind(NamedInput::Kind kind) {
    switch (kind) {
        case NamedInput::Kind::Tensor:
            return ExprOp::INPUT;
        case NamedInput::Kind::RuntimeScalarFp32:
            return ExprOp::RUNTIME_SCALAR;
        case NamedInput::Kind::TensorRuntimeScalar:
            return ExprOp::TENSOR_RUNTIME_SCALAR;
    }
    throw std::invalid_argument("Unknown logical input kind.");
}

ExprNode semanticsWithOp(ExprOp op, ExprNode semantics) {
    semantics.op = op;
    return semantics;
}

}  // namespace

LogicalCudaKernelApplicationPtr LogicalCudaKernelApplication::create(
    std::shared_ptr<const CudaKernelExpression> specification,
    std::vector<LogicalExpression> inputs) {
    if (!specification) {
        throw std::invalid_argument("Logical CUDA kernel application requires a kernel specification.");
    }
    if (inputs.size() != specification->inputs().size()) {
        throw std::invalid_argument("Logical CUDA kernel application input count must match the kernel ABI input count.");
    }
    if (std::any_of(inputs.begin(), inputs.end(), [](const LogicalExpression& input) { return !input; })) {
        throw std::invalid_argument("Logical CUDA kernel application inputs cannot be null.");
    }
    return std::shared_ptr<const LogicalCudaKernelApplication>(
        new LogicalCudaKernelApplication(std::move(specification), std::move(inputs)));
}

LogicalCudaKernelApplication::LogicalCudaKernelApplication(
    std::shared_ptr<const CudaKernelExpression> specification,
    std::vector<LogicalExpression> inputs)
    : specification_(std::move(specification)), inputs_(std::move(inputs)) {}

LogicalExpression LogicalExpressionNode::create(ExprNode semantics,
                                                std::vector<LogicalDependency> dependencies,
                                                std::optional<LogicalInputBinding> input_binding,
                                                std::optional<LogicalInputBinding> ragged_runtime_offsets_binding,
                                                LogicalCudaKernelApplicationPtr cuda_kernel_application) {
    requireNoPhysicalReferences(semantics);
    dependencies = normalizeDependencies(std::move(dependencies));
    validateBasicArity(semantics.op, dependencies);

    if (isInputOp(semantics.op) != input_binding.has_value()) {
        throw std::invalid_argument("Logical input nodes require exactly one logical input binding; non-input nodes must not have one.");
    }
    if (input_binding && input_binding->name.empty()) {
        throw std::invalid_argument("Logical input name cannot be empty.");
    }
    if (input_binding && inputOpForKind(input_binding->kind) != semantics.op) {
        throw std::invalid_argument("Logical input binding kind does not match ExprOp.");
    }
    if (ragged_runtime_offsets_binding && ragged_runtime_offsets_binding->name.empty()) {
        throw std::invalid_argument("Logical ragged runtime-offset input name cannot be empty.");
    }
    if ((semantics.op == ExprOp::CUDA_KERNEL_OUTPUT) != static_cast<bool>(cuda_kernel_application)) {
        throw std::invalid_argument("Logical CUDA kernel output nodes require exactly one logical CUDA application.");
    }

    return std::shared_ptr<const LogicalExpressionNode>(new LogicalExpressionNode(std::move(semantics),
                                                                                  std::move(dependencies),
                                                                                  std::move(input_binding),
                                                                                  std::move(ragged_runtime_offsets_binding),
                                                                                  std::move(cuda_kernel_application)));
}

LogicalExpressionNode::LogicalExpressionNode(ExprNode semantics,
                                             std::vector<LogicalDependency> dependencies,
                                             std::optional<LogicalInputBinding> input_binding,
                                             std::optional<LogicalInputBinding> ragged_runtime_offsets_binding,
                                             LogicalCudaKernelApplicationPtr cuda_kernel_application)
    : semantics_(std::move(semantics)),
      dependencies_(std::move(dependencies)),
      input_binding_(std::move(input_binding)),
      ragged_runtime_offsets_binding_(std::move(ragged_runtime_offsets_binding)),
      cuda_kernel_application_(std::move(cuda_kernel_application)) {}

const LogicalExpression& LogicalExpressionNode::dependency(LogicalDependencyKind kind, uint32_t ordinal) const {
    const auto it = std::lower_bound(dependencies_.begin(), dependencies_.end(), std::pair{kind, ordinal},
                                     [](const LogicalDependency& dependency, const auto& key) {
                                         if (dependency.kind != key.first) {
                                             return static_cast<uint8_t>(dependency.kind) < static_cast<uint8_t>(key.first);
                                         }
                                         return dependency.ordinal < key.second;
                                     });
    if (it == dependencies_.end() || it->kind != kind || it->ordinal != ordinal) {
        throw std::out_of_range(std::string("Logical dependency not present: ") + logicalDependencyKindName(kind));
    }
    return it->node;
}

bool LogicalExpressionNode::hasDependency(LogicalDependencyKind kind, uint32_t ordinal) const {
    return std::any_of(dependencies_.begin(), dependencies_.end(), [&](const LogicalDependency& dependency) {
        return dependency.kind == kind && dependency.ordinal == ordinal;
    });
}

LogicalExpression makeLogicalInput(std::string name,
                                   NamedInput::Kind kind,
                                   std::optional<DataType> input_tensor_dtype,
                                   std::optional<DataType> compute_dtype,
                                   std::optional<DataType> output_dtype) {
    ExprNode semantics{};
    semantics.op = inputOpForKind(kind);
    semantics.input_tensor_dtype = input_tensor_dtype;
    semantics.compute_dtype = compute_dtype;
    semantics.output_dtype = output_dtype;
    return LogicalExpressionNode::create(std::move(semantics), {}, LogicalInputBinding{std::move(name), kind});
}

LogicalExpression makeLogicalScalar(double value,
                                    std::optional<DataType> compute_dtype,
                                    std::optional<DataType> output_dtype) {
    ExprNode semantics{};
    semantics.op = ExprOp::SCALAR_FP;
    semantics.scalar_fp = value;
    semantics.compute_dtype = compute_dtype;
    semantics.output_dtype = output_dtype;
    return LogicalExpressionNode::create(std::move(semantics));
}

LogicalExpression makeLogicalUnary(ExprOp op, const LogicalExpression& input, ExprNode semantics) {
    return LogicalExpressionNode::create(semanticsWithOp(op, std::move(semantics)),
                                         {{LogicalDependencyKind::Lhs, 0, input}});
}

LogicalExpression makeLogicalBinary(ExprOp op,
                                    const LogicalExpression& lhs,
                                    const LogicalExpression& rhs,
                                    ExprNode semantics) {
    return LogicalExpressionNode::create(semanticsWithOp(op, std::move(semantics)),
                                         {{LogicalDependencyKind::Lhs, 0, lhs}, {LogicalDependencyKind::Rhs, 0, rhs}});
}

LogicalExpression makeLogicalTernary(ExprOp op,
                                     const LogicalExpression& lhs,
                                     const LogicalExpression& rhs,
                                     const LogicalExpression& aux,
                                     ExprNode semantics) {
    return LogicalExpressionNode::create(semanticsWithOp(op, std::move(semantics)),
                                         {{LogicalDependencyKind::Lhs, 0, lhs},
                                          {LogicalDependencyKind::Rhs, 0, rhs},
                                          {LogicalDependencyKind::Aux, 0, aux}});
}


namespace {

void assignPhysicalDependency(ExprNode& node, LogicalDependencyKind kind, uint32_t ordinal, uint32_t physical_index) {
    if (ordinal != 0) {
        throw std::runtime_error("Logical scalar dependency ordinal must be zero while lowering.");
    }
    const ScalarDependencyDescriptor* descriptor = scalarDependencyDescriptor(kind);
    node.*(descriptor->member) = physical_index;
}

struct LogicalLoweringState {
    std::shared_ptr<PhysicalExpression> physical = std::make_shared<PhysicalExpression>();
    std::unordered_map<const LogicalExpressionNode*, uint32_t> lowered_nodes;
    std::unordered_map<const LogicalCudaKernelApplication*, uint32_t> cuda_applications;
    std::unordered_set<const LogicalExpressionNode*> active_nodes;

    uint32_t lowerNode(const LogicalExpression& logical) {
        if (!logical) {
            throw std::invalid_argument("Cannot lower a null logical expression node.");
        }
        if (const auto it = lowered_nodes.find(logical.get()); it != lowered_nodes.end()) {
            return it->second;
        }
        if (!active_nodes.insert(logical.get()).second) {
            throw std::runtime_error("Logical expression graph contains a cycle.");
        }

        std::vector<std::tuple<LogicalDependencyKind, uint32_t, uint32_t>> lowered_dependencies;
        lowered_dependencies.reserve(logical->dependencies().size());
        for (const LogicalDependency& dependency : logical->dependencies()) {
            lowered_dependencies.emplace_back(dependency.kind, dependency.ordinal, lowerNode(dependency.node));
        }

        std::vector<uint32_t> lowered_cuda_inputs;
        if (logical->cudaKernelApplication()) {
            lowered_cuda_inputs.reserve(logical->cudaKernelApplication()->inputs().size());
            for (const LogicalExpression& input : logical->cudaKernelApplication()->inputs()) {
                lowered_cuda_inputs.push_back(lowerNode(input));
            }
        }

        ExprNode node = logical->semantics();
        for (const auto& [kind, ordinal, physical_index] : lowered_dependencies) {
            assignPhysicalDependency(node, kind, ordinal, physical_index);
        }

        if (logical->inputBinding()) {
            node.input_slot = physical->getOrCreateInputSlot(logical->inputBinding()->name, logical->inputBinding()->kind);
        }
        if (logical->raggedRuntimeOffsetsBinding()) {
            node.ragged_runtime_offsets_input_slot = physical->getOrCreateInputSlot(
                logical->raggedRuntimeOffsetsBinding()->name, logical->raggedRuntimeOffsetsBinding()->kind);
        }
        if (logical->cudaKernelApplication()) {
            const LogicalCudaKernelApplication* identity = logical->cudaKernelApplication().get();
            auto it = cuda_applications.find(identity);
            if (it == cuda_applications.end()) {
                const uint32_t spec_index = static_cast<uint32_t>(physical->cuda_kernel_expressions.size());
                physical->cuda_kernel_expressions.push_back(logical->cudaKernelApplication()->specification());
                it = cuda_applications.emplace(identity, spec_index).first;
            }
            node.cuda_kernel_spec_index = it->second;
            node.cuda_kernel_input_nodes = std::move(lowered_cuda_inputs);
        }

        const uint32_t physical_index = static_cast<uint32_t>(physical->nodes.size());
        physical->nodes.push_back(std::move(node));
        const bool inserted = lowered_nodes.emplace(logical.get(), physical_index).second;
        if (!inserted) {
            throw std::runtime_error("Logical lowerer attempted to lower one logical node more than once in one context.");
        }
        active_nodes.erase(logical.get());
        return physical_index;
    }
};

}  // namespace

PhysicalOutputs LogicalExpressionLowerer::lower(const std::vector<LogicalNamedOutput>& outputs) {
    if (outputs.empty()) {
        throw std::invalid_argument("LogicalExpressionLowerer requires at least one named output.");
    }

    LogicalLoweringState state;
    std::vector<NamedOutput> physical_outputs;
    physical_outputs.reserve(outputs.size());
    std::unordered_set<std::string> names;
    for (const LogicalNamedOutput& output : outputs) {
        if (output.name.empty()) {
            throw std::invalid_argument("Logical output name cannot be empty.");
        }
        if (!names.insert(output.name).second) {
            throw std::invalid_argument("Logical output names must be unique.");
        }
        const uint32_t physical_index = state.lowerNode(output.node);
        physical_outputs.push_back(NamedOutput{output.name, physical_index, output.materialization});
    }
    state.physical->output_node = outputs.size() == 1 ? physical_outputs.front().node_idx : UINT32_MAX;
    return PhysicalOutputs{.expr = std::move(state.physical), .outputs = std::move(physical_outputs)};
}


namespace {

struct PhysicalDependencyReference {
    LogicalDependencyKind kind;
    uint32_t ordinal;
    uint32_t node_index;
};

std::vector<PhysicalDependencyReference> enumeratePhysicalDependencies(const ExprNode& node) {
    std::vector<PhysicalDependencyReference> dependencies;
    dependencies.reserve(kScalarDependencyDescriptors.size());
    for (const ScalarDependencyDescriptor& descriptor : kScalarDependencyDescriptors) {
        const uint32_t node_index = node.*(descriptor.member);
        if (node_index != UINT32_MAX) {
            dependencies.push_back({descriptor.kind, 0, node_index});
        }
    }
    return dependencies;
}

ExprNode semanticSnapshotFromPhysical(const ExprNode& source) {
    ExprNode semantics = source;
    for (const ScalarDependencyDescriptor& descriptor : kScalarDependencyDescriptors) {
        semantics.*(descriptor.member) = UINT32_MAX;
    }
    semantics.input_slot = UINT32_MAX;
    semantics.cuda_kernel_spec_index = UINT32_MAX;
    semantics.cuda_kernel_input_nodes.clear();
    semantics.ragged_runtime_offsets_input_slot = UINT32_MAX;
    return semantics;
}

LogicalInputBinding importInputBinding(const PhysicalExpression& source, uint32_t slot, const char* role) {
    if (slot == UINT32_MAX || slot >= source.inputs.size()) {
        throw std::runtime_error(std::string("Physical expression ") + role + " references an invalid input slot.");
    }
    const NamedInput& input = source.inputs[slot];
    if (input.slot != slot) {
        throw std::runtime_error(std::string("Physical expression ") + role + " input table is not self-consistent.");
    }
    return LogicalInputBinding{input.name, input.kind};
}

}  // namespace

LogicalExpressionImporter::LogicalExpressionImporter(const PhysicalExpression& source)
    : source_(source),
      memo_(source.nodes.size()),
      visit_state_(source.nodes.size(), 0),
      cuda_applications_(source.cuda_kernel_expressions.size()),
      cuda_application_input_nodes_(source.cuda_kernel_expressions.size()) {}

LogicalExpression LogicalExpressionImporter::importNode(uint32_t node_index) {
    if (node_index >= source_.nodes.size()) {
        throw std::out_of_range("LogicalExpressionImporter node index is out of range.");
    }
    if (memo_[node_index]) {
        return memo_[node_index];
    }
    if (visit_state_[node_index] == 2) {
        throw std::runtime_error("Logical importer memoization invariant was violated.");
    }
    if (visit_state_[node_index] == 1) {
        throw std::runtime_error("Physical expression graph contains a cycle.");
    }
    visit_state_[node_index] = 1;

    const ExprNode& source_node = source_.nodes[node_index];
    std::vector<LogicalDependency> dependencies;
    for (const PhysicalDependencyReference& dependency : enumeratePhysicalDependencies(source_node)) {
        if (dependency.node_index >= source_.nodes.size()) {
            throw std::runtime_error(std::string("Physical expression dependency '") + logicalDependencyKindName(dependency.kind) +
                                     "' is out of range.");
        }
        dependencies.push_back({dependency.kind, dependency.ordinal, importNode(dependency.node_index)});
    }

    std::optional<LogicalInputBinding> input_binding;
    if (isInputOp(source_node.op)) {
        input_binding = importInputBinding(source_, source_node.input_slot, "input node");
    }

    std::optional<LogicalInputBinding> ragged_offsets_binding;
    if (source_node.ragged_runtime_offsets_input_slot != UINT32_MAX) {
        ragged_offsets_binding = importInputBinding(source_, source_node.ragged_runtime_offsets_input_slot, "ragged runtime-offset metadata");
    }

    LogicalCudaKernelApplicationPtr cuda_kernel_application;
    if (source_node.op == ExprOp::CUDA_KERNEL_OUTPUT) {
        const uint32_t spec_index = source_node.cuda_kernel_spec_index;
        if (spec_index >= source_.cuda_kernel_expressions.size() || !source_.cuda_kernel_expressions[spec_index]) {
            throw std::runtime_error("Physical CUDA kernel output references an invalid kernel application/specification index.");
        }

        if (cuda_applications_[spec_index]) {
            if (cuda_application_input_nodes_[spec_index] != source_node.cuda_kernel_input_nodes) {
                throw std::runtime_error(
                    "Physical CUDA outputs sharing one application index must have identical ABI input nodes.");
            }
            cuda_kernel_application = cuda_applications_[spec_index];
        } else {
            std::vector<LogicalExpression> application_inputs;
            application_inputs.reserve(source_node.cuda_kernel_input_nodes.size());
            for (uint32_t input_node : source_node.cuda_kernel_input_nodes) {
                if (input_node >= source_.nodes.size()) {
                    throw std::runtime_error("Physical CUDA kernel application references an out-of-range ABI input node.");
                }
                application_inputs.push_back(importNode(input_node));
            }
            cuda_kernel_application = LogicalCudaKernelApplication::create(
                source_.cuda_kernel_expressions[spec_index], std::move(application_inputs));
            cuda_applications_[spec_index] = cuda_kernel_application;
            cuda_application_input_nodes_[spec_index] = source_node.cuda_kernel_input_nodes;
        }
    } else if (source_node.cuda_kernel_spec_index != UINT32_MAX || !source_node.cuda_kernel_input_nodes.empty()) {
        throw std::runtime_error("Non-CUDA physical expression node unexpectedly references CUDA application state.");
    }

    LogicalExpression imported = LogicalExpressionNode::create(semanticSnapshotFromPhysical(source_node),
                                                                std::move(dependencies),
                                                                std::move(input_binding),
                                                                std::move(ragged_offsets_binding),
                                                                std::move(cuda_kernel_application));
    memo_[node_index] = imported;
    visit_state_[node_index] = 2;
    return imported;
}

std::vector<LogicalNamedOutput> LogicalExpressionImporter::importOutputs(const std::vector<NamedOutput>& outputs) {
    if (outputs.empty()) {
        throw std::invalid_argument("LogicalExpressionImporter requires at least one named output.");
    }
    std::unordered_set<std::string> names;
    std::vector<LogicalNamedOutput> logical_outputs;
    logical_outputs.reserve(outputs.size());
    for (const NamedOutput& output : outputs) {
        if (output.name.empty() || !names.insert(output.name).second) {
            throw std::invalid_argument("Physical output names must be non-empty and unique while importing.");
        }
        logical_outputs.push_back(LogicalNamedOutput{output.name, importNode(output.node_idx), output.materialization});
    }
    return logical_outputs;
}


namespace {

bool isAttentionOp(ExprOp op) {
    return op == ExprOp::ATTENTION || op == ExprOp::ATTENTION_BACKWARD_Q || op == ExprOp::ATTENTION_BACKWARD_K ||
           op == ExprOp::ATTENTION_BACKWARD_V || op == ExprOp::ATTENTION_BACKWARD_BIAS;
}

bool isAttentionBackwardOp(ExprOp op) {
    return op == ExprOp::ATTENTION_BACKWARD_Q || op == ExprOp::ATTENTION_BACKWARD_K ||
           op == ExprOp::ATTENTION_BACKWARD_V || op == ExprOp::ATTENTION_BACKWARD_BIAS;
}

bool dependencyAllowedForOp(ExprOp op, LogicalDependencyKind kind) {
    switch (kind) {
        case LogicalDependencyKind::Lhs:
            return Expression::isUnaryOp(op) || Expression::isBinaryOp(op) || Expression::isTernaryOp(op);
        case LogicalDependencyKind::Rhs:
            return Expression::isBinaryOp(op) || Expression::isTernaryOp(op);
        case LogicalDependencyKind::Aux:
            return Expression::isTernaryOp(op);
        case LogicalDependencyKind::Alpha:
            return op == ExprOp::GEMM || isAttentionOp(op);
        case LogicalDependencyKind::Beta:
            return op == ExprOp::GEMM || isAttentionBackwardOp(op);
        case LogicalDependencyKind::MatmulEpilogueAux:
            return op == ExprOp::MATMUL || op == ExprOp::GEMM;
        case LogicalDependencyKind::RopeEffectiveSequenceLength:
        case LogicalDependencyKind::RopePositionIds:
            return op == ExprOp::ROPE;
        case LogicalDependencyKind::AttentionSeqLenQ:
        case LogicalDependencyKind::AttentionSeqLenKv:
        case LogicalDependencyKind::AttentionRaggedOffsetQ:
        case LogicalDependencyKind::AttentionRaggedOffsetKv:
        case LogicalDependencyKind::AttentionPageTableK:
        case LogicalDependencyKind::AttentionPageTableV:
        case LogicalDependencyKind::AttentionDropoutSeed:
        case LogicalDependencyKind::AttentionDropoutOffset:
            return isAttentionOp(op);
        case LogicalDependencyKind::AttentionDescaleQ:
        case LogicalDependencyKind::AttentionDescaleK:
        case LogicalDependencyKind::AttentionDescaleV:
        case LogicalDependencyKind::AttentionDescaleS:
        case LogicalDependencyKind::AttentionScaleS:
        case LogicalDependencyKind::AttentionScaleO:
        case LogicalDependencyKind::AttentionAmaxS:
        case LogicalDependencyKind::AttentionAmaxO:
            return op == ExprOp::ATTENTION;
        case LogicalDependencyKind::Count:
            return false;
    }
    return false;
}

void requireDependencyFlag(const LogicalExpressionNode& node,
                           bool enabled,
                           LogicalDependencyKind first,
                           LogicalDependencyKind second,
                           const char* feature) {
    const bool first_present = node.hasDependency(first);
    const bool second_present = node.hasDependency(second);
    if (enabled != first_present || enabled != second_present) {
        throw std::runtime_error(std::string("Logical ") + feature +
                                 " dependency presence does not match its semantic enable flag.");
    }
}

void validateLogicalNodeContract(const LogicalExpressionNode& node) {
    const ExprOp op = node.op();
    validateBasicArity(op, node.dependencies());
    requireNoPhysicalReferences(node.semantics());

    for (const LogicalDependency& dependency : node.dependencies()) {
        if (!dependencyAllowedForOp(op, dependency.kind)) {
            throw std::runtime_error(std::string("Logical dependency '") + logicalDependencyKindName(dependency.kind) +
                                     "' is not valid for this ExprOp.");
        }
    }

    if (op == ExprOp::MATMUL || op == ExprOp::GEMM) {
        const bool needs_epilogue_aux = node.semantics().matmul_backward_epilogue != MatmulBackwardEpilogue::Default;
        if (needs_epilogue_aux != node.hasDependency(LogicalDependencyKind::MatmulEpilogueAux)) {
            throw std::runtime_error("Logical matmul backward epilogue auxiliary dependency does not match epilogue semantics.");
        }
    }

    if (isAttentionOp(op)) {
        const ExprNode& semantics = node.semantics();
        requireDependencyFlag(node,
                              semantics.attention_use_padding_mask,
                              LogicalDependencyKind::AttentionSeqLenQ,
                              LogicalDependencyKind::AttentionSeqLenKv,
                              "attention padding-mask");
        requireDependencyFlag(node,
                              semantics.attention_use_ragged_offsets,
                              LogicalDependencyKind::AttentionRaggedOffsetQ,
                              LogicalDependencyKind::AttentionRaggedOffsetKv,
                              "attention ragged-offset");
        requireDependencyFlag(node,
                              semantics.attention_use_paged_kv_cache,
                              LogicalDependencyKind::AttentionPageTableK,
                              LogicalDependencyKind::AttentionPageTableV,
                              "attention paged-KV");
        requireDependencyFlag(node,
                              semantics.attention_dropout_probability > 0.0f,
                              LogicalDependencyKind::AttentionDropoutSeed,
                              LogicalDependencyKind::AttentionDropoutOffset,
                              "attention dropout");

        if (op == ExprOp::ATTENTION) {
            if (semantics.attention_use_bias != node.hasDependency(LogicalDependencyKind::Alpha)) {
                throw std::runtime_error("Logical attention bias dependency does not match attention_use_bias.");
            }
            const std::array<LogicalDependencyKind, 8> fp8_roles = {
                LogicalDependencyKind::AttentionDescaleQ,
                LogicalDependencyKind::AttentionDescaleK,
                LogicalDependencyKind::AttentionDescaleV,
                LogicalDependencyKind::AttentionDescaleS,
                LogicalDependencyKind::AttentionScaleS,
                LogicalDependencyKind::AttentionScaleO,
                LogicalDependencyKind::AttentionAmaxS,
                LogicalDependencyKind::AttentionAmaxO,
            };
            for (LogicalDependencyKind role : fp8_roles) {
                if (semantics.attention_use_fp8_forward_scaling != node.hasDependency(role)) {
                    throw std::runtime_error("Logical attention FP8 dependency presence does not match attention_use_fp8_forward_scaling.");
                }
            }
        } else {
            if (!node.hasDependency(LogicalDependencyKind::Alpha)) {
                throw std::runtime_error("Logical attention backward node is missing the upstream-gradient dependency.");
            }
            if (semantics.attention_use_bias != node.hasDependency(LogicalDependencyKind::Beta)) {
                throw std::runtime_error("Logical attention backward bias dependency does not match attention_use_bias.");
            }
        }
    }

    if (op == ExprOp::CUDA_KERNEL_OUTPUT) {
        const auto& application = node.cudaKernelApplication();
        if (!application || !application->specification()) {
            throw std::runtime_error("Logical CUDA output is missing its kernel application/specification.");
        }
        const auto& kernel = application->specification();
        if (node.semantics().cuda_kernel_output_index >= kernel->outputs().size()) {
            throw std::runtime_error("Logical CUDA output index is outside the kernel specification.");
        }
        if (!node.semantics().output_dtype.has_value() ||
            node.semantics().output_dtype.value() != kernel->outputs()[node.semantics().cuda_kernel_output_index].dtype) {
            throw std::runtime_error("Logical CUDA output dtype does not match the kernel output specification.");
        }
        if (application->inputs().size() != kernel->inputs().size()) {
            throw std::runtime_error("Logical CUDA application ABI input count does not match its kernel specification.");
        }
        for (uint32_t ordinal = 0; ordinal < kernel->inputs().size(); ++ordinal) {
            const LogicalExpression& input = application->inputs()[ordinal];
            const auto& spec = kernel->inputs()[ordinal];
            switch (spec.kind) {
                case CudaKernelExpression::TensorParamSpec::Kind::Tensor:
                    if (input->op() == ExprOp::RUNTIME_SCALAR || input->op() == ExprOp::TENSOR_RUNTIME_SCALAR) {
                        throw std::runtime_error("Logical CUDA tensor ABI input is wired to a runtime scalar node.");
                    }
                    break;
                case CudaKernelExpression::TensorParamSpec::Kind::TensorRuntimeScalar:
                    if (input->op() != ExprOp::TENSOR_RUNTIME_SCALAR) {
                        throw std::runtime_error("Logical CUDA tensor-runtime-scalar ABI input has the wrong logical node kind.");
                    }
                    break;
                case CudaKernelExpression::TensorParamSpec::Kind::HostRuntimeScalar:
                    if (input->op() != ExprOp::RUNTIME_SCALAR) {
                        throw std::runtime_error("Logical CUDA host-runtime-scalar ABI input has the wrong logical node kind.");
                    }
                    break;
            }
            if (input->semantics().output_dtype.has_value() && input->semantics().output_dtype.value() != spec.dtype) {
                throw std::runtime_error("Logical CUDA ABI input dtype does not match the kernel specification.");
            }
        }
    }
}

}  // namespace

void validateLogicalGraph(const std::vector<LogicalNamedOutput>& outputs) {
    if (outputs.empty()) {
        throw std::invalid_argument("Logical graph validation requires at least one output.");
    }
    std::unordered_set<std::string> output_names;
    std::unordered_map<const LogicalExpressionNode*, uint8_t> state;
    std::function<void(const LogicalExpression&)> visit = [&](const LogicalExpression& logical) {
        if (!logical) {
            throw std::runtime_error("Logical graph contains a null node.");
        }
        uint8_t& node_state = state[logical.get()];
        if (node_state == 2) {
            return;
        }
        if (node_state == 1) {
            throw std::runtime_error("Logical graph contains a cycle.");
        }
        node_state = 1;
        validateLogicalNodeContract(*logical);
        for (const LogicalDependency& dependency : logical->dependencies()) {
            visit(dependency.node);
        }
        if (logical->cudaKernelApplication()) {
            for (const LogicalExpression& input : logical->cudaKernelApplication()->inputs()) {
                visit(input);
            }
        }
        node_state = 2;
    };

    for (const LogicalNamedOutput& output : outputs) {
        if (output.name.empty() || !output_names.insert(output.name).second) {
            throw std::runtime_error("Logical graph outputs must have non-empty unique names.");
        }
        visit(output.node);
    }
}

const char* logicalDependencyKindName(LogicalDependencyKind kind) {
    switch (kind) {
        case LogicalDependencyKind::Lhs: return "lhs";
        case LogicalDependencyKind::Rhs: return "rhs";
        case LogicalDependencyKind::Aux: return "aux";
        case LogicalDependencyKind::Alpha: return "alpha";
        case LogicalDependencyKind::Beta: return "beta";
        case LogicalDependencyKind::MatmulEpilogueAux: return "matmul_epilogue_aux";
        case LogicalDependencyKind::RopeEffectiveSequenceLength: return "rope_effective_sequence_length";
        case LogicalDependencyKind::RopePositionIds: return "rope_position_ids";
        case LogicalDependencyKind::AttentionSeqLenQ: return "attention_seq_len_q";
        case LogicalDependencyKind::AttentionSeqLenKv: return "attention_seq_len_kv";
        case LogicalDependencyKind::AttentionRaggedOffsetQ: return "attention_ragged_offset_q";
        case LogicalDependencyKind::AttentionRaggedOffsetKv: return "attention_ragged_offset_kv";
        case LogicalDependencyKind::AttentionPageTableK: return "attention_page_table_k";
        case LogicalDependencyKind::AttentionPageTableV: return "attention_page_table_v";
        case LogicalDependencyKind::AttentionDropoutSeed: return "attention_dropout_seed";
        case LogicalDependencyKind::AttentionDropoutOffset: return "attention_dropout_offset";
        case LogicalDependencyKind::AttentionDescaleQ: return "attention_descale_q";
        case LogicalDependencyKind::AttentionDescaleK: return "attention_descale_k";
        case LogicalDependencyKind::AttentionDescaleV: return "attention_descale_v";
        case LogicalDependencyKind::AttentionDescaleS: return "attention_descale_s";
        case LogicalDependencyKind::AttentionScaleS: return "attention_scale_s";
        case LogicalDependencyKind::AttentionScaleO: return "attention_scale_o";
        case LogicalDependencyKind::AttentionAmaxS: return "attention_amax_s";
        case LogicalDependencyKind::AttentionAmaxO: return "attention_amax_o";
        case LogicalDependencyKind::Count: return "count";
    }
    return "unknown";
}

}  // namespace ThorImplementation
