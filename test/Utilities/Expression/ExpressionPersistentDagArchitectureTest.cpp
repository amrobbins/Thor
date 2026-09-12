#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Expression/CudaKernelExpression.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/ExpressionInternal.h"
#include "Utilities/Expression/ExpressionTestHooks.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/LogicalExpression.h"
#include "Utilities/Expression/RaggedExpression.h"

#include "DeepLearning/Api/Layers/Learning/Convolution2d.h"
#include "DeepLearning/Api/Layers/Learning/Convolution3d.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "Utilities/Expression/ExecutionDiagnostics.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace ThorImplementation;
namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

template <typename T>
concept HasPublicPlannerTestHook = requires(const PhysicalOutputs& outputs) {
    T::planForTests(outputs);
};

static_assert(!HasPublicPlannerTestHook<EquationCompiler>,
              "Planner observability must remain behind ExpressionTestHooks.h, not EquationCompiler public API.");

size_t countLinearBoundaryNodes(const PhysicalOutputs& outputs) {
    if (!outputs.expr) {
        return 0;
    }

    size_t count = 0;
    for (const ExprNode& node : outputs.expr->nodes) {
        count += (node.op == ExprOp::MATMUL || node.op == ExprOp::GEMM) ? 1u : 0u;
    }
    return count;
}

size_t countStagesOfKind(const std::vector<PhysicalExecutionStage>& stages, PhysicalExecutionStage::Kind kind) {
    size_t count = 0;
    for (const PhysicalExecutionStage& stage : stages) {
        count += stage.kind == kind ? 1u : 0u;
    }
    return count;
}

size_t countNodesOfKind(const PhysicalOutputs& outputs, ExprOp op);

struct PlannerIdentityObservation {
    PhysicalOutputs physical;
    std::vector<PhysicalExecutionStage> stages;
    std::vector<CompiledStageOutput> final_outputs;
};

PlannerIdentityObservation observePlannerIdentity(const Expression& a,
                                                   const Expression& b,
                                                   const std::vector<DataType>& input_dtypes) {
    PhysicalOutputs physical = Expression::outputs({{"a", a}, {"b", b}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, input_dtypes);
    detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(physical);
    return PlannerIdentityObservation{
        .physical = std::move(physical),
        .stages = std::move(plan.stages),
        .final_outputs = std::move(plan.final_outputs),
    };
}

std::optional<uint32_t> finalValueIdNamed(const std::vector<CompiledStageOutput>& outputs, const std::string& name) {
    for (const CompiledStageOutput& output : outputs) {
        if (output.name == name) {
            return output.value_id;
        }
    }
    return std::nullopt;
}

size_t countStageOutputsWithValueId(const std::vector<PhysicalExecutionStage>& stages, uint32_t value_id) {
    size_t count = 0;
    for (const PhysicalExecutionStage& stage : stages) {
        for (const CompiledStageOutput& output : stage.outputs) {
            count += output.value_id == value_id ? 1u : 0u;
        }
    }
    return count;
}

void expectDistinctStageBoundaryRootsUseDistinctRuntimeValues(const PlannerIdentityObservation& observation,
                                                              ExprOp op,
                                                              PhysicalExecutionStage::Kind stage_kind) {
    ASSERT_TRUE(observation.physical.expr);
    ASSERT_EQ(observation.physical.outputs.size(), 2u);
    const uint32_t a_root = observation.physical.outputs[0].node_idx;
    const uint32_t b_root = observation.physical.outputs[1].node_idx;
    ASSERT_NE(a_root, b_root);
    ASSERT_LT(a_root, observation.physical.expr->nodes.size());
    ASSERT_LT(b_root, observation.physical.expr->nodes.size());
    EXPECT_EQ(observation.physical.expr->nodes[a_root].op, op);
    EXPECT_EQ(observation.physical.expr->nodes[b_root].op, op);
    EXPECT_EQ(countNodesOfKind(observation.physical, op), 2u);

    const std::optional<uint32_t> a_value = finalValueIdNamed(observation.final_outputs, "a");
    const std::optional<uint32_t> b_value = finalValueIdNamed(observation.final_outputs, "b");
    ASSERT_TRUE(a_value.has_value());
    ASSERT_TRUE(b_value.has_value());
    EXPECT_NE(*a_value, *b_value)
        << "Distinct stage-boundary physical roots must retain distinct runtime value identities.";
    EXPECT_GE(countStagesOfKind(observation.stages, stage_kind), 1u);
    EXPECT_EQ(countStageOutputsWithValueId(observation.stages, *a_value), 1u);
    EXPECT_EQ(countStageOutputsWithValueId(observation.stages, *b_value), 1u);
}

void expectSharedStageBoundaryRootUsesOneRuntimeValue(const PlannerIdentityObservation& observation,
                                                      ExprOp op,
                                                      PhysicalExecutionStage::Kind stage_kind) {
    ASSERT_TRUE(observation.physical.expr);
    ASSERT_EQ(observation.physical.outputs.size(), 2u);
    const uint32_t a_root = observation.physical.outputs[0].node_idx;
    const uint32_t b_root = observation.physical.outputs[1].node_idx;
    ASSERT_EQ(a_root, b_root);
    ASSERT_LT(a_root, observation.physical.expr->nodes.size());
    EXPECT_EQ(observation.physical.expr->nodes[a_root].op, op);
    EXPECT_EQ(countNodesOfKind(observation.physical, op), 1u);

    const std::optional<uint32_t> a_value = finalValueIdNamed(observation.final_outputs, "a");
    const std::optional<uint32_t> b_value = finalValueIdNamed(observation.final_outputs, "b");
    ASSERT_TRUE(a_value.has_value());
    ASSERT_TRUE(b_value.has_value());
    EXPECT_EQ(*a_value, *b_value);
    EXPECT_EQ(countStagesOfKind(observation.stages, stage_kind), 1u);
    EXPECT_EQ(countStageOutputsWithValueId(observation.stages, *a_value), 1u);
}

size_t countNodesOfKind(const PhysicalOutputs& outputs, ExprOp op) {
    if (!outputs.expr) {
        return 0;
    }

    size_t count = 0;
    for (const ExprNode& node : outputs.expr->nodes) {
        count += node.op == op ? 1u : 0u;
    }
    return count;
}

size_t countInputNodesNamed(const PhysicalOutputs& outputs, const std::string& input_name) {
    if (!outputs.expr) {
        return 0;
    }

    size_t count = 0;
    for (const ExprNode& node : outputs.expr->nodes) {
        if (node.op != ExprOp::INPUT && node.op != ExprOp::RUNTIME_SCALAR && node.op != ExprOp::TENSOR_RUNTIME_SCALAR) {
            continue;
        }
        if (node.input_slot >= outputs.expr->inputs.size()) {
            continue;
        }
        count += outputs.expr->inputs[node.input_slot].name == input_name ? 1u : 0u;
    }
    return count;
}

void expectEquivalentPhysicalTopology(const PhysicalOutputs& lhs, const PhysicalOutputs& rhs) {
    ASSERT_TRUE(lhs.expr);
    ASSERT_TRUE(rhs.expr);
    ASSERT_FALSE(lhs.conditional);
    ASSERT_FALSE(rhs.conditional);

    ASSERT_EQ(lhs.expr->inputs.size(), rhs.expr->inputs.size());
    for (size_t i = 0; i < lhs.expr->inputs.size(); ++i) {
        EXPECT_EQ(lhs.expr->inputs[i].name, rhs.expr->inputs[i].name);
        EXPECT_EQ(lhs.expr->inputs[i].slot, rhs.expr->inputs[i].slot);
        EXPECT_EQ(lhs.expr->inputs[i].kind, rhs.expr->inputs[i].kind);
    }

    ASSERT_EQ(lhs.outputs.size(), rhs.outputs.size());
    for (size_t i = 0; i < lhs.outputs.size(); ++i) {
        EXPECT_EQ(lhs.outputs[i].name, rhs.outputs[i].name);
        EXPECT_EQ(lhs.outputs[i].node_idx, rhs.outputs[i].node_idx);
        EXPECT_EQ(lhs.outputs[i].materialization, rhs.outputs[i].materialization);
    }

    ASSERT_EQ(lhs.expr->nodes.size(), rhs.expr->nodes.size());
    for (size_t i = 0; i < lhs.expr->nodes.size(); ++i) {
        const ExprNode& a = lhs.expr->nodes[i];
        const ExprNode& b = rhs.expr->nodes[i];
        EXPECT_EQ(a.op, b.op) << "node " << i;
        EXPECT_EQ(a.lhs, b.lhs) << "node " << i;
        EXPECT_EQ(a.rhs, b.rhs) << "node " << i;
        EXPECT_EQ(a.aux, b.aux) << "node " << i;
        EXPECT_EQ(a.alpha_node, b.alpha_node) << "node " << i;
        EXPECT_EQ(a.beta_node, b.beta_node) << "node " << i;
        EXPECT_EQ(a.matmul_epilogue_aux, b.matmul_epilogue_aux) << "node " << i;
        EXPECT_EQ(a.rope_effective_sequence_length_node, b.rope_effective_sequence_length_node) << "node " << i;
        EXPECT_EQ(a.rope_position_ids_node, b.rope_position_ids_node) << "node " << i;
        EXPECT_EQ(a.attention_seq_len_q_node, b.attention_seq_len_q_node) << "node " << i;
        EXPECT_EQ(a.attention_seq_len_kv_node, b.attention_seq_len_kv_node) << "node " << i;
        EXPECT_EQ(a.attention_ragged_offset_q_node, b.attention_ragged_offset_q_node) << "node " << i;
        EXPECT_EQ(a.attention_ragged_offset_kv_node, b.attention_ragged_offset_kv_node) << "node " << i;
        EXPECT_EQ(a.attention_page_table_k_node, b.attention_page_table_k_node) << "node " << i;
        EXPECT_EQ(a.attention_page_table_v_node, b.attention_page_table_v_node) << "node " << i;
        EXPECT_EQ(a.attention_dropout_seed_node, b.attention_dropout_seed_node) << "node " << i;
        EXPECT_EQ(a.attention_dropout_offset_node, b.attention_dropout_offset_node) << "node " << i;
        EXPECT_EQ(a.attention_descale_q_node, b.attention_descale_q_node) << "node " << i;
        EXPECT_EQ(a.attention_descale_k_node, b.attention_descale_k_node) << "node " << i;
        EXPECT_EQ(a.attention_descale_v_node, b.attention_descale_v_node) << "node " << i;
        EXPECT_EQ(a.attention_descale_s_node, b.attention_descale_s_node) << "node " << i;
        EXPECT_EQ(a.attention_scale_s_node, b.attention_scale_s_node) << "node " << i;
        EXPECT_EQ(a.attention_scale_o_node, b.attention_scale_o_node) << "node " << i;
        EXPECT_EQ(a.attention_amax_s_node, b.attention_amax_s_node) << "node " << i;
        EXPECT_EQ(a.attention_amax_o_node, b.attention_amax_o_node) << "node " << i;
        EXPECT_EQ(a.cuda_kernel_input_nodes, b.cuda_kernel_input_nodes) << "node " << i;
    }

    EXPECT_EQ(canonicalize(lhs), canonicalize(rhs));
}

CudaKernelExpression repeatedInputInspectionKernel() {
    return CudaKernelExpression::builder("shared_input_identity")
        .source(R"cuda(
extern "C" __global__
void shared_input_identity_kernel(const float* a, const float* b, float* out) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = a[i] + b[i];
}
)cuda")
        .entry("shared_input_identity_kernel")
        .input("a", DataType::FP32)
        .input("b", DataType::FP32)
        .outputLike("out", DataType::FP32, "a")
        .launchGrid1D(CudaKernelExpression::DimExpr::numel("out"), 64)
        .build();
}


CudaKernelExpression multiOutputInspectionKernel() {
    return CudaKernelExpression::builder("multi_output_application")
        .source(R"cuda(
extern "C" __global__
void multi_output_application_kernel(const float* a, const float* b, float* out0, float* out1) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    out0[i] = a[i] + b[i];
    out1[i] = a[i] - b[i];
}
)cuda")
        .entry("multi_output_application_kernel")
        .input("a", DataType::FP32)
        .input("b", DataType::FP32)
        .outputLike("out0", DataType::FP32, "a")
        .outputLike("out1", DataType::FP32, "a")
        .launchGrid1D(CudaKernelExpression::DimExpr::numel("out0"), 64)
        .build();
}

CudaKernelExpression runtimeScalarInspectionKernel() {
    return CudaKernelExpression::builder("runtime_scalar_abi")
        .source(R"cuda(
extern "C" __global__
void runtime_scalar_abi_kernel(const float* x, const long long* seed, float scale, float* out) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = x[i] * scale + static_cast<float>(*seed == 0);
}
)cuda")
        .entry("runtime_scalar_abi_kernel")
        .input("x", DataType::FP32)
        .tensorRuntimeScalarInput("seed", DataType::INT64)
        .hostRuntimeScalarInput("scale", DataType::FP32)
        .outputLike("out", DataType::FP32, "x")
        .launchGrid1D(CudaKernelExpression::DimExpr::numel("out"), 64)
        .build();
}

RaggedTensorDescriptor persistentDagRaggedDescriptor() {
    return RaggedTensorDescriptor(DataType::FP32, {4}, 3, 9, 9, DataType::UINT32);
}

Expression fp32Projection() {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w = Expression::input("w", DataType::FP32, DataType::FP32);
    return Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32);
}

BackwardBuildResult buildProjectionBackward(const Expression& y) {
    PhysicalOutputs forward = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP32, DataType::FP32});
    const std::unordered_map<std::string, std::vector<uint64_t>> forward_input_dims = {
        {"x", {2, 3}},
        {"w", {3, 4}},
    };

    return buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x", "w"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>(forward_input_dims));
}

void expectProjectionVjpPairUsesOneAccumulatedAdjoint(const PhysicalOutputs& backward) {
    ASSERT_TRUE(backward.expr);

    std::vector<uint32_t> linear_nodes;
    for (uint32_t node_idx = 0; node_idx < backward.expr->nodes.size(); ++node_idx) {
        const ExprOp op = backward.expr->nodes[node_idx].op;
        if (op == ExprOp::MATMUL || op == ExprOp::GEMM) {
            linear_nodes.push_back(node_idx);
        }
    }
    ASSERT_EQ(linear_nodes.size(), 2u)
        << "One shared forward projection must generate exactly the dInput/dWeight VJP pair.";

    const ExprNode& first = backward.expr->nodes.at(linear_nodes[0]);
    const ExprNode& second = backward.expr->nodes.at(linear_nodes[1]);
    std::vector<uint32_t> common_operands;
    const uint32_t first_operands[] = {first.lhs, first.rhs};
    const uint32_t second_operands[] = {second.lhs, second.rhs};
    for (const uint32_t lhs_operand : first_operands) {
        for (const uint32_t rhs_operand : second_operands) {
            if (lhs_operand == rhs_operand &&
                std::find(common_operands.begin(), common_operands.end(), lhs_operand) == common_operands.end()) {
                common_operands.push_back(lhs_operand);
            }
        }
    }

    ASSERT_EQ(common_operands.size(), 1u)
        << "The dInput and dWeight MATMULs must consume one shared accumulated projection adjoint.";
    ASSERT_LT(common_operands.front(), backward.expr->nodes.size());
    EXPECT_EQ(backward.expr->nodes.at(common_operands.front()).op, ExprOp::ADD)
        << "Branch/output seeds must accumulate before the shared producer VJP is generated.";
}

size_t countRetainedLinearForwardValues(const PhysicalOutputs& forward, const BackwardBuildResult& backward) {
    if (!forward.expr) {
        return 0;
    }

    size_t count = 0;
    for (const ForwardValueRequirement& requirement : backward.forward_value_requirements) {
        if (requirement.kind != ForwardValueRequirementKind::NodeOutput ||
            requirement.forward_node_index >= forward.expr->nodes.size()) {
            continue;
        }
        const ExprOp op = forward.expr->nodes.at(requirement.forward_node_index).op;
        count += (op == ExprOp::MATMUL || op == ExprOp::GEMM) ? 1u : 0u;
    }
    return count;
}

size_t countForwardValueRequirements(const BackwardBuildResult& backward,
                                     uint32_t forward_node_index,
                                     ForwardValueRequirementKind kind) {
    return static_cast<size_t>(std::count_if(
        backward.forward_value_requirements.begin(),
        backward.forward_value_requirements.end(),
        [&](const ForwardValueRequirement& requirement) {
            return requirement.forward_node_index == forward_node_index && requirement.kind == kind;
        }));
}

void expectUniqueForwardValueRequirementIdentities(const BackwardBuildResult& backward) {
    const auto& requirements = backward.forward_value_requirements;
    for (size_t i = 0; i < requirements.size(); ++i) {
        for (size_t j = i + 1; j < requirements.size(); ++j) {
            const bool same_semantic_requirement =
                requirements[i].forward_node_index == requirements[j].forward_node_index &&
                requirements[i].kind == requirements[j].kind &&
                requirements[i].conditional_branch_path == requirements[j].conditional_branch_path;
            EXPECT_FALSE(same_semantic_requirement)
                << "Retained-forward requirements must be unique by forward node, requirement kind, and conditional branch path.";
        }
    }
}

uint32_t findSingleNodeOfKind(const PhysicalOutputs& outputs, ExprOp op) {
    if (!outputs.expr) {
        ADD_FAILURE() << "Expected a physical expression while locating a node.";
        return UINT32_MAX;
    }

    uint32_t result = UINT32_MAX;
    for (uint32_t node_idx = 0; node_idx < outputs.expr->nodes.size(); ++node_idx) {
        if (outputs.expr->nodes[node_idx].op != op) {
            continue;
        }
        if (result != UINT32_MAX) {
            ADD_FAILURE() << "Expected exactly one node of the requested kind.";
            return UINT32_MAX;
        }
        result = node_idx;
    }
    if (result == UINT32_MAX) {
        ADD_FAILURE() << "Expected exactly one node of the requested kind.";
    }
    return result;
}

#ifdef THOR_DEBUG
size_t countNodesWithProvenance(const PhysicalOutputs& outputs,
                                ExprOp op,
                                ExpressionExecutionProvenance provenance) {
    if (!outputs.expr) {
        return 0;
    }

    size_t count = 0;
    for (const ExprNode& node : outputs.expr->nodes) {
        if (node.op == op && node.execution_provenance == provenance) {
            ++count;
        }
    }
    return count;
}

Impl::TensorPlacement persistentDagCpuPlacement(Impl::TensorPlacement::MemDevices::CPU);

uint64_t persistentDagTensorNumel(const Impl::Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t dim : tensor.getDimensions()) {
        numel *= dim;
    }
    return numel;
}

void persistentDagSynchronizeEvents(std::vector<Event>& events) {
    for (Event& event : events) {
        event.synchronize();
    }
    events.clear();
}

void persistentDagFillFp32Tensor(Impl::Tensor& tensor, float value) {
    ASSERT_EQ(tensor.getPlacement(), persistentDagCpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::FP32);
    auto* values = static_cast<float*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < persistentDagTensorNumel(tensor); ++i) {
        values[i] = value;
    }
}

void persistentDagWriteFp32Tensor(Impl::Tensor& tensor, const std::vector<float>& source) {
    ASSERT_EQ(tensor.getPlacement(), persistentDagCpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::FP32);
    ASSERT_EQ(persistentDagTensorNumel(tensor), source.size());
    auto* values = static_cast<float*>(tensor.getMemPtr());
    std::copy(source.begin(), source.end(), values);
}

std::vector<float> persistentDagReadFp32Tensor(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), persistentDagCpuPlacement);
    EXPECT_EQ(tensor.getDataType(), DataType::FP32);
    const auto* values = static_cast<const float*>(tensor.getMemPtr());
    return std::vector<float>(values, values + persistentDagTensorNumel(tensor));
}

Impl::Tensor persistentDagCopyTensorToCpu(const Impl::Tensor& tensor, Stream& stream) {
    Impl::Tensor host = tensor.clone(persistentDagCpuPlacement);
    host.copyFromAsync(tensor, stream);
    Event copied = stream.putEvent();
    copied.synchronize();
    return host;
}

void persistentDagSetParameterTensor(const std::shared_ptr<Impl::PhysicalParameter>& parameter,
                             const std::vector<float>& values,
                             Stream& stream) {
    ASSERT_NE(parameter, nullptr);
    ASSERT_TRUE(parameter->getStorage().has_value());
    Impl::Tensor device = parameter->getStorage().value();
    Impl::Tensor host = device.clone(persistentDagCpuPlacement);
    persistentDagWriteFp32Tensor(host, values);
    device.copyFromAsync(host, stream);
}

void persistentDagExpectAllClose(const std::vector<float>& actual,
                         const std::vector<float>& expected,
                         float atol,
                         float rtol,
                         const std::string& what) {
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        const float tolerance = atol + rtol * std::fabs(expected[i]);
        EXPECT_LE(diff, tolerance)
            << what << " mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

std::vector<float> persistentDagFullyConnectedBackwardErrorReference(const std::vector<float>& error_input,
                                                               const std::vector<float>& weights,
                                                               uint64_t batch_size,
                                                               uint64_t input_features,
                                                               uint64_t output_features) {
    std::vector<float> error_output(batch_size * input_features, 0.0f);
    for (uint64_t batch = 0; batch < batch_size; ++batch) {
        for (uint64_t input = 0; input < input_features; ++input) {
            float sum = 0.0f;
            for (uint64_t output = 0; output < output_features; ++output) {
                sum += error_input[batch * output_features + output] *
                       weights[input * output_features + output];
            }
            error_output[batch * input_features + input] = sum;
        }
    }
    return error_output;
}

std::vector<float> persistentDagFullyConnectedWeightGradReference(const std::vector<float>& input_values,
                                                            const std::vector<float>& error_input,
                                                            uint64_t batch_size,
                                                            uint64_t input_features,
                                                            uint64_t output_features) {
    std::vector<float> gradient(input_features * output_features, 0.0f);
    for (uint64_t input = 0; input < input_features; ++input) {
        for (uint64_t output = 0; output < output_features; ++output) {
            float sum = 0.0f;
            for (uint64_t batch = 0; batch < batch_size; ++batch) {
                sum += input_values[batch * input_features + input] *
                       error_input[batch * output_features + output];
            }
            gradient[input * output_features + output] = sum;
        }
    }
    return gradient;
}

std::vector<float> persistentDagSgdUpdatedReference(const std::vector<float>& initial,
                                             const std::vector<float>& raw_gradient,
                                             uint64_t batch_size,
                                             float learning_rate) {
    const float step = learning_rate /
                       (static_cast<float>(batch_size) * Impl::Loss::getLossScalingFactor());
    std::vector<float> updated(initial.size());
    for (uint64_t i = 0; i < initial.size(); ++i) {
        updated[i] = initial[i] - step * raw_gradient[i];
    }
    return updated;
}

struct PersistentDagPlacedCustomLayerFixture {
    std::shared_ptr<Api::PlacedNetwork> placed_network;
    Impl::StampedNetwork* stamped_network = nullptr;
    std::shared_ptr<Impl::NetworkInput> physical_input;
    std::shared_ptr<Impl::NetworkOutput> physical_output;
    std::shared_ptr<Impl::CustomLayer> physical_layer;
};

template <typename ApiLayer>
PersistentDagPlacedCustomLayerFixture placePersistentDagSingleCustomLayerNetwork(Api::Network& network,
                                                                 const Api::NetworkInput& api_input,
                                                                 const Api::NetworkOutput& api_output,
                                                                 const ApiLayer& api_layer,
                                                                 uint32_t batch_size) {
    std::vector<Event> init_done_events;
    PersistentDagPlacedCustomLayerFixture fixture;
    fixture.placed_network = network.place(batch_size, init_done_events, false);
    persistentDagSynchronizeEvents(init_done_events);
    EXPECT_NE(fixture.placed_network, nullptr);
    if (!fixture.placed_network) {
        return fixture;
    }

    fixture.stamped_network = &fixture.placed_network->getStampedNetwork(0);
    fixture.physical_input = std::dynamic_pointer_cast<Impl::NetworkInput>(
        fixture.stamped_network->getPhysicalLayerFromApiLayer(api_input.getId()));
    fixture.physical_output = std::dynamic_pointer_cast<Impl::NetworkOutput>(
        fixture.stamped_network->getPhysicalLayerFromApiLayer(api_output.getId()));
    fixture.physical_layer = std::dynamic_pointer_cast<Impl::CustomLayer>(
        fixture.stamped_network->getPhysicalLayerFromApiLayer(api_layer.getId()));

    EXPECT_NE(fixture.physical_input, nullptr);
    EXPECT_NE(fixture.physical_output, nullptr);
    EXPECT_NE(fixture.physical_layer, nullptr);
    return fixture;
}

void runPersistentDagForward(PersistentDagPlacedCustomLayerFixture& fixture, Impl::Tensor& feature_input, uint32_t batch_size) {
    ASSERT_NE(fixture.physical_input, nullptr);
    ASSERT_NE(fixture.physical_output, nullptr);
    fixture.physical_input->forward(feature_input, false, batch_size);
    Event output_ready = fixture.physical_output->getOutputReadyEvent();
    output_ready.synchronize();
}

void expectSharedGeluBoundaryPlan(
    const Impl::CustomLayer::GenericSharedBackwardDebugDiagnostic& diagnostic,
    const std::string& boundary_kind) {
    ASSERT_EQ(diagnostic.clearStageKindNames.size(), diagnostic.clearStageDependencyIndices.size());

    size_t boundary_count = 0;
    std::string stage_summary;
    for (uint32_t stage_index = 0; stage_index < diagnostic.clearStageKindNames.size(); ++stage_index) {
        stage_summary += std::to_string(stage_index) + ":" + diagnostic.clearStageKindNames[stage_index] + " deps=[";
        for (uint32_t dependency : diagnostic.clearStageDependencyIndices[stage_index]) {
            stage_summary += std::to_string(dependency) + ",";
        }
        stage_summary += "] ";
        boundary_count += diagnostic.clearStageKindNames[stage_index] == boundary_kind ? 1u : 0u;
    }

    EXPECT_EQ(boundary_count, 2u)
        << "One shared GELU producer must emit exactly the two producer VJPs required for dInput/dWeights. "
        << "Plan: " << stage_summary;
}

void runPersistentDagOneBackwardAndExpectSinglePlanSubmission(PersistentDagPlacedCustomLayerFixture& fixture, uint32_t batch_size) {
    ASSERT_NE(fixture.physical_layer, nullptr);
    ASSERT_TRUE(fixture.physical_layer->getGradientUpdateStream().has_value());
    ASSERT_GT(fixture.physical_layer->getErrorInputs().size(), 0u);
    ASSERT_TRUE(fixture.physical_layer->getErrorInputs()[0].has_value());

    Stream stream = fixture.physical_layer->getStreams()[0];
    Stream gradient_stream = fixture.physical_layer->getGradientUpdateStream().value();
    Impl::Tensor error_input = fixture.physical_layer->getErrorInputs()[0].value();
    Impl::Tensor error_input_host = error_input.clone(persistentDagCpuPlacement);
    persistentDagFillFp32Tensor(error_input_host, 1.0f);
    error_input.copyFromAsync(error_input_host, stream);
    stream.synchronize();

    fixture.physical_layer->backward(error_input, batch_size);
    stream.synchronize();
    gradient_stream.synchronize();

    const auto diagnostic = fixture.physical_layer->genericSharedBackwardDebugDiagnostic();
    ASSERT_TRUE(diagnostic.has_value());
    EXPECT_EQ(diagnostic->clearExecutionCount, 1u)
        << "The producer-gradient stages must execute within one clear plan submission.";
    EXPECT_EQ(diagnostic->accumulateExecutionCount, 0u);
}
#endif


}  // namespace

// Persistent-DAG architecture regression suite.
//
// ExpressionPersistentDagInvariant tests define the fixed production contract:
// ordinary Expression composition owns immutable logical nodes, reusing a handle
// preserves authored identity, Outputs retains logical roots until lowering, and
// persistent graph transforms reuse unchanged structure. Physical->logical import
// is graph-scoped so shared source ancestry remains shared, and semantic logical
// inspection stays in logical IR rather than materializing a physical graph.
//
// Planner/runtime value identity is physical-node based. Structural equivalence
// may still drive execution grouping, implementation caching, and audited
// stage-local CSE, but it must not redefine authored value identity.
//
// Physical cloning inside compiler/transport IR remains legitimate after the
// explicit logical lowering boundary. AutoDiff must consume the correctly shared
// lowered DAG; it is not a structural-deduplication repair layer.

// PERSISTENT-DAG INVARIANT:
// Deriving two branches from one Expression handle and recombining them retains
// one logical projection node. The final logical/physical graph contains exactly
// one MATMUL/GEMM producer, with SIN and COS both referring to that same value.
TEST(ExpressionPersistentDagInvariant, OrdinaryCompositionKeepsDirectLogicalDependencies) {
    const Expression projection = fp32Projection();
    const Expression sine = projection.sin();
    const Expression cosine = projection.cos();

    const std::optional<Expression> sine_input = ExpressionInternalAccess::lhsDependency(sine);
    const std::optional<Expression> cosine_input = ExpressionInternalAccess::lhsDependency(cosine);
    ASSERT_TRUE(sine_input.has_value());
    ASSERT_TRUE(cosine_input.has_value());
    EXPECT_TRUE(sine_input->isSameLogicalNode(projection));
    EXPECT_TRUE(cosine_input->isSameLogicalNode(projection));
    EXPECT_FALSE(sine.isSameLogicalNode(cosine));
}

TEST(ExpressionPersistentDagInvariant, UnaryChainAddsOneLogicalNodePerAuthoredOperation) {
    Expression current = fp32Projection();
    for (size_t i = 0; i < 8; ++i) {
        const Expression previous = current;
        current = current.sin();
        const std::optional<Expression> input = ExpressionInternalAccess::lhsDependency(current);
        ASSERT_TRUE(input.has_value());
        EXPECT_TRUE(input->isSameLogicalNode(previous)) << "step " << i;
        EXPECT_FALSE(current.isSameLogicalNode(previous)) << "step " << i;
    }
}

TEST(ExpressionPersistentDagInvariant, ScanWithIndicesOutputsShareOneLogicalInput) {
    const Expression projection = fp32Projection();
    const auto [scan_values, scan_indices] = projection.scanWithIndices(ScanOp::Max, -1, true);

    const std::optional<Expression> values_input = ExpressionInternalAccess::lhsDependency(scan_values);
    const std::optional<Expression> indices_input = ExpressionInternalAccess::lhsDependency(scan_indices);
    ASSERT_TRUE(values_input.has_value());
    ASSERT_TRUE(indices_input.has_value());
    EXPECT_TRUE(values_input->isSameLogicalNode(projection));
    EXPECT_TRUE(indices_input->isSameLogicalNode(projection));
    EXPECT_FALSE(scan_values.isSameLogicalNode(scan_indices));

    const ExprNode& value_semantics = ExpressionInternalAccess::rootSemantics(scan_values);
    const ExprNode& index_semantics = ExpressionInternalAccess::rootSemantics(scan_indices);
    EXPECT_EQ(value_semantics.op, ExprOp::SCAN);
    EXPECT_EQ(value_semantics.scan_op, ScanOp::Max);
    EXPECT_EQ(value_semantics.scan_mode, ScanMode::Inclusive);
    EXPECT_EQ(value_semantics.scan_axis, UINT64_MAX);
    EXPECT_EQ(index_semantics.op, ExprOp::SCAN);
    EXPECT_EQ(index_semantics.scan_op, ScanOp::ArgMax);
    EXPECT_EQ(index_semantics.scan_mode, ScanMode::Inclusive);
    EXPECT_EQ(index_semantics.scan_axis, UINT64_MAX);

    const PhysicalOutputs outputs =
        Expression::outputs({{"values", scan_values}, {"indices", scan_indices}}).physicalOutputs();
    ASSERT_TRUE(outputs.expr);
    ASSERT_EQ(outputs.outputs.size(), 2u);
    EXPECT_EQ(countLinearBoundaryNodes(outputs), 1u);
    EXPECT_EQ(countNodesOfKind(outputs, ExprOp::SCAN), 2u);

    const ExprNode& lowered_values = outputs.expr->nodes.at(outputs.outputs[0].node_idx);
    const ExprNode& lowered_indices = outputs.expr->nodes.at(outputs.outputs[1].node_idx);
    ASSERT_EQ(lowered_values.op, ExprOp::SCAN);
    ASSERT_EQ(lowered_indices.op, ExprOp::SCAN);
    EXPECT_EQ(lowered_values.lhs, lowered_indices.lhs);
}

TEST(ExpressionPersistentDagInvariant, SegmentedScanWithIndicesOutputsShareLogicalInputAndOffsets) {
    const Expression projection = fp32Projection();
    const Expression offsets = Expression::input("offsets", DataType::UINT32, DataType::UINT32);
    const auto [scan_values, scan_indices] = projection.segmentedScanWithIndices(offsets, ScanOp::Min, false);

    const std::optional<Expression> values_input = ExpressionInternalAccess::lhsDependency(scan_values);
    const std::optional<Expression> indices_input = ExpressionInternalAccess::lhsDependency(scan_indices);
    const std::optional<Expression> values_offsets = ExpressionInternalAccess::rhsDependency(scan_values);
    const std::optional<Expression> indices_offsets = ExpressionInternalAccess::rhsDependency(scan_indices);
    ASSERT_TRUE(values_input.has_value());
    ASSERT_TRUE(indices_input.has_value());
    ASSERT_TRUE(values_offsets.has_value());
    ASSERT_TRUE(indices_offsets.has_value());
    EXPECT_TRUE(values_input->isSameLogicalNode(projection));
    EXPECT_TRUE(indices_input->isSameLogicalNode(projection));
    EXPECT_TRUE(values_offsets->isSameLogicalNode(offsets));
    EXPECT_TRUE(indices_offsets->isSameLogicalNode(offsets));
    EXPECT_FALSE(scan_values.isSameLogicalNode(scan_indices));

    const ExprNode& value_semantics = ExpressionInternalAccess::rootSemantics(scan_values);
    const ExprNode& index_semantics = ExpressionInternalAccess::rootSemantics(scan_indices);
    EXPECT_EQ(value_semantics.op, ExprOp::SEGMENTED_SCAN);
    EXPECT_EQ(value_semantics.scan_op, ScanOp::Min);
    EXPECT_EQ(value_semantics.scan_mode, ScanMode::Exclusive);
    EXPECT_EQ(value_semantics.scan_axis, UINT64_MAX);
    EXPECT_EQ(index_semantics.op, ExprOp::SEGMENTED_SCAN);
    EXPECT_EQ(index_semantics.scan_op, ScanOp::ArgMin);
    EXPECT_EQ(index_semantics.scan_mode, ScanMode::Exclusive);
    EXPECT_EQ(index_semantics.scan_axis, UINT64_MAX);

    const PhysicalOutputs outputs =
        Expression::outputs({{"values", scan_values}, {"indices", scan_indices}}).physicalOutputs();
    ASSERT_TRUE(outputs.expr);
    ASSERT_EQ(outputs.outputs.size(), 2u);
    EXPECT_EQ(countLinearBoundaryNodes(outputs), 1u);
    EXPECT_EQ(countInputNodesNamed(outputs, "offsets"), 1u);
    EXPECT_EQ(countNodesOfKind(outputs, ExprOp::SEGMENTED_SCAN), 2u);

    const ExprNode& lowered_values = outputs.expr->nodes.at(outputs.outputs[0].node_idx);
    const ExprNode& lowered_indices = outputs.expr->nodes.at(outputs.outputs[1].node_idx);
    ASSERT_EQ(lowered_values.op, ExprOp::SEGMENTED_SCAN);
    ASSERT_EQ(lowered_indices.op, ExprOp::SEGMENTED_SCAN);
    EXPECT_EQ(lowered_values.lhs, lowered_indices.lhs);
    EXPECT_EQ(lowered_values.rhs, lowered_indices.rhs);
}

TEST(ExpressionPersistentDagInvariant, UnaryFanoutRecombineSharesOneProducer) {
    const Expression projection = fp32Projection();
    const Expression y = projection.sin() + projection.cos();
    const PhysicalOutputs outputs = Expression::outputs({{"y", y}}).physicalOutputs();

    EXPECT_EQ(countLinearBoundaryNodes(outputs), 1u);
}

// PERSISTENT-DAG INVARIANT:
// Expression::outputs() lowers all output handles by walking one shared logical
// DAG. Separately derived outputs from the same projection therefore contain one
// projection producer, not one copy per output expression.
TEST(ExpressionPersistentDagInvariant, MultiOutputLoweringSharesOneProducer) {
    const Expression projection = fp32Projection();
    const Expression sin_projection = projection.sin();
    const Expression cos_projection = projection.cos();
    const PhysicalOutputs outputs =
        Expression::outputs({{"sin_projection", sin_projection}, {"cos_projection", cos_projection}}).physicalOutputs();

    EXPECT_EQ(countLinearBoundaryNodes(outputs), 1u);
}

TEST(ExpressionPersistentDagInvariant, OutputsRetainAuthoredLogicalRootsUntilPhysicalBoundary) {
    const Expression projection = fp32Projection();
    const Expression sin_projection = projection.sin();
    const Expression cos_projection = projection.cos();
    const Outputs outputs =
        Expression::outputs({{"sin_projection", sin_projection}, {"cos_projection", cos_projection}});

    EXPECT_EQ(outputs.outputNames(), (std::vector<std::string>{"sin_projection", "cos_projection"}));
    EXPECT_TRUE(outputs.outputExpression("sin_projection").isSameLogicalNode(sin_projection));
    EXPECT_TRUE(outputs.outputExpression(1).isSameLogicalNode(cos_projection));

    const PhysicalOutputs first = outputs.physicalOutputs();
    const PhysicalOutputs second = outputs.physicalOutputs();
    ASSERT_TRUE(first.expr);
    ASSERT_TRUE(second.expr);
    EXPECT_NE(first.expr.get(), second.expr.get())
        << "Ordinary Outputs should not retain a mutable physical backing graph between lowering boundaries.";
    EXPECT_EQ(countLinearBoundaryNodes(first), 1u);
    EXPECT_EQ(countLinearBoundaryNodes(second), 1u);
}

TEST(ExpressionPersistentDagInvariant, FromPhysicalOutputsImportsAllOrdinaryRootsThroughOneContext) {
    const Expression projection = fp32Projection();
    PhysicalOutputs physical =
        Expression::outputs({{"sin_projection", projection.sin()}, {"cos_projection", projection.cos()}}).physicalOutputs();

    const Outputs imported = Outputs::fromPhysicalOutputs(std::move(physical));
    const Expression imported_sin = imported.outputExpression("sin_projection");
    const Expression imported_cos = imported.outputExpression("cos_projection");
    const std::optional<Expression> sin_input = ExpressionInternalAccess::lhsDependency(imported_sin);
    const std::optional<Expression> cos_input = ExpressionInternalAccess::lhsDependency(imported_cos);
    ASSERT_TRUE(sin_input.has_value());
    ASSERT_TRUE(cos_input.has_value());
    EXPECT_TRUE(sin_input->isSameLogicalNode(*cos_input));

    const PhysicalOutputs round_tripped = imported.physicalOutputs();
    EXPECT_EQ(countLinearBoundaryNodes(round_tripped), 1u);

    const Outputs independently_transformed = Expression::outputs({
        {"sin_projection", imported_sin + Expression::constantScalar(1.0)},
        {"cos_projection", imported_cos + Expression::constantScalar(2.0)},
    });
    const PhysicalOutputs transformed_round_trip = independently_transformed.physicalOutputs();
    EXPECT_EQ(countLinearBoundaryNodes(transformed_round_trip), 1u)
        << "Independent transforms of roots imported together must retain their shared physical ancestry.";
}

// PERSISTENT-DAG INVARIANT:
// GELU's natural definition `x * Phi(x)` reuses one authored x node across both
// branches, so an expensive producer feeding GELU occurs exactly once without a
// GELU-specific graph-construction workaround.
TEST(ExpressionPersistentDagInvariant, NaturalGeluSharesOneProducer) {
    const Expression projection = fp32Projection();
    const PhysicalOutputs outputs = Expression::outputs({{"y", projection.gelu()}}).physicalOutputs();

    EXPECT_EQ(countLinearBoundaryNodes(outputs), 1u);
}


// PERSISTENT-DAG INVARIANT:
// One forward projection feeding natural GELU accumulates one projection adjoint
// and emits exactly one producer VJP pair: dX and dW. No GELU-specific graph
// construction or structural VJP deduplication is required.
TEST(ExpressionPersistentDagInvariant, NaturalGeluProducesOneProjectionVjpPair) {
    const Expression projection = fp32Projection();
    const BackwardBuildResult backward = buildProjectionBackward(projection.gelu());

    EXPECT_EQ(countLinearBoundaryNodes(backward.outputs), 2u);
}

// PERSISTENT-DAG INVARIANT:
// Multi-step convenience functions compose ordinary Expression handles without
// multiplying shared ancestry. Every case below is derived from one projection
// and lowers to exactly one projection producer.
TEST(ExpressionPersistentDagInvariant, ConvenienceOpsPreserveSharedAncestry) {
    struct Case {
        const char* name;
        std::function<Expression(const Expression&)> build;
    };

    const std::vector<Case> cases = {
        {"sigmoid", [](const Expression& x) { return x.sigmoid(); }},
        {"softplus", [](const Expression& x) { return x.softplus(); }},
        {"elu", [](const Expression& x) { return x.elu(); }},
        {"selu", [](const Expression& x) { return x.selu(); }},
        {"mish", [](const Expression& x) { return x.mish(); }},
        {"hard_swish", [](const Expression& x) { return x.hardSwish(); }},
        {"threshold", [](const Expression& x) { return x.threshold(); }},
        {"swish", [](const Expression& x) { return x.swish(); }},
    };

    for (const Case& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        const Expression projection = fp32Projection();
        const PhysicalOutputs outputs = Expression::outputs({{"y", test_case.build(projection)}}).physicalOutputs();
        EXPECT_EQ(countLinearBoundaryNodes(outputs), 1u);
    }
}

// PERSISTENT-DAG INVARIANT:
// Fan-out is represented by additional outgoing edges, not by copying ancestor
// graphs. Regardless of fan-out depth, this expression has one authored projection
// and retains exactly one projection node.
TEST(ExpressionPersistentDagInvariant, NestedFanoutPreservesOneProducer) {
    const Expression projection = fp32Projection();
    const Expression level1 = projection.sin() + projection.cos();
    const Expression level2 = level1.exp() + level1.tanh();
    const Expression level3 = level2.sqrt() + level2.abs();
    const PhysicalOutputs outputs = Expression::outputs({{"y", level3}}).physicalOutputs();

    EXPECT_EQ(countLinearBoundaryNodes(outputs), 1u);
}

// PERSISTENT-DAG INVARIANT:
// Reusing the exact same Expression in both operand slots preserves one logical
// producer by authored node identity, independent of physical backing allocation.
TEST(ExpressionPersistentDagInvariant, DirectReusePreservesLogicalIdentity) {
    const Expression projection = fp32Projection();
    const PhysicalOutputs outputs = Expression::outputs({{"y", projection * projection}}).physicalOutputs();

    EXPECT_EQ(countLinearBoundaryNodes(outputs), 1u);
}


// PERSISTENT-DAG INVARIANT:
// GEMM consumes authored logical handles directly. Reusing one projection in
// lhs/rhs/addend therefore preserves one producer identity rather than cloning
// that producer once per operand role.
TEST(ExpressionPersistentDagInvariant, GemmBuilderPreservesRepeatedOperandIdentity) {
    const Expression projection = fp32Projection();
    const Expression outer = Expression::gemm(
        projection, projection, projection, 1.0, 1.0, true, false, true, DataType::FP32, DataType::FP16);

    const std::optional<Expression> lhs = ExpressionInternalAccess::lhsDependency(outer);
    const std::optional<Expression> rhs = ExpressionInternalAccess::rhsDependency(outer);
    const std::optional<Expression> aux = ExpressionInternalAccess::auxDependency(outer);
    ASSERT_TRUE(lhs.has_value());
    ASSERT_TRUE(rhs.has_value());
    ASSERT_TRUE(aux.has_value());
    EXPECT_TRUE(lhs->isSameLogicalNode(projection));
    EXPECT_TRUE(rhs->isSameLogicalNode(projection));
    EXPECT_TRUE(aux->isSameLogicalNode(projection));
    EXPECT_FALSE(ExpressionInternalAccess::alphaDependency(outer).has_value());
    EXPECT_FALSE(ExpressionInternalAccess::betaDependency(outer).has_value());

    const ExprNode& logical_semantics = ExpressionInternalAccess::rootSemantics(outer);
    EXPECT_EQ(logical_semantics.op, ExprOp::GEMM);
    EXPECT_DOUBLE_EQ(logical_semantics.alpha_fp, 1.0);
    EXPECT_DOUBLE_EQ(logical_semantics.beta_fp, 1.0);
    EXPECT_TRUE(logical_semantics.transpose_lhs);
    EXPECT_FALSE(logical_semantics.transpose_rhs);
    EXPECT_TRUE(logical_semantics.transpose_aux);
    EXPECT_EQ(logical_semantics.compute_dtype, std::optional<DataType>(DataType::FP32));
    EXPECT_EQ(logical_semantics.output_dtype, std::optional<DataType>(DataType::FP16));

    const PhysicalOutputs outputs = Expression::outputs({{"y", outer}}).physicalOutputs();
    ASSERT_TRUE(outputs.expr);
    ASSERT_EQ(outputs.outputs.size(), 1u);
    const ExprNode& physical_gemm = outputs.expr->nodes.at(outputs.outputs.front().node_idx);
    EXPECT_EQ(countLinearBoundaryNodes(outputs), 2u);
    EXPECT_EQ(physical_gemm.op, ExprOp::GEMM);
    EXPECT_EQ(physical_gemm.lhs, physical_gemm.rhs);
    EXPECT_EQ(physical_gemm.lhs, physical_gemm.aux);
    EXPECT_EQ(physical_gemm.alpha_node, UINT32_MAX);
    EXPECT_EQ(physical_gemm.beta_node, UINT32_MAX);
}

TEST(ExpressionPersistentDagInvariant, GemmScaleEncodingPreservesHistoricalContract) {
    const Expression lhs = Expression::input("gemm_lhs", DataType::FP32, DataType::FP32);
    const Expression rhs = Expression::input("gemm_rhs", DataType::FP32, DataType::FP32);
    const Expression addend = Expression::input("gemm_addend", DataType::FP32, DataType::FP32);

    {
        const Expression gemm = Expression::gemm(lhs, rhs, addend, 2.5, -3.0);
        const ExprNode& semantics = ExpressionInternalAccess::rootSemantics(gemm);
        EXPECT_DOUBLE_EQ(semantics.alpha_fp, 2.5);
        EXPECT_DOUBLE_EQ(semantics.beta_fp, -3.0);
        EXPECT_FALSE(ExpressionInternalAccess::alphaDependency(gemm).has_value());
        EXPECT_FALSE(ExpressionInternalAccess::betaDependency(gemm).has_value());

        const PhysicalOutputs outputs = Expression::outputs({{"y", gemm}}).physicalOutputs();
        const ExprNode& physical = outputs.expr->nodes.at(outputs.outputs.front().node_idx);
        EXPECT_DOUBLE_EQ(physical.alpha_fp, 2.5);
        EXPECT_DOUBLE_EQ(physical.beta_fp, -3.0);
        EXPECT_EQ(physical.alpha_node, UINT32_MAX);
        EXPECT_EQ(physical.beta_node, UINT32_MAX);
    }

    const Expression runtime_alpha = Expression::runtimeScalar("gemm_alpha", DataType::FP32, DataType::FP32);
    const Expression tensor_beta = Expression::tensorRuntimeScalar("gemm_beta", DataType::FP32, DataType::FP32);
    {
        const Expression gemm = Expression::gemm(lhs, rhs, addend, runtime_alpha, tensor_beta);
        const std::optional<Expression> alpha = ExpressionInternalAccess::alphaDependency(gemm);
        const std::optional<Expression> beta = ExpressionInternalAccess::betaDependency(gemm);
        ASSERT_TRUE(alpha.has_value());
        ASSERT_TRUE(beta.has_value());
        EXPECT_TRUE(alpha->isSameLogicalNode(runtime_alpha));
        EXPECT_TRUE(beta->isSameLogicalNode(tensor_beta));
        EXPECT_DOUBLE_EQ(ExpressionInternalAccess::rootSemantics(gemm).alpha_fp, 1.0);
        EXPECT_DOUBLE_EQ(ExpressionInternalAccess::rootSemantics(gemm).beta_fp, 1.0);

        const PhysicalOutputs outputs = Expression::outputs({{"y", gemm}}).physicalOutputs();
        const ExprNode& physical = outputs.expr->nodes.at(outputs.outputs.front().node_idx);
        ASSERT_NE(physical.alpha_node, UINT32_MAX);
        ASSERT_NE(physical.beta_node, UINT32_MAX);
        EXPECT_EQ(outputs.expr->nodes.at(physical.alpha_node).op, ExprOp::RUNTIME_SCALAR);
        EXPECT_EQ(outputs.expr->nodes.at(physical.beta_node).op, ExprOp::TENSOR_RUNTIME_SCALAR);
    }

    {
        const Expression scaled_alpha = Expression::constantScalar(2.5) * runtime_alpha;
        const Expression scaled_beta = tensor_beta * Expression::constantScalar(-4.0);
        const Expression gemm = Expression::gemm(lhs, rhs, addend, scaled_alpha, scaled_beta);
        const std::optional<Expression> alpha = ExpressionInternalAccess::alphaDependency(gemm);
        const std::optional<Expression> beta = ExpressionInternalAccess::betaDependency(gemm);
        ASSERT_TRUE(alpha.has_value());
        ASSERT_TRUE(beta.has_value());
        EXPECT_TRUE(alpha->isSameLogicalNode(runtime_alpha));
        EXPECT_TRUE(beta->isSameLogicalNode(tensor_beta));
        EXPECT_DOUBLE_EQ(ExpressionInternalAccess::rootSemantics(gemm).alpha_fp, 2.5);
        EXPECT_DOUBLE_EQ(ExpressionInternalAccess::rootSemantics(gemm).beta_fp, -4.0);

        const PhysicalOutputs outputs = Expression::outputs({{"y", gemm}}).physicalOutputs();
        const ExprNode& physical = outputs.expr->nodes.at(outputs.outputs.front().node_idx);
        EXPECT_DOUBLE_EQ(physical.alpha_fp, 2.5);
        EXPECT_DOUBLE_EQ(physical.beta_fp, -4.0);
        ASSERT_NE(physical.alpha_node, UINT32_MAX);
        ASSERT_NE(physical.beta_node, UINT32_MAX);
        EXPECT_EQ(outputs.expr->nodes.at(physical.alpha_node).op, ExprOp::RUNTIME_SCALAR);
        EXPECT_EQ(outputs.expr->nodes.at(physical.beta_node).op, ExprOp::TENSOR_RUNTIME_SCALAR);
    }

    {
        const Expression general_scale = runtime_alpha + Expression::constantScalar(1.0);
        const Expression gemm = Expression::gemm(lhs, rhs, addend, general_scale, general_scale);
        const std::optional<Expression> alpha = ExpressionInternalAccess::alphaDependency(gemm);
        const std::optional<Expression> beta = ExpressionInternalAccess::betaDependency(gemm);
        ASSERT_TRUE(alpha.has_value());
        ASSERT_TRUE(beta.has_value());
        EXPECT_TRUE(alpha->isSameLogicalNode(general_scale));
        EXPECT_TRUE(beta->isSameLogicalNode(general_scale));
        EXPECT_TRUE(alpha->isSameLogicalNode(*beta));

        const PhysicalOutputs outputs = Expression::outputs({{"y", gemm}}).physicalOutputs();
        const ExprNode& physical = outputs.expr->nodes.at(outputs.outputs.front().node_idx);
        ASSERT_NE(physical.alpha_node, UINT32_MAX);
        EXPECT_EQ(physical.alpha_node, physical.beta_node);
        EXPECT_EQ(outputs.expr->nodes.at(physical.alpha_node).op, ExprOp::ADD);
    }
}

// PERSISTENT-DAG INVARIANT:
// Attention is one logical node whose operand roles point directly at the authored
// Q/K/V values. Reusing one projection in several roles must preserve that exact
// logical identity and lower to one shared physical producer.
TEST(ExpressionPersistentDagInvariant, AttentionBuilderPreservesRepeatedOperandIdentity) {
    const Expression projection = fp32Projection();
    AttentionOptions options;
    options.q_layout = AttentionTensorLayout::BSHD;
    options.k_layout = AttentionTensorLayout::BSHD;
    options.v_layout = AttentionTensorLayout::BSHD;
    options.o_layout = AttentionTensorLayout::BSHD;
    options.mask_kind = AttentionMaskKind::CausalTopLeft;
    options.diagonal_left_bound = 17;
    options.attention_scale = 0.125f;
    options.use_alibi_mask = true;
    options.compute_dtype = DataType::FP32;
    options.output_dtype = DataType::FP16;
    const Expression attention = Expression::attention(projection, projection, projection, projection, options);

    const std::optional<Expression> q = ExpressionInternalAccess::lhsDependency(attention);
    const std::optional<Expression> k = ExpressionInternalAccess::rhsDependency(attention);
    const std::optional<Expression> v = ExpressionInternalAccess::auxDependency(attention);
    const std::optional<Expression> bias = ExpressionInternalAccess::alphaDependency(attention);
    ASSERT_TRUE(q.has_value());
    ASSERT_TRUE(k.has_value());
    ASSERT_TRUE(v.has_value());
    ASSERT_TRUE(bias.has_value());
    EXPECT_TRUE(q->isSameLogicalNode(projection));
    EXPECT_TRUE(k->isSameLogicalNode(projection));
    EXPECT_TRUE(v->isSameLogicalNode(projection));
    EXPECT_TRUE(bias->isSameLogicalNode(projection));

    const ExprNode& semantics = ExpressionInternalAccess::rootSemantics(attention);
    EXPECT_EQ(semantics.op, ExprOp::ATTENTION);
    EXPECT_TRUE(semantics.attention_use_bias);
    EXPECT_EQ(semantics.attention_q_layout, AttentionTensorLayout::BSHD);
    EXPECT_EQ(semantics.attention_k_layout, AttentionTensorLayout::BSHD);
    EXPECT_EQ(semantics.attention_v_layout, AttentionTensorLayout::BSHD);
    EXPECT_EQ(semantics.attention_o_layout, AttentionTensorLayout::BSHD);
    EXPECT_EQ(semantics.attention_mask_kind, AttentionMaskKind::CausalTopLeft);
    EXPECT_EQ(semantics.attention_diagonal_left_bound, 17);
    EXPECT_TRUE(semantics.attention_has_scale);
    EXPECT_FLOAT_EQ(semantics.attention_scale, 0.125f);
    EXPECT_TRUE(semantics.attention_use_alibi_mask);
    EXPECT_EQ(semantics.compute_dtype, std::optional<DataType>(DataType::FP32));
    EXPECT_EQ(semantics.output_dtype, std::optional<DataType>(DataType::FP16));
    EXPECT_EQ(semantics.lhs, UINT32_MAX);
    EXPECT_EQ(semantics.rhs, UINT32_MAX);
    EXPECT_EQ(semantics.aux, UINT32_MAX);
    EXPECT_EQ(semantics.alpha_node, UINT32_MAX);

    const PhysicalOutputs outputs = Expression::outputs({{"y", attention}}).physicalOutputs();
    ASSERT_EQ(countLinearBoundaryNodes(outputs), 1u);
    const ExprNode& physical = outputs.expr->nodes.at(outputs.outputs.front().node_idx);
    ASSERT_EQ(physical.op, ExprOp::ATTENTION);
    EXPECT_EQ(physical.lhs, physical.rhs);
    EXPECT_EQ(physical.lhs, physical.aux);
    EXPECT_EQ(physical.lhs, physical.alpha_node);
    EXPECT_EQ(physical.attention_q_layout, semantics.attention_q_layout);
    EXPECT_EQ(physical.attention_k_layout, semantics.attention_k_layout);
    EXPECT_EQ(physical.attention_v_layout, semantics.attention_v_layout);
    EXPECT_EQ(physical.attention_o_layout, semantics.attention_o_layout);
    EXPECT_EQ(physical.attention_mask_kind, semantics.attention_mask_kind);
    EXPECT_EQ(physical.attention_diagonal_left_bound, semantics.attention_diagonal_left_bound);
    EXPECT_EQ(physical.attention_has_scale, semantics.attention_has_scale);
    EXPECT_FLOAT_EQ(physical.attention_scale, semantics.attention_scale);
    EXPECT_EQ(physical.attention_use_alibi_mask, semantics.attention_use_alibi_mask);
    EXPECT_EQ(physical.compute_dtype, semantics.compute_dtype);
    EXPECT_EQ(physical.output_dtype, semantics.output_dtype);
}

// PERSISTENT-DAG INVARIANT:
// Attention metadata is represented exclusively as logical dependencies. Reusing
// one authored metadata value in compatible roles must remain reuse, including
// padding lengths, ragged offsets, paged-KV page tables, dropout state, and all
// FP8 scale/descale/amax roles.
TEST(ExpressionPersistentDagInvariant, AttentionMetadataDependenciesPreserveAuthoredIdentity) {
    const Expression q = Expression::input("q", DataType::FP32, DataType::FP32);
    const Expression k = Expression::input("k", DataType::FP32, DataType::FP32);
    const Expression v = Expression::input("v", DataType::FP32, DataType::FP32);
    const Expression shared = Expression::input("shared_meta", DataType::FP32, DataType::FP32);

    auto expect_shared_dependency = [&](const Expression& attention, LogicalDependencyKind kind) {
        const std::optional<Expression> dependency = ExpressionInternalAccess::logicalDependency(attention, kind);
        ASSERT_TRUE(dependency.has_value());
        EXPECT_TRUE(dependency->isSameLogicalNode(shared));
    };

    {
        AttentionOptions options;
        const Expression attention = Expression::scaledDotProductAttention(q, k, v, shared, shared, options);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionSeqLenQ);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionSeqLenKv);

        const PhysicalOutputs outputs = Expression::outputs({{"y", attention}}).physicalOutputs();
        const ExprNode& physical = outputs.expr->nodes.at(outputs.outputs.front().node_idx);
        EXPECT_TRUE(physical.attention_use_padding_mask);
        ASSERT_NE(physical.attention_seq_len_q_node, UINT32_MAX);
        EXPECT_EQ(physical.attention_seq_len_q_node, physical.attention_seq_len_kv_node);
    }

    {
        AttentionOptions options;
        options.q_layout = AttentionTensorLayout::BSHD;
        options.k_layout = AttentionTensorLayout::BSHD;
        options.v_layout = AttentionTensorLayout::BSHD;
        options.o_layout = AttentionTensorLayout::BSHD;
        const Expression attention = Expression::scaledDotProductAttentionRagged(q, k, v, shared, shared, options);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionRaggedOffsetQ);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionRaggedOffsetKv);

        const PhysicalOutputs outputs = Expression::outputs({{"y", attention}}).physicalOutputs();
        const ExprNode& physical = outputs.expr->nodes.at(outputs.outputs.front().node_idx);
        EXPECT_TRUE(physical.attention_use_ragged_offsets);
        ASSERT_NE(physical.attention_ragged_offset_q_node, UINT32_MAX);
        EXPECT_EQ(physical.attention_ragged_offset_q_node, physical.attention_ragged_offset_kv_node);
    }

    {
        AttentionOptions options;
        options.paged_kv_max_sequence_length = 128;
        const Expression attention = Expression::scaledDotProductAttentionPagedKv(q, k, v, shared, shared, shared, shared, options);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionSeqLenQ);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionSeqLenKv);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionPageTableK);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionPageTableV);

        const PhysicalOutputs outputs = Expression::outputs({{"y", attention}}).physicalOutputs();
        const ExprNode& physical = outputs.expr->nodes.at(outputs.outputs.front().node_idx);
        EXPECT_TRUE(physical.attention_use_padding_mask);
        EXPECT_TRUE(physical.attention_use_paged_kv_cache);
        ASSERT_NE(physical.attention_page_table_k_node, UINT32_MAX);
        EXPECT_EQ(physical.attention_page_table_k_node, physical.attention_page_table_v_node);
    }

    {
        AttentionOptions options;
        options.dropout_probability = 0.25f;
        const Expression attention = Expression::scaledDotProductAttentionWithDropout(q, k, v, shared, shared, options);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionDropoutSeed);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionDropoutOffset);

        const PhysicalOutputs outputs = Expression::outputs({{"y", attention}}).physicalOutputs();
        const ExprNode& physical = outputs.expr->nodes.at(outputs.outputs.front().node_idx);
        EXPECT_FLOAT_EQ(physical.attention_dropout_probability, 0.25f);
        ASSERT_NE(physical.attention_dropout_seed_node, UINT32_MAX);
        EXPECT_EQ(physical.attention_dropout_seed_node, physical.attention_dropout_offset_node);
    }

    {
        AttentionOptions options;
        const Expression attention = Expression::scaledDotProductAttentionFp8Forward(
            q, k, v, shared, shared, shared, shared, shared, shared, shared, shared, options);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionDescaleQ);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionDescaleK);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionDescaleV);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionDescaleS);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionScaleS);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionScaleO);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionAmaxS);
        expect_shared_dependency(attention, LogicalDependencyKind::AttentionAmaxO);

        const PhysicalOutputs outputs = Expression::outputs({{"y", attention}}).physicalOutputs();
        const ExprNode& physical = outputs.expr->nodes.at(outputs.outputs.front().node_idx);
        EXPECT_TRUE(physical.attention_use_fp8_forward_scaling);
        const uint32_t shared_node = physical.attention_descale_q_node;
        ASSERT_NE(shared_node, UINT32_MAX);
        EXPECT_EQ(shared_node, physical.attention_descale_k_node);
        EXPECT_EQ(shared_node, physical.attention_descale_v_node);
        EXPECT_EQ(shared_node, physical.attention_descale_s_node);
        EXPECT_EQ(shared_node, physical.attention_scale_s_node);
        EXPECT_EQ(shared_node, physical.attention_scale_o_node);
        EXPECT_EQ(shared_node, physical.attention_amax_s_node);
        EXPECT_EQ(shared_node, physical.attention_amax_o_node);
    }
}

// PERSISTENT-DAG INVARIANT:
// A semantic root transformation path-copies only the node whose value semantics
// change. withOutputDType() creates a distinct SIN root while preserving the exact
// logical projection dependency used by the original SIN root.
TEST(ExpressionPersistentDagInvariant, CloneAndModifySharesUnchangedAncestors) {
    const Expression projection = fp32Projection();
    const Expression original = projection.sin();
    const std::string original_before = canonicalize(Expression::outputs({{"original", original}}).physicalOutputs());

    const Expression modified = original.withOutputDType(DataType::FP16);

    // Graph transformations are functional and never mutate the source Expression.
    EXPECT_EQ(canonicalize(Expression::outputs({{"original", original}}).physicalOutputs()), original_before);

    const PhysicalExpression original_physical = original.expression();
    const PhysicalExpression modified_physical = modified.expression();
    EXPECT_NE(original_physical.nodes.at(original_physical.output_node).output_dtype,
              std::optional<DataType>(DataType::FP16));
    EXPECT_EQ(modified_physical.nodes.at(modified_physical.output_node).output_dtype,
              std::optional<DataType>(DataType::FP16));

    const std::optional<Expression> original_projection = ExpressionInternalAccess::lhsDependency(original);
    const std::optional<Expression> modified_projection = ExpressionInternalAccess::lhsDependency(modified);
    ASSERT_TRUE(original_projection.has_value());
    ASSERT_TRUE(modified_projection.has_value());
    EXPECT_FALSE(original.isSameLogicalNode(modified));
    EXPECT_TRUE(original_projection->isSameLogicalNode(*modified_projection));

    const PhysicalOutputs outputs =
        Expression::outputs({{"original", original}, {"modified", modified}}).physicalOutputs();
    EXPECT_EQ(countLinearBoundaryNodes(outputs), 1u);
}

// PERSISTENT-DAG INVARIANT:
// SIN and COS contribute to one projection adjoint before the producer VJP is
// emitted. One forward MATMUL therefore has exactly two backward MATMULs: dX and
// dW, with one reverse visit for the shared producer node.
TEST(ExpressionPersistentDagInvariant, SharedProjectionProducesOneBackwardVjpPair) {
    const Expression projection = fp32Projection();
    const Expression sine = projection.sin();
    const Expression cosine = projection.cos();
    const Expression y = sine + cosine;

    const PhysicalOutputs forward = Expression::outputs({{"y", y}}).physicalOutputs();
    ASSERT_TRUE(forward.expr);
    ASSERT_EQ(countLinearBoundaryNodes(forward), 1u);
    ASSERT_EQ(forward.outputs.size(), 1u);
    const ExprNode& add = forward.expr->nodes.at(forward.outputs.front().node_idx);
    ASSERT_EQ(add.op, ExprOp::ADD);
    const ExprNode& physical_sine = forward.expr->nodes.at(add.lhs);
    const ExprNode& physical_cosine = forward.expr->nodes.at(add.rhs);
    ASSERT_EQ(physical_sine.op, ExprOp::SIN);
    ASSERT_EQ(physical_cosine.op, ExprOp::COS);
    EXPECT_EQ(physical_sine.lhs, physical_cosine.lhs)
        << "The different derivative branches must fan out from one physical producer node.";

    const BackwardBuildResult backward = buildProjectionBackward(y);
    EXPECT_EQ(countLinearBoundaryNodes(backward.outputs), 2u);
    expectProjectionVjpPairUsesOneAccumulatedAdjoint(backward.outputs);
}

// PERSISTENT-DAG / AUTODIFF INVARIANT:
// Separate public outputs may carry independent upstream seeds while sharing one
// executing producer. Public-output cardinality must not become producer-VJP
// cardinality: both branch contributions accumulate at the shared MATMUL node
// before its one dInput/dWeight VJP pair is generated.
TEST(ExpressionPersistentDagInvariant, SharedProjectionAcrossNamedOutputsProducesOneBackwardVjpPair) {
    const Expression projection = fp32Projection();
    const Expression sine = projection.sin();
    const Expression cosine = projection.cos();
    PhysicalOutputs forward =
        Expression::outputs({{"sin_projection", sine}, {"cos_projection", cosine}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP32, DataType::FP32});

    ASSERT_TRUE(forward.expr);
    ASSERT_EQ(forward.outputs.size(), 2u);
    ASSERT_EQ(countLinearBoundaryNodes(forward), 1u);
    const ExprNode& physical_sine = forward.expr->nodes.at(forward.outputs[0].node_idx);
    const ExprNode& physical_cosine = forward.expr->nodes.at(forward.outputs[1].node_idx);
    ASSERT_EQ(physical_sine.op, ExprOp::SIN);
    ASSERT_EQ(physical_cosine.op, ExprOp::COS);
    EXPECT_EQ(physical_sine.lhs, physical_cosine.lhs);

    const std::unordered_map<std::string, std::vector<uint64_t>> forward_input_dims = {
        {"x", {2, 3}},
        {"w", {3, 4}},
    };
    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x", "w"},
        std::unordered_map<std::string, std::string>{
            {"sin_projection", "d_sin_projection"},
            {"cos_projection", "d_cos_projection"},
        },
        std::unordered_map<std::string, DataType>{
            {"sin_projection", DataType::FP32},
            {"cos_projection", DataType::FP32},
        },
        std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>(forward_input_dims));

    EXPECT_EQ(countLinearBoundaryNodes(backward.outputs), 2u)
        << "Two named outputs sharing one producer must not generate one projection VJP pair per output.";
    expectProjectionVjpPairUsesOneAccumulatedAdjoint(backward.outputs);
}

// PERSISTENT-DAG / AUTODIFF INVARIANT:
// Naming the exact same forward root more than once creates several upstream
// seeds for one forward node, not several executing producers. addContribution()
// must combine those seeds by node index before the producer's reverse rule runs.
TEST(ExpressionPersistentDagInvariant, SameProjectionRootNamedTwiceAccumulatesSeedsBeforeOneBackwardVjpPair) {
    const Expression projection = fp32Projection();
    PhysicalOutputs forward = Expression::outputs({{"a", projection}, {"b", projection}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP32, DataType::FP32});

    ASSERT_TRUE(forward.expr);
    ASSERT_EQ(forward.outputs.size(), 2u);
    ASSERT_EQ(countLinearBoundaryNodes(forward), 1u);
    ASSERT_EQ(forward.outputs[0].node_idx, forward.outputs[1].node_idx)
        << "Two names for one root must preserve one physical producer identity.";

    const std::unordered_map<std::string, std::vector<uint64_t>> forward_input_dims = {
        {"x", {2, 3}},
        {"w", {3, 4}},
    };
    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x", "w"},
        std::unordered_map<std::string, std::string>{{"a", "da"}, {"b", "db"}},
        std::unordered_map<std::string, DataType>{{"a", DataType::FP32}, {"b", DataType::FP32}},
        std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>(forward_input_dims));

    EXPECT_EQ(countLinearBoundaryNodes(backward.outputs), 2u)
        << "Multiple seeds for one root must still execute the producer reverse rule only once.";
    expectProjectionVjpPairUsesOneAccumulatedAdjoint(backward.outputs);
}

// PERSISTENT-DAG / SAVED-FORWARD INVARIANT:
// One authored projection is shared by SIN and COS, and both derivative rules
// request that same computed primal. Saved-forward identity follows the physical
// forward node, so fan-out must publish one retained NodeOutput requirement.
TEST(ExpressionPersistentDagInvariant, SharedProjectionCreatesOneRetainedForwardRequirement) {
    const Expression projection = fp32Projection();
    const Expression y = projection.sin() + projection.cos();
    PhysicalOutputs forward = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP32, DataType::FP32});
    const std::unordered_map<std::string, std::vector<uint64_t>> forward_input_dims = {
        {"x", {2, 3}},
        {"w", {3, 4}},
    };
    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x", "w"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::optional<std::unordered_map<std::string, std::vector<uint64_t>>>(forward_input_dims));

    expectUniqueForwardValueRequirementIdentities(backward);
    ASSERT_EQ(backward.forward_value_requirements.size(), 1u);
    EXPECT_EQ(countRetainedLinearForwardValues(forward, backward), 1u);
    EXPECT_EQ(backward.forward_value_requirements.front().kind, ForwardValueRequirementKind::NodeOutput);
}

// PERSISTENT-DAG / SAVED-FORWARD INVARIANT:
// SIN and COS both need the shared TANH value, and TANH backward itself also
// needs that same materialized output. All forwardValue() requests for the one
// executing TANH node must reuse one retained NodeOutput requirement/binding.
TEST(ExpressionPersistentDagInvariant, SharedTanhFanoutCreatesOneRetainedForwardRequirement) {
    const Expression projection = fp32Projection();
    const Expression shared_tanh = projection.tanh();
    const Expression y = shared_tanh.sin() + shared_tanh.cos();

    PhysicalOutputs forward = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP32, DataType::FP32});
    ASSERT_TRUE(forward.expr);
    const uint32_t tanh_node = findSingleNodeOfKind(forward, ExprOp::TANH);
    ASSERT_NE(tanh_node, UINT32_MAX);

    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x", "w"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {2, 3}},
            {"w", {3, 4}},
        });

    expectUniqueForwardValueRequirementIdentities(backward);
    ASSERT_EQ(backward.forward_value_requirements.size(), 1u)
        << "Fan-out and TANH's own VJP must reuse one retained output of the shared TANH node.";
    const ForwardValueRequirement& requirement = backward.forward_value_requirements.front();
    EXPECT_EQ(requirement.forward_node_index, tanh_node);
    EXPECT_EQ(requirement.kind, ForwardValueRequirementKind::NodeOutput);
    EXPECT_TRUE(requirement.conditional_branch_path.empty());
    EXPECT_EQ(countForwardValueRequirements(backward, tanh_node, ForwardValueRequirementKind::NodeOutput), 1u);
    EXPECT_EQ(countInputNodesNamed(backward.outputs, requirement.backward_input_name), 1u)
        << "Repeated forwardValue() calls for one forward node must bind one synthetic backward input.";
#ifdef THOR_DEBUG
    EXPECT_EQ(countNodesWithProvenance(backward.outputs, ExprOp::TANH, ExpressionExecutionProvenance::Forward), 0u);
    EXPECT_EQ(countNodesWithProvenance(backward.outputs, ExprOp::MATMUL, ExpressionExecutionProvenance::Forward), 0u)
        << "Retained computed primals must not be replayed inside backward.";
#endif
}

// PERSISTENT-DAG / SAVED-FORWARD INVARIANT:
// The MUL VJP requests the opposite operand for each input. When both operands
// are the exact same authored projection, those two operand-role requests name
// one forward node and therefore must publish one retained NodeOutput.
TEST(ExpressionPersistentDagInvariant, RepeatedForwardValueRequestsWithinOneVjpReuseOneBinding) {
    const Expression projection = fp32Projection();
    const Expression y = projection * projection;

    PhysicalOutputs forward = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP32, DataType::FP32});
    ASSERT_TRUE(forward.expr);
    ASSERT_EQ(forward.outputs.size(), 1u);
    const uint32_t mul_node = forward.outputs.front().node_idx;
    ASSERT_LT(mul_node, forward.expr->nodes.size());
    ASSERT_EQ(forward.expr->nodes[mul_node].op, ExprOp::MUL);
    ASSERT_EQ(forward.expr->nodes[mul_node].lhs, forward.expr->nodes[mul_node].rhs)
        << "This gate requires the same projection node in both MUL operand roles.";
    const uint32_t projection_node = forward.expr->nodes[mul_node].lhs;
    ASSERT_LT(projection_node, forward.expr->nodes.size());
    ASSERT_TRUE(forward.expr->nodes[projection_node].op == ExprOp::MATMUL ||
                forward.expr->nodes[projection_node].op == ExprOp::GEMM);

    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x", "w"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {2, 3}},
            {"w", {3, 4}},
        });

    expectUniqueForwardValueRequirementIdentities(backward);
    ASSERT_EQ(backward.forward_value_requirements.size(), 1u);
    const ForwardValueRequirement& requirement = backward.forward_value_requirements.front();
    EXPECT_EQ(requirement.forward_node_index, projection_node);
    EXPECT_EQ(requirement.kind, ForwardValueRequirementKind::NodeOutput);
    EXPECT_EQ(countForwardValueRequirements(backward, projection_node, ForwardValueRequirementKind::NodeOutput), 1u);
    EXPECT_EQ(countInputNodesNamed(backward.outputs, requirement.backward_input_name), 1u);
}

// PERSISTENT-DAG / SAVED-FORWARD INVARIANT:
// A training MATMUL+GELU producer may legitimately expose two different retained
// artifacts: its post-GELU NodeOutput and its backend preactivation auxiliary.
// Fan-out must not multiply either requirement, and the two kinds must not be
// incorrectly collapsed merely because they belong to the same forward node.
TEST(ExpressionPersistentDagInvariant, FusedGeluFanoutRetainsEachForwardRequirementKindOnce) {
    const Expression projection = fp32Projection();
    const Expression y = projection.sin() + projection.cos();

    PhysicalOutputs forward = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP32, DataType::FP32});
    ASSERT_TRUE(forward.expr);
    const uint32_t projection_node = findSingleNodeOfKind(forward, ExprOp::MATMUL);
    ASSERT_NE(projection_node, UINT32_MAX);
    ExprNode& fused_projection = forward.expr->nodes.at(projection_node);
    fused_projection.matmul_epilogue = MatmulEpilogue::Gelu;
    fused_projection.matmul_forward_epilogue_aux = true;

    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x", "w"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {2, 3}},
            {"w", {3, 4}},
        });

    expectUniqueForwardValueRequirementIdentities(backward);
    ASSERT_EQ(backward.forward_value_requirements.size(), 2u)
        << "One fused producer may require NodeOutput and MatmulEpilogueAux, but fan-out must not duplicate either kind.";
    EXPECT_EQ(countForwardValueRequirements(backward, projection_node, ForwardValueRequirementKind::NodeOutput), 1u);
    EXPECT_EQ(countForwardValueRequirements(backward, projection_node, ForwardValueRequirementKind::MatmulEpilogueAux), 1u);

    const auto node_output_it = std::find_if(
        backward.forward_value_requirements.begin(),
        backward.forward_value_requirements.end(),
        [&](const ForwardValueRequirement& requirement) {
            return requirement.forward_node_index == projection_node &&
                   requirement.kind == ForwardValueRequirementKind::NodeOutput;
        });
    const auto aux_it = std::find_if(
        backward.forward_value_requirements.begin(),
        backward.forward_value_requirements.end(),
        [&](const ForwardValueRequirement& requirement) {
            return requirement.forward_node_index == projection_node &&
                   requirement.kind == ForwardValueRequirementKind::MatmulEpilogueAux;
        });
    ASSERT_NE(node_output_it, backward.forward_value_requirements.end());
    ASSERT_NE(aux_it, backward.forward_value_requirements.end());
    EXPECT_NE(node_output_it->backward_input_name, aux_it->backward_input_name)
        << "Different retained artifact kinds for one producer require distinct backward bindings.";
    EXPECT_EQ(countInputNodesNamed(backward.outputs, node_output_it->backward_input_name), 1u);
    EXPECT_EQ(countInputNodesNamed(backward.outputs, aux_it->backward_input_name), 1u);
#ifdef THOR_DEBUG
    EXPECT_EQ(countNodesWithProvenance(backward.outputs, ExprOp::MATMUL, ExpressionExecutionProvenance::Forward), 0u)
        << "GELU backward must consume the retained producer artifacts rather than replaying the forward MATMUL.";
#endif
}

// PERSISTENT-DAG / AUTODIFF BINDING INVARIANT:
// External tensor identity is a runtime binding contract, not executing-producer
// identity. Separately authored INPUT nodes may therefore remain distinct nodes
// while sharing one input_slot, and their reverse contributions must be summed
// into one public gradient for that external binding.
TEST(ExpressionPersistentDagInvariant, SameTensorBindingAcrossDistinctInputNodesAggregatesOnePublicGradient) {
    const Expression x_a = Expression::input("shared_binding_x", DataType::FP32, DataType::FP32);
    const Expression x_b = Expression::input("shared_binding_x", DataType::FP32, DataType::FP32);
    const Expression y = x_a + x_b * Expression::constantScalar(2.0);

    PhysicalOutputs forward = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP32});
    ASSERT_TRUE(forward.expr);
    ASSERT_EQ(forward.expr->inputs.size(), 1u)
        << "Two authored INPUT nodes for one external name must share one binding slot.";
    ASSERT_EQ(forward.expr->inputs.front().name, "shared_binding_x");
    ASSERT_EQ(forward.expr->inputs.front().kind, NamedInput::Kind::Tensor);

    std::vector<uint32_t> bound_input_nodes;
    for (uint32_t node_idx = 0; node_idx < forward.expr->nodes.size(); ++node_idx) {
        const ExprNode& node = forward.expr->nodes[node_idx];
        if (node.op == ExprOp::INPUT && node.input_slot == forward.expr->inputs.front().slot) {
            bound_input_nodes.push_back(node_idx);
        }
    }
    ASSERT_EQ(bound_input_nodes.size(), 2u);
    EXPECT_NE(bound_input_nodes[0], bound_input_nodes[1])
        << "External binding equality must not collapse separately authored INPUT node identities.";
    EXPECT_EQ(forward.expr->nodes.at(bound_input_nodes[0]).input_slot,
              forward.expr->nodes.at(bound_input_nodes[1]).input_slot);

    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"shared_binding_x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"shared_binding_x", {2, 3}}});

    ASSERT_TRUE(backward.outputs.expr);
    ASSERT_EQ(backward.outputs.outputs.size(), 1u);
    EXPECT_EQ(backward.outputs.outputs.front().name, "shared_binding_x_grad");
    const ExprNode& public_grad =
        backward.outputs.expr->nodes.at(backward.outputs.outputs.front().node_idx);
    EXPECT_EQ(public_grad.op, ExprOp::ADD)
        << "The public binding gradient must add contributions from both distinct INPUT nodes.";
}

// PERSISTENT-DAG / AUTODIFF IDENTITY INVARIANT:
// Slot-based aggregation is specific to external INPUT bindings. Two separately
// authored executing MATMUL producers remain two producer identities and each
// must execute its own dInput/dWeight VJP pair when both participate in the graph.
TEST(ExpressionPersistentDagInvariant, DistinctExecutingProducersRemainDistinctFromInputBindingAggregation) {
    const Expression x = Expression::input("distinct_producer_x", DataType::FP32, DataType::FP32);
    const Expression w = Expression::input("distinct_producer_w", DataType::FP32, DataType::FP32);
    const Expression projection_a = Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32);
    const Expression projection_b = Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32);
    const Expression y = projection_a + projection_b;

    PhysicalOutputs forward = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP32, DataType::FP32});
    ASSERT_TRUE(forward.expr);
    EXPECT_EQ(countLinearBoundaryNodes(forward), 2u)
        << "The control requires two separately authored executing producers.";

    const ExprNode& root = forward.expr->nodes.at(forward.outputs.front().node_idx);
    ASSERT_EQ(root.op, ExprOp::ADD);
    ASSERT_NE(root.lhs, root.rhs);
    ASSERT_TRUE(forward.expr->nodes.at(root.lhs).op == ExprOp::MATMUL ||
                forward.expr->nodes.at(root.lhs).op == ExprOp::GEMM);
    ASSERT_TRUE(forward.expr->nodes.at(root.rhs).op == ExprOp::MATMUL ||
                forward.expr->nodes.at(root.rhs).op == ExprOp::GEMM);

    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"distinct_producer_x", "distinct_producer_w"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"distinct_producer_x", {2, 3}},
            {"distinct_producer_w", {3, 4}},
        });

    EXPECT_EQ(countLinearBoundaryNodes(backward.outputs), 4u)
        << "Two executing producer nodes require two dInput/dWeight VJP pairs; producer identity is node-based.";
}

// PERSISTENT-DAG / EXTERNAL-BINDING INVARIANT:
// Runtime scalars follow the same external slot-binding model during forward
// lowering: separately authored handles with one name share a slot while keeping
// distinct node identity. They are intentionally non-differentiable; this test
// documents binding behavior without inventing scalar gradients.
TEST(ExpressionPersistentDagInvariant, RuntimeScalarBindingsShareSlotsButRemainNonDifferentiable) {
    const Expression host_a = Expression::runtimeScalar("shared_host_scale", DataType::FP32, DataType::FP32);
    const Expression host_b = Expression::runtimeScalar("shared_host_scale", DataType::FP32, DataType::FP32);
    const Expression tensor_a = Expression::tensorRuntimeScalar("shared_tensor_scale", DataType::FP32, DataType::FP32);
    const Expression tensor_b = Expression::tensorRuntimeScalar("shared_tensor_scale", DataType::FP32, DataType::FP32);

    const PhysicalOutputs forward = Expression::outputs({
        {"host_a", host_a},
        {"host_b", host_b},
        {"tensor_a", tensor_a},
        {"tensor_b", tensor_b},
    }).physicalOutputs();
    ASSERT_TRUE(forward.expr);
    ASSERT_EQ(forward.outputs.size(), 4u);
    ASSERT_EQ(forward.expr->inputs.size(), 2u);

    const uint32_t host_node_a = forward.outputs[0].node_idx;
    const uint32_t host_node_b = forward.outputs[1].node_idx;
    const uint32_t tensor_node_a = forward.outputs[2].node_idx;
    const uint32_t tensor_node_b = forward.outputs[3].node_idx;
    ASSERT_NE(host_node_a, host_node_b);
    ASSERT_NE(tensor_node_a, tensor_node_b);
    ASSERT_EQ(forward.expr->nodes.at(host_node_a).op, ExprOp::RUNTIME_SCALAR);
    ASSERT_EQ(forward.expr->nodes.at(host_node_b).op, ExprOp::RUNTIME_SCALAR);
    ASSERT_EQ(forward.expr->nodes.at(tensor_node_a).op, ExprOp::TENSOR_RUNTIME_SCALAR);
    ASSERT_EQ(forward.expr->nodes.at(tensor_node_b).op, ExprOp::TENSOR_RUNTIME_SCALAR);
    EXPECT_EQ(forward.expr->nodes.at(host_node_a).input_slot,
              forward.expr->nodes.at(host_node_b).input_slot);
    EXPECT_EQ(forward.expr->nodes.at(tensor_node_a).input_slot,
              forward.expr->nodes.at(tensor_node_b).input_slot);
    EXPECT_NE(forward.expr->nodes.at(host_node_a).input_slot,
              forward.expr->nodes.at(tensor_node_a).input_slot);
    EXPECT_EQ(forward.expr->inputs.at(forward.expr->nodes.at(host_node_a).input_slot).kind,
              NamedInput::Kind::RuntimeScalarFp32);
    EXPECT_EQ(forward.expr->inputs.at(forward.expr->nodes.at(tensor_node_a).input_slot).kind,
              NamedInput::Kind::TensorRuntimeScalar);

    const PhysicalOutputs host_only = Expression::outputs({{"y", host_a}}).physicalOutputs();
    EXPECT_THROW(
        buildBackwardOutputsWithForwardValueRequirements(host_only, {"shared_host_scale"}),
        std::runtime_error);
    const PhysicalOutputs tensor_only = Expression::outputs({{"y", tensor_a}}).physicalOutputs();
    EXPECT_THROW(
        buildBackwardOutputsWithForwardValueRequirements(tensor_only, {"shared_tensor_scale"}),
        std::runtime_error);
}

// PERSISTENT-DAG INVARIANT:
// One authored projection reused by two logical consumers already lowers to one
// physical MATMUL node. EquationCompiler therefore needs only one runtime value
// here without invoking any structural equivalence between distinct nodes.
//
// Genuine physical-node sharing remains one planner runtime value; structural
// equality between different physical node ids never implies runtime identity.
TEST(ExpressionPersistentDagInvariant, ForwardTopologyAlreadyContainsOneSharedProducer) {
    const Expression projection = fp32Projection();
    const Expression y = projection.sin() + projection.cos();
    PhysicalOutputs outputs = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP32, DataType::FP32});

    ASSERT_EQ(countLinearBoundaryNodes(outputs), 1u);
    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(outputs);
    EXPECT_EQ(countStagesOfKind(stages, PhysicalExecutionStage::Kind::Matmul), 1u)
        << "One shared physical producer should naturally require one MATMUL runtime stage.";
}

// PERSISTENT-DAG INVARIANT:
// Distinct physical terminal roots retain distinct runtime value identities even
// when their fused-region structure is identical. Dependency-overlap grouping is
// independent of value identity, so the two outputs may still execute in one fused
// stage as long as that stage exposes both authored values separately.
TEST(ExpressionPersistentDagInvariant, FusedRegionDistinctTerminalRootsUseDistinctRuntimeValues) {
    const Expression x = Expression::input("terminal_x", DataType::FP32, DataType::FP32);
    const Expression a = x.sin().exp();
    const Expression b = x.sin().exp();

    PhysicalOutputs physical = Expression::outputs({{"a", a}, {"b", b}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32});
    ASSERT_TRUE(physical.expr);
    ASSERT_EQ(physical.outputs.size(), 2u);
    EXPECT_NE(physical.outputs[0].node_idx, physical.outputs[1].node_idx);
    EXPECT_EQ(countNodesOfKind(physical, ExprOp::SIN), 2u);
    EXPECT_EQ(countNodesOfKind(physical, ExprOp::EXP), 2u);

    detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(physical);
    const std::optional<uint32_t> a_value = finalValueIdNamed(plan.final_outputs, "a");
    const std::optional<uint32_t> b_value = finalValueIdNamed(plan.final_outputs, "b");
    ASSERT_TRUE(a_value.has_value());
    ASSERT_TRUE(b_value.has_value());
    EXPECT_NE(*a_value, *b_value)
        << "Distinct terminal physical roots must not alias through structural fused-region equality.";

    ASSERT_EQ(countStagesOfKind(plan.stages, PhysicalExecutionStage::Kind::FusedKernel), 1u)
        << "Dependency-overlap grouping should remain available independently of runtime value identity.";
    EXPECT_EQ(countStageOutputsWithValueId(plan.stages, *a_value), 1u);
    EXPECT_EQ(countStageOutputsWithValueId(plan.stages, *b_value), 1u);

    const auto fused_stage_it = std::find_if(plan.stages.begin(), plan.stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::FusedKernel;
    });
    ASSERT_NE(fused_stage_it, plan.stages.end());
    EXPECT_EQ(fused_stage_it->outputs.size(), 2u)
        << "One execution group may carry both structurally equal authored terminal values.";
}

// PERSISTENT-DAG INVARIANT:
// Two names that intentionally reference the same terminal fused root are aliases of
// one authored physical value. Removing exact-signature runtime CSE must retain this
// ordinary same-node reuse.
TEST(ExpressionPersistentDagInvariant, SameTerminalFusedRootReusesOneRuntimeValue) {
    const Expression x = Expression::input("terminal_shared_x", DataType::FP32, DataType::FP32);
    const Expression shared = x.sin().exp();

    PhysicalOutputs physical = Expression::outputs({{"a", shared}, {"b", shared}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32});
    ASSERT_TRUE(physical.expr);
    ASSERT_EQ(physical.outputs.size(), 2u);
    ASSERT_EQ(physical.outputs[0].node_idx, physical.outputs[1].node_idx);

    detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(physical);
    const std::optional<uint32_t> a_value = finalValueIdNamed(plan.final_outputs, "a");
    const std::optional<uint32_t> b_value = finalValueIdNamed(plan.final_outputs, "b");
    ASSERT_TRUE(a_value.has_value());
    ASSERT_TRUE(b_value.has_value());
    EXPECT_EQ(*a_value, *b_value);
    ASSERT_EQ(countStagesOfKind(plan.stages, PhysicalExecutionStage::Kind::FusedKernel), 1u);
    EXPECT_EQ(countStageOutputsWithValueId(plan.stages, *a_value), 1u);
}

// PERSISTENT-DAG INVARIANT:
// Pending terminal scheduling uses physical root identity. A consumer of the exact
// pending root must materialize and consume that pending value, while a different
// physical root with identical structure must receive and consume a different value.
TEST(ExpressionPersistentDagInvariant, PendingTerminalLookupUsesPhysicalRootIdentity) {
    {
        const Expression x = Expression::input("pending_same_x", DataType::FP32, DataType::FP32);
        const Expression pending = x.sin().exp();
        const Expression consumer = pending.reduce_sum({0}, {}, DataType::FP32);

        PhysicalOutputs physical = Expression::outputs({{"pending", pending}, {"consumer", consumer}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::FP32});
        detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(physical);

        const std::optional<uint32_t> pending_value = finalValueIdNamed(plan.final_outputs, "pending");
        ASSERT_TRUE(pending_value.has_value());
        const auto reduction_it = std::find_if(plan.stages.begin(), plan.stages.end(), [](const PhysicalExecutionStage& stage) {
            return stage.kind == PhysicalExecutionStage::Kind::Reduction;
        });
        ASSERT_NE(reduction_it, plan.stages.end());
        ASSERT_EQ(reduction_it->input_value_ids.size(), 1u);
        EXPECT_EQ(reduction_it->input_value_ids[0], *pending_value)
            << "A consumer of the exact pending physical root must use that root's pending runtime value.";
        EXPECT_EQ(countStageOutputsWithValueId(plan.stages, *pending_value), 1u);
    }

    {
        const Expression x = Expression::input("pending_distinct_x", DataType::FP32, DataType::FP32);
        const Expression pending = x.sin().exp();
        const Expression distinct_equal = x.sin().exp();
        const Expression consumer = distinct_equal.reduce_sum({0}, {}, DataType::FP32);

        PhysicalOutputs physical = Expression::outputs({{"pending", pending}, {"consumer", consumer}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::FP32});
        ASSERT_TRUE(physical.expr);
        ASSERT_EQ(physical.outputs.size(), 2u);
        EXPECT_EQ(countNodesOfKind(physical, ExprOp::EXP), 2u);

        detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(physical);
        const std::optional<uint32_t> pending_value = finalValueIdNamed(plan.final_outputs, "pending");
        ASSERT_TRUE(pending_value.has_value());
        const auto reduction_it = std::find_if(plan.stages.begin(), plan.stages.end(), [](const PhysicalExecutionStage& stage) {
            return stage.kind == PhysicalExecutionStage::Kind::Reduction;
        });
        ASSERT_NE(reduction_it, plan.stages.end());
        ASSERT_EQ(reduction_it->input_value_ids.size(), 1u);
        const uint32_t distinct_value = reduction_it->input_value_ids[0];
        EXPECT_NE(distinct_value, *pending_value)
            << "A structurally equal but different pending root must not alias the already-pending terminal value.";
        EXPECT_EQ(countStageOutputsWithValueId(plan.stages, *pending_value), 1u);
        EXPECT_EQ(countStageOutputsWithValueId(plan.stages, distinct_value), 1u);
    }
}

// PERSISTENT-DAG INVARIANT:
// Independently authored nonterminal fused roots retain distinct runtime value
// identities even when their region structure is identical. Each downstream stage
// boundary must consume the runtime value produced for its own physical input root.
TEST(ExpressionPersistentDagInvariant, FusedRegionDistinctIntermediateRootsUseDistinctRuntimeValues) {
    const Expression x = Expression::input("intermediate_x", DataType::FP32, DataType::FP32);
    const Expression producer_a = x.sin().exp();
    const Expression producer_b = x.sin().exp();
    const Expression a = producer_a.reduce_sum({0}, {}, DataType::FP32);
    const Expression b = producer_b.reduce_sum({1}, {}, DataType::FP32);

    PhysicalOutputs physical = Expression::outputs({{"a", a}, {"b", b}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32});
    ASSERT_TRUE(physical.expr);
    EXPECT_EQ(countNodesOfKind(physical, ExprOp::SIN), 2u);
    EXPECT_EQ(countNodesOfKind(physical, ExprOp::EXP), 2u);
    EXPECT_EQ(countNodesOfKind(physical, ExprOp::REDUCE_SUM), 2u);

    detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(physical);
    ASSERT_EQ(countStagesOfKind(plan.stages, PhysicalExecutionStage::Kind::Reduction), 2u)
        << "The different reduction axes keep the downstream stage-boundary roots distinct.";

    std::vector<const PhysicalExecutionStage*> reductions;
    for (const PhysicalExecutionStage& stage : plan.stages) {
        if (stage.kind == PhysicalExecutionStage::Kind::Reduction) {
            reductions.push_back(&stage);
        }
    }
    ASSERT_EQ(reductions.size(), 2u);
    ASSERT_EQ(reductions[0]->input_value_ids.size(), 1u);
    ASSERT_EQ(reductions[1]->input_value_ids.size(), 1u);
    const uint32_t producer_a_value = reductions[0]->input_value_ids[0];
    const uint32_t producer_b_value = reductions[1]->input_value_ids[0];
    EXPECT_NE(producer_a_value, producer_b_value)
        << "Distinct intermediate physical roots must retain distinct runtime value identities.";
    EXPECT_EQ(countStageOutputsWithValueId(plan.stages, producer_a_value), 1u);
    EXPECT_EQ(countStageOutputsWithValueId(plan.stages, producer_b_value), 1u);

    const std::optional<uint32_t> a_value = finalValueIdNamed(plan.final_outputs, "a");
    const std::optional<uint32_t> b_value = finalValueIdNamed(plan.final_outputs, "b");
    ASSERT_TRUE(a_value.has_value());
    ASSERT_TRUE(b_value.has_value());
    EXPECT_NE(*a_value, *b_value);
}

// PERSISTENT-DAG INVARIANT:
// Terminal-region grouping is an execution-fusion decision, not value identity. Two
// different terminal regions that share their external dependency may execute in one
// fused stage while retaining different runtime value_ids and different stage outputs.
// Execution grouping must remain independent from runtime value identity.
TEST(ExpressionPersistentDagInvariant, DistinctTerminalRegionsCanShareOneFusedStageWithoutAliasingValues) {
    const Expression x = Expression::input("terminal_group_x", DataType::FP32, DataType::FP32);
    const Expression a = x.sin();
    const Expression b = x.cos();

    PhysicalOutputs physical = Expression::outputs({{"a", a}, {"b", b}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32});
    ASSERT_TRUE(physical.expr);
    ASSERT_EQ(physical.outputs.size(), 2u);
    EXPECT_NE(physical.outputs[0].node_idx, physical.outputs[1].node_idx);

    detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(physical);
    const std::optional<uint32_t> a_value = finalValueIdNamed(plan.final_outputs, "a");
    const std::optional<uint32_t> b_value = finalValueIdNamed(plan.final_outputs, "b");
    ASSERT_TRUE(a_value.has_value());
    ASSERT_TRUE(b_value.has_value());
    EXPECT_NE(*a_value, *b_value);
    ASSERT_EQ(countStagesOfKind(plan.stages, PhysicalExecutionStage::Kind::FusedKernel), 1u);
    EXPECT_EQ(countStageOutputsWithValueId(plan.stages, *a_value), 1u);
    EXPECT_EQ(countStageOutputsWithValueId(plan.stages, *b_value), 1u);

    const auto fused_stage_it = std::find_if(plan.stages.begin(), plan.stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::FusedKernel;
    });
    ASSERT_NE(fused_stage_it, plan.stages.end());
    EXPECT_EQ(fused_stage_it->outputs.size(), 2u)
        << "Different terminal roots may be grouped by shared dependencies while retaining separate outputs.";
}

// PERSISTENT-DAG INVARIANT:
// Stage-local structural deduplication happens after planner runtime value ids have
// already been assigned. Two distinct terminal roots can therefore retain distinct
// runtime identities while independently authored equivalent internal arithmetic is
// represented once inside the generated fused stage. This is code-generation-local
// CSE rather than planner value-identity aliasing.
//
// This optimization is valid only while distinct requested outputs keep distinct
// value_ids/local output roots and StageNodeKey fully captures every semantic
// attribute of an operation it is allowed to merge.
TEST(ExpressionPersistentDagInvariant, StageLocalCsePreservesDistinctRuntimeValuesAndOutputs) {
    const Expression x = Expression::input("stage_local_x", DataType::FP32, DataType::FP32);
    const Expression duplicate_a = x.sin();
    const Expression duplicate_b = x.sin();
    const Expression a = duplicate_a.exp();
    const Expression b = duplicate_b.cos();

    PhysicalOutputs physical = Expression::outputs({{"a", a}, {"b", b}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32});
    ASSERT_TRUE(physical.expr);
    EXPECT_EQ(countNodesOfKind(physical, ExprOp::SIN), 2u)
        << "Lowering must preserve the two independently authored internal SIN nodes.";

    detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(physical);
    const std::optional<uint32_t> a_value = finalValueIdNamed(plan.final_outputs, "a");
    const std::optional<uint32_t> b_value = finalValueIdNamed(plan.final_outputs, "b");
    ASSERT_TRUE(a_value.has_value());
    ASSERT_TRUE(b_value.has_value());
    EXPECT_NE(*a_value, *b_value)
        << "Stage-local CSE must not redefine planner runtime value identity.";

    std::vector<const PhysicalExecutionStage*> fused_stages;
    for (const PhysicalExecutionStage& stage : plan.stages) {
        if (stage.kind == PhysicalExecutionStage::Kind::FusedKernel) {
            fused_stages.push_back(&stage);
        }
    }
    ASSERT_EQ(fused_stages.size(), 1u)
        << "The two different terminal wrappers should remain eligible for one fused execution group.";
    const PhysicalExecutionStage& stage = *fused_stages.front();
    ASSERT_EQ(stage.input_value_ids.size(), 1u);
    EXPECT_EQ(stage.input_value_ids.front(), 0u)
        << "Stage-local CSE must not manufacture or alias external input value identities.";
    ASSERT_EQ(stage.outputs.size(), 2u);
    EXPECT_NE(stage.outputs[0].value_id, stage.outputs[1].value_id);
    EXPECT_NE(stage.outputs[0].local_node_idx, stage.outputs[1].local_node_idx)
        << "Distinct terminal roots must remain distinct local stage outputs.";
    EXPECT_EQ(stage.outputs[0].materialized_layout, MaterializedTensorLayout::RowMajor);
    EXPECT_EQ(stage.outputs[1].materialized_layout, MaterializedTensorLayout::RowMajor);

    size_t stage_sin_count = 0;
    for (const ExprNode& node : stage.expr.nodes) {
        stage_sin_count += node.op == ExprOp::SIN ? 1u : 0u;
    }
    EXPECT_EQ(stage_sin_count, 1u)
        << "Audited pure equivalent SIN nodes remain eligible for stage-local implementation CSE.";

    std::optional<uint32_t> exp_parent;
    std::optional<uint32_t> cos_parent;
    for (const CompiledStageOutput& output : stage.outputs) {
        ASSERT_LT(output.local_node_idx, stage.expr.nodes.size());
        const ExprNode& root = stage.expr.nodes[output.local_node_idx];
        if (root.op == ExprOp::EXP) {
            exp_parent = root.lhs;
        } else if (root.op == ExprOp::COS) {
            cos_parent = root.lhs;
        }
    }
    ASSERT_TRUE(exp_parent.has_value());
    ASSERT_TRUE(cos_parent.has_value());
    EXPECT_EQ(*exp_parent, *cos_parent)
        << "Equivalent internal arithmetic is represented by one stage-local node after deduplication.";
    ASSERT_LT(*exp_parent, stage.expr.nodes.size());
    EXPECT_EQ(stage.expr.nodes[*exp_parent].op, ExprOp::SIN);
}

// PERSISTENT-DAG INVARIANT:
// Stage-local CSE is allowed to merge only semantics-complete equivalent operations.
// RESHAPE target dimensions are part of operation semantics, so two reshapes of the
// same parent to different shapes must remain distinct stage-local nodes.
TEST(ExpressionPersistentDagInvariant, StageLocalCseRespectsReshapeDimensions) {
    const Expression x = Expression::input("stage_local_reshape_x", DataType::FP32, DataType::FP32);
    const Expression reshape_a = x.reshape({2, 6});
    const Expression reshape_b = x.reshape({3, 4});
    const Expression a = reshape_a.sin();
    const Expression b = reshape_b.cos();

    PhysicalOutputs physical = Expression::outputs({{"a", a}, {"b", b}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32});
    ASSERT_TRUE(physical.expr);
    EXPECT_EQ(countNodesOfKind(physical, ExprOp::RESHAPE), 2u);

    std::vector<std::vector<uint64_t>> authored_reshape_dims;
    for (const ExprNode& node : physical.expr->nodes) {
        if (node.op == ExprOp::RESHAPE) {
            authored_reshape_dims.push_back(node.reshape_dims);
        }
    }
    ASSERT_EQ(authored_reshape_dims.size(), 2u);
    EXPECT_NE(authored_reshape_dims[0], authored_reshape_dims[1]);

    detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(physical);
    const std::optional<uint32_t> a_value = finalValueIdNamed(plan.final_outputs, "a");
    const std::optional<uint32_t> b_value = finalValueIdNamed(plan.final_outputs, "b");
    ASSERT_TRUE(a_value.has_value());
    ASSERT_TRUE(b_value.has_value());
    EXPECT_NE(*a_value, *b_value)
        << "The planner still preserves the distinct terminal runtime values in this isolation case.";

    std::vector<const PhysicalExecutionStage*> fused_stages;
    for (const PhysicalExecutionStage& stage : plan.stages) {
        if (stage.kind == PhysicalExecutionStage::Kind::FusedKernel) {
            fused_stages.push_back(&stage);
        }
    }
    ASSERT_EQ(fused_stages.size(), 1u);
    const PhysicalExecutionStage& stage = *fused_stages.front();
    ASSERT_EQ(stage.outputs.size(), 2u);
    EXPECT_NE(stage.outputs[0].value_id, stage.outputs[1].value_id);
    EXPECT_NE(stage.outputs[0].local_node_idx, stage.outputs[1].local_node_idx);

    std::vector<uint32_t> stage_reshape_nodes;
    for (uint32_t i = 0; i < stage.expr.nodes.size(); ++i) {
        if (stage.expr.nodes[i].op == ExprOp::RESHAPE) {
            stage_reshape_nodes.push_back(i);
        }
    }
    ASSERT_EQ(stage_reshape_nodes.size(), 2u)
        << "RESHAPE target dimensions must participate in stage-local CSE equality.";

    std::optional<uint32_t> sin_parent;
    std::optional<uint32_t> cos_parent;
    for (const CompiledStageOutput& output : stage.outputs) {
        ASSERT_LT(output.local_node_idx, stage.expr.nodes.size());
        const ExprNode& root = stage.expr.nodes[output.local_node_idx];
        if (root.op == ExprOp::SIN) {
            sin_parent = root.lhs;
        } else if (root.op == ExprOp::COS) {
            cos_parent = root.lhs;
        }
    }
    ASSERT_TRUE(sin_parent.has_value());
    ASSERT_TRUE(cos_parent.has_value());
    EXPECT_NE(*sin_parent, *cos_parent)
        << "Different RESHAPE semantics must remain different stage-local parents.";

    ASSERT_LT(*sin_parent, stage.expr.nodes.size());
    ASSERT_LT(*cos_parent, stage.expr.nodes.size());
    ASSERT_EQ(stage.expr.nodes[*sin_parent].op, ExprOp::RESHAPE);
    ASSERT_EQ(stage.expr.nodes[*cos_parent].op, ExprOp::RESHAPE);
    EXPECT_NE(stage.expr.nodes[*sin_parent].reshape_dims, stage.expr.nodes[*cos_parent].reshape_dims);

    std::vector<std::vector<uint64_t>> stage_reshape_dims{
        stage.expr.nodes[stage_reshape_nodes[0]].reshape_dims,
        stage.expr.nodes[stage_reshape_nodes[1]].reshape_dims,
    };
    std::sort(stage_reshape_dims.begin(), stage_reshape_dims.end());
    std::sort(authored_reshape_dims.begin(), authored_reshape_dims.end());
    EXPECT_EQ(stage_reshape_dims, authored_reshape_dims);
}

// PERSISTENT-DAG INVARIANT:
// STRIDED_VIEW_BACKWARD source/fill dimensions are part of its scatter semantics.
// Identical view dimensions/strides/offsets do not make two backward views equal
// when their source shapes differ.
TEST(ExpressionPersistentDagInvariant, StageLocalCseRespectsStridedViewBackwardSourceDimensions) {
    const Expression x = Expression::input("stage_local_svbw_x", DataType::FP32, DataType::FP32);
    const Expression scatter_a = x.stridedViewBackward({2, 6}, {2, 3}, {3, 1}, 0);
    const Expression scatter_b = x.stridedViewBackward({3, 4}, {2, 3}, {3, 1}, 0);
    const Expression a = scatter_a.sin();
    const Expression b = scatter_b.cos();

    PhysicalOutputs physical = Expression::outputs({{"a", a}, {"b", b}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32});
    ASSERT_TRUE(physical.expr);
    EXPECT_EQ(countNodesOfKind(physical, ExprOp::STRIDED_VIEW_BACKWARD), 2u);

    std::vector<std::vector<uint64_t>> authored_source_dims;
    for (const ExprNode& node : physical.expr->nodes) {
        if (node.op == ExprOp::STRIDED_VIEW_BACKWARD) {
            authored_source_dims.push_back(node.fill_dims);
        }
    }
    ASSERT_EQ(authored_source_dims.size(), 2u);
    EXPECT_NE(authored_source_dims[0], authored_source_dims[1]);

    detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(physical);
    const std::optional<uint32_t> a_value = finalValueIdNamed(plan.final_outputs, "a");
    const std::optional<uint32_t> b_value = finalValueIdNamed(plan.final_outputs, "b");
    ASSERT_TRUE(a_value.has_value());
    ASSERT_TRUE(b_value.has_value());
    EXPECT_NE(*a_value, *b_value);

    const PhysicalExecutionStage* fused_stage = nullptr;
    for (const PhysicalExecutionStage& stage : plan.stages) {
        if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel) {
            continue;
        }
        ASSERT_EQ(fused_stage, nullptr);
        fused_stage = &stage;
    }
    ASSERT_NE(fused_stage, nullptr);
    ASSERT_EQ(fused_stage->outputs.size(), 2u);
    EXPECT_NE(fused_stage->outputs[0].value_id, fused_stage->outputs[1].value_id);
    EXPECT_NE(fused_stage->outputs[0].local_node_idx, fused_stage->outputs[1].local_node_idx);

    std::vector<uint32_t> stage_scatter_nodes;
    for (uint32_t i = 0; i < fused_stage->expr.nodes.size(); ++i) {
        if (fused_stage->expr.nodes[i].op == ExprOp::STRIDED_VIEW_BACKWARD) {
            stage_scatter_nodes.push_back(i);
        }
    }
    ASSERT_EQ(stage_scatter_nodes.size(), 2u)
        << "STRIDED_VIEW_BACKWARD source dimensions must participate in stage-local CSE equality.";

    std::optional<uint32_t> sin_parent;
    std::optional<uint32_t> cos_parent;
    for (const CompiledStageOutput& output : fused_stage->outputs) {
        ASSERT_LT(output.local_node_idx, fused_stage->expr.nodes.size());
        const ExprNode& root = fused_stage->expr.nodes[output.local_node_idx];
        if (root.op == ExprOp::SIN) {
            sin_parent = root.lhs;
        } else if (root.op == ExprOp::COS) {
            cos_parent = root.lhs;
        }
    }
    ASSERT_TRUE(sin_parent.has_value());
    ASSERT_TRUE(cos_parent.has_value());
    EXPECT_NE(*sin_parent, *cos_parent)
        << "Different STRIDED_VIEW_BACKWARD source shapes must remain different stage-local parents.";

    ASSERT_LT(*sin_parent, fused_stage->expr.nodes.size());
    ASSERT_LT(*cos_parent, fused_stage->expr.nodes.size());
    ASSERT_EQ(fused_stage->expr.nodes[*sin_parent].op, ExprOp::STRIDED_VIEW_BACKWARD);
    ASSERT_EQ(fused_stage->expr.nodes[*cos_parent].op, ExprOp::STRIDED_VIEW_BACKWARD);
    EXPECT_NE(fused_stage->expr.nodes[*sin_parent].fill_dims, fused_stage->expr.nodes[*cos_parent].fill_dims);

    std::vector<std::vector<uint64_t>> stage_source_dims{
        fused_stage->expr.nodes[stage_scatter_nodes[0]].fill_dims,
        fused_stage->expr.nodes[stage_scatter_nodes[1]].fill_dims,
    };
    std::sort(stage_source_dims.begin(), stage_source_dims.end());
    std::sort(authored_source_dims.begin(), authored_source_dims.end());
    EXPECT_EQ(stage_source_dims, authored_source_dims);
}

// PERSISTENT-DAG INVARIANT:
// Independently authored stage-boundary operations retain distinct runtime values
// even when op, inputs, dtype annotations, and all operation attributes are
// structurally identical. Structural signatures may still drive implementation
// caching or execution grouping, but physical node identity is authoritative for
// runtime value identity.
TEST(ExpressionPersistentDagInvariant, StageBoundaryDistinctAuthoredRootsUseDistinctRuntimeValues) {
    {
        SCOPED_TRACE("MATMUL stage boundary");
        const Expression x = Expression::input("matmul_x", DataType::FP32, DataType::FP32);
        const Expression w = Expression::input("matmul_w", DataType::FP32, DataType::FP32);
        const Expression a = Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32);
        const Expression b = Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32);
        expectDistinctStageBoundaryRootsUseDistinctRuntimeValues(
            observePlannerIdentity(a, b, {DataType::FP32, DataType::FP32}),
            ExprOp::MATMUL,
            PhysicalExecutionStage::Kind::Matmul);
    }

    {
        SCOPED_TRACE("reduction stage boundary");
        const Expression x = Expression::input("reduce_x", DataType::FP32, DataType::FP32);
        const Expression a = x.reduce_sum({1}, {}, DataType::FP32);
        const Expression b = x.reduce_sum({1}, {}, DataType::FP32);
        expectDistinctStageBoundaryRootsUseDistinctRuntimeValues(
            observePlannerIdentity(a, b, {DataType::FP32}),
            ExprOp::REDUCE_SUM,
            PhysicalExecutionStage::Kind::Reduction);
    }

    {
        SCOPED_TRACE("softmax stage boundary");
        const Expression x = Expression::input("softmax_x", DataType::FP32, DataType::FP32);
        const Expression a = x.softmax();
        const Expression b = x.softmax();
        expectDistinctStageBoundaryRootsUseDistinctRuntimeValues(
            observePlannerIdentity(a, b, {DataType::FP32}),
            ExprOp::SOFTMAX,
            PhysicalExecutionStage::Kind::Softmax);
    }

    {
        SCOPED_TRACE("RMSNorm stage boundary");
        const Expression x = Expression::input("rms_x", DataType::FP32, DataType::FP32);
        const Expression scale = Expression::input("rms_scale", DataType::FP32, DataType::FP32);
        const Expression a = Expression::rmsNorm(x, scale, 4, 1.0e-5, DataType::FP32, DataType::FP32);
        const Expression b = Expression::rmsNorm(x, scale, 4, 1.0e-5, DataType::FP32, DataType::FP32);
        expectDistinctStageBoundaryRootsUseDistinctRuntimeValues(
            observePlannerIdentity(a, b, {DataType::FP32, DataType::FP32}),
            ExprOp::RMSNORM,
            PhysicalExecutionStage::Kind::RmsNorm);
    }
}

// PERSISTENT-DAG INVARIANT:
// GELU epilogue recognition requires the exact same authored physical source to
// feed both x and normcdf(x). Two independently authored MATMUL nodes remain
// distinct even when every structural attribute and input is equal.
TEST(ExpressionPersistentDagInvariant, MatmulEpilogueMatcherRejectsDistinctAuthoredSources) {
    const Expression x = Expression::input("gelu_match_x", DataType::FP32, DataType::FP32);
    const Expression w = Expression::input("gelu_match_w", DataType::FP32, DataType::FP32);
    const Expression source_a = Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32);
    const Expression source_b = Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32);
    const Expression y = source_a * source_b.normcdf();

    PhysicalOutputs physical = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32, DataType::FP32});
    ASSERT_TRUE(physical.expr);
    ASSERT_EQ(physical.outputs.size(), 1u);
    EXPECT_EQ(countNodesOfKind(physical, ExprOp::MATMUL), 2u);

    const uint32_t root_idx = physical.outputs.front().node_idx;
    ASSERT_LT(root_idx, physical.expr->nodes.size());
    const ExprNode& root = physical.expr->nodes[root_idx];
    ASSERT_EQ(root.op, ExprOp::MUL);
    ASSERT_LT(root.lhs, physical.expr->nodes.size());
    ASSERT_LT(root.rhs, physical.expr->nodes.size());
    ASSERT_EQ(physical.expr->nodes[root.lhs].op, ExprOp::MATMUL);
    ASSERT_EQ(physical.expr->nodes[root.rhs].op, ExprOp::NORMCDF);
    const uint32_t normcdf_source = physical.expr->nodes[root.rhs].lhs;
    ASSERT_LT(normcdf_source, physical.expr->nodes.size());
    ASSERT_EQ(physical.expr->nodes[normcdf_source].op, ExprOp::MATMUL);
    ASSERT_NE(root.lhs, normcdf_source)
        << "The invariant setup requires two separately authored physical MATMUL values.";

    const TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
    const std::unordered_map<std::string, Tensor> inputs = {
        {"gelu_match_x", Tensor(cpu, TensorDescriptor(DataType::FP32, {2, 3}))},
        {"gelu_match_w", Tensor(cpu, TensorDescriptor(DataType::FP32, {3, 4}))},
    };
    const PhysicalOutputs optimized = detail::optimizeGemmPatternsForTests(physical, inputs);
    ASSERT_TRUE(optimized.expr);
    ASSERT_EQ(optimized.outputs.size(), 1u);
    EXPECT_EQ(countNodesOfKind(optimized, ExprOp::MATMUL), 2u);

    const ExprNode& optimized_root = optimized.expr->nodes.at(optimized.outputs.front().node_idx);
    ASSERT_EQ(optimized_root.op, ExprOp::MUL)
        << "Distinct authored MATMUL values must not be substituted for one GELU source.";
    ASSERT_LT(optimized_root.lhs, optimized.expr->nodes.size());
    ASSERT_LT(optimized_root.rhs, optimized.expr->nodes.size());
    ASSERT_EQ(optimized.expr->nodes[optimized_root.lhs].op, ExprOp::MATMUL);
    ASSERT_EQ(optimized.expr->nodes[optimized_root.rhs].op, ExprOp::NORMCDF);
    const uint32_t optimized_normcdf_source = optimized.expr->nodes[optimized_root.rhs].lhs;
    ASSERT_LT(optimized_normcdf_source, optimized.expr->nodes.size());
    ASSERT_EQ(optimized.expr->nodes[optimized_normcdf_source].op, ExprOp::MATMUL);
    EXPECT_NE(optimized_root.lhs, optimized_normcdf_source);
    EXPECT_EQ(optimized.expr->nodes[optimized_root.lhs].matmul_epilogue, MatmulEpilogue::Default);
    EXPECT_EQ(optimized.expr->nodes[optimized_normcdf_source].matmul_epilogue, MatmulEpilogue::Default);
}

// PERSISTENT-DAG INVARIANT:
// GELU epilogue recognition is legitimate only when the exact same authored
// MATMUL node feeds both x and normcdf(x).
TEST(ExpressionPersistentDagInvariant, MatmulEpilogueMatcherAcceptsSameAuthoredSource) {
    const Expression x = Expression::input("gelu_shared_x", DataType::FP32, DataType::FP32);
    const Expression w = Expression::input("gelu_shared_w", DataType::FP32, DataType::FP32);
    const Expression source = Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32);
    const Expression y = source * source.normcdf();

    PhysicalOutputs physical = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32, DataType::FP32});
    ASSERT_TRUE(physical.expr);
    ASSERT_EQ(physical.outputs.size(), 1u);
    EXPECT_EQ(countNodesOfKind(physical, ExprOp::MATMUL), 1u);

    const uint32_t root_idx = physical.outputs.front().node_idx;
    const ExprNode& root = physical.expr->nodes.at(root_idx);
    ASSERT_EQ(root.op, ExprOp::MUL);
    ASSERT_EQ(physical.expr->nodes.at(root.rhs).op, ExprOp::NORMCDF);
    EXPECT_EQ(root.lhs, physical.expr->nodes.at(root.rhs).lhs);

    const TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
    const std::unordered_map<std::string, Tensor> inputs = {
        {"gelu_shared_x", Tensor(cpu, TensorDescriptor(DataType::FP32, {2, 3}))},
        {"gelu_shared_w", Tensor(cpu, TensorDescriptor(DataType::FP32, {3, 4}))},
    };
    const PhysicalOutputs optimized = detail::optimizeGemmPatternsForTests(physical, inputs);
    ASSERT_TRUE(optimized.expr);
    const ExprNode& optimized_root = optimized.expr->nodes.at(optimized.outputs.front().node_idx);
    ASSERT_EQ(optimized_root.op, ExprOp::MATMUL);
    EXPECT_EQ(optimized_root.matmul_epilogue, MatmulEpilogue::Gelu);
}

// PERSISTENT-DAG INVARIANT:
// Two names that intentionally reference the exact same physical stage-boundary
// root are aliases of one authored value, while distinct physical roots remain
// distinct runtime values even when structurally equal.
TEST(ExpressionPersistentDagInvariant, SameStageBoundaryRootReusesOneRuntimeValue) {
    {
        SCOPED_TRACE("MATMUL shared root");
        const Expression x = Expression::input("matmul_x", DataType::FP32, DataType::FP32);
        const Expression w = Expression::input("matmul_w", DataType::FP32, DataType::FP32);
        const Expression shared = Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32);
        expectSharedStageBoundaryRootUsesOneRuntimeValue(
            observePlannerIdentity(shared, shared, {DataType::FP32, DataType::FP32}),
            ExprOp::MATMUL,
            PhysicalExecutionStage::Kind::Matmul);
    }

    {
        SCOPED_TRACE("reduction shared root");
        const Expression x = Expression::input("reduce_x", DataType::FP32, DataType::FP32);
        const Expression shared = x.reduce_sum({1}, {}, DataType::FP32);
        expectSharedStageBoundaryRootUsesOneRuntimeValue(
            observePlannerIdentity(shared, shared, {DataType::FP32}),
            ExprOp::REDUCE_SUM,
            PhysicalExecutionStage::Kind::Reduction);
    }

    {
        SCOPED_TRACE("softmax shared root");
        const Expression x = Expression::input("softmax_x", DataType::FP32, DataType::FP32);
        const Expression shared = x.softmax();
        expectSharedStageBoundaryRootUsesOneRuntimeValue(
            observePlannerIdentity(shared, shared, {DataType::FP32}),
            ExprOp::SOFTMAX,
            PhysicalExecutionStage::Kind::Softmax);
    }

    {
        SCOPED_TRACE("RMSNorm shared root");
        const Expression x = Expression::input("rms_x", DataType::FP32, DataType::FP32);
        const Expression scale = Expression::input("rms_scale", DataType::FP32, DataType::FP32);
        const Expression shared = Expression::rmsNorm(x, scale, 4, 1.0e-5, DataType::FP32, DataType::FP32);
        expectSharedStageBoundaryRootUsesOneRuntimeValue(
            observePlannerIdentity(shared, shared, {DataType::FP32, DataType::FP32}),
            ExprOp::RMSNORM,
            PhysicalExecutionStage::Kind::RmsNorm);
    }
}

// PERSISTENT-DAG INVARIANT:
// One authored Expression reused in several custom-CUDA ABI positions remains
// one logical application input. ABI argument multiplicity must not clone the
// producer computation.
TEST(ExpressionPersistentDagInvariant, CustomCudaRepeatedInputPreservesSharedProducerPerArgument) {
    const Expression projection = fp32Projection();
    const Outputs applied = repeatedInputInspectionKernel().apply({
        {"a", projection},
        {"b", projection},
    });
    const Expression output = applied.outputExpression("out");

    const LogicalCudaKernelApplicationPtr application = ExpressionInternalAccess::cudaKernelApplication(output);
    ASSERT_TRUE(application);
    ASSERT_EQ(application->inputs().size(), 2u);
    const auto input0 = ExpressionInternalAccess::cudaKernelApplicationInput(output, 0);
    const auto input1 = ExpressionInternalAccess::cudaKernelApplicationInput(output, 1);
    ASSERT_TRUE(input0.has_value());
    ASSERT_TRUE(input1.has_value());
    EXPECT_TRUE(input0->isSameLogicalNode(projection));
    EXPECT_TRUE(input1->isSameLogicalNode(projection));
    EXPECT_TRUE(input0->isSameLogicalNode(*input1));

    const PhysicalOutputs outputs = applied.physicalOutputs();
    ASSERT_EQ(outputs.outputs.size(), 1u);
    ASSERT_TRUE(outputs.expr);
    ASSERT_EQ(outputs.expr->cuda_kernel_expressions.size(), 1u);
    const ExprNode& kernel_output = outputs.expr->nodes.at(outputs.outputs.front().node_idx);
    ASSERT_EQ(kernel_output.op, ExprOp::CUDA_KERNEL_OUTPUT);
    ASSERT_EQ(kernel_output.cuda_kernel_input_nodes.size(), 2u);
    EXPECT_EQ(countLinearBoundaryNodes(outputs), 1u);
    EXPECT_EQ(kernel_output.cuda_kernel_input_nodes[0], kernel_output.cuda_kernel_input_nodes[1]);
}

// One apply() call is one application identity, shared by all of its outputs.
// A separately authored apply() call is a distinct application even when it
// uses the same kernel object and exact same input handles. Input map iteration
// order must not affect ABI ordering.
TEST(ExpressionPersistentDagInvariant, CustomCudaApplicationIdentityAndAbiOrderAreAuthoredExplicitly) {
    const CudaKernelExpression kernel = multiOutputInspectionKernel();
    const Expression a = Expression::input("a", DataType::FP32, DataType::FP32);
    const Expression b = Expression::input("b", DataType::FP32, DataType::FP32);

    const Outputs first = kernel.apply({{"b", b}, {"a", a}});
    const Expression first0 = first.outputExpression("out0");
    const Expression first1 = first.outputExpression("out1");
    const LogicalCudaKernelApplicationPtr application0 = ExpressionInternalAccess::cudaKernelApplication(first0);
    const LogicalCudaKernelApplicationPtr application1 = ExpressionInternalAccess::cudaKernelApplication(first1);
    ASSERT_TRUE(application0);
    ASSERT_TRUE(application1);
    EXPECT_EQ(application0.get(), application1.get());
    ASSERT_EQ(application0->inputs().size(), 2u);

    const auto abi0 = ExpressionInternalAccess::cudaKernelApplicationInput(first0, 0);
    const auto abi1 = ExpressionInternalAccess::cudaKernelApplicationInput(first0, 1);
    ASSERT_TRUE(abi0.has_value());
    ASSERT_TRUE(abi1.has_value());
    EXPECT_TRUE(abi0->isSameLogicalNode(a));
    EXPECT_TRUE(abi1->isSameLogicalNode(b));

    const Outputs second = kernel.apply({{"a", a}, {"b", b}});
    const Expression second0 = second.outputExpression("out0");
    const LogicalCudaKernelApplicationPtr application2 = ExpressionInternalAccess::cudaKernelApplication(second0);
    ASSERT_TRUE(application2);
    EXPECT_NE(application0.get(), application2.get());
    EXPECT_EQ(application0->specification()->cacheSignature(), application2->specification()->cacheSignature());

    PhysicalOutputs lowered = Expression::outputs({
        {"first0", first0},
        {"first1", first1},
        {"second0", second0},
    }).physicalOutputs();
    ASSERT_TRUE(lowered.expr);
    ASSERT_EQ(lowered.expr->cuda_kernel_expressions.size(), 2u);
    const ExprNode& physical_first0 = lowered.expr->nodes.at(lowered.outputs[0].node_idx);
    const ExprNode& physical_first1 = lowered.expr->nodes.at(lowered.outputs[1].node_idx);
    const ExprNode& physical_second0 = lowered.expr->nodes.at(lowered.outputs[2].node_idx);
    EXPECT_EQ(physical_first0.cuda_kernel_spec_index, physical_first1.cuda_kernel_spec_index);
    EXPECT_NE(physical_first0.cuda_kernel_spec_index, physical_second0.cuda_kernel_spec_index);
    EXPECT_EQ(physical_first0.cuda_kernel_input_nodes, physical_first1.cuda_kernel_input_nodes);
    ASSERT_EQ(physical_first0.cuda_kernel_input_nodes.size(), 2u);
    EXPECT_EQ(physical_first0.cuda_kernel_input_nodes, physical_second0.cuda_kernel_input_nodes);

    resolveOutputsDTypesInPlace(lowered, {DataType::FP32, DataType::FP32});
    detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(lowered);
    const std::vector<PhysicalExecutionStage>& stages = plan.stages;
    const std::vector<CompiledStageOutput>& final_outputs = plan.final_outputs;
    EXPECT_EQ(countStagesOfKind(stages, PhysicalExecutionStage::Kind::CudaKernel), 2u)
        << "Separate custom-CUDA application identities must remain separate execution applications.";
    const std::optional<uint32_t> first0_id = finalValueIdNamed(final_outputs, "first0");
    const std::optional<uint32_t> first1_id = finalValueIdNamed(final_outputs, "first1");
    const std::optional<uint32_t> second0_id = finalValueIdNamed(final_outputs, "second0");
    ASSERT_TRUE(first0_id.has_value());
    ASSERT_TRUE(first1_id.has_value());
    ASSERT_TRUE(second0_id.has_value());
    EXPECT_NE(*first0_id, *first1_id);
    EXPECT_NE(*first0_id, *second0_id);
    EXPECT_NE(*first1_id, *second0_id);
}

// PERSISTENT-DAG INVARIANT:
// One custom-CUDA application may legitimately execute as one physical stage while
// producing several authored outputs. Those outputs are not aliases: each physical
// output root receives its own runtime value_id even though application identity is
// shared and stage count is one.
TEST(ExpressionPersistentDagInvariant, CustomCudaMultiOutputApplicationUsesDistinctRuntimeValuesInOneStage) {
    const CudaKernelExpression kernel = multiOutputInspectionKernel();
    const Expression a = Expression::input("cuda_a", DataType::FP32, DataType::FP32);
    const Expression b = Expression::input("cuda_b", DataType::FP32, DataType::FP32);
    const Outputs applied = kernel.apply({{"a", a}, {"b", b}});

    PhysicalOutputs physical = applied.physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32, DataType::FP32});
    ASSERT_TRUE(physical.expr);
    ASSERT_EQ(physical.outputs.size(), 2u);
    ASSERT_NE(physical.outputs[0].node_idx, physical.outputs[1].node_idx);
    const ExprNode& first = physical.expr->nodes.at(physical.outputs[0].node_idx);
    const ExprNode& second = physical.expr->nodes.at(physical.outputs[1].node_idx);
    ASSERT_EQ(first.op, ExprOp::CUDA_KERNEL_OUTPUT);
    ASSERT_EQ(second.op, ExprOp::CUDA_KERNEL_OUTPUT);
    EXPECT_EQ(first.cuda_kernel_spec_index, second.cuda_kernel_spec_index);
    EXPECT_NE(first.cuda_kernel_output_index, second.cuda_kernel_output_index);

    detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(physical);
    const std::vector<PhysicalExecutionStage>& stages = plan.stages;
    const std::vector<CompiledStageOutput>& final_outputs = plan.final_outputs;
    ASSERT_EQ(countStagesOfKind(stages, PhysicalExecutionStage::Kind::CudaKernel), 1u);
    const std::optional<uint32_t> out0_value = finalValueIdNamed(final_outputs, "out0");
    const std::optional<uint32_t> out1_value = finalValueIdNamed(final_outputs, "out1");
    ASSERT_TRUE(out0_value.has_value());
    ASSERT_TRUE(out1_value.has_value());
    EXPECT_NE(*out0_value, *out1_value);
    EXPECT_EQ(countStageOutputsWithValueId(stages, *out0_value), 1u);
    EXPECT_EQ(countStageOutputsWithValueId(stages, *out1_value), 1u);

    const auto cuda_stage_it = std::find_if(stages.begin(), stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::CudaKernel;
    });
    ASSERT_NE(cuda_stage_it, stages.end());
    EXPECT_EQ(cuda_stage_it->outputs.size(), 2u);
}

// PERSISTENT-DAG INVARIANT:
// scanWithIndices authors a paired value/index request represented by two physical
// SCAN roots. The planner recognizes that compatible pair and intentionally executes
// it in one Scan stage while allocating distinct runtime value_ids. One-stage
// execution does not imply one-value identity.
TEST(ExpressionPersistentDagInvariant, PairedScanOutputsUseDistinctRuntimeValuesInOneStage) {
    const Expression x = Expression::input("scan_x", DataType::FP32, DataType::FP32);
    const auto [values, indices] = x.scanWithIndices(ScanOp::Max, -1, true);
    PhysicalOutputs physical = Expression::outputs({{"values", values}, {"indices", indices}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32});

    ASSERT_TRUE(physical.expr);
    ASSERT_EQ(physical.outputs.size(), 2u);
    ASSERT_NE(physical.outputs[0].node_idx, physical.outputs[1].node_idx);
    EXPECT_EQ(physical.expr->nodes.at(physical.outputs[0].node_idx).op, ExprOp::SCAN);
    EXPECT_EQ(physical.expr->nodes.at(physical.outputs[1].node_idx).op, ExprOp::SCAN);

    detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(physical);
    const std::vector<PhysicalExecutionStage>& stages = plan.stages;
    const std::vector<CompiledStageOutput>& final_outputs = plan.final_outputs;
    ASSERT_EQ(countStagesOfKind(stages, PhysicalExecutionStage::Kind::Scan), 1u);
    const std::optional<uint32_t> values_id = finalValueIdNamed(final_outputs, "values");
    const std::optional<uint32_t> indices_id = finalValueIdNamed(final_outputs, "indices");
    ASSERT_TRUE(values_id.has_value());
    ASSERT_TRUE(indices_id.has_value());
    EXPECT_NE(*values_id, *indices_id);
    EXPECT_EQ(countStageOutputsWithValueId(stages, *values_id), 1u);
    EXPECT_EQ(countStageOutputsWithValueId(stages, *indices_id), 1u);

    const auto scan_stage_it = std::find_if(stages.begin(), stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::Scan;
    });
    ASSERT_NE(scan_stage_it, stages.end());
    EXPECT_EQ(scan_stage_it->outputs.size(), 2u);
}

// PERSISTENT-DAG INVARIANT:
// Special EquationCompiler post-planning merges may reduce dispatch count without
// redefining runtime value identity. RMSNorm backward dX/dScale are separate physical
// outputs and keep separate value_ids even though mergeRmsNormBackwardStages combines
// their compatible routes into one backend stage.
TEST(ExpressionPersistentDagInvariant, RmsNormBackwardMergePreservesDistinctRuntimeValues) {
    const Expression x = Expression::input("rms_bwd_x", DataType::FP32, DataType::FP32);
    const Expression scale = Expression::input("rms_bwd_scale", DataType::FP32, DataType::FP32);
    const Expression y = Expression::rmsNorm(x, scale, 8, 1.0e-5, DataType::FP32, DataType::FP32);

    PhysicalOutputs forward = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP32, DataType::FP32});
    PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"rms_bwd_x", "rms_bwd_scale"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"rms_bwd_x", {4, 8}},
            {"rms_bwd_scale", {8}},
        });
    resolveOutputsDTypesInPlace(backward, std::vector<DataType>(backward.expr->inputs.size(), DataType::FP32));

    detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(backward);
    ASSERT_EQ(countStagesOfKind(plan.stages, PhysicalExecutionStage::Kind::RmsNormBackward), 1u);
    const auto stage_it = std::find_if(plan.stages.begin(), plan.stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::RmsNormBackward;
    });
    ASSERT_NE(stage_it, plan.stages.end());
    ASSERT_EQ(stage_it->outputs.size(), 2u);
    EXPECT_NE(stage_it->outputs[0].value_id, stage_it->outputs[1].value_id);
    EXPECT_NE(stage_it->outputs[0].local_node_idx, stage_it->outputs[1].local_node_idx);

    const std::optional<uint32_t> dx_value = finalValueIdNamed(plan.final_outputs, "rms_bwd_x_grad");
    const std::optional<uint32_t> dscale_value = finalValueIdNamed(plan.final_outputs, "rms_bwd_scale_grad");
    ASSERT_TRUE(dx_value.has_value());
    ASSERT_TRUE(dscale_value.has_value());
    EXPECT_NE(*dx_value, *dscale_value);
    EXPECT_EQ(countStageOutputsWithValueId(plan.stages, *dx_value), 1u);
    EXPECT_EQ(countStageOutputsWithValueId(plan.stages, *dscale_value), 1u);
}

// PERSISTENT-DAG INVARIANT:
// Attention backward dQ/dK/dV are one authored multi-output backend application in
// execution terms, but each output remains its own planner value. The structural
// attentionBackwardMergeKey may batch compatible routes; it must never make those
// routes aliases.
TEST(ExpressionPersistentDagInvariant, AttentionBackwardMergePreservesDistinctRuntimeValues) {
    const Expression q = Expression::input("attn_bwd_q", DataType::FP32, DataType::FP32);
    const Expression k = Expression::input("attn_bwd_k", DataType::FP32, DataType::FP32);
    const Expression v = Expression::input("attn_bwd_v", DataType::FP32, DataType::FP32);
    AttentionOptions options;
    options.compute_dtype = DataType::FP32;
    options.output_dtype = DataType::FP32;
    const Expression y = Expression::attention(q, k, v, options);

    PhysicalOutputs forward = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP32, DataType::FP32, DataType::FP32});
    PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"attn_bwd_q", "attn_bwd_k", "attn_bwd_v"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"attn_bwd_q", {2, 2, 3, 4}},
            {"attn_bwd_k", {2, 2, 3, 4}},
            {"attn_bwd_v", {2, 2, 3, 4}},
        });
    resolveOutputsDTypesInPlace(backward, std::vector<DataType>(backward.expr->inputs.size(), DataType::FP32));

    detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(backward);
    ASSERT_EQ(countStagesOfKind(plan.stages, PhysicalExecutionStage::Kind::AttentionBackward), 1u);
    const auto stage_it = std::find_if(plan.stages.begin(), plan.stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::AttentionBackward;
    });
    ASSERT_NE(stage_it, plan.stages.end());
    ASSERT_EQ(stage_it->outputs.size(), 3u);

    const std::optional<uint32_t> dq = finalValueIdNamed(plan.final_outputs, "attn_bwd_q_grad");
    const std::optional<uint32_t> dk = finalValueIdNamed(plan.final_outputs, "attn_bwd_k_grad");
    const std::optional<uint32_t> dv = finalValueIdNamed(plan.final_outputs, "attn_bwd_v_grad");
    ASSERT_TRUE(dq.has_value());
    ASSERT_TRUE(dk.has_value());
    ASSERT_TRUE(dv.has_value());
    EXPECT_NE(*dq, *dk);
    EXPECT_NE(*dq, *dv);
    EXPECT_NE(*dk, *dv);
    EXPECT_EQ(countStageOutputsWithValueId(plan.stages, *dq), 1u);
    EXPECT_EQ(countStageOutputsWithValueId(plan.stages, *dk), 1u);
    EXPECT_EQ(countStageOutputsWithValueId(plan.stages, *dv), 1u);
}

// PERSISTENT-DAG INVARIANT:
// Q/K RoPE materialization is deliberately grouped before an Attention boundary when
// both roots are available. Grouping the two index-aware expressions into one fused
// stage is an execution optimization: the two RoPE physical roots must keep distinct
// planner values and Attention must consume both distinct values.
TEST(ExpressionPersistentDagInvariant, GroupedRopeMaterializationPreservesDistinctRuntimeValues) {
    const Expression q = Expression::input("rope_q", DataType::FP32, DataType::FP32);
    const Expression k = Expression::input("rope_k", DataType::FP32, DataType::FP32);
    const Expression v = Expression::input("rope_v", DataType::FP32, DataType::FP32);
    RotaryPositionEmbeddingOptions rope_options;
    rope_options.rotary_dim = 4;
    rope_options.compute_dtype = DataType::FP32;
    rope_options.output_dtype = DataType::FP32;

    const Expression rope_q = q.rotaryPositionEmbedding(rope_options);
    const Expression rope_k = k.rotaryPositionEmbedding(rope_options);
    AttentionOptions attention_options;
    attention_options.compute_dtype = DataType::FP32;
    attention_options.output_dtype = DataType::FP32;
    const Expression y = Expression::attention(rope_q, rope_k, v, attention_options);

    PhysicalOutputs physical = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32, DataType::FP32, DataType::FP32});
    detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(physical);

    ASSERT_EQ(countStagesOfKind(plan.stages, PhysicalExecutionStage::Kind::Attention), 1u);
    const auto grouped_rope_it = std::find_if(plan.stages.begin(), plan.stages.end(), [](const PhysicalExecutionStage& stage) {
        if (stage.kind != PhysicalExecutionStage::Kind::FusedKernel || stage.outputs.size() != 2u) {
            return false;
        }
        size_t rope_nodes = 0;
        for (const CompiledStageOutput& output : stage.outputs) {
            if (output.local_node_idx < stage.expr.nodes.size() && stage.expr.nodes[output.local_node_idx].op == ExprOp::ROPE) {
                ++rope_nodes;
            }
        }
        return rope_nodes == 2u;
    });
    ASSERT_NE(grouped_rope_it, plan.stages.end());
    ASSERT_EQ(grouped_rope_it->outputs.size(), 2u);
    EXPECT_NE(grouped_rope_it->outputs[0].value_id, grouped_rope_it->outputs[1].value_id);

    const auto attention_it = std::find_if(plan.stages.begin(), plan.stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::Attention;
    });
    ASSERT_NE(attention_it, plan.stages.end());
    ASSERT_GE(attention_it->input_value_ids.size(), 3u);
    EXPECT_NE(attention_it->input_value_ids[0], attention_it->input_value_ids[1]);
    EXPECT_TRUE(attention_it->input_value_ids[0] == grouped_rope_it->outputs[0].value_id ||
                attention_it->input_value_ids[0] == grouped_rope_it->outputs[1].value_id);
    EXPECT_TRUE(attention_it->input_value_ids[1] == grouped_rope_it->outputs[0].value_id ||
                attention_it->input_value_ids[1] == grouped_rope_it->outputs[1].value_id);
}

// PERSISTENT-DAG INVARIANT:
// A cuBLASLt backward-epilogue matmul may expose its axis-0 bias reduction as a
// secondary output of the same Matmul stage. The paired output is a backend execution
// optimization only: matrix output and bias-gradient output retain distinct value_ids.
TEST(ExpressionPersistentDagInvariant, MatmulBiasGradientPairPreservesDistinctRuntimeValues) {
    const Expression lhs = Expression::input("bgrad_lhs", DataType::FP32, DataType::FP32);
    const Expression rhs = Expression::input("bgrad_rhs", DataType::FP32, DataType::FP32);
    PhysicalOutputs physical =
        Expression::outputs({{"matrix", Expression::matmul(lhs, rhs, false, false, DataType::FP32, DataType::FP32)}})
            .physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32, DataType::FP32});

    ASSERT_TRUE(physical.expr);
    ASSERT_EQ(physical.outputs.size(), 1u);
    const uint32_t matmul_idx = physical.outputs.front().node_idx;
    ASSERT_LT(matmul_idx, physical.expr->nodes.size());
    ExprNode& matmul = physical.expr->nodes[matmul_idx];
    ASSERT_EQ(matmul.op, ExprOp::MATMUL);
    matmul.matmul_backward_epilogue = MatmulBackwardEpilogue::DRelu;
    matmul.matmul_epilogue_aux = matmul.lhs;

    ExprNode bgrad;
    bgrad.op = ExprOp::REDUCE_SUM;
    bgrad.lhs = matmul_idx;
    bgrad.reduction_axes = {0};
    bgrad.squeeze_axes = {0};
    bgrad.compute_dtype = DataType::FP32;
    bgrad.output_dtype = DataType::FP32;
    physical.expr->nodes.push_back(std::move(bgrad));
    const uint32_t bgrad_idx = static_cast<uint32_t>(physical.expr->nodes.size() - 1);
    physical.outputs.push_back(NamedOutput{.name = "bias_grad", .node_idx = bgrad_idx});

    detail::EquationCompilerPlanForTests plan = detail::planEquationCompilerForTests(physical);
    ASSERT_EQ(countStagesOfKind(plan.stages, PhysicalExecutionStage::Kind::Matmul), 1u);
    const auto stage_it = std::find_if(plan.stages.begin(), plan.stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::Matmul;
    });
    ASSERT_NE(stage_it, plan.stages.end());
    ASSERT_EQ(stage_it->outputs.size(), 2u);
    EXPECT_NE(stage_it->outputs[0].value_id, stage_it->outputs[1].value_id);
    EXPECT_NE(stage_it->outputs[0].local_node_idx, stage_it->outputs[1].local_node_idx);

    const std::optional<uint32_t> matrix_value = finalValueIdNamed(plan.final_outputs, "matrix");
    const std::optional<uint32_t> bgrad_value = finalValueIdNamed(plan.final_outputs, "bias_grad");
    ASSERT_TRUE(matrix_value.has_value());
    ASSERT_TRUE(bgrad_value.has_value());
    EXPECT_NE(*matrix_value, *bgrad_value);
    EXPECT_EQ(countStageOutputsWithValueId(plan.stages, *matrix_value), 1u);
    EXPECT_EQ(countStageOutputsWithValueId(plan.stages, *bgrad_value), 1u);
}

// Runtime-scalar ABI inputs historically had all forward/backward dtype fields
// stamped to the ABI-declared dtype during physical assembly. The logical-native
// builder must preserve that contract without mutating the authored input.
TEST(ExpressionPersistentDagInvariant, CustomCudaRuntimeScalarAbiDTypeAdaptationIsImmutable) {
    const CudaKernelExpression kernel = runtimeScalarInspectionKernel();
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression seed = Expression::tensorRuntimeScalar("seed");
    const Expression scale = Expression::runtimeScalar("scale");

    ASSERT_FALSE(ExpressionInternalAccess::rootSemantics(seed).output_dtype.has_value());
    ASSERT_FALSE(ExpressionInternalAccess::rootSemantics(scale).output_dtype.has_value());

    const Outputs applied = kernel.apply({{"scale", scale}, {"x", x}, {"seed", seed}});
    const Expression output = applied.outputExpression("out");
    const auto abi_x = ExpressionInternalAccess::cudaKernelApplicationInput(output, 0);
    const auto abi_seed = ExpressionInternalAccess::cudaKernelApplicationInput(output, 1);
    const auto abi_scale = ExpressionInternalAccess::cudaKernelApplicationInput(output, 2);
    ASSERT_TRUE(abi_x.has_value());
    ASSERT_TRUE(abi_seed.has_value());
    ASSERT_TRUE(abi_scale.has_value());

    EXPECT_TRUE(abi_x->isSameLogicalNode(x));
    EXPECT_FALSE(abi_seed->isSameLogicalNode(seed));
    EXPECT_FALSE(abi_scale->isSameLogicalNode(scale));
    EXPECT_EQ(ExpressionInternalAccess::rootOp(*abi_seed), ExprOp::TENSOR_RUNTIME_SCALAR);
    EXPECT_EQ(ExpressionInternalAccess::rootOp(*abi_scale), ExprOp::RUNTIME_SCALAR);

    auto expectAllDTypes = [](const ExprNode& semantics, DataType dtype) {
        ASSERT_TRUE(semantics.input_tensor_dtype.has_value());
        ASSERT_TRUE(semantics.compute_dtype.has_value());
        ASSERT_TRUE(semantics.output_dtype.has_value());
        ASSERT_TRUE(semantics.backward_output_dtype.has_value());
        ASSERT_TRUE(semantics.backward_compute_dtype.has_value());
        EXPECT_EQ(*semantics.input_tensor_dtype, dtype);
        EXPECT_EQ(*semantics.compute_dtype, dtype);
        EXPECT_EQ(*semantics.output_dtype, dtype);
        EXPECT_EQ(*semantics.backward_output_dtype, dtype);
        EXPECT_EQ(*semantics.backward_compute_dtype, dtype);
    };
    expectAllDTypes(ExpressionInternalAccess::rootSemantics(*abi_seed), DataType::INT64);
    expectAllDTypes(ExpressionInternalAccess::rootSemantics(*abi_scale), DataType::FP32);

    // The original authored scalar Expressions remain unchanged.
    EXPECT_FALSE(ExpressionInternalAccess::rootSemantics(seed).output_dtype.has_value());
    EXPECT_FALSE(ExpressionInternalAccess::rootSemantics(scale).output_dtype.has_value());

    const PhysicalOutputs lowered = applied.physicalOutputs();
    const ExprNode& physical_output = lowered.expr->nodes.at(lowered.outputs.front().node_idx);
    ASSERT_EQ(physical_output.cuda_kernel_input_nodes.size(), 3u);
    EXPECT_EQ(lowered.expr->nodes.at(physical_output.cuda_kernel_input_nodes[0]).op, ExprOp::INPUT);
    const ExprNode& physical_seed = lowered.expr->nodes.at(physical_output.cuda_kernel_input_nodes[1]);
    const ExprNode& physical_scale = lowered.expr->nodes.at(physical_output.cuda_kernel_input_nodes[2]);
    EXPECT_EQ(physical_seed.op, ExprOp::TENSOR_RUNTIME_SCALAR);
    EXPECT_EQ(physical_scale.op, ExprOp::RUNTIME_SCALAR);
    expectAllDTypes(physical_seed, DataType::INT64);
    expectAllDTypes(physical_scale, DataType::FP32);
}

// The optional cuDNN RMSNorm+Swish fusion is an immutable logical semantic
// rewrite. The fused root is distinct, but its authored input/scale
// dependencies must remain exactly shared with the original RMSNorm root.
TEST(ExpressionPersistentDagInvariant, RmsNormFusedActivationRewriteIsLogicalAndImmutable) {
    const Expression input = Expression::input("rms_input", DataType::BF16, DataType::BF16);
    const Expression scale = Expression::input("rms_scale", DataType::BF16, DataType::BF16);
    const Expression ordinary =
        Expression::rmsNorm(input, scale, 32, 1.0e-5, DataType::FP32, DataType::BF16);
    const Expression fused = ExpressionInternalAccess::withRmsNormFusedActivation(
        ordinary, CudnnRmsNormFusedActivation::SWISH);

    EXPECT_FALSE(ordinary.isSameLogicalNode(fused));
    EXPECT_EQ(ExpressionInternalAccess::rootOp(ordinary), ExprOp::RMSNORM);
    EXPECT_EQ(ExpressionInternalAccess::rootOp(fused), ExprOp::RMSNORM);
    EXPECT_EQ(ExpressionInternalAccess::rootSemantics(ordinary).rms_norm_fused_activation,
              CudnnRmsNormFusedActivation::NONE);
    EXPECT_EQ(ExpressionInternalAccess::rootSemantics(fused).rms_norm_fused_activation,
              CudnnRmsNormFusedActivation::SWISH);

    const auto ordinary_input = ExpressionInternalAccess::lhsDependency(ordinary);
    const auto ordinary_scale = ExpressionInternalAccess::rhsDependency(ordinary);
    const auto fused_input = ExpressionInternalAccess::lhsDependency(fused);
    const auto fused_scale = ExpressionInternalAccess::rhsDependency(fused);
    ASSERT_TRUE(ordinary_input.has_value());
    ASSERT_TRUE(ordinary_scale.has_value());
    ASSERT_TRUE(fused_input.has_value());
    ASSERT_TRUE(fused_scale.has_value());
    EXPECT_TRUE(ordinary_input->isSameLogicalNode(input));
    EXPECT_TRUE(ordinary_scale->isSameLogicalNode(scale));
    EXPECT_TRUE(fused_input->isSameLogicalNode(input));
    EXPECT_TRUE(fused_scale->isSameLogicalNode(scale));
    EXPECT_TRUE(ordinary_input->isSameLogicalNode(*fused_input));
    EXPECT_TRUE(ordinary_scale->isSameLogicalNode(*fused_scale));

    // The rewrite is deliberately narrow and cannot be used as a generic
    // semantic mutation escape hatch.
    EXPECT_THROW((void)ExpressionInternalAccess::withRmsNormFusedActivation(input, CudnnRmsNormFusedActivation::SWISH),
                 std::invalid_argument);

    const PhysicalOutputs lowered =
        Expression::outputs({{"ordinary", ordinary}, {"fused", fused}}).physicalOutputs();
    ASSERT_TRUE(lowered.expr);
    ASSERT_EQ(lowered.outputs.size(), 2u);
    EXPECT_EQ(countNodesOfKind(lowered, ExprOp::RMSNORM), 2u);
    EXPECT_EQ(countInputNodesNamed(lowered, "rms_input"), 1u);
    EXPECT_EQ(countInputNodesNamed(lowered, "rms_scale"), 1u);

    const ExprNode& ordinary_physical = lowered.expr->nodes.at(lowered.outputs[0].node_idx);
    const ExprNode& fused_physical = lowered.expr->nodes.at(lowered.outputs[1].node_idx);
    EXPECT_EQ(ordinary_physical.rms_norm_fused_activation, CudnnRmsNormFusedActivation::NONE);
    EXPECT_EQ(fused_physical.rms_norm_fused_activation, CudnnRmsNormFusedActivation::SWISH);
    EXPECT_EQ(ordinary_physical.lhs, fused_physical.lhs);
    EXPECT_EQ(ordinary_physical.rhs, fused_physical.rhs);
}

// PERSISTENT-DAG INVARIANT:
// Several roots imported from one already-physical graph share one importer
// context, so one source physical node becomes one logical DAG node. Physical
// import is a legitimate boundary and is not a logical graph-construction path.
TEST(ExpressionPersistentDagInvariant, MultiRootPhysicalImportPreservesSharingThroughOneImporter) {
    const Expression projection = fp32Projection();
    const auto [scan_values, scan_indices] = projection.scanWithIndices(ScanOp::Max, -1, true);

    PhysicalOutputs source =
        Expression::outputs({{"values", scan_values}, {"indices", scan_indices}}).physicalOutputs();
    ASSERT_TRUE(source.expr);
    ASSERT_EQ(source.outputs.size(), 2u);

    const Outputs imported = Outputs::fromPhysicalOutputs(source);
    const PhysicalOutputs outputs =
        Expression::outputs({{"values", imported.outputExpression("values")},
                             {"indices", imported.outputExpression("indices")}})
            .physicalOutputs();

    EXPECT_EQ(countLinearBoundaryNodes(outputs), 1u);
}

// PERSISTENT-DAG INVARIANT:
// Several roots imported from one PhysicalExpression share one import context, and
// independent root-only transformations retain their common unchanged ancestor by
// logical identity.
TEST(ExpressionPersistentDagInvariant, IndependentlyTransformedPhysicalRootImportsPreserveSharedAncestor) {
    const Expression projection = fp32Projection();
    const auto [scan_values, scan_indices] = projection.scanWithIndices(ScanOp::Max, -1, true);

    PhysicalOutputs source =
        Expression::outputs({{"values", scan_values}, {"indices", scan_indices}}).physicalOutputs();
    ASSERT_TRUE(source.expr);
    ASSERT_EQ(source.outputs.size(), 2u);
    const Outputs imported = Outputs::fromPhysicalOutputs(source);

    const Expression transformed_values = imported.outputExpression("values").withOutputDType(DataType::FP32);
    const Expression transformed_indices = imported.outputExpression("indices").withOutputDType(DataType::UINT32);
    const PhysicalOutputs outputs =
        Expression::outputs({{"values", transformed_values}, {"indices", transformed_indices}}).physicalOutputs();

    EXPECT_EQ(countLinearBoundaryNodes(outputs), 1u);
}

// PERSISTENT-DAG INVARIANT:
// substituteInput() path-copies only nodes on paths that contain the substituted
// logical input. Unaffected immutable sibling subgraphs remain shared between the
// original and transformed expressions, and the original remains unchanged.
TEST(ExpressionPersistentDagInvariant, SubstituteInputSharesUnaffectedSiblingSubgraph) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression wx = Expression::input("wx", DataType::FP32, DataType::FP32);
    const Expression y = Expression::input("y", DataType::FP32, DataType::FP32);
    const Expression wy = Expression::input("wy", DataType::FP32, DataType::FP32);

    const Expression affected = Expression::matmul(x, wx, false, false, DataType::FP32, DataType::FP32).sin();
    const Expression unaffected = Expression::matmul(y, wy, false, false, DataType::FP32, DataType::FP32).cos().exp();
    const Expression original = affected + unaffected;
    const std::string original_before = canonicalize(Expression::outputs({{"original", original}}).physicalOutputs());

    const Expression replacement = Expression::input("x_replacement", DataType::FP32, DataType::FP32);
    const Expression transformed = original.substituteInput("x", replacement);

    // Functional immutability is a permanent invariant.
    EXPECT_EQ(canonicalize(Expression::outputs({{"original", original}}).physicalOutputs()), original_before);
    EXPECT_TRUE(original.getInputNames().contains("x"));
    EXPECT_FALSE(original.getInputNames().contains("x_replacement"));
    EXPECT_FALSE(transformed.getInputNames().contains("x"));
    EXPECT_TRUE(transformed.getInputNames().contains("x_replacement"));

    const PhysicalOutputs combined =
        Expression::outputs({{"original", original}, {"transformed", transformed}}).physicalOutputs();
    EXPECT_EQ(countLinearBoundaryNodes(combined), 3u);
}

// PERSISTENT-DAG INVARIANT:
// Memoization during substitution preserves authored sharing inside the changed
// region too. A shared affected producer is transformed once, and every changed
// parent edge points at that one transformed logical value.
TEST(ExpressionPersistentDagInvariant, SubstituteInputTransformsSharedAffectedSubgraphOnce) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w = Expression::input("w", DataType::FP32, DataType::FP32);
    const Expression projection = Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32);
    const Expression shared = projection.sin();
    const Expression original = shared + shared;

    const Expression replacement = Expression::input("x_replacement", DataType::FP32, DataType::FP32);
    const Expression transformed = original.substituteInput("x", replacement);
    const PhysicalOutputs outputs = Expression::outputs({{"transformed", transformed}}).physicalOutputs();

    EXPECT_EQ(countLinearBoundaryNodes(outputs), 1u);
    EXPECT_FALSE(transformed.getInputNames().contains("x"));
    EXPECT_TRUE(transformed.getInputNames().contains("x_replacement"));
}

// PERSISTENT-DAG INVARIANT:
// Flattening the same immutable logical Outputs multiple times produces
// semantically and topologically equivalent PhysicalExpressions with
// deterministic indexing/order wherever the existing compiler contract requires
// determinism.
TEST(ExpressionPersistentDagInvariant, IndependentPhysicalLoweringsAreDeterministic) {
    const Expression projection = fp32Projection();
    const Expression y = projection.sin() + projection.cos();

    const PhysicalOutputs first =
        Expression::outputs({{"primary", y}, {"projection", projection}}).physicalOutputs();
    const PhysicalOutputs second =
        Expression::outputs({{"primary", y}, {"projection", projection}}).physicalOutputs();

    expectEquivalentPhysicalTopology(first, second);
}

// Positive authored-identity control: logical identity is based on authored/reused
// values, not structural equivalence. Reusing one Expression handle means one
// logical value, while calling the same operation twice creates two distinct values.
TEST(ExpressionPersistentDagInvariant, ReusedHandleAndReauthoredOperationHaveDifferentLogicalMultiplicity) {
    const Expression reused_projection = fp32Projection();
    const PhysicalOutputs reused =
        Expression::outputs({{"y", reused_projection * reused_projection}}).physicalOutputs();
    EXPECT_EQ(countLinearBoundaryNodes(reused), 1u);

    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w = Expression::input("w", DataType::FP32, DataType::FP32);
    const Expression projection_a = Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32);
    const Expression projection_b = Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32);
    const PhysicalOutputs reauthored = Expression::outputs({{"y", projection_a * projection_b}}).physicalOutputs();
    EXPECT_EQ(countLinearBoundaryNodes(reauthored), 2u);
}

// PERSISTENT-DAG INVARIANT:
// Logical identity for ragged values and their associated metadata follows the
// same rules as dense Expressions. Deriving two values from one ragged logical
// value does not duplicate its producer or silently create distinct metadata
// values.
TEST(ExpressionPersistentDagInvariant, RaggedDerivedOutputsShareValueAndPartitionNodes) {
    const RaggedExpression ragged = RaggedExpression::input("ragged", persistentDagRaggedDescriptor());
    const RaggedExpression absolute = ragged.abs();
    const RaggedExpression exponential = ragged.exp();

    const PhysicalOutputs outputs = Expression::outputs(
                                        {{"absolute", absolute.getValues()}, {"exponential", exponential.getValues()}})
                                        .physicalOutputs();

    EXPECT_EQ(countInputNodesNamed(outputs, "ragged.values"), 1u);
    EXPECT_EQ(countInputNodesNamed(outputs, "ragged.offsets"), 1u);
}

// PERSISTENT-DAG INVARIANT:
// A chain of N newly-authored operations contains the original logical graph plus
// N new logical nodes. Retaining and lowering intermediate handles does not copy
// their common ancestry; all versions remain views into one persistent DAG.
TEST(ExpressionPersistentDagInvariant, RetainedUnaryChainHasLinearPersistentTopology) {
    constexpr size_t unary_steps = 6;
    const Expression projection = fp32Projection();
    const size_t base_nodes = projection.expression().nodes.size();

    std::vector<std::pair<std::string, Expression>> versions;
    versions.reserve(unary_steps + 1);
    Expression current = projection;
    versions.emplace_back("v0", current);
    for (size_t i = 1; i <= unary_steps; ++i) {
        current = current.sin();
        versions.emplace_back("v" + std::to_string(i), current);
    }

    const PhysicalOutputs outputs = Expression::outputs(versions).physicalOutputs();
    const size_t expected_persistent_nodes = base_nodes + unary_steps;
    EXPECT_EQ(outputs.expr->nodes.size(), expected_persistent_nodes);
    EXPECT_EQ(countLinearBoundaryNodes(outputs), 1u);
}

// PERSISTENT-DAG INVARIANT:
// RoPE metadata is represented by logical dependency roles. Metadata normalization
// remains an authored CAST, but that CAST and the RoPE value edge must share the
// exact original producer when one Expression is reused across roles.
TEST(ExpressionPersistentDagInvariant, RopeMetadataBuilderPreservesReusedProducerAcrossRoles) {
    const Expression projection = fp32Projection();
    const Expression rope = projection.rotaryPositionEmbeddingWithPositionIds(projection);

    const std::optional<Expression> value = ExpressionInternalAccess::lhsDependency(rope);
    const std::optional<Expression> positions = ExpressionInternalAccess::ropePositionIdsDependency(rope);
    ASSERT_TRUE(value.has_value());
    ASSERT_TRUE(positions.has_value());
    EXPECT_TRUE(value->isSameLogicalNode(projection));
    EXPECT_EQ(ExpressionInternalAccess::rootOp(*positions), ExprOp::CAST);

    const std::optional<Expression> positions_source = ExpressionInternalAccess::lhsDependency(*positions);
    ASSERT_TRUE(positions_source.has_value());
    EXPECT_TRUE(positions_source->isSameLogicalNode(projection));
    ASSERT_TRUE(ExpressionInternalAccess::rootSemantics(*positions).output_dtype.has_value());
    EXPECT_EQ(ExpressionInternalAccess::rootSemantics(*positions).output_dtype.value(), DataType::FP32);

    const PhysicalOutputs outputs = Expression::outputs({{"y", rope}}).physicalOutputs();
    EXPECT_EQ(countLinearBoundaryNodes(outputs), 1u);
    EXPECT_EQ(countNodesOfKind(outputs, ExprOp::CAST), 1u);
    EXPECT_EQ(countNodesOfKind(outputs, ExprOp::ROPE), 1u);

    const ExprNode* physical_rope = nullptr;
    for (const ExprNode& candidate : outputs.expr->nodes) {
        if (candidate.op == ExprOp::ROPE) {
            ASSERT_EQ(physical_rope, nullptr);
            physical_rope = &candidate;
        }
    }
    ASSERT_NE(physical_rope, nullptr);
    ASSERT_LT(physical_rope->lhs, outputs.expr->nodes.size());
    ASSERT_LT(physical_rope->rope_position_ids_node, outputs.expr->nodes.size());
    EXPECT_EQ(outputs.expr->nodes.at(physical_rope->lhs).op, ExprOp::MATMUL);
    const ExprNode& physical_positions = outputs.expr->nodes.at(physical_rope->rope_position_ids_node);
    EXPECT_EQ(physical_positions.op, ExprOp::CAST);
    EXPECT_EQ(physical_positions.lhs, physical_rope->lhs);
}

TEST(ExpressionPersistentDagInvariant, RopeEffectiveSequenceLengthUsesNormalizedLogicalDependency) {
    const Expression projection = fp32Projection();
    const Expression lengths = Expression::input("lengths", DataType::UINT32, DataType::UINT32);

    RotaryPositionEmbeddingOptions options;
    options.sequence_axis = 0;
    options.head_dim_axis = 1;
    options.rotary_dim = 4;
    options.scaling_kind = RotaryScalingKind::DynamicNTK;
    options.scaling_factor = 2.0;
    options.original_max_position_embeddings = 128;

    const Expression rope = projection.rotaryPositionEmbeddingWithEffectiveSequenceLength(lengths, options);
    const std::optional<Expression> value = ExpressionInternalAccess::lhsDependency(rope);
    const std::optional<Expression> effective_length =
        ExpressionInternalAccess::ropeEffectiveSequenceLengthDependency(rope);
    ASSERT_TRUE(value.has_value());
    ASSERT_TRUE(effective_length.has_value());
    EXPECT_TRUE(value->isSameLogicalNode(projection));
    EXPECT_EQ(ExpressionInternalAccess::rootOp(*effective_length), ExprOp::CAST);
    ASSERT_TRUE(ExpressionInternalAccess::rootSemantics(*effective_length).output_dtype.has_value());
    EXPECT_EQ(ExpressionInternalAccess::rootSemantics(*effective_length).output_dtype.value(), DataType::FP32);

    const std::optional<Expression> length_source = ExpressionInternalAccess::lhsDependency(*effective_length);
    ASSERT_TRUE(length_source.has_value());
    EXPECT_TRUE(length_source->isSameLogicalNode(lengths));

    const PhysicalOutputs outputs = Expression::outputs({{"y", rope}}).physicalOutputs();
    EXPECT_EQ(countInputNodesNamed(outputs, "lengths"), 1u);
    EXPECT_EQ(countNodesOfKind(outputs, ExprOp::CAST), 1u);
    EXPECT_EQ(countNodesOfKind(outputs, ExprOp::ROPE), 1u);
}

TEST(ExpressionPersistentDagInvariant, RopeCombinedMetadataSharesSourceAndPreservesLongRopeSemantics) {
    const Expression projection = fp32Projection();
    const Expression metadata = Expression::input("rope_metadata", DataType::UINT32, DataType::UINT32);

    RotaryPositionEmbeddingOptions options;
    options.sequence_axis = 1;
    options.head_dim_axis = 3;
    options.rotary_dim = 4;
    options.base = 500000.0;
    options.position_offset = 0;
    options.interleaved = true;
    options.inverse = true;
    options.scaling_kind = RotaryScalingKind::LongRope;
    options.scaling_factor = 8.0;
    options.original_max_position_embeddings = 8192;
    options.attention_factor = 1.25;
    options.yarn_beta_fast = 48.0;
    options.yarn_beta_slow = 2.0;
    options.llama3_low_freq_factor = 1.5;
    options.llama3_high_freq_factor = 6.0;
    options.long_rope_short_factors = {1.0, 1.5};
    options.long_rope_long_factors = {2.0, 3.0};
    options.output_dtype = DataType::BF16;
    options.compute_dtype = DataType::FP32;
    options.allow_in_place_materialization = true;

    const Expression rope =
        projection.rotaryPositionEmbeddingWithPositionIdsAndEffectiveSequenceLength(metadata, metadata, options);
    const std::optional<Expression> positions = ExpressionInternalAccess::ropePositionIdsDependency(rope);
    const std::optional<Expression> effective_length =
        ExpressionInternalAccess::ropeEffectiveSequenceLengthDependency(rope);
    ASSERT_TRUE(positions.has_value());
    ASSERT_TRUE(effective_length.has_value());
    EXPECT_EQ(ExpressionInternalAccess::rootOp(*positions), ExprOp::CAST);
    EXPECT_EQ(ExpressionInternalAccess::rootOp(*effective_length), ExprOp::CAST);
    EXPECT_FALSE(positions->isSameLogicalNode(*effective_length))
        << "Position IDs and effective length are separately authored normalization operations.";

    const std::optional<Expression> positions_source = ExpressionInternalAccess::lhsDependency(*positions);
    const std::optional<Expression> length_source = ExpressionInternalAccess::lhsDependency(*effective_length);
    ASSERT_TRUE(positions_source.has_value());
    ASSERT_TRUE(length_source.has_value());
    EXPECT_TRUE(positions_source->isSameLogicalNode(metadata));
    EXPECT_TRUE(length_source->isSameLogicalNode(metadata));

    const ExprNode& semantics = ExpressionInternalAccess::rootSemantics(rope);
    EXPECT_EQ(semantics.op, ExprOp::ROPE);
    EXPECT_EQ(semantics.rope_sequence_axis, options.sequence_axis);
    EXPECT_EQ(semantics.rope_head_dim_axis, options.head_dim_axis);
    EXPECT_EQ(semantics.rope_rotary_dim, options.rotary_dim);
    EXPECT_DOUBLE_EQ(semantics.rope_base, options.base);
    EXPECT_EQ(semantics.rope_position_offset, options.position_offset);
    EXPECT_EQ(semantics.rope_interleaved, options.interleaved);
    EXPECT_EQ(semantics.rope_inverse, options.inverse);
    EXPECT_EQ(semantics.rope_scaling_kind, options.scaling_kind);
    EXPECT_DOUBLE_EQ(semantics.rope_scaling_factor, options.scaling_factor);
    EXPECT_EQ(semantics.rope_original_max_position_embeddings, options.original_max_position_embeddings);
    EXPECT_DOUBLE_EQ(semantics.rope_attention_factor, options.attention_factor.value());
    EXPECT_DOUBLE_EQ(semantics.rope_yarn_beta_fast, options.yarn_beta_fast);
    EXPECT_DOUBLE_EQ(semantics.rope_yarn_beta_slow, options.yarn_beta_slow);
    EXPECT_DOUBLE_EQ(semantics.rope_llama3_low_freq_factor, options.llama3_low_freq_factor);
    EXPECT_DOUBLE_EQ(semantics.rope_llama3_high_freq_factor, options.llama3_high_freq_factor);
    EXPECT_EQ(semantics.rope_long_rope_short_factors, options.long_rope_short_factors);
    EXPECT_EQ(semantics.rope_long_rope_long_factors, options.long_rope_long_factors);
    ASSERT_TRUE(semantics.output_dtype.has_value());
    ASSERT_TRUE(semantics.compute_dtype.has_value());
    EXPECT_EQ(semantics.output_dtype.value(), options.output_dtype.value());
    EXPECT_EQ(semantics.compute_dtype.value(), options.compute_dtype.value());
    EXPECT_TRUE(semantics.rope_allow_in_place_materialization);

    const PhysicalOutputs outputs = Expression::outputs({{"y", rope}}).physicalOutputs();
    EXPECT_EQ(countLinearBoundaryNodes(outputs), 1u);
    EXPECT_EQ(countInputNodesNamed(outputs, "rope_metadata"), 1u);
    EXPECT_EQ(countNodesOfKind(outputs, ExprOp::CAST), 2u);
    EXPECT_EQ(countNodesOfKind(outputs, ExprOp::ROPE), 1u);

    const ExprNode* physical_rope = nullptr;
    for (const ExprNode& candidate : outputs.expr->nodes) {
        if (candidate.op == ExprOp::ROPE) {
            ASSERT_EQ(physical_rope, nullptr);
            physical_rope = &candidate;
        }
    }
    ASSERT_NE(physical_rope, nullptr);
    ASSERT_LT(physical_rope->rope_position_ids_node, outputs.expr->nodes.size());
    ASSERT_LT(physical_rope->rope_effective_sequence_length_node, outputs.expr->nodes.size());
    const ExprNode& physical_positions = outputs.expr->nodes.at(physical_rope->rope_position_ids_node);
    const ExprNode& physical_length = outputs.expr->nodes.at(physical_rope->rope_effective_sequence_length_node);
    EXPECT_EQ(physical_positions.op, ExprOp::CAST);
    EXPECT_EQ(physical_length.op, ExprOp::CAST);
    EXPECT_EQ(physical_positions.lhs, physical_length.lhs)
        << "Both normalized metadata roles must retain one shared authored metadata producer.";
    EXPECT_EQ(physical_rope->rope_long_rope_short_factors, options.long_rope_short_factors);
    EXPECT_EQ(physical_rope->rope_long_rope_long_factors, options.long_rope_long_factors);
    EXPECT_DOUBLE_EQ(physical_rope->rope_attention_factor, options.attention_factor.value());
}

#ifdef THOR_DEBUG
// PERSISTENT-DAG INVARIANT:
// Natural GELU retains one logical convolution producer. Backward contains zero
// replayed forward CONV2D nodes, and one requested input gradient emits exactly
// one CONV2D_BACKWARD_DATA node.
TEST(ExpressionPersistentDagInvariant, Convolution2dGeluBuildsOneBackwardDataVjpWithoutForwardReplay) {
    const Expression input = Expression::input("input", DataType::FP32, DataType::FP32);
    const Expression filter = Expression::input("filter", DataType::FP32, DataType::FP32);
    ConvolutionSpatial2d spatial;
    spatial.pre_padding_h = 1;
    spatial.post_padding_h = 1;
    spatial.pre_padding_w = 1;
    spatial.post_padding_w = 1;

    const Expression convolution =
        Expression::conv2d(input, filter, spatial, DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", convolution.gelu()}}).physicalOutputs();
    const BackwardBuildResult backward_build = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"input"},
        std::optional<std::string>{"dy"},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"input", {1, 2, 5, 5}},
            {"filter", {3, 2, 3, 3}},
        });
    const PhysicalOutputs& backward = backward_build.outputs;
    EXPECT_FALSE(backward_build.forward_value_requirements.empty());

    EXPECT_EQ(countNodesWithProvenance(
                  backward, ExprOp::CONV2D, ExpressionExecutionProvenance::Forward),
              0u)
        << "Backward must consume retained real-forward state rather than replaying CONV2D.";
    EXPECT_EQ(countNodesWithProvenance(
                  backward, ExprOp::CONV2D_BACKWARD_DATA, ExpressionExecutionProvenance::BackwardGradient),
              1u)
        << "One shared CONV2D producer must emit one input-gradient VJP.";
}

// PERSISTENT-DAG INVARIANT:
// A single authored CONV3D value feeding natural GELU remains one logical value.
// Backward contains zero replayed CONV3D nodes and exactly one
// CONV3D_BACKWARD_DATA node for one requested input gradient.
TEST(ExpressionPersistentDagInvariant, Convolution3dGeluBuildsOneBackwardDataVjpWithoutForwardReplay) {
    const Expression input = Expression::input("input", DataType::FP32, DataType::FP32);
    const Expression filter = Expression::input("filter", DataType::FP32, DataType::FP32);
    ConvolutionSpatial3d spatial;
    spatial.pre_padding_d = 1;
    spatial.post_padding_d = 1;
    spatial.pre_padding_h = 1;
    spatial.post_padding_h = 1;
    spatial.pre_padding_w = 1;
    spatial.post_padding_w = 1;

    const Expression convolution =
        Expression::conv3d(input, filter, spatial, DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", convolution.gelu()}}).physicalOutputs();
    const BackwardBuildResult backward_build = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"input"},
        std::optional<std::string>{"dy"},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"input", {1, 2, 4, 5, 5}},
            {"filter", {3, 2, 3, 3, 3}},
        });
    const PhysicalOutputs& backward = backward_build.outputs;
    EXPECT_FALSE(backward_build.forward_value_requirements.empty());

    EXPECT_EQ(countNodesWithProvenance(
                  backward, ExprOp::CONV3D, ExpressionExecutionProvenance::Forward),
              0u);
    EXPECT_EQ(countNodesWithProvenance(
                  backward, ExprOp::CONV3D_BACKWARD_DATA, ExpressionExecutionProvenance::BackwardGradient),
              1u)
        << "One shared CONV3D producer must emit one input-gradient VJP.";
}

// PERSISTENT-DAG INVARIANT:
// One affine producer feeding natural GELU produces one accumulated affine adjoint
// and exactly two physical/runtime backward MATMULs: dInput and dWeights. Forward
// executes one GEMM, backward replays zero forward GEMMs, and the shared-clear plan
// is submitted exactly once.
TEST(ExpressionPersistentDagInvariant, DefaultGeluFullyConnectedUsesOneProducerVjpPairAtPlanAndRuntime) {
    constexpr uint32_t batch_size = 2;
    constexpr uint32_t input_features = 3;
    constexpr uint32_t output_features = 2;
    constexpr float learning_rate = 0.2f;

    // Retain the numerical control alongside the cardinality assertions: duplicate
    // producer work must never be treated as permission for an incorrect gradient.
    const std::vector<float> input_values = {
        1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f,
    };
    const std::vector<float> weight_values = {
         1.0f,  0.0f,
         0.0f,  1.0f,
        -1.0f, -1.0f,
    };
    const std::vector<float> error_input_values = {
         1.0f, 2.0f,
        -1.0f, 0.5f,
    };
    std::vector<float> activation_adjoint = error_input_values;
    for (float& value : activation_adjoint) {
        value *= 0.5f;
    }

    Api::Network network("persistent_dag_fc_default_gelu_vjp_cardinality");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("input")
                                  .dimensions({input_features})
                                  .dataType(DataType::FP32)
                                  .build();
    Api::GradientRivet input_rivet =
        Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(input_rivet.getFeatureOutput().value())
                                 .numOutputFeatures(output_features)
                                 .hasBias(false)
                                 .weightsDataType(DataType::FP32)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(DataType::FP32)
                                 .build();
    Api::GradientRivet output_rivet =
        Api::GradientRivet::Builder().network(network).tensor(fc.getFeatureOutput().value()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(output_rivet.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();
    std::shared_ptr<Api::Sgd> sgd = Api::Sgd::Builder()
                                        .network(network)
                                        .initialLearningRate(learning_rate)
                                        .decay(0.0f)
                                        .momentum(0.0f)
                                        .build();
    (void)sgd;

    PersistentDagPlacedCustomLayerFixture fixture =
        placePersistentDagSingleCustomLayerNetwork(network, input, output, fc, batch_size);
    ASSERT_NE(fixture.physical_layer, nullptr);
    ASSERT_TRUE(fixture.physical_layer->getGradientUpdateStream().has_value());
    Stream stream = fixture.physical_layer->getStreams()[0];
    Stream gradient_stream = fixture.physical_layer->getGradientUpdateStream().value();
    persistentDagSetParameterTensor(fixture.physical_layer->getParameter("weights"), weight_values, stream);
    stream.synchronize();

    Impl::Tensor feature_input_host(
        persistentDagCpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batch_size, input_features}));
    persistentDagWriteFp32Tensor(feature_input_host, input_values);
    Impl::resetExpressionTestExecutionCounters();
    runPersistentDagForward(fixture, feature_input_host, batch_size);
    const Impl::ExpressionTestExecutionCounters forward_counters = Impl::expressionTestExecutionCounters();
    EXPECT_EQ(forward_counters.matmul.forward, 1u)
        << "The authored affine producer must execute exactly once in forward.";
    EXPECT_EQ(forward_counters.matmul.backward_gradient, 0u);
    persistentDagExpectAllClose(
        persistentDagReadFp32Tensor(fixture.physical_output->getFeatureOutput().value()),
        std::vector<float>(batch_size * output_features, 0.0f),
        1e-6f,
        1e-6f,
        "GELU forward");

    const auto diagnostic_before = fixture.physical_layer->genericSharedBackwardDebugDiagnostic();
    ASSERT_TRUE(diagnostic_before.has_value());
    expectSharedGeluBoundaryPlan(diagnostic_before.value(), "Matmul");
    EXPECT_EQ(diagnostic_before->clearExecutionCount, 0u);
    EXPECT_EQ(diagnostic_before->accumulateExecutionCount, 0u);

    ASSERT_GT(fixture.physical_layer->getErrorInputs().size(), 0u);
    ASSERT_TRUE(fixture.physical_layer->getErrorInputs()[0].has_value());
    ASSERT_GT(fixture.physical_layer->getErrorOutputs().size(), 0u);
    ASSERT_TRUE(fixture.physical_layer->getErrorOutputs()[0].has_value());
    ASSERT_TRUE(fixture.physical_layer->getParameter("weights")->getOptimizer()->getWeightsGradient().has_value());

    Impl::Tensor error_input = fixture.physical_layer->getErrorInputs()[0].value();
    Impl::Tensor error_input_host = error_input.clone(persistentDagCpuPlacement);
    persistentDagWriteFp32Tensor(error_input_host, error_input_values);
    error_input.copyFromAsync(error_input_host, stream);
    stream.synchronize();

    Impl::resetExpressionTestExecutionCounters();
    fixture.physical_layer->backward(error_input, batch_size);

    Impl::Tensor error_output_host =
        persistentDagCopyTensorToCpu(fixture.physical_layer->getErrorOutputs()[0].value(), stream);
    Impl::Tensor weights_gradient_host = persistentDagCopyTensorToCpu(
        fixture.physical_layer->getParameter("weights")->getOptimizer()->getWeightsGradient().value(),
        gradient_stream);
    Impl::Tensor weights_after_host = persistentDagCopyTensorToCpu(
        fixture.physical_layer->getParameter("weights")->getStorage().value(), gradient_stream);
    stream.synchronize();
    gradient_stream.synchronize();

    const std::vector<float> expected_error_output = persistentDagFullyConnectedBackwardErrorReference(
        activation_adjoint, weight_values, batch_size, input_features, output_features);
    const std::vector<float> expected_weights_gradient = persistentDagFullyConnectedWeightGradReference(
        input_values, activation_adjoint, batch_size, input_features, output_features);
    const std::vector<float> expected_weights_after =
        persistentDagSgdUpdatedReference(weight_values, expected_weights_gradient, batch_size, learning_rate);

    persistentDagExpectAllClose(
        persistentDagReadFp32Tensor(error_output_host), expected_error_output, 1e-5f, 1e-5f, "default GELU dInput");
    persistentDagExpectAllClose(
        persistentDagReadFp32Tensor(weights_gradient_host), expected_weights_gradient, 1e-5f, 1e-5f, "default GELU dWeights");
    persistentDagExpectAllClose(
        persistentDagReadFp32Tensor(weights_after_host), expected_weights_after, 1e-5f, 1e-5f, "default GELU weights after");

    const Impl::ExpressionTestExecutionCounters backward_counters = Impl::expressionTestExecutionCounters();
    EXPECT_EQ(backward_counters.matmul.forward, 0u)
        << "Backward must not replay the forward affine GEMM.";
    EXPECT_EQ(backward_counters.matmul.backward_gradient, 2u)
        << "One shared affine producer must execute exactly its dInput/dWeights VJP pair.";

    const auto diagnostic_after = fixture.physical_layer->genericSharedBackwardDebugDiagnostic();
    ASSERT_TRUE(diagnostic_after.has_value());
    EXPECT_EQ(diagnostic_after->clearExecutionCount, 1u)
        << "The shared-clear backward plan must be submitted exactly once.";
    EXPECT_EQ(diagnostic_after->accumulateExecutionCount, 0u);
}

// PERSISTENT-DAG INVARIANT:
// One authored CONV2D producer feeding natural GELU emits exactly two physical
// backward convolutions (dInput and dWeights), executes those two exactly once,
// never replays the forward convolution, and submits the shared-clear plan once.
TEST(ExpressionPersistentDagInvariant, DefaultGeluConvolution2dUsesOneProducerVjpPairAtPlanAndRuntime) {
    constexpr uint32_t batch_size = 2;
    constexpr uint32_t channels = 2;
    constexpr uint32_t height = 5;
    constexpr uint32_t width = 5;
    constexpr uint32_t output_channels = 3;

    std::shared_ptr<Api::Sgd> weights_sgd =
        Api::Sgd::Builder().initialLearningRate(0.001f).decay(0.0f).momentum(0.0f).build();
    std::shared_ptr<Api::Sgd> biases_sgd =
        Api::Sgd::Builder().initialLearningRate(0.001f).decay(0.0f).momentum(0.0f).build();

    Api::Network network("persistent_dag_conv2d_default_gelu_vjp_cardinality");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("input")
                                  .dimensions({channels, height, width})
                                  .dataType(DataType::FP32)
                                  .build();
    Api::GradientRivet input_rivet =
        Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();
    Api::Convolution2d convolution = Api::Convolution2d::Builder()
                                         .network(network)
                                         .featureInput(input_rivet.getFeatureOutput().value())
                                         .numOutputChannels(output_channels)
                                         .filterHeight(3)
                                         .filterWidth(3)
                                         .padding(1, 1, 1, 1)
                                         .hasBias(true)
                                         .weightsOptimizer(weights_sgd)
                                         .biasesOptimizer(biases_sgd)
                                         .build();
    Api::GradientRivet output_rivet = Api::GradientRivet::Builder()
                                          .network(network)
                                          .tensor(convolution.getFeatureOutput().value())
                                          .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(output_rivet.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();

    PersistentDagPlacedCustomLayerFixture fixture =
        placePersistentDagSingleCustomLayerNetwork(network, input, output, convolution, batch_size);
    ASSERT_NE(fixture.physical_layer, nullptr);

    Impl::Tensor feature_input_host(
        persistentDagCpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batch_size, channels, height, width}));
    persistentDagFillFp32Tensor(feature_input_host, 0.125f);
    Impl::resetExpressionTestExecutionCounters();
    runPersistentDagForward(fixture, feature_input_host, batch_size);
    const Impl::ExpressionTestExecutionCounters forward_counters = Impl::expressionTestExecutionCounters();
    EXPECT_EQ(forward_counters.convolution.forward, 1u);
    EXPECT_EQ(forward_counters.convolution.backward_gradient, 0u);

    const auto diagnostic_before = fixture.physical_layer->genericSharedBackwardDebugDiagnostic();
    ASSERT_TRUE(diagnostic_before.has_value());
    expectSharedGeluBoundaryPlan(diagnostic_before.value(), "ConvolutionBackward");
    EXPECT_EQ(diagnostic_before->clearExecutionCount, 0u);
    EXPECT_EQ(diagnostic_before->accumulateExecutionCount, 0u);

    Impl::resetExpressionTestExecutionCounters();
    runPersistentDagOneBackwardAndExpectSinglePlanSubmission(fixture, batch_size);
    const Impl::ExpressionTestExecutionCounters backward_counters = Impl::expressionTestExecutionCounters();
    EXPECT_EQ(backward_counters.convolution.forward, 0u);
    EXPECT_EQ(backward_counters.convolution.backward_gradient, 2u)
        << "One shared convolution producer must execute exactly its dInput/dWeights VJP pair.";
}

// PERSISTENT-DAG INVARIANT:
// One authored CONV3D producer feeding natural GELU emits and executes exactly
// two backward convolutions (dInput and dWeights), with zero forward replay and
// exactly one shared-clear plan submission.
TEST(ExpressionPersistentDagInvariant, DefaultGeluConvolution3dUsesOneProducerVjpPairAtPlanAndRuntime) {
    constexpr uint32_t batch_size = 1;
    constexpr uint32_t channels = 2;
    constexpr uint32_t depth = 4;
    constexpr uint32_t height = 4;
    constexpr uint32_t width = 4;
    constexpr uint32_t output_channels = 3;

    std::shared_ptr<Api::Sgd> weights_sgd =
        Api::Sgd::Builder().initialLearningRate(0.001f).decay(0.0f).momentum(0.0f).build();
    std::shared_ptr<Api::Sgd> biases_sgd =
        Api::Sgd::Builder().initialLearningRate(0.001f).decay(0.0f).momentum(0.0f).build();

    Api::Network network("persistent_dag_conv3d_default_gelu_vjp_cardinality");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("input")
                                  .dimensions({channels, depth, height, width})
                                  .dataType(DataType::FP32)
                                  .build();
    Api::GradientRivet input_rivet =
        Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();
    Api::Convolution3d convolution = Api::Convolution3d::Builder()
                                         .network(network)
                                         .featureInput(input_rivet.getFeatureOutput().value())
                                         .numOutputChannels(output_channels)
                                         .filterDepth(3)
                                         .filterHeight(3)
                                         .filterWidth(3)
                                         .depthPadding(1)
                                         .verticalPadding(1)
                                         .horizontalPadding(1)
                                         .hasBias(true)
                                         .weightsOptimizer(weights_sgd)
                                         .biasesOptimizer(biases_sgd)
                                         .build();
    Api::GradientRivet output_rivet = Api::GradientRivet::Builder()
                                          .network(network)
                                          .tensor(convolution.getFeatureOutput().value())
                                          .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(output_rivet.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();

    PersistentDagPlacedCustomLayerFixture fixture =
        placePersistentDagSingleCustomLayerNetwork(network, input, output, convolution, batch_size);
    ASSERT_NE(fixture.physical_layer, nullptr);

    Impl::Tensor feature_input_host(
        persistentDagCpuPlacement,
        Impl::TensorDescriptor(DataType::FP32, {batch_size, channels, depth, height, width}));
    persistentDagFillFp32Tensor(feature_input_host, 0.125f);
    Impl::resetExpressionTestExecutionCounters();
    runPersistentDagForward(fixture, feature_input_host, batch_size);
    const Impl::ExpressionTestExecutionCounters forward_counters = Impl::expressionTestExecutionCounters();
    EXPECT_EQ(forward_counters.convolution.forward, 1u);
    EXPECT_EQ(forward_counters.convolution.backward_gradient, 0u);

    const auto diagnostic_before = fixture.physical_layer->genericSharedBackwardDebugDiagnostic();
    ASSERT_TRUE(diagnostic_before.has_value());
    expectSharedGeluBoundaryPlan(diagnostic_before.value(), "ConvolutionBackward");
    EXPECT_EQ(diagnostic_before->clearExecutionCount, 0u);
    EXPECT_EQ(diagnostic_before->accumulateExecutionCount, 0u);

    Impl::resetExpressionTestExecutionCounters();
    runPersistentDagOneBackwardAndExpectSinglePlanSubmission(fixture, batch_size);
    const Impl::ExpressionTestExecutionCounters backward_counters = Impl::expressionTestExecutionCounters();
    EXPECT_EQ(backward_counters.convolution.forward, 0u);
    EXPECT_EQ(backward_counters.convolution.backward_gradient, 2u)
        << "One shared convolution producer must execute exactly its dInput/dWeights VJP pair.";
}
#endif
