#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Expression/ExecutionDiagnostics.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                         \
    do {                                                                                                               \
        int cuda_device_count_for_test = 0;                                                                            \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                      \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                  \
            GTEST_SKIP() << "CUDA device is required for Expression forward-replay instrumentation tests.";          \
        }                                                                                                              \
    } while (false)

PhysicalOutputs matmulTanhForward() {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w = Expression::input("w", DataType::FP32, DataType::FP32);
    return Expression::outputs({{"y", Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32).tanh()}})
        .physicalOutputs();
}

PhysicalOutputs affineGeluForward() {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w = Expression::input("w", DataType::FP32, DataType::FP32);
    const Expression bias = Expression::input("bias", DataType::FP32, DataType::FP32);
    const Expression affine = Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32) + bias;
    return Expression::outputs({{"y", affine.gelu()}}).physicalOutputs();
}

std::vector<float> copyFp32ToHost(const Tensor& gpu_tensor, Stream& stream) {
    Tensor cpu = gpu_tensor.clone(TensorPlacement(TensorPlacement::MemDevices::CPU));
    cpu.copyFromAsync(gpu_tensor, stream);
    stream.synchronize();
    const float* values = cpu.getMemPtr<float>();
    return std::vector<float>(values, values + cpu.getTotalNumElements());
}

float cublasLtGeluApprox(float x) {
    constexpr float sqrt_two_over_pi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + std::tanh(sqrt_two_over_pi * (x + 0.044715f * x * x * x)));
}

float cublasLtGeluApproxDerivative(float x) {
    constexpr float sqrt_two_over_pi = 0.7978845608028654f;
    constexpr float cubic = 0.044715f;
    const float x2 = x * x;
    const float tanh_arg = sqrt_two_over_pi * (x + cubic * x * x2);
    const float tanh_value = std::tanh(tanh_arg);
    const float sech2 = 1.0f - tanh_value * tanh_value;
    return 0.5f * (1.0f + tanh_value) +
           0.5f * x * sech2 * sqrt_two_over_pi * (1.0f + 3.0f * cubic * x2);
}


#ifdef THOR_DEBUG
size_t countNodesWithProvenance(const PhysicalOutputs& outputs,
                                ExprOp op,
                                ExpressionExecutionProvenance provenance) {
    size_t count = 0;
    for (const ExprNode& node : outputs.expr->nodes) {
        if (node.op == op && node.execution_provenance == provenance) {
            ++count;
        }
    }
    return count;
}
#endif

}  // namespace

#ifdef THOR_DEBUG
TEST(ExpressionForwardReplayInstrumentation, Br60BLegacyPhysicalOutputsApiRejectsUndisclosedForwardRequirement) {
    const PhysicalOutputs forward = matmulTanhForward();
    try {
        (void)buildBackwardOutputs(
            forward,
            {"x"},
            std::optional<std::string>{"dy"},
            std::unordered_map<std::string, std::vector<uint64_t>>{
                {"x", {2, 3}},
                {"w", {3, 4}},
            });
        FAIL() << "Legacy PhysicalOutputs-only AutoDiff must reject a newly discovered retained-forward requirement.";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("buildBackwardOutputsWithForwardValueRequirements()"), std::string::npos);
        EXPECT_NE(message.find("retained real-forward"), std::string::npos);
    }
}

TEST(ExpressionForwardReplayInstrumentation, Br60BLegacyPhysicalOutputsApiAcceptsExplicitSavedForwardBinding) {
    const PhysicalOutputs forward = matmulTanhForward();
    ASSERT_EQ(forward.outputs.size(), 1U);
    const uint32_t tanh_node = forward.outputs.front().node_idx;
    const std::string saved_name = "saved_tanh_output";

    const PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {2, 3}},
            {"w", {3, 4}},
        },
        false,
        SavedForwardValueInputNames{{tanh_node, saved_name}});

    ASSERT_NE(backward.expr, nullptr);
    EXPECT_EQ(countNodesWithProvenance(
                  backward, ExprOp::MATMUL, ExpressionExecutionProvenance::Forward),
              0U);
    EXPECT_EQ(countNodesWithProvenance(
                  backward, ExprOp::TANH, ExpressionExecutionProvenance::Forward),
              0U);
    EXPECT_EQ(countNodesWithProvenance(
                  backward, ExprOp::MATMUL, ExpressionExecutionProvenance::BackwardGradient),
              1U);
    EXPECT_TRUE(std::any_of(backward.expr->inputs.begin(), backward.expr->inputs.end(), [&](const NamedInput& input) {
        return input.name == saved_name;
    }));
}

TEST(ExpressionForwardReplayInstrumentation, Br60BLegacyPhysicalOutputsApiStillSupportsRootOnlyVjp) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", x * x}}).physicalOutputs();

    EXPECT_NO_THROW({
        const PhysicalOutputs backward = buildBackwardOutputs(
            forward,
            {"x"},
            std::unordered_map<std::string, std::string>{{"y", "dy"}},
            std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
            std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 4}}});
        EXPECT_FALSE(std::any_of(backward.expr->inputs.begin(), backward.expr->inputs.end(), [](const NamedInput& input) {
            return input.name.rfind("__thor_saved_forward_value_", 0) == 0;
        }));
    });
}

TEST(ExpressionForwardReplayInstrumentation, SavedForwardAutodiffDoesNotRetainRootInputs) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", x * x}}).physicalOutputs();

    const BackwardBuildResult result = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 4}}});

    EXPECT_TRUE(result.forward_value_requirements.empty());
    EXPECT_EQ(countNodesWithProvenance(
                  result.outputs, ExprOp::MUL, ExpressionExecutionProvenance::Forward),
              0U);
}

TEST(ExpressionForwardReplayInstrumentation, SavedForwardAutodiffDeclaresTanhOutputInsteadOfReplayingProducer) {
    const PhysicalOutputs forward = matmulTanhForward();
    ASSERT_EQ(forward.outputs.size(), 1U);

    const BackwardBuildResult result = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {2, 3}},
            {"w", {3, 4}},
        });

    ASSERT_NE(result.outputs.expr, nullptr);
    ASSERT_EQ(result.forward_value_requirements.size(), 1U);
    EXPECT_EQ(result.forward_value_requirements.front().forward_node_index, forward.outputs.front().node_idx);
    EXPECT_FALSE(result.forward_value_requirements.front().backward_input_name.empty());

    EXPECT_EQ(countNodesWithProvenance(
                  result.outputs, ExprOp::MATMUL, ExpressionExecutionProvenance::Forward),
              0U);
    EXPECT_EQ(countNodesWithProvenance(
                  result.outputs, ExprOp::TANH, ExpressionExecutionProvenance::Forward),
              0U);
    EXPECT_EQ(countNodesWithProvenance(
                  result.outputs, ExprOp::MATMUL, ExpressionExecutionProvenance::BackwardGradient),
              1U);

    const std::string& saved_name = result.forward_value_requirements.front().backward_input_name;
    auto has_input = [&](const std::string& name) {
        return std::find_if(result.outputs.expr->inputs.begin(), result.outputs.expr->inputs.end(), [&](const NamedInput& input) {
                   return input.name == name;
               }) != result.outputs.expr->inputs.end();
    };
    EXPECT_TRUE(has_input(saved_name));
    EXPECT_TRUE(has_input("w"));
    EXPECT_TRUE(has_input("dy"));
    EXPECT_TRUE(has_input("x"))
        << "Saved-forward inputs replace primal computation, not the public forward-root ABI of a standalone backward equation.";
}

TEST(ExpressionForwardReplayInstrumentation, SavedForwardBooleanRequirementPreservesBooleanStorageDType) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression parameter = Expression::input("parameter", DataType::FP32, DataType::FP32);
    const Expression condition = x > Expression::constantScalar(0.0);
    const PhysicalOutputs forward =
        Expression::outputs({{"y", Expression::where(condition, x, parameter)}}).physicalOutputs();

    const BackwardBuildResult result = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"parameter"},
        std::nullopt,
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {2, 4}},
            {"parameter", {4}},
        });

    ASSERT_EQ(result.forward_value_requirements.size(), 1U);
    const std::string& saved_name = result.forward_value_requirements.front().backward_input_name;
    const auto input_it = std::find_if(
        result.outputs.expr->inputs.begin(), result.outputs.expr->inputs.end(), [&](const NamedInput& input) {
            return input.name == saved_name;
        });
    ASSERT_NE(input_it, result.outputs.expr->inputs.end());

    const auto node_it = std::find_if(result.outputs.expr->nodes.begin(), result.outputs.expr->nodes.end(), [&](const ExprNode& node) {
        return node.op == ExprOp::INPUT && node.input_slot == input_it->slot;
    });
    ASSERT_NE(node_it, result.outputs.expr->nodes.end());
    ASSERT_TRUE(node_it->input_tensor_dtype.has_value());
    ASSERT_TRUE(node_it->output_dtype.has_value());
    EXPECT_EQ(node_it->input_tensor_dtype.value(), DataType::BOOLEAN);
    EXPECT_EQ(node_it->output_dtype.value(), DataType::BOOLEAN);
}

TEST(ExpressionForwardReplayInstrumentation, SavedForwardAutodiffPreservesForwardRootInputAbi) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression scale = Expression::runtimeScalar("scale", DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", x * scale}}).physicalOutputs();

    const BackwardBuildResult result = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 4}}});

    ASSERT_NE(result.outputs.expr, nullptr);
    EXPECT_TRUE(result.forward_value_requirements.empty());

    auto find_input = [&](const std::string& name) {
        return std::find_if(result.outputs.expr->inputs.begin(), result.outputs.expr->inputs.end(), [&](const NamedInput& input) {
            return input.name == name;
        });
    };
    EXPECT_NE(find_input("x"), result.outputs.expr->inputs.end())
        << "Backward equations preserve the original forward-root ABI even when a particular VJP does not read that root.";
    const auto scale_input = find_input("scale");
    ASSERT_NE(scale_input, result.outputs.expr->inputs.end());
    EXPECT_EQ(scale_input->kind, NamedInput::Kind::RuntimeScalarFp32)
        << "Preserving the forward-root ABI must also preserve runtime-scalar input kinds.";
    EXPECT_NE(find_input("dy"), result.outputs.expr->inputs.end());
}

TEST(ExpressionForwardReplayInstrumentation, SavedForwardAutodiffRebuildsViewsButRetainsMaterializedAncestor) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w = Expression::input("w", DataType::FP32, DataType::FP32);
    const Expression product = Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", product.reshape({8}).ln()}}).physicalOutputs();

    uint32_t matmul_node = UINT32_MAX;
    for (uint32_t i = 0; i < forward.expr->nodes.size(); ++i) {
        if (forward.expr->nodes[i].op == ExprOp::MATMUL) {
            matmul_node = i;
            break;
        }
    }
    ASSERT_NE(matmul_node, UINT32_MAX);

    const BackwardBuildResult result = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {2, 3}},
            {"w", {3, 4}},
        });

    ASSERT_EQ(result.forward_value_requirements.size(), 1U);
    EXPECT_EQ(result.forward_value_requirements.front().forward_node_index, matmul_node)
        << "RESHAPE is metadata-only and should be reconstructed around the saved MATMUL value.";
    EXPECT_EQ(countNodesWithProvenance(
                  result.outputs, ExprOp::MATMUL, ExpressionExecutionProvenance::Forward),
              0U);
    EXPECT_GE(countNodesWithProvenance(
                  result.outputs, ExprOp::RESHAPE, ExpressionExecutionProvenance::Forward),
              1U);
}

TEST(ExpressionForwardReplayInstrumentation, SavedForwardAutodiffUsesStableExplicitBindingName) {
    const PhysicalOutputs forward = matmulTanhForward();
    ASSERT_EQ(forward.outputs.size(), 1U);

    const SavedForwardValueInputNames saved_forward_values = {
        {forward.outputs.front().node_idx, "retained_y"},
    };
    const BackwardBuildResult result = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {2, 3}},
            {"w", {3, 4}},
        },
        false,
        saved_forward_values);

    ASSERT_EQ(result.forward_value_requirements.size(), 1U);
    EXPECT_EQ(result.forward_value_requirements.front().forward_node_index, forward.outputs.front().node_idx);
    EXPECT_EQ(result.forward_value_requirements.front().backward_input_name, "retained_y");
}

TEST(ExpressionForwardReplayInstrumentation, SavedForwardAutodiffDeclaresReductionOutputInsteadOfReplayingReduction) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward =
        Expression::outputs({{"y", x.reduce_norm2({1}, {1}, DataType::FP32)}}).physicalOutputs();

    const BackwardBuildResult result = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 4}}});

    ASSERT_EQ(result.forward_value_requirements.size(), 1U);
    EXPECT_EQ(result.forward_value_requirements.front().forward_node_index, forward.outputs.front().node_idx);
    EXPECT_EQ(countNodesWithProvenance(
                  result.outputs, ExprOp::REDUCE_NORM2, ExpressionExecutionProvenance::Forward),
              0U);
}

TEST(ExpressionForwardReplayInstrumentation, SavedForwardBindingUsesStableInputWithoutReplay) {
    const PhysicalOutputs forward = matmulTanhForward();
    ASSERT_EQ(forward.outputs.size(), 1U);

    const SavedForwardValueInputNames saved_forward_values = {
        {forward.outputs.front().node_idx, "saved_y"},
    };
    const PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {2, 3}},
            {"w", {3, 4}},
        },
        false,
        saved_forward_values);

    EXPECT_EQ(countNodesWithProvenance(
                  backward, ExprOp::MATMUL, ExpressionExecutionProvenance::Forward),
              0U);
    EXPECT_EQ(countNodesWithProvenance(
                  backward, ExprOp::TANH, ExpressionExecutionProvenance::Forward),
              0U);
    EXPECT_EQ(countNodesWithProvenance(
                  backward, ExprOp::MATMUL, ExpressionExecutionProvenance::BackwardGradient),
              1U);
}

TEST(ExpressionForwardReplayInstrumentation, OutputDependentReductionBecomesSavedForwardInput) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", x.reduce_norm2({1}, {1}, DataType::FP32)}}).physicalOutputs();
    const BackwardBuildResult build = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::optional<std::string>{"dy"},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 4}}});
    const PhysicalOutputs& backward = build.outputs;

    EXPECT_EQ(countNodesWithProvenance(
                  backward, ExprOp::REDUCE_NORM2, ExpressionExecutionProvenance::Forward),
              0U);
    ASSERT_EQ(build.forward_value_requirements.size(), 1U);
    EXPECT_EQ(build.forward_value_requirements.front().forward_node_index, forward.outputs.front().node_idx);
}

TEST(ExpressionForwardReplayInstrumentation, ReduceProdOutputBecomesSavedForwardInput) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", x.reduce_prod({1}, {1})}}).physicalOutputs();
    const BackwardBuildResult build = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::optional<std::string>{"dy"},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 4}}});
    const PhysicalOutputs& backward = build.outputs;

    EXPECT_EQ(countNodesWithProvenance(
                  backward, ExprOp::REDUCE_PROD, ExpressionExecutionProvenance::Forward),
              0U);
    ASSERT_EQ(build.forward_value_requirements.size(), 1U);
    EXPECT_EQ(build.forward_value_requirements.front().forward_node_index, forward.outputs.front().node_idx);
}

TEST(ExpressionForwardReplayInstrumentation, ConvolutionGeluUsesSavedForwardValuesWithoutReplay) {
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

    // Exact GELU still contains two logical references to the convolution
    // producer, but BR6.0A turns any computed primal dependency into a retained
    // forward input. No CONV2D producer may be cloned into the backward graph.
    EXPECT_EQ(countNodesWithProvenance(
                  backward, ExprOp::CONV2D, ExpressionExecutionProvenance::Forward),
              0U);
    EXPECT_EQ(countNodesWithProvenance(
                  backward, ExprOp::CONV2D_BACKWARD_DATA, ExpressionExecutionProvenance::BackwardGradient),
              2U);
}


TEST(ExpressionForwardReplayInstrumentation, Convolution3dGeluUsesSavedForwardValuesWithoutReplay) {
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

    // See the 2D case above: duplicated logical references may remain, but the
    // forward convolution itself is never reconstructed during autodiff.
    EXPECT_EQ(countNodesWithProvenance(
                  backward, ExprOp::CONV3D, ExpressionExecutionProvenance::Forward),
              0U);
    EXPECT_EQ(countNodesWithProvenance(
                  backward, ExprOp::CONV3D_BACKWARD_DATA, ExpressionExecutionProvenance::BackwardGradient),
              2U);
}

TEST(ExpressionForwardReplayInstrumentation, PhysicalNorm2BackwardConsumesRetainedOutputWithoutReplay) {
    REQUIRE_CUDA_DEVICE();

    const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward =
        Expression::outputs({{"y", x_expr.reduce_norm2({1}, {1}, DataType::FP32)}}).physicalOutputs();
    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 4}}});
    ASSERT_EQ(backward.forward_value_requirements.size(), 1U);

    Stream stream(0);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {2, 4}));
    Tensor dy(gpu, TensorDescriptor(DataType::FP32, {2}));
    x.fill(0.5, stream);
    dy.fill(1.0, stream);
    stream.synchronize();

    const ForwardValueRequirement& requirement = backward.forward_value_requirements.front();
    FusedEquation forward_equation = FusedEquation::compile(forward, 0);
    StampedExecutionPlan forward_plan =
        forward_equation.stampRetainingForwardValues({requirement.forward_node_index}, {{"x", x}}, stream);
    forward_plan.run();
    stream.synchronize();

    FusedEquation backward_equation = FusedEquation::compile(backward.outputs, 0);
    StampedExecutionPlan backward_plan = backward_equation.stamp(
        {{"x", x},
         {"dy", dy},
         {requirement.backward_input_name, forward_plan.retainedForwardValue(requirement.forward_node_index)}},
        stream);
    resetExpressionTestExecutionCounters();
    backward_plan.run();
    stream.synchronize();

    const ExpressionTestExecutionCounters counters = expressionTestExecutionCounters();
    EXPECT_EQ(counters.reduction.forward, 0U);
}

TEST(ExpressionForwardReplayInstrumentation, PhysicalSoftmaxBackwardConsumesRetainedOutputWithoutReplay) {
    REQUIRE_CUDA_DEVICE();

    const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", x_expr.softmax()}}).physicalOutputs();
    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 4}}});
    ASSERT_EQ(backward.forward_value_requirements.size(), 1U);

    Stream stream(0);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {2, 4}));
    Tensor dy(gpu, TensorDescriptor(DataType::FP32, {2, 4}));
    x.fill(0.5, stream);
    dy.fill(1.0, stream);
    stream.synchronize();

    const ForwardValueRequirement& requirement = backward.forward_value_requirements.front();
    FusedEquation forward_equation = FusedEquation::compile(forward, 0);
    StampedExecutionPlan forward_plan =
        forward_equation.stampRetainingForwardValues({requirement.forward_node_index}, {{"x", x}}, stream);
    forward_plan.run();
    stream.synchronize();

    FusedEquation backward_equation = FusedEquation::compile(backward.outputs, 0);
    StampedExecutionPlan backward_plan = backward_equation.stamp(
        {{"x", x},
         {"dy", dy},
         {requirement.backward_input_name, forward_plan.retainedForwardValue(requirement.forward_node_index)}},
        stream);
    resetExpressionTestExecutionCounters();
    backward_plan.run();
    stream.synchronize();

    const ExpressionTestExecutionCounters counters = expressionTestExecutionCounters();
    EXPECT_EQ(counters.softmax.forward, 0U);
}

TEST(ExpressionForwardReplayInstrumentation, Br2RetainedForwardPlanFeedsBackwardRequirementWithoutReplay) {
    REQUIRE_CUDA_DEVICE();

    const PhysicalOutputs forward = matmulTanhForward();
    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {2, 3}},
            {"w", {3, 4}},
        });
    ASSERT_EQ(backward.forward_value_requirements.size(), 1U);

    Stream stream(0);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {2, 3}));
    Tensor w(gpu, TensorDescriptor(DataType::FP32, {3, 4}));
    Tensor dy(gpu, TensorDescriptor(DataType::FP32, {2, 4}));
    x.fill(0.25, stream);
    w.fill(0.5, stream);
    dy.fill(1.0, stream);
    stream.synchronize();

    std::vector<uint32_t> retained_nodes;
    for (const ForwardValueRequirement& requirement : backward.forward_value_requirements) {
        retained_nodes.push_back(requirement.forward_node_index);
    }

    FusedEquation forward_equation = FusedEquation::compile(forward, 0);
    StampedExecutionPlan forward_plan =
        forward_equation.stampRetainingForwardValues(retained_nodes, {{"x", x}, {"w", w}}, stream);

    const std::vector<std::string> public_output_names = forward_plan.outputNames();
    ASSERT_EQ(public_output_names.size(), 1U);
    EXPECT_EQ(public_output_names.front(), "y")
        << "Retained primal values are internal execution state, not additional public layer outputs.";
    EXPECT_EQ(forward_plan.retainedForwardValues().size(), retained_nodes.size());
    for (uint32_t retained_node : retained_nodes) {
        EXPECT_TRUE(forward_plan.hasRetainedForwardValue(retained_node));
    }
    ASSERT_EQ(retained_nodes.size(), 1U);
    EXPECT_TRUE(forward_plan.retainedForwardValue(retained_nodes.front()) == forward_plan.output("y"))
        << "When the required primal is already the public forward output, BR2 must reuse that producer tensor rather than creating a hidden duplicate.";

    forward_plan.run();
    stream.synchronize();

    std::unordered_map<std::string, Tensor> backward_inputs{{"x", x}, {"w", w}, {"dy", dy}};
    for (const ForwardValueRequirement& requirement : backward.forward_value_requirements) {
        backward_inputs.emplace(requirement.backward_input_name,
                                forward_plan.retainedForwardValue(requirement.forward_node_index));
    }

    FusedEquation backward_equation = FusedEquation::compile(backward.outputs, 0);
    StampedExecutionPlan backward_plan = backward_equation.stamp(backward_inputs, stream);

    resetExpressionTestExecutionCounters();
    backward_plan.run();
    stream.synchronize();

    const ExpressionTestExecutionCounters counters = expressionTestExecutionCounters();
    EXPECT_EQ(counters.matmul.backward_gradient, 1U);
    EXPECT_EQ(counters.matmul.forward, 0U)
        << "A BR1 forward-value requirement bound to BR2 retained state must not launch a forward MATMUL during backward.";
}

TEST(ExpressionForwardReplayInstrumentation, Br5SavedForwardAutodiffRefusesFusedActivationWithoutForwardProvider) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w = Expression::input("w", DataType::FP32, DataType::FP32);
    PhysicalOutputs forward =
        Expression::outputs({{"y", Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32)}})
            .physicalOutputs();
    ASSERT_EQ(forward.outputs.size(), 1U);
    ExprNode& fused_output = forward.expr->nodes.at(forward.outputs.front().node_idx);
    ASSERT_EQ(fused_output.op, ExprOp::MATMUL);
    fused_output.matmul_epilogue = MatmulEpilogue::Relu;
    fused_output.matmul_forward_epilogue_aux = false;

    EXPECT_THROW(
        (void)buildBackwardOutputsWithForwardValueRequirements(
            forward,
            {"x"},
            std::unordered_map<std::string, std::string>{{"y", "dy"}},
            std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
            std::unordered_map<std::string, std::vector<uint64_t>>{
                {"x", {2, 3}},
                {"w", {3, 4}},
            }),
        std::runtime_error)
        << "BR5 saved-forward autodiff must never silently replay a fused affine preactivation.";
}

TEST(ExpressionForwardReplayInstrumentation, Br5ExpConsumesRetainedForwardOutputWithoutReplay) {
    REQUIRE_CUDA_DEVICE();

    const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", x_expr.exp()}}).physicalOutputs();
    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 4}}});

    ASSERT_EQ(backward.forward_value_requirements.size(), 1U);
    EXPECT_EQ(backward.forward_value_requirements.front().kind, ForwardValueRequirementKind::NodeOutput);
    EXPECT_EQ(backward.forward_value_requirements.front().forward_node_index, forward.outputs.front().node_idx);

    Stream stream(0);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {2, 4}));
    Tensor dy(gpu, TensorDescriptor(DataType::FP32, {2, 4}));
    x.fill(0.25, stream);
    dy.fill(1.0, stream);
    stream.synchronize();

    FusedEquation forward_equation = FusedEquation::compile(forward, 0);
    StampedExecutionPlan forward_plan = forward_equation.stampRetainingForwardValues(
        {backward.forward_value_requirements.front().forward_node_index}, {{"x", x}}, stream);
    forward_plan.run();
    stream.synchronize();

    std::unordered_map<std::string, Tensor> backward_inputs{{"x", x}, {"dy", dy}};
    backward_inputs.emplace(backward.forward_value_requirements.front().backward_input_name,
                            forward_plan.retainedForwardValue(backward.forward_value_requirements.front().forward_node_index));
    StampedExecutionPlan backward_plan = FusedEquation::compile(backward.outputs, 0).stamp(backward_inputs, stream);

    resetExpressionTestExecutionCounters();
    backward_plan.run();
    stream.synchronize();
    EXPECT_EQ(expressionTestExecutionCounters().fused_kernel.forward, 0U)
        << "EXP backward must consume the already-produced forward EXP output.";
}

TEST(ExpressionForwardReplayInstrumentation, Br5Expm1UsesRetainedOutputInsteadOfLaunchingExpInBackward) {
    const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", x_expr.expm1()}}).physicalOutputs();
    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 4}}});

    ASSERT_EQ(backward.forward_value_requirements.size(), 1U);
    EXPECT_EQ(backward.forward_value_requirements.front().kind, ForwardValueRequirementKind::NodeOutput);
    EXPECT_EQ(backward.forward_value_requirements.front().forward_node_index, forward.outputs.front().node_idx);
    EXPECT_EQ(std::count_if(backward.outputs.expr->nodes.begin(),
                            backward.outputs.expr->nodes.end(),
                            [](const ExprNode& node) { return node.op == ExprOp::EXP; }),
              0)
        << "EXPM1 backward should use expm1(x)+1 from retained forward output rather than recomputing exp(x).";
}

TEST(ExpressionForwardReplayInstrumentation, Br5SoftmaxConsumesRetainedForwardOutputWithoutReplay) {
    REQUIRE_CUDA_DEVICE();

    const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", x_expr.softmax()}}).physicalOutputs();
    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 4}}});

    ASSERT_EQ(backward.forward_value_requirements.size(), 1U);
    EXPECT_EQ(backward.forward_value_requirements.front().kind, ForwardValueRequirementKind::NodeOutput);
    EXPECT_EQ(backward.forward_value_requirements.front().forward_node_index, forward.outputs.front().node_idx);

    Stream stream(0);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {2, 4}));
    Tensor dy(gpu, TensorDescriptor(DataType::FP32, {2, 4}));
    x.fill(0.25, stream);
    dy.fill(1.0, stream);
    stream.synchronize();

    FusedEquation forward_equation = FusedEquation::compile(forward, 0);
    StampedExecutionPlan forward_plan = forward_equation.stampRetainingForwardValues(
        {backward.forward_value_requirements.front().forward_node_index}, {{"x", x}}, stream);
    resetExpressionTestExecutionCounters();
    forward_plan.run();
    stream.synchronize();
    ExpressionTestExecutionCounters counters = expressionTestExecutionCounters();
    EXPECT_EQ(counters.softmax.forward, 1U);

    std::unordered_map<std::string, Tensor> backward_inputs{{"x", x}, {"dy", dy}};
    backward_inputs.emplace(backward.forward_value_requirements.front().backward_input_name,
                            forward_plan.retainedForwardValue(backward.forward_value_requirements.front().forward_node_index));
    StampedExecutionPlan backward_plan = FusedEquation::compile(backward.outputs, 0).stamp(backward_inputs, stream);

    resetExpressionTestExecutionCounters();
    backward_plan.run();
    stream.synchronize();
    counters = expressionTestExecutionCounters();
    EXPECT_EQ(counters.softmax.forward, 0U);
}

TEST(ExpressionForwardReplayInstrumentation, Br5LogSoftmaxUsesRetainedOutputWithoutSecondSoftmax) {
    REQUIRE_CUDA_DEVICE();

    const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", x_expr.logSoftmax()}}).physicalOutputs();
    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 4}}});

    ASSERT_EQ(backward.forward_value_requirements.size(), 1U);
    EXPECT_EQ(backward.forward_value_requirements.front().kind, ForwardValueRequirementKind::NodeOutput);
    EXPECT_EQ(backward.forward_value_requirements.front().forward_node_index, forward.outputs.front().node_idx);
    EXPECT_EQ(std::count_if(backward.outputs.expr->nodes.begin(),
                            backward.outputs.expr->nodes.end(),
                            [](const ExprNode& node) { return node.op == ExprOp::SOFTMAX; }),
              0)
        << "LOG_SOFTMAX backward should recover probabilities from exp(retained log-softmax output), not launch Softmax again.";

    Stream stream(0);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {2, 4}));
    Tensor dy(gpu, TensorDescriptor(DataType::FP32, {2, 4}));
    x.fill(0.25, stream);
    dy.fill(1.0, stream);
    stream.synchronize();

    FusedEquation forward_equation = FusedEquation::compile(forward, 0);
    StampedExecutionPlan forward_plan = forward_equation.stampRetainingForwardValues(
        {backward.forward_value_requirements.front().forward_node_index}, {{"x", x}}, stream);
    forward_plan.run();
    stream.synchronize();

    std::unordered_map<std::string, Tensor> backward_inputs{{"x", x}, {"dy", dy}};
    backward_inputs.emplace(backward.forward_value_requirements.front().backward_input_name,
                            forward_plan.retainedForwardValue(backward.forward_value_requirements.front().forward_node_index));
    StampedExecutionPlan backward_plan = FusedEquation::compile(backward.outputs, 0).stamp(backward_inputs, stream);

    resetExpressionTestExecutionCounters();
    backward_plan.run();
    stream.synchronize();
    const ExpressionTestExecutionCounters counters = expressionTestExecutionCounters();
    EXPECT_EQ(counters.softmax.backward_gradient, 0U);
    EXPECT_EQ(counters.softmax.forward, 0U);
}

TEST(ExpressionForwardReplayInstrumentation, Br5Norm2ConsumesRetainedReductionOutputWithoutReplay) {
    REQUIRE_CUDA_DEVICE();

    const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward =
        Expression::outputs({{"y", x_expr.reduce_norm2({1}, {1}, DataType::FP32)}}).physicalOutputs();
    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 4}}});

    ASSERT_EQ(backward.forward_value_requirements.size(), 1U);
    EXPECT_EQ(backward.forward_value_requirements.front().kind, ForwardValueRequirementKind::NodeOutput);
    EXPECT_EQ(backward.forward_value_requirements.front().forward_node_index, forward.outputs.front().node_idx);

    Stream stream(0);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {2, 4}));
    Tensor dy(gpu, TensorDescriptor(DataType::FP32, {2}));
    x.fill(0.5, stream);
    dy.fill(1.0, stream);
    stream.synchronize();

    FusedEquation forward_equation = FusedEquation::compile(forward, 0);
    StampedExecutionPlan forward_plan = forward_equation.stampRetainingForwardValues(
        {backward.forward_value_requirements.front().forward_node_index}, {{"x", x}}, stream);
    forward_plan.run();
    stream.synchronize();

    std::unordered_map<std::string, Tensor> backward_inputs{{"x", x}, {"dy", dy}};
    backward_inputs.emplace(backward.forward_value_requirements.front().backward_input_name,
                            forward_plan.retainedForwardValue(backward.forward_value_requirements.front().forward_node_index));
    StampedExecutionPlan backward_plan = FusedEquation::compile(backward.outputs, 0).stamp(backward_inputs, stream);

    resetExpressionTestExecutionCounters();
    backward_plan.run();
    stream.synchronize();
    const ExpressionTestExecutionCounters counters = expressionTestExecutionCounters();
    EXPECT_EQ(counters.reduction.forward, 0U);
}

TEST(ExpressionForwardReplayInstrumentation, Br5ReduceProdConsumesRetainedReductionOutputWithoutReplay) {
    REQUIRE_CUDA_DEVICE();

    const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", x_expr.reduce_prod({1}, {1})}}).physicalOutputs();
    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 4}}});

    ASSERT_EQ(backward.forward_value_requirements.size(), 1U);
    EXPECT_EQ(backward.forward_value_requirements.front().kind, ForwardValueRequirementKind::NodeOutput);
    EXPECT_EQ(backward.forward_value_requirements.front().forward_node_index, forward.outputs.front().node_idx);

    Stream stream(0);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {2, 4}));
    Tensor dy(gpu, TensorDescriptor(DataType::FP32, {2}));
    x.fill(0.5, stream);
    dy.fill(1.0, stream);
    stream.synchronize();

    FusedEquation forward_equation = FusedEquation::compile(forward, 0);
    StampedExecutionPlan forward_plan = forward_equation.stampRetainingForwardValues(
        {backward.forward_value_requirements.front().forward_node_index}, {{"x", x}}, stream);
    forward_plan.run();
    stream.synchronize();

    std::unordered_map<std::string, Tensor> backward_inputs{{"x", x}, {"dy", dy}};
    backward_inputs.emplace(backward.forward_value_requirements.front().backward_input_name,
                            forward_plan.retainedForwardValue(backward.forward_value_requirements.front().forward_node_index));
    StampedExecutionPlan backward_plan = FusedEquation::compile(backward.outputs, 0).stamp(backward_inputs, stream);

    resetExpressionTestExecutionCounters();
    backward_plan.run();
    stream.synchronize();
    const ExpressionTestExecutionCounters counters = expressionTestExecutionCounters();
    EXPECT_EQ(counters.reduction.forward, 0U);
}

TEST(ExpressionForwardReplayInstrumentation, Br2RetainsIntermediateWithoutAddingPublicOutput) {
    REQUIRE_CUDA_DEVICE();

    const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w_expr = Expression::input("w", DataType::FP32, DataType::FP32);
    const Expression product = Expression::matmul(x_expr, w_expr, false, false, DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", product.tanh()}}).physicalOutputs();

    uint32_t matmul_node = UINT32_MAX;
    for (uint32_t i = 0; i < forward.expr->nodes.size(); ++i) {
        if (forward.expr->nodes[i].op == ExprOp::MATMUL) {
            matmul_node = i;
            break;
        }
    }
    ASSERT_NE(matmul_node, UINT32_MAX);

    Stream stream(0);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {2, 3}));
    Tensor w(gpu, TensorDescriptor(DataType::FP32, {3, 4}));
    x.fill(0.25, stream);
    w.fill(0.5, stream);
    stream.synchronize();

    FusedEquation forward_equation = FusedEquation::compile(forward, 0);
    StampedExecutionPlan forward_plan =
        forward_equation.stampRetainingForwardValues({matmul_node}, {{"x", x}, {"w", w}}, stream);

    EXPECT_EQ(forward_plan.outputNames().size(), 1U);
    ASSERT_TRUE(forward_plan.hasRetainedForwardValue(matmul_node));
    EXPECT_EQ(forward_plan.retainedForwardValue(matmul_node).getDimensions(), (std::vector<uint64_t>{2, 4}));

    resetExpressionTestExecutionCounters();
    forward_plan.run();
    stream.synchronize();
    const ExpressionTestExecutionCounters counters = expressionTestExecutionCounters();
    EXPECT_EQ(counters.matmul.forward, 1U)
        << "Retaining an already-materialized MATMUL value must not introduce another MATMUL launch.";
}

TEST(ExpressionForwardReplayInstrumentation, Br3RetainedMatmulDescendantPreventsGemmLoweringDuplicateProducer) {
    REQUIRE_CUDA_DEVICE();

    const Expression x_expr = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w_expr = Expression::input("w", DataType::FP32, DataType::FP32);
    const Expression bias_expr = Expression::input("bias", DataType::FP32, DataType::FP32);
    const Expression product = Expression::matmul(x_expr, w_expr, false, false, DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", product + bias_expr}}).physicalOutputs();

    uint32_t matmul_node = UINT32_MAX;
    for (uint32_t i = 0; i < forward.expr->nodes.size(); ++i) {
        if (forward.expr->nodes[i].op == ExprOp::MATMUL) {
            matmul_node = i;
            break;
        }
    }
    ASSERT_NE(matmul_node, UINT32_MAX);

    Stream stream(0);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {2, 3}));
    Tensor w(gpu, TensorDescriptor(DataType::FP32, {3, 4}));
    Tensor bias(gpu, TensorDescriptor(DataType::FP32, {4}));
    x.fill(0.25, stream);
    w.fill(0.5, stream);
    bias.fill(0.1, stream);
    stream.synchronize();

    FusedEquation forward_equation = FusedEquation::compile(forward, 0);
    StampedExecutionPlan forward_plan =
        forward_equation.stampRetainingForwardValues({matmul_node}, {{"x", x}, {"w", w}, {"bias", bias}}, stream);

    const std::vector<std::string> stage_kinds = forward_plan.stageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1)
        << "Retaining the MATMUL result must prevent matmul+bias lowering from creating a second GEMM producer.";

    resetExpressionTestExecutionCounters();
    forward_plan.run();
    stream.synchronize();
    const ExpressionTestExecutionCounters counters = expressionTestExecutionCounters();
    EXPECT_EQ(counters.matmul.forward, 1U)
        << "BR3 must prefer one MATMUL plus pointwise bias over a retained MATMUL plus duplicate fused GEMM.";
}

TEST(ExpressionForwardReplayInstrumentation, Br3RetainedGeluPrerequisitesPreventActivationEpilogueDuplicateProducer) {
    REQUIRE_CUDA_DEVICE();

    const PhysicalOutputs forward = affineGeluForward();
    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {2, 3}},
            {"w", {3, 4}},
            {"bias", {4}},
        });
    ASSERT_FALSE(backward.forward_value_requirements.empty());

    std::vector<uint32_t> retained_nodes;
    retained_nodes.reserve(backward.forward_value_requirements.size());
    for (const ForwardValueRequirement& requirement : backward.forward_value_requirements) {
        retained_nodes.push_back(requirement.forward_node_index);
    }
    std::sort(retained_nodes.begin(), retained_nodes.end());
    retained_nodes.erase(std::unique(retained_nodes.begin(), retained_nodes.end()), retained_nodes.end());

    Stream stream(0);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {2, 3}));
    Tensor w(gpu, TensorDescriptor(DataType::FP32, {3, 4}));
    Tensor bias(gpu, TensorDescriptor(DataType::FP32, {4}));
    x.fill(0.25, stream);
    w.fill(0.5, stream);
    bias.fill(0.1, stream);
    stream.synchronize();

    FusedEquation forward_equation = FusedEquation::compile(forward, 0);
    StampedExecutionPlan forward_plan =
        forward_equation.stampRetainingForwardValues(retained_nodes, {{"x", x}, {"w", w}, {"bias", bias}}, stream);

    const std::vector<std::string> stage_kinds = forward_plan.stageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1)
        << "A retained GELU prerequisite must disable an epilogue rewrite that would otherwise require a second affine GEMM.";
    EXPECT_EQ(forward_plan.retainedForwardValues().size(), retained_nodes.size());

    resetExpressionTestExecutionCounters();
    forward_plan.run();
    stream.synchronize();
    const ExpressionTestExecutionCounters counters = expressionTestExecutionCounters();
    EXPECT_EQ(counters.matmul.forward, 1U)
        << "Before BR4 supplies a cuBLASLt forward auxiliary value, FC-style GELU retention must execute exactly one affine GEMM.";
}

TEST(ExpressionForwardReplayInstrumentation, Br4FusedGeluAuxFeedsBackwardWithoutAffineReplay) {
    REQUIRE_CUDA_DEVICE();

    // cuBLASLt's GELU_AUX leading dimension is expressed in the backend's
    // column-major view of Thor's public row-major output, so an output width
    // divisible by eight satisfies the documented AUX_LD contract.
    constexpr uint64_t batch = 2;
    constexpr uint64_t input_features = 8;
    constexpr uint64_t output_features = 8;
    constexpr float x_value = 0.25f;
    constexpr float w_value = 0.5f;
    constexpr float bias_value = 0.1f;

    Stream stream(0);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {batch, input_features}));
    Tensor w(gpu, TensorDescriptor(DataType::FP32, {input_features, output_features}));
    Tensor bias(gpu, TensorDescriptor(DataType::FP32, {output_features}));
    Tensor dy(gpu, TensorDescriptor(DataType::FP32, {batch, output_features}));
    x.fill(x_value, stream);
    w.fill(w_value, stream);
    bias.fill(bias_value, stream);
    dy.fill(1.0, stream);
    stream.synchronize();

    FusedEquation forward_equation = FusedEquation::compile(affineGeluForward(), 0);
    const std::unordered_map<std::string, Tensor> forward_inputs{{"x", x}, {"w", w}, {"bias", bias}};
    const PhysicalOutputs training_forward =
        forward_equation.physicalOutputsForTrainingBackward(forward_inputs);
    ASSERT_EQ(training_forward.outputs.size(), 1U);
    const uint32_t y_node = training_forward.outputs.front().node_idx;
    ASSERT_LT(y_node, training_forward.expr->nodes.size());
    const ExprNode& fused_y = training_forward.expr->nodes[y_node];
    if (!((fused_y.op == ExprOp::MATMUL || fused_y.op == ExprOp::GEMM) &&
          fused_y.matmul_epilogue == MatmulEpilogue::Gelu && fused_y.matmul_forward_epilogue_aux)) {
        GTEST_SKIP() << "This GPU/cuBLASLt combination has no qualified GELU_AUX algorithm for the BR4 test shape.";
    }

    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        training_forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {batch, input_features}},
            {"w", {input_features, output_features}},
            {"bias", {output_features}},
        });

    ASSERT_EQ(backward.forward_value_requirements.size(), 1U);
    const ForwardValueRequirement& aux_requirement = backward.forward_value_requirements.front();
    EXPECT_EQ(aux_requirement.forward_node_index, y_node);
    EXPECT_EQ(aux_requirement.kind, ForwardValueRequirementKind::MatmulEpilogueAux);
    EXPECT_FALSE(aux_requirement.backward_input_name.empty());

    // Retain both the public post-GELU value and the backend preactivation at
    // the same logical node. This is the shape used when a fused CustomLoss
    // needs predictions while GELU backward needs MatmulEpilogueAux; the two
    // requirement kinds must remain distinct and must not add another producer.
    StampedExecutionPlan forward_plan = forward_equation.stampRetainingForwardValues(
        {y_node}, {y_node}, forward_inputs, stream);
    EXPECT_TRUE(forward_plan.hasRetainedForwardEpilogueAux(y_node));
    EXPECT_TRUE(forward_plan.hasRetainedForwardValue(y_node));
    EXPECT_EQ(forward_plan.retainedForwardValues().size(), 1U);
    EXPECT_EQ(forward_plan.retainedForwardEpilogueAuxValues().size(), 1U);
    EXPECT_TRUE(forward_plan.retainedForwardValue(y_node) == forward_plan.getFinalOutputs().at("y"))
        << "NodeOutput retention at a public fused GELU output must reuse the post-GELU producer tensor.";
    EXPECT_FALSE(forward_plan.retainedForwardValue(y_node) == forward_plan.retainedForwardEpilogueAux(y_node))
        << "NodeOutput and MatmulEpilogueAux at the same logical node are semantically distinct saved values.";
    const std::vector<std::string> forward_stage_kinds = forward_plan.stageKindNames();
    EXPECT_EQ(std::count(forward_stage_kinds.begin(), forward_stage_kinds.end(), "Matmul"), 1)
        << "BR4 GELU_AUX must come from the same affine GEMM that produces the forward output.";

    resetExpressionTestExecutionCounters();
    forward_plan.run();
    stream.synchronize();
    ExpressionTestExecutionCounters counters = expressionTestExecutionCounters();
    EXPECT_EQ(counters.matmul.forward, 1U);

    const float expected_preactivation =
        static_cast<float>(input_features) * x_value * w_value + bias_value;
    const std::vector<float> aux_values =
        copyFp32ToHost(forward_plan.retainedForwardEpilogueAux(y_node), stream);
    ASSERT_EQ(aux_values.size(), batch * output_features);
    for (float value : aux_values) {
        EXPECT_NEAR(value, expected_preactivation, 2e-4f)
            << "GELU_AUX must retain the affine preactivation, not the post-GELU output.";
    }

    const std::vector<float> y_values = copyFp32ToHost(forward_plan.getFinalOutputs().at("y"), stream);
    const float expected_y = cublasLtGeluApprox(expected_preactivation);
    for (float value : y_values) {
        EXPECT_NEAR(value, expected_y, 3e-4f);
    }

    FusedEquation backward_equation = FusedEquation::compile(backward.outputs, 0);
    const Tensor saved_aux = forward_plan.retainedForwardEpilogueAux(y_node);
    std::unordered_map<std::string, Tensor> backward_inputs{
        {"x", x},
        {"w", w},
        {"bias", bias},
        {"dy", dy},
        {aux_requirement.backward_input_name, saved_aux},
    };
    StampedExecutionPlan backward_plan = backward_equation.stamp(backward_inputs, stream);

    resetExpressionTestExecutionCounters();
    backward_plan.run();
    stream.synchronize();
    counters = expressionTestExecutionCounters();
    EXPECT_EQ(counters.matmul.forward, 0U)
        << "A BR4 saved GELU_AUX binding must never launch a forward affine GEMM during backward.";
    EXPECT_EQ(counters.matmul.backward_gradient, 1U);

    const std::vector<float> dx_values = copyFp32ToHost(backward_plan.getFinalOutputs().at("x_grad"), stream);
    const float expected_dx = static_cast<float>(output_features) * w_value *
                              cublasLtGeluApproxDerivative(expected_preactivation);
    ASSERT_EQ(dx_values.size(), batch * input_features);
    for (float value : dx_values) {
        EXPECT_NEAR(value, expected_dx, 8e-4f)
            << "Backward must consume the retained cuBLASLt preactivation with matching DGELU approximation semantics.";
    }
}

TEST(ExpressionForwardReplayInstrumentation, Br4IneligibleGeluAuxShapeFallsBackToBr3Retention) {
    REQUIRE_CUDA_DEVICE();

    // Width four cannot satisfy cuBLASLt GELU_AUX's multiple-of-eight AUX_LD
    // requirement in Thor's row-major mapping. The training preview must leave
    // the exact expression unfused so BR1/BR2 ordinary retained values and BR3's
    // one-affine-producer protection remain the fallback.
    Stream stream(0);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {2, 3}));
    Tensor w(gpu, TensorDescriptor(DataType::FP32, {3, 4}));
    Tensor bias(gpu, TensorDescriptor(DataType::FP32, {4}));
    x.fill(0.25, stream);
    w.fill(0.5, stream);
    bias.fill(0.1, stream);
    stream.synchronize();

    FusedEquation forward_equation = FusedEquation::compile(affineGeluForward(), 0);
    const PhysicalOutputs training_forward = forward_equation.physicalOutputsForTrainingBackward(
        {{"x", x}, {"w", w}, {"bias", bias}});
    ASSERT_EQ(training_forward.outputs.size(), 1U);
    const ExprNode& output_node = training_forward.expr->nodes.at(training_forward.outputs.front().node_idx);
    EXPECT_FALSE(output_node.matmul_forward_epilogue_aux);

    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        training_forward,
        {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {2, 3}},
            {"w", {3, 4}},
            {"bias", {4}},
        });
    ASSERT_FALSE(backward.forward_value_requirements.empty());
    EXPECT_TRUE(std::none_of(backward.forward_value_requirements.begin(),
                             backward.forward_value_requirements.end(),
                             [](const ForwardValueRequirement& requirement) {
                                 return requirement.kind == ForwardValueRequirementKind::MatmulEpilogueAux;
                             }));

    std::vector<uint32_t> retained_nodes;
    for (const ForwardValueRequirement& requirement : backward.forward_value_requirements) {
        ASSERT_EQ(requirement.kind, ForwardValueRequirementKind::NodeOutput);
        retained_nodes.push_back(requirement.forward_node_index);
    }
    std::sort(retained_nodes.begin(), retained_nodes.end());
    retained_nodes.erase(std::unique(retained_nodes.begin(), retained_nodes.end()), retained_nodes.end());

    StampedExecutionPlan forward_plan = forward_equation.stampRetainingForwardValues(
        retained_nodes, {{"x", x}, {"w", w}, {"bias", bias}}, stream);
    const std::vector<std::string> stage_kinds = forward_plan.stageKindNames();
    EXPECT_EQ(std::count(stage_kinds.begin(), stage_kinds.end(), "Matmul"), 1)
        << "If GELU_AUX is ineligible, BR4 must preserve BR3's single-affine-producer fallback.";
    resetExpressionTestExecutionCounters();
    forward_plan.run();
    stream.synchronize();
    const ExpressionTestExecutionCounters counters = expressionTestExecutionCounters();
    EXPECT_EQ(counters.matmul.forward, 1U);
}

TEST(ExpressionForwardReplayInstrumentation, Br60AOrdinaryAutoDiffRefusesFusedActivationWithoutForwardProvider) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w = Expression::input("w", DataType::FP32, DataType::FP32);
    PhysicalOutputs forward =
        Expression::outputs({{"y", Expression::matmul(x, w, false, false, DataType::FP32, DataType::FP32)}})
            .physicalOutputs();
    ASSERT_EQ(forward.outputs.size(), 1U);
    ExprNode& fused_output = forward.expr->nodes.at(forward.outputs.front().node_idx);
    ASSERT_EQ(fused_output.op, ExprOp::MATMUL);
    fused_output.matmul_epilogue = MatmulEpilogue::Relu;
    fused_output.matmul_forward_epilogue_aux = false;

    EXPECT_THROW(
        (void)buildBackwardOutputs(
            forward,
            {"x"},
            std::unordered_map<std::string, std::string>{{"y", "dy"}},
            std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
            std::unordered_map<std::string, std::vector<uint64_t>>{
                {"x", {2, 3}},
                {"w", {3, 4}},
            }),
        std::runtime_error)
        << "BR6.0A removes the legacy affine-preamble replay path from every AutoDiff entry point.";
}


#endif
