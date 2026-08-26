#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_runtime.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {

const ExprNode& singleOutputNode(const PhysicalOutputs& outputs) {
    if (!outputs.expr || outputs.outputs.size() != 1) {
        throw std::runtime_error("test expected one physical output");
    }
    return outputs.expr->nodes.at(outputs.outputs.front().node_idx);
}


#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                               \
        int cuda_device_count_for_test = 0;                                                                            \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                      \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                  \
            GTEST_SKIP() << "CUDA device is required for BROADCAST_TO execution tests.";                              \
        }                                                                                                              \
    } while (false)

Tensor makeFp32Tensor(const std::vector<uint64_t>& dims, const std::vector<float>& values, Stream& stream) {
    TensorPlacement cpu_placement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu_placement(TensorPlacement::MemDevices::GPU, 0);
    Tensor cpu(cpu_placement, TensorDescriptor(DataType::FP32, dims));
    if (cpu.getTotalNumElements() != values.size()) {
        throw std::runtime_error("test tensor value count does not match dimensions");
    }
    auto* ptr = static_cast<float*>(cpu.getMemPtr());
    for (size_t i = 0; i < values.size(); ++i) {
        ptr[i] = values[i];
    }
    Tensor gpu(gpu_placement, TensorDescriptor(DataType::FP32, dims));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

std::vector<float> copyFp32ToCpu(const Tensor& gpu, Stream& stream) {
    TensorPlacement cpu_placement(TensorPlacement::MemDevices::CPU);
    Tensor cpu(cpu_placement, gpu.getDescriptor());
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    const auto* ptr = static_cast<const float*>(cpu.getMemPtr());
    return std::vector<float>(ptr, ptr + cpu.getTotalNumElements());
}

void expectFp32Values(const Tensor& tensor, const std::vector<float>& expected, Stream& stream) {
    const std::vector<float> actual = copyFp32ToCpu(tensor, stream);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
    }
}

template <typename Fn>
void expectRuntimeErrorContaining(Fn&& fn, const std::string& expected_fragment) {
    try {
        fn();
        FAIL() << "Expected std::runtime_error containing: " << expected_fragment;
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(expected_fragment), std::string::npos) << error.what();
    }
}

bool isZeroFill(const PhysicalExpression& expr, uint32_t node_idx) {
    if (node_idx >= expr.nodes.size()) {
        return false;
    }
    const ExprNode& node = expr.nodes.at(node_idx);
    return node.op == ExprOp::FILL && node.scalar_fp == 0.0;
}

bool isZeroAddBroadcast(const PhysicalExpression& expr, const ExprNode& node) {
    return node.op == ExprOp::ADD && (isZeroFill(expr, node.lhs) || isZeroFill(expr, node.rhs));
}

}  // namespace

TEST(BroadcastToExpression, BuilderUsesDedicatedTargetDimensionsAndUnaryIdentity) {
    const Expression x = Expression::input("x");
    const PhysicalOutputs physical = Expression::outputs({{"y", x.broadcastTo({2, 3})}}).physicalOutputs();

    const ExprNode& node = singleOutputNode(physical);
    ASSERT_EQ(node.op, ExprOp::BROADCAST_TO);
    EXPECT_EQ(node.broadcast_dims, (std::vector<uint64_t>{2, 3}));
    EXPECT_TRUE(node.fill_dims.empty());
    EXPECT_TRUE(Expression::isUnaryOp(ExprOp::BROADCAST_TO));
    ASSERT_LT(node.lhs, physical.expr->nodes.size());
    EXPECT_EQ(physical.expr->nodes[node.lhs].op, ExprOp::INPUT);

    EXPECT_THROW((void)x.broadcastTo({}), std::invalid_argument);
    EXPECT_THROW((void)x.broadcastTo({2, 0}), std::invalid_argument);
    EXPECT_THROW((void)x.broadcastTo({2, std::numeric_limits<uint64_t>::max()}), std::invalid_argument);
}

TEST(BroadcastToExpression, ShapeInferenceUsesStandardTrailingAxisBroadcastRules) {
    EXPECT_EQ(inferBroadcastToOutputDims({}, {2, 3}), (std::vector<uint64_t>{2, 3}));
    EXPECT_EQ(inferBroadcastToOutputDims({3}, {2, 3}), (std::vector<uint64_t>{2, 3}));
    EXPECT_EQ(inferBroadcastToOutputDims({1, 4, 1, 8}, {2, 4, 7, 8}),
              (std::vector<uint64_t>{2, 4, 7, 8}));
    EXPECT_EQ(inferBroadcastToOutputDims({2, 3}, {2, 3}), (std::vector<uint64_t>{2, 3}));

    EXPECT_THROW((void)inferBroadcastToOutputDims({2, 3}, {3}), std::invalid_argument);
    EXPECT_THROW((void)inferBroadcastToOutputDims({2, 3}, {4, 3}), std::invalid_argument);
    EXPECT_THROW((void)inferBroadcastToOutputDims({2, 3}, {2, 0, 3}), std::invalid_argument);
}

TEST(BroadcastToExpression, DTypeResolutionPreservesInputDTypeAndKeepsCastOrthogonal) {
    {
        const Expression x = Expression::input("x", DataType::BF16, DataType::BF16);
        PhysicalOutputs physical = Expression::outputs({{"y", x.broadcastTo({2, 3})}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::BF16});

        const ExprNode& node = singleOutputNode(physical);
        ASSERT_TRUE(node.output_dtype.has_value());
        EXPECT_EQ(node.output_dtype.value(), DataType::BF16);
        ASSERT_TRUE(node.compute_dtype.has_value());
        EXPECT_EQ(node.compute_dtype.value(), DataType::BF16);
    }

    {
        const Expression x = Expression::input("x", DataType::UINT32, DataType::UINT32);
        PhysicalOutputs physical = Expression::outputs({{"y", x.broadcastTo({2, 3})}}).physicalOutputs();
        resolveOutputsDTypesInPlace(physical, {DataType::UINT32});

        const ExprNode& node = singleOutputNode(physical);
        ASSERT_TRUE(node.output_dtype.has_value());
        EXPECT_EQ(node.output_dtype.value(), DataType::UINT32);
        ASSERT_TRUE(node.compute_dtype.has_value());
        EXPECT_EQ(node.compute_dtype.value(), DataType::UINT32);
    }

    {
        const Expression x = Expression::input("x", DataType::BF16, DataType::BF16);
        PhysicalOutputs physical = Expression::outputs({{"y", x.broadcastTo({2, 3})}}).physicalOutputs();
        physical.expr->nodes.at(physical.outputs.front().node_idx).output_dtype = DataType::FP32;
        expectRuntimeErrorContaining(
            [&]() { resolveOutputsDTypesInPlace(physical, {DataType::BF16}); },
            "BROADCAST_TO preserves dtype; use CAST as a separate operation");
    }
}

TEST(BroadcastToExpression, CanonicalIdentityAndSerializationIncludeTargetDimensions) {
    const Expression x = Expression::input("x");
    const ExpressionDefinition a = ExpressionDefinition::fromOutputs(Expression::outputs({{"y", x.broadcastTo({2, 3})}}));
    const ExpressionDefinition b = ExpressionDefinition::fromOutputs(Expression::outputs({{"y", x.broadcastTo({4, 3})}}));

    EXPECT_NE(a.canonical_hash, b.canonical_hash);

    const nlohmann::json json = a.architectureJson();
    bool found_broadcast = false;
    for (const auto& node : json.at("nodes")) {
        if (node.at("op").get<std::string>() != "broadcast_to") {
            continue;
        }
        found_broadcast = true;
        EXPECT_EQ(node.at("broadcast_dims").get<std::vector<uint64_t>>(), (std::vector<uint64_t>{2, 3}));
    }
    EXPECT_TRUE(found_broadcast);

    const ExpressionDefinition round_trip = ExpressionDefinition::deserialize(json);
    EXPECT_EQ(round_trip.canonical_hash, a.canonical_hash);
    const ExprNode& round_trip_node = singleOutputNode(round_trip.outputs);
    EXPECT_EQ(round_trip_node.op, ExprOp::BROADCAST_TO);
    EXPECT_EQ(round_trip_node.broadcast_dims, (std::vector<uint64_t>{2, 3}));
}

TEST(BroadcastToExpression, LegacyNodesDoNotGainSerializedBroadcastMetadata) {
    const Expression x = Expression::input("x");
    const ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Expression::outputs({{"y", x + 1.0}}));
    const nlohmann::json json = definition.architectureJson();

    for (const auto& node : json.at("nodes")) {
        EXPECT_FALSE(node.contains("broadcast_dims"));
    }

    EXPECT_NO_THROW((void)ExpressionDefinition::deserialize(json));
}

TEST(BroadcastToExpression, ValidationRejectsMissingOrInvalidTargetDimensions) {
    const Expression x = Expression::input("x");
    const ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Expression::outputs({{"y", x.broadcastTo({2, 3})}}));

    for (const std::vector<uint64_t>& invalid_dims :
         {std::vector<uint64_t>{}, std::vector<uint64_t>{2, 0},
          std::vector<uint64_t>{2, std::numeric_limits<uint64_t>::max()}}) {
        nlohmann::json json = definition.architectureJson();
        for (auto& node : json["nodes"]) {
            if (node.at("op").get<std::string>() == "broadcast_to") {
                node["broadcast_dims"] = invalid_dims;
            }
        }
        expectRuntimeErrorContaining(
            [&]() { (void)ExpressionDefinition::deserialize(json); },
            "BROADCAST_TO");
    }
}


TEST(BroadcastToExpression, ValidationRejectsBroadcastMetadataOnOtherOps) {
    const Expression x = Expression::input("x");
    const ExpressionDefinition definition = ExpressionDefinition::fromOutputs(Expression::outputs({{"y", x + 1.0}}));
    nlohmann::json json = definition.architectureJson();

    bool mutated = false;
    for (auto& node : json["nodes"]) {
        if (node.at("op").get<std::string>() == "add") {
            node["broadcast_dims"] = std::vector<uint64_t>{2, 3};
            mutated = true;
            break;
        }
    }
    ASSERT_TRUE(mutated);
    expectRuntimeErrorContaining(
        [&]() { (void)ExpressionDefinition::deserialize(json); },
        "broadcast_dims metadata is valid only for BROADCAST_TO");
}

TEST(BroadcastToExpression, RaggedRuntimeExtentRequiresSegmentedBroadcast) {
    const Expression values = Expression::input("values", DataType::FP32, DataType::FP32);
    const Expression offsets = Expression::input("offsets", DataType::UINT32, DataType::UINT32);
    const Expression ragged_values = values.withRaggedRuntimeExtent(offsets, 2, 8, 4);
    PhysicalOutputs physical = Expression::outputs({{"y", ragged_values.broadcastTo({8, 4})}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32, DataType::UINT32});

    expectRuntimeErrorContaining(
        [&]() { (void)EquationCompiler::splitAtReductionBoundaries(physical); },
        "use SEGMENTED_BROADCAST for ragged row-wise expansion");
}


TEST(BroadcastToExpression, UnreachableBroadcastDoesNotBlockPlanningOtherOutputs) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    PhysicalOutputs physical = Expression::outputs({{"y", x + 1.0}}).physicalOutputs();
    resolveOutputsDTypesInPlace(physical, {DataType::FP32});

    ExprNode dead_broadcast{};
    dead_broadcast.op = ExprOp::BROADCAST_TO;
    dead_broadcast.lhs = physical.outputs.front().node_idx;
    dead_broadcast.broadcast_dims = {2, 3};
    dead_broadcast.output_dtype = DataType::FP32;
    dead_broadcast.compute_dtype = DataType::FP32;
    physical.expr->nodes.push_back(dead_broadcast);

    EXPECT_NO_THROW((void)EquationCompiler::splitAtReductionBoundaries(physical));
}

TEST(BroadcastToExpression, DenseExecutionUsesFusedBroadcastIndexing) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor x_value = makeFp32Tensor({3}, {1.0f, 2.0f, 3.0f}, stream);

    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    FusedEquation equation = FusedEquation::compile(
        Expression::outputs({{"y", x.broadcastTo({2, 3})}}).physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"x", x_value}}, stream);

    ASSERT_EQ(plan.output("y").getDimensions(), (std::vector<uint64_t>{2, 3}));
    plan.run();
    stream.synchronize();
    expectFp32Values(plan.output("y"), {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f}, stream);
}

TEST(BroadcastToExpression, DenseExecutionHandlesMultipleSingletonAxes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor x_value = makeFp32Tensor({1, 2, 1}, {10.0f, 20.0f}, stream);

    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    FusedEquation equation = FusedEquation::compile(
        Expression::outputs({{"y", x.broadcastTo({3, 2, 4})}}).physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"x", x_value}}, stream);
    plan.run();
    stream.synchronize();

    std::vector<float> expected;
    expected.reserve(24);
    for (size_t batch = 0; batch < 3; ++batch) {
        (void)batch;
        for (float value : {10.0f, 20.0f}) {
            for (size_t inner = 0; inner < 4; ++inner) {
                (void)inner;
                expected.push_back(value);
            }
        }
    }
    EXPECT_EQ(plan.output("y").getDimensions(), (std::vector<uint64_t>{3, 2, 4}));
    expectFp32Values(plan.output("y"), expected, stream);
}

TEST(BroadcastToExpression, BackwardGraphUsesExplicitReductionToOriginalShape) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    PhysicalOutputs forward = Expression::outputs({{"y", x.broadcastTo({2, 3})}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP32});

    PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"x"},
        std::optional<std::string>{"dy"},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {3}}});

    ASSERT_EQ(backward.outputs.size(), 1u);
    const ExprNode& root = backward.expr->nodes.at(backward.outputs.front().node_idx);
    EXPECT_EQ(root.op, ExprOp::REDUCE_SUM);
    EXPECT_EQ(root.reduction_axes, (std::vector<uint64_t>{0}));
    EXPECT_EQ(root.squeeze_axes, (std::vector<uint64_t>{0}));
}

TEST(BroadcastToExpression, BackwardReducesBroadcastAxesToOriginalInputShape) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor x_value = makeFp32Tensor({3}, {1.0f, 2.0f, 3.0f}, stream);
    Tensor dy = makeFp32Tensor({2, 3}, {1.0f, 10.0f, 100.0f, 2.0f, 20.0f, 200.0f}, stream);

    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    FusedEquation forward = FusedEquation::compile(
        Expression::outputs({{"y", x.broadcastTo({2, 3})}}).physicalOutputs(), 0);
    FusedEquation backward = forward.compileBackward({"x"}, "dy");
    StampedExecutionPlan plan = backward.stamp({{"x", x_value}, {"dy", dy}}, stream);

    plan.run();
    stream.synchronize();
    ASSERT_EQ(plan.output("x_grad").getDimensions(), (std::vector<uint64_t>{3}));
    expectFp32Values(plan.output("x_grad"), {3.0f, 30.0f, 300.0f}, stream);
}

TEST(BroadcastToExpression, BackwardNoOpBroadcastPreservesGradientWithoutReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor x_value = makeFp32Tensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, stream);
    Tensor dy = makeFp32Tensor({2, 3}, {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}, stream);

    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    FusedEquation forward = FusedEquation::compile(
        Expression::outputs({{"y", x.broadcastTo({2, 3})}}).physicalOutputs(), 0);
    FusedEquation backward = forward.compileBackward({"x"}, "dy");
    StampedExecutionPlan plan = backward.stamp({{"x", x_value}, {"dy", dy}}, stream);

    plan.run();
    stream.synchronize();
    expectFp32Values(plan.output("x_grad"), {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}, stream);
}

TEST(BroadcastToExpression, ReductionBackwardUsesExplicitBroadcastToInsteadOfZeroAdd) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    PhysicalOutputs forward = Expression::outputs({{"y", x.reduce_sum({0}, {})}}).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP32});

    PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"x"},
        std::optional<std::string>{"dy"},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {2, 3}}});

    ASSERT_EQ(backward.outputs.size(), 1u);
    const ExprNode& root = backward.expr->nodes.at(backward.outputs.front().node_idx);
    EXPECT_EQ(root.op, ExprOp::BROADCAST_TO);
    EXPECT_EQ(root.broadcast_dims, (std::vector<uint64_t>{2, 3}));

    for (const ExprNode& node : backward.expr->nodes) {
        EXPECT_FALSE(isZeroAddBroadcast(*backward.expr, node))
            << "dense reduction-gradient expansion must use BROADCAST_TO, not fill(0)+grad";
    }
}

TEST(BroadcastToExpression, AttentionScoreBiasExpansionUsesCastThenBroadcastToInsteadOfZeroAdd) {
    const char* experimental_surface = std::getenv("THOR_EXPERIMENTAL_CUDNN_ATTENTION_SUPPORT_SURFACE");
    if (experimental_surface != nullptr && std::string_view(experimental_surface) == "1") {
        GTEST_SKIP() << "The experimental cuDNN attention support-surface bypass intentionally disables "
                        "production bias materialization.";
    }

    const Expression q = Expression::input("q", DataType::FP32, DataType::FP32);
    const Expression k = Expression::input("k", DataType::FP32, DataType::FP32);
    const Expression v = Expression::input("v", DataType::FP32, DataType::FP32);
    const Expression bias = Expression::input("bias", DataType::BF16, DataType::BF16);

    AttentionOptions options;
    options.compute_dtype = DataType::FP32;
    options.output_dtype = DataType::FP32;
    PhysicalOutputs forward = Expression::outputs({{"y", Expression::attention(q, k, v, bias, options)}}).physicalOutputs();

    std::vector<DataType> input_dtypes(forward.expr->inputs.size(), DataType::FP32);
    for (const NamedInput& input : forward.expr->inputs) {
        if (input.name == "bias") {
            input_dtypes.at(input.slot) = DataType::BF16;
        }
    }
    resolveOutputsDTypesInPlace(forward, input_dtypes);

    PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"bias"},
        std::optional<std::string>{"dy"},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"q", {2, 4, 3, 8}},
            {"k", {2, 4, 5, 8}},
            {"v", {2, 4, 5, 6}},
            {"bias", {1, 4, 1, 5}},
        });

    size_t score_bias_broadcasts = 0;
    for (const ExprNode& node : backward.expr->nodes) {
        EXPECT_FALSE(isZeroAddBroadcast(*backward.expr, node))
            << "attention score-bias materialization must not use fill(0)+bias";
        if (node.op != ExprOp::BROADCAST_TO ||
            node.broadcast_dims != std::vector<uint64_t>({2, 4, 3, 5})) {
            continue;
        }
        ++score_bias_broadcasts;
        ASSERT_LT(node.lhs, backward.expr->nodes.size());
        const ExprNode& cast = backward.expr->nodes.at(node.lhs);
        EXPECT_EQ(cast.op, ExprOp::CAST);
        ASSERT_TRUE(cast.output_dtype.has_value());
        EXPECT_EQ(cast.output_dtype.value(), DataType::FP32);
    }
    EXPECT_EQ(score_bias_broadcasts, 1u);
}
