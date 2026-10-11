#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {
PhysicalOutputs backwardForLayerNorm(const std::vector<std::string>& targets,
                                     bool packed = false,
                                     uint64_t hidden = 5,
                                     double epsilon = 0.00031,
                                     RaggedRuntimeExtentSource source = RaggedRuntimeExtentSource::HOST_EXTENT) {
    Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression gamma = Expression::input("gamma", DataType::FP32, DataType::FP32);
    const Expression beta = Expression::input("beta", DataType::FP32, DataType::FP32);
    const Expression extent = Expression::input("extent", DataType::UINT32, DataType::UINT32);
    if (packed) {
        x = x.withRaggedRuntimeExtent(extent, 2, 14, hidden, source);
    }
    const Expression y = Expression::layerNorm(x, gamma, beta, hidden, epsilon,
                                             DataType::FP32, DataType::FP32,
                                             packed ? std::optional<uint64_t>(14) : std::nullopt);
    PhysicalOutputs forward = Expression::outputs({{"y", y}}).physicalOutputs();
    std::vector<DataType> forward_dtypes(forward.expr->inputs.size(), DataType::FP32);
    for (const NamedInput& input : forward.expr->inputs) {
        if (input.name == "extent") forward_dtypes.at(input.slot) = DataType::UINT32;
    }
    resolveOutputsDTypesInPlace(forward, forward_dtypes);

    PhysicalOutputs backward = buildBackwardOutputs(
        forward, targets,
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {packed ? 14u : 3u, hidden}},
            {"gamma", {hidden}},
            {"beta", {hidden}},
            {"extent", {packed ? 3u : 1u}},
        });
    std::vector<DataType> backward_dtypes(backward.expr->inputs.size(), DataType::FP32);
    for (const NamedInput& input : backward.expr->inputs) {
        if (input.name == "extent") backward_dtypes.at(input.slot) = DataType::UINT32;
    }
    resolveOutputsDTypesInPlace(backward, backward_dtypes);
    return backward;
}

const ExprNode* findRoute(const PhysicalOutputs& outputs, ExprOp op) {
    const ExprNode* found = nullptr;
    for (const ExprNode& node : outputs.expr->nodes) {
        if (node.op == op) {
            if (found != nullptr) throw std::runtime_error("duplicate LayerNorm gradient route");
            found = &node;
        }
    }
    return found;
}
} // namespace

TEST(LayerNormAutoDiffGraph1B, FilledConstantRemainsANondifferentiableLeafWhenSeeded) {
    // A vector FILL can participate in a differentiable result and also be a
    // separately seeded forward output. Neither route differentiates the fill.
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression constant = Expression::fill(2.5, {3}, DataType::FP32);
    PhysicalOutputs forward = Expression::outputs({
        {"y", x + constant}, {"constant", constant}
    }).physicalOutputs();
    resolveOutputsDTypesInPlace(forward, {DataType::FP32});

    const PhysicalOutputs backward = buildBackwardOutputs(
        forward, {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}, {"constant", "dconstant"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}, {"constant", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{{"x", {3}}});
    ASSERT_EQ(backward.outputs.size(), 1u);
}

TEST(LayerNormAutoDiffGraph1B, ConstantAffineFillShapesSpecializePackedBackward) {
    // AdaptiveLayerNorm normalizes with FILL(1, [hidden]) and
    // FILL(0, [hidden]) before applying its per-example conditioning.
    // Autodiff must infer those leaf shapes from fill_dims, even when
    // specializing the forward graph for a physical packed placement.
    constexpr uint64_t hidden = 3;
    constexpr uint64_t capacity = 14;
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression extent = Expression::input("extent", DataType::UINT32, DataType::UINT32);
    const Expression packed = x.withRaggedRuntimeExtent(
        extent, 2, capacity, hidden, RaggedRuntimeExtentSource::HOST_EXTENT);
    const Expression unit_scale = Expression::fill(1.0, {hidden}, DataType::FP32);
    const Expression zero_bias = Expression::fill(0.0, {hidden}, DataType::FP32);
    const Expression y = Expression::layerNorm(packed, unit_scale, zero_bias,
                                              hidden, 1.0e-5, DataType::FP32,
                                              DataType::FP32, capacity);
    PhysicalOutputs forward = Expression::outputs({{"y", y}}).physicalOutputs();
    std::vector<DataType> input_dtypes(forward.expr->inputs.size(), DataType::FP32);
    for (const NamedInput& input : forward.expr->inputs) {
        if (input.name == "extent") input_dtypes.at(input.slot) = DataType::UINT32;
    }
    resolveOutputsDTypesInPlace(forward, input_dtypes);

    // The packed extent marker is a non-executing alias reconstructed from
    // x and the runtime carrier. LayerNorm dX also needs the forward gamma;
    // here it is a materialized FILL(1, [hidden]), so autodiff must request
    // that value from the real forward rather than re-executing the producer.
    // The legacy PhysicalOutputs-only API rejects the new requirement.
    const BackwardBuildResult build = buildBackwardOutputsWithForwardValueRequirements(
        forward, {"x"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {capacity, hidden}}, {"extent", {3}},
        });
    const PhysicalOutputs& backward = build.outputs;
    ASSERT_EQ(build.forward_value_requirements.size(), 1u);
    const ForwardValueRequirement& required = build.forward_value_requirements.front();
    EXPECT_EQ(required.kind, ForwardValueRequirementKind::NodeOutput);
    EXPECT_TRUE(required.conditional_branch_path.empty());
    ASSERT_LT(required.forward_node_index, forward.expr->nodes.size());
    const ExprNode& retained = forward.expr->nodes.at(required.forward_node_index);
    ASSERT_EQ(retained.op, ExprOp::FILL);
    EXPECT_EQ(retained.fill_dims, (std::vector<uint64_t>{hidden}));
    EXPECT_DOUBLE_EQ(retained.scalar_fp, 1.0);
    ASSERT_TRUE(retained.output_dtype.has_value());
    EXPECT_EQ(retained.output_dtype.value(), DataType::FP32);
    size_t retained_input_bindings = 0;
    for (const NamedInput& input : backward.expr->inputs) {
        if (input.name == required.backward_input_name) ++retained_input_bindings;
    }
    EXPECT_EQ(retained_input_bindings, 1u)
        << "The retained value must be bound from the real forward execution.";
    ASSERT_EQ(backward.outputs.size(), 1u);
    EXPECT_NE(findRoute(backward, ExprOp::LAYERNORM_BACKWARD_X), nullptr);
    EXPECT_EQ(findRoute(backward, ExprOp::LAYERNORM_BACKWARD_SCALE), nullptr);
    EXPECT_EQ(findRoute(backward, ExprOp::LAYERNORM_BACKWARD_BIAS), nullptr);
}

TEST(LayerNormAutoDiffGraph1B, DenseCreatesThreeIndependentAffineAndValueRoutes) {
    const PhysicalOutputs backward = backwardForLayerNorm({"x", "gamma", "beta"});
    for (const ExprOp op : {ExprOp::LAYERNORM_BACKWARD_X, ExprOp::LAYERNORM_BACKWARD_SCALE,
                            ExprOp::LAYERNORM_BACKWARD_BIAS}) {
        const ExprNode* route = findRoute(backward, op);
        ASSERT_NE(route, nullptr);
        EXPECT_EQ(route->layer_norm_normalized_feature_count, 5u);
        EXPECT_DOUBLE_EQ(route->layer_norm_epsilon, 0.00031);
        EXPECT_EQ(route->layer_norm_packed_row_capacity, 0u);
        ASSERT_TRUE(route->output_dtype.has_value());
        EXPECT_EQ(route->output_dtype.value(), DataType::FP32);
    }
    ASSERT_EQ(backward.outputs.size(), 3u);
}

TEST(LayerNormAutoDiffGraph1B, RequestedGradientSubsetsDoNotInventOtherRoutes) {
    for (const auto& target : {"x", "gamma", "beta"}) {
        const PhysicalOutputs backward = backwardForLayerNorm({target});
        EXPECT_EQ(backward.outputs.size(), 1u);
        EXPECT_EQ(findRoute(backward, ExprOp::LAYERNORM_BACKWARD_X) != nullptr, target == std::string("x"));
        EXPECT_EQ(findRoute(backward, ExprOp::LAYERNORM_BACKWARD_SCALE) != nullptr, target == std::string("gamma"));
        EXPECT_EQ(findRoute(backward, ExprOp::LAYERNORM_BACKWARD_BIAS) != nullptr, target == std::string("beta"));
    }
}

TEST(LayerNormAutoDiffGraph1B, PackedHostExtentRemainsStructuralAndProducesThreeRoutes) {
    const PhysicalOutputs backward = backwardForLayerNorm({"x", "gamma", "beta"}, true, 7, 0.0005);
    for (const ExprOp op : {ExprOp::LAYERNORM_BACKWARD_X, ExprOp::LAYERNORM_BACKWARD_SCALE,
                            ExprOp::LAYERNORM_BACKWARD_BIAS}) {
        const ExprNode* route = findRoute(backward, op);
        ASSERT_NE(route, nullptr);
        EXPECT_EQ(route->layer_norm_packed_row_capacity, 14u);
        EXPECT_EQ(route->layer_norm_normalized_feature_count, 7u);
        EXPECT_DOUBLE_EQ(route->layer_norm_epsilon, 0.0005);
    }
    size_t host_extent_count = 0;
    for (const ExprNode& node : backward.expr->nodes) {
        if (node.op != ExprOp::RAGGED_VALUEWISE_EXTENT) continue;
        ++host_extent_count;
        EXPECT_EQ(node.ragged_runtime_extent_source, RaggedRuntimeExtentSource::HOST_EXTENT);
        EXPECT_EQ(node.ragged_runtime_max_active_values, 14u);
    }
    EXPECT_GE(host_extent_count, 1u);
}

TEST(LayerNormAutoDiffGraph1B, PackedDeviceOffsetsForAdaptiveAffineDoNotChangeExtentSource) {
    const PhysicalOutputs backward = backwardForLayerNorm(
        {"x", "gamma", "beta"}, true, 5, 0.00031, RaggedRuntimeExtentSource::DEVICE_OFFSETS);
    ASSERT_NE(findRoute(backward, ExprOp::LAYERNORM_BACKWARD_X), nullptr);
    ASSERT_NE(findRoute(backward, ExprOp::LAYERNORM_BACKWARD_SCALE), nullptr);
    ASSERT_NE(findRoute(backward, ExprOp::LAYERNORM_BACKWARD_BIAS), nullptr);
    bool retained = false;
    for (const ExprNode& node : backward.expr->nodes) {
        if (node.op != ExprOp::RAGGED_VALUEWISE_EXTENT) continue;
        retained = true;
        EXPECT_EQ(node.ragged_runtime_extent_source, RaggedRuntimeExtentSource::DEVICE_OFFSETS);
    }
    EXPECT_TRUE(retained);
}

TEST(LayerNormAutoDiffGraph1B, BackwardRoutesSerializeAndRoundTripWithoutPhysicalPlacement) {
    const PhysicalOutputs backward = backwardForLayerNorm({"x", "gamma", "beta"}, true, 7, 0.001);
    const ExpressionDefinition definition =
        ExpressionDefinition::fromOutputs(Outputs::fromPhysicalOutputs(backward));
    const nlohmann::json payload = definition.architectureJson();
    size_t routes = 0;
    for (const auto& node : payload.at("nodes")) {
        const std::string op = node.at("op").get<std::string>();
        if (op != "layernorm_backward_x" && op != "layernorm_backward_scale" &&
            op != "layernorm_backward_bias") continue;
        ++routes;
        EXPECT_EQ(node.at("layer_norm_normalized_feature_count").get<uint64_t>(), 7u);
        EXPECT_EQ(node.at("layer_norm_packed_row_capacity").get<uint64_t>(), 14u);
        EXPECT_DOUBLE_EQ(node.at("layer_norm_epsilon").get<double>(), 0.001);
    }
    EXPECT_EQ(routes, 3u);
    const ExpressionDefinition loaded = ExpressionDefinition::deserialize(payload);
    EXPECT_EQ(loaded.architectureJson(), payload);
}
