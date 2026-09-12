#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Expression/Expression.h"

#include "gtest/gtest.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {

bool hasTensorInputNamed(const PhysicalExpression& expr, const std::string& name) {
    for (const NamedInput& input : expr.inputs) {
        if (input.kind == NamedInput::Kind::Tensor && input.name == name) {
            return true;
        }
    }
    return false;
}

PhysicalOutputs makeTwoInputProductForward() {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w = Expression::input("w", DataType::FP32, DataType::FP32);
    return Expression::outputs({{"y", x * w}}).physicalOutputs();
}

}  // namespace

TEST(SelectiveGradientAccumulation, OnlySelectedWrtReadsExistingGradient) {
    const PhysicalOutputs forward = makeTwoInputProductForward();

    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x", "w"},
        std::optional<std::string>("dy"),
        std::nullopt,
        GradientAccumulationTargets{"w"});

    ASSERT_NE(backward.outputs.expr, nullptr);
    EXPECT_FALSE(hasTensorInputNamed(*backward.outputs.expr, "x_grad"));
    EXPECT_TRUE(hasTensorInputNamed(*backward.outputs.expr, "w_grad"));
}

TEST(SelectiveGradientAccumulation, EmptyAndAllTargetSetsHaveExpectedTerminalContracts) {
    const PhysicalOutputs forward = makeTwoInputProductForward();

    const BackwardBuildResult overwrite_all = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x", "w"},
        std::optional<std::string>("dy"),
        std::nullopt,
        GradientAccumulationTargets{});
    ASSERT_NE(overwrite_all.outputs.expr, nullptr);
    EXPECT_FALSE(hasTensorInputNamed(*overwrite_all.outputs.expr, "x_grad"));
    EXPECT_FALSE(hasTensorInputNamed(*overwrite_all.outputs.expr, "w_grad"));

    const BackwardBuildResult accumulate_all = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x", "w"},
        std::optional<std::string>("dy"),
        std::nullopt,
        GradientAccumulationTargets{"x", "w"});
    ASSERT_NE(accumulate_all.outputs.expr, nullptr);
    EXPECT_TRUE(hasTensorInputNamed(*accumulate_all.outputs.expr, "x_grad"));
    EXPECT_TRUE(hasTensorInputNamed(*accumulate_all.outputs.expr, "w_grad"));
}

TEST(SelectiveGradientAccumulation, LegacyBoolCompatibilityStillMeansNoneOrAllRequested) {
    const PhysicalOutputs forward = makeTwoInputProductForward();

    const BackwardBuildResult overwrite_all = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x", "w"},
        std::optional<std::string>("dy"),
        std::nullopt,
        false);
    ASSERT_NE(overwrite_all.outputs.expr, nullptr);
    EXPECT_FALSE(hasTensorInputNamed(*overwrite_all.outputs.expr, "x_grad"));
    EXPECT_FALSE(hasTensorInputNamed(*overwrite_all.outputs.expr, "w_grad"));

    const BackwardBuildResult accumulate_all = buildBackwardOutputsWithForwardValueRequirements(
        forward,
        {"x", "w"},
        std::optional<std::string>("dy"),
        std::nullopt,
        true);
    ASSERT_NE(accumulate_all.outputs.expr, nullptr);
    EXPECT_TRUE(hasTensorInputNamed(*accumulate_all.outputs.expr, "x_grad"));
    EXPECT_TRUE(hasTensorInputNamed(*accumulate_all.outputs.expr, "w_grad"));
}

TEST(SelectiveGradientAccumulation, RejectsTargetOutsideRequestedWrtSet) {
    const PhysicalOutputs forward = makeTwoInputProductForward();

    EXPECT_THROW(
        (void)buildBackwardOutputsWithForwardValueRequirements(
            forward,
            {"x"},
            std::optional<std::string>("dy"),
            std::nullopt,
            GradientAccumulationTargets{"w"}),
        std::runtime_error);
}

TEST(SelectiveGradientAccumulation, ConditionalMissingInputsHonorPerTargetPolicy) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w = Expression::input("w", DataType::FP32, DataType::FP32);
    const Expression predicate_value = Expression::input("predicate_value", DataType::FP32, DataType::FP32);
    const Outputs forward = Outputs::conditional(
        predicate_value.greaterThan(Expression::constantScalar(0.0)),
        Expression::outputs({{"y", x * Expression::constantScalar(2.0)}}),
        Expression::outputs({{"y", w * Expression::constantScalar(3.0)}}));

    const std::unordered_map<std::string, std::vector<uint64_t>> forward_input_dims = {
        {"x", {4}},
        {"w", {4}},
        {"predicate_value", {1}},
    };
    const BackwardBuildResult backward = buildBackwardOutputsWithForwardValueRequirements(
        forward.physicalOutputs(),
        {"x", "w"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        forward_input_dims,
        GradientAccumulationTargets{"w"});

    ASSERT_TRUE(backward.outputs.isConditional());
    ASSERT_NE(backward.outputs.conditional, nullptr);
    ASSERT_NE(backward.outputs.conditional->then_branch.expr, nullptr);
    ASSERT_NE(backward.outputs.conditional->else_branch.expr, nullptr);

    // then: x contributes and overwrites; w is inactive and must preserve its
    // existing accumulated value. else: x is inactive and overwrites with zero;
    // w contributes and accumulates. Neither branch should ever read x_grad.
    EXPECT_FALSE(hasTensorInputNamed(*backward.outputs.conditional->then_branch.expr, "x_grad"));
    EXPECT_TRUE(hasTensorInputNamed(*backward.outputs.conditional->then_branch.expr, "w_grad"));
    EXPECT_FALSE(hasTensorInputNamed(*backward.outputs.conditional->else_branch.expr, "x_grad"));
    EXPECT_TRUE(hasTensorInputNamed(*backward.outputs.conditional->else_branch.expr, "w_grad"));
}
