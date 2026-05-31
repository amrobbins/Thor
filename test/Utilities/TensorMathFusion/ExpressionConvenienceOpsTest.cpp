#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_runtime.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {

const ExprNode& outputNode(const PhysicalOutputs& outputs) {
    if (!outputs.expr) {
        throw std::runtime_error("test output graph is null");
    }
    if (outputs.outputs.size() != 1) {
        throw std::runtime_error("test expected exactly one output");
    }
    return outputs.expr->nodes.at(outputs.outputs[0].node_idx);
}

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cuda_device_count_for_test = 0;                                                                             \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                       \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                    \
            GTEST_SKIP() << "CUDA device is required for Expression numerical execution tests.";                        \
        }                                                                                                               \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

uint64_t tensorNumel(const Tensor& tensor) {
    uint64_t n = 1;
    for (uint64_t d : tensor.getDimensions()) {
        n *= d;
    }
    return n;
}

Tensor makeGpuTensor(const std::vector<uint64_t>& dims, const std::vector<float>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP32, dims));
    if (tensorNumel(cpu) != values.size()) {
        throw std::runtime_error("makeGpuTensor value count mismatch.");
    }

    auto* ptr = static_cast<float*>(cpu.getMemPtr());
    for (size_t i = 0; i < values.size(); ++i) {
        ptr[i] = values[i];
    }

    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::FP32, dims));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

std::vector<float> copyToCpuValues(const Tensor& gpu, Stream& stream) {
    Tensor cpu = gpu.clone(cpuPlacement);
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();

    std::vector<float> values(tensorNumel(cpu));
    const auto* ptr = static_cast<const float*>(cpu.getMemPtr());
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = ptr[i];
    }
    return values;
}

void expectNear(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 1.0e-5f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], atol) << "index " << i;
    }
}

template <typename Fn>
std::vector<float> mapFloatValues(const std::vector<float>& values, Fn&& fn) {
    std::vector<float> expected;
    expected.reserve(values.size());
    for (float value : values) {
        expected.push_back(static_cast<float>(fn(static_cast<double>(value))));
    }
    return expected;
}

Tensor runExpressionOutput(const Outputs& expression_outputs,
                           const std::unordered_map<std::string, Tensor>& inputs,
                           const std::string& output_name,
                           Stream& stream) {
    FusedEquation eq = FusedEquation::compile(expression_outputs.physicalOutputs(), 0);
    StampedExecutionPlan plan = eq.stamp(inputs, stream);
    plan.run();
    return plan.output(output_name);
}

}  // namespace

TEST(ExpressionConvenienceOps, ClampWithScalarBoundsLowersToMaxThenMin) {
    auto x = Expression::input("x");

    auto outputs = Expression::outputs({{"y", x.clamp(-1.0, 2.0)}}).physicalOutputs();

    const ExprNode& minNode = outputNode(outputs);
    ASSERT_EQ(minNode.op, ExprOp::MIN);
    ASSERT_NE(minNode.lhs, UINT32_MAX);
    ASSERT_NE(minNode.rhs, UINT32_MAX);

    const ExprNode& maxNode = outputs.expr->nodes.at(minNode.lhs);
    ASSERT_EQ(maxNode.op, ExprOp::MAX);
    ASSERT_NE(maxNode.lhs, UINT32_MAX);
    ASSERT_NE(maxNode.rhs, UINT32_MAX);
    EXPECT_EQ(outputs.expr->nodes.at(maxNode.lhs).op, ExprOp::INPUT);

    const ExprNode& lowerBoundNode = outputs.expr->nodes.at(maxNode.rhs);
    ASSERT_EQ(lowerBoundNode.op, ExprOp::SCALAR_FP);
    EXPECT_DOUBLE_EQ(lowerBoundNode.scalar_fp, -1.0);

    const ExprNode& upperBoundNode = outputs.expr->nodes.at(minNode.rhs);
    ASSERT_EQ(upperBoundNode.op, ExprOp::SCALAR_FP);
    EXPECT_DOUBLE_EQ(upperBoundNode.scalar_fp, 2.0);

    resolveOutputsDTypesInPlace(outputs, {DataType::FP32});
    const auto stages = EquationCompiler::splitAtReductionBoundaries(outputs);
    ASSERT_EQ(stages.size(), 1);
    EXPECT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
}

TEST(ExpressionConvenienceOps, ClampWithExpressionBoundsLowersToMaxThenMin) {
    auto x = Expression::input("x");
    auto lower = Expression::input("lower");
    auto upper = Expression::input("upper");

    auto outputs = Expression::outputs({{"y", Expression::clamp(x, lower, upper)}}).physicalOutputs();

    const ExprNode& minNode = outputNode(outputs);
    ASSERT_EQ(minNode.op, ExprOp::MIN);
    const ExprNode& maxNode = outputs.expr->nodes.at(minNode.lhs);
    EXPECT_EQ(maxNode.op, ExprOp::MAX);
    EXPECT_EQ(outputs.expr->nodes.at(maxNode.lhs).op, ExprOp::INPUT);
    EXPECT_EQ(outputs.expr->nodes.at(maxNode.rhs).op, ExprOp::INPUT);
    EXPECT_EQ(outputs.expr->nodes.at(minNode.rhs).op, ExprOp::INPUT);
}

TEST(ExpressionConvenienceOps, ClampRejectsReversedScalarBounds) {
    auto x = Expression::input("x");
    EXPECT_THROW((void)x.clamp(2.0, -1.0), std::invalid_argument);
    EXPECT_THROW((void)Expression::clamp(x, 2.0, -1.0), std::invalid_argument);
}

TEST(ExpressionConvenienceOps, DotProductLowersToMultiplyThenReduceSum) {
    auto a = Expression::input("a");
    auto b = Expression::input("b");

    auto outputs = Expression::outputs({{"dot", Expression::dotProduct(a, b)}}).physicalOutputs();

    const ExprNode& reduceNode = outputNode(outputs);
    ASSERT_EQ(reduceNode.op, ExprOp::REDUCE_SUM);
    EXPECT_TRUE(reduceNode.reduction_axes.empty());
    ASSERT_EQ(reduceNode.squeeze_axes, std::vector<uint64_t>{UINT64_MAX});
    ASSERT_NE(reduceNode.lhs, UINT32_MAX);

    const ExprNode& mulNode = outputs.expr->nodes.at(reduceNode.lhs);
    EXPECT_EQ(mulNode.op, ExprOp::MUL);

    resolveOutputsDTypesInPlace(outputs, {DataType::FP32, DataType::FP32});
    const auto stages = EquationCompiler::splitAtReductionBoundaries(outputs);
    ASSERT_EQ(stages.size(), 2);
    EXPECT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    EXPECT_EQ(stages[1].kind, PhysicalExecutionStage::Kind::Reduction);
}

TEST(ExpressionConvenienceOps, OuterProductLowersToUnsqueezeThenMatmul) {
    auto a = Expression::input("a");
    auto b = Expression::input("b");

    auto outputs = Expression::outputs({{"outer", a.outerProduct(b)}}).physicalOutputs();

    const ExprNode& matmulNode = outputNode(outputs);
    ASSERT_EQ(matmulNode.op, ExprOp::MATMUL);
    ASSERT_NE(matmulNode.lhs, UINT32_MAX);
    ASSERT_NE(matmulNode.rhs, UINT32_MAX);

    const ExprNode& lhsUnsqueeze = outputs.expr->nodes.at(matmulNode.lhs);
    const ExprNode& rhsUnsqueeze = outputs.expr->nodes.at(matmulNode.rhs);
    ASSERT_EQ(lhsUnsqueeze.op, ExprOp::UNSQUEEZE);
    ASSERT_EQ(rhsUnsqueeze.op, ExprOp::UNSQUEEZE);
    EXPECT_EQ(lhsUnsqueeze.unsqueeze_axes, std::vector<uint64_t>{1});
    EXPECT_EQ(rhsUnsqueeze.unsqueeze_axes, std::vector<uint64_t>{0});
    EXPECT_FALSE(matmulNode.transpose_lhs);
    EXPECT_FALSE(matmulNode.transpose_rhs);
}

TEST(ExpressionConvenienceOps, ClampWithScalarBoundsProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 3}, {-3.0f, -1.0f, 0.0f, 1.5f, 2.0f, 5.0f}, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").clamp(-1.0, 2.0)}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), {-1.0f, -1.0f, 0.0f, 1.5f, 2.0f, 2.0f});
}

TEST(ExpressionConvenienceOps, ClampWithExpressionBoundsBroadcastsAndProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 3}, {-3.0f, 0.5f, 4.0f, -0.5f, 2.0f, 8.0f}, stream);
    Tensor lower = makeGpuTensor({1, 3}, {-1.0f, 0.0f, 2.0f}, stream);
    Tensor upper = makeGpuTensor({1, 3}, {0.0f, 1.0f, 5.0f}, stream);

    auto expression_outputs = Expression::outputs({
        {"y", Expression::clamp(Expression::input("x"), Expression::input("lower"), Expression::input("upper"))},
    });
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}, {"lower", lower}, {"upper", upper}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), {-1.0f, 0.5f, 4.0f, -0.5f, 1.0f, 5.0f});
}

TEST(ExpressionConvenienceOps, DotProductProducesExpectedScalarValue) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({4}, {1.0f, 2.0f, -3.0f, 0.5f}, stream);
    Tensor b = makeGpuTensor({4}, {4.0f, -5.0f, 6.0f, 8.0f}, stream);

    auto expression_outputs = Expression::outputs({{"dot", Expression::dotProduct(Expression::input("a"), Expression::input("b"))}});
    Tensor dot = runExpressionOutput(expression_outputs, {{"a", a}, {"b", b}}, "dot", stream);

    EXPECT_EQ(dot.getDimensions(), (std::vector<uint64_t>{1}));
    expectNear(copyToCpuValues(dot, stream), {-20.0f});
}

TEST(ExpressionConvenienceOps, OuterProductProducesExpectedMatrixValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({3}, {1.0f, -3.0f, 0.5f}, stream);
    Tensor b = makeGpuTensor({2}, {4.0f, -2.0f}, stream);

    auto expression_outputs = Expression::outputs({{"outer", Expression::input("a").outerProduct(Expression::input("b"))}});
    Tensor outer = runExpressionOutput(expression_outputs, {{"a", a}, {"b", b}}, "outer", stream);

    EXPECT_EQ(outer.getDimensions(), (std::vector<uint64_t>{3, 2}));
    expectNear(copyToCpuValues(outer, stream), {4.0f, -2.0f, -12.0f, 6.0f, 2.0f, -1.0f});
}

TEST(ExpressionRoundingOps, LowerToUnaryExpressionNodes) {
    auto x = Expression::input("x");

    const std::vector<std::pair<ExprOp, Expression>> cases = {
        {ExprOp::CEIL, x.ceil()},
        {ExprOp::FLOOR, x.floor()},
        {ExprOp::ROUND, x.round()},
        {ExprOp::TRUNC, x.trunc()},
    };

    for (const auto& [expectedOp, expr] : cases) {
        auto outputs = Expression::outputs({{"y", expr}}).physicalOutputs();
        const ExprNode& node = outputNode(outputs);
        EXPECT_EQ(node.op, expectedOp);
        ASSERT_NE(node.lhs, UINT32_MAX);
        EXPECT_EQ(outputs.expr->nodes.at(node.lhs).op, ExprOp::INPUT);
    }
}

TEST(ExpressionRoundingOps, StayInSingleFusedStage) {
    auto x = Expression::input("x");
    auto y = x.ceil() + x.floor() + x.round() + x.trunc();

    auto outputs = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP32});

    const auto stages = EquationCompiler::splitAtReductionBoundaries(outputs);
    ASSERT_EQ(stages.size(), 1);
    EXPECT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
}

TEST(ExpressionRoundingOps, AutodiffRejectsNondifferentiableRoundingOps) {
    auto x = Expression::input("x");

    const std::vector<Expression> cases = {x.ceil(), x.floor(), x.round(), x.trunc()};
    for (const auto& expr : cases) {
        auto outputs = Expression::outputs({{"y", expr}}).physicalOutputs();
        EXPECT_THROW((void)buildBackwardOutputs(outputs, {"x"}), std::runtime_error);
    }
}

TEST(ExpressionRoundingOps, CeilProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor x = makeGpuTensor({3, 4}, {-2.7f, -2.5f, -2.1f, -1.0f, -0.5f, -0.1f, 0.0f, 0.1f, 0.5f, 1.5f, 2.1f, 2.7f}, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").ceil()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{3, 4}));
    expectNear(copyToCpuValues(y, stream), {-2.0f, -2.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 3.0f, 3.0f});
}

TEST(ExpressionRoundingOps, FloorProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor x = makeGpuTensor({3, 4}, {-2.7f, -2.5f, -2.1f, -1.0f, -0.5f, -0.1f, 0.0f, 0.1f, 0.5f, 1.5f, 2.1f, 2.7f}, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").floor()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{3, 4}));
    expectNear(copyToCpuValues(y, stream), {-3.0f, -3.0f, -3.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 2.0f});
}

TEST(ExpressionRoundingOps, RoundProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor x = makeGpuTensor({3, 4}, {-2.7f, -2.5f, -2.1f, -1.0f, -0.5f, -0.1f, 0.0f, 0.1f, 0.5f, 1.5f, 2.1f, 2.7f}, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").round()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{3, 4}));
    expectNear(copyToCpuValues(y, stream), {-3.0f, -3.0f, -2.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 2.0f, 3.0f});
}

TEST(ExpressionRoundingOps, TruncProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor x = makeGpuTensor({3, 4}, {-2.7f, -2.5f, -2.1f, -1.0f, -0.5f, -0.1f, 0.0f, 0.1f, 0.5f, 1.5f, 2.1f, 2.7f}, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").trunc()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{3, 4}));
    expectNear(copyToCpuValues(y, stream), {-2.0f, -2.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 2.0f});
}

TEST(ExpressionTrigOps, PrimitiveCircularTrigOpsLowerToUnaryExpressionNodes) {
    auto x = Expression::input("x");

    const std::vector<std::pair<ExprOp, Expression>> cases = {
        {ExprOp::SIN, x.sin()},
        {ExprOp::COS, x.cos()},
        {ExprOp::TAN, x.tan()},
        {ExprOp::ASIN, x.asin()},
        {ExprOp::ACOS, x.acos()},
        {ExprOp::ATAN, x.atan()},
    };

    for (const auto& [expectedOp, expr] : cases) {
        auto outputs = Expression::outputs({{"y", expr}}).physicalOutputs();
        const ExprNode& node = outputNode(outputs);
        EXPECT_EQ(node.op, expectedOp);
        ASSERT_NE(node.lhs, UINT32_MAX);
        EXPECT_EQ(outputs.expr->nodes.at(node.lhs).op, ExprOp::INPUT);
    }
}

TEST(ExpressionTrigOps, ReciprocalCircularTrigHelpersLowerToDocumentedCompositeGraphs) {
    auto x = Expression::input("x");

    const std::vector<std::pair<ExprOp, Expression>> cases = {
        {ExprOp::SIN, x.csc()},
        {ExprOp::COS, x.sec()},
        {ExprOp::TAN, x.cot()},
    };

    for (const auto& [expectedInnerOp, expr] : cases) {
        auto outputs = Expression::outputs({{"y", expr}}).physicalOutputs();
        const ExprNode& divNode = outputNode(outputs);
        ASSERT_EQ(divNode.op, ExprOp::DIV);
        ASSERT_NE(divNode.lhs, UINT32_MAX);
        ASSERT_NE(divNode.rhs, UINT32_MAX);

        const ExprNode& numeratorNode = outputs.expr->nodes.at(divNode.lhs);
        ASSERT_EQ(numeratorNode.op, ExprOp::SCALAR_FP);
        EXPECT_DOUBLE_EQ(numeratorNode.scalar_fp, 1.0);

        const ExprNode& innerNode = outputs.expr->nodes.at(divNode.rhs);
        ASSERT_EQ(innerNode.op, expectedInnerOp);
        ASSERT_NE(innerNode.lhs, UINT32_MAX);
        EXPECT_EQ(outputs.expr->nodes.at(innerNode.lhs).op, ExprOp::INPUT);
    }
}

TEST(ExpressionTrigOps, InverseReciprocalCircularTrigHelpersLowerToDocumentedCompositeGraphs) {
    auto x = Expression::input("x");

    const std::vector<std::pair<ExprOp, Expression>> cases = {
        {ExprOp::ASIN, x.acsc()},
        {ExprOp::ACOS, x.asec()},
        {ExprOp::ATAN, x.acot()},
    };

    for (const auto& [expectedRootOp, expr] : cases) {
        auto outputs = Expression::outputs({{"y", expr}}).physicalOutputs();
        const ExprNode& rootNode = outputNode(outputs);
        ASSERT_EQ(rootNode.op, expectedRootOp);
        ASSERT_NE(rootNode.lhs, UINT32_MAX);

        const ExprNode& divNode = outputs.expr->nodes.at(rootNode.lhs);
        ASSERT_EQ(divNode.op, ExprOp::DIV);
        ASSERT_NE(divNode.lhs, UINT32_MAX);
        ASSERT_NE(divNode.rhs, UINT32_MAX);

        const ExprNode& numeratorNode = outputs.expr->nodes.at(divNode.lhs);
        ASSERT_EQ(numeratorNode.op, ExprOp::SCALAR_FP);
        EXPECT_DOUBLE_EQ(numeratorNode.scalar_fp, 1.0);
        EXPECT_EQ(outputs.expr->nodes.at(divNode.rhs).op, ExprOp::INPUT);
    }
}

TEST(ExpressionTrigOps, CircularTrigOpsStayInSingleFusedStage) {
    auto x = Expression::input("x");
    auto y = x.sin() + x.cos() + x.tan() + x.asin() + x.acos() + x.atan() + x.csc() + x.sec() + x.cot() + x.acsc() + x.asec() + x.acot();

    auto outputs = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP32});

    const auto stages = EquationCompiler::splitAtReductionBoundaries(outputs);
    ASSERT_EQ(stages.size(), 1);
    EXPECT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
}

TEST(ExpressionTrigOps, CircularTrigPrimitiveAutodiffRulesAreSupported) {
    auto x = Expression::input("x");
    auto y = x.sin() + x.cos() + x.tan() + x.asin() + x.acos() + x.atan();

    auto outputs = Expression::outputs({{"y", y}}).physicalOutputs();
    EXPECT_NO_THROW((void)buildBackwardOutputs(outputs, {"x"}));
}

TEST(ExpressionTrigOps, SinProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f};
    Tensor x = makeGpuTensor({7}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").sin()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::sin(v); }), 1.0e-5f);
}

TEST(ExpressionTrigOps, CosProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f};
    Tensor x = makeGpuTensor({7}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").cos()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::cos(v); }), 1.0e-5f);
}

TEST(ExpressionTrigOps, TanProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-0.75f, -0.5f, -0.25f, 0.25f, 0.5f, 0.75f};
    Tensor x = makeGpuTensor({2, 3}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").tan()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::tan(v); }), 1.0e-5f);
}

TEST(ExpressionTrigOps, AsinProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f};
    Tensor x = makeGpuTensor({7}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").asin()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::asin(v); }), 1.0e-5f);
}

TEST(ExpressionTrigOps, AcosProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f};
    Tensor x = makeGpuTensor({7}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").acos()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::acos(v); }), 1.0e-5f);
}

TEST(ExpressionTrigOps, AtanProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-3.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 3.0f};
    Tensor x = makeGpuTensor({7}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").atan()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::atan(v); }), 1.0e-5f);
}

TEST(ExpressionTrigOps, CscProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-1.25f, -0.75f, -0.25f, 0.25f, 0.75f, 1.25f};
    Tensor x = makeGpuTensor({2, 3}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").csc()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return 1.0 / std::sin(v); }), 2.0e-5f);
}

TEST(ExpressionTrigOps, SecProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-1.25f, -0.75f, -0.25f, 0.25f, 0.75f, 1.25f};
    Tensor x = makeGpuTensor({2, 3}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").sec()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return 1.0 / std::cos(v); }), 2.0e-5f);
}

TEST(ExpressionTrigOps, CotProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-1.25f, -0.75f, -0.25f, 0.25f, 0.75f, 1.25f};
    Tensor x = makeGpuTensor({2, 3}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").cot()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return 1.0 / std::tan(v); }), 2.0e-5f);
}

TEST(ExpressionTrigOps, AcscProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-4.0f, -2.0f, -1.25f, 1.25f, 2.0f, 4.0f};
    Tensor x = makeGpuTensor({2, 3}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").acsc()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::asin(1.0 / v); }), 1.0e-5f);
}

TEST(ExpressionTrigOps, AsecProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-4.0f, -2.0f, -1.25f, 1.25f, 2.0f, 4.0f};
    Tensor x = makeGpuTensor({2, 3}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").asec()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::acos(1.0 / v); }), 1.0e-5f);
}

TEST(ExpressionTrigOps, AcotProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-4.0f, -2.0f, -0.5f, 0.5f, 2.0f, 4.0f};
    Tensor x = makeGpuTensor({2, 3}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").acot()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::atan(1.0 / v); }), 1.0e-5f);
}

TEST(ExpressionHyperbolicTrigOps, PrimitiveHyperbolicTrigOpsLowerToUnaryExpressionNodes) {
    auto x = Expression::input("x");

    const std::vector<std::pair<ExprOp, Expression>> cases = {
        {ExprOp::SINH, x.sinh()},
        {ExprOp::COSH, x.cosh()},
        {ExprOp::ASINH, x.asinh()},
        {ExprOp::ACOSH, x.acosh()},
        {ExprOp::ATANH, x.atanh()},
    };

    for (const auto& [expectedOp, expr] : cases) {
        auto outputs = Expression::outputs({{"y", expr}}).physicalOutputs();
        const ExprNode& node = outputNode(outputs);
        EXPECT_EQ(node.op, expectedOp);
        ASSERT_NE(node.lhs, UINT32_MAX);
        EXPECT_EQ(outputs.expr->nodes.at(node.lhs).op, ExprOp::INPUT);
    }
}

TEST(ExpressionHyperbolicTrigOps, ReciprocalHyperbolicTrigHelpersLowerToDocumentedCompositeGraphs) {
    auto x = Expression::input("x");

    const std::vector<std::pair<ExprOp, Expression>> cases = {
        {ExprOp::SINH, x.csch()},
        {ExprOp::COSH, x.sech()},
        {ExprOp::TANH, x.coth()},
    };

    for (const auto& [expectedInnerOp, expr] : cases) {
        auto outputs = Expression::outputs({{"y", expr}}).physicalOutputs();
        const ExprNode& divNode = outputNode(outputs);
        ASSERT_EQ(divNode.op, ExprOp::DIV);
        ASSERT_NE(divNode.lhs, UINT32_MAX);
        ASSERT_NE(divNode.rhs, UINT32_MAX);

        const ExprNode& numeratorNode = outputs.expr->nodes.at(divNode.lhs);
        ASSERT_EQ(numeratorNode.op, ExprOp::SCALAR_FP);
        EXPECT_DOUBLE_EQ(numeratorNode.scalar_fp, 1.0);

        const ExprNode& innerNode = outputs.expr->nodes.at(divNode.rhs);
        ASSERT_EQ(innerNode.op, expectedInnerOp);
        ASSERT_NE(innerNode.lhs, UINT32_MAX);
        EXPECT_EQ(outputs.expr->nodes.at(innerNode.lhs).op, ExprOp::INPUT);
    }
}

TEST(ExpressionHyperbolicTrigOps, InverseReciprocalHyperbolicTrigHelpersLowerToDocumentedCompositeGraphs) {
    auto x = Expression::input("x");

    const std::vector<std::pair<ExprOp, Expression>> cases = {
        {ExprOp::ASINH, x.acsch()},
        {ExprOp::ACOSH, x.asech()},
        {ExprOp::ATANH, x.acoth()},
    };

    for (const auto& [expectedRootOp, expr] : cases) {
        auto outputs = Expression::outputs({{"y", expr}}).physicalOutputs();
        const ExprNode& rootNode = outputNode(outputs);
        ASSERT_EQ(rootNode.op, expectedRootOp);
        ASSERT_NE(rootNode.lhs, UINT32_MAX);

        const ExprNode& divNode = outputs.expr->nodes.at(rootNode.lhs);
        ASSERT_EQ(divNode.op, ExprOp::DIV);
        ASSERT_NE(divNode.lhs, UINT32_MAX);
        ASSERT_NE(divNode.rhs, UINT32_MAX);

        const ExprNode& numeratorNode = outputs.expr->nodes.at(divNode.lhs);
        ASSERT_EQ(numeratorNode.op, ExprOp::SCALAR_FP);
        EXPECT_DOUBLE_EQ(numeratorNode.scalar_fp, 1.0);
        EXPECT_EQ(outputs.expr->nodes.at(divNode.rhs).op, ExprOp::INPUT);
    }
}

TEST(ExpressionHyperbolicTrigOps, HyperbolicTrigOpsStayInSingleFusedStage) {
    auto x = Expression::input("x");
    auto y = x.sinh() + x.cosh() + x.tanh() + x.asinh() + x.acosh() + x.atanh() + x.csch() + x.sech() + x.coth() + x.acsch() +
             x.asech() + x.acoth();

    auto outputs = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP32});

    const auto stages = EquationCompiler::splitAtReductionBoundaries(outputs);
    ASSERT_EQ(stages.size(), 1);
    EXPECT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
}

TEST(ExpressionHyperbolicTrigOps, HyperbolicTrigPrimitiveAutodiffRulesAreSupported) {
    auto x = Expression::input("x");
    auto y = x.sinh() + x.cosh() + x.asinh() + x.acosh() + x.atanh();

    auto outputs = Expression::outputs({{"y", y}}).physicalOutputs();
    EXPECT_NO_THROW((void)buildBackwardOutputs(outputs, {"x"}));
}

TEST(ExpressionHyperbolicTrigOps, SinhProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-2.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 2.0f};
    Tensor x = makeGpuTensor({7}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").sinh()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::sinh(v); }), 2.0e-5f);
}

TEST(ExpressionHyperbolicTrigOps, CoshProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-2.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 2.0f};
    Tensor x = makeGpuTensor({7}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").cosh()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::cosh(v); }), 2.0e-5f);
}

TEST(ExpressionHyperbolicTrigOps, AsinhProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-8.0f, -2.0f, -0.25f, 0.0f, 0.25f, 2.0f, 8.0f};
    Tensor x = makeGpuTensor({7}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").asinh()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::asinh(v); }), 1.0e-5f);
}

TEST(ExpressionHyperbolicTrigOps, AcoshProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {1.0f, 1.125f, 1.5f, 2.0f, 4.0f, 8.0f};
    Tensor x = makeGpuTensor({2, 3}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").acosh()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::acosh(v); }), 1.0e-5f);
}

TEST(ExpressionHyperbolicTrigOps, AtanhProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f};
    Tensor x = makeGpuTensor({7}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").atanh()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::atanh(v); }), 1.0e-5f);
}

TEST(ExpressionHyperbolicTrigOps, CschProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f};
    Tensor x = makeGpuTensor({2, 3}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").csch()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return 1.0 / std::sinh(v); }), 2.0e-5f);
}

TEST(ExpressionHyperbolicTrigOps, SechProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    Tensor x = makeGpuTensor({7}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").sech()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return 1.0 / std::cosh(v); }), 2.0e-5f);
}

TEST(ExpressionHyperbolicTrigOps, CothProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f};
    Tensor x = makeGpuTensor({2, 3}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").coth()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return 1.0 / std::tanh(v); }), 2.0e-5f);
}

TEST(ExpressionHyperbolicTrigOps, AcschProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-4.0f, -2.0f, -0.5f, 0.5f, 2.0f, 4.0f};
    Tensor x = makeGpuTensor({2, 3}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").acsch()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::asinh(1.0 / v); }), 1.0e-5f);
}

TEST(ExpressionHyperbolicTrigOps, AsechProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {0.125f, 0.25f, 0.5f, 0.75f, 1.0f};
    Tensor x = makeGpuTensor({5}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").asech()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{5}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::acosh(1.0 / v); }), 1.0e-5f);
}

TEST(ExpressionHyperbolicTrigOps, AcothProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-4.0f, -2.0f, -1.25f, 1.25f, 2.0f, 4.0f};
    Tensor x = makeGpuTensor({2, 3}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").acoth()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::atanh(1.0 / v); }), 1.0e-5f);
}

TEST(ExpressionErrorFunctionOps, PrimitiveErrorFunctionOpsLowerToUnaryExpressionNodes) {
    auto x = Expression::input("x");

    const std::vector<std::pair<ExprOp, Expression>> cases = {
        {ExprOp::ERF, x.erf()},
        {ExprOp::ERFC, x.erfc()},
        {ExprOp::ERFCX, x.erfcx()},
        {ExprOp::ERFINV, x.erfinv()},
        {ExprOp::ERFCINV, x.erfcinv()},
    };

    for (const auto& [expectedOp, expr] : cases) {
        auto outputs = Expression::outputs({{"y", expr}}).physicalOutputs();
        const ExprNode& node = outputNode(outputs);
        EXPECT_EQ(node.op, expectedOp);
        ASSERT_NE(node.lhs, UINT32_MAX);
        EXPECT_EQ(outputs.expr->nodes.at(node.lhs).op, ExprOp::INPUT);
    }
}

TEST(ExpressionErrorFunctionOps, ErrorFunctionOpsStayInSingleFusedStage) {
    auto x = Expression::input("x");
    auto y = x.erf() + x.erfc() + x.erfcx() + x.erfinv() + x.erfcinv();

    auto outputs = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP32});

    const auto stages = EquationCompiler::splitAtReductionBoundaries(outputs);
    ASSERT_EQ(stages.size(), 1);
    EXPECT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
}

TEST(ExpressionErrorFunctionOps, ErrorFunctionPrimitiveAutodiffRulesAreSupported) {
    auto x = Expression::input("x");
    auto y = x.erf() + x.erfc() + x.erfcx() + x.erfinv() + x.erfcinv();

    auto outputs = Expression::outputs({{"y", y}}).physicalOutputs();
    EXPECT_NO_THROW((void)buildBackwardOutputs(outputs, {"x"}));
}

TEST(ExpressionErrorFunctionOps, ErfProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    Tensor x = makeGpuTensor({7}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").erf()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::erf(v); }), 1.0e-5f);
}

TEST(ExpressionErrorFunctionOps, ErfcProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    Tensor x = makeGpuTensor({7}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").erfc()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::erfc(v); }), 1.0e-5f);
}

TEST(ExpressionErrorFunctionOps, ErfcxProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f};
    Tensor x = makeGpuTensor({7}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").erfcx()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::exp(v * v) * std::erfc(v); }), 1.0e-4f);
}

TEST(ExpressionErrorFunctionOps, ErfinvProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-0.9f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.9f};
    Tensor x = makeGpuTensor({7}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").erfinv()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(y, stream), {-1.163087154f, -0.476936276f, -0.225312055f, 0.0f, 0.225312055f, 0.476936276f, 1.163087154f}, 2.0e-5f);
}

TEST(ExpressionErrorFunctionOps, ErfcinvProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {0.1f, 0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.9f};
    Tensor x = makeGpuTensor({2, 4}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").erfcinv()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 4}));
    expectNear(copyToCpuValues(y, stream), {1.163087154f, 0.813419848f, 0.476936276f, 0.225312055f, 0.0f, -0.225312055f, -0.476936276f, -1.163087154f}, 2.0e-5f);
}

TEST(ExpressionErrorFunctionOps, InverseErrorFunctionsRoundTripThroughForwardFunctions) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> erf_values = {-0.9f, -0.5f, -0.1f, 0.0f, 0.1f, 0.5f, 0.9f};
    Tensor erf_x = makeGpuTensor({7}, erf_values, stream);
    auto erf_outputs = Expression::outputs({{"y", Expression::input("x").erfinv().erf()}});
    Tensor erf_roundtrip = runExpressionOutput(erf_outputs, {{"x", erf_x}}, "y", stream);

    EXPECT_EQ(erf_roundtrip.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(erf_roundtrip, stream), erf_values, 2.0e-5f);

    const std::vector<float> erfc_values = {0.1f, 0.25f, 0.5f, 0.9f, 1.1f, 1.5f, 1.9f};
    Tensor erfc_x = makeGpuTensor({7}, erfc_values, stream);
    auto erfc_outputs = Expression::outputs({{"y", Expression::input("x").erfcinv().erfc()}});
    Tensor erfc_roundtrip = runExpressionOutput(erfc_outputs, {{"x", erfc_x}}, "y", stream);

    EXPECT_EQ(erfc_roundtrip.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(erfc_roundtrip, stream), erfc_values, 3.0e-5f);
}


TEST(ExpressionGammaFunctionOps, PrimitiveGammaFunctionOpsLowerToUnaryExpressionNodes) {
    auto x = Expression::input("x");

    const std::vector<std::pair<ExprOp, Expression>> cases = {
        {ExprOp::TGAMMA, x.tgamma()},
        {ExprOp::LGAMMA, x.lgamma()},
        {ExprOp::DIGAMMA, x.digamma()},
    };

    for (const auto& [expectedOp, expr] : cases) {
        auto outputs = Expression::outputs({{"y", expr}}).physicalOutputs();
        const ExprNode& node = outputNode(outputs);
        EXPECT_EQ(node.op, expectedOp);
        ASSERT_NE(node.lhs, UINT32_MAX);
        EXPECT_EQ(outputs.expr->nodes.at(node.lhs).op, ExprOp::INPUT);
    }
}

TEST(ExpressionGammaFunctionOps, GammaFunctionOpsStayInSingleFusedStage) {
    auto x = Expression::input("x");
    auto y = x.tgamma() + x.lgamma() + x.digamma();

    auto outputs = Expression::outputs({{"y", y}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP32});

    const auto stages = EquationCompiler::splitAtReductionBoundaries(outputs);
    ASSERT_EQ(stages.size(), 1);
    EXPECT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
}

TEST(ExpressionGammaFunctionOps, TgammaAndLgammaAutodiffUseDigamma) {
    auto x = Expression::input("x");
    auto y = x.tgamma() + x.lgamma();

    auto outputs = Expression::outputs({{"y", y}}).physicalOutputs();
    auto backward = buildBackwardOutputs(outputs, {"x"});

    ASSERT_TRUE(backward.expr != nullptr);
    bool foundDigamma = false;
    for (const ExprNode& node : backward.expr->nodes) {
        foundDigamma = foundDigamma || node.op == ExprOp::DIGAMMA;
    }
    EXPECT_TRUE(foundDigamma);
}

TEST(ExpressionGammaFunctionOps, DigammaAutodiffRejectsUntilTrigammaExists) {
    auto x = Expression::input("x");
    auto outputs = Expression::outputs({{"y", x.digamma()}}).physicalOutputs();

    EXPECT_THROW((void)buildBackwardOutputs(outputs, {"x"}), std::runtime_error);
}

TEST(ExpressionGammaFunctionOps, TgammaProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-0.5f, 0.5f, 1.0f, 1.5f, 2.0f, 3.5f, 5.0f};
    Tensor x = makeGpuTensor({7}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").tgamma()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{7}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::tgamma(v); }), 2.0e-5f);
}

TEST(ExpressionGammaFunctionOps, LgammaProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-0.5f, 0.25f, 0.5f, 1.0f, 1.5f, 2.0f, 3.5f, 5.0f};
    Tensor x = makeGpuTensor({2, 4}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").lgamma()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 4}));
    expectNear(copyToCpuValues(y, stream), mapFloatValues(values, [](double v) { return std::lgamma(v); }), 2.0e-5f);
}

TEST(ExpressionGammaFunctionOps, DigammaProducesExpectedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<float> values = {-0.5f, 0.25f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 8.0f};
    Tensor x = makeGpuTensor({9}, values, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").digamma()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{9}));
    expectNear(copyToCpuValues(y, stream), {
        0.036489974f,
        -4.227453709f,
        -1.963510036f,
        -0.577215672f,
        0.036489974f,
        0.422784328f,
        0.922784328f,
        1.256117702f,
        2.015641451f,
    }, 2.0e-5f);
}

TEST(ExpressionGammaFunctionOps, DigammaProducesNanAtNonpositiveIntegerPoles) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor x = makeGpuTensor({4}, {0.0f, -1.0f, -2.0f, -3.0f}, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("x").digamma()}});
    Tensor y = runExpressionOutput(expression_outputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{4}));
    const std::vector<float> values = copyToCpuValues(y, stream);
    ASSERT_EQ(values.size(), 4U);
    for (float value : values) {
        EXPECT_TRUE(std::isnan(value));
    }
}
