#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/NewtonSchulzOrthogonalization.h"

#include "cuda_runtime.h"

#include "gtest/gtest.h"

#include <cmath>
#include <functional>
#include <limits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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

Tensor makeGpuUint32Tensor(const std::vector<uint64_t>& dims, const std::vector<uint32_t>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::UINT32, dims));
    if (tensorNumel(cpu) != values.size()) {
        throw std::runtime_error("makeGpuUint32Tensor value count mismatch.");
    }

    auto* ptr = static_cast<uint32_t*>(cpu.getMemPtr());
    for (size_t i = 0; i < values.size(); ++i) {
        ptr[i] = values[i];
    }

    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::UINT32, dims));
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

std::unordered_map<std::string, std::vector<float>> runBackwardValues(const Outputs& forward_outputs,
                                                                     const std::unordered_map<std::string, Tensor>& inputs,
                                                                     const std::vector<std::string>& wrt_names,
                                                                     const std::string& upstream_input_name,
                                                                     Stream& stream) {
    FusedEquation forward = FusedEquation::compile(forward_outputs.physicalOutputs(), 0);
    FusedEquation backward = forward.compileBackward(wrt_names, upstream_input_name);

    StampedExecutionPlan plan = backward.stamp(inputs, stream);
    plan.run();

    std::unordered_map<std::string, std::vector<float>> gradients;
    for (const std::string& wrt_name : wrt_names) {
        const std::string grad_name = wrt_name + "_grad";
        gradients.emplace(grad_name, copyToCpuValues(plan.output(grad_name), stream));
    }
    return gradients;
}

template <typename DerivativeFn>
std::vector<float> expectedElementwiseGradient(const std::vector<float>& values,
                                               const std::vector<float>& upstream,
                                               DerivativeFn&& derivative_fn) {
    if (values.size() != upstream.size()) {
        throw std::runtime_error("expectedElementwiseGradient value count mismatch.");
    }

    std::vector<float> expected;
    expected.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        expected.push_back(static_cast<float>(static_cast<double>(upstream[i]) * derivative_fn(static_cast<double>(values[i]))));
    }
    return expected;
}

std::vector<float> scaleDerivatives(const std::vector<float>& derivative_values, const std::vector<float>& upstream) {
    if (derivative_values.size() != upstream.size()) {
        throw std::runtime_error("scaleDerivatives value count mismatch.");
    }

    std::vector<float> expected;
    expected.reserve(derivative_values.size());
    for (size_t i = 0; i < derivative_values.size(); ++i) {
        expected.push_back(derivative_values[i] * upstream[i]);
    }
    return expected;
}

void expectUnaryBackwardValues(const std::string& case_name,
                               const std::vector<float>& values,
                               const std::vector<float>& upstream,
                               const std::function<Expression(const Expression&)>& expression_fn,
                               const std::vector<float>& expected_grad,
                               float atol,
                               Stream& stream) {
    SCOPED_TRACE(case_name);
    Tensor x = makeGpuTensor({static_cast<uint64_t>(values.size())}, values, stream);
    Tensor dy = makeGpuTensor({static_cast<uint64_t>(upstream.size())}, upstream, stream);

    auto forward_outputs = Expression::outputs({{"y", expression_fn(Expression::input("x"))}});
    auto gradients = runBackwardValues(forward_outputs, {{"x", x}, {"dy", dy}}, {"x"}, "dy", stream);

    expectNear(gradients.at("x_grad"), expected_grad, atol);
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

TEST(ExpressionConvenienceOps, ClampWithExpressionBoundsBroadcastsBackwardGradients) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 3}, {-2.0f, -0.5f, 3.0f, -0.25f, 2.5f, 0.5f}, stream);
    Tensor lower = makeGpuTensor({1, 3}, {-1.0f, -1.0f, -1.0f}, stream);
    Tensor upper = makeGpuTensor({1, 3}, {1.0f, 1.0f, 1.0f}, stream);
    Tensor dy = makeGpuTensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, stream);

    auto forward_outputs = Expression::outputs({{
        "y", Expression::clamp(Expression::input("x"), Expression::input("lower"), Expression::input("upper")),
    }});
    auto gradients = runBackwardValues(forward_outputs, {{"x", x}, {"lower", lower}, {"upper", upper}, {"dy", dy}}, {"x", "lower", "upper"}, "dy", stream);

    expectNear(gradients.at("x_grad"), {0.0f, 2.0f, 0.0f, 4.0f, 0.0f, 6.0f});
    expectNear(gradients.at("lower_grad"), {1.0f, 0.0f, 0.0f});
    expectNear(gradients.at("upper_grad"), {0.0f, 5.0f, 3.0f});
}


TEST(ExpressionConvenienceOps, TakeAlongAxisLowersAsIndexAwareBinaryOp) {
    auto values = Expression::input("values");
    auto indices = Expression::input("indices");

    auto outputs = Expression::outputs({{"y", values.takeAlongAxis(indices, 1)}}).physicalOutputs();

    const ExprNode& takeNode = outputNode(outputs);
    ASSERT_EQ(takeNode.op, ExprOp::TAKE_ALONG_AXIS);
    EXPECT_EQ(takeNode.reduction_axes, (std::vector<uint64_t>{1}));
    ASSERT_NE(takeNode.lhs, UINT32_MAX);
    ASSERT_NE(takeNode.rhs, UINT32_MAX);
    EXPECT_EQ(outputs.expr->nodes.at(takeNode.lhs).op, ExprOp::INPUT);
    EXPECT_EQ(outputs.expr->nodes.at(takeNode.rhs).op, ExprOp::INPUT);

    resolveOutputsDTypesInPlace(outputs, {DataType::FP32, DataType::UINT32});
    const auto stages = EquationCompiler::splitAtReductionBoundaries(outputs);
    ASSERT_EQ(stages.size(), 1);
    EXPECT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
}

TEST(ExpressionConvenienceOps, TakeAlongAxisRejectsNonIntegralIndices) {
    auto values = Expression::input("values");
    auto indices = Expression::input("indices");
    auto outputs = Expression::outputs({{"y", values.takeAlongAxis(indices)}}).physicalOutputs();

    EXPECT_THROW(resolveOutputsDTypesInPlace(outputs, {DataType::FP32, DataType::FP32}), std::runtime_error);
}

TEST(ExpressionConvenienceOps, TakeAlongAxisGathersAlongLastAxis) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor values = makeGpuTensor({2, 4}, {10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f}, stream);
    Tensor indices = makeGpuUint32Tensor({2, 3}, {3, 0, 2, 1, 1, 0}, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("values").takeAlongAxis(Expression::input("indices"))}});
    Tensor y = runExpressionOutput(expression_outputs, {{"values", values}, {"indices", indices}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), {13.0f, 10.0f, 12.0f, 21.0f, 21.0f, 20.0f});
}

TEST(ExpressionConvenienceOps, TakeAlongAxisGathersAlongLeadingAxis) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor values = makeGpuTensor({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, stream);
    Tensor indices = makeGpuUint32Tensor({2, 2}, {2, 0, 0, 1}, stream);

    auto expression_outputs = Expression::outputs({{"y", Expression::input("values").takeAlongAxis(Expression::input("indices"), 0)}});
    Tensor y = runExpressionOutput(expression_outputs, {{"values", values}, {"indices", indices}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 2}));
    expectNear(copyToCpuValues(y, stream), {5.0f, 2.0f, 1.0f, 4.0f});
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

TEST(ExpressionConvenienceOps, DotProductBackwardProducesExpectedGradients) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({4}, {1.0f, 2.0f, -3.0f, 0.5f}, stream);
    Tensor b = makeGpuTensor({4}, {4.0f, -5.0f, 6.0f, 8.0f}, stream);
    Tensor dy = makeGpuTensor({1}, {2.5f}, stream);

    auto forward_outputs = Expression::outputs({{"dot", Expression::dotProduct(Expression::input("a"), Expression::input("b"))}});
    auto gradients = runBackwardValues(forward_outputs, {{"a", a}, {"b", b}, {"dy", dy}}, {"a", "b"}, "dy", stream);

    expectNear(gradients.at("a_grad"), {10.0f, -12.5f, 15.0f, 20.0f});
    expectNear(gradients.at("b_grad"), {2.5f, 5.0f, -7.5f, 1.25f});
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

TEST(ExpressionConvenienceOps, OuterProductBackwardProducesExpectedGradients) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor a = makeGpuTensor({3}, {1.0f, -3.0f, 0.5f}, stream);
    Tensor b = makeGpuTensor({2}, {4.0f, -2.0f}, stream);
    Tensor dy = makeGpuTensor({3, 2}, {1.0f, 2.0f, -3.0f, 4.0f, 5.0f, -6.0f}, stream);

    auto forward_outputs = Expression::outputs({{"outer", Expression::input("a").outerProduct(Expression::input("b"))}});
    auto gradients = runBackwardValues(forward_outputs, {{"a", a}, {"b", b}, {"dy", dy}}, {"a", "b"}, "dy", stream);

    expectNear(gradients.at("a_grad"), {0.0f, -20.0f, 32.0f});
    expectNear(gradients.at("b_grad"), {12.5f, -13.0f});
}

TEST(ExpressionBooleanComparisonOps, ComparisonsLowerToBinaryExpressionNodes) {
    auto x = Expression::input("x");
    auto y = Expression::input("y");

    const std::vector<std::pair<ExprOp, Expression>> cases = {
        {ExprOp::EQUAL, x.equal(y)},
        {ExprOp::NOT_EQUAL, x.notEqual(y)},
        {ExprOp::LESS, x.lessThan(y)},
        {ExprOp::LESS_EQUAL, x.lessEqual(y)},
        {ExprOp::GREATER, x.greaterThan(y)},
        {ExprOp::GREATER_EQUAL, x.greaterEqual(y)},
        {ExprOp::EQUAL, x == y},
        {ExprOp::NOT_EQUAL, x != y},
        {ExprOp::LESS, x < y},
        {ExprOp::LESS_EQUAL, x <= y},
        {ExprOp::GREATER, x > y},
        {ExprOp::GREATER_EQUAL, x >= y},
    };

    for (const auto& [expectedOp, expr] : cases) {
        auto outputs = Expression::outputs({{"y", expr}}).physicalOutputs();
        const ExprNode& node = outputNode(outputs);
        EXPECT_EQ(node.op, expectedOp);
        ASSERT_NE(node.lhs, UINT32_MAX);
        ASSERT_NE(node.rhs, UINT32_MAX);
        EXPECT_EQ(outputs.expr->nodes.at(node.lhs).op, ExprOp::INPUT);
        EXPECT_EQ(outputs.expr->nodes.at(node.rhs).op, ExprOp::INPUT);
    }
}

TEST(ExpressionBooleanComparisonOps, BooleanOutputsDefaultToBooleanAndCanBeOverridden) {
    auto x = Expression::input("x");
    auto y = Expression::input("y");

    auto bool_outputs = Expression::outputs({{"mask", x.greaterEqual(y)}}).physicalOutputs();
    resolveOutputsDTypesInPlace(bool_outputs, {DataType::FP32, DataType::FP32});
    const ExprNode& bool_node = outputNode(bool_outputs);
    ASSERT_TRUE(bool_node.output_dtype.has_value());
    EXPECT_EQ(bool_node.output_dtype.value(), DataType::BOOLEAN);
    ASSERT_TRUE(bool_node.compute_dtype.has_value());
    EXPECT_EQ(bool_node.compute_dtype.value(), DataType::FP32);

    auto fp32_outputs = Expression::outputs({{"mask", x.greaterEqual(y).withOutputDType(DataType::FP32)}}).physicalOutputs();
    resolveOutputsDTypesInPlace(fp32_outputs, {DataType::FP32, DataType::FP32});
    const ExprNode& fp32_node = outputNode(fp32_outputs);
    ASSERT_TRUE(fp32_node.output_dtype.has_value());
    EXPECT_EQ(fp32_node.output_dtype.value(), DataType::FP32);
    ASSERT_TRUE(fp32_node.compute_dtype.has_value());
    EXPECT_EQ(fp32_node.compute_dtype.value(), DataType::FP32);

    auto integer_outputs = Expression::outputs({{"mask", x.equal(y)}}).physicalOutputs();
    resolveOutputsDTypesInPlace(integer_outputs, {DataType::UINT32, DataType::UINT32});
    const ExprNode& integer_node = outputNode(integer_outputs);
    ASSERT_TRUE(integer_node.output_dtype.has_value());
    EXPECT_EQ(integer_node.output_dtype.value(), DataType::BOOLEAN);
    ASSERT_TRUE(integer_node.compute_dtype.has_value());
    EXPECT_EQ(integer_node.compute_dtype.value(), DataType::FP32);
}

TEST(ExpressionBooleanComparisonOps, LogicalOpsComposeBooleanMasksInSingleFusedStage) {
    auto x = Expression::input("x");
    auto y = Expression::input("y");
    auto mask = x.greaterEqual(y).logicalAnd(x.notEqual(y)).logicalOr(!x.lessThan(y));

    auto outputs = Expression::outputs({{"mask", mask}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP32, DataType::FP32});

    const ExprNode& root = outputNode(outputs);
    ASSERT_EQ(root.op, ExprOp::LOGICAL_OR);
    ASSERT_TRUE(root.output_dtype.has_value());
    EXPECT_EQ(root.output_dtype.value(), DataType::BOOLEAN);
    ASSERT_TRUE(root.compute_dtype.has_value());
    EXPECT_EQ(root.compute_dtype.value(), DataType::BOOLEAN);

    const auto stages = EquationCompiler::splitAtReductionBoundaries(outputs);
    ASSERT_EQ(stages.size(), 1);
    EXPECT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
}

TEST(ExpressionBooleanComparisonOps, AutodiffRejectsNondifferentiableBooleanOps) {
    auto x = Expression::input("x");
    auto y = Expression::input("y");

    const std::vector<Expression> cases = {
        x.equal(y).withOutputDType(DataType::FP32),
        x.notEqual(y).withOutputDType(DataType::FP32),
        x.lessThan(y).withOutputDType(DataType::FP32),
        x.lessEqual(y).withOutputDType(DataType::FP32),
        x.greaterThan(y).withOutputDType(DataType::FP32),
        x.greaterEqual(y).withOutputDType(DataType::FP32),
        x.greaterEqual(y).logicalAnd(x.notEqual(y)).withOutputDType(DataType::FP32),
        x.greaterEqual(y).logicalOr(x.notEqual(y)).withOutputDType(DataType::FP32),
        x.greaterEqual(y).logicalNot().withOutputDType(DataType::FP32),
    };

    for (const auto& expr : cases) {
        auto outputs = Expression::outputs({{"y", expr}}).physicalOutputs();
        EXPECT_THROW((void)buildBackwardOutputs(outputs, {"x"}), std::runtime_error);
    }
}

TEST(ExpressionBooleanComparisonOps, ComparisonProducesExpectedValuesWhenCastedToFp32) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 4}, {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f, 4.0f}, stream);
    Tensor y = makeGpuTensor({2, 4}, {-2.0f, 0.0f, 0.0f, 1.0f, 0.5f, 2.0f, 4.0f, 3.0f}, stream);

    auto expression_outputs = Expression::outputs({{"mask", Expression::input("x").greaterEqual(Expression::input("y")).withOutputDType(DataType::FP32)}});
    Tensor mask = runExpressionOutput(expression_outputs, {{"x", x}, {"y", y}}, "mask", stream);

    EXPECT_EQ(mask.getDimensions(), (std::vector<uint64_t>{2, 4}));
    expectNear(copyToCpuValues(mask, stream), {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});
}

TEST(ExpressionBooleanComparisonOps, LogicalCompositionProducesExpectedValuesWhenCastedToFp32) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor x = makeGpuTensor({6}, {-1.0f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f}, stream);
    Tensor label = makeGpuTensor({6}, {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f}, stream);

    auto prediction = Expression::input("x").greaterEqual(Expression::constantScalar(0.5));
    auto truth = Expression::input("label").notEqual(Expression::constantScalar(0.0));
    auto correct = prediction.equal(truth).withOutputDType(DataType::FP32);
    auto expression_outputs = Expression::outputs({{"correct", correct}});
    Tensor out = runExpressionOutput(expression_outputs, {{"x", x}, {"label", label}}, "correct", stream);

    EXPECT_EQ(out.getDimensions(), (std::vector<uint64_t>{6}));
    expectNear(copyToCpuValues(out, stream), {1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f});
}


TEST(ExpressionWhereSelectOps, WhereAndSelectLowerToTernaryExpressionNodes) {
    auto x = Expression::input("x");
    auto y = Expression::input("y");
    auto condition = x.greaterThan(Expression::constantScalar(0.0));

    auto where_outputs = Expression::outputs({{"z", Expression::where(condition, x, y)}}).physicalOutputs();
    const ExprNode& where_node = outputNode(where_outputs);
    EXPECT_EQ(where_node.op, ExprOp::WHERE);
    ASSERT_NE(where_node.lhs, UINT32_MAX);
    ASSERT_NE(where_node.rhs, UINT32_MAX);
    ASSERT_NE(where_node.aux, UINT32_MAX);
    EXPECT_EQ(where_outputs.expr->nodes.at(where_node.lhs).op, ExprOp::GREATER);
    EXPECT_EQ(where_outputs.expr->nodes.at(where_node.rhs).op, ExprOp::INPUT);
    EXPECT_EQ(where_outputs.expr->nodes.at(where_node.aux).op, ExprOp::INPUT);

    auto select_outputs = Expression::outputs({{"z", condition.select(x, y)}}).physicalOutputs();
    const ExprNode& select_node = outputNode(select_outputs);
    EXPECT_EQ(select_node.op, ExprOp::WHERE);
    ASSERT_NE(select_node.lhs, UINT32_MAX);
    ASSERT_NE(select_node.rhs, UINT32_MAX);
    ASSERT_NE(select_node.aux, UINT32_MAX);
}

TEST(ExpressionWhereSelectOps, DefaultsToBranchDTypeAndRequiresBooleanCondition) {
    auto x = Expression::input("x");
    auto y = Expression::input("y");
    auto condition = x.greaterEqual(y);

    auto outputs = Expression::outputs({{"z", Expression::where(condition, x, y)}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP32, DataType::FP32});
    const ExprNode& node = outputNode(outputs);
    ASSERT_TRUE(node.output_dtype.has_value());
    EXPECT_EQ(node.output_dtype.value(), DataType::FP32);
    ASSERT_TRUE(node.compute_dtype.has_value());
    EXPECT_EQ(node.compute_dtype.value(), DataType::FP32);

    auto bool_outputs = Expression::outputs({{"z", Expression::where(condition, condition, condition.logicalNot())}}).physicalOutputs();
    resolveOutputsDTypesInPlace(bool_outputs, {DataType::FP32, DataType::FP32});
    const ExprNode& bool_node = outputNode(bool_outputs);
    ASSERT_TRUE(bool_node.output_dtype.has_value());
    EXPECT_EQ(bool_node.output_dtype.value(), DataType::BOOLEAN);
    ASSERT_TRUE(bool_node.compute_dtype.has_value());
    EXPECT_EQ(bool_node.compute_dtype.value(), DataType::BOOLEAN);

    auto invalid_outputs = Expression::outputs({{"z", Expression::where(x, x, y)}}).physicalOutputs();
    EXPECT_THROW(resolveOutputsDTypesInPlace(invalid_outputs, {DataType::FP32, DataType::FP32}), std::runtime_error);
}

TEST(ExpressionWhereSelectOps, ComposesWithBooleanMasksInSingleFusedStage) {
    auto x = Expression::input("x");
    auto y = Expression::input("y");
    auto condition = x.greaterThan(Expression::constantScalar(0.0)).logicalAnd(y.lessThan(Expression::constantScalar(10.0)));
    auto z = Expression::where(condition, x * y, x - y);

    auto outputs = Expression::outputs({{"z", z}}).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP32, DataType::FP32});

    const ExprNode& root = outputNode(outputs);
    ASSERT_EQ(root.op, ExprOp::WHERE);
    ASSERT_TRUE(root.output_dtype.has_value());
    EXPECT_EQ(root.output_dtype.value(), DataType::FP32);
    ASSERT_TRUE(root.compute_dtype.has_value());
    EXPECT_EQ(root.compute_dtype.value(), DataType::FP32);

    const auto stages = EquationCompiler::splitAtReductionBoundaries(outputs);
    ASSERT_EQ(stages.size(), 1);
    EXPECT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
}

TEST(ExpressionWhereSelectOps, WhereProducesExpectedBroadcastedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f}, stream);
    Tensor fallback = makeGpuTensor({1, 3}, {10.0f, 20.0f, 30.0f}, stream);

    auto expression_outputs = Expression::outputs(
        {{"z", Expression::where(Expression::input("x").greaterThan(Expression::constantScalar(0.0)),
                                  Expression::input("x"),
                                  Expression::input("fallback"))}});
    Tensor out = runExpressionOutput(expression_outputs, {{"x", x}, {"fallback", fallback}}, "z", stream);

    EXPECT_EQ(out.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(out, stream), {10.0f, 20.0f, 30.0f, 1.0f, 2.0f, 3.0f});
}

TEST(ExpressionWhereSelectOps, WhereBackwardMasksBranchGradientsAndIgnoresConditionGradient) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor x = makeGpuTensor({6}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f}, stream);
    Tensor dy = makeGpuTensor({6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, stream);

    auto x_expr = Expression::input("x");
    auto forward_outputs = Expression::outputs({{"z", Expression::where(x_expr.greaterThan(Expression::constantScalar(0.0)),
                                                                          x_expr * x_expr,
                                                                          x_expr * Expression::constantScalar(3.0))}});
    auto gradients = runBackwardValues(forward_outputs, {{"x", x}, {"dy", dy}}, {"x"}, "dy", stream);

    expectNear(gradients.at("x_grad"), {3.0f, 6.0f, 9.0f, 8.0f, 20.0f, 36.0f});
}


TEST(ExpressionGraphConditionalOps, ConditionalOutputsExposeContractAndRejectMismatchedBranchNames) {
    auto x = Expression::input("x");
    auto predicate = Expression::input("predicate_value").greaterThan(Expression::constantScalar(0.0));

    Outputs then_outputs = Expression::outputs({{"y", x + Expression::constantScalar(1.0)}});
    Outputs else_outputs = Expression::outputs({{"y", x - Expression::constantScalar(1.0)}});
    Outputs conditional = Outputs::conditional(predicate, then_outputs, else_outputs);

    EXPECT_TRUE(conditional.isConditional());
    PhysicalOutputs physical = conditional.physicalOutputs();
    ASSERT_TRUE(physical.isConditional());
    ASSERT_NE(physical.conditional, nullptr);
    ASSERT_EQ(physical.outputs.size(), 1);
    EXPECT_EQ(physical.outputs[0].name, "y");

    Outputs mismatched_else = Expression::outputs({{"z", x - Expression::constantScalar(1.0)}});
    EXPECT_THROW((void)Outputs::conditional(predicate, then_outputs, mismatched_else), std::runtime_error);
}

TEST(ExpressionGraphConditionalOps, CompileBackwardAndExpressionDefinitionRejectConditionalOutputs) {
    auto x = Expression::input("x");
    auto predicate = Expression::input("predicate_value").greaterThan(Expression::constantScalar(0.0));
    Outputs conditional = Outputs::conditional(predicate,
                                               Expression::outputs({{"y", x + Expression::constantScalar(1.0)}}),
                                               Expression::outputs({{"y", x - Expression::constantScalar(1.0)}}));

    EXPECT_THROW((void)ExpressionDefinition::fromOutputs(conditional), std::runtime_error);

    FusedEquation equation = FusedEquation::compile(conditional.physicalOutputs(), 0);
    EXPECT_THROW((void)equation.compileBackward({"x"}, "dy"), std::runtime_error);
}

TEST(ExpressionGraphConditionalOps, RunsThenAndElseBranchesWithDeviceScalarPredicate) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    auto x_expr = Expression::input("x");
    auto predicate = Expression::input("predicate_value").greaterThan(Expression::constantScalar(0.0));
    Outputs conditional = Outputs::conditional(predicate,
                                               Expression::outputs({{"y", x_expr + Expression::constantScalar(10.0)}}),
                                               Expression::outputs({{"y", x_expr - Expression::constantScalar(10.0)}}));
    FusedEquation equation = FusedEquation::compile(conditional.physicalOutputs(), 0);

    Tensor x = makeGpuTensor({4}, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    Tensor true_predicate = makeGpuTensor({1}, {1.0f}, stream);
    StampedExecutionPlan true_plan = equation.stamp({{"x", x}, {"predicate_value", true_predicate}}, stream);
    EXPECT_EQ(true_plan.stageKindNames(), (std::vector<std::string>{"Conditional"}));
    true_plan.run();
    expectNear(copyToCpuValues(true_plan.output("y"), stream), {11.0f, 12.0f, 13.0f, 14.0f});

    Tensor false_predicate = makeGpuTensor({1}, {-1.0f}, stream);
    StampedExecutionPlan false_plan = equation.stamp({{"x", x}, {"predicate_value", false_predicate}}, stream);
    EXPECT_EQ(false_plan.stageKindNames(), (std::vector<std::string>{"Conditional"}));
    false_plan.run();
    expectNear(copyToCpuValues(false_plan.output("y"), stream), {-9.0f, -8.0f, -7.0f, -6.0f});
}

TEST(ExpressionGraphConditionalOps, MultiOutputBranchesShareFinalOutputTensorsAndSelectCorrectBranch) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    auto x_expr = Expression::input("x");
    auto predicate = Expression::input("predicate_value").greaterThan(Expression::constantScalar(0.0));
    Outputs then_outputs = Expression::outputs({
        {"sum", x_expr + Expression::constantScalar(5.0)},
        {"scaled", x_expr * Expression::constantScalar(2.0)},
    });
    Outputs else_outputs = Expression::outputs({
        {"sum", x_expr - Expression::constantScalar(5.0)},
        {"scaled", x_expr * Expression::constantScalar(3.0)},
    });
    FusedEquation equation = FusedEquation::compile(Outputs::conditional(predicate, then_outputs, else_outputs).physicalOutputs(), 0);

    Tensor x = makeGpuTensor({3}, {1.0f, 2.0f, 3.0f}, stream);
    Tensor true_predicate = makeGpuTensor({1}, {1.0f}, stream);
    StampedExecutionPlan true_plan = equation.stamp({{"x", x}, {"predicate_value", true_predicate}}, stream);
    true_plan.run();
    expectNear(copyToCpuValues(true_plan.output("sum"), stream), {6.0f, 7.0f, 8.0f});
    expectNear(copyToCpuValues(true_plan.output("scaled"), stream), {2.0f, 4.0f, 6.0f});

    Tensor false_predicate = makeGpuTensor({1}, {-1.0f}, stream);
    StampedExecutionPlan false_plan = equation.stamp({{"x", x}, {"predicate_value", false_predicate}}, stream);
    false_plan.run();
    expectNear(copyToCpuValues(false_plan.output("sum"), stream), {-4.0f, -3.0f, -2.0f});
    expectNear(copyToCpuValues(false_plan.output("scaled"), stream), {3.0f, 6.0f, 9.0f});
}

TEST(ExpressionGraphConditionalOps, StampRejectsNonBooleanPredicateOutput) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    auto x_expr = Expression::input("x");
    Outputs conditional = Outputs::conditional(Expression::input("predicate_value"),
                                               Expression::outputs({{"y", x_expr + Expression::constantScalar(1.0)}}),
                                               Expression::outputs({{"y", x_expr - Expression::constantScalar(1.0)}}));
    FusedEquation equation = FusedEquation::compile(conditional.physicalOutputs(), 0);

    Tensor x = makeGpuTensor({2}, {1.0f, 2.0f}, stream);
    Tensor predicate = makeGpuTensor({1}, {1.0f}, stream);
    EXPECT_THROW((void)equation.stamp({{"x", x}, {"predicate_value", predicate}}, stream), std::runtime_error);
}

TEST(ExpressionGraphConditionalOps, StampRejectsNonScalarPredicateOutput) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    auto x_expr = Expression::input("x");
    auto predicate = Expression::input("predicate_value").greaterThan(Expression::constantScalar(0.0));
    Outputs conditional = Outputs::conditional(predicate,
                                               Expression::outputs({{"y", x_expr + Expression::constantScalar(1.0)}}),
                                               Expression::outputs({{"y", x_expr - Expression::constantScalar(1.0)}}));
    FusedEquation equation = FusedEquation::compile(conditional.physicalOutputs(), 0);

    Tensor x = makeGpuTensor({2}, {1.0f, 2.0f}, stream);
    Tensor predicate_values = makeGpuTensor({2}, {1.0f, -1.0f}, stream);
    EXPECT_THROW((void)equation.stamp({{"x", x}, {"predicate_value", predicate_values}}, stream), std::runtime_error);
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

TEST(ExpressionTrigOps, CircularTrigPrimitiveBackwardProducesExpectedGradients) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::vector<float> small_values = {-0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f};
    const std::vector<float> small_upstream = {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 2.5f, -0.75f};

    expectUnaryBackwardValues("sin",
                              small_values,
                              small_upstream,
                              [](const Expression& x) { return x.sin(); },
                              expectedElementwiseGradient(small_values, small_upstream, [](double v) { return std::cos(v); }),
                              2.0e-5f,
                              stream);
    expectUnaryBackwardValues("cos",
                              small_values,
                              small_upstream,
                              [](const Expression& x) { return x.cos(); },
                              expectedElementwiseGradient(small_values, small_upstream, [](double v) { return -std::sin(v); }),
                              2.0e-5f,
                              stream);
    expectUnaryBackwardValues("tan",
                              small_values,
                              small_upstream,
                              [](const Expression& x) { return x.tan(); },
                              expectedElementwiseGradient(small_values, small_upstream, [](double v) {
                                  const double c = std::cos(v);
                                  return 1.0 / (c * c);
                              }),
                              3.0e-5f,
                              stream);
    expectUnaryBackwardValues("asin",
                              small_values,
                              small_upstream,
                              [](const Expression& x) { return x.asin(); },
                              expectedElementwiseGradient(small_values, small_upstream, [](double v) {
                                  return 1.0 / std::sqrt(1.0 - v * v);
                              }),
                              3.0e-5f,
                              stream);
    expectUnaryBackwardValues("acos",
                              small_values,
                              small_upstream,
                              [](const Expression& x) { return x.acos(); },
                              expectedElementwiseGradient(small_values, small_upstream, [](double v) {
                                  return -1.0 / std::sqrt(1.0 - v * v);
                              }),
                              3.0e-5f,
                              stream);

    const std::vector<float> atan_values = {-3.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 3.0f};
    const std::vector<float> atan_upstream = {0.25f, -1.0f, 2.0f, -3.0f, 1.5f, -0.5f, 4.0f};
    expectUnaryBackwardValues("atan",
                              atan_values,
                              atan_upstream,
                              [](const Expression& x) { return x.atan(); },
                              expectedElementwiseGradient(atan_values, atan_upstream, [](double v) { return 1.0 / (1.0 + v * v); }),
                              2.0e-5f,
                              stream);
}

TEST(ExpressionTrigOps, ReciprocalCircularTrigBackwardProducesExpectedGradients) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::vector<float> reciprocal_values = {-1.25f, -0.75f, -0.25f, 0.25f, 0.75f, 1.25f};
    const std::vector<float> reciprocal_upstream = {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 2.5f};

    expectUnaryBackwardValues("csc",
                              reciprocal_values,
                              reciprocal_upstream,
                              [](const Expression& x) { return x.csc(); },
                              expectedElementwiseGradient(reciprocal_values, reciprocal_upstream, [](double v) {
                                  const double s = std::sin(v);
                                  return -std::cos(v) / (s * s);
                              }),
                              5.0e-4f,
                              stream);
    expectUnaryBackwardValues("sec",
                              reciprocal_values,
                              reciprocal_upstream,
                              [](const Expression& x) { return x.sec(); },
                              expectedElementwiseGradient(reciprocal_values, reciprocal_upstream, [](double v) {
                                  const double c = std::cos(v);
                                  return std::sin(v) / (c * c);
                              }),
                              5.0e-4f,
                              stream);
    expectUnaryBackwardValues("cot",
                              reciprocal_values,
                              reciprocal_upstream,
                              [](const Expression& x) { return x.cot(); },
                              expectedElementwiseGradient(reciprocal_values, reciprocal_upstream, [](double v) {
                                  const double s = std::sin(v);
                                  return -1.0 / (s * s);
                              }),
                              5.0e-4f,
                              stream);

    const std::vector<float> inverse_reciprocal_values = {-4.0f, -2.0f, -1.25f, 1.25f, 2.0f, 4.0f};
    const std::vector<float> inverse_reciprocal_upstream = {-1.0f, 0.5f, -2.0f, 3.0f, -0.75f, 1.25f};
    expectUnaryBackwardValues("acsc",
                              inverse_reciprocal_values,
                              inverse_reciprocal_upstream,
                              [](const Expression& x) { return x.acsc(); },
                              expectedElementwiseGradient(inverse_reciprocal_values, inverse_reciprocal_upstream, [](double v) {
                                  return (-1.0 / (v * v)) / std::sqrt(1.0 - 1.0 / (v * v));
                              }),
                              2.0e-4f,
                              stream);
    expectUnaryBackwardValues("asec",
                              inverse_reciprocal_values,
                              inverse_reciprocal_upstream,
                              [](const Expression& x) { return x.asec(); },
                              expectedElementwiseGradient(inverse_reciprocal_values, inverse_reciprocal_upstream, [](double v) {
                                  return (1.0 / (v * v)) / std::sqrt(1.0 - 1.0 / (v * v));
                              }),
                              2.0e-4f,
                              stream);

    const std::vector<float> acot_values = {-4.0f, -2.0f, -0.5f, 0.5f, 2.0f, 4.0f};
    const std::vector<float> acot_upstream = {1.25f, -0.75f, 3.0f, -2.0f, 0.5f, -1.0f};
    expectUnaryBackwardValues("acot",
                              acot_values,
                              acot_upstream,
                              [](const Expression& x) { return x.acot(); },
                              expectedElementwiseGradient(acot_values, acot_upstream, [](double v) { return -1.0 / (1.0 + v * v); }),
                              2.0e-4f,
                              stream);
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

TEST(ExpressionHyperbolicTrigOps, HyperbolicTrigPrimitiveBackwardProducesExpectedGradients) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::vector<float> values = {-2.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 2.0f};
    const std::vector<float> upstream = {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 2.5f, -0.75f};

    expectUnaryBackwardValues("sinh",
                              values,
                              upstream,
                              [](const Expression& x) { return x.sinh(); },
                              expectedElementwiseGradient(values, upstream, [](double v) { return std::cosh(v); }),
                              4.0e-5f,
                              stream);
    expectUnaryBackwardValues("cosh",
                              values,
                              upstream,
                              [](const Expression& x) { return x.cosh(); },
                              expectedElementwiseGradient(values, upstream, [](double v) { return std::sinh(v); }),
                              4.0e-5f,
                              stream);
    expectUnaryBackwardValues("tanh",
                              values,
                              upstream,
                              [](const Expression& x) { return x.tanh(); },
                              expectedElementwiseGradient(values, upstream, [](double v) {
                                  const double t = std::tanh(v);
                                  return 1.0 - t * t;
                              }),
                              4.0e-5f,
                              stream);

    const std::vector<float> asinh_values = {-8.0f, -2.0f, -0.25f, 0.0f, 0.25f, 2.0f, 8.0f};
    const std::vector<float> asinh_upstream = {-1.0f, 0.5f, -2.0f, 3.0f, -0.75f, 1.25f, 2.0f};
    expectUnaryBackwardValues("asinh",
                              asinh_values,
                              asinh_upstream,
                              [](const Expression& x) { return x.asinh(); },
                              expectedElementwiseGradient(asinh_values, asinh_upstream, [](double v) { return 1.0 / std::sqrt(v * v + 1.0); }),
                              4.0e-5f,
                              stream);

    const std::vector<float> acosh_values = {1.125f, 1.5f, 2.0f, 4.0f, 8.0f};
    const std::vector<float> acosh_upstream = {1.0f, -2.0f, 0.5f, -0.75f, 1.25f};
    expectUnaryBackwardValues("acosh",
                              acosh_values,
                              acosh_upstream,
                              [](const Expression& x) { return x.acosh(); },
                              expectedElementwiseGradient(acosh_values, acosh_upstream, [](double v) {
                                  return 1.0 / (std::sqrt(v - 1.0) * std::sqrt(v + 1.0));
                              }),
                              6.0e-5f,
                              stream);

    const std::vector<float> atanh_values = {-0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f};
    const std::vector<float> atanh_upstream = {0.25f, -1.0f, 2.0f, -3.0f, 1.5f, -0.5f, 4.0f};
    expectUnaryBackwardValues("atanh",
                              atanh_values,
                              atanh_upstream,
                              [](const Expression& x) { return x.atanh(); },
                              expectedElementwiseGradient(atanh_values, atanh_upstream, [](double v) { return 1.0 / (1.0 - v * v); }),
                              5.0e-5f,
                              stream);
}

TEST(ExpressionHyperbolicTrigOps, ReciprocalHyperbolicTrigBackwardProducesExpectedGradients) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::vector<float> reciprocal_values = {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f};
    const std::vector<float> reciprocal_upstream = {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 2.5f};

    expectUnaryBackwardValues("csch",
                              reciprocal_values,
                              reciprocal_upstream,
                              [](const Expression& x) { return x.csch(); },
                              expectedElementwiseGradient(reciprocal_values, reciprocal_upstream, [](double v) {
                                  const double s = std::sinh(v);
                                  return -std::cosh(v) / (s * s);
                              }),
                              8.0e-5f,
                              stream);
    expectUnaryBackwardValues("coth",
                              reciprocal_values,
                              reciprocal_upstream,
                              [](const Expression& x) { return x.coth(); },
                              expectedElementwiseGradient(reciprocal_values, reciprocal_upstream, [](double v) {
                                  const double s = std::sinh(v);
                                  return -1.0 / (s * s);
                              }),
                              8.0e-5f,
                              stream);

    const std::vector<float> sech_values = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    const std::vector<float> sech_upstream = {-1.0f, 0.5f, -2.0f, 3.0f, -0.75f, 1.25f, 2.0f};
    expectUnaryBackwardValues("sech",
                              sech_values,
                              sech_upstream,
                              [](const Expression& x) { return x.sech(); },
                              expectedElementwiseGradient(sech_values, sech_upstream, [](double v) {
                                  const double c = std::cosh(v);
                                  return -std::sinh(v) / (c * c);
                              }),
                              6.0e-5f,
                              stream);

    const std::vector<float> inverse_values = {-4.0f, -2.0f, -0.5f, 0.5f, 2.0f, 4.0f};
    const std::vector<float> inverse_upstream = {1.25f, -0.75f, 3.0f, -2.0f, 0.5f, -1.0f};
    expectUnaryBackwardValues("acsch",
                              inverse_values,
                              inverse_upstream,
                              [](const Expression& x) { return x.acsch(); },
                              expectedElementwiseGradient(inverse_values, inverse_upstream, [](double v) {
                                  return (-1.0 / (v * v)) / std::sqrt(1.0 + 1.0 / (v * v));
                              }),
                              6.0e-5f,
                              stream);

    const std::vector<float> asech_values = {0.125f, 0.25f, 0.5f, 0.75f};
    const std::vector<float> asech_upstream = {1.0f, -2.0f, 0.5f, -0.75f};
    expectUnaryBackwardValues("asech",
                              asech_values,
                              asech_upstream,
                              [](const Expression& x) { return x.asech(); },
                              expectedElementwiseGradient(asech_values, asech_upstream, [](double v) {
                                  const double inv = 1.0 / v;
                                  return (-1.0 / (v * v)) / (std::sqrt(inv - 1.0) * std::sqrt(inv + 1.0));
                              }),
                              8.0e-5f,
                              stream);

    const std::vector<float> acoth_values = {-4.0f, -2.0f, -1.25f, 1.25f, 2.0f, 4.0f};
    const std::vector<float> acoth_upstream = {-1.0f, 0.5f, -2.0f, 3.0f, -0.75f, 1.25f};
    expectUnaryBackwardValues("acoth",
                              acoth_values,
                              acoth_upstream,
                              [](const Expression& x) { return x.acoth(); },
                              expectedElementwiseGradient(acoth_values, acoth_upstream, [](double v) { return -1.0 / (v * v - 1.0); }),
                              8.0e-5f,
                              stream);
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

TEST(ExpressionErrorFunctionOps, ErrorFunctionBackwardProducesExpectedGradients) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr double two_over_sqrt_pi = 1.1283791670955126;
    constexpr double sqrt_pi_over_two = 0.8862269254527580;

    const std::vector<float> values = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f};
    const std::vector<float> upstream = {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 2.5f, -0.75f};

    expectUnaryBackwardValues("erf",
                              values,
                              upstream,
                              [](const Expression& x) { return x.erf(); },
                              expectedElementwiseGradient(values, upstream, [two_over_sqrt_pi](double v) {
                                  return two_over_sqrt_pi * std::exp(-(v * v));
                              }),
                              5.0e-5f,
                              stream);
    expectUnaryBackwardValues("erfc",
                              values,
                              upstream,
                              [](const Expression& x) { return x.erfc(); },
                              expectedElementwiseGradient(values, upstream, [two_over_sqrt_pi](double v) {
                                  return -two_over_sqrt_pi * std::exp(-(v * v));
                              }),
                              5.0e-5f,
                              stream);
    expectUnaryBackwardValues("erfcx",
                              values,
                              upstream,
                              [](const Expression& x) { return x.erfcx(); },
                              expectedElementwiseGradient(values, upstream, [two_over_sqrt_pi](double v) {
                                  return 2.0 * v * std::exp(v * v) * std::erfc(v) - two_over_sqrt_pi;
                              }),
                              1.0e-3f,
                              stream);

    const std::vector<float> erfinv_values = {-0.9f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.9f};
    const std::vector<float> erfinv_upstream = {-1.0f, 0.5f, -2.0f, 3.0f, -0.75f, 1.25f, 2.0f};
    const std::vector<float> erfinv_outputs = {-1.163087154f, -0.476936276f, -0.225312055f, 0.0f, 0.225312055f, 0.476936276f, 1.163087154f};
    std::vector<float> erfinv_derivatives;
    erfinv_derivatives.reserve(erfinv_outputs.size());
    for (float inverse : erfinv_outputs) {
        erfinv_derivatives.push_back(static_cast<float>(sqrt_pi_over_two * std::exp(static_cast<double>(inverse) * inverse)));
    }
    expectUnaryBackwardValues("erfinv",
                              erfinv_values,
                              erfinv_upstream,
                              [](const Expression& x) { return x.erfinv(); },
                              scaleDerivatives(erfinv_derivatives, erfinv_upstream),
                              1.0e-4f,
                              stream);

    const std::vector<float> erfcinv_values = {0.1f, 0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.9f};
    const std::vector<float> erfcinv_upstream = {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 2.5f, -0.75f, 1.25f};
    const std::vector<float> erfcinv_outputs = {1.163087154f, 0.813419848f, 0.476936276f, 0.225312055f,
                                                0.0f, -0.225312055f, -0.476936276f, -1.163087154f};
    std::vector<float> erfcinv_derivatives;
    erfcinv_derivatives.reserve(erfcinv_outputs.size());
    for (float inverse : erfcinv_outputs) {
        erfcinv_derivatives.push_back(static_cast<float>(-sqrt_pi_over_two * std::exp(static_cast<double>(inverse) * inverse)));
    }
    expectUnaryBackwardValues("erfcinv",
                              erfcinv_values,
                              erfcinv_upstream,
                              [](const Expression& x) { return x.erfcinv(); },
                              scaleDerivatives(erfcinv_derivatives, erfcinv_upstream),
                              1.0e-4f,
                              stream);
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

TEST(ExpressionGammaFunctionOps, TgammaAndLgammaBackwardProduceExpectedGradients) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr double euler_gamma = 0.5772156649015329;

    const std::vector<float> values = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> upstream = {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 2.5f};
    const std::vector<float> digamma_values = {
        static_cast<float>(-euler_gamma - 2.0 * std::log(2.0)),
        static_cast<float>(-euler_gamma),
        static_cast<float>(-euler_gamma - 2.0 * std::log(2.0) + 2.0),
        static_cast<float>(1.0 - euler_gamma),
        static_cast<float>(1.5 - euler_gamma),
        static_cast<float>(1.0 + 0.5 + 1.0 / 3.0 - euler_gamma),
    };

    std::vector<float> tgamma_derivatives;
    tgamma_derivatives.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        tgamma_derivatives.push_back(static_cast<float>(std::tgamma(static_cast<double>(values[i])) * digamma_values[i]));
    }

    expectUnaryBackwardValues("tgamma",
                              values,
                              upstream,
                              [](const Expression& x) { return x.tgamma(); },
                              scaleDerivatives(tgamma_derivatives, upstream),
                              3.0e-4f,
                              stream);
    expectUnaryBackwardValues("lgamma",
                              values,
                              upstream,
                              [](const Expression& x) { return x.lgamma(); },
                              scaleDerivatives(digamma_values, upstream),
                              1.0e-4f,
                              stream);
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

namespace {

std::vector<double> matmulCpu(const std::vector<double>& a,
                              uint64_t a_rows,
                              uint64_t a_cols,
                              const std::vector<double>& b,
                              uint64_t b_rows,
                              uint64_t b_cols) {
    if (a_cols != b_rows) {
        throw std::runtime_error("matmulCpu dimension mismatch.");
    }
    std::vector<double> out(a_rows * b_cols, 0.0);
    for (uint64_t i = 0; i < a_rows; ++i) {
        for (uint64_t k = 0; k < a_cols; ++k) {
            const double av = a[i * a_cols + k];
            for (uint64_t j = 0; j < b_cols; ++j) {
                out[i * b_cols + j] += av * b[k * b_cols + j];
            }
        }
    }
    return out;
}

std::vector<double> transposeCpu(const std::vector<double>& x, uint64_t rows, uint64_t cols) {
    std::vector<double> out(rows * cols);
    for (uint64_t i = 0; i < rows; ++i) {
        for (uint64_t j = 0; j < cols; ++j) {
            out[j * rows + i] = x[i * cols + j];
        }
    }
    return out;
}

std::vector<double> newtonSchulzCpu(const std::vector<float>& input,
                                    uint64_t numRows,
                                    uint64_t numCols,
                                    NewtonSchulzOrthogonalizationOptions options) {
    std::vector<double> x(input.begin(), input.end());
    if (options.transposeTallMatrices && numRows > numCols) {
        x = transposeCpu(x, numRows, numCols);
        std::swap(numRows, numCols);
    }

    double sumSquares = 0.0;
    for (double v : x) {
        sumSquares += v * v;
    }
    const double invNorm = 1.0 / (std::sqrt(sumSquares) + options.epsilon);
    for (double& v : x) {
        v *= invNorm;
    }

    for (uint32_t step = 0; step < options.numIterations; ++step) {
        if (numRows <= numCols) {
            const auto xT = transposeCpu(x, numRows, numCols);
            const auto gram = matmulCpu(x, numRows, numCols, xT, numCols, numRows);
            const auto gram2 = matmulCpu(gram, numRows, numRows, gram, numRows, numRows);
            std::vector<double> polynomial(numRows * numRows);
            for (size_t i = 0; i < polynomial.size(); ++i) {
                polynomial[i] = options.coefficientB * gram[i] + options.coefficientC * gram2[i];
            }
            const auto polynomialX = matmulCpu(polynomial, numRows, numRows, x, numRows, numCols);
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = options.coefficientA * x[i] + polynomialX[i];
            }
        } else {
            const auto xT = transposeCpu(x, numRows, numCols);
            const auto gram = matmulCpu(xT, numCols, numRows, x, numRows, numCols);
            const auto gram2 = matmulCpu(gram, numCols, numCols, gram, numCols, numCols);
            std::vector<double> polynomial(numCols * numCols);
            for (size_t i = 0; i < polynomial.size(); ++i) {
                polynomial[i] = options.coefficientB * gram[i] + options.coefficientC * gram2[i];
            }
            const auto xPolynomial = matmulCpu(x, numRows, numCols, polynomial, numCols, numCols);
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = options.coefficientA * x[i] + xPolynomial[i];
            }
        }
    }

    return x;
}

std::vector<double> newtonSchulzCpuOriginalShape(const std::vector<float>& input,
                                                 uint64_t numRows,
                                                 uint64_t numCols,
                                                 NewtonSchulzOrthogonalizationOptions options) {
    const bool transposed = options.transposeTallMatrices && numRows > numCols;
    auto out = newtonSchulzCpu(input, numRows, numCols, options);
    if (transposed) {
        out = transposeCpu(out, numCols, numRows);
    }
    return out;
}

std::vector<float> toFloatValues(const std::vector<double>& values) {
    std::vector<float> out;
    out.reserve(values.size());
    for (double value : values) {
        out.push_back(static_cast<float>(value));
    }
    return out;
}

}  // namespace


TEST(ExpressionConvenienceOps, FusedKernelConsumesNonDenseStridedInputView) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor boxes = makeGpuTensor({2, 4},
                                 {0.10f, 0.20f, 1.40f, 1.80f,
                                  2.00f, 0.50f, 3.50f, 1.90f},
                                 stream);
    Tensor x1 = boxes.aliasView({2}, {4}, 0);
    ASSERT_FALSE(x1.isDenseContiguous());

    auto x = Expression::input("x");
    auto expressionOutputs = Expression::outputs({{"y", x * Expression::constantScalar(2.0) + Expression::constantScalar(1.0)}});
    Tensor y = runExpressionOutput(expressionOutputs, {{"x", x1}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2}));
    expectNear(copyToCpuValues(y, stream), {1.20f, 5.00f});
}


TEST(ExpressionConvenienceOps, ConvenienceRunPlanCacheKeysNonDenseInputStrides) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor storage = makeGpuTensor({3, 4},
                                   {0.0f, 1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f, 7.0f,
                                    8.0f, 9.0f, 10.0f, 11.0f},
                                   stream);
    Tensor stride4 = storage.aliasView({3}, {4}, 0);
    Tensor stride2 = storage.aliasView({3}, {2}, 0);
    ASSERT_FALSE(stride4.isDenseContiguous());
    ASSERT_FALSE(stride2.isDenseContiguous());
    ASSERT_NE(stride4.getStridesElements(), stride2.getStridesElements());

    auto x = Expression::input("x");
    auto expressionOutputs = Expression::outputs({{"y", x * Expression::constantScalar(3.0)}});
    FusedEquation eq = FusedEquation::compile(expressionOutputs.physicalOutputs(), 0);

    auto stride4Plan = eq.prepareConvenienceRunPlanForInputs({{"x", stride4}});
    auto stride4PlanAgain = eq.prepareConvenienceRunPlanForInputs({{"x", stride4}});
    EXPECT_EQ(stride4Plan.get(), stride4PlanAgain.get());

    auto stride2Plan = eq.prepareConvenienceRunPlanForInputs({{"x", stride2}});
    EXPECT_NE(stride4Plan.get(), stride2Plan.get());

    Tensor yStride4(gpuPlacement, TensorDescriptor(DataType::FP32, {3}));
    eq.run({{"x", stride4}}, yStride4, stream);
    expectNear(copyToCpuValues(yStride4, stream), {0.0f, 12.0f, 24.0f});

    Tensor yStride2(gpuPlacement, TensorDescriptor(DataType::FP32, {3}));
    eq.run({{"x", stride2}}, yStride2, stream);
    expectNear(copyToCpuValues(yStride2, stream), {0.0f, 6.0f, 12.0f});
}


TEST(ExpressionConvenienceOps, InternalStridedViewAliasExecutesInStampedAndConvenienceRuns) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor storage = makeGpuTensor({3, 4},
                                   {0.0f, 1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f, 7.0f,
                                    8.0f, 9.0f, 10.0f, 11.0f},
                                   stream);

    auto x = Expression::input("x");
    auto view = x.stridedView({3}, {4}, 2);
    auto expressionOutputs = Expression::outputs({{"y", view * Expression::constantScalar(2.0) + Expression::constantScalar(1.0)}});

    Tensor stamped = runExpressionOutput(expressionOutputs, {{"x", storage}}, "y", stream);
    EXPECT_EQ(stamped.getDimensions(), (std::vector<uint64_t>{3}));
    expectNear(copyToCpuValues(stamped, stream), {5.0f, 13.0f, 21.0f});

    FusedEquation eq = FusedEquation::compile(expressionOutputs.physicalOutputs(), 0);
    auto plan = eq.prepareConvenienceRunPlanForInputs({{"x", storage}});
    ASSERT_EQ(plan->stages.size(), 1);
    Tensor convenience(gpuPlacement, TensorDescriptor(DataType::FP32, {3}));
    eq.run({{"x", storage}}, convenience, stream);
    expectNear(copyToCpuValues(convenience, stream), {5.0f, 13.0f, 21.0f});
}

TEST(ExpressionConvenienceOps, InternalRankTwoStridedViewUsesBothRuntimeStrides) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor storage = makeGpuTensor({3, 5},
                                   {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                                    5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                                    10.0f, 11.0f, 12.0f, 13.0f, 14.0f},
                                   stream);

    auto x = Expression::input("x");
    auto view = x.stridedView({3, 2}, {5, 2}, 1);
    auto expressionOutputs = Expression::outputs({{"y", view + Expression::constantScalar(0.5)}});
    FusedEquation eq = FusedEquation::compile(expressionOutputs.physicalOutputs(), 0);

    Tensor y(gpuPlacement, TensorDescriptor(DataType::FP32, {3, 2}));
    eq.run({{"x", storage}}, y, stream);

    expectNear(copyToCpuValues(y, stream), {1.5f, 3.5f, 6.5f, 8.5f, 11.5f, 13.5f});
}


TEST(ExpressionConvenienceOps, InternalStridedViewCanAliasFromNonDenseRootViewBase) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor storage = makeGpuTensor({3, 5},
                                   {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                                    5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                                    10.0f, 11.0f, 12.0f, 13.0f, 14.0f},
                                   stream);
    Tensor paddedRows = storage.aliasView({3, 4}, {5, 1}, 0);
    ASSERT_FALSE(paddedRows.isDenseContiguous());

    auto x = Expression::input("x");
    auto column = x.stridedView({3}, {5}, 2);
    auto expressionOutputs = Expression::outputs({{"y", column - Expression::constantScalar(1.0)}});
    FusedEquation eq = FusedEquation::compile(expressionOutputs.physicalOutputs(), 0);

    Tensor y(gpuPlacement, TensorDescriptor(DataType::FP32, {3}));
    eq.run({{"x", paddedRows}}, y, stream);

    expectNear(copyToCpuValues(y, stream), {1.0f, 6.0f, 11.0f});
}

TEST(ExpressionConvenienceOps, NonDenseRootViewBroadcastsWithDenseInput) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor storage = makeGpuTensor({3, 4},
                                   {0.0f, 1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f, 7.0f,
                                    8.0f, 9.0f, 10.0f, 11.0f},
                                   stream);
    Tensor column = storage.aliasView({3, 1}, {4, 1}, 0);
    Tensor bias = makeGpuTensor({1, 2}, {1.0f, 2.0f}, stream);
    ASSERT_FALSE(column.isDenseContiguous());

    auto a = Expression::input("a");
    auto b = Expression::input("b");
    auto expressionOutputs = Expression::outputs({{"y", a + b}});
    FusedEquation eq = FusedEquation::compile(expressionOutputs.physicalOutputs(), 0);

    Tensor y(gpuPlacement, TensorDescriptor(DataType::FP32, {3, 2}));
    eq.run({{"a", column}, {"b", bias}}, y, stream);

    expectNear(copyToCpuValues(y, stream), {1.0f, 2.0f, 5.0f, 6.0f, 9.0f, 10.0f});
}

TEST(ExpressionConvenienceOps, MultipleInternalStridedViewsExecuteCorrectlyThroughConvenienceRun) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor storage = makeGpuTensor({2, 4},
                                   {0.0f, 1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f, 7.0f},
                                   stream);

    auto x = Expression::input("x");
    auto first = x.stridedView({2}, {4}, 0);
    auto third = x.stridedView({2}, {4}, 2);
    auto expressionOutputs = Expression::outputs({
        {"first", first + Expression::constantScalar(10.0)},
        {"third", third * Expression::constantScalar(3.0)},
    });
    FusedEquation eq = FusedEquation::compile(expressionOutputs.physicalOutputs(), 0);
    auto plan = eq.prepareConvenienceRunPlanForInputs({{"x", storage}});
    ASSERT_FALSE(plan->stages.empty());

    Tensor firstOut(gpuPlacement, TensorDescriptor(DataType::FP32, {2}));
    Tensor thirdOut(gpuPlacement, TensorDescriptor(DataType::FP32, {2}));
    std::unordered_map<std::string, Tensor> outputs{{"first", firstOut}, {"third", thirdOut}};
    eq.run({{"x", storage}}, outputs, stream);

    expectNear(copyToCpuValues(outputs.at("first"), stream), {10.0f, 14.0f});
    expectNear(copyToCpuValues(outputs.at("third"), stream), {6.0f, 18.0f});
}


TEST(ExpressionConvenienceOps, BroadcastStatsSurviveFlattenAndUnflattenReshapeFusion) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor x = makeGpuTensor({2, 3, 4},
                             {0.0f, 1.0f, 2.0f, 3.0f,
                              4.0f, 5.0f, 6.0f, 7.0f,
                              8.0f, 9.0f, 10.0f, 11.0f,
                              12.0f, 13.0f, 14.0f, 15.0f,
                              16.0f, 17.0f, 18.0f, 19.0f,
                              20.0f, 21.0f, 22.0f, 23.0f},
                             stream);
    Tensor stats = makeGpuTensor({6, 1}, {100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f}, stream);

    auto xin = Expression::input("x");
    auto stats_in = Expression::input("stats");
    auto expressionOutputs = Expression::outputs({{"y", (xin.reshape({6, 4}) + stats_in).reshape({2, 3, 4})}});
    Tensor y = runExpressionOutput(expressionOutputs, {{"x", x}, {"stats", stats}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3, 4}));
    expectNear(copyToCpuValues(y, stream),
               {100.0f, 101.0f, 102.0f, 103.0f,
                204.0f, 205.0f, 206.0f, 207.0f,
                308.0f, 309.0f, 310.0f, 311.0f,
                412.0f, 413.0f, 414.0f, 415.0f,
                516.0f, 517.0f, 518.0f, 519.0f,
                620.0f, 621.0f, 622.0f, 623.0f});
}

TEST(ExpressionConvenienceOps, BackwardReshapesPublicOutputAdjointToInternalMulShape) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor x = makeGpuTensor({6, 4},
                             {0.0f, 1.0f, 2.0f, 3.0f,
                              4.0f, 5.0f, 6.0f, 7.0f,
                              8.0f, 9.0f, 10.0f, 11.0f,
                              12.0f, 13.0f, 14.0f, 15.0f,
                              16.0f, 17.0f, 18.0f, 19.0f,
                              20.0f, 21.0f, 22.0f, 23.0f},
                             stream);
    Tensor dy = makeGpuTensor({2, 3, 4},
                              {1.0f, 0.5f, -1.0f, 2.0f,
                               1.5f, -0.5f, 0.25f, 3.0f,
                               -2.0f, 4.0f, 0.75f, -1.5f,
                               2.5f, -3.0f, 1.25f, 0.0f,
                               0.5f, 2.0f, -0.25f, 1.0f,
                               -1.0f, 0.125f, 3.5f, -2.5f},
                              stream);

    auto xin = Expression::input("x");
    auto forwardOutputs = Expression::outputs({{"y", (xin * xin).reshape({2, 3, 4})}});
    auto gradients = runBackwardValues(forwardOutputs, {{"x", x}, {"dy", dy}}, {"x"}, "dy", stream);

    expectNear(gradients.at("x_grad"),
               {0.0f, 1.0f, -4.0f, 12.0f,
                12.0f, -5.0f, 3.0f, 42.0f,
                -32.0f, 72.0f, 15.0f, -33.0f,
                60.0f, -78.0f, 35.0f, 0.0f,
                16.0f, 68.0f, -9.0f, 38.0f,
                -40.0f, 5.25f, 154.0f, -115.0f});
}

TEST(ExpressionConvenienceOps, TerminalStridedViewMaterializesIntoCallerProvidedDenseOutput) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor storage = makeGpuTensor({2, 6, 2},
                                   {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                                    12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f},
                                   stream);

    auto x = Expression::input("x");
    auto tail = x.stridedView({2, 3, 2}, {12, 2, 1}, 6);
    auto expressionOutputs = Expression::outputs({{"y", tail}});
    FusedEquation equation = FusedEquation::compile(expressionOutputs.physicalOutputs(), 0);

    Tensor denseOutput(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3, 2}));
    StampedExecutionPlan plan = equation.stamp({{"x", storage}}, stream, {}, {{"y", denseOutput}});
    EXPECT_EQ(plan.output("y"), denseOutput);
    plan.run();

    expectNear(copyToCpuValues(denseOutput, stream),
               {6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f});
}

TEST(ExpressionConvenienceOps, StridedViewBackwardScattersToDenseSourceGradient) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor storage = makeGpuTensor({3, 4},
                                   {0.0f, 1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f, 7.0f,
                                    8.0f, 9.0f, 10.0f, 11.0f},
                                   stream);
    Tensor upstream = makeGpuTensor({3}, {1.0f, 0.5f, -2.0f}, stream);

    auto x = Expression::input("x");
    auto column = x.stridedView({3}, {4}, 1);
    auto forwardOutputs = Expression::outputs({{"y", column * column}});
    auto gradients = runBackwardValues(forwardOutputs, {{"x", storage}, {"dy", upstream}}, {"x"}, "dy", stream);

    expectNear(gradients.at("x_grad"),
               {0.0f, 2.0f, 0.0f, 0.0f,
                0.0f, 5.0f, 0.0f, 0.0f,
                0.0f, -36.0f, 0.0f, 0.0f});
}

TEST(ExpressionConvenienceOps, DenseReshapeAliasRejectsNonDenseStridedSourceView) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor storage = makeGpuTensor({2, 4},
                                   {0.0f, 1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f, 7.0f},
                                   stream);

    auto x = Expression::input("x");
    auto view = x.stridedView({2}, {4}, 0).reshape({1, 2});
    auto expressionOutputs = Expression::outputs({{"y", view + Expression::constantScalar(1.0)}});
    FusedEquation eq = FusedEquation::compile(expressionOutputs.physicalOutputs(), 0);

    EXPECT_THROW((void)eq.stamp({{"x", storage}}, stream), std::runtime_error);
    Tensor y(gpuPlacement, TensorDescriptor(DataType::FP32, {1, 2}));
    EXPECT_THROW(eq.run({{"x", storage}}, y, stream), std::runtime_error);
}

TEST(ExpressionConvenienceOps, DenseValueReductionsExecuteThroughTheCentralReductionUtility) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor({2, 3}, {-2.0f, 1.0f, 3.0f, 4.0f, -1.0f, 2.0f}, stream);
    auto x = Expression::input("x");
    auto expression_outputs = Expression::outputs({
        {"sum", x.reduce_sum({1}, {1})},
        {"product", x.reduce_prod({1}, {1})},
        {"minimum", x.reduce_min({1}, {1})},
        {"maximum", x.reduce_max({1}, {1})},
        {"mean", x.reduce_mean({1}, {1})},
        {"norm1", x.reduce_norm1({1}, {1})},
        {"norm2", x.reduce_norm2({1}, {1})},
    });

    FusedEquation equation = FusedEquation::compile(expression_outputs.physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"x", input}}, stream);
    plan.run();

    expectNear(copyToCpuValues(plan.output("sum"), stream), {2.0f, 5.0f});
    expectNear(copyToCpuValues(plan.output("product"), stream), {-6.0f, -8.0f});
    expectNear(copyToCpuValues(plan.output("minimum"), stream), {-2.0f, -1.0f});
    expectNear(copyToCpuValues(plan.output("maximum"), stream), {3.0f, 4.0f});
    expectNear(copyToCpuValues(plan.output("mean"), stream), {2.0f / 3.0f, 5.0f / 3.0f});
    expectNear(copyToCpuValues(plan.output("norm1"), stream), {6.0f, 7.0f});
    expectNear(copyToCpuValues(plan.output("norm2"), stream), {std::sqrt(14.0f), std::sqrt(21.0f)});
}

TEST(NewtonSchulzOrthogonalization, RejectsInvalidOptions) {
    auto x = Expression::input("x");
    EXPECT_THROW((void)newtonSchulzOrthogonalize(x, 0, 4), std::logic_error);
    EXPECT_THROW((void)newtonSchulzOrthogonalize(x, 4, 0), std::logic_error);

    NewtonSchulzOrthogonalizationOptions options;
    options.epsilon = 0.0;
    EXPECT_THROW((void)newtonSchulzOrthogonalize(x, 4, 4, options), std::logic_error);

    options.epsilon = 1.0e-8;
    options.coefficientA = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void)newtonSchulzOrthogonalize(x, 4, 4, options), std::logic_error);
}

TEST(NewtonSchulzOrthogonalization, WideMatrixMatchesCpuReference) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t rows = 2;
    constexpr uint64_t cols = 4;
    const std::vector<float> inputValues = {
        0.8f, -0.4f, 1.2f, 0.5f,
        -1.1f, 0.3f, 0.7f, -0.9f,
    };
    Tensor x = makeGpuTensor({rows, cols}, inputValues, stream);

    NewtonSchulzOrthogonalizationOptions options;
    options.numIterations = 3;
    auto expressionOutputs = Expression::outputs({{"y", newtonSchulzOrthogonalize(Expression::input("x"), rows, cols, options)}});
    Tensor y = runExpressionOutput(expressionOutputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{rows, cols}));
    expectNear(copyToCpuValues(y, stream), toFloatValues(newtonSchulzCpuOriginalShape(inputValues, rows, cols, options)), 2.0e-4f);
}

TEST(NewtonSchulzOrthogonalization, TallMatrixTransposePathMatchesCpuReference) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t rows = 4;
    constexpr uint64_t cols = 2;
    const std::vector<float> inputValues = {
        0.8f, -0.4f,
        1.2f, 0.5f,
        -1.1f, 0.3f,
        0.7f, -0.9f,
    };
    Tensor x = makeGpuTensor({rows, cols}, inputValues, stream);

    NewtonSchulzOrthogonalizationOptions options;
    options.numIterations = 3;
    options.transposeTallMatrices = true;
    auto expressionOutputs = Expression::outputs({{"y", newtonSchulzOrthogonalize(Expression::input("x"), rows, cols, options)}});
    Tensor y = runExpressionOutput(expressionOutputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{rows, cols}));
    expectNear(copyToCpuValues(y, stream), toFloatValues(newtonSchulzCpuOriginalShape(inputValues, rows, cols, options)), 2.0e-4f);
}

TEST(NewtonSchulzOrthogonalization, TallMatrixRightPolynomialPathMatchesCpuReference) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t rows = 4;
    constexpr uint64_t cols = 2;
    const std::vector<float> inputValues = {
        0.8f, -0.4f,
        1.2f, 0.5f,
        -1.1f, 0.3f,
        0.7f, -0.9f,
    };
    Tensor x = makeGpuTensor({rows, cols}, inputValues, stream);

    NewtonSchulzOrthogonalizationOptions options;
    options.numIterations = 3;
    options.transposeTallMatrices = false;
    auto expressionOutputs = Expression::outputs({{"y", newtonSchulzOrthogonalize(Expression::input("x"), rows, cols, options)}});
    Tensor y = runExpressionOutput(expressionOutputs, {{"x", x}}, "y", stream);

    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{rows, cols}));
    expectNear(copyToCpuValues(y, stream), toFloatValues(newtonSchulzCpuOriginalShape(inputValues, rows, cols, options)), 2.0e-4f);
}
