#include "DeepLearning/Api/Optimizers/CustomOptimizer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/CustomOptimizer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
Impl::TensorPlacement gpuPlacement(Impl::TensorPlacement::MemDevices::GPU, 0);

using DataType = Impl::DataType;

void writeCpuFp32Tensor(Impl::Tensor& tensor, const std::vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::FP32);
    ASSERT_EQ(tensor.getTotalNumElements(), values.size());

    float* ptr = tensor.getMemPtr<float>();
    for (uint64_t i = 0; i < values.size(); ++i)
        ptr[i] = values[i];
}

std::vector<float> readCpuFp32Tensor(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);
    EXPECT_EQ(tensor.getDataType(), DataType::FP32);

    std::vector<float> values(tensor.getTotalNumElements());
    const float* ptr = tensor.getMemPtr<float>();
    for (uint64_t i = 0; i < values.size(); ++i)
        values[i] = ptr[i];

    return values;
}

void copyValuesToGpuFp32Tensor(Impl::Tensor& gpuTensor, const std::vector<float>& values, Stream& stream) {
    ASSERT_EQ(gpuTensor.getPlacement(), gpuPlacement);
    ASSERT_EQ(gpuTensor.getDataType(), DataType::FP32);
    ASSERT_EQ(gpuTensor.getTotalNumElements(), values.size());

    Impl::Tensor host(cpuPlacement, gpuTensor.getDescriptor());
    writeCpuFp32Tensor(host, values);

    gpuTensor.copyFromAsync(host, stream);
    stream.synchronize();
}

std::vector<float> copyGpuFp32TensorToValues(const Impl::Tensor& gpuTensor, Stream& stream) {
    EXPECT_EQ(gpuTensor.getPlacement(), gpuPlacement);
    EXPECT_EQ(gpuTensor.getDataType(), DataType::FP32);

    Impl::Tensor host = gpuTensor.clone(cpuPlacement);
    host.copyFromAsync(gpuTensor, stream);
    stream.synchronize();

    return readCpuFp32Tensor(host);
}

void expectAllClose(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 1e-5f, float rtol = 1e-5f) {
    ASSERT_EQ(actual.size(), expected.size());

    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        const float tol = atol + rtol * std::fabs(expected[i]);
        EXPECT_LE(diff, tol) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

Impl::CustomOptimizerUpdateExpression plainSgdExpression(const Impl::CustomOptimizerUpdateContext& ctx) {
    auto step = ctx.runtimeScalar("step", DataType::FP32, DataType::FP32);
    auto w = ctx.weights(DataType::FP32, DataType::FP32);
    auto g = ctx.gradient();
    return Impl::CustomOptimizerUpdateExpression{{{"weights", (w - step * g).withOutputDType(ctx.weightsTensor().getDataType())}}};
}

std::unordered_map<std::string, float> stepRuntimeScalar(float learningRate, uint32_t batchSize, const std::string& prefix) {
    const float step = learningRate / (static_cast<float>(batchSize) * Impl::Loss::getLossScalingFactor());
    return {{prefix + "step", step}};
}

}  // namespace

TEST(CustomOptimizerImplementation, DenseStatelessExpressionUpdatesWeightsNumerically) {
    Stream stream(gpuPlacement);
    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f}, stream);

    constexpr float learningRate = 0.2f;
    Impl::CustomOptimizer optimizer(
        123,
        {},
        &plainSgdExpression,
        [=](uint32_t batchSize, const std::string& prefix) { return stepRuntimeScalar(learningRate, batchSize, prefix); });

    optimizer.compile(weights, stream);
    ASSERT_TRUE(optimizer.isCompiled());
    ASSERT_TRUE(optimizer.getWeightsGradient().has_value());
    EXPECT_TRUE(optimizer.supportsDenseUpdateFusion());
    EXPECT_FALSE(optimizer.supportsSparseRowGradients());

    const uint32_t batchSize = 2;
    Impl::Tensor gradient = optimizer.getWeightsGradient().value();
    copyValuesToGpuFp32Tensor(gradient, {0.5f, -1.0f, 1.5f, -2.0f, 2.5f, -3.0f}, stream);
    optimizer.updateWeights(batchSize);
    stream.synchronize();

    const float step = learningRate / (static_cast<float>(batchSize) * Impl::Loss::getLossScalingFactor());
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), {1.0f - step * 0.5f,
                                                               -2.0f - step * -1.0f,
                                                               3.0f - step * 1.5f,
                                                               -4.0f - step * -2.0f,
                                                               5.0f - step * 2.5f,
                                                               -6.0f - step * -3.0f});
}

TEST(CustomOptimizerImplementation, DenseStatefulExpressionUpdatesWeightsAndStateNumerically) {
    Stream stream(gpuPlacement);
    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {4}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, -3.0f, 4.0f}, stream);

    constexpr float learningRate = 0.1f;
    constexpr float momentum = 0.5f;
    Impl::CustomOptimizer optimizer(
        456,
        {Impl::CustomOptimizerStateSpec::sameShapeAsWeights("velocity", DataType::FP32)},
        [=](const Impl::CustomOptimizerUpdateContext& ctx) {
            auto step = ctx.runtimeScalar("step", DataType::FP32, DataType::FP32);
            auto w = ctx.weights(DataType::FP32, DataType::FP32);
            auto g = ctx.gradient();
            auto v = ctx.state("velocity", DataType::FP32, DataType::FP32);
            auto vNext = Impl::Expression::constantScalar(momentum) * v - step * g;
            auto wNext = w + vNext;
            return Impl::CustomOptimizerUpdateExpression{{{"weights", wNext.withOutputDType(ctx.weightsTensor().getDataType())},
                                                         {"velocity", vNext.withOutputDType(DataType::FP32)}}};
        },
        [=](uint32_t batchSize, const std::string& prefix) { return stepRuntimeScalar(learningRate, batchSize, prefix); });

    optimizer.compile(weights, stream);
    ASSERT_TRUE(optimizer.hasParameter("velocity"));
    Impl::Tensor velocity = optimizer.getParameter("velocity")->getStorage().value();
    Impl::Tensor gradient = optimizer.getWeightsGradient().value();

    const float firstStep = learningRate / (1.0f * Impl::Loss::getLossScalingFactor());
    const float secondStep = learningRate / (2.0f * Impl::Loss::getLossScalingFactor());

    std::vector<float> expectedWeights{1.0f, 2.0f, -3.0f, 4.0f};
    std::vector<float> expectedVelocity{0.0f, 0.0f, 0.0f, 0.0f};

    const std::vector<float> firstGradient{1.0f, -2.0f, 3.0f, -4.0f};
    copyValuesToGpuFp32Tensor(gradient, firstGradient, stream);
    optimizer.updateWeights(/*batchSize=*/1);
    stream.synchronize();

    for (uint64_t i = 0; i < expectedWeights.size(); ++i) {
        expectedVelocity[i] = momentum * expectedVelocity[i] - firstStep * firstGradient[i];
        expectedWeights[i] += expectedVelocity[i];
    }
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expectedWeights);
    expectAllClose(copyGpuFp32TensorToValues(velocity, stream), expectedVelocity);

    const std::vector<float> secondGradient{2.0f, 2.0f, -2.0f, -2.0f};
    copyValuesToGpuFp32Tensor(gradient, secondGradient, stream);
    optimizer.updateWeights(/*batchSize=*/2);
    stream.synchronize();

    for (uint64_t i = 0; i < expectedWeights.size(); ++i) {
        expectedVelocity[i] = momentum * expectedVelocity[i] - secondStep * secondGradient[i];
        expectedWeights[i] += expectedVelocity[i];
    }
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expectedWeights);
    expectAllClose(copyGpuFp32TensorToValues(velocity, stream), expectedVelocity);
}

TEST(CustomOptimizerImplementation, CompileWithoutDenseGradientSupportsFusionSurface) {
    Stream stream(gpuPlacement);
    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));

    Impl::CustomOptimizer optimizer(789,
                                    {},
                                    &plainSgdExpression,
                                    [](uint32_t batchSize, const std::string& prefix) {
                                        return stepRuntimeScalar(0.01f, batchSize, prefix);
                                    });
    optimizer.compile(weights, stream, /*materializeDenseGradient=*/false);

    EXPECT_TRUE(optimizer.isCompiled());
    EXPECT_FALSE(optimizer.getWeightsGradient().has_value());

    auto grad = Impl::Expression::input("grad_from_layer", DataType::FP32, DataType::FP32);
    Impl::DenseOptimizerExpression expression = optimizer.toDenseUpdateExpression(weights, grad, "__test__");
    ASSERT_TRUE(expression.inputs.contains("__test__weights_in"));
    ASSERT_TRUE(expression.preallocatedOutputs.contains("weights"));
    ASSERT_EQ(expression.outputs.outputs.size(), 1u);
    EXPECT_EQ(expression.outputs.outputs[0].name, "weights");

    std::unordered_map<std::string, float> scalars = optimizer.denseUpdateRuntimeScalars(/*batchSize=*/4, "__test__");
    ASSERT_TRUE(scalars.contains("__test__step"));
    EXPECT_FLOAT_EQ(scalars.at("__test__step"), 0.01f / (4.0f * Impl::Loss::getLossScalingFactor()));
}

TEST(CustomOptimizerApi, BuilderStampsPhysicalCustomOptimizerAndRejectsSerialization) {
    std::shared_ptr<Api::CustomOptimizer> optimizer = Api::CustomOptimizer::Builder()
                                                         .state("velocity", Api::DataType::FP32)
                                                         .updateExpression([](const Impl::CustomOptimizerUpdateContext& ctx) {
                                                             auto step = ctx.runtimeScalar("step", DataType::FP32, DataType::FP32);
                                                             auto w = ctx.weights(DataType::FP32, DataType::FP32);
                                                             auto g = ctx.gradient();
                                                             auto v = ctx.state("velocity", DataType::FP32, DataType::FP32);
                                                             auto vNext = Impl::Expression::constantScalar(0.9f) * v - step * g;
                                                             return Impl::CustomOptimizerUpdateExpression{{
                                                                 {"weights", (w + vNext).withOutputDType(ctx.weightsTensor().getDataType())},
                                                                 {"velocity", vNext.withOutputDType(DataType::FP32)},
                                                             }};
                                                         })
                                                         .runtimeScalars([](uint32_t batchSize, const std::string& prefix) {
                                                             return stepRuntimeScalar(0.01f, batchSize, prefix);
                                                         })
                                                         .supportsSparseRowGradients(true)
                                                         .build();

    ASSERT_NE(optimizer, nullptr);
    EXPECT_EQ(optimizer->getType(), "CustomOptimizer");
    ASSERT_EQ(optimizer->getStateSpecs().size(), 1u);
    EXPECT_EQ(optimizer->getStateSpecs()[0].name, "velocity");
    EXPECT_TRUE(optimizer->getSupportsSparseRowGradients());
    EXPECT_THROW((void)optimizer->architectureJson(), std::runtime_error);

    std::shared_ptr<Impl::Optimizer> physicalBase = optimizer->stamp(nullptr);
    auto physicalCustom = std::dynamic_pointer_cast<Impl::CustomOptimizer>(physicalBase);
    ASSERT_NE(physicalCustom, nullptr);
    EXPECT_TRUE(physicalCustom->supportsSparseRowGradients());
}
