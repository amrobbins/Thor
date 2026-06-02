#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Lars.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

void expectMapHasValue(const std::unordered_map<std::string, float>& values, const std::string& name, float expected) {
    ASSERT_TRUE(values.contains(name)) << "Missing hyperparameter: " << name;
    EXPECT_FLOAT_EQ(values.at(name), expected) << "Hyperparameter mismatch for " << name;
}

void writeCpuFp32Tensor(Tensor& tensor, const std::vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::FP32);
    ASSERT_EQ(tensor.getTotalNumElements(), values.size());

    float* ptr = tensor.getMemPtr<float>();
    for (uint64_t i = 0; i < values.size(); ++i)
        ptr[i] = values[i];
}

std::vector<float> readCpuFp32Tensor(const Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);
    EXPECT_EQ(tensor.getDataType(), DataType::FP32);

    std::vector<float> values(tensor.getTotalNumElements());
    const float* ptr = tensor.getMemPtr<float>();
    for (uint64_t i = 0; i < values.size(); ++i)
        values[i] = ptr[i];

    return values;
}

void copyValuesToGpuFp32Tensor(Tensor& gpuTensor, const std::vector<float>& values, Stream& stream) {
    ASSERT_EQ(gpuTensor.getPlacement(), gpuPlacement);
    ASSERT_EQ(gpuTensor.getDataType(), DataType::FP32);
    ASSERT_EQ(gpuTensor.getTotalNumElements(), values.size());

    Tensor host(cpuPlacement, gpuTensor.getDescriptor());
    writeCpuFp32Tensor(host, values);
    gpuTensor.copyFromAsync(host, stream);
    stream.synchronize();
}

std::vector<float> copyGpuFp32TensorToValues(const Tensor& gpuTensor, Stream& stream) {
    EXPECT_EQ(gpuTensor.getPlacement(), gpuPlacement);
    EXPECT_EQ(gpuTensor.getDataType(), DataType::FP32);

    Tensor host = gpuTensor.clone(cpuPlacement);
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

Tensor requireOptimizerStorage(const Lars& lars, const std::string& name) {
    if (!lars.hasParameter(name))
        throw std::runtime_error("LARS optimizer missing parameter: " + name);
    std::optional<Tensor> storage = lars.getParameter(name)->getStorage();
    if (!storage.has_value())
        throw std::runtime_error("LARS optimizer parameter has no storage: " + name);
    return storage.value();
}

struct LarsReferenceState {
    std::vector<float> weights;
    std::vector<float> velocity;
};

void applyLarsReferenceStep(LarsReferenceState& state,
                            const std::vector<float>& rawGradient,
                            const std::vector<uint64_t>& dims,
                            uint32_t batchSize,
                            float alpha,
                            float momentum,
                            float weightDecay,
                            float trustCoefficient,
                            float epsilon,
                            bool useNesterovMomentum) {
    ASSERT_EQ(state.weights.size(), rawGradient.size());
    ASSERT_EQ(state.velocity.size(), rawGradient.size());
    ASSERT_GT(batchSize, 0u);

    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());

    std::vector<float> g(rawGradient.size());
    for (uint64_t i = 0; i < rawGradient.size(); ++i)
        g[i] = rawGradient[i] * invBatchLossScale;

    float trustRatio = 1.0f;
    if (dims.size() > 1) {
        double weightNormSquared = 0.0;
        double gradientNormSquared = 0.0;
        for (uint64_t i = 0; i < state.weights.size(); ++i) {
            weightNormSquared += static_cast<double>(state.weights[i]) * static_cast<double>(state.weights[i]);
            gradientNormSquared += static_cast<double>(g[i]) * static_cast<double>(g[i]);
        }
        const float weightNorm = static_cast<float>(std::sqrt(weightNormSquared));
        const float gradientNorm = static_cast<float>(std::sqrt(gradientNormSquared));
        trustRatio = trustCoefficient * weightNorm / (gradientNorm + weightDecay * weightNorm + epsilon);
    }

    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float update = g[i] + weightDecay * state.weights[i];
        const float scaledUpdate = alpha * trustRatio * update;
        const float velocityNext = momentum * state.velocity[i] + scaledUpdate;
        const float weightsUpdate = useNesterovMomentum ? momentum * velocityNext + scaledUpdate : velocityNext;
        state.weights[i] -= weightsUpdate;
        state.velocity[i] = velocityNext;
    }
}

void runLarsStep(Lars& lars, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    std::optional<Tensor> gradientOpt = lars.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());
    Tensor gradient = gradientOpt.value();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);
    lars.updateWeights(batchSize);
    stream.synchronize();
}

}  // namespace

TEST(LarsTest, ConstructorGettersSettersAndHyperParameters) {
    constexpr uint64_t id = 43;
    Lars lars(id, 0.1f, 0.9f, 0.01f, 0.001f, 1e-8f, false);

    EXPECT_EQ(lars.getId(), id);
    EXPECT_FLOAT_EQ(lars.getAlpha(), 0.1f);
    EXPECT_FLOAT_EQ(lars.getMomentum(), 0.9f);
    EXPECT_FLOAT_EQ(lars.getWeightDecay(), 0.01f);
    EXPECT_FLOAT_EQ(lars.getTrustCoefficient(), 0.001f);
    EXPECT_FLOAT_EQ(lars.getEpsilon(), 1e-8f);
    EXPECT_FALSE(lars.getUseNesterovMomentum());

    lars.setAlpha(0.2f);
    lars.setMomentum(0.8f);
    lars.setWeightDecay(0.02f);
    lars.setTrustCoefficient(0.002f);
    lars.setEpsilon(1e-6f);
    lars.setUseNesterovMomentum(true);

    EXPECT_FLOAT_EQ(lars.getAlpha(), 0.2f);
    EXPECT_FLOAT_EQ(lars.getMomentum(), 0.8f);
    EXPECT_FLOAT_EQ(lars.getWeightDecay(), 0.02f);
    EXPECT_FLOAT_EQ(lars.getTrustCoefficient(), 0.002f);
    EXPECT_FLOAT_EQ(lars.getEpsilon(), 1e-6f);
    EXPECT_TRUE(lars.getUseNesterovMomentum());

    std::unordered_map<std::string, float> params = lars.getAllHyperParameters();
    ASSERT_EQ(params.size(), 6u);
    expectMapHasValue(params, "alpha", 0.2f);
    expectMapHasValue(params, "momentum", 0.8f);
    expectMapHasValue(params, "weightDecay", 0.02f);
    expectMapHasValue(params, "trustCoefficient", 0.002f);
    expectMapHasValue(params, "epsilon", 1e-6f);
    expectMapHasValue(params, "useNesterovMomentum", 1.0f);
}

TEST(LarsTest, DenseUpdateMatchesReferenceWithLayerTrustRatio) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.2f;
    constexpr float momentum = 0.8f;
    constexpr float weightDecay = 0.01f;
    constexpr float trustCoefficient = 0.002f;
    constexpr float epsilon = 1e-6f;
    constexpr bool useNesterovMomentum = false;
    const std::vector<uint64_t> dims{2, 3};

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, dims));
    const std::vector<float> initialWeights{1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Lars lars(45, alpha, momentum, weightDecay, trustCoefficient, epsilon, useNesterovMomentum);
    lars.compile(weights, stream);
    stream.synchronize();

    Tensor velocity = requireOptimizerStorage(lars, "velocity");
    copyValuesToGpuFp32Tensor(velocity, std::vector<float>(initialWeights.size(), 0.0f), stream);

    LarsReferenceState reference{initialWeights, std::vector<float>(initialWeights.size(), 0.0f)};
    const std::vector<float> gradient1{0.5f, -0.25f, 1.0f, -1.5f, 0.75f, -0.5f};
    const std::vector<float> gradient2{-0.1f, 0.2f, -0.3f, 0.4f, -0.5f, 0.6f};

    applyLarsReferenceStep(reference, gradient1, dims, 2, alpha, momentum, weightDecay, trustCoefficient, epsilon, useNesterovMomentum);
    runLarsStep(lars, gradient1, 2, stream);
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), reference.weights, 2e-5f, 2e-5f);
    expectAllClose(copyGpuFp32TensorToValues(velocity, stream), reference.velocity, 2e-5f, 2e-5f);

    applyLarsReferenceStep(reference, gradient2, dims, 4, alpha, momentum, weightDecay, trustCoefficient, epsilon, useNesterovMomentum);
    runLarsStep(lars, gradient2, 4, stream);
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), reference.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(velocity, stream), reference.velocity, 3e-5f, 3e-5f);
}

TEST(LarsTest, OneDimensionalWeightsUseTrustRatioOfOne) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.05f;
    constexpr float momentum = 0.7f;
    constexpr float weightDecay = 0.02f;
    constexpr float trustCoefficient = 0.001f;
    constexpr float epsilon = 1e-6f;
    constexpr bool useNesterovMomentum = true;
    const std::vector<uint64_t> dims{4};

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, dims));
    const std::vector<float> initialWeights{0.5f, -1.0f, 1.5f, -2.0f};
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Lars lars(47, alpha, momentum, weightDecay, trustCoefficient, epsilon, useNesterovMomentum);
    lars.compile(weights, stream);
    stream.synchronize();

    Tensor velocity = requireOptimizerStorage(lars, "velocity");
    copyValuesToGpuFp32Tensor(velocity, std::vector<float>(initialWeights.size(), 0.0f), stream);

    LarsReferenceState reference{initialWeights, std::vector<float>(initialWeights.size(), 0.0f)};
    const std::vector<float> gradient{0.1f, -0.2f, 0.3f, -0.4f};

    applyLarsReferenceStep(reference, gradient, dims, 2, alpha, momentum, weightDecay, trustCoefficient, epsilon, useNesterovMomentum);
    runLarsStep(lars, gradient, 2, stream);

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), reference.weights, 2e-5f, 2e-5f);
    expectAllClose(copyGpuFp32TensorToValues(velocity, stream), reference.velocity, 2e-5f, 2e-5f);
}

TEST(LarsTest, SparseRowsAreRejectedBecauseLayerWideNormsAreRequired) {
    Stream stream(gpuPlacement);
    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {4, 3}));
    Lars lars(49, 0.1f, 0.9f, 0.01f, 0.001f, 1e-8f, false);

    EXPECT_FALSE(lars.supportsSparseRowGradients());
    EXPECT_FALSE(lars.supportsSparseRowUpdateFusion());
    EXPECT_THROW(lars.compileSparseRows(weights, 2, stream), std::runtime_error);
}
