#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Lamb.h"
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

Tensor requireOptimizerStorage(const Lamb& lamb, const std::string& name) {
    if (!lamb.hasParameter(name))
        throw std::runtime_error("Lamb optimizer missing parameter: " + name);
    std::optional<Tensor> storage = lamb.getParameter(name)->getStorage();
    if (!storage.has_value())
        throw std::runtime_error("Lamb optimizer parameter has no storage: " + name);
    return storage.value();
}

struct LambReferenceState {
    std::vector<float> weights;
    std::vector<float> m;
    std::vector<float> v;
    float t = 0.0f;
};

void applyLambReferenceStep(LambReferenceState& state,
                            const std::vector<float>& rawGradient,
                            const std::vector<uint64_t>& dims,
                            uint32_t batchSize,
                            float alpha,
                            float beta1,
                            float beta2,
                            float epsilon,
                            float weightDecay,
                            float trustRatioEpsilon) {
    ASSERT_EQ(state.weights.size(), rawGradient.size());
    ASSERT_EQ(state.m.size(), rawGradient.size());
    ASSERT_EQ(state.v.size(), rawGradient.size());
    ASSERT_GT(batchSize, 0u);

    state.t += 1.0f;
    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());
    const float invBiasCorrection1 = static_cast<float>(1.0 / (1.0 - std::pow(static_cast<double>(beta1), state.t)));
    const float invBiasCorrection2 = static_cast<float>(1.0 / (1.0 - std::pow(static_cast<double>(beta2), state.t)));

    std::vector<float> update(rawGradient.size());
    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float g = rawGradient[i] * invBatchLossScale;
        state.m[i] = beta1 * state.m[i] + (1.0f - beta1) * g;
        state.v[i] = beta2 * state.v[i] + (1.0f - beta2) * g * g;

        const float mHat = state.m[i] * invBiasCorrection1;
        const float vHat = state.v[i] * invBiasCorrection2;
        update[i] = mHat / (std::sqrt(vHat) + epsilon) + weightDecay * state.weights[i];
    }

    float trustRatio = 1.0f;
    if (dims.size() > 1) {
        double weightNormSquared = 0.0;
        double updateNormSquared = 0.0;
        for (uint64_t i = 0; i < state.weights.size(); ++i) {
            weightNormSquared += static_cast<double>(state.weights[i]) * static_cast<double>(state.weights[i]);
            updateNormSquared += static_cast<double>(update[i]) * static_cast<double>(update[i]);
        }
        trustRatio = static_cast<float>(std::sqrt(weightNormSquared) / (std::sqrt(updateNormSquared) + trustRatioEpsilon));
    }

    for (uint64_t i = 0; i < state.weights.size(); ++i)
        state.weights[i] -= alpha * trustRatio * update[i];
}

void runLambStep(Lamb& lamb, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    std::optional<Tensor> gradientOpt = lamb.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());
    Tensor gradient = gradientOpt.value();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);
    lamb.updateWeights(batchSize);
    stream.synchronize();
}

}  // namespace

TEST(LambTest, ConstructorGettersSettersAndHyperParameters) {
    constexpr uint64_t id = 29;
    Lamb lamb(id, 0.01f, 0.9f, 0.999f, 1e-6f, 0.01f, 1e-6f);

    EXPECT_EQ(lamb.getId(), id);
    EXPECT_FLOAT_EQ(lamb.getT(), 0.0f);
    EXPECT_FLOAT_EQ(lamb.getAlpha(), 0.01f);
    EXPECT_FLOAT_EQ(lamb.getBeta1(), 0.9f);
    EXPECT_FLOAT_EQ(lamb.getBeta2(), 0.999f);
    EXPECT_FLOAT_EQ(lamb.getEpsilon(), 1e-6f);
    EXPECT_FLOAT_EQ(lamb.getWeightDecay(), 0.01f);
    EXPECT_FLOAT_EQ(lamb.getTrustRatioEpsilon(), 1e-6f);

    lamb.setT(3.0f);
    lamb.setAlpha(0.02f);
    lamb.setBeta1(0.8f);
    lamb.setBeta2(0.99f);
    lamb.setEpsilon(1e-5f);
    lamb.setWeightDecay(0.02f);
    lamb.setTrustRatioEpsilon(1e-5f);

    EXPECT_FLOAT_EQ(lamb.getT(), 3.0f);
    EXPECT_FLOAT_EQ(lamb.getAlpha(), 0.02f);
    EXPECT_FLOAT_EQ(lamb.getBeta1(), 0.8f);
    EXPECT_FLOAT_EQ(lamb.getBeta2(), 0.99f);
    EXPECT_FLOAT_EQ(lamb.getEpsilon(), 1e-5f);
    EXPECT_FLOAT_EQ(lamb.getWeightDecay(), 0.02f);
    EXPECT_FLOAT_EQ(lamb.getTrustRatioEpsilon(), 1e-5f);

    std::unordered_map<std::string, float> params = lamb.getAllHyperParameters();
    ASSERT_EQ(params.size(), 7u);
    expectMapHasValue(params, "t", 3.0f);
    expectMapHasValue(params, "alpha", 0.02f);
    expectMapHasValue(params, "beta1", 0.8f);
    expectMapHasValue(params, "beta2", 0.99f);
    expectMapHasValue(params, "epsilon", 1e-5f);
    expectMapHasValue(params, "weightDecay", 0.02f);
    expectMapHasValue(params, "trustRatioEpsilon", 1e-5f);
}

TEST(LambTest, DenseUpdateMatchesReferenceWithLayerTrustRatio) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.01f;
    constexpr float beta1 = 0.8f;
    constexpr float beta2 = 0.9f;
    constexpr float epsilon = 1e-6f;
    constexpr float weightDecay = 0.02f;
    constexpr float trustRatioEpsilon = 1e-6f;
    const std::vector<uint64_t> dims{2, 3};

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, dims));
    const std::vector<float> initialWeights{1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Lamb lamb(31, alpha, beta1, beta2, epsilon, weightDecay, trustRatioEpsilon);
    lamb.compile(weights, stream);
    stream.synchronize();

    Tensor m = requireOptimizerStorage(lamb, "m");
    Tensor v = requireOptimizerStorage(lamb, "v");
    copyValuesToGpuFp32Tensor(m, std::vector<float>(initialWeights.size(), 0.0f), stream);
    copyValuesToGpuFp32Tensor(v, std::vector<float>(initialWeights.size(), 0.0f), stream);

    LambReferenceState reference{initialWeights, std::vector<float>(initialWeights.size(), 0.0f), std::vector<float>(initialWeights.size(), 0.0f)};
    const std::vector<float> gradient1{0.5f, -0.25f, 1.0f, -1.5f, 0.75f, -0.5f};
    const std::vector<float> gradient2{-0.1f, 0.2f, -0.3f, 0.4f, -0.5f, 0.6f};

    applyLambReferenceStep(reference, gradient1, dims, 2, alpha, beta1, beta2, epsilon, weightDecay, trustRatioEpsilon);
    runLambStep(lamb, gradient1, 2, stream);
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), reference.weights, 2e-5f, 2e-5f);
    expectAllClose(copyGpuFp32TensorToValues(m, stream), reference.m);
    expectAllClose(copyGpuFp32TensorToValues(v, stream), reference.v);
    EXPECT_FLOAT_EQ(lamb.getT(), 1.0f);

    applyLambReferenceStep(reference, gradient2, dims, 4, alpha, beta1, beta2, epsilon, weightDecay, trustRatioEpsilon);
    runLambStep(lamb, gradient2, 4, stream);
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), reference.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(m, stream), reference.m);
    expectAllClose(copyGpuFp32TensorToValues(v, stream), reference.v);
    EXPECT_FLOAT_EQ(lamb.getT(), 2.0f);
}

TEST(LambTest, OneDimensionalWeightsUseAdamWStyleTrustRatioOfOne) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.01f;
    constexpr float beta1 = 0.8f;
    constexpr float beta2 = 0.9f;
    constexpr float epsilon = 1e-6f;
    constexpr float weightDecay = 0.02f;
    constexpr float trustRatioEpsilon = 1e-6f;
    const std::vector<uint64_t> dims{4};

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, dims));
    const std::vector<float> initialWeights{0.0f, 0.5f, -1.0f, 1.5f};
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Lamb lamb(33, alpha, beta1, beta2, epsilon, weightDecay, trustRatioEpsilon);
    lamb.compile(weights, stream);
    stream.synchronize();

    Tensor m = requireOptimizerStorage(lamb, "m");
    Tensor v = requireOptimizerStorage(lamb, "v");
    copyValuesToGpuFp32Tensor(m, std::vector<float>(initialWeights.size(), 0.0f), stream);
    copyValuesToGpuFp32Tensor(v, std::vector<float>(initialWeights.size(), 0.0f), stream);

    LambReferenceState reference{initialWeights, std::vector<float>(initialWeights.size(), 0.0f), std::vector<float>(initialWeights.size(), 0.0f)};
    const std::vector<float> gradient{0.1f, -0.2f, 0.3f, -0.4f};

    applyLambReferenceStep(reference, gradient, dims, 2, alpha, beta1, beta2, epsilon, weightDecay, trustRatioEpsilon);
    runLambStep(lamb, gradient, 2, stream);

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), reference.weights, 2e-5f, 2e-5f);
}

TEST(LambTest, SparseRowsAreRejectedBecauseLayerWideNormsAreRequired) {
    Stream stream(gpuPlacement);
    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {4, 3}));
    Lamb lamb(35, 0.01f, 0.9f, 0.999f, 1e-6f, 0.01f, 1e-6f);

    EXPECT_FALSE(lamb.supportsSparseRowGradients());
    EXPECT_FALSE(lamb.supportsSparseRowUpdateFusion());
    EXPECT_THROW(lamb.compileSparseRows(weights, 2, stream), std::runtime_error);
}
