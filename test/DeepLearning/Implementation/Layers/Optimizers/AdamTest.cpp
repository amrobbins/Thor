#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adam.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;
using DataType = TensorDescriptor::DataType;

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

struct AdamReferenceState {
    std::vector<float> weights;
    std::vector<float> m;
    std::vector<float> v;
    float t = 0.0f;
};

void applyAdamReferenceStep(AdamReferenceState& state,
                            const std::vector<float>& rawGradient,
                            uint32_t batchSize,
                            float alpha,
                            float beta1,
                            float beta2,
                            float epsilon) {
    ASSERT_EQ(state.weights.size(), rawGradient.size());
    ASSERT_EQ(state.m.size(), rawGradient.size());
    ASSERT_EQ(state.v.size(), rawGradient.size());
    ASSERT_GT(batchSize, 0u);

    state.t += 1.0f;

    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());
    const double alphaT64 = static_cast<double>(alpha) * std::sqrt(1.0 - std::pow(static_cast<double>(beta2), state.t)) /
                            (1.0 - std::pow(static_cast<double>(beta1), state.t));
    const float alphaT = static_cast<float>(alphaT64);

    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float g = rawGradient[i] * invBatchLossScale;

        state.m[i] = beta1 * state.m[i] + (1.0f - beta1) * g;
        state.v[i] = beta2 * state.v[i] + (1.0f - beta2) * g * g;

        state.weights[i] = state.weights[i] - alphaT * state.m[i] / (std::sqrt(state.v[i]) + epsilon);
    }
}

void runAdamStep(Adam& adam, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    Optional<Tensor> gradientOpt = adam.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.isPresent());

    Tensor gradient = gradientOpt.get();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);

    adam.updateWeights(batchSize);
    stream.synchronize();
}

float quantizeE5M2ToFloat(float value) { return static_cast<float>(__nv_fp8_e5m2(value)); }

std::vector<float> quantizeE5M2ToFloat(const std::vector<float>& values) {
    std::vector<float> quantized(values.size());
    for (uint64_t i = 0; i < values.size(); ++i)
        quantized[i] = quantizeE5M2ToFloat(values[i]);
    return quantized;
}

void writeCpuFp8E5M2Tensor(Tensor& tensor, const std::vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::FP8_E5M2);
    ASSERT_EQ(tensor.getTotalNumElements(), values.size());

    auto* ptr = tensor.getMemPtr<__nv_fp8_e5m2>();
    for (uint64_t i = 0; i < values.size(); ++i)
        ptr[i] = __nv_fp8_e5m2(values[i]);
}

std::vector<float> readCpuFp8E5M2TensorAsFloat(const Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);
    EXPECT_EQ(tensor.getDataType(), DataType::FP8_E5M2);

    std::vector<float> values(tensor.getTotalNumElements());
    const auto* ptr = tensor.getMemPtr<__nv_fp8_e5m2>();
    for (uint64_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<float>(ptr[i]);

    return values;
}

void copyValuesToGpuFp8E5M2Tensor(Tensor& gpuTensor, const std::vector<float>& values, Stream& stream) {
    ASSERT_EQ(gpuTensor.getPlacement(), gpuPlacement);
    ASSERT_EQ(gpuTensor.getDataType(), DataType::FP8_E5M2);
    ASSERT_EQ(gpuTensor.getTotalNumElements(), values.size());

    Tensor host(cpuPlacement, gpuTensor.getDescriptor());
    writeCpuFp8E5M2Tensor(host, values);

    gpuTensor.copyFromAsync(host, stream);
    stream.synchronize();
}

std::vector<float> copyGpuFp8E5M2TensorToFloatValues(const Tensor& gpuTensor, Stream& stream) {
    EXPECT_EQ(gpuTensor.getPlacement(), gpuPlacement);
    EXPECT_EQ(gpuTensor.getDataType(), DataType::FP8_E5M2);

    Tensor host = gpuTensor.clone(cpuPlacement);
    host.copyFromAsync(gpuTensor, stream);
    stream.synchronize();

    return readCpuFp8E5M2TensorAsFloat(host);
}

void applyAdamReferenceStepE5M2Weights(AdamReferenceState& state,
                                       const std::vector<float>& rawGradient,
                                       uint32_t batchSize,
                                       float alpha,
                                       float beta1,
                                       float beta2,
                                       float epsilon) {
    ASSERT_EQ(state.weights.size(), rawGradient.size());
    ASSERT_EQ(state.m.size(), rawGradient.size());
    ASSERT_EQ(state.v.size(), rawGradient.size());
    ASSERT_GT(batchSize, 0u);

    state.t += 1.0f;

    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());
    const double alphaT64 = static_cast<double>(alpha) * std::sqrt(1.0 - std::pow(static_cast<double>(beta2), state.t)) /
                            (1.0 - std::pow(static_cast<double>(beta1), state.t));
    const float alphaT = static_cast<float>(alphaT64);

    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        // The gradient tensor is cloned from FP8 weights, so the injected gradient
        // is quantized before Adam reads it as FP32.
        const float g = quantizeE5M2ToFloat(rawGradient[i]) * invBatchLossScale;

        state.m[i] = beta1 * state.m[i] + (1.0f - beta1) * g;
        state.v[i] = beta2 * state.v[i] + (1.0f - beta2) * g * g;

        const float wNext = state.weights[i] - alphaT * state.m[i] / (std::sqrt(state.v[i]) + epsilon);

        // Adam writes weights with output dtype equal to the original weights dtype.
        state.weights[i] = quantizeE5M2ToFloat(wNext);
    }
}

void runAdamStepFp8E5M2Gradient(Adam& adam, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    Optional<Tensor> gradientOpt = adam.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.isPresent());

    Tensor gradient = gradientOpt.get();
    ASSERT_EQ(gradient.getDataType(), DataType::FP8_E5M2);

    copyValuesToGpuFp8E5M2Tensor(gradient, rawGradient, stream);

    adam.updateWeights(batchSize);
    stream.synchronize();
}

}  // namespace

TEST(AdamTest, ConstructorGettersSettersAndHyperParameters) {
    constexpr uint64_t id = 17;
    Adam adam(id, 0.001f, 0.9f, 0.999f, 1e-8f);

    EXPECT_EQ(adam.getId(), id);
    EXPECT_FLOAT_EQ(adam.getAlpha(), 0.001f);
    EXPECT_FLOAT_EQ(adam.getBeta1(), 0.9f);
    EXPECT_FLOAT_EQ(adam.getBeta2(), 0.999f);
    EXPECT_FLOAT_EQ(adam.getEpsilon(), 1e-8f);
    EXPECT_FLOAT_EQ(adam.getT(), 0.0f);

    adam.setAlpha(0.01f);
    adam.setBeta1(0.8f);
    adam.setBeta2(0.95f);
    adam.setEpsilon(1e-5f);
    adam.setT(7.0f);

    EXPECT_FLOAT_EQ(adam.getAlpha(), 0.01f);
    EXPECT_FLOAT_EQ(adam.getBeta1(), 0.8f);
    EXPECT_FLOAT_EQ(adam.getBeta2(), 0.95f);
    EXPECT_FLOAT_EQ(adam.getEpsilon(), 1e-5f);
    EXPECT_FLOAT_EQ(adam.getT(), 7.0f);

    std::unordered_map<std::string, float> updated = adam.updateHyperParameters(/*epoch=*/3, /*batch=*/4, /*batchesPerEpoch=*/5);
    ASSERT_EQ(updated.size(), 1u);
    expectMapHasValue(updated, "t", 7.0f);

    std::unordered_map<std::string, float> all = adam.getAllHyperParameters();
    ASSERT_EQ(all.size(), 5u);
    expectMapHasValue(all, "t", 7.0f);
    expectMapHasValue(all, "alpha", 0.01f);
    expectMapHasValue(all, "beta1", 0.8f);
    expectMapHasValue(all, "beta2", 0.95f);
    expectMapHasValue(all, "epsilon", 1e-5f);
}

TEST(AdamTest, CompileCreatesGradientMomentsAndNamedOutputs) {
    Stream stream(gpuPlacement);

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP16, {2, 3}));

    Adam adam(1, 0.001f, 0.9f, 0.999f, 1e-8f);
    EXPECT_FALSE(adam.isCompiled());

    adam.compile(weights, stream);
    stream.synchronize();

    EXPECT_TRUE(adam.isCompiled());

    Optional<Tensor> gradientOpt = adam.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.isPresent());

    Tensor gradient = gradientOpt.get();
    EXPECT_EQ(gradient.getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.getDataType(), DataType::FP16);
    EXPECT_EQ(gradient.getDimensions(), weights.getDimensions());

    EXPECT_TRUE(adam.hasParameter("m"));
    EXPECT_TRUE(adam.hasParameter("v"));

    Tensor weightsOut = adam.getOptimizerParameterTensor("weights");
    Tensor m = adam.getOptimizerParameterTensor("m");
    Tensor v = adam.getOptimizerParameterTensor("v");

    EXPECT_EQ(weightsOut, weights);
    EXPECT_EQ(m.getPlacement(), gpuPlacement);
    EXPECT_EQ(v.getPlacement(), gpuPlacement);
    EXPECT_EQ(m.getDataType(), DataType::FP32);
    EXPECT_EQ(v.getDataType(), DataType::FP32);
    EXPECT_EQ(m.getDimensions(), weights.getDimensions());
    EXPECT_EQ(v.getDimensions(), weights.getDimensions());

    std::vector<std::string> outputNames = adam.getOptimizerParameterNames();
    std::set<std::string> names(outputNames.begin(), outputNames.end());
    EXPECT_EQ(names, (std::set<std::string>{"weights", "m", "v"}));

    EXPECT_THROW((void)adam.getOptimizerParameterTensor("missing"), std::runtime_error);
}

TEST(AdamTest, CompileInitializesMomentsToZero) {
    Stream stream(gpuPlacement);

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    Adam adam(2, 0.001f, 0.9f, 0.999f, 1e-8f);
    adam.compile(weights, stream);
    stream.synchronize();

    Tensor m = adam.getOptimizerParameterTensor("m");
    Tensor v = adam.getOptimizerParameterTensor("v");

    expectAllClose(copyGpuFp32TensorToValues(m, stream), {0.0f, 0.0f, 0.0f, 0.0f});
    expectAllClose(copyGpuFp32TensorToValues(v, stream), {0.0f, 0.0f, 0.0f, 0.0f});
}

TEST(AdamTest, SingleStepFromPrecomputedGradientMatchesCpuReference) {
    Stream stream(gpuPlacement);

    const std::vector<float> initialWeights{1.0f, -2.0f, 3.0f, -4.0f};
    const std::vector<float> gradient{8.0f, -12.0f, 0.5f, -2.0f};

    constexpr uint32_t batchSize = 2;
    constexpr float alpha = 0.1f;
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float epsilon = 1e-4f;

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Adam adam(3, alpha, beta1, beta2, epsilon);
    adam.compile(weights, stream);
    stream.synchronize();

    AdamReferenceState expected;
    expected.weights = initialWeights;
    expected.m.assign(initialWeights.size(), 0.0f);
    expected.v.assign(initialWeights.size(), 0.0f);

    applyAdamReferenceStep(expected, gradient, batchSize, alpha, beta1, beta2, epsilon);
    runAdamStep(adam, gradient, batchSize, stream);

    Tensor m = adam.getOptimizerParameterTensor("m");
    Tensor v = adam.getOptimizerParameterTensor("v");

    EXPECT_FLOAT_EQ(adam.getT(), 1.0f);
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 2e-5f, 2e-5f);
    expectAllClose(copyGpuFp32TensorToValues(m, stream), expected.m, 2e-5f, 2e-5f);
    expectAllClose(copyGpuFp32TensorToValues(v, stream), expected.v, 2e-5f, 2e-5f);
}

TEST(AdamTest, TwoStepsCarryMomentsAndUseBiasCorrection) {
    Stream stream(gpuPlacement);

    const std::vector<float> initialWeights{1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f};
    const std::vector<float> gradient1{10.0f, -20.0f, 0.5f, -4.0f, 8.0f, -16.0f};
    const std::vector<float> gradient2{-3.0f, 5.0f, 7.0f, -11.0f, 13.0f, -17.0f};

    constexpr float alpha = 0.05f;
    constexpr float beta1 = 0.8f;
    constexpr float beta2 = 0.95f;
    constexpr float epsilon = 1e-3f;

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Adam adam(4, alpha, beta1, beta2, epsilon);
    adam.compile(weights, stream);
    stream.synchronize();

    AdamReferenceState expected;
    expected.weights = initialWeights;
    expected.m.assign(initialWeights.size(), 0.0f);
    expected.v.assign(initialWeights.size(), 0.0f);

    applyAdamReferenceStep(expected, gradient1, /*batchSize=*/2, alpha, beta1, beta2, epsilon);
    runAdamStep(adam, gradient1, /*batchSize=*/2, stream);

    applyAdamReferenceStep(expected, gradient2, /*batchSize=*/4, alpha, beta1, beta2, epsilon);
    runAdamStep(adam, gradient2, /*batchSize=*/4, stream);

    Tensor m = adam.getOptimizerParameterTensor("m");
    Tensor v = adam.getOptimizerParameterTensor("v");

    EXPECT_FLOAT_EQ(adam.getT(), 2.0f);
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(m, stream), expected.m, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(v, stream), expected.v, 3e-5f, 3e-5f);

    std::unordered_map<std::string, float> updated = adam.updateHyperParameters(/*epoch=*/0, /*batch=*/0, /*batchesPerEpoch=*/1);
    ASSERT_EQ(updated.size(), 1u);
    expectMapHasValue(updated, "t", 2.0f);

    std::unordered_map<std::string, float> all = adam.getAllHyperParameters();
    expectMapHasValue(all, "t", 2.0f);
}

TEST(AdamTest, ThreeStepsWithFp8E5M2WeightsCarryMomentsAndQuantizeWeights) {
    Stream stream(gpuPlacement);

    const std::vector<float> initialWeights{
        1.0f,
        -2.0f,
        4.0f,
        -8.0f,
        0.5f,
        -0.25f,
    };

    const std::vector<float> gradient1{
        8.0f,
        -4.0f,
        2.0f,
        -1.0f,
        0.5f,
        -0.25f,
    };

    const std::vector<float> gradient2{
        -2.0f,
        4.0f,
        -8.0f,
        1.0f,
        -0.5f,
        0.25f,
    };

    const std::vector<float> gradient3{
        1.0f,
        2.0f,
        -4.0f,
        -8.0f,
        0.25f,
        -0.5f,
    };

    constexpr float alpha = 0.25f;
    constexpr float beta1 = 0.8f;
    constexpr float beta2 = 0.95f;
    constexpr float epsilon = 1e-3f;

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP8_E5M2, {2, 3}));
    copyValuesToGpuFp8E5M2Tensor(weights, initialWeights, stream);

    Adam adam(5, alpha, beta1, beta2, epsilon);
    adam.compile(weights, stream);
    stream.synchronize();

    Optional<Tensor> gradientOpt = adam.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.isPresent());
    EXPECT_EQ(gradientOpt.get().getDataType(), DataType::FP8_E5M2);

    Tensor m = adam.getOptimizerParameterTensor("m");
    Tensor v = adam.getOptimizerParameterTensor("v");

    EXPECT_EQ(m.getDataType(), DataType::FP32);
    EXPECT_EQ(v.getDataType(), DataType::FP32);
    EXPECT_EQ(m.getDimensions(), weights.getDimensions());
    EXPECT_EQ(v.getDimensions(), weights.getDimensions());

    AdamReferenceState expected;
    expected.weights = quantizeE5M2ToFloat(initialWeights);
    expected.m.assign(initialWeights.size(), 0.0f);
    expected.v.assign(initialWeights.size(), 0.0f);

    applyAdamReferenceStepE5M2Weights(expected, gradient1, /*batchSize=*/2, alpha, beta1, beta2, epsilon);
    runAdamStepFp8E5M2Gradient(adam, gradient1, /*batchSize=*/2, stream);

    applyAdamReferenceStepE5M2Weights(expected, gradient2, /*batchSize=*/4, alpha, beta1, beta2, epsilon);
    runAdamStepFp8E5M2Gradient(adam, gradient2, /*batchSize=*/4, stream);

    applyAdamReferenceStepE5M2Weights(expected, gradient3, /*batchSize=*/1, alpha, beta1, beta2, epsilon);
    runAdamStepFp8E5M2Gradient(adam, gradient3, /*batchSize=*/1, stream);

    EXPECT_FLOAT_EQ(adam.getT(), 3.0f);

    // Weights are FP8, so exact agreement after CPU-side E5M2 quantization is expected.
    expectAllClose(copyGpuFp8E5M2TensorToFloatValues(weights, stream), expected.weights, 0.0f, 0.0f);

    // Moments remain FP32, but the fused GPU path can differ by small FP32 rounding.
    expectAllClose(copyGpuFp32TensorToValues(m, stream), expected.m, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(v, stream), expected.v, 3e-5f, 3e-5f);

    std::unordered_map<std::string, float> all = adam.getAllHyperParameters();
    expectMapHasValue(all, "t", 3.0f);
}
