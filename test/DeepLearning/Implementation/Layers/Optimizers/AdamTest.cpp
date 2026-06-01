#include <optional>
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adam.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <set>
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

void writeCpuUint16Tensor(Tensor& tensor, const std::vector<uint64_t>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::UINT16);
    ASSERT_EQ(tensor.getTotalNumElements(), values.size());

    uint16_t* ptr = tensor.getMemPtr<uint16_t>();
    for (uint64_t i = 0; i < values.size(); ++i)
        ptr[i] = static_cast<uint16_t>(values[i]);
}

void writeCpuUint32Tensor(Tensor& tensor, const std::vector<uint64_t>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::UINT32);
    ASSERT_EQ(tensor.getTotalNumElements(), values.size());

    uint32_t* ptr = tensor.getMemPtr<uint32_t>();
    for (uint64_t i = 0; i < values.size(); ++i)
        ptr[i] = static_cast<uint32_t>(values[i]);
}

void writeCpuUint64Tensor(Tensor& tensor, const std::vector<uint64_t>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::UINT64);
    ASSERT_EQ(tensor.getTotalNumElements(), values.size());

    uint64_t* ptr = tensor.getMemPtr<uint64_t>();
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


void enqueueValuesToGpuFp32Tensor(Tensor& gpuTensor,
                                  const std::vector<float>& values,
                                  Stream& stream,
                                  std::deque<Tensor>& hostKeepAlive) {
    ASSERT_EQ(gpuTensor.getPlacement(), gpuPlacement);
    ASSERT_EQ(gpuTensor.getDataType(), DataType::FP32);
    ASSERT_EQ(gpuTensor.getTotalNumElements(), values.size());

    hostKeepAlive.emplace_back(cpuPlacement, gpuTensor.getDescriptor());
    writeCpuFp32Tensor(hostKeepAlive.back(), values);
    gpuTensor.copyFromAsync(hostKeepAlive.back(), stream);
}

void enqueueRowValuesToGpuTensor(Tensor& gpuTensor,
                                 const std::vector<uint64_t>& values,
                                 Stream& stream,
                                 std::deque<Tensor>& hostKeepAlive) {
    ASSERT_EQ(gpuTensor.getPlacement(), gpuPlacement);
    ASSERT_EQ(gpuTensor.getTotalNumElements(), values.size());

    hostKeepAlive.emplace_back(cpuPlacement, gpuTensor.getDescriptor());
    switch (gpuTensor.getDataType()) {
        case DataType::UINT16:
            writeCpuUint16Tensor(hostKeepAlive.back(), values);
            break;
        case DataType::UINT32:
            writeCpuUint32Tensor(hostKeepAlive.back(), values);
            break;
        case DataType::UINT64:
            writeCpuUint64Tensor(hostKeepAlive.back(), values);
            break;
        default:
            FAIL() << "Unsupported sparse row dtype in enqueueRowValuesToGpuTensor.";
    }

    gpuTensor.copyFromAsync(hostKeepAlive.back(), stream);
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

void copyRowValuesToGpuTensor(Tensor& gpuTensor, const std::vector<uint64_t>& values, Stream& stream) {
    ASSERT_EQ(gpuTensor.getPlacement(), gpuPlacement);
    ASSERT_EQ(gpuTensor.getTotalNumElements(), values.size());

    Tensor host(cpuPlacement, gpuTensor.getDescriptor());
    switch (gpuTensor.getDataType()) {
        case DataType::UINT16:
            writeCpuUint16Tensor(host, values);
            break;
        case DataType::UINT32:
            writeCpuUint32Tensor(host, values);
            break;
        case DataType::UINT64:
            writeCpuUint64Tensor(host, values);
            break;
        default:
            FAIL() << "Unsupported sparse row dtype in copyRowValuesToGpuTensor.";
    }

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


void applySparseAdamReferenceStep(AdamReferenceState& state,
                                  const std::vector<uint64_t>& rows,
                                  const std::vector<float>& sparseGradientValues,
                                  uint64_t numRows,
                                  uint64_t embeddingDim,
                                  uint32_t batchSize,
                                  float alpha,
                                  float beta1,
                                  float beta2,
                                  float epsilon) {
    ASSERT_LE(numRows, rows.size());
    ASSERT_GE(sparseGradientValues.size(), numRows * embeddingDim);
    ASSERT_EQ(state.weights.size(), state.m.size());
    ASSERT_EQ(state.weights.size(), state.v.size());
    ASSERT_GT(batchSize, 0u);

    state.t += 1.0f;

    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());
    const double alphaT64 = static_cast<double>(alpha) * std::sqrt(1.0 - std::pow(static_cast<double>(beta2), state.t)) /
                            (1.0 - std::pow(static_cast<double>(beta1), state.t));
    const float alphaT = static_cast<float>(alphaT64);

    for (uint64_t u = 0; u < numRows; ++u) {
        const uint64_t row = rows[u];
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            const uint64_t denseIndex = row * embeddingDim + d;
            ASSERT_LT(denseIndex, state.weights.size());
            const float g = sparseGradientValues[u * embeddingDim + d] * invBatchLossScale;

            state.m[denseIndex] = beta1 * state.m[denseIndex] + (1.0f - beta1) * g;
            state.v[denseIndex] = beta2 * state.v[denseIndex] + (1.0f - beta2) * g * g;
            state.weights[denseIndex] -= alphaT * state.m[denseIndex] / (std::sqrt(state.v[denseIndex]) + epsilon);
        }
    }
}

void writeSparseGradient(SparseRowGradient& gradient,
                         const std::vector<uint64_t>& rows,
                         const std::vector<float>& values,
                         uint64_t numRows,
                         Stream& stream) {
    ASSERT_LE(numRows, gradient.capacity);
    ASSERT_LE(rows.size(), gradient.rows.getTotalNumElements());
    ASSERT_EQ(values.size(), gradient.capacity * gradient.embeddingDim);

    std::vector<uint64_t> rowStorage(gradient.rows.getTotalNumElements(), 0);
    std::copy(rows.begin(), rows.end(), rowStorage.begin());
    copyRowValuesToGpuTensor(gradient.rows, rowStorage, stream);
    copyValuesToGpuFp32Tensor(gradient.values, values, stream);
    copyRowValuesToGpuTensor(gradient.numRows, {numRows}, stream);
}


void enqueueSparseGradient(SparseRowGradient& gradient,
                           const std::vector<uint64_t>& rows,
                           const std::vector<float>& values,
                           uint64_t numRows,
                           Stream& stream,
                           std::deque<Tensor>& hostKeepAlive) {
    ASSERT_LE(numRows, gradient.capacity);
    ASSERT_LE(rows.size(), gradient.rows.getTotalNumElements());
    ASSERT_EQ(values.size(), gradient.capacity * gradient.embeddingDim);

    std::vector<uint64_t> rowStorage(gradient.rows.getTotalNumElements(), 0);
    std::copy(rows.begin(), rows.end(), rowStorage.begin());
    enqueueRowValuesToGpuTensor(gradient.rows, rowStorage, stream, hostKeepAlive);
    enqueueValuesToGpuFp32Tensor(gradient.values, values, stream, hostKeepAlive);
    enqueueRowValuesToGpuTensor(gradient.numRows, {numRows}, stream, hostKeepAlive);
}

void runSparseAdamStep(Adam& adam,
                       SparseRowGradient& gradient,
                       const std::vector<uint64_t>& rows,
                       const std::vector<float>& values,
                       uint64_t numRows,
                       uint32_t batchSize,
                       Stream& stream) {
    writeSparseGradient(gradient, rows, values, numRows, stream);
    adam.updateSparseRows(batchSize);
    stream.synchronize();
}

void runAdamStep(Adam& adam, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    std::optional<Tensor> gradientOpt = adam.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Tensor gradient = gradientOpt.value();
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
    std::optional<Tensor> gradientOpt = adam.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Tensor gradient = gradientOpt.value();
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

    std::optional<Tensor> gradientOpt = adam.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Tensor gradient = gradientOpt.value();
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


TEST(AdamTest, QueuedDenseUpdatesCaptureRuntimeScalarsBeforeRuntimeStateMutation) {
    Stream stream(gpuPlacement);

    constexpr uint64_t numQueuedSteps = 12;
    constexpr float beta1 = 0.65f;
    constexpr float beta2 = 0.82f;
    constexpr float epsilon = 1e-4f;

    const std::vector<float> initialWeights{1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Adam adam(71, /*alpha=*/0.01f, beta1, beta2, epsilon);
    adam.compile(weights, stream);
    stream.synchronize();

    std::optional<Tensor> gradientOpt = adam.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());
    Tensor gradient = gradientOpt.value();

    AdamReferenceState expected;
    expected.weights = initialWeights;
    expected.m.assign(initialWeights.size(), 0.0f);
    expected.v.assign(initialWeights.size(), 0.0f);

    std::deque<Tensor> queuedHostCopies;
    for (uint64_t stepIdx = 0; stepIdx < numQueuedSteps; ++stepIdx) {
        std::vector<float> rawGradient(initialWeights.size());
        for (uint64_t i = 0; i < rawGradient.size(); ++i) {
            const float sign = ((stepIdx + i) % 2 == 0) ? 1.0f : -1.0f;
            rawGradient[i] = sign * 0.25f * static_cast<float>((stepIdx + 1) * (i + 2));
        }

        const uint32_t batchSize = static_cast<uint32_t>((stepIdx % 4) + 1);
        const float alpha = 0.008f + 0.0015f * static_cast<float>(stepIdx);

        adam.setT(expected.t);
        adam.setAlpha(alpha);
        enqueueValuesToGpuFp32Tensor(gradient, rawGradient, stream, queuedHostCopies);
        applyAdamReferenceStep(expected, rawGradient, batchSize, alpha, beta1, beta2, epsilon);
        adam.updateWeights(batchSize);

        // Poison the mutable CPU-side RuntimeState after the launch has been queued.
        // Correct execution depends only on the scalar values captured for that launch,
        // not on whatever RuntimeState contains by the time the GPU reaches the queued work.
        adam.setAlpha(100.0f + static_cast<float>(stepIdx));
        adam.setT(1000.0f + static_cast<float>(stepIdx));
    }

    stream.synchronize();
    adam.setT(expected.t);

    EXPECT_FLOAT_EQ(adam.getT(), static_cast<float>(numQueuedSteps));
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 1e-4f, 1e-4f);
    expectAllClose(copyGpuFp32TensorToValues(adam.getOptimizerParameterTensor("m"), stream), expected.m, 1e-4f, 1e-4f);
    expectAllClose(copyGpuFp32TensorToValues(adam.getOptimizerParameterTensor("v"), stream), expected.v, 1e-4f, 1e-4f);
}

TEST(AdamTest, CompileSparseRowsCreatesSparseGradientMomentsAndNoDenseGradient) {
    Stream stream(gpuPlacement);

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {5, 4}));

    Adam adam(61, /*alpha=*/0.01f, /*beta1=*/0.8f, /*beta2=*/0.9f, /*epsilon=*/1e-5f);
    EXPECT_TRUE(adam.supportsSparseRowGradients());
    EXPECT_TRUE(adam.supportsSparseRowUpdateFusion());

    SparseRowGradient gradient = adam.compileSparseRows(weights, /*maxSparseRows=*/3, stream);
    stream.synchronize();

    EXPECT_TRUE(adam.isCompiled());
    EXPECT_FALSE(adam.getWeightsGradient().has_value());
    ASSERT_TRUE(adam.getSparseRowGradient().has_value());

    gradient.validate();
    EXPECT_EQ(gradient.capacity, 3u);
    EXPECT_EQ(gradient.vocabularySize, 5u);
    EXPECT_EQ(gradient.embeddingDim, 4u);
    EXPECT_EQ(gradient.values.getDataType(), DataType::FP32);
    EXPECT_EQ(gradient.values.getDimensions(), (std::vector<uint64_t>{3, 4}));
    EXPECT_EQ(gradient.rows.getDataType(), DataType::UINT16);
    EXPECT_EQ(gradient.numRows.getDataType(), DataType::UINT16);

    Tensor weightsOut = adam.getOptimizerParameterTensor("weights");
    Tensor m = adam.getOptimizerParameterTensor("m");
    Tensor v = adam.getOptimizerParameterTensor("v");
    EXPECT_EQ(weightsOut, weights);
    EXPECT_EQ(m.getDataType(), DataType::FP32);
    EXPECT_EQ(v.getDataType(), DataType::FP32);
    EXPECT_EQ(m.getDimensions(), weights.getDimensions());
    EXPECT_EQ(v.getDimensions(), weights.getDimensions());

    std::vector<std::string> outputNames = adam.getOptimizerParameterNames();
    std::set<std::string> names(outputNames.begin(), outputNames.end());
    EXPECT_EQ(names, (std::set<std::string>{"weights", "m", "v"}));
}

TEST(AdamTest, SparseRowUpdateTwoStepsCarryMomentsOnlyForTouchedRows) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabularySize = 5;
    constexpr uint64_t embeddingDim = 4;
    constexpr uint64_t capacity = 4;
    constexpr float alpha = 0.02f;
    constexpr float beta1 = 0.7f;
    constexpr float beta2 = 0.8f;
    constexpr float epsilon = 1e-4f;

    const std::vector<float> initialWeights{
        1.0f, 1.1f, 1.2f, 1.3f,
        2.0f, 2.1f, 2.2f, 2.3f,
        3.0f, 3.1f, 3.2f, 3.3f,
        4.0f, 4.1f, 4.2f, 4.3f,
        5.0f, 5.1f, 5.2f, 5.3f,
    };
    const std::vector<uint64_t> rows1{3, 1, 4, 0};
    const std::vector<float> values1{
        0.4f, -0.8f, 1.2f, -1.6f,
        2.0f, -2.4f, 2.8f, -3.2f,
        1000.0f, 1000.0f, 1000.0f, 1000.0f,
        -1000.0f, -1000.0f, -1000.0f, -1000.0f,
    };
    const std::vector<uint64_t> rows2{1, 2, 3, 4};
    const std::vector<float> values2{
        -1.0f, 1.5f, -2.0f, 2.5f,
        3.0f, -3.5f, 4.0f, -4.5f,
        -0.25f, 0.5f, -0.75f, 1.0f,
        999.0f, 999.0f, 999.0f, 999.0f,
    };

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {vocabularySize, embeddingDim}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Adam adam(62, alpha, beta1, beta2, epsilon);
    SparseRowGradient gradient = adam.compileSparseRows(weights, capacity, stream);
    stream.synchronize();

    AdamReferenceState expected;
    expected.weights = initialWeights;
    expected.m.assign(initialWeights.size(), 0.0f);
    expected.v.assign(initialWeights.size(), 0.0f);

    applySparseAdamReferenceStep(expected,
                                 rows1,
                                 values1,
                                 /*numRows=*/2,
                                 embeddingDim,
                                 /*batchSize=*/2,
                                 alpha,
                                 beta1,
                                 beta2,
                                 epsilon);
    runSparseAdamStep(adam, gradient, rows1, values1, /*numRows=*/2, /*batchSize=*/2, stream);

    applySparseAdamReferenceStep(expected,
                                 rows2,
                                 values2,
                                 /*numRows=*/3,
                                 embeddingDim,
                                 /*batchSize=*/5,
                                 alpha,
                                 beta1,
                                 beta2,
                                 epsilon);
    runSparseAdamStep(adam, gradient, rows2, values2, /*numRows=*/3, /*batchSize=*/5, stream);

    EXPECT_FLOAT_EQ(adam.getT(), 2.0f);
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(adam.getOptimizerParameterTensor("m"), stream), expected.m, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(adam.getOptimizerParameterTensor("v"), stream), expected.v, 3e-5f, 3e-5f);
}


TEST(AdamTest, QueuedSparseRowUpdatesCaptureRuntimeScalarsBeforeRuntimeStateMutation) {
    Stream stream(gpuPlacement);

    constexpr uint64_t numQueuedSteps = 12;
    constexpr uint64_t vocabularySize = 8;
    constexpr uint64_t embeddingDim = 4;
    constexpr uint64_t capacity = 4;
    constexpr uint64_t activeRows = 3;
    constexpr float beta1 = 0.6f;
    constexpr float beta2 = 0.85f;
    constexpr float epsilon = 1e-4f;

    std::vector<float> initialWeights(vocabularySize * embeddingDim);
    for (uint64_t i = 0; i < initialWeights.size(); ++i) {
        initialWeights[i] = -2.0f + 0.125f * static_cast<float>(i);
    }

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {vocabularySize, embeddingDim}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Adam adam(72, /*alpha=*/0.01f, beta1, beta2, epsilon);
    SparseRowGradient gradient = adam.compileSparseRows(weights, capacity, stream);
    stream.synchronize();

    AdamReferenceState expected;
    expected.weights = initialWeights;
    expected.m.assign(initialWeights.size(), 0.0f);
    expected.v.assign(initialWeights.size(), 0.0f);

    std::deque<Tensor> queuedHostCopies;
    for (uint64_t stepIdx = 0; stepIdx < numQueuedSteps; ++stepIdx) {
        const uint64_t row0 = stepIdx % vocabularySize;
        const std::vector<uint64_t> rows{row0, (row0 + 3) % vocabularySize, (row0 + 5) % vocabularySize};

        std::vector<float> values(capacity * embeddingDim, 12345.0f);
        for (uint64_t r = 0; r < activeRows; ++r) {
            for (uint64_t d = 0; d < embeddingDim; ++d) {
                const float sign = ((stepIdx + r + d) % 2 == 0) ? 1.0f : -1.0f;
                values[r * embeddingDim + d] = sign * 0.125f * static_cast<float>((stepIdx + 1) * (r + 1) * (d + 1));
            }
        }

        const uint32_t batchSize = static_cast<uint32_t>((stepIdx % 5) + 1);
        const float alpha = 0.006f + 0.00125f * static_cast<float>(stepIdx);

        adam.setT(expected.t);
        adam.setAlpha(alpha);
        enqueueSparseGradient(gradient, rows, values, activeRows, stream, queuedHostCopies);
        applySparseAdamReferenceStep(expected, rows, values, activeRows, embeddingDim, batchSize, alpha, beta1, beta2, epsilon);
        adam.updateSparseRows(batchSize);

        // Mutate the CPU RuntimeState immediately after enqueue. The sparse update
        // must have copied alphaT/invBatchLossScale into this launch's parameters.
        adam.setAlpha(200.0f + static_cast<float>(stepIdx));
        adam.setT(2000.0f + static_cast<float>(stepIdx));
    }

    stream.synchronize();
    adam.setT(expected.t);

    EXPECT_FLOAT_EQ(adam.getT(), static_cast<float>(numQueuedSteps));
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 1e-4f, 1e-4f);
    expectAllClose(copyGpuFp32TensorToValues(adam.getOptimizerParameterTensor("m"), stream), expected.m, 1e-4f, 1e-4f);
    expectAllClose(copyGpuFp32TensorToValues(adam.getOptimizerParameterTensor("v"), stream), expected.v, 1e-4f, 1e-4f);
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

    std::optional<Tensor> gradientOpt = adam.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());
    EXPECT_EQ(gradientOpt.value().getDataType(), DataType::FP8_E5M2);

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
