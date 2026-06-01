#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adamax.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
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
            FAIL() << "Unsupported sparse row dtype.";
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

struct AdamaxReferenceState {
    std::vector<float> weights;
    std::vector<float> m;
    std::vector<float> u;
    float t = 0.0f;
};

void applyAdamaxReferenceStep(AdamaxReferenceState& state,
                              const std::vector<float>& rawGradient,
                              uint32_t batchSize,
                              float alpha,
                              float beta1,
                              float beta2,
                              float epsilon) {
    ASSERT_EQ(state.weights.size(), rawGradient.size());
    ASSERT_EQ(state.m.size(), rawGradient.size());
    ASSERT_EQ(state.u.size(), rawGradient.size());
    ASSERT_GT(batchSize, 0u);

    state.t += 1.0f;
    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());
    const double alphaT64 = static_cast<double>(alpha) / (1.0 - std::pow(static_cast<double>(beta1), state.t));
    const float alphaT = static_cast<float>(alphaT64);

    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float g = rawGradient[i] * invBatchLossScale;
        state.m[i] = beta1 * state.m[i] + (1.0f - beta1) * g;
        state.u[i] = std::max(beta2 * state.u[i], std::fabs(g));
        state.weights[i] -= alphaT * state.m[i] / (state.u[i] + epsilon);
    }
}

void applySparseAdamaxReferenceStep(AdamaxReferenceState& state,
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
    ASSERT_EQ(state.weights.size(), state.u.size());
    ASSERT_GT(batchSize, 0u);

    state.t += 1.0f;
    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());
    const double alphaT64 = static_cast<double>(alpha) / (1.0 - std::pow(static_cast<double>(beta1), state.t));
    const float alphaT = static_cast<float>(alphaT64);

    for (uint64_t r = 0; r < numRows; ++r) {
        const uint64_t row = rows[r];
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            const uint64_t denseIndex = row * embeddingDim + d;
            ASSERT_LT(denseIndex, state.weights.size());
            const float g = sparseGradientValues[r * embeddingDim + d] * invBatchLossScale;
            state.m[denseIndex] = beta1 * state.m[denseIndex] + (1.0f - beta1) * g;
            state.u[denseIndex] = std::max(beta2 * state.u[denseIndex], std::fabs(g));
            state.weights[denseIndex] -= alphaT * state.m[denseIndex] / (state.u[denseIndex] + epsilon);
        }
    }
}

void runAdamaxStep(Adamax& adamax, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    std::optional<Tensor> gradientOpt = adamax.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Tensor gradient = gradientOpt.value();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);

    adamax.updateWeights(batchSize);
    stream.synchronize();
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

void runSparseAdamaxStep(Adamax& adamax,
                         SparseRowGradient& gradient,
                         const std::vector<uint64_t>& rows,
                         const std::vector<float>& values,
                         uint64_t numRows,
                         uint32_t batchSize,
                         Stream& stream) {
    writeSparseGradient(gradient, rows, values, numRows, stream);
    adamax.updateSparseRows(batchSize);
    stream.synchronize();
}

}  // namespace

TEST(AdamaxTest, ConstructorGettersSettersAndHyperParameters) {
    constexpr uint64_t id = 23;
    Adamax adamax(id, 0.002f, 0.9f, 0.999f, 1e-7f);

    EXPECT_EQ(adamax.getId(), id);
    EXPECT_FLOAT_EQ(adamax.getAlpha(), 0.002f);
    EXPECT_FLOAT_EQ(adamax.getBeta1(), 0.9f);
    EXPECT_FLOAT_EQ(adamax.getBeta2(), 0.999f);
    EXPECT_FLOAT_EQ(adamax.getEpsilon(), 1e-7f);
    EXPECT_FLOAT_EQ(adamax.getT(), 0.0f);

    adamax.setAlpha(0.01f);
    adamax.setBeta1(0.8f);
    adamax.setBeta2(0.95f);
    adamax.setEpsilon(1e-5f);
    adamax.setT(7.0f);

    EXPECT_FLOAT_EQ(adamax.getAlpha(), 0.01f);
    EXPECT_FLOAT_EQ(adamax.getBeta1(), 0.8f);
    EXPECT_FLOAT_EQ(adamax.getBeta2(), 0.95f);
    EXPECT_FLOAT_EQ(adamax.getEpsilon(), 1e-5f);
    EXPECT_FLOAT_EQ(adamax.getT(), 7.0f);

    std::unordered_map<std::string, float> updated = adamax.updateHyperParameters(/*epoch=*/3, /*batch=*/4, /*batchesPerEpoch=*/5);
    ASSERT_EQ(updated.size(), 1u);
    expectMapHasValue(updated, "t", 7.0f);

    std::unordered_map<std::string, float> all = adamax.getAllHyperParameters();
    ASSERT_EQ(all.size(), 5u);
    expectMapHasValue(all, "t", 7.0f);
    expectMapHasValue(all, "alpha", 0.01f);
    expectMapHasValue(all, "beta1", 0.8f);
    expectMapHasValue(all, "beta2", 0.95f);
    expectMapHasValue(all, "epsilon", 1e-5f);
}

TEST(AdamaxTest, CompileCreatesGradientMomentInfinityNormAndNamedOutputs) {
    Stream stream(gpuPlacement);

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP16, {2, 3}));

    Adamax adamax(1, 0.002f, 0.9f, 0.999f, 1e-7f);
    EXPECT_FALSE(adamax.isCompiled());

    adamax.compile(weights, stream);
    stream.synchronize();

    EXPECT_TRUE(adamax.isCompiled());

    std::optional<Tensor> gradientOpt = adamax.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Tensor gradient = gradientOpt.value();
    EXPECT_EQ(gradient.getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.getDataType(), DataType::FP16);
    EXPECT_EQ(gradient.getDimensions(), weights.getDimensions());

    EXPECT_TRUE(adamax.hasParameter("m"));
    EXPECT_TRUE(adamax.hasParameter("u"));

    Tensor weightsOut = adamax.getOptimizerParameterTensor("weights");
    Tensor m = adamax.getOptimizerParameterTensor("m");
    Tensor u = adamax.getOptimizerParameterTensor("u");

    EXPECT_EQ(weightsOut, weights);
    EXPECT_EQ(m.getPlacement(), gpuPlacement);
    EXPECT_EQ(m.getDataType(), DataType::FP32);
    EXPECT_EQ(m.getDimensions(), weights.getDimensions());
    EXPECT_EQ(u.getPlacement(), gpuPlacement);
    EXPECT_EQ(u.getDataType(), DataType::FP32);
    EXPECT_EQ(u.getDimensions(), weights.getDimensions());

    std::vector<std::string> outputNames = adamax.getOptimizerParameterNames();
    std::set<std::string> names(outputNames.begin(), outputNames.end());
    EXPECT_EQ(names, (std::set<std::string>{"weights", "m", "u"}));
}

TEST(AdamaxTest, CompileInitializesStateToZero) {
    Stream stream(gpuPlacement);

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    Adamax adamax(2, 0.002f, 0.9f, 0.999f, 1e-7f);
    adamax.compile(weights, stream);
    stream.synchronize();

    Tensor m = adamax.getOptimizerParameterTensor("m");
    Tensor u = adamax.getOptimizerParameterTensor("u");
    expectAllClose(copyGpuFp32TensorToValues(m, stream), {0.0f, 0.0f, 0.0f, 0.0f});
    expectAllClose(copyGpuFp32TensorToValues(u, stream), {0.0f, 0.0f, 0.0f, 0.0f});
}

TEST(AdamaxTest, TwoDenseStepsMatchCpuReference) {
    Stream stream(gpuPlacement);

    const std::vector<float> initialWeights{1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f};
    const std::vector<float> gradient1{10.0f, -20.0f, 0.5f, -4.0f, 8.0f, -16.0f};
    const std::vector<float> gradient2{-3.0f, 5.0f, 7.0f, -11.0f, 13.0f, -17.0f};

    constexpr float alpha = 0.03f;
    constexpr float beta1 = 0.8f;
    constexpr float beta2 = 0.95f;
    constexpr float epsilon = 1e-4f;

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Adamax adamax(3, alpha, beta1, beta2, epsilon);
    adamax.compile(weights, stream);
    stream.synchronize();

    AdamaxReferenceState expected;
    expected.weights = initialWeights;
    expected.m.assign(initialWeights.size(), 0.0f);
    expected.u.assign(initialWeights.size(), 0.0f);

    applyAdamaxReferenceStep(expected, gradient1, /*batchSize=*/2, alpha, beta1, beta2, epsilon);
    runAdamaxStep(adamax, gradient1, /*batchSize=*/2, stream);

    applyAdamaxReferenceStep(expected, gradient2, /*batchSize=*/4, alpha, beta1, beta2, epsilon);
    runAdamaxStep(adamax, gradient2, /*batchSize=*/4, stream);

    Tensor m = adamax.getOptimizerParameterTensor("m");
    Tensor u = adamax.getOptimizerParameterTensor("u");

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(m, stream), expected.m, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(u, stream), expected.u, 3e-5f, 3e-5f);
    EXPECT_FLOAT_EQ(adamax.getT(), 2.0f);
}

TEST(AdamaxTest, SparseRowUpdatesOnlyTouchedRowsAndMatchCpuReference) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabSize = 5;
    constexpr uint64_t embeddingDim = 3;
    constexpr uint64_t maxSparseRows = 4;
    constexpr float alpha = 0.04f;
    constexpr float beta1 = 0.85f;
    constexpr float beta2 = 0.9f;
    constexpr float epsilon = 1e-4f;

    const std::vector<float> initialWeights{1.0f,  2.0f,  3.0f,
                                            -1.0f, -2.0f, -3.0f,
                                            4.0f,  5.0f,  6.0f,
                                            -4.0f, -5.0f, -6.0f,
                                            7.0f,  8.0f,  9.0f};

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {vocabSize, embeddingDim}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Adamax adamax(4, alpha, beta1, beta2, epsilon);
    SparseRowGradient gradient = adamax.compileSparseRows(weights, maxSparseRows, stream);
    stream.synchronize();

    AdamaxReferenceState expected;
    expected.weights = initialWeights;
    expected.m.assign(initialWeights.size(), 0.0f);
    expected.u.assign(initialWeights.size(), 0.0f);

    std::vector<uint64_t> rows1{1, 3};
    std::vector<float> values1{1.0f, -2.0f, 3.0f,
                               4.0f, -5.0f, 6.0f,
                               0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f};
    applySparseAdamaxReferenceStep(expected, rows1, values1, /*numRows=*/2, embeddingDim, /*batchSize=*/2, alpha, beta1, beta2, epsilon);
    runSparseAdamaxStep(adamax, gradient, rows1, values1, /*numRows=*/2, /*batchSize=*/2, stream);

    std::vector<uint64_t> rows2{0, 3, 4};
    std::vector<float> values2{-1.0f, 2.0f, -3.0f,
                               0.5f, -0.25f, 0.75f,
                               8.0f, -9.0f, 10.0f,
                               0.0f, 0.0f, 0.0f};
    applySparseAdamaxReferenceStep(expected, rows2, values2, /*numRows=*/3, embeddingDim, /*batchSize=*/4, alpha, beta1, beta2, epsilon);
    runSparseAdamaxStep(adamax, gradient, rows2, values2, /*numRows=*/3, /*batchSize=*/4, stream);

    Tensor m = adamax.getOptimizerParameterTensor("m");
    Tensor u = adamax.getOptimizerParameterTensor("u");

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 4e-5f, 4e-5f);
    expectAllClose(copyGpuFp32TensorToValues(m, stream), expected.m, 4e-5f, 4e-5f);
    expectAllClose(copyGpuFp32TensorToValues(u, stream), expected.u, 4e-5f, 4e-5f);
    EXPECT_FLOAT_EQ(adamax.getT(), 2.0f);
}

TEST(AdamaxTest, DenseAndSparseRuntimeScalarsIncludeBiasCorrectionAndLossScaling) {
    Adamax adamax(5, 0.03f, 0.8f, 0.95f, 1e-7f);

    std::unordered_map<std::string, float> dense = adamax.denseUpdateRuntimeScalars(/*batchSize=*/5, "__test__");
    ASSERT_EQ(dense.size(), 2u);
    expectMapHasValue(dense, "__test__alphaT", 0.03f / (1.0f - 0.8f));
    expectMapHasValue(dense, "__test__invBatchLossScale", 1.0f / (5.0f * Loss::getLossScalingFactor()));
    EXPECT_FLOAT_EQ(adamax.getT(), 1.0f);

    std::unordered_map<std::string, float> sparse = adamax.sparseRowUpdateRuntimeScalars(/*batchSize=*/7);
    ASSERT_EQ(sparse.size(), 2u);
    expectMapHasValue(sparse, "alphaT", 0.03f / (1.0f - 0.8f * 0.8f));
    expectMapHasValue(sparse, "invBatchLossScale", 1.0f / (7.0f * Loss::getLossScalingFactor()));
    EXPECT_FLOAT_EQ(adamax.getT(), 2.0f);
}
