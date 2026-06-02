#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/RAdam.h"
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

struct RAdamReferenceState {
    std::vector<float> weights;
    std::vector<float> m;
    std::vector<float> v;
    float t = 0.0f;
};

struct RAdamStepScalars {
    float rectifiedAlphaT = 0.0f;
    float unrectifiedAlphaT = 0.0f;
    float useRectified = 0.0f;
};

RAdamStepScalars computeRAdamStepScalars(float t, float alpha, float beta1, float beta2) {
    const double beta1PowT = std::pow(static_cast<double>(beta1), static_cast<double>(t));
    const double beta2PowT = std::pow(static_cast<double>(beta2), static_cast<double>(t));
    const double oneMinusBeta1PowT = 1.0 - beta1PowT;
    const double oneMinusBeta2PowT = 1.0 - beta2PowT;
    const double rhoInf = 2.0 / (1.0 - static_cast<double>(beta2)) - 1.0;
    const double rhoT = rhoInf - (2.0 * static_cast<double>(t) * beta2PowT / oneMinusBeta2PowT);

    RAdamStepScalars scalars;
    scalars.unrectifiedAlphaT = static_cast<float>(static_cast<double>(alpha) / oneMinusBeta1PowT);
    if (rhoT >= 5.0) {
        const double rectification = std::sqrt(((rhoT - 4.0) * (rhoT - 2.0) * rhoInf) /
                                               ((rhoInf - 4.0) * (rhoInf - 2.0) * rhoT));
        scalars.rectifiedAlphaT = static_cast<float>(static_cast<double>(alpha) * rectification *
                                                     std::sqrt(oneMinusBeta2PowT) / oneMinusBeta1PowT);
        scalars.useRectified = 1.0f;
    }
    return scalars;
}

void applyRAdamReferenceStep(RAdamReferenceState& state,
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
    const RAdamStepScalars scalars = computeRAdamStepScalars(state.t, alpha, beta1, beta2);

    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float g = rawGradient[i] * invBatchLossScale;
        state.m[i] = beta1 * state.m[i] + (1.0f - beta1) * g;
        state.v[i] = beta2 * state.v[i] + (1.0f - beta2) * g * g;
        const float step = scalars.useRectified > 0.5f
                               ? scalars.rectifiedAlphaT * state.m[i] / (std::sqrt(state.v[i]) + epsilon)
                               : scalars.unrectifiedAlphaT * state.m[i];
        state.weights[i] -= step;
    }
}

void applySparseRAdamReferenceStep(RAdamReferenceState& state,
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
    const RAdamStepScalars scalars = computeRAdamStepScalars(state.t, alpha, beta1, beta2);

    for (uint64_t r = 0; r < numRows; ++r) {
        const uint64_t row = rows[r];
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            const uint64_t denseIndex = row * embeddingDim + d;
            ASSERT_LT(denseIndex, state.weights.size());
            const float g = sparseGradientValues[r * embeddingDim + d] * invBatchLossScale;
            state.m[denseIndex] = beta1 * state.m[denseIndex] + (1.0f - beta1) * g;
            state.v[denseIndex] = beta2 * state.v[denseIndex] + (1.0f - beta2) * g * g;
            const float step = scalars.useRectified > 0.5f
                                   ? scalars.rectifiedAlphaT * state.m[denseIndex] / (std::sqrt(state.v[denseIndex]) + epsilon)
                                   : scalars.unrectifiedAlphaT * state.m[denseIndex];
            state.weights[denseIndex] -= step;
        }
    }
}

void runRAdamStep(RAdam& radam, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    std::optional<Tensor> gradientOpt = radam.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Tensor gradient = gradientOpt.value();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);

    radam.updateWeights(batchSize);
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

void runSparseRAdamStep(RAdam& radam,
                         SparseRowGradient& gradient,
                         const std::vector<uint64_t>& rows,
                         const std::vector<float>& values,
                         uint64_t numRows,
                         uint32_t batchSize,
                         Stream& stream) {
    writeSparseGradient(gradient, rows, values, numRows, stream);
    radam.updateSparseRows(batchSize);
    stream.synchronize();
}

}  // namespace

TEST(RAdamTest, ConstructorGettersSettersAndHyperParameters) {
    constexpr uint64_t id = 23;
    RAdam radam(id, 0.002f, 0.9f, 0.999f, 1e-7f);

    EXPECT_EQ(radam.getId(), id);
    EXPECT_FLOAT_EQ(radam.getAlpha(), 0.002f);
    EXPECT_FLOAT_EQ(radam.getBeta1(), 0.9f);
    EXPECT_FLOAT_EQ(radam.getBeta2(), 0.999f);
    EXPECT_FLOAT_EQ(radam.getEpsilon(), 1e-7f);
    EXPECT_FLOAT_EQ(radam.getT(), 0.0f);

    radam.setAlpha(0.01f);
    radam.setBeta1(0.8f);
    radam.setBeta2(0.95f);
    radam.setEpsilon(1e-5f);
    radam.setT(7.0f);

    EXPECT_FLOAT_EQ(radam.getAlpha(), 0.01f);
    EXPECT_FLOAT_EQ(radam.getBeta1(), 0.8f);
    EXPECT_FLOAT_EQ(radam.getBeta2(), 0.95f);
    EXPECT_FLOAT_EQ(radam.getEpsilon(), 1e-5f);
    EXPECT_FLOAT_EQ(radam.getT(), 7.0f);

    std::unordered_map<std::string, float> updated = radam.updateHyperParameters(/*epoch=*/3, /*batch=*/4, /*batchesPerEpoch=*/5);
    ASSERT_EQ(updated.size(), 1u);
    expectMapHasValue(updated, "t", 7.0f);

    std::unordered_map<std::string, float> all = radam.getAllHyperParameters();
    ASSERT_EQ(all.size(), 5u);
    expectMapHasValue(all, "t", 7.0f);
    expectMapHasValue(all, "alpha", 0.01f);
    expectMapHasValue(all, "beta1", 0.8f);
    expectMapHasValue(all, "beta2", 0.95f);
    expectMapHasValue(all, "epsilon", 1e-5f);
}

TEST(RAdamTest, CompileCreatesGradientMomentSecondMomentAndNamedOutputs) {
    Stream stream(gpuPlacement);

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP16, {2, 3}));

    RAdam radam(1, 0.002f, 0.9f, 0.999f, 1e-7f);
    EXPECT_FALSE(radam.isCompiled());

    radam.compile(weights, stream);
    stream.synchronize();

    EXPECT_TRUE(radam.isCompiled());

    std::optional<Tensor> gradientOpt = radam.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Tensor gradient = gradientOpt.value();
    EXPECT_EQ(gradient.getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.getDataType(), DataType::FP16);
    EXPECT_EQ(gradient.getDimensions(), weights.getDimensions());

    EXPECT_TRUE(radam.hasParameter("m"));
    EXPECT_TRUE(radam.hasParameter("v"));

    Tensor weightsOut = radam.getOptimizerParameterTensor("weights");
    Tensor m = radam.getOptimizerParameterTensor("m");
    Tensor v = radam.getOptimizerParameterTensor("v");

    EXPECT_EQ(weightsOut, weights);
    EXPECT_EQ(m.getPlacement(), gpuPlacement);
    EXPECT_EQ(m.getDataType(), DataType::FP32);
    EXPECT_EQ(m.getDimensions(), weights.getDimensions());
    EXPECT_EQ(v.getPlacement(), gpuPlacement);
    EXPECT_EQ(v.getDataType(), DataType::FP32);
    EXPECT_EQ(v.getDimensions(), weights.getDimensions());

    std::vector<std::string> outputNames = radam.getOptimizerParameterNames();
    std::set<std::string> names(outputNames.begin(), outputNames.end());
    EXPECT_EQ(names, (std::set<std::string>{"weights", "m", "v"}));
}

TEST(RAdamTest, CompileInitializesStateToZero) {
    Stream stream(gpuPlacement);

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    RAdam radam(2, 0.002f, 0.9f, 0.999f, 1e-7f);
    radam.compile(weights, stream);
    stream.synchronize();

    Tensor m = radam.getOptimizerParameterTensor("m");
    Tensor v = radam.getOptimizerParameterTensor("v");
    expectAllClose(copyGpuFp32TensorToValues(m, stream), {0.0f, 0.0f, 0.0f, 0.0f});
    expectAllClose(copyGpuFp32TensorToValues(v, stream), {0.0f, 0.0f, 0.0f, 0.0f});
}

TEST(RAdamTest, DenseStepsCoverUnrectifiedAndRectifiedBranchesAndMatchCpuReference) {
    Stream stream(gpuPlacement);

    const std::vector<float> initialWeights{1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f};
    const std::vector<std::vector<float>> gradients{
        {10.0f, -20.0f, 0.5f, -4.0f, 8.0f, -16.0f},
        {-3.0f, 5.0f, 7.0f, -11.0f, 13.0f, -17.0f},
        {1.5f, -2.5f, 3.5f, -4.5f, 5.5f, -6.5f},
        {-0.25f, 0.5f, -0.75f, 1.0f, -1.25f, 1.5f},
        {2.0f, -1.0f, 0.25f, -0.5f, 0.75f, -1.5f},
        {-1.25f, 2.25f, -3.25f, 4.25f, -5.25f, 6.25f},
    };
    const std::vector<uint32_t> batchSizes{2, 4, 3, 5, 6, 7};

    constexpr float alpha = 0.03f;
    constexpr float beta1 = 0.8f;
    constexpr float beta2 = 0.95f;
    constexpr float epsilon = 1e-4f;

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    RAdam radam(3, alpha, beta1, beta2, epsilon);
    radam.compile(weights, stream);
    stream.synchronize();

    RAdamReferenceState expected;
    expected.weights = initialWeights;
    expected.m.assign(initialWeights.size(), 0.0f);
    expected.v.assign(initialWeights.size(), 0.0f);

    for (uint64_t step = 0; step < gradients.size(); ++step) {
        applyRAdamReferenceStep(expected, gradients[step], batchSizes[step], alpha, beta1, beta2, epsilon);
        runRAdamStep(radam, gradients[step], batchSizes[step], stream);
    }

    Tensor m = radam.getOptimizerParameterTensor("m");
    Tensor v = radam.getOptimizerParameterTensor("v");

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 5e-5f, 5e-5f);
    expectAllClose(copyGpuFp32TensorToValues(m, stream), expected.m, 5e-5f, 5e-5f);
    expectAllClose(copyGpuFp32TensorToValues(v, stream), expected.v, 5e-5f, 5e-5f);
    EXPECT_FLOAT_EQ(radam.getT(), 6.0f);
    EXPECT_FLOAT_EQ(computeRAdamStepScalars(5.0f, alpha, beta1, beta2).useRectified, 0.0f);
    EXPECT_FLOAT_EQ(computeRAdamStepScalars(6.0f, alpha, beta1, beta2).useRectified, 1.0f);
}

TEST(RAdamTest, SparseRowUpdatesOnlyTouchedRowsAndMatchCpuReference) {
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

    RAdam radam(4, alpha, beta1, beta2, epsilon);
    SparseRowGradient gradient = radam.compileSparseRows(weights, maxSparseRows, stream);
    stream.synchronize();

    RAdamReferenceState expected;
    expected.weights = initialWeights;
    expected.m.assign(initialWeights.size(), 0.0f);
    expected.v.assign(initialWeights.size(), 0.0f);

    std::vector<uint64_t> rows1{1, 3};
    std::vector<float> values1{1.0f, -2.0f, 3.0f,
                               4.0f, -5.0f, 6.0f,
                               0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f};
    applySparseRAdamReferenceStep(expected, rows1, values1, /*numRows=*/2, embeddingDim, /*batchSize=*/2, alpha, beta1, beta2, epsilon);
    runSparseRAdamStep(radam, gradient, rows1, values1, /*numRows=*/2, /*batchSize=*/2, stream);

    std::vector<uint64_t> rows2{0, 3, 4};
    std::vector<float> values2{-1.0f, 2.0f, -3.0f,
                               0.5f, -0.25f, 0.75f,
                               8.0f, -9.0f, 10.0f,
                               0.0f, 0.0f, 0.0f};
    applySparseRAdamReferenceStep(expected, rows2, values2, /*numRows=*/3, embeddingDim, /*batchSize=*/4, alpha, beta1, beta2, epsilon);
    runSparseRAdamStep(radam, gradient, rows2, values2, /*numRows=*/3, /*batchSize=*/4, stream);

    Tensor m = radam.getOptimizerParameterTensor("m");
    Tensor v = radam.getOptimizerParameterTensor("v");

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 4e-5f, 4e-5f);
    expectAllClose(copyGpuFp32TensorToValues(m, stream), expected.m, 4e-5f, 4e-5f);
    expectAllClose(copyGpuFp32TensorToValues(v, stream), expected.v, 4e-5f, 4e-5f);
    EXPECT_FLOAT_EQ(radam.getT(), 2.0f);
}

TEST(RAdamTest, DenseAndSparseRuntimeScalarsIncludeRectificationAndLossScaling) {
    constexpr float alpha = 0.03f;
    constexpr float beta1 = 0.8f;
    constexpr float beta2 = 0.95f;
    RAdam radam(5, alpha, beta1, beta2, 1e-7f);

    std::unordered_map<std::string, float> dense = radam.denseUpdateRuntimeScalars(/*batchSize=*/5, "__test__");
    ASSERT_EQ(dense.size(), 4u);
    RAdamStepScalars denseScalars = computeRAdamStepScalars(1.0f, alpha, beta1, beta2);
    expectMapHasValue(dense, "__test__rectifiedAlphaT", denseScalars.rectifiedAlphaT);
    expectMapHasValue(dense, "__test__unrectifiedAlphaT", denseScalars.unrectifiedAlphaT);
    expectMapHasValue(dense, "__test__useRectified", denseScalars.useRectified);
    expectMapHasValue(dense, "__test__invBatchLossScale", 1.0f / (5.0f * Loss::getLossScalingFactor()));
    EXPECT_FLOAT_EQ(radam.getT(), 1.0f);
    EXPECT_FLOAT_EQ(denseScalars.useRectified, 0.0f);

    for (int i = 0; i < 4; ++i)
        (void)radam.denseUpdateRuntimeScalars(/*batchSize=*/5, "__warmup__");

    std::unordered_map<std::string, float> sparse = radam.sparseRowUpdateRuntimeScalars(/*batchSize=*/7);
    ASSERT_EQ(sparse.size(), 4u);
    RAdamStepScalars sparseScalars = computeRAdamStepScalars(6.0f, alpha, beta1, beta2);
    expectMapHasValue(sparse, "rectifiedAlphaT", sparseScalars.rectifiedAlphaT);
    expectMapHasValue(sparse, "unrectifiedAlphaT", sparseScalars.unrectifiedAlphaT);
    expectMapHasValue(sparse, "useRectified", sparseScalars.useRectified);
    expectMapHasValue(sparse, "invBatchLossScale", 1.0f / (7.0f * Loss::getLossScalingFactor()));
    EXPECT_FLOAT_EQ(radam.getT(), 6.0f);
    EXPECT_FLOAT_EQ(sparseScalars.useRectified, 1.0f);
}
