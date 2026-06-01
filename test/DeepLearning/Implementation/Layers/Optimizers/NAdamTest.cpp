#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/NAdam.h"
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

struct NAdamReferenceState {
    std::vector<float> weights;
    std::vector<float> m;
    std::vector<float> v;
    float t = 0.0f;
};

void applyNAdamReferenceStep(NAdamReferenceState& state,
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
    const double sqrtOneMinusBeta2PowT = std::sqrt(1.0 - std::pow(static_cast<double>(beta2), state.t));
    const double mScale64 = static_cast<double>(alpha) * static_cast<double>(beta1) * sqrtOneMinusBeta2PowT /
                            (1.0 - std::pow(static_cast<double>(beta1), state.t + 1.0f));
    const double gradientScale64 = static_cast<double>(alpha) * (1.0 - static_cast<double>(beta1)) * sqrtOneMinusBeta2PowT /
                                   (1.0 - std::pow(static_cast<double>(beta1), state.t));
    const float mScale = static_cast<float>(mScale64);
    const float gradientScale = static_cast<float>(gradientScale64);

    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float g = rawGradient[i] * invBatchLossScale;
        state.m[i] = beta1 * state.m[i] + (1.0f - beta1) * g;
        state.v[i] = beta2 * state.v[i] + (1.0f - beta2) * g * g;
        const float nesterovMomentum = mScale * state.m[i] + gradientScale * g;
        state.weights[i] -= nesterovMomentum / (std::sqrt(state.v[i]) + epsilon);
    }
}

void applySparseNAdamReferenceStep(NAdamReferenceState& state,
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
    const double sqrtOneMinusBeta2PowT = std::sqrt(1.0 - std::pow(static_cast<double>(beta2), state.t));
    const double mScale64 = static_cast<double>(alpha) * static_cast<double>(beta1) * sqrtOneMinusBeta2PowT /
                            (1.0 - std::pow(static_cast<double>(beta1), state.t + 1.0f));
    const double gradientScale64 = static_cast<double>(alpha) * (1.0 - static_cast<double>(beta1)) * sqrtOneMinusBeta2PowT /
                                   (1.0 - std::pow(static_cast<double>(beta1), state.t));
    const float mScale = static_cast<float>(mScale64);
    const float gradientScale = static_cast<float>(gradientScale64);

    for (uint64_t r = 0; r < numRows; ++r) {
        const uint64_t row = rows[r];
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            const uint64_t denseIndex = row * embeddingDim + d;
            ASSERT_LT(denseIndex, state.weights.size());
            const float g = sparseGradientValues[r * embeddingDim + d] * invBatchLossScale;
            state.m[denseIndex] = beta1 * state.m[denseIndex] + (1.0f - beta1) * g;
            state.v[denseIndex] = beta2 * state.v[denseIndex] + (1.0f - beta2) * g * g;
            const float nesterovMomentum = mScale * state.m[denseIndex] + gradientScale * g;
            state.weights[denseIndex] -= nesterovMomentum / (std::sqrt(state.v[denseIndex]) + epsilon);
        }
    }
}

void runNAdamStep(NAdam& nadam, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    std::optional<Tensor> gradientOpt = nadam.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Tensor gradient = gradientOpt.value();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);

    nadam.updateWeights(batchSize);
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

void runSparseNAdamStep(NAdam& nadam,
                         SparseRowGradient& gradient,
                         const std::vector<uint64_t>& rows,
                         const std::vector<float>& values,
                         uint64_t numRows,
                         uint32_t batchSize,
                         Stream& stream) {
    writeSparseGradient(gradient, rows, values, numRows, stream);
    nadam.updateSparseRows(batchSize);
    stream.synchronize();
}

}  // namespace

TEST(NAdamTest, ConstructorGettersSettersAndHyperParameters) {
    constexpr uint64_t id = 23;
    NAdam nadam(id, 0.002f, 0.9f, 0.999f, 1e-7f);

    EXPECT_EQ(nadam.getId(), id);
    EXPECT_FLOAT_EQ(nadam.getAlpha(), 0.002f);
    EXPECT_FLOAT_EQ(nadam.getBeta1(), 0.9f);
    EXPECT_FLOAT_EQ(nadam.getBeta2(), 0.999f);
    EXPECT_FLOAT_EQ(nadam.getEpsilon(), 1e-7f);
    EXPECT_FLOAT_EQ(nadam.getT(), 0.0f);

    nadam.setAlpha(0.01f);
    nadam.setBeta1(0.8f);
    nadam.setBeta2(0.95f);
    nadam.setEpsilon(1e-5f);
    nadam.setT(7.0f);

    EXPECT_FLOAT_EQ(nadam.getAlpha(), 0.01f);
    EXPECT_FLOAT_EQ(nadam.getBeta1(), 0.8f);
    EXPECT_FLOAT_EQ(nadam.getBeta2(), 0.95f);
    EXPECT_FLOAT_EQ(nadam.getEpsilon(), 1e-5f);
    EXPECT_FLOAT_EQ(nadam.getT(), 7.0f);

    std::unordered_map<std::string, float> updated = nadam.updateHyperParameters(/*epoch=*/3, /*batch=*/4, /*batchesPerEpoch=*/5);
    ASSERT_EQ(updated.size(), 1u);
    expectMapHasValue(updated, "t", 7.0f);

    std::unordered_map<std::string, float> all = nadam.getAllHyperParameters();
    ASSERT_EQ(all.size(), 5u);
    expectMapHasValue(all, "t", 7.0f);
    expectMapHasValue(all, "alpha", 0.01f);
    expectMapHasValue(all, "beta1", 0.8f);
    expectMapHasValue(all, "beta2", 0.95f);
    expectMapHasValue(all, "epsilon", 1e-5f);
}

TEST(NAdamTest, CompileCreatesGradientMomentSecondMomentAndNamedOutputs) {
    Stream stream(gpuPlacement);

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP16, {2, 3}));

    NAdam nadam(1, 0.002f, 0.9f, 0.999f, 1e-7f);
    EXPECT_FALSE(nadam.isCompiled());

    nadam.compile(weights, stream);
    stream.synchronize();

    EXPECT_TRUE(nadam.isCompiled());

    std::optional<Tensor> gradientOpt = nadam.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Tensor gradient = gradientOpt.value();
    EXPECT_EQ(gradient.getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.getDataType(), DataType::FP16);
    EXPECT_EQ(gradient.getDimensions(), weights.getDimensions());

    EXPECT_TRUE(nadam.hasParameter("m"));
    EXPECT_TRUE(nadam.hasParameter("v"));

    Tensor weightsOut = nadam.getOptimizerParameterTensor("weights");
    Tensor m = nadam.getOptimizerParameterTensor("m");
    Tensor v = nadam.getOptimizerParameterTensor("v");

    EXPECT_EQ(weightsOut, weights);
    EXPECT_EQ(m.getPlacement(), gpuPlacement);
    EXPECT_EQ(m.getDataType(), DataType::FP32);
    EXPECT_EQ(m.getDimensions(), weights.getDimensions());
    EXPECT_EQ(v.getPlacement(), gpuPlacement);
    EXPECT_EQ(v.getDataType(), DataType::FP32);
    EXPECT_EQ(v.getDimensions(), weights.getDimensions());

    std::vector<std::string> outputNames = nadam.getOptimizerParameterNames();
    std::set<std::string> names(outputNames.begin(), outputNames.end());
    EXPECT_EQ(names, (std::set<std::string>{"weights", "m", "v"}));
}

TEST(NAdamTest, CompileInitializesStateToZero) {
    Stream stream(gpuPlacement);

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    NAdam nadam(2, 0.002f, 0.9f, 0.999f, 1e-7f);
    nadam.compile(weights, stream);
    stream.synchronize();

    Tensor m = nadam.getOptimizerParameterTensor("m");
    Tensor v = nadam.getOptimizerParameterTensor("v");
    expectAllClose(copyGpuFp32TensorToValues(m, stream), {0.0f, 0.0f, 0.0f, 0.0f});
    expectAllClose(copyGpuFp32TensorToValues(v, stream), {0.0f, 0.0f, 0.0f, 0.0f});
}

TEST(NAdamTest, TwoDenseStepsMatchCpuReference) {
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

    NAdam nadam(3, alpha, beta1, beta2, epsilon);
    nadam.compile(weights, stream);
    stream.synchronize();

    NAdamReferenceState expected;
    expected.weights = initialWeights;
    expected.m.assign(initialWeights.size(), 0.0f);
    expected.v.assign(initialWeights.size(), 0.0f);

    applyNAdamReferenceStep(expected, gradient1, /*batchSize=*/2, alpha, beta1, beta2, epsilon);
    runNAdamStep(nadam, gradient1, /*batchSize=*/2, stream);

    applyNAdamReferenceStep(expected, gradient2, /*batchSize=*/4, alpha, beta1, beta2, epsilon);
    runNAdamStep(nadam, gradient2, /*batchSize=*/4, stream);

    Tensor m = nadam.getOptimizerParameterTensor("m");
    Tensor v = nadam.getOptimizerParameterTensor("v");

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(m, stream), expected.m, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(v, stream), expected.v, 3e-5f, 3e-5f);
    EXPECT_FLOAT_EQ(nadam.getT(), 2.0f);
}

TEST(NAdamTest, SparseRowUpdatesOnlyTouchedRowsAndMatchCpuReference) {
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

    NAdam nadam(4, alpha, beta1, beta2, epsilon);
    SparseRowGradient gradient = nadam.compileSparseRows(weights, maxSparseRows, stream);
    stream.synchronize();

    NAdamReferenceState expected;
    expected.weights = initialWeights;
    expected.m.assign(initialWeights.size(), 0.0f);
    expected.v.assign(initialWeights.size(), 0.0f);

    std::vector<uint64_t> rows1{1, 3};
    std::vector<float> values1{1.0f, -2.0f, 3.0f,
                               4.0f, -5.0f, 6.0f,
                               0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f};
    applySparseNAdamReferenceStep(expected, rows1, values1, /*numRows=*/2, embeddingDim, /*batchSize=*/2, alpha, beta1, beta2, epsilon);
    runSparseNAdamStep(nadam, gradient, rows1, values1, /*numRows=*/2, /*batchSize=*/2, stream);

    std::vector<uint64_t> rows2{0, 3, 4};
    std::vector<float> values2{-1.0f, 2.0f, -3.0f,
                               0.5f, -0.25f, 0.75f,
                               8.0f, -9.0f, 10.0f,
                               0.0f, 0.0f, 0.0f};
    applySparseNAdamReferenceStep(expected, rows2, values2, /*numRows=*/3, embeddingDim, /*batchSize=*/4, alpha, beta1, beta2, epsilon);
    runSparseNAdamStep(nadam, gradient, rows2, values2, /*numRows=*/3, /*batchSize=*/4, stream);

    Tensor m = nadam.getOptimizerParameterTensor("m");
    Tensor v = nadam.getOptimizerParameterTensor("v");

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 4e-5f, 4e-5f);
    expectAllClose(copyGpuFp32TensorToValues(m, stream), expected.m, 4e-5f, 4e-5f);
    expectAllClose(copyGpuFp32TensorToValues(v, stream), expected.v, 4e-5f, 4e-5f);
    EXPECT_FLOAT_EQ(nadam.getT(), 2.0f);
}

TEST(NAdamTest, DenseAndSparseRuntimeScalarsIncludeBiasCorrectionAndLossScaling) {
    constexpr float alpha = 0.03f;
    constexpr float beta1 = 0.8f;
    constexpr float beta2 = 0.95f;
    NAdam nadam(5, alpha, beta1, beta2, 1e-7f);

    std::unordered_map<std::string, float> dense = nadam.denseUpdateRuntimeScalars(/*batchSize=*/5, "__test__");
    ASSERT_EQ(dense.size(), 3u);
    const float denseMScale = static_cast<float>(static_cast<double>(alpha) * static_cast<double>(beta1) *
                                                 std::sqrt(1.0 - std::pow(static_cast<double>(beta2), 1.0)) /
                                                 (1.0 - std::pow(static_cast<double>(beta1), 2.0)));
    const float denseGradientScale = static_cast<float>(static_cast<double>(alpha) * (1.0 - static_cast<double>(beta1)) *
                                                        std::sqrt(1.0 - std::pow(static_cast<double>(beta2), 1.0)) /
                                                        (1.0 - std::pow(static_cast<double>(beta1), 1.0)));
    expectMapHasValue(dense, "__test__mScale", denseMScale);
    expectMapHasValue(dense, "__test__gradientScale", denseGradientScale);
    expectMapHasValue(dense, "__test__invBatchLossScale", 1.0f / (5.0f * Loss::getLossScalingFactor()));
    EXPECT_FLOAT_EQ(nadam.getT(), 1.0f);

    std::unordered_map<std::string, float> sparse = nadam.sparseRowUpdateRuntimeScalars(/*batchSize=*/7);
    ASSERT_EQ(sparse.size(), 3u);
    const float sparseMScale = static_cast<float>(static_cast<double>(alpha) * static_cast<double>(beta1) *
                                                  std::sqrt(1.0 - std::pow(static_cast<double>(beta2), 2.0)) /
                                                  (1.0 - std::pow(static_cast<double>(beta1), 3.0)));
    const float sparseGradientScale = static_cast<float>(static_cast<double>(alpha) * (1.0 - static_cast<double>(beta1)) *
                                                         std::sqrt(1.0 - std::pow(static_cast<double>(beta2), 2.0)) /
                                                         (1.0 - std::pow(static_cast<double>(beta1), 2.0)));
    expectMapHasValue(sparse, "mScale", sparseMScale);
    expectMapHasValue(sparse, "gradientScale", sparseGradientScale);
    expectMapHasValue(sparse, "invBatchLossScale", 1.0f / (7.0f * Loss::getLossScalingFactor()));
    EXPECT_FLOAT_EQ(nadam.getT(), 2.0f);
}
