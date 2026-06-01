#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adagrad.h"
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

struct AdagradReferenceState {
    std::vector<float> weights;
    std::vector<float> accumulator;
};

void applyAdagradReferenceStep(AdagradReferenceState& state,
                               const std::vector<float>& rawGradient,
                               uint32_t batchSize,
                               float alpha,
                               float epsilon) {
    ASSERT_EQ(state.weights.size(), rawGradient.size());
    ASSERT_EQ(state.accumulator.size(), rawGradient.size());
    ASSERT_GT(batchSize, 0u);

    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());
    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float g = rawGradient[i] * invBatchLossScale;
        state.accumulator[i] += g * g;
        state.weights[i] -= alpha * g / (std::sqrt(state.accumulator[i]) + epsilon);
    }
}

void applySparseAdagradReferenceStep(AdagradReferenceState& state,
                                     const std::vector<uint64_t>& rows,
                                     const std::vector<float>& sparseGradientValues,
                                     uint64_t numRows,
                                     uint64_t embeddingDim,
                                     uint32_t batchSize,
                                     float alpha,
                                     float epsilon) {
    ASSERT_LE(numRows, rows.size());
    ASSERT_GE(sparseGradientValues.size(), numRows * embeddingDim);
    ASSERT_EQ(state.weights.size(), state.accumulator.size());
    ASSERT_GT(batchSize, 0u);

    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());
    for (uint64_t u = 0; u < numRows; ++u) {
        const uint64_t row = rows[u];
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            const uint64_t denseIndex = row * embeddingDim + d;
            ASSERT_LT(denseIndex, state.weights.size());
            const float g = sparseGradientValues[u * embeddingDim + d] * invBatchLossScale;
            state.accumulator[denseIndex] += g * g;
            state.weights[denseIndex] -= alpha * g / (std::sqrt(state.accumulator[denseIndex]) + epsilon);
        }
    }
}

void runAdagradStep(Adagrad& adagrad, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    std::optional<Tensor> gradientOpt = adagrad.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Tensor gradient = gradientOpt.value();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);

    adagrad.updateWeights(batchSize);
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

void runSparseAdagradStep(Adagrad& adagrad,
                          SparseRowGradient& gradient,
                          const std::vector<uint64_t>& rows,
                          const std::vector<float>& values,
                          uint64_t numRows,
                          uint32_t batchSize,
                          Stream& stream) {
    writeSparseGradient(gradient, rows, values, numRows, stream);
    adagrad.updateSparseRows(batchSize);
    stream.synchronize();
}

}  // namespace

TEST(AdagradTest, ConstructorGettersSettersAndHyperParameters) {
    constexpr uint64_t id = 19;
    Adagrad adagrad(id, 0.01f, 1e-7f);

    EXPECT_EQ(adagrad.getId(), id);
    EXPECT_FLOAT_EQ(adagrad.getAlpha(), 0.01f);
    EXPECT_FLOAT_EQ(adagrad.getEpsilon(), 1e-7f);

    adagrad.setAlpha(0.02f);
    adagrad.setEpsilon(1e-5f);

    EXPECT_FLOAT_EQ(adagrad.getAlpha(), 0.02f);
    EXPECT_FLOAT_EQ(adagrad.getEpsilon(), 1e-5f);

    std::unordered_map<std::string, float> updated = adagrad.updateHyperParameters(/*epoch=*/3, /*batch=*/4, /*batchesPerEpoch=*/5);
    EXPECT_TRUE(updated.empty());

    std::unordered_map<std::string, float> all = adagrad.getAllHyperParameters();
    ASSERT_EQ(all.size(), 2u);
    expectMapHasValue(all, "alpha", 0.02f);
    expectMapHasValue(all, "epsilon", 1e-5f);
}

TEST(AdagradTest, CompileCreatesGradientAccumulatorAndNamedOutputs) {
    Stream stream(gpuPlacement);

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP16, {2, 3}));

    Adagrad adagrad(1, 0.01f, 1e-7f);
    EXPECT_FALSE(adagrad.isCompiled());

    adagrad.compile(weights, stream);
    stream.synchronize();

    EXPECT_TRUE(adagrad.isCompiled());

    std::optional<Tensor> gradientOpt = adagrad.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Tensor gradient = gradientOpt.value();
    EXPECT_EQ(gradient.getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.getDataType(), DataType::FP16);
    EXPECT_EQ(gradient.getDimensions(), weights.getDimensions());

    EXPECT_TRUE(adagrad.hasParameter("accumulator"));

    Tensor weightsOut = adagrad.getOptimizerParameterTensor("weights");
    Tensor accumulator = adagrad.getOptimizerParameterTensor("accumulator");

    EXPECT_EQ(weightsOut, weights);
    EXPECT_EQ(accumulator.getPlacement(), gpuPlacement);
    EXPECT_EQ(accumulator.getDataType(), DataType::FP32);
    EXPECT_EQ(accumulator.getDimensions(), weights.getDimensions());

    std::vector<std::string> outputNames = adagrad.getOptimizerParameterNames();
    std::set<std::string> names(outputNames.begin(), outputNames.end());
    EXPECT_EQ(names, (std::set<std::string>{"weights", "accumulator"}));
}

TEST(AdagradTest, CompileInitializesAccumulatorToZero) {
    Stream stream(gpuPlacement);

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    Adagrad adagrad(2, 0.01f, 1e-7f);
    adagrad.compile(weights, stream);
    stream.synchronize();

    Tensor accumulator = adagrad.getOptimizerParameterTensor("accumulator");
    expectAllClose(copyGpuFp32TensorToValues(accumulator, stream), {0.0f, 0.0f, 0.0f, 0.0f});
}

TEST(AdagradTest, TwoDenseStepsMatchCpuReference) {
    Stream stream(gpuPlacement);

    const std::vector<float> initialWeights{1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f};
    const std::vector<float> gradient1{10.0f, -20.0f, 0.5f, -4.0f, 8.0f, -16.0f};
    const std::vector<float> gradient2{-3.0f, 5.0f, 7.0f, -11.0f, 13.0f, -17.0f};

    constexpr float alpha = 0.05f;
    constexpr float epsilon = 1e-3f;

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Adagrad adagrad(3, alpha, epsilon);
    adagrad.compile(weights, stream);
    stream.synchronize();

    AdagradReferenceState expected;
    expected.weights = initialWeights;
    expected.accumulator.assign(initialWeights.size(), 0.0f);

    applyAdagradReferenceStep(expected, gradient1, /*batchSize=*/2, alpha, epsilon);
    runAdagradStep(adagrad, gradient1, /*batchSize=*/2, stream);

    applyAdagradReferenceStep(expected, gradient2, /*batchSize=*/4, alpha, epsilon);
    runAdagradStep(adagrad, gradient2, /*batchSize=*/4, stream);

    Tensor accumulator = adagrad.getOptimizerParameterTensor("accumulator");

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(accumulator, stream), expected.accumulator, 3e-5f, 3e-5f);
}

TEST(AdagradTest, SparseRowUpdatesOnlyTouchedRowsAndMatchCpuReference) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabSize = 5;
    constexpr uint64_t embeddingDim = 3;
    constexpr uint64_t maxSparseRows = 4;
    constexpr float alpha = 0.07f;
    constexpr float epsilon = 1e-4f;

    const std::vector<float> initialWeights{1.0f,  2.0f,  3.0f,
                                            -1.0f, -2.0f, -3.0f,
                                            4.0f,  5.0f,  6.0f,
                                            -4.0f, -5.0f, -6.0f,
                                            7.0f,  8.0f,  9.0f};

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {vocabSize, embeddingDim}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Adagrad adagrad(4, alpha, epsilon);
    SparseRowGradient gradient = adagrad.compileSparseRows(weights, maxSparseRows, stream);
    stream.synchronize();

    AdagradReferenceState expected;
    expected.weights = initialWeights;
    expected.accumulator.assign(initialWeights.size(), 0.0f);

    std::vector<uint64_t> rows1{1, 3};
    std::vector<float> values1{1.0f, -2.0f, 3.0f,
                               4.0f, -5.0f, 6.0f,
                               0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f};
    applySparseAdagradReferenceStep(expected, rows1, values1, /*numRows=*/2, embeddingDim, /*batchSize=*/2, alpha, epsilon);
    runSparseAdagradStep(adagrad, gradient, rows1, values1, /*numRows=*/2, /*batchSize=*/2, stream);

    std::vector<uint64_t> rows2{0, 3, 4};
    std::vector<float> values2{-1.0f, 2.0f, -3.0f,
                               0.5f, -0.25f, 0.75f,
                               8.0f, -9.0f, 10.0f,
                               0.0f, 0.0f, 0.0f};
    applySparseAdagradReferenceStep(expected, rows2, values2, /*numRows=*/3, embeddingDim, /*batchSize=*/4, alpha, epsilon);
    runSparseAdagradStep(adagrad, gradient, rows2, values2, /*numRows=*/3, /*batchSize=*/4, stream);

    Tensor accumulator = adagrad.getOptimizerParameterTensor("accumulator");

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 4e-5f, 4e-5f);
    expectAllClose(copyGpuFp32TensorToValues(accumulator, stream), expected.accumulator, 4e-5f, 4e-5f);
}

TEST(AdagradTest, DenseAndSparseRuntimeScalarsIncludeLossScaling) {
    Adagrad adagrad(5, 0.03f, 1e-7f);

    std::unordered_map<std::string, float> dense = adagrad.denseUpdateRuntimeScalars(/*batchSize=*/5, "__test__");
    ASSERT_EQ(dense.size(), 2u);
    expectMapHasValue(dense, "__test__alpha", 0.03f);
    expectMapHasValue(dense, "__test__invBatchLossScale", 1.0f / (5.0f * Loss::getLossScalingFactor()));

    std::unordered_map<std::string, float> sparse = adagrad.sparseRowUpdateRuntimeScalars(/*batchSize=*/7);
    ASSERT_EQ(sparse.size(), 2u);
    expectMapHasValue(sparse, "alpha", 0.03f);
    expectMapHasValue(sparse, "invBatchLossScale", 1.0f / (7.0f * Loss::getLossScalingFactor()));
}

