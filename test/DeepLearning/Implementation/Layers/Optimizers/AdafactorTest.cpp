#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adafactor.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
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

struct FactoredReferenceState {
    std::vector<float> weights;
    std::vector<float> row_second_moment;
    std::vector<float> column_second_moment;
};

struct UnfactoredReferenceState {
    std::vector<float> weights;
    std::vector<float> second_moment;
};

void applyFactoredAdafactorReferenceStep(FactoredReferenceState& state,
                                         const std::vector<float>& rawGradient,
                                         uint64_t rows,
                                         uint64_t columns,
                                         uint32_t batchSize,
                                         float alpha,
                                         float beta2,
                                         float epsilon,
                                         float weightDecay) {
    ASSERT_EQ(state.weights.size(), rows * columns);
    ASSERT_EQ(rawGradient.size(), rows * columns);
    ASSERT_EQ(state.row_second_moment.size(), rows);
    ASSERT_EQ(state.column_second_moment.size(), columns);
    ASSERT_GT(batchSize, 0u);

    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());
    std::vector<float> g(rawGradient.size());
    for (uint64_t i = 0; i < rawGradient.size(); ++i)
        g[i] = rawGradient[i] * invBatchLossScale;

    for (uint64_t r = 0; r < rows; ++r) {
        float mean = 0.0f;
        for (uint64_t c = 0; c < columns; ++c)
            mean += g[r * columns + c] * g[r * columns + c];
        mean /= static_cast<float>(columns);
        state.row_second_moment[r] = beta2 * state.row_second_moment[r] + (1.0f - beta2) * mean;
    }

    for (uint64_t c = 0; c < columns; ++c) {
        float mean = 0.0f;
        for (uint64_t r = 0; r < rows; ++r)
            mean += g[r * columns + c] * g[r * columns + c];
        mean /= static_cast<float>(rows);
        state.column_second_moment[c] = beta2 * state.column_second_moment[c] + (1.0f - beta2) * mean;
    }

    float rowMean = 0.0f;
    for (float v : state.row_second_moment)
        rowMean += v;
    rowMean /= static_cast<float>(rows);

    for (uint64_t r = 0; r < rows; ++r) {
        for (uint64_t c = 0; c < columns; ++c) {
            const uint64_t idx = r * columns + c;
            const float secondMomentEstimate = state.row_second_moment[r] * state.column_second_moment[c] / (rowMean + epsilon);
            const float update = g[idx] / std::sqrt(secondMomentEstimate + epsilon);
            state.weights[idx] -= alpha * weightDecay * state.weights[idx] + alpha * update;
        }
    }
}

void applyUnfactoredAdafactorReferenceStep(UnfactoredReferenceState& state,
                                           const std::vector<float>& rawGradient,
                                           uint32_t batchSize,
                                           float alpha,
                                           float beta2,
                                           float epsilon,
                                           float weightDecay) {
    ASSERT_EQ(state.weights.size(), rawGradient.size());
    ASSERT_EQ(state.second_moment.size(), rawGradient.size());
    ASSERT_GT(batchSize, 0u);

    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());
    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float g = rawGradient[i] * invBatchLossScale;
        state.second_moment[i] = beta2 * state.second_moment[i] + (1.0f - beta2) * g * g;
        const float update = g / std::sqrt(state.second_moment[i] + epsilon);
        state.weights[i] -= alpha * weightDecay * state.weights[i] + alpha * update;
    }
}

void applySparseUnfactoredAdafactorReferenceStep(UnfactoredReferenceState& state,
                                                 const std::vector<uint64_t>& rows,
                                                 const std::vector<float>& sparseGradientValues,
                                                 uint64_t numRows,
                                                 uint64_t embeddingDim,
                                                 uint32_t batchSize,
                                                 float alpha,
                                                 float beta2,
                                                 float epsilon,
                                                 float weightDecay) {
    ASSERT_LE(numRows, rows.size());
    ASSERT_GE(sparseGradientValues.size(), numRows * embeddingDim);
    ASSERT_EQ(state.weights.size(), state.second_moment.size());
    ASSERT_GT(batchSize, 0u);

    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());
    for (uint64_t u = 0; u < numRows; ++u) {
        const uint64_t row = rows[u];
        for (uint64_t d = 0; d < embeddingDim; ++d) {
            const uint64_t denseIndex = row * embeddingDim + d;
            ASSERT_LT(denseIndex, state.weights.size());
            const float g = sparseGradientValues[u * embeddingDim + d] * invBatchLossScale;
            state.second_moment[denseIndex] = beta2 * state.second_moment[denseIndex] + (1.0f - beta2) * g * g;
            const float update = g / std::sqrt(state.second_moment[denseIndex] + epsilon);
            state.weights[denseIndex] -= alpha * weightDecay * state.weights[denseIndex] + alpha * update;
        }
    }
}

void runAdafactorStep(Adafactor& adafactor, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    std::optional<Tensor> gradientOpt = adafactor.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Tensor gradient = gradientOpt.value();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);

    adafactor.updateWeights(batchSize);
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

void runSparseAdafactorStep(Adafactor& adafactor,
                            SparseRowGradient& gradient,
                            const std::vector<uint64_t>& rows,
                            const std::vector<float>& values,
                            uint64_t numRows,
                            uint32_t batchSize,
                            Stream& stream) {
    writeSparseGradient(gradient, rows, values, numRows, stream);
    adafactor.updateSparseRows(batchSize);
    stream.synchronize();
}

}  // namespace

TEST(AdafactorTest, ConstructorGettersSettersAndHyperParameters) {
    constexpr uint64_t id = 23;
    Adafactor adafactor(id, 0.001f, 0.999f, 1e-30f, 0.0f, true);

    EXPECT_EQ(adafactor.getId(), id);
    EXPECT_FLOAT_EQ(adafactor.getAlpha(), 0.001f);
    EXPECT_FLOAT_EQ(adafactor.getBeta2(), 0.999f);
    EXPECT_FLOAT_EQ(adafactor.getEpsilon(), 1e-30f);
    EXPECT_FLOAT_EQ(adafactor.getWeightDecay(), 0.0f);
    EXPECT_TRUE(adafactor.getFactorSecondMoment());

    adafactor.setAlpha(0.01f);
    adafactor.setBeta2(0.98f);
    adafactor.setEpsilon(1e-6f);
    adafactor.setWeightDecay(0.02f);
    adafactor.setFactorSecondMoment(false);

    EXPECT_FLOAT_EQ(adafactor.getAlpha(), 0.01f);
    EXPECT_FLOAT_EQ(adafactor.getBeta2(), 0.98f);
    EXPECT_FLOAT_EQ(adafactor.getEpsilon(), 1e-6f);
    EXPECT_FLOAT_EQ(adafactor.getWeightDecay(), 0.02f);
    EXPECT_FALSE(adafactor.getFactorSecondMoment());

    std::unordered_map<std::string, float> updated = adafactor.updateHyperParameters(/*epoch=*/3, /*batch=*/4, /*batchesPerEpoch=*/5);
    EXPECT_TRUE(updated.empty());

    std::unordered_map<std::string, float> all = adafactor.getAllHyperParameters();
    ASSERT_EQ(all.size(), 5u);
    expectMapHasValue(all, "alpha", 0.01f);
    expectMapHasValue(all, "beta2", 0.98f);
    expectMapHasValue(all, "epsilon", 1e-6f);
    expectMapHasValue(all, "weightDecay", 0.02f);
    expectMapHasValue(all, "factorSecondMoment", 0.0f);

    EXPECT_THROW(adafactor.setAlpha(0.0f), std::exception);
    EXPECT_THROW(adafactor.setBeta2(1.0f), std::exception);
    EXPECT_THROW(adafactor.setEpsilon(0.0f), std::exception);
    EXPECT_THROW(adafactor.setWeightDecay(-0.01f), std::exception);
}

TEST(AdafactorTest, DenseMatrixUsesFactoredStateAndMatchesCpuReference) {
    Stream stream(gpuPlacement);

    constexpr uint64_t rows = 2;
    constexpr uint64_t columns = 3;
    constexpr float alpha = 0.003f;
    constexpr float beta2 = 0.75f;
    constexpr float epsilon = 1e-6f;
    constexpr float weightDecay = 0.01f;

    const std::vector<float> initialWeights{1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f};
    const std::vector<float> gradient1{10.0f, -20.0f, 0.5f, -4.0f, 8.0f, -16.0f};
    const std::vector<float> gradient2{-3.0f, 5.0f, 7.0f, -11.0f, 13.0f, -17.0f};

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {rows, columns}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Adafactor adafactor(1, alpha, beta2, epsilon, weightDecay, true);
    adafactor.compile(weights, stream);
    stream.synchronize();

    ASSERT_TRUE(adafactor.isUsingFactoredPath());
    std::shared_ptr<Optimizer> selected = adafactor.getSelectedOptimizer();
    ASSERT_NE(selected, nullptr);
    ASSERT_TRUE(selected->hasParameter("row_second_moment"));
    ASSERT_TRUE(selected->hasParameter("column_second_moment"));

    FactoredReferenceState expected;
    expected.weights = initialWeights;
    expected.row_second_moment.assign(rows, 0.0f);
    expected.column_second_moment.assign(columns, 0.0f);

    applyFactoredAdafactorReferenceStep(expected, gradient1, rows, columns, /*batchSize=*/2, alpha, beta2, epsilon, weightDecay);
    runAdafactorStep(adafactor, gradient1, /*batchSize=*/2, stream);

    applyFactoredAdafactorReferenceStep(expected, gradient2, rows, columns, /*batchSize=*/4, alpha, beta2, epsilon, weightDecay);
    runAdafactorStep(adafactor, gradient2, /*batchSize=*/4, stream);

    Tensor rowSecondMoment = selected->getParameter("row_second_moment")->getStorage().value();
    Tensor columnSecondMoment = selected->getParameter("column_second_moment")->getStorage().value();

    EXPECT_EQ(rowSecondMoment.getDimensions(), (std::vector<uint64_t>{rows, 1}));
    EXPECT_EQ(columnSecondMoment.getDimensions(), (std::vector<uint64_t>{1, columns}));
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 5e-5f, 5e-5f);
    expectAllClose(copyGpuFp32TensorToValues(rowSecondMoment, stream), expected.row_second_moment, 5e-5f, 5e-5f);
    expectAllClose(copyGpuFp32TensorToValues(columnSecondMoment, stream), expected.column_second_moment, 5e-5f, 5e-5f);
}

TEST(AdafactorTest, DenseVectorUsesUnfactoredStateAndMatchesCpuReference) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.004f;
    constexpr float beta2 = 0.8f;
    constexpr float epsilon = 1e-6f;
    constexpr float weightDecay = 0.02f;

    const std::vector<float> initialWeights{1.0f, -2.0f, 3.0f, -4.0f};
    const std::vector<float> gradient1{4.0f, -8.0f, 12.0f, -16.0f};
    const std::vector<float> gradient2{-5.0f, 7.0f, -9.0f, 11.0f};

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {4}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Adafactor adafactor(2, alpha, beta2, epsilon, weightDecay, true);
    adafactor.compile(weights, stream);
    stream.synchronize();

    ASSERT_TRUE(adafactor.isUsingUnfactoredPath());
    std::shared_ptr<Optimizer> selected = adafactor.getSelectedOptimizer();
    ASSERT_NE(selected, nullptr);
    ASSERT_TRUE(selected->hasParameter("second_moment"));

    UnfactoredReferenceState expected;
    expected.weights = initialWeights;
    expected.second_moment.assign(initialWeights.size(), 0.0f);

    applyUnfactoredAdafactorReferenceStep(expected, gradient1, /*batchSize=*/2, alpha, beta2, epsilon, weightDecay);
    runAdafactorStep(adafactor, gradient1, /*batchSize=*/2, stream);

    applyUnfactoredAdafactorReferenceStep(expected, gradient2, /*batchSize=*/3, alpha, beta2, epsilon, weightDecay);
    runAdafactorStep(adafactor, gradient2, /*batchSize=*/3, stream);

    Tensor secondMoment = selected->getParameter("second_moment")->getStorage().value();
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 5e-5f, 5e-5f);
    expectAllClose(copyGpuFp32TensorToValues(secondMoment, stream), expected.second_moment, 5e-5f, 5e-5f);
}

TEST(AdafactorTest, SparseRowsUseUnfactoredStateAndOnlyUpdateTouchedRows) {
    Stream stream(gpuPlacement);

    constexpr uint64_t vocabSize = 5;
    constexpr uint64_t embeddingDim = 3;
    constexpr uint64_t maxSparseRows = 4;
    constexpr float alpha = 0.006f;
    constexpr float beta2 = 0.7f;
    constexpr float epsilon = 1e-6f;
    constexpr float weightDecay = 0.01f;

    const std::vector<float> initialWeights{1.0f,  2.0f,  3.0f,
                                            -1.0f, -2.0f, -3.0f,
                                            4.0f,  5.0f,  6.0f,
                                            -4.0f, -5.0f, -6.0f,
                                            7.0f,  8.0f,  9.0f};

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {vocabSize, embeddingDim}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Adafactor adafactor(3, alpha, beta2, epsilon, weightDecay, true);
    SparseRowGradient gradient = adafactor.compileSparseRows(weights, maxSparseRows, stream);
    stream.synchronize();

    ASSERT_TRUE(adafactor.isUsingUnfactoredPath());
    std::shared_ptr<Optimizer> selected = adafactor.getSelectedOptimizer();
    ASSERT_NE(selected, nullptr);
    ASSERT_TRUE(selected->hasParameter("second_moment"));

    UnfactoredReferenceState expected;
    expected.weights = initialWeights;
    expected.second_moment.assign(initialWeights.size(), 0.0f);

    std::vector<uint64_t> rows1{1, 3};
    std::vector<float> values1{1.0f, -2.0f, 3.0f,
                               4.0f, -5.0f, 6.0f,
                               0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f};
    applySparseUnfactoredAdafactorReferenceStep(expected, rows1, values1, /*numRows=*/2, embeddingDim, /*batchSize=*/2, alpha, beta2, epsilon, weightDecay);
    runSparseAdafactorStep(adafactor, gradient, rows1, values1, /*numRows=*/2, /*batchSize=*/2, stream);

    std::vector<uint64_t> rows2{0, 3, 4};
    std::vector<float> values2{-1.0f, 2.0f, -3.0f,
                               0.5f, -0.25f, 0.75f,
                               8.0f, -9.0f, 10.0f,
                               0.0f, 0.0f, 0.0f};
    applySparseUnfactoredAdafactorReferenceStep(expected, rows2, values2, /*numRows=*/3, embeddingDim, /*batchSize=*/4, alpha, beta2, epsilon, weightDecay);
    runSparseAdafactorStep(adafactor, gradient, rows2, values2, /*numRows=*/3, /*batchSize=*/4, stream);

    Tensor secondMoment = selected->getParameter("second_moment")->getStorage().value();
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 5e-5f, 5e-5f);
    expectAllClose(copyGpuFp32TensorToValues(secondMoment, stream), expected.second_moment, 5e-5f, 5e-5f);
}

TEST(AdafactorTest, RuntimeScalarsIncludeLossScalingAndDecoupledWeightDecay) {
    Adafactor adafactor(4, 0.03f, 0.9f, 1e-6f, 0.2f, true);

    std::unordered_map<std::string, float> dense = adafactor.denseUpdateRuntimeScalars(/*batchSize=*/5, "__test__");
    ASSERT_EQ(dense.size(), 3u);
    expectMapHasValue(dense, "__test__alpha", 0.03f);
    expectMapHasValue(dense, "__test__alphaWeightDecay", 0.006f);
    expectMapHasValue(dense, "__test__invBatchLossScale", 1.0f / (5.0f * Loss::getLossScalingFactor()));

    std::unordered_map<std::string, float> sparse = adafactor.sparseRowUpdateRuntimeScalars(/*batchSize=*/7);
    ASSERT_EQ(sparse.size(), 3u);
    expectMapHasValue(sparse, "alpha", 0.03f);
    expectMapHasValue(sparse, "alphaWeightDecay", 0.006f);
    expectMapHasValue(sparse, "invBatchLossScale", 1.0f / (7.0f * Loss::getLossScalingFactor()));
}
