#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/AdamW.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Muon.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

using namespace ThorImplementation;

namespace {

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

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

void expectAllClose(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 1e-4f, float rtol = 1e-4f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        const float tol = atol + rtol * std::fabs(expected[i]);
        EXPECT_LE(diff, tol) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

std::vector<float> matmul(const std::vector<float>& a,
                          uint64_t aRows,
                          uint64_t aCols,
                          const std::vector<float>& b,
                          uint64_t bRows,
                          uint64_t bCols) {
    EXPECT_EQ(aCols, bRows);
    std::vector<float> out(aRows * bCols, 0.0f);
    for (uint64_t i = 0; i < aRows; ++i) {
        for (uint64_t k = 0; k < aCols; ++k) {
            const float av = a[i * aCols + k];
            for (uint64_t j = 0; j < bCols; ++j)
                out[i * bCols + j] += av * b[k * bCols + j];
        }
    }
    return out;
}

std::vector<float> transpose(const std::vector<float>& x, uint64_t rows, uint64_t cols) {
    std::vector<float> out(rows * cols);
    for (uint64_t r = 0; r < rows; ++r) {
        for (uint64_t c = 0; c < cols; ++c)
            out[c * rows + r] = x[r * cols + c];
    }
    return out;
}

std::vector<float> newtonSchulzReference(std::vector<float> x,
                                         uint64_t rows,
                                         uint64_t cols,
                                         const NewtonSchulzOrthogonalizationOptions& options) {
    bool transposed = options.transposeTallMatrices && rows > cols;
    if (transposed) {
        x = transpose(x, rows, cols);
        std::swap(rows, cols);
    }

    float norm = 0.0f;
    for (float v : x)
        norm += v * v;
    norm = std::sqrt(norm);
    for (float& v : x)
        v /= norm + static_cast<float>(options.epsilon);

    for (uint32_t it = 0; it < options.numIterations; ++it) {
        if (rows <= cols) {
            std::vector<float> xT = transpose(x, rows, cols);
            std::vector<float> xxT = matmul(x, rows, cols, xT, cols, rows);
            std::vector<float> xxTSquared = matmul(xxT, rows, rows, xxT, rows, rows);
            std::vector<float> polynomial(xxT.size());
            for (uint64_t i = 0; i < polynomial.size(); ++i)
                polynomial[i] = static_cast<float>(options.coefficientB) * xxT[i] +
                                static_cast<float>(options.coefficientC) * xxTSquared[i];
            std::vector<float> px = matmul(polynomial, rows, rows, x, rows, cols);
            for (uint64_t i = 0; i < x.size(); ++i)
                x[i] = static_cast<float>(options.coefficientA) * x[i] + px[i];
        } else {
            std::vector<float> xT = transpose(x, rows, cols);
            std::vector<float> xTx = matmul(xT, cols, rows, x, rows, cols);
            std::vector<float> xTxSquared = matmul(xTx, cols, cols, xTx, cols, cols);
            std::vector<float> polynomial(xTx.size());
            for (uint64_t i = 0; i < polynomial.size(); ++i)
                polynomial[i] = static_cast<float>(options.coefficientB) * xTx[i] +
                                static_cast<float>(options.coefficientC) * xTxSquared[i];
            std::vector<float> xp = matmul(x, rows, cols, polynomial, cols, cols);
            for (uint64_t i = 0; i < x.size(); ++i)
                x[i] = static_cast<float>(options.coefficientA) * x[i] + xp[i];
        }
    }

    if (transposed)
        x = transpose(x, rows, cols);
    return x;
}

}  // namespace

TEST(MuonImplementation, DenseRankTwoUsesMuonMatrixPathAndUpdatesNumerically) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.03f;
    constexpr float beta = 0.8f;
    constexpr float epsilon = 1.0e-6f;
    constexpr float weightDecay = 0.01f;
    NewtonSchulzOrthogonalizationOptions options;
    options.numIterations = 3;
    options.epsilon = epsilon;

    auto fallback = std::make_shared<AdamW>(123, 0.001f, 0.9f, 0.999f, 1e-7f, 0.01f);
    Muon muon(7, alpha, beta, epsilon, weightDecay, true, options, fallback);

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3}));
    const std::vector<float> initialWeights{1.0f, -2.0f, 0.5f, 3.0f, -4.0f, 2.0f};
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    muon.compile(weights, stream);
    ASSERT_TRUE(muon.isCompiled());
    EXPECT_TRUE(muon.isUsingMuonMatrixPath());
    ASSERT_NE(muon.getSelectedOptimizer(), nullptr);
    EXPECT_TRUE(muon.getSelectedOptimizer()->hasParameter("momentum"));

    std::optional<Tensor> gradientOpt = muon.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());
    Tensor gradient = gradientOpt.value();
    const std::vector<float> rawGradient{0.5f, -1.0f, 1.5f, -2.0f, 2.5f, -3.0f};
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);

    muon.updateWeights(/*batchSize=*/2);
    stream.synchronize();

    const float invBatchLossScale = 1.0f / (2.0f * Loss::getLossScalingFactor());
    std::vector<float> g(rawGradient.size());
    std::vector<float> momentumNext(rawGradient.size());
    std::vector<float> updateSource(rawGradient.size());
    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        g[i] = rawGradient[i] * invBatchLossScale;
        momentumNext[i] = (1.0f - beta) * g[i];
        updateSource[i] = beta * momentumNext[i] + (1.0f - beta) * g[i];
    }
    std::vector<float> orthogonalUpdate = newtonSchulzReference(updateSource, 2, 3, options);

    std::vector<float> expectedWeights(initialWeights.size());
    for (uint64_t i = 0; i < initialWeights.size(); ++i)
        expectedWeights[i] = initialWeights[i] - alpha * weightDecay * initialWeights[i] - alpha * orthogonalUpdate[i];

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expectedWeights, 2e-4f, 2e-4f);

    Tensor momentum = muon.getSelectedOptimizer()->getParameter("momentum")->getStorage().value();
    expectAllClose(copyGpuFp32TensorToValues(momentum, stream), momentumNext, 1e-6f, 1e-6f);
}

TEST(MuonImplementation, DenseNonMatrixUsesAdamWFallback) {
    Stream stream(gpuPlacement);

    auto fallback = std::make_shared<AdamW>(123, 0.001f, 0.9f, 0.999f, 1e-7f, 0.01f);
    Muon muon(7, 0.02f, 0.95f, 1e-8f, 0.0f, true, NewtonSchulzOrthogonalizationOptions{}, fallback);

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {4}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, -2.0f, 3.0f, -4.0f}, stream);

    muon.compile(weights, stream);
    ASSERT_TRUE(muon.isCompiled());
    EXPECT_TRUE(muon.isUsingFallbackPath());
    ASSERT_NE(std::dynamic_pointer_cast<AdamW>(muon.getSelectedOptimizer()), nullptr);
    EXPECT_TRUE(muon.getSelectedOptimizer()->hasParameter("m"));
    EXPECT_TRUE(muon.getSelectedOptimizer()->hasParameter("v"));
    EXPECT_TRUE(muon.getWeightsGradient().has_value());
}

TEST(MuonImplementation, SparseRowsUseFallbackOptimizer) {
    Stream stream(gpuPlacement);

    auto fallback = std::make_shared<AdamW>(123, 0.001f, 0.9f, 0.999f, 1e-7f, 0.01f);
    Muon muon(7, 0.02f, 0.95f, 1e-8f, 0.0f, true, NewtonSchulzOrthogonalizationOptions{}, fallback);

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {8, 3}));
    copyValuesToGpuFp32Tensor(weights, std::vector<float>(24, 0.5f), stream);

    SparseRowGradient gradient = muon.compileSparseRows(weights, /*maxSparseRows=*/4, stream);
    ASSERT_TRUE(muon.isCompiled());
    EXPECT_TRUE(muon.isUsingFallbackPath());
    ASSERT_NE(std::dynamic_pointer_cast<AdamW>(muon.getSelectedOptimizer()), nullptr);
    EXPECT_EQ(gradient.embeddingDim, 3u);
    EXPECT_EQ(gradient.vocabularySize, 8u);
}
