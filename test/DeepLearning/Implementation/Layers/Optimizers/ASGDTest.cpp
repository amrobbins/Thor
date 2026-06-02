#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/ASGD.h"
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

Tensor requireOptimizerStorage(const ASGD& asgd, const std::string& name) {
    if (!asgd.hasParameter(name))
        throw std::runtime_error("ASGD optimizer missing parameter: " + name);
    std::optional<Tensor> storage = asgd.getParameter(name)->getStorage();
    if (!storage.has_value())
        throw std::runtime_error("ASGD optimizer parameter has no storage: " + name);
    return storage.value();
}

struct ASGDReferenceState {
    std::vector<float> weights;
    std::vector<float> averagedWeights;
    float t;
};

void applyASGDReferenceStep(ASGDReferenceState& state,
                            const std::vector<float>& rawGradient,
                            uint32_t batchSize,
                            float alpha,
                            float lambd,
                            float power,
                            float t0,
                            float weightDecay) {
    ASSERT_EQ(state.weights.size(), rawGradient.size());
    ASSERT_EQ(state.averagedWeights.size(), rawGradient.size());
    ASSERT_GT(batchSize, 0u);

    state.t += 1.0f;
    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());
    const double eta64 = static_cast<double>(alpha) /
                         std::pow(1.0 + static_cast<double>(lambd) * static_cast<double>(alpha) * state.t,
                                  static_cast<double>(power));
    const float eta = static_cast<float>(eta64);
    const float lambdEta = static_cast<float>(static_cast<double>(lambd) * eta64);
    const float averagingScale = state.t >= t0 ? 1.0f / (state.t - t0 + 1.0f) : 0.0f;

    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float g = rawGradient[i] * invBatchLossScale;
        const float update = g + weightDecay * state.weights[i];
        const float wNext = (1.0f - lambdEta) * state.weights[i] - eta * update;
        state.weights[i] = wNext;
        state.averagedWeights[i] += averagingScale * (wNext - state.averagedWeights[i]);
    }
}

void runASGDStep(ASGD& asgd, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    std::optional<Tensor> gradientOpt = asgd.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());
    Tensor gradient = gradientOpt.value();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);
    asgd.updateWeights(batchSize);
    stream.synchronize();
}

}  // namespace

TEST(ASGDTest, ConstructorGettersSettersAndHyperParameters) {
    constexpr uint64_t id = 53;
    ASGD asgd(id, 0.01f, 1e-4f, 0.75f, 1000.0f, 0.02f);

    EXPECT_EQ(asgd.getId(), id);
    EXPECT_FLOAT_EQ(asgd.getAlpha(), 0.01f);
    EXPECT_FLOAT_EQ(asgd.getLambd(), 1e-4f);
    EXPECT_FLOAT_EQ(asgd.getPower(), 0.75f);
    EXPECT_FLOAT_EQ(asgd.getT0(), 1000.0f);
    EXPECT_FLOAT_EQ(asgd.getWeightDecay(), 0.02f);
    EXPECT_FLOAT_EQ(asgd.getT(), 0.0f);

    asgd.setAlpha(0.02f);
    asgd.setLambd(2e-4f);
    asgd.setPower(0.5f);
    asgd.setT0(10.0f);
    asgd.setWeightDecay(0.03f);
    asgd.setT(7.0f);

    EXPECT_FLOAT_EQ(asgd.getAlpha(), 0.02f);
    EXPECT_FLOAT_EQ(asgd.getLambd(), 2e-4f);
    EXPECT_FLOAT_EQ(asgd.getPower(), 0.5f);
    EXPECT_FLOAT_EQ(asgd.getT0(), 10.0f);
    EXPECT_FLOAT_EQ(asgd.getWeightDecay(), 0.03f);
    EXPECT_FLOAT_EQ(asgd.getT(), 7.0f);

    std::unordered_map<std::string, float> params = asgd.getAllHyperParameters();
    ASSERT_EQ(params.size(), 6u);
    expectMapHasValue(params, "t", 7.0f);
    expectMapHasValue(params, "alpha", 0.02f);
    expectMapHasValue(params, "lambd", 2e-4f);
    expectMapHasValue(params, "power", 0.5f);
    expectMapHasValue(params, "t0", 10.0f);
    expectMapHasValue(params, "weightDecay", 0.03f);
}

TEST(ASGDTest, DenseUpdateAndAveragedWeightsMatchReference) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.2f;
    constexpr float lambd = 0.01f;
    constexpr float power = 0.5f;
    constexpr float t0 = 2.0f;
    constexpr float weightDecay = 0.03f;

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3}));
    const std::vector<float> initialWeights{1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    ASGD asgd(55, alpha, lambd, power, t0, weightDecay);
    asgd.compile(weights, stream);
    stream.synchronize();

    Tensor averagedWeights = requireOptimizerStorage(asgd, "averaged_weights");
    copyValuesToGpuFp32Tensor(averagedWeights, std::vector<float>(initialWeights.size(), 0.0f), stream);

    ASGDReferenceState reference{initialWeights, std::vector<float>(initialWeights.size(), 0.0f), 0.0f};
    const std::vector<float> gradient1{0.5f, -0.25f, 1.0f, -1.5f, 0.75f, -0.5f};
    const std::vector<float> gradient2{-0.1f, 0.2f, -0.3f, 0.4f, -0.5f, 0.6f};
    const std::vector<float> gradient3{0.05f, -0.15f, 0.25f, -0.35f, 0.45f, -0.55f};

    applyASGDReferenceStep(reference, gradient1, 2, alpha, lambd, power, t0, weightDecay);
    runASGDStep(asgd, gradient1, 2, stream);
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), reference.weights, 2e-5f, 2e-5f);
    expectAllClose(copyGpuFp32TensorToValues(averagedWeights, stream), reference.averagedWeights, 2e-5f, 2e-5f);

    applyASGDReferenceStep(reference, gradient2, 4, alpha, lambd, power, t0, weightDecay);
    runASGDStep(asgd, gradient2, 4, stream);
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), reference.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(averagedWeights, stream), reference.averagedWeights, 3e-5f, 3e-5f);

    applyASGDReferenceStep(reference, gradient3, 1, alpha, lambd, power, t0, weightDecay);
    runASGDStep(asgd, gradient3, 1, stream);
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), reference.weights, 4e-5f, 4e-5f);
    expectAllClose(copyGpuFp32TensorToValues(averagedWeights, stream), reference.averagedWeights, 4e-5f, 4e-5f);

    EXPECT_FLOAT_EQ(asgd.getT(), 3.0f);
}

TEST(ASGDTest, SparseRowsAreRejectedBecauseAveragedWeightsAreFullTensorState) {
    Stream stream(gpuPlacement);
    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {4, 3}));
    ASGD asgd(57, 0.01f, 1e-4f, 0.75f, 1000.0f, 0.0f);

    EXPECT_FALSE(asgd.supportsSparseRowGradients());
    EXPECT_FALSE(asgd.supportsSparseRowUpdateFusion());
    EXPECT_THROW(asgd.compileSparseRows(weights, 2, stream), std::runtime_error);
}
