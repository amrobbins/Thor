#include "DeepLearning/Api/Optimizers/NAdam.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/NAdam.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
Impl::TensorPlacement gpuPlacement(Impl::TensorPlacement::MemDevices::GPU, 0);

using DataType = Impl::DataType;

void synchronizeEvents(std::vector<Event> events) {
    for (Event& event : events)
        event.synchronize();
}

void expectHyperParameter(const std::unordered_map<std::string, float>& parameters, const std::string& name, float expected) {
    ASSERT_TRUE(parameters.contains(name)) << "Missing hyperparameter: " << name;
    EXPECT_FLOAT_EQ(parameters.at(name), expected) << "Mismatch for hyperparameter: " << name;
}

void expectAllClose(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 1e-5f, float rtol = 1e-5f) {
    ASSERT_EQ(actual.size(), expected.size());

    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        const float tol = atol + rtol * std::fabs(expected[i]);
        EXPECT_LE(diff, tol) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

void writeCpuFp32Tensor(Impl::Tensor& tensor, const std::vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::FP32);
    ASSERT_EQ(tensor.getTotalNumElements(), values.size());

    float* ptr = tensor.getMemPtr<float>();
    for (uint64_t i = 0; i < values.size(); ++i)
        ptr[i] = values[i];
}

std::vector<float> readCpuFp32Tensor(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);
    EXPECT_EQ(tensor.getDataType(), DataType::FP32);

    std::vector<float> values(tensor.getTotalNumElements());
    const float* ptr = tensor.getMemPtr<float>();
    for (uint64_t i = 0; i < values.size(); ++i)
        values[i] = ptr[i];

    return values;
}

void copyValuesToGpuFp32Tensor(Impl::Tensor& gpuTensor, const std::vector<float>& values, Stream& stream) {
    ASSERT_EQ(gpuTensor.getPlacement(), gpuPlacement);
    ASSERT_EQ(gpuTensor.getDataType(), DataType::FP32);
    ASSERT_EQ(gpuTensor.getTotalNumElements(), values.size());

    Impl::Tensor host(cpuPlacement, gpuTensor.getDescriptor());
    writeCpuFp32Tensor(host, values);

    gpuTensor.copyFromAsync(host, stream);
    stream.synchronize();
}

std::vector<float> copyGpuFp32TensorToValues(const Impl::Tensor& gpuTensor, Stream& stream) {
    EXPECT_EQ(gpuTensor.getPlacement(), gpuPlacement);
    EXPECT_EQ(gpuTensor.getDataType(), DataType::FP32);

    Impl::Tensor host = gpuTensor.clone(cpuPlacement);
    host.copyFromAsync(gpuTensor, stream);
    stream.synchronize();

    return readCpuFp32Tensor(host);
}

Impl::Tensor requireOptimizerStorage(const std::shared_ptr<Impl::NAdam>& nadam, const std::string& name) {
    if (nadam == nullptr)
        throw std::runtime_error("NAdam optimizer is null.");

    if (!nadam->hasParameter(name))
        throw std::runtime_error("NAdam optimizer is missing parameter: " + name);

    std::optional<Impl::Tensor> storage = nadam->getParameter(name)->getStorage();
    if (!storage.has_value())
        throw std::runtime_error("NAdam optimizer parameter has no storage: " + name);

    return storage.value();
}

std::shared_ptr<Impl::NAdam> stampCompileNAdam(Api::NAdam& nadam, Impl::Tensor& weights, Stream& stream) {
    std::shared_ptr<Impl::Optimizer> physicalOptimizer = nadam.stamp(nullptr);
    std::shared_ptr<Impl::NAdam> physicalNAdam = std::dynamic_pointer_cast<Impl::NAdam>(physicalOptimizer);
    if (physicalNAdam == nullptr)
        throw std::runtime_error("Api::NAdam did not stamp an Impl::NAdam.");

    nadam.compile(physicalOptimizer, weights, stream);
    stream.synchronize();

    return physicalNAdam;
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
    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Impl::Loss::getLossScalingFactor());
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

void runNAdamStep(Impl::NAdam& nadam, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    std::optional<Impl::Tensor> gradientOpt = nadam.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Impl::Tensor gradient = gradientOpt.value();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);

    nadam.updateWeights(batchSize);
    stream.synchronize();
}

}  // namespace

TEST(NAdamApi, BuilderDefaultsSettersAndArchitectureJson) {
    std::shared_ptr<Api::NAdam> nadam = Api::NAdam::Builder().build();
    ASSERT_NE(nadam, nullptr);

    EXPECT_EQ(nadam->getType(), "NAdam");
    EXPECT_FLOAT_EQ(nadam->getAlpha(), 0.002f);
    EXPECT_FLOAT_EQ(nadam->getBeta1(), 0.9f);
    EXPECT_FLOAT_EQ(nadam->getBeta2(), 0.999f);
    EXPECT_FLOAT_EQ(nadam->getEpsilon(), 1e-7f);
    EXPECT_EQ(nadam->getOriginalId(), nadam->getId());

    nadam->setAlpha(0.02f, nullptr);
    nadam->setBeta1(0.85f, nullptr);
    nadam->setBeta2(0.95f, nullptr);
    nadam->setEpsilon(1e-5f, nullptr);

    EXPECT_FLOAT_EQ(nadam->getAlpha(), 0.02f);
    EXPECT_FLOAT_EQ(nadam->getBeta1(), 0.85f);
    EXPECT_FLOAT_EQ(nadam->getBeta2(), 0.95f);
    EXPECT_FLOAT_EQ(nadam->getEpsilon(), 1e-5f);

    json j = nadam->architectureJson();
    ASSERT_EQ(j.at("optimizer_type").get<std::string>(), "nadam");
    ASSERT_EQ(j.at("version").get<std::string>(), nadam->getVersion());
    ASSERT_EQ(j.at("id").get<uint64_t>(), nadam->getId());
    EXPECT_FLOAT_EQ(j.at("t").get<float>(), 0.0f);
    EXPECT_FLOAT_EQ(j.at("alpha").get<float>(), 0.02f);
    EXPECT_FLOAT_EQ(j.at("beta1").get<float>(), 0.85f);
    EXPECT_FLOAT_EQ(j.at("beta2").get<float>(), 0.95f);
    EXPECT_FLOAT_EQ(j.at("epsilon").get<float>(), 1e-5f);
}

TEST(NAdamApi, BuilderCustomValuesStampAndCompilePhysicalNAdam) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.03f;
    constexpr float beta1 = 0.85f;
    constexpr float beta2 = 0.95f;
    constexpr float epsilon = 1e-4f;

    std::shared_ptr<Api::NAdam> nadam = Api::NAdam::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).build();
    ASSERT_NE(nadam, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f}, stream);

    std::shared_ptr<Impl::NAdam> physicalNAdam = stampCompileNAdam(*nadam, weights, stream);
    ASSERT_NE(physicalNAdam, nullptr);

    EXPECT_TRUE(physicalNAdam->isCompiled());
    EXPECT_EQ(physicalNAdam->getId(), nadam->getId());
    EXPECT_FLOAT_EQ(physicalNAdam->getAlpha(), alpha);
    EXPECT_FLOAT_EQ(physicalNAdam->getBeta1(), beta1);
    EXPECT_FLOAT_EQ(physicalNAdam->getBeta2(), beta2);
    EXPECT_FLOAT_EQ(physicalNAdam->getEpsilon(), epsilon);
    EXPECT_FLOAT_EQ(physicalNAdam->getT(), 0.0f);

    std::optional<Impl::Tensor> gradient = physicalNAdam->getWeightsGradient();
    ASSERT_TRUE(gradient.has_value());
    EXPECT_EQ(gradient.value().getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.value().getDataType(), DataType::FP32);
    EXPECT_EQ(gradient.value().getDimensions(), weights.getDimensions());

    Impl::Tensor m = requireOptimizerStorage(physicalNAdam, "m");
    Impl::Tensor v = requireOptimizerStorage(physicalNAdam, "v");

    EXPECT_EQ(m.getPlacement(), gpuPlacement);
    EXPECT_EQ(m.getDataType(), DataType::FP32);
    EXPECT_EQ(m.getDimensions(), weights.getDimensions());
    EXPECT_EQ(v.getPlacement(), gpuPlacement);
    EXPECT_EQ(v.getDataType(), DataType::FP32);
    EXPECT_EQ(v.getDimensions(), weights.getDimensions());

    std::unordered_map<std::string, float> params = physicalNAdam->getAllHyperParameters();
    ASSERT_EQ(params.size(), 5u);
    expectHyperParameter(params, "t", 0.0f);
    expectHyperParameter(params, "alpha", alpha);
    expectHyperParameter(params, "beta1", beta1);
    expectHyperParameter(params, "beta2", beta2);
    expectHyperParameter(params, "epsilon", epsilon);
}

TEST(NAdamApi, InitializeFirstStampZerosStateParameters) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::NAdam> nadam = Api::NAdam::Builder().alpha(0.03f).beta1(0.85f).beta2(0.95f).epsilon(1e-4f).build();
    ASSERT_NE(nadam, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    std::shared_ptr<Impl::NAdam> physicalNAdam = stampCompileNAdam(*nadam, weights, stream);

    Impl::Tensor m = requireOptimizerStorage(physicalNAdam, "m");
    Impl::Tensor v = requireOptimizerStorage(physicalNAdam, "v");
    copyValuesToGpuFp32Tensor(m, {1.0f, -2.0f, 3.0f, -4.0f}, stream);
    copyValuesToGpuFp32Tensor(v, {5.0f, 6.0f, 7.0f, 8.0f}, stream);

    synchronizeEvents(nadam->initialize(physicalNAdam, /*isFirstStamp=*/true, nullptr, std::nullopt));
    expectAllClose(copyGpuFp32TensorToValues(m, stream), {0.0f, 0.0f, 0.0f, 0.0f});
    expectAllClose(copyGpuFp32TensorToValues(v, stream), {0.0f, 0.0f, 0.0f, 0.0f});
}

TEST(NAdamApi, InitializeCopiesStateFromSisterOptimizer) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::NAdam> nadam = Api::NAdam::Builder().alpha(0.03f).beta1(0.85f).beta2(0.95f).epsilon(1e-4f).build();
    ASSERT_NE(nadam, nullptr);

    Impl::Tensor weights1(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    Impl::Tensor weights2(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights1, {1.0f, 2.0f, 3.0f, 4.0f}, stream);
    copyValuesToGpuFp32Tensor(weights2, {-1.0f, -2.0f, -3.0f, -4.0f}, stream);

    std::shared_ptr<Impl::NAdam> sister = stampCompileNAdam(*nadam, weights1, stream);
    std::shared_ptr<Impl::NAdam> physical = stampCompileNAdam(*nadam, weights2, stream);

    Impl::Tensor sisterM = requireOptimizerStorage(sister, "m");
    Impl::Tensor sisterV = requireOptimizerStorage(sister, "v");
    Impl::Tensor m = requireOptimizerStorage(physical, "m");
    Impl::Tensor v = requireOptimizerStorage(physical, "v");

    copyValuesToGpuFp32Tensor(sisterM, {0.1f, 0.2f, 0.3f, 0.4f}, stream);
    copyValuesToGpuFp32Tensor(sisterV, {0.5f, 0.6f, 0.7f, 0.8f}, stream);
    copyValuesToGpuFp32Tensor(m, {9.0f, 9.0f, 9.0f, 9.0f}, stream);
    copyValuesToGpuFp32Tensor(v, {8.0f, 8.0f, 8.0f, 8.0f}, stream);

    synchronizeEvents(nadam->initialize(physical, /*isFirstStamp=*/false, sister, std::nullopt));
    expectAllClose(copyGpuFp32TensorToValues(m, stream), {0.1f, 0.2f, 0.3f, 0.4f});
    expectAllClose(copyGpuFp32TensorToValues(v, stream), {0.5f, 0.6f, 0.7f, 0.8f});
}

TEST(NAdamApi, PhysicalUpdateThroughApiStampedOptimizerMatchesCpuReference) {
    Stream stream(gpuPlacement);

    const std::vector<float> initialWeights{1.0f, -2.0f, 3.0f, -4.0f};
    const std::vector<float> gradient1{8.0f, -12.0f, 0.5f, -2.0f};
    const std::vector<float> gradient2{-1.0f, 4.0f, -8.0f, 16.0f};

    constexpr uint32_t batchSize1 = 2;
    constexpr uint32_t batchSize2 = 4;
    constexpr float alpha = 0.04f;
    constexpr float beta1 = 0.8f;
    constexpr float beta2 = 0.9f;
    constexpr float epsilon = 1e-4f;

    std::shared_ptr<Api::NAdam> nadam = Api::NAdam::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).build();
    ASSERT_NE(nadam, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    std::shared_ptr<Impl::NAdam> physicalNAdam = stampCompileNAdam(*nadam, weights, stream);

    NAdamReferenceState expected;
    expected.weights = initialWeights;
    expected.m.assign(initialWeights.size(), 0.0f);
    expected.v.assign(initialWeights.size(), 0.0f);

    applyNAdamReferenceStep(expected, gradient1, batchSize1, alpha, beta1, beta2, epsilon);
    runNAdamStep(*physicalNAdam, gradient1, batchSize1, stream);

    applyNAdamReferenceStep(expected, gradient2, batchSize2, alpha, beta1, beta2, epsilon);
    runNAdamStep(*physicalNAdam, gradient2, batchSize2, stream);

    Impl::Tensor m = requireOptimizerStorage(physicalNAdam, "m");
    Impl::Tensor v = requireOptimizerStorage(physicalNAdam, "v");
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(m, stream), expected.m, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(v, stream), expected.v, 3e-5f, 3e-5f);
    EXPECT_FLOAT_EQ(physicalNAdam->getT(), 2.0f);
}
