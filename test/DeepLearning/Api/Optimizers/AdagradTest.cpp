#include "DeepLearning/Api/Optimizers/Adagrad.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adagrad.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

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

Impl::Tensor requireOptimizerStorage(const std::shared_ptr<Impl::Adagrad>& adagrad, const std::string& name) {
    if (adagrad == nullptr)
        throw std::runtime_error("Adagrad optimizer is null.");

    if (!adagrad->hasParameter(name))
        throw std::runtime_error("Adagrad optimizer is missing parameter: " + name);

    std::optional<Impl::Tensor> storage = adagrad->getParameter(name)->getStorage();
    if (!storage.has_value())
        throw std::runtime_error("Adagrad optimizer parameter has no storage: " + name);

    return storage.value();
}

std::shared_ptr<Impl::Adagrad> stampCompileAdagrad(Api::Adagrad& adagrad, Impl::Tensor& weights, Stream& stream) {
    std::shared_ptr<Impl::Optimizer> physicalOptimizer = adagrad.stamp(nullptr);
    std::shared_ptr<Impl::Adagrad> physicalAdagrad = std::dynamic_pointer_cast<Impl::Adagrad>(physicalOptimizer);
    if (physicalAdagrad == nullptr)
        throw std::runtime_error("Api::Adagrad did not stamp an Impl::Adagrad.");

    adagrad.compile(physicalOptimizer, weights, stream);
    stream.synchronize();

    return physicalAdagrad;
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

    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Impl::Loss::getLossScalingFactor());
    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float g = rawGradient[i] * invBatchLossScale;
        state.accumulator[i] += g * g;
        state.weights[i] -= alpha * g / (std::sqrt(state.accumulator[i]) + epsilon);
    }
}

void runAdagradStep(Impl::Adagrad& adagrad, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    std::optional<Impl::Tensor> gradientOpt = adagrad.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Impl::Tensor gradient = gradientOpt.value();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);

    adagrad.updateWeights(batchSize);
    stream.synchronize();
}

}  // namespace

TEST(AdagradApi, BuilderDefaultsSettersAndArchitectureJson) {
    std::shared_ptr<Api::Adagrad> adagrad = Api::Adagrad::Builder().build();
    ASSERT_NE(adagrad, nullptr);

    EXPECT_EQ(adagrad->getType(), "Adagrad");
    EXPECT_FLOAT_EQ(adagrad->getAlpha(), 0.01f);
    EXPECT_FLOAT_EQ(adagrad->getEpsilon(), 1e-7f);
    EXPECT_EQ(adagrad->getOriginalId(), adagrad->getId());

    adagrad->setAlpha(0.02f, nullptr);
    adagrad->setEpsilon(1e-5f, nullptr);

    EXPECT_FLOAT_EQ(adagrad->getAlpha(), 0.02f);
    EXPECT_FLOAT_EQ(adagrad->getEpsilon(), 1e-5f);

    json j = adagrad->architectureJson();
    ASSERT_EQ(j.at("optimizer_type").get<std::string>(), "adagrad");
    ASSERT_EQ(j.at("version").get<std::string>(), adagrad->getVersion());
    ASSERT_EQ(j.at("id").get<uint64_t>(), adagrad->getId());
    EXPECT_FLOAT_EQ(j.at("alpha").get<float>(), 0.02f);
    EXPECT_FLOAT_EQ(j.at("epsilon").get<float>(), 1e-5f);
}

TEST(AdagradApi, BuilderCustomValuesStampAndCompilePhysicalAdagrad) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.03f;
    constexpr float epsilon = 1e-4f;

    std::shared_ptr<Api::Adagrad> adagrad = Api::Adagrad::Builder().alpha(alpha).epsilon(epsilon).build();
    ASSERT_NE(adagrad, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f}, stream);

    std::shared_ptr<Impl::Adagrad> physicalAdagrad = stampCompileAdagrad(*adagrad, weights, stream);
    ASSERT_NE(physicalAdagrad, nullptr);

    EXPECT_TRUE(physicalAdagrad->isCompiled());
    EXPECT_EQ(physicalAdagrad->getId(), adagrad->getId());
    EXPECT_FLOAT_EQ(physicalAdagrad->getAlpha(), alpha);
    EXPECT_FLOAT_EQ(physicalAdagrad->getEpsilon(), epsilon);

    std::optional<Impl::Tensor> gradient = physicalAdagrad->getWeightsGradient();
    ASSERT_TRUE(gradient.has_value());
    EXPECT_EQ(gradient.value().getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.value().getDataType(), DataType::FP32);
    EXPECT_EQ(gradient.value().getDimensions(), weights.getDimensions());

    Impl::Tensor accumulator = requireOptimizerStorage(physicalAdagrad, "accumulator");

    EXPECT_EQ(accumulator.getPlacement(), gpuPlacement);
    EXPECT_EQ(accumulator.getDataType(), DataType::FP32);
    EXPECT_EQ(accumulator.getDimensions(), weights.getDimensions());

    std::unordered_map<std::string, float> params = physicalAdagrad->getAllHyperParameters();
    ASSERT_EQ(params.size(), 2u);
    expectHyperParameter(params, "alpha", alpha);
    expectHyperParameter(params, "epsilon", epsilon);
}

TEST(AdagradApi, InitializeFirstStampZerosAccumulatorParameter) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::Adagrad> adagrad = Api::Adagrad::Builder().alpha(0.03f).epsilon(1e-4f).build();
    ASSERT_NE(adagrad, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    std::shared_ptr<Impl::Adagrad> physicalAdagrad = stampCompileAdagrad(*adagrad, weights, stream);

    Impl::Tensor accumulator = requireOptimizerStorage(physicalAdagrad, "accumulator");
    copyValuesToGpuFp32Tensor(accumulator, {1.0f, -2.0f, 3.0f, -4.0f}, stream);

    synchronizeEvents(adagrad->initialize(physicalAdagrad, /*isFirstStamp=*/true, nullptr, std::nullopt));
    expectAllClose(copyGpuFp32TensorToValues(accumulator, stream), {0.0f, 0.0f, 0.0f, 0.0f});
}

TEST(AdagradApi, InitializeCopiesAccumulatorFromSisterOptimizer) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::Adagrad> adagrad = Api::Adagrad::Builder().alpha(0.03f).epsilon(1e-4f).build();
    ASSERT_NE(adagrad, nullptr);

    Impl::Tensor weights1(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    Impl::Tensor weights2(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights1, {1.0f, 2.0f, 3.0f, 4.0f}, stream);
    copyValuesToGpuFp32Tensor(weights2, {-1.0f, -2.0f, -3.0f, -4.0f}, stream);

    std::shared_ptr<Impl::Adagrad> sister = stampCompileAdagrad(*adagrad, weights1, stream);
    std::shared_ptr<Impl::Adagrad> physical = stampCompileAdagrad(*adagrad, weights2, stream);

    Impl::Tensor sisterAccumulator = requireOptimizerStorage(sister, "accumulator");
    Impl::Tensor accumulator = requireOptimizerStorage(physical, "accumulator");

    copyValuesToGpuFp32Tensor(sisterAccumulator, {0.1f, 0.2f, 0.3f, 0.4f}, stream);
    copyValuesToGpuFp32Tensor(accumulator, {9.0f, 9.0f, 9.0f, 9.0f}, stream);

    synchronizeEvents(adagrad->initialize(physical, /*isFirstStamp=*/false, sister, std::nullopt));
    expectAllClose(copyGpuFp32TensorToValues(accumulator, stream), {0.1f, 0.2f, 0.3f, 0.4f});
}

TEST(AdagradApi, PhysicalUpdateThroughApiStampedOptimizerMatchesCpuReference) {
    Stream stream(gpuPlacement);

    const std::vector<float> initialWeights{1.0f, -2.0f, 3.0f, -4.0f};
    const std::vector<float> gradient1{8.0f, -12.0f, 0.5f, -2.0f};
    const std::vector<float> gradient2{-1.0f, 4.0f, -8.0f, 16.0f};

    constexpr uint32_t batchSize1 = 2;
    constexpr uint32_t batchSize2 = 4;
    constexpr float alpha = 0.04f;
    constexpr float epsilon = 1e-4f;

    std::shared_ptr<Api::Adagrad> adagrad = Api::Adagrad::Builder().alpha(alpha).epsilon(epsilon).build();
    ASSERT_NE(adagrad, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    std::shared_ptr<Impl::Adagrad> physicalAdagrad = stampCompileAdagrad(*adagrad, weights, stream);

    AdagradReferenceState expected;
    expected.weights = initialWeights;
    expected.accumulator.assign(initialWeights.size(), 0.0f);

    applyAdagradReferenceStep(expected, gradient1, batchSize1, alpha, epsilon);
    runAdagradStep(*physicalAdagrad, gradient1, batchSize1, stream);

    applyAdagradReferenceStep(expected, gradient2, batchSize2, alpha, epsilon);
    runAdagradStep(*physicalAdagrad, gradient2, batchSize2, stream);

    Impl::Tensor accumulator = requireOptimizerStorage(physicalAdagrad, "accumulator");
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(accumulator, stream), expected.accumulator, 3e-5f, 3e-5f);
}

