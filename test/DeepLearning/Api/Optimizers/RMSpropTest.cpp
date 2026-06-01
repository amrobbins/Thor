#include "DeepLearning/Api/Optimizers/RMSprop.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/RMSprop.h"
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

Impl::Tensor requireOptimizerStorage(const std::shared_ptr<Impl::RMSprop>& rmsprop, const std::string& name) {
    if (rmsprop == nullptr)
        throw std::runtime_error("RMSprop optimizer is null.");

    if (!rmsprop->hasParameter(name))
        throw std::runtime_error("RMSprop optimizer is missing parameter: " + name);

    std::optional<Impl::Tensor> storage = rmsprop->getParameter(name)->getStorage();
    if (!storage.has_value())
        throw std::runtime_error("RMSprop optimizer parameter has no storage: " + name);

    return storage.value();
}

std::shared_ptr<Impl::RMSprop> stampCompileRMSprop(Api::RMSprop& rmsprop, Impl::Tensor& weights, Stream& stream) {
    std::shared_ptr<Impl::Optimizer> physicalOptimizer = rmsprop.stamp(nullptr);
    std::shared_ptr<Impl::RMSprop> physicalRMSprop = std::dynamic_pointer_cast<Impl::RMSprop>(physicalOptimizer);
    if (physicalRMSprop == nullptr)
        throw std::runtime_error("Api::RMSprop did not stamp an Impl::RMSprop.");

    rmsprop.compile(physicalOptimizer, weights, stream);
    stream.synchronize();

    return physicalRMSprop;
}

struct RMSpropReferenceState {
    std::vector<float> weights;
    std::vector<float> square_average;
};

void applyRMSpropReferenceStep(RMSpropReferenceState& state,
                               const std::vector<float>& rawGradient,
                               uint32_t batchSize,
                               float alpha,
                               float rho,
                               float epsilon) {
    ASSERT_EQ(state.weights.size(), rawGradient.size());
    ASSERT_EQ(state.square_average.size(), rawGradient.size());
    ASSERT_GT(batchSize, 0u);

    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Impl::Loss::getLossScalingFactor());
    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float g = rawGradient[i] * invBatchLossScale;
        state.square_average[i] = rho * state.square_average[i] + (1.0f - rho) * g * g;
        state.weights[i] -= alpha * g / (std::sqrt(state.square_average[i]) + epsilon);
    }
}

void runRMSpropStep(Impl::RMSprop& rmsprop, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    std::optional<Impl::Tensor> gradientOpt = rmsprop.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Impl::Tensor gradient = gradientOpt.value();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);

    rmsprop.updateWeights(batchSize);
    stream.synchronize();
}

}  // namespace

TEST(RMSpropApi, BuilderDefaultsSettersAndArchitectureJson) {
    std::shared_ptr<Api::RMSprop> rmsprop = Api::RMSprop::Builder().build();
    ASSERT_NE(rmsprop, nullptr);

    EXPECT_EQ(rmsprop->getType(), "RMSprop");
    EXPECT_FLOAT_EQ(rmsprop->getAlpha(), 0.001f);
    EXPECT_FLOAT_EQ(rmsprop->getRho(), 0.9f);
    EXPECT_FLOAT_EQ(rmsprop->getEpsilon(), 1e-7f);
    EXPECT_EQ(rmsprop->getOriginalId(), rmsprop->getId());

    rmsprop->setAlpha(0.02f, nullptr);
    rmsprop->setRho(0.95f, nullptr);
    rmsprop->setEpsilon(1e-5f, nullptr);

    EXPECT_FLOAT_EQ(rmsprop->getAlpha(), 0.02f);
    EXPECT_FLOAT_EQ(rmsprop->getRho(), 0.95f);
    EXPECT_FLOAT_EQ(rmsprop->getEpsilon(), 1e-5f);

    json j = rmsprop->architectureJson();
    ASSERT_EQ(j.at("optimizer_type").get<std::string>(), "rmsprop");
    ASSERT_EQ(j.at("version").get<std::string>(), rmsprop->getVersion());
    ASSERT_EQ(j.at("id").get<uint64_t>(), rmsprop->getId());
    EXPECT_FLOAT_EQ(j.at("alpha").get<float>(), 0.02f);
    EXPECT_FLOAT_EQ(j.at("rho").get<float>(), 0.95f);
    EXPECT_FLOAT_EQ(j.at("epsilon").get<float>(), 1e-5f);
}

TEST(RMSpropApi, BuilderCustomValuesStampAndCompilePhysicalRMSprop) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.03f;
    constexpr float rho = 0.95f;
    constexpr float epsilon = 1e-4f;

    std::shared_ptr<Api::RMSprop> rmsprop = Api::RMSprop::Builder().alpha(alpha).rho(rho).epsilon(epsilon).build();
    ASSERT_NE(rmsprop, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f}, stream);

    std::shared_ptr<Impl::RMSprop> physicalRMSprop = stampCompileRMSprop(*rmsprop, weights, stream);
    ASSERT_NE(physicalRMSprop, nullptr);

    EXPECT_TRUE(physicalRMSprop->isCompiled());
    EXPECT_EQ(physicalRMSprop->getId(), rmsprop->getId());
    EXPECT_FLOAT_EQ(physicalRMSprop->getAlpha(), alpha);
    EXPECT_FLOAT_EQ(physicalRMSprop->getRho(), rho);
    EXPECT_FLOAT_EQ(physicalRMSprop->getEpsilon(), epsilon);

    std::optional<Impl::Tensor> gradient = physicalRMSprop->getWeightsGradient();
    ASSERT_TRUE(gradient.has_value());
    EXPECT_EQ(gradient.value().getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.value().getDataType(), DataType::FP32);
    EXPECT_EQ(gradient.value().getDimensions(), weights.getDimensions());

    Impl::Tensor square_average = requireOptimizerStorage(physicalRMSprop, "square_average");

    EXPECT_EQ(square_average.getPlacement(), gpuPlacement);
    EXPECT_EQ(square_average.getDataType(), DataType::FP32);
    EXPECT_EQ(square_average.getDimensions(), weights.getDimensions());

    std::unordered_map<std::string, float> params = physicalRMSprop->getAllHyperParameters();
    ASSERT_EQ(params.size(), 3u);
    expectHyperParameter(params, "alpha", alpha);
    expectHyperParameter(params, "rho", rho);
    expectHyperParameter(params, "epsilon", epsilon);
}

TEST(RMSpropApi, InitializeFirstStampZerosSquareAverageParameter) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::RMSprop> rmsprop = Api::RMSprop::Builder().alpha(0.03f).rho(0.95f).epsilon(1e-4f).build();
    ASSERT_NE(rmsprop, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    std::shared_ptr<Impl::RMSprop> physicalRMSprop = stampCompileRMSprop(*rmsprop, weights, stream);

    Impl::Tensor square_average = requireOptimizerStorage(physicalRMSprop, "square_average");
    copyValuesToGpuFp32Tensor(square_average, {1.0f, -2.0f, 3.0f, -4.0f}, stream);

    synchronizeEvents(rmsprop->initialize(physicalRMSprop, /*isFirstStamp=*/true, nullptr, std::nullopt));
    expectAllClose(copyGpuFp32TensorToValues(square_average, stream), {0.0f, 0.0f, 0.0f, 0.0f});
}

TEST(RMSpropApi, InitializeCopiesSquareAverageFromSisterOptimizer) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::RMSprop> rmsprop = Api::RMSprop::Builder().alpha(0.03f).rho(0.95f).epsilon(1e-4f).build();
    ASSERT_NE(rmsprop, nullptr);

    Impl::Tensor weights1(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    Impl::Tensor weights2(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights1, {1.0f, 2.0f, 3.0f, 4.0f}, stream);
    copyValuesToGpuFp32Tensor(weights2, {-1.0f, -2.0f, -3.0f, -4.0f}, stream);

    std::shared_ptr<Impl::RMSprop> sister = stampCompileRMSprop(*rmsprop, weights1, stream);
    std::shared_ptr<Impl::RMSprop> physical = stampCompileRMSprop(*rmsprop, weights2, stream);

    Impl::Tensor sisterSquareAverage = requireOptimizerStorage(sister, "square_average");
    Impl::Tensor square_average = requireOptimizerStorage(physical, "square_average");

    copyValuesToGpuFp32Tensor(sisterSquareAverage, {0.1f, 0.2f, 0.3f, 0.4f}, stream);
    copyValuesToGpuFp32Tensor(square_average, {9.0f, 9.0f, 9.0f, 9.0f}, stream);

    synchronizeEvents(rmsprop->initialize(physical, /*isFirstStamp=*/false, sister, std::nullopt));
    expectAllClose(copyGpuFp32TensorToValues(square_average, stream), {0.1f, 0.2f, 0.3f, 0.4f});
}

TEST(RMSpropApi, PhysicalUpdateThroughApiStampedOptimizerMatchesCpuReference) {
    Stream stream(gpuPlacement);

    const std::vector<float> initialWeights{1.0f, -2.0f, 3.0f, -4.0f};
    const std::vector<float> gradient1{8.0f, -12.0f, 0.5f, -2.0f};
    const std::vector<float> gradient2{-1.0f, 4.0f, -8.0f, 16.0f};

    constexpr uint32_t batchSize1 = 2;
    constexpr uint32_t batchSize2 = 4;
    constexpr float alpha = 0.04f;
    constexpr float rho = 0.8f;
    constexpr float epsilon = 1e-4f;

    std::shared_ptr<Api::RMSprop> rmsprop = Api::RMSprop::Builder().alpha(alpha).rho(rho).epsilon(epsilon).build();
    ASSERT_NE(rmsprop, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    std::shared_ptr<Impl::RMSprop> physicalRMSprop = stampCompileRMSprop(*rmsprop, weights, stream);

    RMSpropReferenceState expected;
    expected.weights = initialWeights;
    expected.square_average.assign(initialWeights.size(), 0.0f);

    applyRMSpropReferenceStep(expected, gradient1, batchSize1, alpha, rho, epsilon);
    runRMSpropStep(*physicalRMSprop, gradient1, batchSize1, stream);

    applyRMSpropReferenceStep(expected, gradient2, batchSize2, alpha, rho, epsilon);
    runRMSpropStep(*physicalRMSprop, gradient2, batchSize2, stream);

    Impl::Tensor square_average = requireOptimizerStorage(physicalRMSprop, "square_average");
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(square_average, stream), expected.square_average, 3e-5f, 3e-5f);
}

