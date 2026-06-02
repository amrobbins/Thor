#include "DeepLearning/Api/Optimizers/Adadelta.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adadelta.h"
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

Impl::Tensor requireOptimizerStorage(const std::shared_ptr<Impl::Adadelta>& adadelta, const std::string& name) {
    if (adadelta == nullptr)
        throw std::runtime_error("Adadelta optimizer is null.");

    if (!adadelta->hasParameter(name))
        throw std::runtime_error("Adadelta optimizer is missing parameter: " + name);

    std::optional<Impl::Tensor> storage = adadelta->getParameter(name)->getStorage();
    if (!storage.has_value())
        throw std::runtime_error("Adadelta optimizer parameter has no storage: " + name);

    return storage.value();
}

std::shared_ptr<Impl::Adadelta> stampCompileAdadelta(Api::Adadelta& adadelta, Impl::Tensor& weights, Stream& stream) {
    std::shared_ptr<Impl::Optimizer> physicalOptimizer = adadelta.stamp(nullptr);
    std::shared_ptr<Impl::Adadelta> physicalAdadelta = std::dynamic_pointer_cast<Impl::Adadelta>(physicalOptimizer);
    if (physicalAdadelta == nullptr)
        throw std::runtime_error("Api::Adadelta did not stamp an Impl::Adadelta.");

    adadelta.compile(physicalOptimizer, weights, stream);
    stream.synchronize();

    return physicalAdadelta;
}

struct AdadeltaReferenceState {
    std::vector<float> weights;
    std::vector<float> gradient_square_average;
    std::vector<float> update_square_average;
};

void applyAdadeltaReferenceStep(AdadeltaReferenceState& state,
                                const std::vector<float>& rawGradient,
                                uint32_t batchSize,
                                float alpha,
                                float rho,
                                float epsilon) {
    ASSERT_EQ(state.weights.size(), rawGradient.size());
    ASSERT_EQ(state.gradient_square_average.size(), rawGradient.size());
    ASSERT_EQ(state.update_square_average.size(), rawGradient.size());
    ASSERT_GT(batchSize, 0u);

    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Impl::Loss::getLossScalingFactor());
    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float g = rawGradient[i] * invBatchLossScale;
        state.gradient_square_average[i] = rho * state.gradient_square_average[i] + (1.0f - rho) * g * g;
        const float update = std::sqrt(state.update_square_average[i] + epsilon) /
                             std::sqrt(state.gradient_square_average[i] + epsilon) * g;
        state.update_square_average[i] = rho * state.update_square_average[i] + (1.0f - rho) * update * update;
        state.weights[i] -= alpha * update;
    }
}

void runAdadeltaStep(Impl::Adadelta& adadelta, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    std::optional<Impl::Tensor> gradientOpt = adadelta.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Impl::Tensor gradient = gradientOpt.value();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);

    adadelta.updateWeights(batchSize);
    stream.synchronize();
}

}  // namespace

TEST(AdadeltaApi, BuilderDefaultsSettersAndArchitectureJson) {
    std::shared_ptr<Api::Adadelta> adadelta = Api::Adadelta::Builder().build();
    ASSERT_NE(adadelta, nullptr);

    EXPECT_EQ(adadelta->getType(), "Adadelta");
    EXPECT_FLOAT_EQ(adadelta->getAlpha(), 1.0f);
    EXPECT_FLOAT_EQ(adadelta->getRho(), 0.95f);
    EXPECT_FLOAT_EQ(adadelta->getEpsilon(), 1e-7f);
    EXPECT_EQ(adadelta->getOriginalId(), adadelta->getId());

    adadelta->setAlpha(0.5f, nullptr);
    adadelta->setRho(0.9f, nullptr);
    adadelta->setEpsilon(1e-5f, nullptr);

    EXPECT_FLOAT_EQ(adadelta->getAlpha(), 0.5f);
    EXPECT_FLOAT_EQ(adadelta->getRho(), 0.9f);
    EXPECT_FLOAT_EQ(adadelta->getEpsilon(), 1e-5f);

    json j = adadelta->architectureJson();
    ASSERT_EQ(j.at("optimizer_type").get<std::string>(), "adadelta");
    ASSERT_EQ(j.at("version").get<std::string>(), adadelta->getVersion());
    ASSERT_EQ(j.at("id").get<uint64_t>(), adadelta->getId());
    EXPECT_FLOAT_EQ(j.at("alpha").get<float>(), 0.5f);
    EXPECT_FLOAT_EQ(j.at("rho").get<float>(), 0.9f);
    EXPECT_FLOAT_EQ(j.at("epsilon").get<float>(), 1e-5f);
}

TEST(AdadeltaApi, BuilderCustomValuesStampAndCompilePhysicalAdadelta) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.75f;
    constexpr float rho = 0.9f;
    constexpr float epsilon = 1e-5f;

    std::shared_ptr<Api::Adadelta> adadelta = Api::Adadelta::Builder().alpha(alpha).rho(rho).epsilon(epsilon).build();
    ASSERT_NE(adadelta, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f}, stream);

    std::shared_ptr<Impl::Adadelta> physicalAdadelta = stampCompileAdadelta(*adadelta, weights, stream);
    ASSERT_NE(physicalAdadelta, nullptr);

    EXPECT_TRUE(physicalAdadelta->isCompiled());
    EXPECT_EQ(physicalAdadelta->getId(), adadelta->getId());
    EXPECT_FLOAT_EQ(physicalAdadelta->getAlpha(), alpha);
    EXPECT_FLOAT_EQ(physicalAdadelta->getRho(), rho);
    EXPECT_FLOAT_EQ(physicalAdadelta->getEpsilon(), epsilon);

    std::optional<Impl::Tensor> gradient = physicalAdadelta->getWeightsGradient();
    ASSERT_TRUE(gradient.has_value());
    EXPECT_EQ(gradient.value().getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.value().getDataType(), DataType::FP32);
    EXPECT_EQ(gradient.value().getDimensions(), weights.getDimensions());

    Impl::Tensor gradientSquareAverage = requireOptimizerStorage(physicalAdadelta, "gradient_square_average");
    Impl::Tensor updateSquareAverage = requireOptimizerStorage(physicalAdadelta, "update_square_average");

    EXPECT_EQ(gradientSquareAverage.getPlacement(), gpuPlacement);
    EXPECT_EQ(gradientSquareAverage.getDataType(), DataType::FP32);
    EXPECT_EQ(gradientSquareAverage.getDimensions(), weights.getDimensions());
    EXPECT_EQ(updateSquareAverage.getPlacement(), gpuPlacement);
    EXPECT_EQ(updateSquareAverage.getDataType(), DataType::FP32);
    EXPECT_EQ(updateSquareAverage.getDimensions(), weights.getDimensions());

    std::unordered_map<std::string, float> params = physicalAdadelta->getAllHyperParameters();
    ASSERT_EQ(params.size(), 3u);
    expectHyperParameter(params, "alpha", alpha);
    expectHyperParameter(params, "rho", rho);
    expectHyperParameter(params, "epsilon", epsilon);
}

TEST(AdadeltaApi, InitializeFirstStampZerosStateParameters) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::Adadelta> adadelta = Api::Adadelta::Builder().alpha(0.75f).rho(0.9f).epsilon(1e-5f).build();
    ASSERT_NE(adadelta, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    std::shared_ptr<Impl::Adadelta> physicalAdadelta = stampCompileAdadelta(*adadelta, weights, stream);

    Impl::Tensor gradientSquareAverage = requireOptimizerStorage(physicalAdadelta, "gradient_square_average");
    Impl::Tensor updateSquareAverage = requireOptimizerStorage(physicalAdadelta, "update_square_average");
    copyValuesToGpuFp32Tensor(gradientSquareAverage, {1.0f, -2.0f, 3.0f, -4.0f}, stream);
    copyValuesToGpuFp32Tensor(updateSquareAverage, {-1.0f, 2.0f, -3.0f, 4.0f}, stream);

    synchronizeEvents(adadelta->initialize(physicalAdadelta, /*isFirstStamp=*/true, nullptr, std::nullopt));
    expectAllClose(copyGpuFp32TensorToValues(gradientSquareAverage, stream), {0.0f, 0.0f, 0.0f, 0.0f});
    expectAllClose(copyGpuFp32TensorToValues(updateSquareAverage, stream), {0.0f, 0.0f, 0.0f, 0.0f});
}

TEST(AdadeltaApi, InitializeCopiesStateFromSisterOptimizer) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::Adadelta> adadelta = Api::Adadelta::Builder().alpha(0.75f).rho(0.9f).epsilon(1e-5f).build();
    ASSERT_NE(adadelta, nullptr);

    Impl::Tensor weights1(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    Impl::Tensor weights2(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights1, {1.0f, 2.0f, 3.0f, 4.0f}, stream);
    copyValuesToGpuFp32Tensor(weights2, {-1.0f, -2.0f, -3.0f, -4.0f}, stream);

    std::shared_ptr<Impl::Adadelta> sister = stampCompileAdadelta(*adadelta, weights1, stream);
    std::shared_ptr<Impl::Adadelta> physical = stampCompileAdadelta(*adadelta, weights2, stream);

    Impl::Tensor sisterGradientSquareAverage = requireOptimizerStorage(sister, "gradient_square_average");
    Impl::Tensor sisterUpdateSquareAverage = requireOptimizerStorage(sister, "update_square_average");
    Impl::Tensor gradientSquareAverage = requireOptimizerStorage(physical, "gradient_square_average");
    Impl::Tensor updateSquareAverage = requireOptimizerStorage(physical, "update_square_average");

    copyValuesToGpuFp32Tensor(sisterGradientSquareAverage, {0.1f, 0.2f, 0.3f, 0.4f}, stream);
    copyValuesToGpuFp32Tensor(sisterUpdateSquareAverage, {0.5f, 0.6f, 0.7f, 0.8f}, stream);
    copyValuesToGpuFp32Tensor(gradientSquareAverage, {9.0f, 9.0f, 9.0f, 9.0f}, stream);
    copyValuesToGpuFp32Tensor(updateSquareAverage, {-9.0f, -9.0f, -9.0f, -9.0f}, stream);

    synchronizeEvents(adadelta->initialize(physical, /*isFirstStamp=*/false, sister, std::nullopt));
    expectAllClose(copyGpuFp32TensorToValues(gradientSquareAverage, stream), {0.1f, 0.2f, 0.3f, 0.4f});
    expectAllClose(copyGpuFp32TensorToValues(updateSquareAverage, stream), {0.5f, 0.6f, 0.7f, 0.8f});
}

TEST(AdadeltaApi, PhysicalUpdateThroughApiStampedOptimizerMatchesCpuReference) {
    Stream stream(gpuPlacement);

    const std::vector<float> initialWeights{1.0f, -2.0f, 3.0f, -4.0f};
    const std::vector<float> gradient1{8.0f, -12.0f, 0.5f, -2.0f};
    const std::vector<float> gradient2{-1.0f, 4.0f, -8.0f, 16.0f};

    constexpr uint32_t batchSize1 = 2;
    constexpr uint32_t batchSize2 = 4;
    constexpr float alpha = 0.5f;
    constexpr float rho = 0.85f;
    constexpr float epsilon = 1e-5f;

    std::shared_ptr<Api::Adadelta> adadelta = Api::Adadelta::Builder().alpha(alpha).rho(rho).epsilon(epsilon).build();
    ASSERT_NE(adadelta, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    std::shared_ptr<Impl::Adadelta> physicalAdadelta = stampCompileAdadelta(*adadelta, weights, stream);

    AdadeltaReferenceState expected;
    expected.weights = initialWeights;
    expected.gradient_square_average.assign(initialWeights.size(), 0.0f);
    expected.update_square_average.assign(initialWeights.size(), 0.0f);

    applyAdadeltaReferenceStep(expected, gradient1, batchSize1, alpha, rho, epsilon);
    runAdadeltaStep(*physicalAdadelta, gradient1, batchSize1, stream);

    applyAdadeltaReferenceStep(expected, gradient2, batchSize2, alpha, rho, epsilon);
    runAdadeltaStep(*physicalAdadelta, gradient2, batchSize2, stream);

    Impl::Tensor gradientSquareAverage = requireOptimizerStorage(physicalAdadelta, "gradient_square_average");
    Impl::Tensor updateSquareAverage = requireOptimizerStorage(physicalAdadelta, "update_square_average");
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 4e-5f, 4e-5f);
    expectAllClose(copyGpuFp32TensorToValues(gradientSquareAverage, stream), expected.gradient_square_average, 4e-5f, 4e-5f);
    expectAllClose(copyGpuFp32TensorToValues(updateSquareAverage, stream), expected.update_square_average, 4e-5f, 4e-5f);
}
