#include "DeepLearning/Api/Optimizers/Adamax.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adamax.h"
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

Impl::Tensor requireOptimizerStorage(const std::shared_ptr<Impl::Adamax>& adamax, const std::string& name) {
    if (adamax == nullptr)
        throw std::runtime_error("Adamax optimizer is null.");

    if (!adamax->hasParameter(name))
        throw std::runtime_error("Adamax optimizer is missing parameter: " + name);

    std::optional<Impl::Tensor> storage = adamax->getParameter(name)->getStorage();
    if (!storage.has_value())
        throw std::runtime_error("Adamax optimizer parameter has no storage: " + name);

    return storage.value();
}

std::shared_ptr<Impl::Adamax> stampCompileAdamax(Api::Adamax& adamax, Impl::Tensor& weights, Stream& stream) {
    std::shared_ptr<Impl::Optimizer> physicalOptimizer = adamax.stamp(nullptr);
    std::shared_ptr<Impl::Adamax> physicalAdamax = std::dynamic_pointer_cast<Impl::Adamax>(physicalOptimizer);
    if (physicalAdamax == nullptr)
        throw std::runtime_error("Api::Adamax did not stamp an Impl::Adamax.");

    adamax.compile(physicalOptimizer, weights, stream);
    stream.synchronize();

    return physicalAdamax;
}

struct AdamaxReferenceState {
    std::vector<float> weights;
    std::vector<float> m;
    std::vector<float> u;
    float t = 0.0f;
};

void applyAdamaxReferenceStep(AdamaxReferenceState& state,
                              const std::vector<float>& rawGradient,
                              uint32_t batchSize,
                              float alpha,
                              float beta1,
                              float beta2,
                              float epsilon) {
    ASSERT_EQ(state.weights.size(), rawGradient.size());
    ASSERT_EQ(state.m.size(), rawGradient.size());
    ASSERT_EQ(state.u.size(), rawGradient.size());
    ASSERT_GT(batchSize, 0u);

    state.t += 1.0f;
    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Impl::Loss::getLossScalingFactor());
    const double alphaT64 = static_cast<double>(alpha) / (1.0 - std::pow(static_cast<double>(beta1), state.t));
    const float alphaT = static_cast<float>(alphaT64);

    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float g = rawGradient[i] * invBatchLossScale;
        state.m[i] = beta1 * state.m[i] + (1.0f - beta1) * g;
        state.u[i] = std::max(beta2 * state.u[i], std::fabs(g));
        state.weights[i] -= alphaT * state.m[i] / (state.u[i] + epsilon);
    }
}

void runAdamaxStep(Impl::Adamax& adamax, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    std::optional<Impl::Tensor> gradientOpt = adamax.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Impl::Tensor gradient = gradientOpt.value();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);

    adamax.updateWeights(batchSize);
    stream.synchronize();
}

}  // namespace

TEST(AdamaxApi, BuilderDefaultsSettersAndArchitectureJson) {
    std::shared_ptr<Api::Adamax> adamax = Api::Adamax::Builder().build();
    ASSERT_NE(adamax, nullptr);

    EXPECT_EQ(adamax->getType(), "Adamax");
    EXPECT_FLOAT_EQ(adamax->getAlpha(), 0.002f);
    EXPECT_FLOAT_EQ(adamax->getBeta1(), 0.9f);
    EXPECT_FLOAT_EQ(adamax->getBeta2(), 0.999f);
    EXPECT_FLOAT_EQ(adamax->getEpsilon(), 1e-7f);
    EXPECT_EQ(adamax->getOriginalId(), adamax->getId());

    adamax->setAlpha(0.02f, nullptr);
    adamax->setBeta1(0.85f, nullptr);
    adamax->setBeta2(0.95f, nullptr);
    adamax->setEpsilon(1e-5f, nullptr);

    EXPECT_FLOAT_EQ(adamax->getAlpha(), 0.02f);
    EXPECT_FLOAT_EQ(adamax->getBeta1(), 0.85f);
    EXPECT_FLOAT_EQ(adamax->getBeta2(), 0.95f);
    EXPECT_FLOAT_EQ(adamax->getEpsilon(), 1e-5f);

    json j = adamax->architectureJson();
    ASSERT_EQ(j.at("optimizer_type").get<std::string>(), "adamax");
    ASSERT_EQ(j.at("version").get<std::string>(), adamax->getVersion());
    ASSERT_EQ(j.at("id").get<uint64_t>(), adamax->getId());
    EXPECT_FLOAT_EQ(j.at("t").get<float>(), 0.0f);
    EXPECT_FLOAT_EQ(j.at("alpha").get<float>(), 0.02f);
    EXPECT_FLOAT_EQ(j.at("beta1").get<float>(), 0.85f);
    EXPECT_FLOAT_EQ(j.at("beta2").get<float>(), 0.95f);
    EXPECT_FLOAT_EQ(j.at("epsilon").get<float>(), 1e-5f);
}

TEST(AdamaxApi, BuilderCustomValuesStampAndCompilePhysicalAdamax) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.03f;
    constexpr float beta1 = 0.85f;
    constexpr float beta2 = 0.95f;
    constexpr float epsilon = 1e-4f;

    std::shared_ptr<Api::Adamax> adamax = Api::Adamax::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).build();
    ASSERT_NE(adamax, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f}, stream);

    std::shared_ptr<Impl::Adamax> physicalAdamax = stampCompileAdamax(*adamax, weights, stream);
    ASSERT_NE(physicalAdamax, nullptr);

    EXPECT_TRUE(physicalAdamax->isCompiled());
    EXPECT_EQ(physicalAdamax->getId(), adamax->getId());
    EXPECT_FLOAT_EQ(physicalAdamax->getAlpha(), alpha);
    EXPECT_FLOAT_EQ(physicalAdamax->getBeta1(), beta1);
    EXPECT_FLOAT_EQ(physicalAdamax->getBeta2(), beta2);
    EXPECT_FLOAT_EQ(physicalAdamax->getEpsilon(), epsilon);
    EXPECT_FLOAT_EQ(physicalAdamax->getT(), 0.0f);

    std::optional<Impl::Tensor> gradient = physicalAdamax->getWeightsGradient();
    ASSERT_TRUE(gradient.has_value());
    EXPECT_EQ(gradient.value().getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.value().getDataType(), DataType::FP32);
    EXPECT_EQ(gradient.value().getDimensions(), weights.getDimensions());

    Impl::Tensor m = requireOptimizerStorage(physicalAdamax, "m");
    Impl::Tensor u = requireOptimizerStorage(physicalAdamax, "u");

    EXPECT_EQ(m.getPlacement(), gpuPlacement);
    EXPECT_EQ(m.getDataType(), DataType::FP32);
    EXPECT_EQ(m.getDimensions(), weights.getDimensions());
    EXPECT_EQ(u.getPlacement(), gpuPlacement);
    EXPECT_EQ(u.getDataType(), DataType::FP32);
    EXPECT_EQ(u.getDimensions(), weights.getDimensions());

    std::unordered_map<std::string, float> params = physicalAdamax->getAllHyperParameters();
    ASSERT_EQ(params.size(), 5u);
    expectHyperParameter(params, "t", 0.0f);
    expectHyperParameter(params, "alpha", alpha);
    expectHyperParameter(params, "beta1", beta1);
    expectHyperParameter(params, "beta2", beta2);
    expectHyperParameter(params, "epsilon", epsilon);
}

TEST(AdamaxApi, InitializeFirstStampZerosStateParameters) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::Adamax> adamax = Api::Adamax::Builder().alpha(0.03f).beta1(0.85f).beta2(0.95f).epsilon(1e-4f).build();
    ASSERT_NE(adamax, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    std::shared_ptr<Impl::Adamax> physicalAdamax = stampCompileAdamax(*adamax, weights, stream);

    Impl::Tensor m = requireOptimizerStorage(physicalAdamax, "m");
    Impl::Tensor u = requireOptimizerStorage(physicalAdamax, "u");
    copyValuesToGpuFp32Tensor(m, {1.0f, -2.0f, 3.0f, -4.0f}, stream);
    copyValuesToGpuFp32Tensor(u, {5.0f, 6.0f, 7.0f, 8.0f}, stream);

    synchronizeEvents(adamax->initialize(physicalAdamax, /*isFirstStamp=*/true, nullptr, std::nullopt));
    expectAllClose(copyGpuFp32TensorToValues(m, stream), {0.0f, 0.0f, 0.0f, 0.0f});
    expectAllClose(copyGpuFp32TensorToValues(u, stream), {0.0f, 0.0f, 0.0f, 0.0f});
}

TEST(AdamaxApi, InitializeCopiesStateFromSisterOptimizer) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::Adamax> adamax = Api::Adamax::Builder().alpha(0.03f).beta1(0.85f).beta2(0.95f).epsilon(1e-4f).build();
    ASSERT_NE(adamax, nullptr);

    Impl::Tensor weights1(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    Impl::Tensor weights2(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights1, {1.0f, 2.0f, 3.0f, 4.0f}, stream);
    copyValuesToGpuFp32Tensor(weights2, {-1.0f, -2.0f, -3.0f, -4.0f}, stream);

    std::shared_ptr<Impl::Adamax> sister = stampCompileAdamax(*adamax, weights1, stream);
    std::shared_ptr<Impl::Adamax> physical = stampCompileAdamax(*adamax, weights2, stream);

    Impl::Tensor sisterM = requireOptimizerStorage(sister, "m");
    Impl::Tensor sisterU = requireOptimizerStorage(sister, "u");
    Impl::Tensor m = requireOptimizerStorage(physical, "m");
    Impl::Tensor u = requireOptimizerStorage(physical, "u");

    copyValuesToGpuFp32Tensor(sisterM, {0.1f, 0.2f, 0.3f, 0.4f}, stream);
    copyValuesToGpuFp32Tensor(sisterU, {0.5f, 0.6f, 0.7f, 0.8f}, stream);
    copyValuesToGpuFp32Tensor(m, {9.0f, 9.0f, 9.0f, 9.0f}, stream);
    copyValuesToGpuFp32Tensor(u, {8.0f, 8.0f, 8.0f, 8.0f}, stream);

    synchronizeEvents(adamax->initialize(physical, /*isFirstStamp=*/false, sister, std::nullopt));
    expectAllClose(copyGpuFp32TensorToValues(m, stream), {0.1f, 0.2f, 0.3f, 0.4f});
    expectAllClose(copyGpuFp32TensorToValues(u, stream), {0.5f, 0.6f, 0.7f, 0.8f});
}

TEST(AdamaxApi, PhysicalUpdateThroughApiStampedOptimizerMatchesCpuReference) {
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

    std::shared_ptr<Api::Adamax> adamax = Api::Adamax::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).build();
    ASSERT_NE(adamax, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    std::shared_ptr<Impl::Adamax> physicalAdamax = stampCompileAdamax(*adamax, weights, stream);

    AdamaxReferenceState expected;
    expected.weights = initialWeights;
    expected.m.assign(initialWeights.size(), 0.0f);
    expected.u.assign(initialWeights.size(), 0.0f);

    applyAdamaxReferenceStep(expected, gradient1, batchSize1, alpha, beta1, beta2, epsilon);
    runAdamaxStep(*physicalAdamax, gradient1, batchSize1, stream);

    applyAdamaxReferenceStep(expected, gradient2, batchSize2, alpha, beta1, beta2, epsilon);
    runAdamaxStep(*physicalAdamax, gradient2, batchSize2, stream);

    Impl::Tensor m = requireOptimizerStorage(physicalAdamax, "m");
    Impl::Tensor u = requireOptimizerStorage(physicalAdamax, "u");
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(m, stream), expected.m, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(u, stream), expected.u, 3e-5f, 3e-5f);
    EXPECT_FLOAT_EQ(physicalAdamax->getT(), 2.0f);
}
