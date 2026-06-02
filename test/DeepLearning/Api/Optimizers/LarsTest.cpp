#include "DeepLearning/Api/Optimizers/Lars.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Lars.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

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

Impl::Tensor requireOptimizerStorage(const std::shared_ptr<Impl::Lars>& lars, const std::string& name) {
    if (lars == nullptr)
        throw std::runtime_error("LARS optimizer is null.");
    if (!lars->hasParameter(name))
        throw std::runtime_error("LARS optimizer is missing parameter: " + name);
    std::optional<Impl::Tensor> storage = lars->getParameter(name)->getStorage();
    if (!storage.has_value())
        throw std::runtime_error("LARS optimizer parameter has no storage: " + name);
    return storage.value();
}

std::shared_ptr<Impl::Lars> stampCompileLars(Api::Lars& lars, Impl::Tensor& weights, Stream& stream) {
    std::shared_ptr<Impl::Optimizer> physicalOptimizer = lars.stamp(nullptr);
    std::shared_ptr<Impl::Lars> physicalLars = std::dynamic_pointer_cast<Impl::Lars>(physicalOptimizer);
    if (physicalLars == nullptr)
        throw std::runtime_error("Api::Lars did not stamp an Impl::Lars.");

    lars.compile(physicalOptimizer, weights, stream);
    stream.synchronize();
    return physicalLars;
}

}  // namespace

TEST(LarsApi, BuilderDefaultsSettersAndArchitectureJson) {
    std::shared_ptr<Api::Lars> lars = Api::Lars::Builder().build();
    ASSERT_NE(lars, nullptr);

    EXPECT_EQ(lars->getType(), "Lars");
    EXPECT_FLOAT_EQ(lars->getAlpha(), 0.01f);
    EXPECT_FLOAT_EQ(lars->getMomentum(), 0.9f);
    EXPECT_FLOAT_EQ(lars->getWeightDecay(), 0.0f);
    EXPECT_FLOAT_EQ(lars->getTrustCoefficient(), 0.001f);
    EXPECT_FLOAT_EQ(lars->getEpsilon(), 1e-8f);
    EXPECT_FALSE(lars->getUseNesterovMomentum());
    EXPECT_EQ(lars->getOriginalId(), lars->getId());

    lars->setAlpha(0.2f, nullptr);
    lars->setMomentum(0.8f, nullptr);
    lars->setWeightDecay(0.02f, nullptr);
    lars->setTrustCoefficient(0.002f, nullptr);
    lars->setEpsilon(1e-6f, nullptr);
    lars->setUseNesterovMomentum(true, nullptr);

    json j = lars->architectureJson();
    ASSERT_EQ(j.at("optimizer_type").get<std::string>(), "lars");
    ASSERT_EQ(j.at("version").get<std::string>(), lars->getVersion());
    ASSERT_EQ(j.at("id").get<uint64_t>(), lars->getId());
    EXPECT_FLOAT_EQ(j.at("alpha").get<float>(), 0.2f);
    EXPECT_FLOAT_EQ(j.at("momentum").get<float>(), 0.8f);
    EXPECT_FLOAT_EQ(j.at("weight_decay").get<float>(), 0.02f);
    EXPECT_FLOAT_EQ(j.at("trust_coefficient").get<float>(), 0.002f);
    EXPECT_FLOAT_EQ(j.at("epsilon").get<float>(), 1e-6f);
    EXPECT_TRUE(j.at("use_nesterov").get<bool>());
}

TEST(LarsApi, BuilderCustomValuesStampAndCompilePhysicalLars) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.3f;
    constexpr float momentum = 0.75f;
    constexpr float weightDecay = 0.04f;
    constexpr float trustCoefficient = 0.003f;
    constexpr float epsilon = 1e-5f;
    constexpr bool useNesterovMomentum = true;

    std::shared_ptr<Api::Lars> lars = Api::Lars::Builder()
                                        .alpha(alpha)
                                        .momentum(momentum)
                                        .weightDecay(weightDecay)
                                        .trustCoefficient(trustCoefficient)
                                        .epsilon(epsilon)
                                        .useNesterovMomentum(useNesterovMomentum)
                                        .build();
    ASSERT_NE(lars, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f}, stream);

    std::shared_ptr<Impl::Lars> physicalLars = stampCompileLars(*lars, weights, stream);
    ASSERT_NE(physicalLars, nullptr);

    EXPECT_TRUE(physicalLars->isCompiled());
    EXPECT_EQ(physicalLars->getId(), lars->getId());
    EXPECT_FLOAT_EQ(physicalLars->getAlpha(), alpha);
    EXPECT_FLOAT_EQ(physicalLars->getMomentum(), momentum);
    EXPECT_FLOAT_EQ(physicalLars->getWeightDecay(), weightDecay);
    EXPECT_FLOAT_EQ(physicalLars->getTrustCoefficient(), trustCoefficient);
    EXPECT_FLOAT_EQ(physicalLars->getEpsilon(), epsilon);
    EXPECT_TRUE(physicalLars->getUseNesterovMomentum());

    std::optional<Impl::Tensor> gradient = physicalLars->getWeightsGradient();
    ASSERT_TRUE(gradient.has_value());
    EXPECT_EQ(gradient.value().getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.value().getDataType(), DataType::FP32);
    EXPECT_EQ(gradient.value().getDimensions(), weights.getDimensions());

    Impl::Tensor velocity = requireOptimizerStorage(physicalLars, "velocity");
    EXPECT_EQ(velocity.getPlacement(), gpuPlacement);
    EXPECT_EQ(velocity.getDataType(), DataType::FP32);
    EXPECT_EQ(velocity.getDimensions(), weights.getDimensions());

    std::unordered_map<std::string, float> params = physicalLars->getAllHyperParameters();
    ASSERT_EQ(params.size(), 6u);
    EXPECT_FLOAT_EQ(params.at("alpha"), alpha);
    EXPECT_FLOAT_EQ(params.at("momentum"), momentum);
    EXPECT_FLOAT_EQ(params.at("weightDecay"), weightDecay);
    EXPECT_FLOAT_EQ(params.at("trustCoefficient"), trustCoefficient);
    EXPECT_FLOAT_EQ(params.at("epsilon"), epsilon);
    EXPECT_FLOAT_EQ(params.at("useNesterovMomentum"), 1.0f);
}

TEST(LarsApi, InitializeFirstStampZerosVelocityParameter) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::Lars> lars = Api::Lars::Builder().alpha(0.1f).momentum(0.8f).weightDecay(0.01f).build();
    ASSERT_NE(lars, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    std::shared_ptr<Impl::Lars> physicalLars = stampCompileLars(*lars, weights, stream);
    Impl::Tensor velocity = requireOptimizerStorage(physicalLars, "velocity");
    copyValuesToGpuFp32Tensor(velocity, {1.0f, -2.0f, 3.0f, -4.0f}, stream);

    synchronizeEvents(lars->initialize(physicalLars, true, nullptr, std::nullopt));

    std::vector<float> expected(4, 0.0f);
    EXPECT_EQ(copyGpuFp32TensorToValues(velocity, stream), expected);
}
