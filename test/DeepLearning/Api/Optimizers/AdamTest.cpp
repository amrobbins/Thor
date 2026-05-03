#include "DeepLearning/Api/Optimizers/Adam.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adam.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
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

using DataType = Impl::TensorDescriptor::DataType;

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

Impl::Tensor requireOptimizerStorage(const std::shared_ptr<Impl::Adam>& adam, const std::string& name) {
    if (adam == nullptr)
        throw std::runtime_error("Adam optimizer is null.");

    if (!adam->hasParameter(name))
        throw std::runtime_error("Adam optimizer is missing parameter: " + name);

    Optional<Impl::Tensor> storage = adam->getParameter(name)->getStorage();
    if (storage.isEmpty())
        throw std::runtime_error("Adam optimizer parameter has no storage: " + name);

    return storage.get();
}

std::shared_ptr<Impl::Adam> stampCompileAdam(Api::Adam& adam, Impl::Tensor& weights, Stream& stream) {
    std::shared_ptr<Impl::Optimizer> physicalOptimizer = adam.stamp(nullptr);
    std::shared_ptr<Impl::Adam> physicalAdam = std::dynamic_pointer_cast<Impl::Adam>(physicalOptimizer);
    if (physicalAdam == nullptr)
        throw std::runtime_error("Api::Adam did not stamp an Impl::Adam.");

    adam.compile(physicalOptimizer, weights, stream);
    stream.synchronize();

    return physicalAdam;
}

}  // namespace

TEST(AdamApi, BuilderDefaultsSettersAndArchitectureJson) {
    std::shared_ptr<Api::Adam> adam = Api::Adam::Builder().build();
    ASSERT_NE(adam, nullptr);

    EXPECT_EQ(adam->getType(), "Adam");
    EXPECT_FLOAT_EQ(adam->getAlpha(), 0.001f);
    EXPECT_FLOAT_EQ(adam->getBeta1(), 0.9f);
    EXPECT_FLOAT_EQ(adam->getBeta2(), 0.999f);
    EXPECT_FLOAT_EQ(adam->getEpsilon(), 1e-7f);
    EXPECT_EQ(adam->getOriginalId(), adam->getId());

    adam->setAlpha(0.02f, nullptr);
    adam->setBeta1(0.75f, nullptr);
    adam->setBeta2(0.92f, nullptr);
    adam->setEpsilon(1e-5f, nullptr);

    EXPECT_FLOAT_EQ(adam->getAlpha(), 0.02f);
    EXPECT_FLOAT_EQ(adam->getBeta1(), 0.75f);
    EXPECT_FLOAT_EQ(adam->getBeta2(), 0.92f);
    EXPECT_FLOAT_EQ(adam->getEpsilon(), 1e-5f);

    json j = adam->architectureJson();
    ASSERT_EQ(j.at("optimizer_type").get<std::string>(), "adam");
    ASSERT_EQ(j.at("version").get<std::string>(), adam->getVersion());
    ASSERT_EQ(j.at("id").get<uint64_t>(), adam->getId());
    EXPECT_FLOAT_EQ(j.at("t").get<float>(), 0.0f);
    EXPECT_FLOAT_EQ(j.at("alpha").get<float>(), 0.02f);
    EXPECT_FLOAT_EQ(j.at("beta1").get<float>(), 0.75f);
    EXPECT_FLOAT_EQ(j.at("beta2").get<float>(), 0.92f);
    EXPECT_FLOAT_EQ(j.at("epsilon").get<float>(), 1e-5f);
}

TEST(AdamApi, BuilderCustomValuesStampAndCompilePhysicalAdam) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.01f;
    constexpr float beta1 = 0.8f;
    constexpr float beta2 = 0.95f;
    constexpr float epsilon = 1e-4f;

    std::shared_ptr<Api::Adam> adam = Api::Adam::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).build();
    ASSERT_NE(adam, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f}, stream);

    std::shared_ptr<Impl::Adam> physicalAdam = stampCompileAdam(*adam, weights, stream);
    ASSERT_NE(physicalAdam, nullptr);

    EXPECT_TRUE(physicalAdam->isCompiled());
    EXPECT_EQ(physicalAdam->getId(), adam->getId());
    EXPECT_FLOAT_EQ(physicalAdam->getT(), 0.0f);
    EXPECT_FLOAT_EQ(physicalAdam->getAlpha(), alpha);
    EXPECT_FLOAT_EQ(physicalAdam->getBeta1(), beta1);
    EXPECT_FLOAT_EQ(physicalAdam->getBeta2(), beta2);
    EXPECT_FLOAT_EQ(physicalAdam->getEpsilon(), epsilon);

    Optional<Impl::Tensor> gradient = physicalAdam->getWeightsGradient();
    ASSERT_TRUE(gradient.isPresent());
    EXPECT_EQ(gradient.get().getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.get().getDataType(), DataType::FP32);
    EXPECT_EQ(gradient.get().getDimensions(), weights.getDimensions());

    Impl::Tensor m = requireOptimizerStorage(physicalAdam, "m");
    Impl::Tensor v = requireOptimizerStorage(physicalAdam, "v");

    EXPECT_EQ(m.getPlacement(), gpuPlacement);
    EXPECT_EQ(v.getPlacement(), gpuPlacement);
    EXPECT_EQ(m.getDataType(), DataType::FP32);
    EXPECT_EQ(v.getDataType(), DataType::FP32);
    EXPECT_EQ(m.getDimensions(), weights.getDimensions());
    EXPECT_EQ(v.getDimensions(), weights.getDimensions());

    std::unordered_map<std::string, float> params = physicalAdam->getAllHyperParameters();
    ASSERT_EQ(params.size(), 5u);
    expectHyperParameter(params, "t", 0.0f);
    expectHyperParameter(params, "alpha", alpha);
    expectHyperParameter(params, "beta1", beta1);
    expectHyperParameter(params, "beta2", beta2);
    expectHyperParameter(params, "epsilon", epsilon);
}

TEST(AdamApi, InitializeFirstStampZerosMomentParameters) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::Adam> adam = Api::Adam::Builder().alpha(0.01f).beta1(0.8f).beta2(0.95f).epsilon(1e-4f).build();
    ASSERT_NE(adam, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    std::shared_ptr<Impl::Adam> physicalAdam = stampCompileAdam(*adam, weights, stream);

    Impl::Tensor m = requireOptimizerStorage(physicalAdam, "m");
    Impl::Tensor v = requireOptimizerStorage(physicalAdam, "v");

    copyValuesToGpuFp32Tensor(m, {1.0f, -2.0f, 3.0f, -4.0f}, stream);
    copyValuesToGpuFp32Tensor(v, {5.0f, 6.0f, 7.0f, 8.0f}, stream);

    std::vector<Event> initEvents = adam->initialize(physicalAdam, /*isFirstStamp=*/true, nullptr, Optional<Event>::empty());
    synchronizeEvents(initEvents);
    stream.synchronize();

    expectAllClose(copyGpuFp32TensorToValues(m, stream), {0.0f, 0.0f, 0.0f, 0.0f});
    expectAllClose(copyGpuFp32TensorToValues(v, stream), {0.0f, 0.0f, 0.0f, 0.0f});
}

TEST(AdamApi, InitializeNonFirstStampCopiesMomentParametersFromSisterOptimizer) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::Adam> adam = Api::Adam::Builder().alpha(0.01f).beta1(0.8f).beta2(0.95f).epsilon(1e-4f).build();
    ASSERT_NE(adam, nullptr);

    Impl::Tensor sisterWeights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    Impl::Tensor targetWeights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(sisterWeights, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, stream);
    copyValuesToGpuFp32Tensor(targetWeights, {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f}, stream);

    std::shared_ptr<Impl::Adam> sisterAdam = stampCompileAdam(*adam, sisterWeights, stream);
    std::shared_ptr<Impl::Adam> targetAdam = stampCompileAdam(*adam, targetWeights, stream);

    Impl::Tensor sisterM = requireOptimizerStorage(sisterAdam, "m");
    Impl::Tensor sisterV = requireOptimizerStorage(sisterAdam, "v");
    Impl::Tensor targetM = requireOptimizerStorage(targetAdam, "m");
    Impl::Tensor targetV = requireOptimizerStorage(targetAdam, "v");

    const std::vector<float> expectedM{1.5f, -2.5f, 3.5f, -4.5f, 5.5f, -6.5f};
    const std::vector<float> expectedV{0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f};

    copyValuesToGpuFp32Tensor(sisterM, expectedM, stream);
    copyValuesToGpuFp32Tensor(sisterV, expectedV, stream);

    copyValuesToGpuFp32Tensor(targetM, {100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f}, stream);
    copyValuesToGpuFp32Tensor(targetV, {200.0f, 201.0f, 202.0f, 203.0f, 204.0f, 205.0f}, stream);

    Optional<Event> sisterReadyEvent = stream.putEvent(false, true);

    std::vector<Event> initEvents = adam->initialize(targetAdam, /*isFirstStamp=*/false, sisterAdam, sisterReadyEvent);
    synchronizeEvents(initEvents);
    stream.synchronize();

    expectAllClose(copyGpuFp32TensorToValues(targetM, stream), expectedM);
    expectAllClose(copyGpuFp32TensorToValues(targetV, stream), expectedV);
}

TEST(AdamApi, SerializeArchitectureOnlyAndDeserialize) {
    constexpr float alpha = 0.123f;
    constexpr float beta1 = 0.456f;
    constexpr float beta2 = 0.789f;
    constexpr float epsilon = 1e-4f;

    std::shared_ptr<Api::Adam> adam = Api::Adam::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).build();
    ASSERT_NE(adam, nullptr);

    json j = adam->architectureJson();

    std::shared_ptr<thor_file::TarReader> archiveReader;
    std::shared_ptr<Api::Optimizer> optimizer = Api::Optimizer::deserialize(archiveReader, j, nullptr);
    std::shared_ptr<Api::Adam> deserializedAdam = std::dynamic_pointer_cast<Api::Adam>(optimizer);
    ASSERT_NE(deserializedAdam, nullptr);

    EXPECT_EQ(deserializedAdam->getOriginalId(), adam->getId());
    EXPECT_FLOAT_EQ(deserializedAdam->getAlpha(), alpha);
    EXPECT_FLOAT_EQ(deserializedAdam->getBeta1(), beta1);
    EXPECT_FLOAT_EQ(deserializedAdam->getBeta2(), beta2);
    EXPECT_FLOAT_EQ(deserializedAdam->getEpsilon(), epsilon);

    Stream stream(gpuPlacement);
    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    std::shared_ptr<Impl::Adam> physicalAdam = stampCompileAdam(*deserializedAdam, weights, stream);
    ASSERT_NE(physicalAdam, nullptr);

    EXPECT_EQ(physicalAdam->getId(), deserializedAdam->getId());
    EXPECT_FLOAT_EQ(physicalAdam->getAlpha(), alpha);
    EXPECT_FLOAT_EQ(physicalAdam->getBeta1(), beta1);
    EXPECT_FLOAT_EQ(physicalAdam->getBeta2(), beta2);
    EXPECT_FLOAT_EQ(physicalAdam->getEpsilon(), epsilon);
}

TEST(AdamApi, SerializeWithStateRecordsMomentFilesAndPhysicalTime) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.01f;
    constexpr float beta1 = 0.8f;
    constexpr float beta2 = 0.95f;
    constexpr float epsilon = 1e-4f;

    std::shared_ptr<Api::Adam> adam = Api::Adam::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).build();
    ASSERT_NE(adam, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    std::shared_ptr<Impl::Adam> physicalAdam = stampCompileAdam(*adam, weights, stream);

    Impl::Tensor m = requireOptimizerStorage(physicalAdam, "m");
    Impl::Tensor v = requireOptimizerStorage(physicalAdam, "v");
    copyValuesToGpuFp32Tensor(m, {0.1f, 0.2f, 0.3f, 0.4f}, stream);
    copyValuesToGpuFp32Tensor(v, {1.1f, 1.2f, 1.3f, 1.4f}, stream);
    physicalAdam->setT(12.0f);

    thor_file::TarWriter archiveWriter("adam_api_test");
    json stateJson = adam->serialize(archiveWriter, stream, physicalAdam, "layer123_weights", /*saveOptimizerState=*/true);

    ASSERT_EQ(stateJson.at("optimizer_type").get<std::string>(), "adam");
    ASSERT_EQ(stateJson.at("version").get<std::string>(), adam->getVersion());
    EXPECT_EQ(stateJson.at("id").get<uint64_t>(), adam->getId());
    EXPECT_FLOAT_EQ(stateJson.at("t").get<float>(), 12.0f);
    EXPECT_FLOAT_EQ(stateJson.at("alpha").get<float>(), alpha);
    EXPECT_FLOAT_EQ(stateJson.at("beta1").get<float>(), beta1);
    EXPECT_FLOAT_EQ(stateJson.at("beta2").get<float>(), beta2);
    EXPECT_FLOAT_EQ(stateJson.at("epsilon").get<float>(), epsilon);

    ASSERT_TRUE(stateJson.contains("m_tensor"));
    ASSERT_TRUE(stateJson.contains("v_tensor"));
    EXPECT_EQ(stateJson.at("m_tensor").get<std::string>(), "layer123_weights_adam_m.gds");
    EXPECT_EQ(stateJson.at("v_tensor").get<std::string>(), "layer123_weights_adam_v.gds");

    EXPECT_FALSE(stateJson.contains("m_bias_tensor"));
    EXPECT_FALSE(stateJson.contains("v_bias_tensor"));

    thor_file::TarWriter architectureOnlyWriter("adam_api_architecture_only_test");
    json architectureOnlyJson = adam->serialize(architectureOnlyWriter, stream, nullptr, "", /*saveOptimizerState=*/false);

    EXPECT_FALSE(architectureOnlyJson.contains("m_tensor"));
    EXPECT_FALSE(architectureOnlyJson.contains("v_tensor"));
    EXPECT_FLOAT_EQ(architectureOnlyJson.at("t").get<float>(), 0.0f);
}
