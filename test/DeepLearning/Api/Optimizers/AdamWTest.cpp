#include <optional>
#include "DeepLearning/Api/Optimizers/AdamW.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/AdamW.h"
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

Impl::Tensor requireOptimizerStorage(const std::shared_ptr<Impl::AdamW>& adamw, const std::string& name) {
    if (adamw == nullptr)
        throw std::runtime_error("AdamW optimizer is null.");

    if (!adamw->hasParameter(name))
        throw std::runtime_error("AdamW optimizer is missing parameter: " + name);

    std::optional<Impl::Tensor> storage = adamw->getParameter(name)->getStorage();
    if (!storage.has_value())
        throw std::runtime_error("AdamW optimizer parameter has no storage: " + name);

    return storage.value();
}

std::shared_ptr<Impl::AdamW> stampCompileAdamW(Api::AdamW& adamw, Impl::Tensor& weights, Stream& stream) {
    std::shared_ptr<Impl::Optimizer> physicalOptimizer = adamw.stamp(nullptr);
    std::shared_ptr<Impl::AdamW> physicalAdamW = std::dynamic_pointer_cast<Impl::AdamW>(physicalOptimizer);
    if (physicalAdamW == nullptr)
        throw std::runtime_error("Api::AdamW did not stamp an Impl::AdamW.");

    adamw.compile(physicalOptimizer, weights, stream);
    stream.synchronize();

    return physicalAdamW;
}

struct AdamWReferenceState {
    std::vector<float> weights;
    std::vector<float> m;
    std::vector<float> v;
    float t = 0.0f;
};

void applyAdamWReferenceStep(AdamWReferenceState& state,
                             const std::vector<float>& rawGradient,
                             uint32_t batchSize,
                             float alpha,
                             float beta1,
                             float beta2,
                             float epsilon,
                             float weightDecay) {
    ASSERT_EQ(state.weights.size(), rawGradient.size());
    ASSERT_EQ(state.m.size(), rawGradient.size());
    ASSERT_EQ(state.v.size(), rawGradient.size());
    ASSERT_GT(batchSize, 0u);

    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Impl::Loss::getLossScalingFactor());
    state.t += 1.0f;
    const float alphaT = static_cast<float>(static_cast<double>(alpha) * std::sqrt(1.0 - std::pow(static_cast<double>(beta2), state.t)) /
                                           (1.0 - std::pow(static_cast<double>(beta1), state.t)));
    const float alphaWeightDecay = alpha * weightDecay;

    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float g = rawGradient[i] * invBatchLossScale;
        const float mNext = beta1 * state.m[i] + (1.0f - beta1) * g;
        const float vNext = beta2 * state.v[i] + (1.0f - beta2) * g * g;
        const float wNext = state.weights[i] - alphaWeightDecay * state.weights[i] - alphaT * mNext / (std::sqrt(vNext) + epsilon);

        state.weights[i] = wNext;
        state.m[i] = mNext;
        state.v[i] = vNext;
    }
}

void runAdamWStep(Impl::AdamW& adamw, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    std::optional<Impl::Tensor> gradientOpt = adamw.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());

    Impl::Tensor gradient = gradientOpt.value();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);

    adamw.updateWeights(batchSize);
    stream.synchronize();
}

}  // namespace

TEST(AdamWApi, BuilderDefaultsSettersAndArchitectureJson) {
    std::shared_ptr<Api::AdamW> adamw = Api::AdamW::Builder().build();
    ASSERT_NE(adamw, nullptr);

    EXPECT_EQ(adamw->getType(), "AdamW");
    EXPECT_FLOAT_EQ(adamw->getAlpha(), 0.001f);
    EXPECT_FLOAT_EQ(adamw->getBeta1(), 0.9f);
    EXPECT_FLOAT_EQ(adamw->getBeta2(), 0.999f);
    EXPECT_FLOAT_EQ(adamw->getEpsilon(), 1e-7f);
    EXPECT_FLOAT_EQ(adamw->getWeightDecay(), 0.01f);
    EXPECT_EQ(adamw->getOriginalId(), adamw->getId());

    adamw->setAlpha(0.02f, nullptr);
    adamw->setBeta1(0.75f, nullptr);
    adamw->setBeta2(0.92f, nullptr);
    adamw->setEpsilon(1e-5f, nullptr);
    adamw->setWeightDecay(0.03f, nullptr);

    EXPECT_FLOAT_EQ(adamw->getAlpha(), 0.02f);
    EXPECT_FLOAT_EQ(adamw->getBeta1(), 0.75f);
    EXPECT_FLOAT_EQ(adamw->getBeta2(), 0.92f);
    EXPECT_FLOAT_EQ(adamw->getEpsilon(), 1e-5f);
    EXPECT_FLOAT_EQ(adamw->getWeightDecay(), 0.03f);

    json j = adamw->architectureJson();
    ASSERT_EQ(j.at("optimizer_type").get<std::string>(), "adamw");
    ASSERT_EQ(j.at("version").get<std::string>(), adamw->getVersion());
    ASSERT_EQ(j.at("id").get<uint64_t>(), adamw->getId());
    EXPECT_FLOAT_EQ(j.at("t").get<float>(), 0.0f);
    EXPECT_FLOAT_EQ(j.at("alpha").get<float>(), 0.02f);
    EXPECT_FLOAT_EQ(j.at("beta1").get<float>(), 0.75f);
    EXPECT_FLOAT_EQ(j.at("beta2").get<float>(), 0.92f);
    EXPECT_FLOAT_EQ(j.at("epsilon").get<float>(), 1e-5f);
    EXPECT_FLOAT_EQ(j.at("weight_decay").get<float>(), 0.03f);
}

TEST(AdamWApi, BuilderCustomValuesStampAndCompilePhysicalAdamW) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.01f;
    constexpr float beta1 = 0.8f;
    constexpr float beta2 = 0.95f;
    constexpr float epsilon = 1e-4f;
    constexpr float weightDecay = 0.02f;

    std::shared_ptr<Api::AdamW> adamw =
        Api::AdamW::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).weightDecay(weightDecay).build();
    ASSERT_NE(adamw, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f}, stream);

    std::shared_ptr<Impl::AdamW> physicalAdamW = stampCompileAdamW(*adamw, weights, stream);
    ASSERT_NE(physicalAdamW, nullptr);

    EXPECT_TRUE(physicalAdamW->isCompiled());
    EXPECT_EQ(physicalAdamW->getId(), adamw->getId());
    EXPECT_FLOAT_EQ(physicalAdamW->getT(), 0.0f);
    EXPECT_FLOAT_EQ(physicalAdamW->getAlpha(), alpha);
    EXPECT_FLOAT_EQ(physicalAdamW->getBeta1(), beta1);
    EXPECT_FLOAT_EQ(physicalAdamW->getBeta2(), beta2);
    EXPECT_FLOAT_EQ(physicalAdamW->getEpsilon(), epsilon);
    EXPECT_FLOAT_EQ(physicalAdamW->getWeightDecay(), weightDecay);

    std::optional<Impl::Tensor> gradient = physicalAdamW->getWeightsGradient();
    ASSERT_TRUE(gradient.has_value());
    EXPECT_EQ(gradient.value().getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.value().getDataType(), DataType::FP32);
    EXPECT_EQ(gradient.value().getDimensions(), weights.getDimensions());

    Impl::Tensor m = requireOptimizerStorage(physicalAdamW, "m");
    Impl::Tensor v = requireOptimizerStorage(physicalAdamW, "v");

    EXPECT_EQ(m.getPlacement(), gpuPlacement);
    EXPECT_EQ(v.getPlacement(), gpuPlacement);
    EXPECT_EQ(m.getDataType(), DataType::FP32);
    EXPECT_EQ(v.getDataType(), DataType::FP32);
    EXPECT_EQ(m.getDimensions(), weights.getDimensions());
    EXPECT_EQ(v.getDimensions(), weights.getDimensions());

    Impl::Tensor weightsOut = physicalAdamW->getOptimizerParameterTensor("weights");
    EXPECT_EQ(weightsOut, weights);

    std::unordered_map<std::string, float> params = physicalAdamW->getAllHyperParameters();
    ASSERT_EQ(params.size(), 6u);
    expectHyperParameter(params, "t", 0.0f);
    expectHyperParameter(params, "alpha", alpha);
    expectHyperParameter(params, "beta1", beta1);
    expectHyperParameter(params, "beta2", beta2);
    expectHyperParameter(params, "epsilon", epsilon);
    expectHyperParameter(params, "weightDecay", weightDecay);
}

TEST(AdamWApi, InitializeFirstStampZerosMomentParameters) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::AdamW> adamw = Api::AdamW::Builder().alpha(0.01f).beta1(0.8f).beta2(0.95f).epsilon(1e-4f).build();
    ASSERT_NE(adamw, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    std::shared_ptr<Impl::AdamW> physicalAdamW = stampCompileAdamW(*adamw, weights, stream);

    Impl::Tensor m = requireOptimizerStorage(physicalAdamW, "m");
    Impl::Tensor v = requireOptimizerStorage(physicalAdamW, "v");

    copyValuesToGpuFp32Tensor(m, {1.0f, -2.0f, 3.0f, -4.0f}, stream);
    copyValuesToGpuFp32Tensor(v, {5.0f, 6.0f, 7.0f, 8.0f}, stream);

    std::vector<Event> initEvents = adamw->initialize(physicalAdamW, /*isFirstStamp=*/true, nullptr, std::nullopt);
    synchronizeEvents(initEvents);
    stream.synchronize();

    expectAllClose(copyGpuFp32TensorToValues(m, stream), {0.0f, 0.0f, 0.0f, 0.0f});
    expectAllClose(copyGpuFp32TensorToValues(v, stream), {0.0f, 0.0f, 0.0f, 0.0f});
}

TEST(AdamWApi, InitializeNonFirstStampCopiesMomentParametersFromSisterOptimizer) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::AdamW> adamw = Api::AdamW::Builder().alpha(0.01f).beta1(0.8f).beta2(0.95f).epsilon(1e-4f).build();
    ASSERT_NE(adamw, nullptr);

    Impl::Tensor sisterWeights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    Impl::Tensor targetWeights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(sisterWeights, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, stream);
    copyValuesToGpuFp32Tensor(targetWeights, {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f}, stream);

    std::shared_ptr<Impl::AdamW> sisterAdamW = stampCompileAdamW(*adamw, sisterWeights, stream);
    std::shared_ptr<Impl::AdamW> targetAdamW = stampCompileAdamW(*adamw, targetWeights, stream);

    Impl::Tensor sisterM = requireOptimizerStorage(sisterAdamW, "m");
    Impl::Tensor sisterV = requireOptimizerStorage(sisterAdamW, "v");
    Impl::Tensor targetM = requireOptimizerStorage(targetAdamW, "m");
    Impl::Tensor targetV = requireOptimizerStorage(targetAdamW, "v");

    const std::vector<float> expectedM{1.5f, -2.5f, 3.5f, -4.5f, 5.5f, -6.5f};
    const std::vector<float> expectedV{0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f};

    copyValuesToGpuFp32Tensor(sisterM, expectedM, stream);
    copyValuesToGpuFp32Tensor(sisterV, expectedV, stream);

    copyValuesToGpuFp32Tensor(targetM, {100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f}, stream);
    copyValuesToGpuFp32Tensor(targetV, {200.0f, 201.0f, 202.0f, 203.0f, 204.0f, 205.0f}, stream);

    std::optional<Event> sisterReadyEvent = stream.putEvent(false, true);

    std::vector<Event> initEvents = adamw->initialize(targetAdamW, /*isFirstStamp=*/false, sisterAdamW, sisterReadyEvent);
    synchronizeEvents(initEvents);
    stream.synchronize();

    expectAllClose(copyGpuFp32TensorToValues(targetM, stream), expectedM);
    expectAllClose(copyGpuFp32TensorToValues(targetV, stream), expectedV);
}

TEST(AdamWApi, StampedPhysicalAdamWStepMatchesCpuReference) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.01f;
    constexpr float beta1 = 0.8f;
    constexpr float beta2 = 0.95f;
    constexpr float epsilon = 1e-4f;
    constexpr float weightDecay = 0.1f;

    const std::vector<float> initialWeights{2.0f, -1.0f, 3.0f, -4.0f};
    const std::vector<float> gradient1{4.0f, -8.0f, 1.0f, -2.0f};
    const std::vector<float> gradient2{-5.0f, 6.0f, -7.0f, 8.0f};

    std::shared_ptr<Api::AdamW> adamw =
        Api::AdamW::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).weightDecay(weightDecay).build();
    ASSERT_NE(adamw, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    std::shared_ptr<Impl::AdamW> physicalAdamW = stampCompileAdamW(*adamw, weights, stream);
    Impl::Tensor m = physicalAdamW->getOptimizerParameterTensor("m");
    Impl::Tensor v = physicalAdamW->getOptimizerParameterTensor("v");

    AdamWReferenceState expected;
    expected.weights = initialWeights;
    expected.m.assign(initialWeights.size(), 0.0f);
    expected.v.assign(initialWeights.size(), 0.0f);

    applyAdamWReferenceStep(expected, gradient1, /*batchSize=*/1, alpha, beta1, beta2, epsilon, weightDecay);
    runAdamWStep(*physicalAdamW, gradient1, /*batchSize=*/1, stream);

    applyAdamWReferenceStep(expected, gradient2, /*batchSize=*/2, alpha, beta1, beta2, epsilon, weightDecay);
    runAdamWStep(*physicalAdamW, gradient2, /*batchSize=*/2, stream);

    EXPECT_FLOAT_EQ(physicalAdamW->getT(), 2.0f);
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(m, stream), expected.m, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(v, stream), expected.v, 3e-5f, 3e-5f);
}

TEST(AdamWApi, SerializeArchitectureOnlyAndDeserialize) {
    constexpr float alpha = 0.123f;
    constexpr float beta1 = 0.456f;
    constexpr float beta2 = 0.789f;
    constexpr float epsilon = 1e-4f;
    constexpr float weightDecay = 0.02f;

    std::shared_ptr<Api::AdamW> adamw =
        Api::AdamW::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).weightDecay(weightDecay).build();
    ASSERT_NE(adamw, nullptr);

    json j = adamw->architectureJson();

    std::shared_ptr<thor_file::TarReader> archiveReader;
    std::shared_ptr<Api::Optimizer> optimizer = Api::Optimizer::deserialize(archiveReader, j, nullptr);
    std::shared_ptr<Api::AdamW> deserializedAdamW = std::dynamic_pointer_cast<Api::AdamW>(optimizer);
    ASSERT_NE(deserializedAdamW, nullptr);

    EXPECT_EQ(deserializedAdamW->getOriginalId(), adamw->getId());
    EXPECT_FLOAT_EQ(deserializedAdamW->getAlpha(), alpha);
    EXPECT_FLOAT_EQ(deserializedAdamW->getBeta1(), beta1);
    EXPECT_FLOAT_EQ(deserializedAdamW->getBeta2(), beta2);
    EXPECT_FLOAT_EQ(deserializedAdamW->getEpsilon(), epsilon);
    EXPECT_FLOAT_EQ(deserializedAdamW->getWeightDecay(), weightDecay);

    Stream stream(gpuPlacement);
    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    std::shared_ptr<Impl::AdamW> physicalAdamW = stampCompileAdamW(*deserializedAdamW, weights, stream);
    ASSERT_NE(physicalAdamW, nullptr);

    EXPECT_EQ(physicalAdamW->getId(), deserializedAdamW->getId());
    EXPECT_FLOAT_EQ(physicalAdamW->getAlpha(), alpha);
    EXPECT_FLOAT_EQ(physicalAdamW->getBeta1(), beta1);
    EXPECT_FLOAT_EQ(physicalAdamW->getBeta2(), beta2);
    EXPECT_FLOAT_EQ(physicalAdamW->getEpsilon(), epsilon);
    EXPECT_FLOAT_EQ(physicalAdamW->getWeightDecay(), weightDecay);
}

TEST(AdamWApi, SerializeWithStateRecordsMomentFilesAndPhysicalTime) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.01f;
    constexpr float beta1 = 0.8f;
    constexpr float beta2 = 0.95f;
    constexpr float epsilon = 1e-4f;
    constexpr float weightDecay = 0.02f;

    std::shared_ptr<Api::AdamW> adamw =
        Api::AdamW::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).weightDecay(weightDecay).build();
    ASSERT_NE(adamw, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    std::shared_ptr<Impl::AdamW> physicalAdamW = stampCompileAdamW(*adamw, weights, stream);

    Impl::Tensor m = requireOptimizerStorage(physicalAdamW, "m");
    Impl::Tensor v = requireOptimizerStorage(physicalAdamW, "v");
    copyValuesToGpuFp32Tensor(m, {0.1f, 0.2f, 0.3f, 0.4f}, stream);
    copyValuesToGpuFp32Tensor(v, {1.1f, 1.2f, 1.3f, 1.4f}, stream);
    physicalAdamW->setT(12.0f);

    thor_file::TarWriter archiveWriter("adamw_api_test");
    json stateJson = adamw->serialize(archiveWriter, stream, physicalAdamW, "layer123_weights", /*saveOptimizerState=*/true);

    ASSERT_EQ(stateJson.at("optimizer_type").get<std::string>(), "adamw");
    ASSERT_EQ(stateJson.at("version").get<std::string>(), adamw->getVersion());
    EXPECT_EQ(stateJson.at("id").get<uint64_t>(), adamw->getId());
    EXPECT_FLOAT_EQ(stateJson.at("t").get<float>(), 12.0f);
    EXPECT_FLOAT_EQ(stateJson.at("alpha").get<float>(), alpha);
    EXPECT_FLOAT_EQ(stateJson.at("beta1").get<float>(), beta1);
    EXPECT_FLOAT_EQ(stateJson.at("beta2").get<float>(), beta2);
    EXPECT_FLOAT_EQ(stateJson.at("epsilon").get<float>(), epsilon);
    EXPECT_FLOAT_EQ(stateJson.at("weight_decay").get<float>(), weightDecay);

    ASSERT_TRUE(stateJson.contains("m_tensor"));
    ASSERT_TRUE(stateJson.contains("v_tensor"));
    EXPECT_EQ(stateJson.at("m_tensor").get<std::string>(), "layer123_weights_adamw_m.gds");
    EXPECT_EQ(stateJson.at("v_tensor").get<std::string>(), "layer123_weights_adamw_v.gds");

    thor_file::TarWriter architectureOnlyWriter("adamw_api_architecture_only_test");
    json architectureOnlyJson = adamw->serialize(architectureOnlyWriter, stream, nullptr, "", /*saveOptimizerState=*/false);

    EXPECT_FALSE(architectureOnlyJson.contains("m_tensor"));
    EXPECT_FALSE(architectureOnlyJson.contains("v_tensor"));
    EXPECT_FLOAT_EQ(architectureOnlyJson.at("t").get<float>(), 0.0f);
}
