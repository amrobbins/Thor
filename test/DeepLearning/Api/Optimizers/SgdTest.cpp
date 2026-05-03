#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <set>
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

float computeCurrentLearningRate(float initialLearningRate, float decay, uint64_t epoch) {
    return static_cast<float>(static_cast<double>(initialLearningRate) * std::pow(1.0 - static_cast<double>(decay), epoch));
}

float computeStep(float currentLearningRate, uint32_t batchSize) {
    EXPECT_GT(batchSize, 0u);
    return currentLearningRate / (static_cast<float>(batchSize) * Impl::Loss::getLossScalingFactor());
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

std::shared_ptr<Impl::Sgd> stampCompileSgd(Api::Sgd& sgd, Impl::Tensor& weights, Stream& stream) {
    std::shared_ptr<Impl::Optimizer> physicalOptimizer = sgd.stamp(nullptr);
    std::shared_ptr<Impl::Sgd> physicalSgd = std::dynamic_pointer_cast<Impl::Sgd>(physicalOptimizer);
    if (physicalSgd == nullptr)
        throw std::runtime_error("Api::Sgd did not stamp an Impl::Sgd.");

    sgd.compile(physicalOptimizer, weights, stream);
    stream.synchronize();

    return physicalSgd;
}

struct SgdReferenceState {
    std::vector<float> weights;
    std::vector<float> velocity;
};

void applyPlainSgdReferenceStep(SgdReferenceState& state,
                                const std::vector<float>& rawGradient,
                                uint32_t batchSize,
                                float currentLearningRate) {
    ASSERT_EQ(state.weights.size(), rawGradient.size());

    const float step = computeStep(currentLearningRate, batchSize);
    for (uint64_t i = 0; i < rawGradient.size(); ++i)
        state.weights[i] -= step * rawGradient[i];
}

void applyMomentumSgdReferenceStep(SgdReferenceState& state,
                                   const std::vector<float>& rawGradient,
                                   uint32_t batchSize,
                                   float currentLearningRate,
                                   float momentum,
                                   bool useNesterovMomentum) {
    ASSERT_EQ(state.weights.size(), rawGradient.size());
    ASSERT_EQ(state.velocity.size(), rawGradient.size());

    const float step = computeStep(currentLearningRate, batchSize);
    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float vNext = momentum * state.velocity[i] - step * rawGradient[i];

        if (useNesterovMomentum) {
            state.weights[i] += momentum * vNext - step * rawGradient[i];
        } else {
            state.weights[i] += vNext;
        }

        state.velocity[i] = vNext;
    }
}

void runSgdStep(Impl::Sgd& sgd, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    Optional<Impl::Tensor> gradientOpt = sgd.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.isPresent());

    Impl::Tensor gradient = gradientOpt.get();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);

    sgd.updateWeights(batchSize);
    stream.synchronize();
}

Impl::Tensor getMomentumStorage(Impl::Sgd& sgd) {
    EXPECT_TRUE(sgd.hasParameter("momentum"));

    Optional<Impl::Tensor> storage = sgd.getParameter("momentum")->getStorage();
    EXPECT_TRUE(storage.isPresent());

    return storage.get();
}

}  // namespace

TEST(SgdApi, BuilderDefaultsSettersAndArchitectureJson) {
    std::shared_ptr<Api::Sgd> sgd = Api::Sgd::Builder().build();
    ASSERT_NE(sgd, nullptr);

    EXPECT_EQ(sgd->getType(), "SGD");
    EXPECT_FLOAT_EQ(sgd->getInitialLearningRate(), 0.01f);
    EXPECT_FLOAT_EQ(sgd->getDecay(), 0.0f);
    EXPECT_FLOAT_EQ(sgd->getMomentum(), 0.0f);
    EXPECT_FALSE(sgd->getUseNesterovMomentum());
    EXPECT_EQ(sgd->getEpoch(), 0u);
    EXPECT_EQ(sgd->getOriginalId(), sgd->getId());

    sgd->setInitialLearningRate(0.2f, nullptr);
    sgd->setDecay(0.3f, nullptr);
    sgd->setMomentum(0.4f, nullptr);
    sgd->setUseNesterovMomentum(true, nullptr);

    EXPECT_FLOAT_EQ(sgd->getInitialLearningRate(), 0.2f);
    EXPECT_FLOAT_EQ(sgd->getDecay(), 0.3f);
    EXPECT_FLOAT_EQ(sgd->getMomentum(), 0.4f);
    EXPECT_TRUE(sgd->getUseNesterovMomentum());

    sgd->setConstantLearningRate(0.125f, nullptr);
    EXPECT_FLOAT_EQ(sgd->getInitialLearningRate(), 0.125f);
    EXPECT_FLOAT_EQ(sgd->getDecay(), 0.0f);

    json j = sgd->architectureJson();
    ASSERT_EQ(j.at("optimizer_type").get<std::string>(), "sgd");
    ASSERT_EQ(j.at("version").get<std::string>(), sgd->getVersion());
    ASSERT_EQ(j.at("id").get<uint64_t>(), sgd->getId());
    EXPECT_FLOAT_EQ(j.at("initial_learning_rate").get<float>(), 0.125f);
    EXPECT_FLOAT_EQ(j.at("decay").get<float>(), 0.0f);
    EXPECT_FLOAT_EQ(j.at("momentum").get<float>(), 0.4f);
    EXPECT_TRUE(j.at("use_nesterov").get<bool>());
    EXPECT_EQ(j.at("epoch").get<uint64_t>(), 0u);
}

TEST(SgdApi, BuilderCustomValuesStampAndCompilePhysicalSgdWithoutMomentum) {
    Stream stream(gpuPlacement);

    constexpr float initialLearningRate = 0.25f;
    constexpr float decay = 0.1f;
    constexpr float momentum = 0.0f;
    constexpr bool useNesterovMomentum = false;

    std::shared_ptr<Api::Sgd> sgd = Api::Sgd::Builder()
                                        .initialLearningRate(initialLearningRate)
                                        .decay(decay)
                                        .momentum(momentum)
                                        .useNesterovMomentum(useNesterovMomentum)
                                        .build();
    ASSERT_NE(sgd, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f}, stream);

    std::shared_ptr<Impl::Sgd> physicalSgd = stampCompileSgd(*sgd, weights, stream);
    ASSERT_NE(physicalSgd, nullptr);

    EXPECT_TRUE(physicalSgd->isCompiled());
    EXPECT_EQ(physicalSgd->getId(), sgd->getId());
    EXPECT_FLOAT_EQ(physicalSgd->getInitialLearningRate(), initialLearningRate);
    EXPECT_FLOAT_EQ(physicalSgd->getDecay(), decay);
    EXPECT_FLOAT_EQ(physicalSgd->getMomentum(), momentum);
    EXPECT_EQ(physicalSgd->getUseNesterovMomentum(), useNesterovMomentum);
    EXPECT_FALSE(physicalSgd->hasParameter("momentum"));

    Optional<Impl::Tensor> gradient = physicalSgd->getWeightsGradient();
    ASSERT_TRUE(gradient.isPresent());
    EXPECT_EQ(gradient.get().getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.get().getDataType(), DataType::FP32);
    EXPECT_EQ(gradient.get().getDimensions(), weights.getDimensions());

    Impl::Tensor weightsOut = physicalSgd->getOptimizerParameterTensor("weights");
    EXPECT_EQ(weightsOut, weights);

    std::vector<std::string> outputNames = physicalSgd->getOptimizerParameterNames();
    std::set<std::string> names(outputNames.begin(), outputNames.end());
    EXPECT_EQ(names, (std::set<std::string>{"weights"}));

    std::unordered_map<std::string, float> params = physicalSgd->getAllHyperParameters();
    ASSERT_EQ(params.size(), 5u);
    expectHyperParameter(params, "currentLearningRate", initialLearningRate);
    expectHyperParameter(params, "initialLearningRate", initialLearningRate);
    expectHyperParameter(params, "decay", decay);
    expectHyperParameter(params, "momentum", momentum);
    expectHyperParameter(params, "useNesterovMomentum", 0.0f);
}

TEST(SgdApi, BuilderCustomValuesStampAndCompilePhysicalSgdWithMomentum) {
    Stream stream(gpuPlacement);

    constexpr float initialLearningRate = 0.125f;
    constexpr float decay = 0.2f;
    constexpr float momentum = 0.75f;
    constexpr bool useNesterovMomentum = true;

    std::shared_ptr<Api::Sgd> sgd = Api::Sgd::Builder()
                                        .initialLearningRate(initialLearningRate)
                                        .decay(decay)
                                        .momentum(momentum)
                                        .useNesterovMomentum(useNesterovMomentum)
                                        .build();
    ASSERT_NE(sgd, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, -2.0f, 3.0f, -4.0f}, stream);

    std::shared_ptr<Impl::Sgd> physicalSgd = stampCompileSgd(*sgd, weights, stream);
    ASSERT_NE(physicalSgd, nullptr);

    EXPECT_TRUE(physicalSgd->isCompiled());
    EXPECT_EQ(physicalSgd->getId(), sgd->getId());
    EXPECT_FLOAT_EQ(physicalSgd->getInitialLearningRate(), initialLearningRate);
    EXPECT_FLOAT_EQ(physicalSgd->getDecay(), decay);
    EXPECT_FLOAT_EQ(physicalSgd->getMomentum(), momentum);
    EXPECT_EQ(physicalSgd->getUseNesterovMomentum(), useNesterovMomentum);
    EXPECT_TRUE(physicalSgd->hasParameter("momentum"));

    Impl::Tensor velocity = physicalSgd->getOptimizerParameterTensor("velocity");
    Impl::Tensor momentumStorage = getMomentumStorage(*physicalSgd);
    EXPECT_EQ(velocity, momentumStorage);
    EXPECT_EQ(momentumStorage.getPlacement(), gpuPlacement);
    EXPECT_EQ(momentumStorage.getDataType(), DataType::FP32);
    EXPECT_EQ(momentumStorage.getDimensions(), weights.getDimensions());
    expectAllClose(copyGpuFp32TensorToValues(momentumStorage, stream), {0.0f, 0.0f, 0.0f, 0.0f});

    std::vector<std::string> outputNames = physicalSgd->getOptimizerParameterNames();
    std::set<std::string> names(outputNames.begin(), outputNames.end());
    EXPECT_EQ(names, (std::set<std::string>{"weights", "velocity"}));

    std::unordered_map<std::string, float> params = physicalSgd->getAllHyperParameters();
    ASSERT_EQ(params.size(), 5u);
    expectHyperParameter(params, "currentLearningRate", initialLearningRate);
    expectHyperParameter(params, "initialLearningRate", initialLearningRate);
    expectHyperParameter(params, "decay", decay);
    expectHyperParameter(params, "momentum", momentum);
    expectHyperParameter(params, "useNesterovMomentum", 1.0f);
}

TEST(SgdApi, StampedPhysicalPlainSgdStepMatchesCpuReference) {
    Stream stream(gpuPlacement);

    const std::vector<float> initialWeights{1.0f, -2.0f, 3.0f, -4.0f};
    const std::vector<float> gradient{8.0f, -12.0f, 0.5f, -2.0f};

    constexpr uint32_t batchSize = 2;
    constexpr float initialLearningRate = 0.1f;
    constexpr float decay = 0.0f;

    std::shared_ptr<Api::Sgd> sgd =
        Api::Sgd::Builder().initialLearningRate(initialLearningRate).decay(decay).momentum(0.0f).useNesterovMomentum(false).build();
    ASSERT_NE(sgd, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    std::shared_ptr<Impl::Sgd> physicalSgd = stampCompileSgd(*sgd, weights, stream);

    SgdReferenceState expected;
    expected.weights = initialWeights;

    applyPlainSgdReferenceStep(expected, gradient, batchSize, initialLearningRate);
    runSgdStep(*physicalSgd, gradient, batchSize, stream);

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 2e-5f, 2e-5f);
}

TEST(SgdApi, StampedPhysicalMomentumSgdStepMatchesCpuReference) {
    Stream stream(gpuPlacement);

    const std::vector<float> initialWeights{2.0f, -1.0f, 3.0f, -4.0f};
    const std::vector<float> gradient1{4.0f, -8.0f, 1.0f, -2.0f};
    const std::vector<float> gradient2{-5.0f, 6.0f, -7.0f, 8.0f};

    constexpr float initialLearningRate = 0.08f;
    constexpr float decay = 0.2f;
    constexpr float momentum = 0.6f;
    constexpr bool useNesterovMomentum = true;

    std::shared_ptr<Api::Sgd> sgd = Api::Sgd::Builder()
                                        .initialLearningRate(initialLearningRate)
                                        .decay(decay)
                                        .momentum(momentum)
                                        .useNesterovMomentum(useNesterovMomentum)
                                        .build();
    ASSERT_NE(sgd, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    std::shared_ptr<Impl::Sgd> physicalSgd = stampCompileSgd(*sgd, weights, stream);
    Impl::Tensor velocity = physicalSgd->getOptimizerParameterTensor("velocity");

    SgdReferenceState expected;
    expected.weights = initialWeights;
    expected.velocity.assign(initialWeights.size(), 0.0f);

    physicalSgd->updateHyperParameters(/*epoch=*/0, /*batch=*/0, /*batchesPerEpoch=*/10);
    float currentLearningRate = computeCurrentLearningRate(initialLearningRate, decay, 0);
    applyMomentumSgdReferenceStep(expected,
                                  gradient1,
                                  /*batchSize=*/1,
                                  currentLearningRate,
                                  momentum,
                                  useNesterovMomentum);
    runSgdStep(*physicalSgd, gradient1, /*batchSize=*/1, stream);

    physicalSgd->updateHyperParameters(/*epoch=*/2, /*batch=*/5, /*batchesPerEpoch=*/10);
    currentLearningRate = computeCurrentLearningRate(initialLearningRate, decay, 2);
    applyMomentumSgdReferenceStep(expected,
                                  gradient2,
                                  /*batchSize=*/2,
                                  currentLearningRate,
                                  momentum,
                                  useNesterovMomentum);
    runSgdStep(*physicalSgd, gradient2, /*batchSize=*/2, stream);

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(velocity, stream), expected.velocity, 3e-5f, 3e-5f);
}

TEST(SgdApi, SerializeArchitectureOnlyAndDeserialize) {
    constexpr float initialLearningRate = 0.123f;
    constexpr float decay = 0.25f;
    constexpr float momentum = 0.75f;
    constexpr bool useNesterovMomentum = true;

    std::shared_ptr<Api::Sgd> sgd = Api::Sgd::Builder()
                                        .initialLearningRate(initialLearningRate)
                                        .decay(decay)
                                        .momentum(momentum)
                                        .useNesterovMomentum(useNesterovMomentum)
                                        .build();
    ASSERT_NE(sgd, nullptr);

    json j = sgd->architectureJson();

    std::shared_ptr<thor_file::TarReader> archiveReader;
    std::shared_ptr<Api::Optimizer> optimizer = Api::Optimizer::deserialize(archiveReader, j, nullptr);
    std::shared_ptr<Api::Sgd> deserializedSgd = std::dynamic_pointer_cast<Api::Sgd>(optimizer);
    ASSERT_NE(deserializedSgd, nullptr);

    EXPECT_EQ(deserializedSgd->getOriginalId(), sgd->getId());
    EXPECT_FLOAT_EQ(deserializedSgd->getInitialLearningRate(), initialLearningRate);
    EXPECT_FLOAT_EQ(deserializedSgd->getDecay(), decay);
    EXPECT_FLOAT_EQ(deserializedSgd->getMomentum(), momentum);
    EXPECT_EQ(deserializedSgd->getUseNesterovMomentum(), useNesterovMomentum);
    EXPECT_EQ(deserializedSgd->getEpoch(), 0u);

    Stream stream(gpuPlacement);
    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    std::shared_ptr<Impl::Sgd> physicalSgd = stampCompileSgd(*deserializedSgd, weights, stream);
    ASSERT_NE(physicalSgd, nullptr);

    EXPECT_EQ(physicalSgd->getId(), deserializedSgd->getId());
    EXPECT_FLOAT_EQ(physicalSgd->getInitialLearningRate(), initialLearningRate);
    EXPECT_FLOAT_EQ(physicalSgd->getDecay(), decay);
    EXPECT_FLOAT_EQ(physicalSgd->getMomentum(), momentum);
    EXPECT_EQ(physicalSgd->getUseNesterovMomentum(), useNesterovMomentum);
    EXPECT_EQ(physicalSgd->getEpoch(), 0u);
}

TEST(SgdApi, SerializeWithStateRecordsPhysicalEpoch) {
    Stream stream(gpuPlacement);

    constexpr float initialLearningRate = 0.2f;
    constexpr float decay = 0.1f;
    constexpr float momentum = 0.5f;
    constexpr bool useNesterovMomentum = false;

    std::shared_ptr<Api::Sgd> sgd = Api::Sgd::Builder()
                                        .initialLearningRate(initialLearningRate)
                                        .decay(decay)
                                        .momentum(momentum)
                                        .useNesterovMomentum(useNesterovMomentum)
                                        .build();
    ASSERT_NE(sgd, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    std::shared_ptr<Impl::Sgd> physicalSgd = stampCompileSgd(*sgd, weights, stream);
    physicalSgd->updateHyperParameters(/*epoch=*/7, /*batch=*/3, /*batchesPerEpoch=*/10);

    thor_file::TarWriter archiveWriter("sgd_api_test");
    json stateJson = sgd->serialize(archiveWriter, stream, physicalSgd, "layer123_weights", /*saveOptimizerState=*/true);

    ASSERT_EQ(stateJson.at("optimizer_type").get<std::string>(), "sgd");
    ASSERT_EQ(stateJson.at("version").get<std::string>(), sgd->getVersion());
    EXPECT_EQ(stateJson.at("id").get<uint64_t>(), sgd->getId());
    EXPECT_FLOAT_EQ(stateJson.at("initial_learning_rate").get<float>(), initialLearningRate);
    EXPECT_FLOAT_EQ(stateJson.at("decay").get<float>(), decay);
    EXPECT_FLOAT_EQ(stateJson.at("momentum").get<float>(), momentum);
    EXPECT_EQ(stateJson.at("use_nesterov").get<bool>(), useNesterovMomentum);
    EXPECT_EQ(stateJson.at("epoch").get<uint64_t>(), 7u);

    std::shared_ptr<thor_file::TarReader> archiveReader;
    std::shared_ptr<Api::Optimizer> optimizer = Api::Optimizer::deserialize(archiveReader, stateJson, nullptr);
    std::shared_ptr<Api::Sgd> deserializedSgd = std::dynamic_pointer_cast<Api::Sgd>(optimizer);
    ASSERT_NE(deserializedSgd, nullptr);

    EXPECT_EQ(deserializedSgd->getOriginalId(), sgd->getId());
    EXPECT_EQ(deserializedSgd->getEpoch(), 7u);
    EXPECT_FLOAT_EQ(deserializedSgd->getInitialLearningRate(), initialLearningRate);
    EXPECT_FLOAT_EQ(deserializedSgd->getDecay(), decay);
    EXPECT_FLOAT_EQ(deserializedSgd->getMomentum(), momentum);
    EXPECT_EQ(deserializedSgd->getUseNesterovMomentum(), useNesterovMomentum);

    thor_file::TarWriter architectureOnlyWriter("sgd_api_architecture_only_test");
    json architectureOnlyJson = sgd->serialize(architectureOnlyWriter, stream, nullptr, "", /*saveOptimizerState=*/false);

    EXPECT_EQ(architectureOnlyJson.at("epoch").get<uint64_t>(), 0u);
}
