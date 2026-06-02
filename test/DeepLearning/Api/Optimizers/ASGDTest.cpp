#include "DeepLearning/Api/Optimizers/ASGD.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/ASGD.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
Impl::TensorPlacement gpuPlacement(Impl::TensorPlacement::MemDevices::GPU, 0);

using DataType = Impl::DataType;

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

void expectAllClose(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 1e-5f, float rtol = 1e-5f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        const float tol = atol + rtol * std::fabs(expected[i]);
        EXPECT_LE(diff, tol) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

std::shared_ptr<Impl::ASGD> stampCompileASGD(Api::ASGD& asgd, Impl::Tensor& weights, Stream& stream) {
    std::shared_ptr<Impl::Optimizer> physicalOptimizer = asgd.stamp(nullptr);
    std::shared_ptr<Impl::ASGD> physicalASGD = std::dynamic_pointer_cast<Impl::ASGD>(physicalOptimizer);
    if (physicalASGD == nullptr)
        throw std::runtime_error("Api::ASGD did not stamp an Impl::ASGD.");

    asgd.compile(physicalOptimizer, weights, stream);
    stream.synchronize();

    return physicalASGD;
}

Impl::Tensor requireOptimizerStorage(const std::shared_ptr<Impl::ASGD>& asgd, const std::string& name) {
    if (!asgd->hasParameter(name))
        throw std::runtime_error("ASGD optimizer missing parameter: " + name);
    std::optional<Impl::Tensor> storage = asgd->getParameter(name)->getStorage();
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
    const float invBatchLossScale = 1.0f / (static_cast<float>(batchSize) * Impl::Loss::getLossScalingFactor());
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

void runASGDStep(Impl::ASGD& asgd, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    std::optional<Impl::Tensor> gradientOpt = asgd.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.has_value());
    Impl::Tensor gradient = gradientOpt.value();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);
    asgd.updateWeights(batchSize);
    stream.synchronize();
}

}  // namespace

TEST(ASGDApi, BuilderDefaultsSettersAndArchitectureJson) {
    std::shared_ptr<Api::ASGD> asgd = Api::ASGD::Builder().build();
    ASSERT_NE(asgd, nullptr);

    EXPECT_EQ(asgd->getType(), "ASGD");
    EXPECT_FLOAT_EQ(asgd->getAlpha(), 0.01f);
    EXPECT_FLOAT_EQ(asgd->getLambd(), 1e-4f);
    EXPECT_FLOAT_EQ(asgd->getPower(), 0.75f);
    EXPECT_FLOAT_EQ(asgd->getT0(), 1e6f);
    EXPECT_FLOAT_EQ(asgd->getWeightDecay(), 0.0f);
    EXPECT_FLOAT_EQ(asgd->getT(), 0.0f);
    EXPECT_EQ(asgd->getOriginalId(), asgd->getId());

    asgd->setAlpha(0.2f, nullptr);
    asgd->setLambd(0.01f, nullptr);
    asgd->setPower(0.5f, nullptr);
    asgd->setT0(2.0f, nullptr);
    asgd->setWeightDecay(0.03f, nullptr);

    EXPECT_FLOAT_EQ(asgd->getAlpha(), 0.2f);
    EXPECT_FLOAT_EQ(asgd->getLambd(), 0.01f);
    EXPECT_FLOAT_EQ(asgd->getPower(), 0.5f);
    EXPECT_FLOAT_EQ(asgd->getT0(), 2.0f);
    EXPECT_FLOAT_EQ(asgd->getWeightDecay(), 0.03f);

    json j = asgd->architectureJson();
    ASSERT_EQ(j.at("optimizer_type").get<std::string>(), "asgd");
    ASSERT_EQ(j.at("version").get<std::string>(), asgd->getVersion());
    ASSERT_EQ(j.at("id").get<uint64_t>(), asgd->getId());
    EXPECT_FLOAT_EQ(j.at("t").get<float>(), 0.0f);
    EXPECT_FLOAT_EQ(j.at("alpha").get<float>(), 0.2f);
    EXPECT_FLOAT_EQ(j.at("lambd").get<float>(), 0.01f);
    EXPECT_FLOAT_EQ(j.at("power").get<float>(), 0.5f);
    EXPECT_FLOAT_EQ(j.at("t0").get<float>(), 2.0f);
    EXPECT_FLOAT_EQ(j.at("weight_decay").get<float>(), 0.03f);
}

TEST(ASGDApi, BuilderCustomValuesStampCompileAndRunPhysicalASGD) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.2f;
    constexpr float lambd = 0.01f;
    constexpr float power = 0.5f;
    constexpr float t0 = 2.0f;
    constexpr float weightDecay = 0.03f;

    std::shared_ptr<Api::ASGD> asgd =
        Api::ASGD::Builder().alpha(alpha).lambd(lambd).power(power).t0(t0).weightDecay(weightDecay).build();
    ASSERT_NE(asgd, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    const std::vector<float> initialWeights{1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    std::shared_ptr<Impl::ASGD> physicalASGD = stampCompileASGD(*asgd, weights, stream);
    ASSERT_NE(physicalASGD, nullptr);

    EXPECT_TRUE(physicalASGD->isCompiled());
    EXPECT_EQ(physicalASGD->getId(), asgd->getId());
    EXPECT_FLOAT_EQ(physicalASGD->getAlpha(), alpha);
    EXPECT_FLOAT_EQ(physicalASGD->getLambd(), lambd);
    EXPECT_FLOAT_EQ(physicalASGD->getPower(), power);
    EXPECT_FLOAT_EQ(physicalASGD->getT0(), t0);
    EXPECT_FLOAT_EQ(physicalASGD->getWeightDecay(), weightDecay);

    Impl::Tensor averagedWeights = requireOptimizerStorage(physicalASGD, "averaged_weights");
    copyValuesToGpuFp32Tensor(averagedWeights, std::vector<float>(initialWeights.size(), 0.0f), stream);

    ASGDReferenceState reference{initialWeights, std::vector<float>(initialWeights.size(), 0.0f), 0.0f};
    const std::vector<float> gradient1{0.5f, -0.25f, 1.0f, -1.5f, 0.75f, -0.5f};
    const std::vector<float> gradient2{-0.1f, 0.2f, -0.3f, 0.4f, -0.5f, 0.6f};

    applyASGDReferenceStep(reference, gradient1, 2, alpha, lambd, power, t0, weightDecay);
    runASGDStep(*physicalASGD, gradient1, 2, stream);
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), reference.weights, 2e-5f, 2e-5f);
    expectAllClose(copyGpuFp32TensorToValues(averagedWeights, stream), reference.averagedWeights, 2e-5f, 2e-5f);

    applyASGDReferenceStep(reference, gradient2, 4, alpha, lambd, power, t0, weightDecay);
    runASGDStep(*physicalASGD, gradient2, 4, stream);
    expectAllClose(copyGpuFp32TensorToValues(weights, stream), reference.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(averagedWeights, stream), reference.averagedWeights, 3e-5f, 3e-5f);
    EXPECT_FLOAT_EQ(physicalASGD->getT(), 2.0f);
}
