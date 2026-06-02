#include "DeepLearning/Api/Optimizers/Adafactor.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adafactor.h"
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

Impl::Tensor requireOptimizerStorage(const std::shared_ptr<Impl::Adafactor>& adafactor, const std::string& name) {
    if (adafactor == nullptr)
        throw std::runtime_error("Adafactor optimizer is null.");

    std::shared_ptr<Impl::Optimizer> selected = adafactor->getSelectedOptimizer();
    if (selected == nullptr)
        throw std::runtime_error("Adafactor selected optimizer is null.");

    if (!selected->hasParameter(name))
        throw std::runtime_error("Adafactor optimizer is missing parameter: " + name);

    std::optional<Impl::Tensor> storage = selected->getParameter(name)->getStorage();
    if (!storage.has_value())
        throw std::runtime_error("Adafactor optimizer parameter has no storage: " + name);

    return storage.value();
}

std::shared_ptr<Impl::Adafactor> stampCompileAdafactor(Api::Adafactor& adafactor, Impl::Tensor& weights, Stream& stream) {
    std::shared_ptr<Impl::Optimizer> physicalOptimizer = adafactor.stamp(nullptr);
    std::shared_ptr<Impl::Adafactor> physicalAdafactor = std::dynamic_pointer_cast<Impl::Adafactor>(physicalOptimizer);
    if (physicalAdafactor == nullptr)
        throw std::runtime_error("Api::Adafactor did not stamp an Impl::Adafactor.");

    adafactor.compile(physicalOptimizer, weights, stream);
    stream.synchronize();

    return physicalAdafactor;
}

}  // namespace

TEST(AdafactorApi, BuilderDefaultsSettersAndArchitectureJson) {
    std::shared_ptr<Api::Adafactor> adafactor = Api::Adafactor::Builder().build();
    ASSERT_NE(adafactor, nullptr);

    EXPECT_EQ(adafactor->getType(), "Adafactor");
    EXPECT_FLOAT_EQ(adafactor->getAlpha(), 0.001f);
    EXPECT_FLOAT_EQ(adafactor->getBeta2(), 0.999f);
    EXPECT_FLOAT_EQ(adafactor->getEpsilon(), 1e-30f);
    EXPECT_FLOAT_EQ(adafactor->getWeightDecay(), 0.0f);
    EXPECT_TRUE(adafactor->getFactorSecondMoment());
    EXPECT_EQ(adafactor->getOriginalId(), adafactor->getId());

    adafactor->setAlpha(0.02f, nullptr);
    adafactor->setBeta2(0.9f, nullptr);
    adafactor->setEpsilon(1e-6f, nullptr);
    adafactor->setWeightDecay(0.01f, nullptr);
    adafactor->setFactorSecondMoment(false, nullptr);

    EXPECT_FLOAT_EQ(adafactor->getAlpha(), 0.02f);
    EXPECT_FLOAT_EQ(adafactor->getBeta2(), 0.9f);
    EXPECT_FLOAT_EQ(adafactor->getEpsilon(), 1e-6f);
    EXPECT_FLOAT_EQ(adafactor->getWeightDecay(), 0.01f);
    EXPECT_FALSE(adafactor->getFactorSecondMoment());

    json j = adafactor->architectureJson();
    ASSERT_EQ(j.at("optimizer_type").get<std::string>(), "adafactor");
    ASSERT_EQ(j.at("version").get<std::string>(), adafactor->getVersion());
    ASSERT_EQ(j.at("id").get<uint64_t>(), adafactor->getId());
    EXPECT_FLOAT_EQ(j.at("alpha").get<float>(), 0.02f);
    EXPECT_FLOAT_EQ(j.at("beta2").get<float>(), 0.9f);
    EXPECT_FLOAT_EQ(j.at("epsilon").get<float>(), 1e-6f);
    EXPECT_FLOAT_EQ(j.at("weight_decay").get<float>(), 0.01f);
    EXPECT_FALSE(j.at("factor_second_moment").get<bool>());
}

TEST(AdafactorApi, BuilderCustomValuesStampAndCompilePhysicalFactoredAdafactor) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.03f;
    constexpr float beta2 = 0.8f;
    constexpr float epsilon = 1e-6f;
    constexpr float weightDecay = 0.02f;

    std::shared_ptr<Api::Adafactor> adafactor = Api::Adafactor::Builder()
                                                    .alpha(alpha)
                                                    .beta2(beta2)
                                                    .epsilon(epsilon)
                                                    .weightDecay(weightDecay)
                                                    .factorSecondMoment(true)
                                                    .build();
    ASSERT_NE(adafactor, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f}, stream);

    std::shared_ptr<Impl::Adafactor> physicalAdafactor = stampCompileAdafactor(*adafactor, weights, stream);
    ASSERT_NE(physicalAdafactor, nullptr);

    EXPECT_TRUE(physicalAdafactor->isCompiled());
    EXPECT_TRUE(physicalAdafactor->isUsingFactoredPath());
    EXPECT_EQ(physicalAdafactor->getId(), adafactor->getId());
    EXPECT_FLOAT_EQ(physicalAdafactor->getAlpha(), alpha);
    EXPECT_FLOAT_EQ(physicalAdafactor->getBeta2(), beta2);
    EXPECT_FLOAT_EQ(physicalAdafactor->getEpsilon(), epsilon);
    EXPECT_FLOAT_EQ(physicalAdafactor->getWeightDecay(), weightDecay);
    EXPECT_TRUE(physicalAdafactor->getFactorSecondMoment());

    std::optional<Impl::Tensor> gradient = physicalAdafactor->getWeightsGradient();
    ASSERT_TRUE(gradient.has_value());
    EXPECT_EQ(gradient.value().getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.value().getDataType(), DataType::FP32);
    EXPECT_EQ(gradient.value().getDimensions(), weights.getDimensions());

    Impl::Tensor rowSecondMoment = requireOptimizerStorage(physicalAdafactor, "row_second_moment");
    Impl::Tensor columnSecondMoment = requireOptimizerStorage(physicalAdafactor, "column_second_moment");

    EXPECT_EQ(rowSecondMoment.getPlacement(), gpuPlacement);
    EXPECT_EQ(rowSecondMoment.getDataType(), DataType::FP32);
    EXPECT_EQ(rowSecondMoment.getDimensions(), (std::vector<uint64_t>{2, 1}));
    EXPECT_EQ(columnSecondMoment.getPlacement(), gpuPlacement);
    EXPECT_EQ(columnSecondMoment.getDataType(), DataType::FP32);
    EXPECT_EQ(columnSecondMoment.getDimensions(), (std::vector<uint64_t>{1, 3}));

    std::unordered_map<std::string, float> params = physicalAdafactor->getAllHyperParameters();
    ASSERT_EQ(params.size(), 5u);
    expectHyperParameter(params, "alpha", alpha);
    expectHyperParameter(params, "beta2", beta2);
    expectHyperParameter(params, "epsilon", epsilon);
    expectHyperParameter(params, "weightDecay", weightDecay);
    expectHyperParameter(params, "factorSecondMoment", 1.0f);
}

TEST(AdafactorApi, DenseVectorOrDisabledFactoringUsesUnfactoredState) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::Adafactor> adafactor = Api::Adafactor::Builder().factorSecondMoment(false).build();
    ASSERT_NE(adafactor, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, stream);

    std::shared_ptr<Impl::Adafactor> physicalAdafactor = stampCompileAdafactor(*adafactor, weights, stream);
    ASSERT_NE(physicalAdafactor, nullptr);
    ASSERT_TRUE(physicalAdafactor->isUsingUnfactoredPath());

    Impl::Tensor secondMoment = requireOptimizerStorage(physicalAdafactor, "second_moment");
    EXPECT_EQ(secondMoment.getPlacement(), gpuPlacement);
    EXPECT_EQ(secondMoment.getDataType(), DataType::FP32);
    EXPECT_EQ(secondMoment.getDimensions(), weights.getDimensions());
}

TEST(AdafactorApi, InitializeFirstStampZerosFactoredStateParameters) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::Adafactor> adafactor = Api::Adafactor::Builder().epsilon(1e-6f).build();
    ASSERT_NE(adafactor, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    std::shared_ptr<Impl::Adafactor> physicalAdafactor = stampCompileAdafactor(*adafactor, weights, stream);

    Impl::Tensor rowSecondMoment = requireOptimizerStorage(physicalAdafactor, "row_second_moment");
    Impl::Tensor columnSecondMoment = requireOptimizerStorage(physicalAdafactor, "column_second_moment");
    copyValuesToGpuFp32Tensor(rowSecondMoment, {1.0f, -2.0f}, stream);
    copyValuesToGpuFp32Tensor(columnSecondMoment, {-1.0f, 2.0f}, stream);

    synchronizeEvents(adafactor->initialize(physicalAdafactor, /*isFirstStamp=*/true, nullptr, std::nullopt));
    expectAllClose(copyGpuFp32TensorToValues(rowSecondMoment, stream), {0.0f, 0.0f});
    expectAllClose(copyGpuFp32TensorToValues(columnSecondMoment, stream), {0.0f, 0.0f});
}

TEST(AdafactorApi, InitializeCopiesStateFromSisterOptimizer) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::Adafactor> adafactor = Api::Adafactor::Builder().epsilon(1e-6f).build();
    ASSERT_NE(adafactor, nullptr);

    Impl::Tensor weights1(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    Impl::Tensor weights2(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights1, {1.0f, 2.0f, 3.0f, 4.0f}, stream);
    copyValuesToGpuFp32Tensor(weights2, {-1.0f, -2.0f, -3.0f, -4.0f}, stream);

    std::shared_ptr<Impl::Adafactor> sister = stampCompileAdafactor(*adafactor, weights1, stream);
    std::shared_ptr<Impl::Adafactor> physical = stampCompileAdafactor(*adafactor, weights2, stream);

    Impl::Tensor sisterRow = requireOptimizerStorage(sister, "row_second_moment");
    Impl::Tensor sisterColumn = requireOptimizerStorage(sister, "column_second_moment");
    Impl::Tensor row = requireOptimizerStorage(physical, "row_second_moment");
    Impl::Tensor column = requireOptimizerStorage(physical, "column_second_moment");

    copyValuesToGpuFp32Tensor(sisterRow, {0.1f, 0.2f}, stream);
    copyValuesToGpuFp32Tensor(sisterColumn, {0.3f, 0.4f}, stream);
    copyValuesToGpuFp32Tensor(row, {9.0f, 9.0f}, stream);
    copyValuesToGpuFp32Tensor(column, {8.0f, 8.0f}, stream);

    synchronizeEvents(adafactor->initialize(physical, /*isFirstStamp=*/false, sister, std::nullopt));
    expectAllClose(copyGpuFp32TensorToValues(row, stream), {0.1f, 0.2f});
    expectAllClose(copyGpuFp32TensorToValues(column, stream), {0.3f, 0.4f});
}
