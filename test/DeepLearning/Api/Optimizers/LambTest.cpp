#include "DeepLearning/Api/Optimizers/Lamb.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Lamb.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
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

Impl::Tensor requireOptimizerStorage(const std::shared_ptr<Impl::Lamb>& lamb, const std::string& name) {
    if (lamb == nullptr)
        throw std::runtime_error("Lamb optimizer is null.");
    if (!lamb->hasParameter(name))
        throw std::runtime_error("Lamb optimizer is missing parameter: " + name);
    std::optional<Impl::Tensor> storage = lamb->getParameter(name)->getStorage();
    if (!storage.has_value())
        throw std::runtime_error("Lamb optimizer parameter has no storage: " + name);
    return storage.value();
}

std::shared_ptr<Impl::Lamb> stampCompileLamb(Api::Lamb& lamb, Impl::Tensor& weights, Stream& stream) {
    std::shared_ptr<Impl::Optimizer> physicalOptimizer = lamb.stamp(nullptr);
    std::shared_ptr<Impl::Lamb> physicalLamb = std::dynamic_pointer_cast<Impl::Lamb>(physicalOptimizer);
    if (physicalLamb == nullptr)
        throw std::runtime_error("Api::Lamb did not stamp an Impl::Lamb.");

    lamb.compile(physicalOptimizer, weights, stream);
    stream.synchronize();
    return physicalLamb;
}

}  // namespace

TEST(LambApi, BuilderDefaultsSettersAndArchitectureJson) {
    std::shared_ptr<Api::Lamb> lamb = Api::Lamb::Builder().build();
    ASSERT_NE(lamb, nullptr);

    EXPECT_EQ(lamb->getType(), "Lamb");
    EXPECT_FLOAT_EQ(lamb->getAlpha(), 0.001f);
    EXPECT_FLOAT_EQ(lamb->getBeta1(), 0.9f);
    EXPECT_FLOAT_EQ(lamb->getBeta2(), 0.999f);
    EXPECT_FLOAT_EQ(lamb->getEpsilon(), 1e-6f);
    EXPECT_FLOAT_EQ(lamb->getWeightDecay(), 0.01f);
    EXPECT_FLOAT_EQ(lamb->getTrustRatioEpsilon(), 1e-6f);
    EXPECT_EQ(lamb->getOriginalId(), lamb->getId());

    lamb->setAlpha(0.02f, nullptr);
    lamb->setBeta1(0.8f, nullptr);
    lamb->setBeta2(0.99f, nullptr);
    lamb->setEpsilon(1e-5f, nullptr);
    lamb->setWeightDecay(0.02f, nullptr);
    lamb->setTrustRatioEpsilon(1e-5f, nullptr);

    json j = lamb->architectureJson();
    ASSERT_EQ(j.at("optimizer_type").get<std::string>(), "lamb");
    ASSERT_EQ(j.at("version").get<std::string>(), lamb->getVersion());
    ASSERT_EQ(j.at("id").get<uint64_t>(), lamb->getId());
    EXPECT_FLOAT_EQ(j.at("t").get<float>(), 0.0f);
    EXPECT_FLOAT_EQ(j.at("alpha").get<float>(), 0.02f);
    EXPECT_FLOAT_EQ(j.at("beta1").get<float>(), 0.8f);
    EXPECT_FLOAT_EQ(j.at("beta2").get<float>(), 0.99f);
    EXPECT_FLOAT_EQ(j.at("epsilon").get<float>(), 1e-5f);
    EXPECT_FLOAT_EQ(j.at("weight_decay").get<float>(), 0.02f);
    EXPECT_FLOAT_EQ(j.at("trust_ratio_epsilon").get<float>(), 1e-5f);
}

TEST(LambApi, BuilderCustomValuesStampAndCompilePhysicalLamb) {
    Stream stream(gpuPlacement);

    constexpr float alpha = 0.03f;
    constexpr float beta1 = 0.8f;
    constexpr float beta2 = 0.97f;
    constexpr float epsilon = 1e-5f;
    constexpr float weightDecay = 0.04f;
    constexpr float trustRatioEpsilon = 1e-4f;

    std::shared_ptr<Api::Lamb> lamb = Api::Lamb::Builder()
                                          .alpha(alpha)
                                          .beta1(beta1)
                                          .beta2(beta2)
                                          .epsilon(epsilon)
                                          .weightDecay(weightDecay)
                                          .trustRatioEpsilon(trustRatioEpsilon)
                                          .build();
    ASSERT_NE(lamb, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f}, stream);

    std::shared_ptr<Impl::Lamb> physicalLamb = stampCompileLamb(*lamb, weights, stream);
    ASSERT_NE(physicalLamb, nullptr);

    EXPECT_TRUE(physicalLamb->isCompiled());
    EXPECT_EQ(physicalLamb->getId(), lamb->getId());
    EXPECT_FLOAT_EQ(physicalLamb->getT(), 0.0f);
    EXPECT_FLOAT_EQ(physicalLamb->getAlpha(), alpha);
    EXPECT_FLOAT_EQ(physicalLamb->getBeta1(), beta1);
    EXPECT_FLOAT_EQ(physicalLamb->getBeta2(), beta2);
    EXPECT_FLOAT_EQ(physicalLamb->getEpsilon(), epsilon);
    EXPECT_FLOAT_EQ(physicalLamb->getWeightDecay(), weightDecay);
    EXPECT_FLOAT_EQ(physicalLamb->getTrustRatioEpsilon(), trustRatioEpsilon);

    std::optional<Impl::Tensor> gradient = physicalLamb->getWeightsGradient();
    ASSERT_TRUE(gradient.has_value());
    EXPECT_EQ(gradient.value().getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.value().getDataType(), DataType::FP32);
    EXPECT_EQ(gradient.value().getDimensions(), weights.getDimensions());

    Impl::Tensor m = requireOptimizerStorage(physicalLamb, "m");
    Impl::Tensor v = requireOptimizerStorage(physicalLamb, "v");
    EXPECT_EQ(m.getPlacement(), gpuPlacement);
    EXPECT_EQ(m.getDataType(), DataType::FP32);
    EXPECT_EQ(m.getDimensions(), weights.getDimensions());
    EXPECT_EQ(v.getPlacement(), gpuPlacement);
    EXPECT_EQ(v.getDataType(), DataType::FP32);
    EXPECT_EQ(v.getDimensions(), weights.getDimensions());

    std::unordered_map<std::string, float> params = physicalLamb->getAllHyperParameters();
    ASSERT_EQ(params.size(), 7u);
    EXPECT_FLOAT_EQ(params.at("t"), 0.0f);
    EXPECT_FLOAT_EQ(params.at("alpha"), alpha);
    EXPECT_FLOAT_EQ(params.at("beta1"), beta1);
    EXPECT_FLOAT_EQ(params.at("beta2"), beta2);
    EXPECT_FLOAT_EQ(params.at("epsilon"), epsilon);
    EXPECT_FLOAT_EQ(params.at("weightDecay"), weightDecay);
    EXPECT_FLOAT_EQ(params.at("trustRatioEpsilon"), trustRatioEpsilon);
}

TEST(LambApi, InitializeFirstStampZerosMomentParameters) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::Lamb> lamb = Api::Lamb::Builder().alpha(0.03f).beta1(0.8f).beta2(0.97f).epsilon(1e-5f).build();
    ASSERT_NE(lamb, nullptr);

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    std::shared_ptr<Impl::Lamb> physicalLamb = stampCompileLamb(*lamb, weights, stream);
    Impl::Tensor m = requireOptimizerStorage(physicalLamb, "m");
    Impl::Tensor v = requireOptimizerStorage(physicalLamb, "v");
    copyValuesToGpuFp32Tensor(m, {1.0f, -2.0f, 3.0f, -4.0f}, stream);
    copyValuesToGpuFp32Tensor(v, {-1.0f, 2.0f, -3.0f, 4.0f}, stream);

    synchronizeEvents(lamb->initialize(physicalLamb, true, nullptr, std::nullopt));

    std::vector<float> expected(4, 0.0f);
    EXPECT_EQ(copyGpuFp32TensorToValues(m, stream), expected);
    EXPECT_EQ(copyGpuFp32TensorToValues(v, stream), expected);
}

TEST(LambApi, DeserializeRoundTripArchitectureJson) {
    std::shared_ptr<Api::Lamb> lamb =
        Api::Lamb::Builder().alpha(0.02f).beta1(0.8f).beta2(0.97f).epsilon(1e-5f).weightDecay(0.04f).trustRatioEpsilon(1e-4f).build();
    ASSERT_NE(lamb, nullptr);

    json j = lamb->architectureJson();
    std::shared_ptr<thor_file::TarReader> reader;
    std::shared_ptr<Api::Optimizer> restoredBase = Api::Optimizer::deserialize(reader, j, nullptr);
    std::shared_ptr<Api::Lamb> restored = std::dynamic_pointer_cast<Api::Lamb>(restoredBase);
    ASSERT_NE(restored, nullptr);

    EXPECT_EQ(restored->getOriginalId(), lamb->getId());
    EXPECT_FLOAT_EQ(restored->getAlpha(), 0.02f);
    EXPECT_FLOAT_EQ(restored->getBeta1(), 0.8f);
    EXPECT_FLOAT_EQ(restored->getBeta2(), 0.97f);
    EXPECT_FLOAT_EQ(restored->getEpsilon(), 1e-5f);
    EXPECT_FLOAT_EQ(restored->getWeightDecay(), 0.04f);
    EXPECT_FLOAT_EQ(restored->getTrustRatioEpsilon(), 1e-4f);
}
