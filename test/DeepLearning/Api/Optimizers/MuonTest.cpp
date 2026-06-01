#include "DeepLearning/Api/Optimizers/AdamW.h"
#include "DeepLearning/Api/Optimizers/Muon.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/Optimizers/AdamW.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Muon.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include "gtest/gtest.h"

#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>

using json = nlohmann::json;
namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

Impl::TensorPlacement gpuPlacement(Impl::TensorPlacement::MemDevices::GPU, 0);
using DataType = Impl::DataType;

void expectHyperParameter(const std::unordered_map<std::string, float>& parameters, const std::string& name, float expected) {
    ASSERT_TRUE(parameters.contains(name)) << "Missing hyperparameter: " << name;
    EXPECT_FLOAT_EQ(parameters.at(name), expected) << "Mismatch for hyperparameter: " << name;
}

std::shared_ptr<Impl::Muon> stampCompileMuon(Api::Muon& muon, Impl::Tensor& weights, Stream& stream) {
    std::shared_ptr<Impl::Optimizer> physicalOptimizer = muon.stamp(nullptr);
    std::shared_ptr<Impl::Muon> physicalMuon = std::dynamic_pointer_cast<Impl::Muon>(physicalOptimizer);
    if (physicalMuon == nullptr)
        throw std::runtime_error("Api::Muon did not stamp an Impl::Muon.");

    muon.compile(physicalOptimizer, weights, stream);
    stream.synchronize();
    return physicalMuon;
}

}  // namespace

TEST(MuonApi, BuilderDefaultsUseAdamWFallbackAndArchitectureJson) {
    std::shared_ptr<Api::Muon> muon = Api::Muon::Builder().build();
    ASSERT_NE(muon, nullptr);

    EXPECT_EQ(muon->getType(), "Muon");
    EXPECT_FLOAT_EQ(muon->getAlpha(), 0.02f);
    EXPECT_FLOAT_EQ(muon->getBeta(), 0.95f);
    EXPECT_FLOAT_EQ(muon->getEpsilon(), 1.0e-8f);
    EXPECT_FLOAT_EQ(muon->getWeightDecay(), 0.0f);
    EXPECT_TRUE(muon->getNesterov());
    EXPECT_EQ(muon->getNumIterations(), 5u);
    ASSERT_NE(muon->getFallbackOptimizer(), nullptr);
    EXPECT_EQ(muon->getFallbackOptimizer()->getType(), "AdamW");

    json j = muon->architectureJson();
    ASSERT_EQ(j.at("optimizer_type").get<std::string>(), "muon");
    ASSERT_EQ(j.at("version").get<std::string>(), muon->getVersion());
    ASSERT_EQ(j.at("id").get<uint64_t>(), muon->getId());
    EXPECT_FLOAT_EQ(j.at("alpha").get<float>(), 0.02f);
    EXPECT_FLOAT_EQ(j.at("beta").get<float>(), 0.95f);
    EXPECT_FLOAT_EQ(j.at("epsilon").get<float>(), 1.0e-8f);
    EXPECT_FLOAT_EQ(j.at("weight_decay").get<float>(), 0.0f);
    EXPECT_TRUE(j.at("nesterov").get<bool>());
    EXPECT_EQ(j.at("num_iterations").get<uint32_t>(), 5u);
    ASSERT_TRUE(j.contains("fallback_optimizer"));
    EXPECT_EQ(j.at("fallback_optimizer").at("optimizer_type").get<std::string>(), "adamw");
}

TEST(MuonApi, BuilderCustomValuesAndFallbackStampCorrectly) {
    Stream stream(gpuPlacement);

    std::shared_ptr<Api::Sgd> fallback = Api::Sgd::Builder().initialLearningRate(0.04f).build();
    std::shared_ptr<Api::Muon> muon = Api::Muon::Builder()
                                           .alpha(0.03f)
                                           .beta(0.8f)
                                           .epsilon(1.0e-6f)
                                           .weightDecay(0.02f)
                                           .nesterov(false)
                                           .numIterations(3)
                                           .coefficientA(3.0f)
                                           .coefficientB(-4.0f)
                                           .coefficientC(2.0f)
                                           .transposeTallMatrices(false)
                                           .fallbackOptimizer(fallback)
                                           .build();
    ASSERT_NE(muon, nullptr);

    json j = muon->architectureJson();
    EXPECT_EQ(j.at("fallback_optimizer").at("optimizer_type").get<std::string>(), "sgd");
    EXPECT_FLOAT_EQ(j.at("alpha").get<float>(), 0.03f);
    EXPECT_FLOAT_EQ(j.at("beta").get<float>(), 0.8f);
    EXPECT_FLOAT_EQ(j.at("epsilon").get<float>(), 1.0e-6f);
    EXPECT_FLOAT_EQ(j.at("weight_decay").get<float>(), 0.02f);
    EXPECT_FALSE(j.at("nesterov").get<bool>());
    EXPECT_EQ(j.at("num_iterations").get<uint32_t>(), 3u);
    EXPECT_FLOAT_EQ(j.at("coefficient_a").get<float>(), 3.0f);
    EXPECT_FLOAT_EQ(j.at("coefficient_b").get<float>(), -4.0f);
    EXPECT_FLOAT_EQ(j.at("coefficient_c").get<float>(), 2.0f);
    EXPECT_FALSE(j.at("transpose_tall_matrices").get<bool>());

    Impl::Tensor matrixWeights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 3}));
    std::shared_ptr<Impl::Muon> matrixPhysical = stampCompileMuon(*muon, matrixWeights, stream);
    EXPECT_TRUE(matrixPhysical->isUsingMuonMatrixPath());
    ASSERT_NE(matrixPhysical->getSelectedOptimizer(), nullptr);
    EXPECT_TRUE(matrixPhysical->getSelectedOptimizer()->hasParameter("momentum"));

    Impl::Tensor vectorWeights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {3}));
    std::shared_ptr<Impl::Muon> fallbackPhysical = stampCompileMuon(*muon, vectorWeights, stream);
    EXPECT_TRUE(fallbackPhysical->isUsingFallbackPath());
    ASSERT_NE(fallbackPhysical->getSelectedOptimizer(), nullptr);
    std::unordered_map<std::string, float> fallbackParams = fallbackPhysical->getSelectedOptimizer()->getAllHyperParameters();
    expectHyperParameter(fallbackParams, "initialLearningRate", 0.04f);
}

TEST(MuonApi, PhysicalMatrixHyperParameterSnapshotMatchesBuilder) {
    Stream stream(gpuPlacement);
    std::shared_ptr<Api::Muon> muon = Api::Muon::Builder().alpha(0.05f).beta(0.7f).epsilon(1e-5f).weightDecay(0.03f).build();

    Impl::Tensor weights(gpuPlacement, Impl::TensorDescriptor(DataType::FP32, {2, 2}));
    std::shared_ptr<Impl::Muon> physical = stampCompileMuon(*muon, weights, stream);
    ASSERT_TRUE(physical->isUsingMuonMatrixPath());

    std::unordered_map<std::string, float> params = physical->getAllHyperParameters();
    expectHyperParameter(params, "alpha", 0.05f);
    expectHyperParameter(params, "beta", 0.7f);
    expectHyperParameter(params, "epsilon", 1e-5f);
    expectHyperParameter(params, "weightDecay", 0.03f);
    expectHyperParameter(params, "nesterov", 1.0f);
}
