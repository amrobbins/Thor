#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/WassersteinGANCriticLoss.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kRealScoresName = "real_scores";
constexpr const char* kFakeScoresName = "fake_scores";
constexpr const char* kLossName = "loss";
constexpr const char* kRealScoresGradientName = "real_scores_grad";
constexpr const char* kFakeScoresGradientName = "fake_scores_grad";

void validateScoreDType(const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported WassersteinGANCriticLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateWassersteinGANCriticDTypes(DataType realScoresDType, DataType fakeScoresDType) {
    validateScoreDType("real_scores", realScoresDType);
    validateScoreDType("fake_scores", fakeScoresDType);
    if (realScoresDType != fakeScoresDType)
        throw runtime_error("WassersteinGANCriticLoss real_scores and fake_scores tensors must have the same dtype.");
}

ThorImplementation::DynamicExpression makeWassersteinGANCriticLossExpression(DataType lossDataType) {
    validateScoreDType("loss", lossDataType);

    ThorImplementation::Expression realScores = ThorImplementation::Expression::input(kRealScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression fakeScores = ThorImplementation::Expression::input(kFakeScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression loss = (fakeScores - realScores).withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWassersteinGANCriticGradientExpression(DataType scoresDataType) {
    validateScoreDType("scores", scoresDataType);

    ThorImplementation::Expression realScores = ThorImplementation::Expression::input(kRealScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression fakeScores = ThorImplementation::Expression::input(kFakeScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression scale(ThorImplementation::Loss::getLossScalingFactor());
    ThorImplementation::Expression realGradient = ((realScores * ThorImplementation::Expression(0.0)) - scale).withOutputDType(scoresDataType);
    ThorImplementation::Expression fakeGradient = ((fakeScores * ThorImplementation::Expression(0.0)) + scale).withOutputDType(scoresDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kRealScoresGradientName, realGradient}, {kFakeScoresGradientName, fakeGradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void WassersteinGANCriticLoss::buildSupportLayersAndAddToNetwork() {
    validateWassersteinGANCriticDTypes(realScoresTensor.getDataType(), fakeScoresTensor.getDataType());
    THOR_THROW_IF_FALSE(realScoresTensor.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(realScoresTensor.getDimensions()[0] > 0);
    THOR_THROW_IF_FALSE(realScoresTensor.getDimensions() == fakeScoresTensor.getDimensions());

    MultiInputCustomLoss rawWassersteinGANCriticLoss = MultiInputCustomLoss::Builder()
                                                           .network(*network)
                                                           .lossExpression(makeWassersteinGANCriticLossExpression(lossDataType))
                                                           .gradientExpression(
                                                               makeWassersteinGANCriticGradientExpression(realScoresTensor.getDataType()))
                                                           .input(kRealScoresName, realScoresTensor, kRealScoresGradientName)
                                                           .input(kFakeScoresName, fakeScoresTensor, kFakeScoresGradientName)
                                                           .lossName(kLossName)
                                                           .lossDataType(lossDataType)
                                                           .reportsRawLoss()
                                                           .build();

    lossShaperInput = rawWassersteinGANCriticLoss.getLoss();

    if (lossShape == LossShape::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::ELEMENTWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossShape == LossShape::CLASSWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossShaperInput).reportsClasswiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        THOR_THROW_IF_FALSE(lossShape == LossShape::RAW);
        lossTensor = lossShaperInput;
    }
}

json WassersteinGANCriticLoss::architectureJson() const {
    json j;
    j["factory"] = Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "wasserstein_gan_critic_loss";
    j["layer_name"] = string("layer") + to_string(getId());
    j["loss_shape"] = lossShape;
    j["loss_data_type"] = lossDataType;
    j["real_scores_tensor"] = realScoresTensor.architectureJson();
    j["fake_scores_tensor"] = fakeScoresTensor.architectureJson();
    j["loss_shaper_input_tensor"] = lossShaperInput.architectureJson();
    j["loss_tensor"] = lossTensor.architectureJson();
    return j;
}

void WassersteinGANCriticLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in WassersteinGANCriticLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "wasserstein_gan_critic_loss")
        throw runtime_error("Layer type mismatch in WassersteinGANCriticLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["real_scores_tensor"].at("id").get<uint64_t>();
    Tensor realScores = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["fake_scores_tensor"].at("id").get<uint64_t>();
    Tensor fakeScores = network->getApiTensorByOriginalId(originalTensorId);

    WassersteinGANCriticLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();
    loss.realScoresTensor = realScores;
    loss.fakeScoresTensor = fakeScores;
    loss.network = network;
    loss.initialized = true;
    loss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("wasserstein_gan_critic_loss", &Thor::WassersteinGANCriticLoss::deserialize);
    return true;
}();
}  // namespace
