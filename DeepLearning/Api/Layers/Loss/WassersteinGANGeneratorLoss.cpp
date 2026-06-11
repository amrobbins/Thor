#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/WassersteinGANGeneratorLoss.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kFakeScoresName = "fake_scores";
constexpr const char* kLossName = "loss";
constexpr const char* kFakeScoresGradientName = "fake_scores_grad";

void validateScoreDType(const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported WassersteinGANGeneratorLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

ThorImplementation::DynamicExpression makeWassersteinGANGeneratorLossExpression(DataType lossDataType) {
    validateScoreDType("loss", lossDataType);

    ThorImplementation::Expression fakeScores = ThorImplementation::Expression::input(kFakeScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression loss = (-fakeScores).withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWassersteinGANGeneratorGradientExpression(DataType scoresDataType) {
    validateScoreDType("scores", scoresDataType);

    ThorImplementation::Expression fakeScores = ThorImplementation::Expression::input(kFakeScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression scale(ThorImplementation::Loss::getLossScalingFactor());
    ThorImplementation::Expression gradient = ((fakeScores * ThorImplementation::Expression(0.0)) - scale).withOutputDType(scoresDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kFakeScoresGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void WassersteinGANGeneratorLoss::buildSupportLayersAndAddToNetwork() {
    validateScoreDType("fake_scores", fakeScoresTensor.getDataType());
    THOR_THROW_IF_FALSE(fakeScoresTensor.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(fakeScoresTensor.getDimensions()[0] > 0);

    MultiInputCustomLoss rawWassersteinGANGeneratorLoss = MultiInputCustomLoss::Builder()
                                                              .network(*network)
                                                              .lossExpression(makeWassersteinGANGeneratorLossExpression(lossDataType))
                                                              .gradientExpression(makeWassersteinGANGeneratorGradientExpression(
                                                                  fakeScoresTensor.getDataType()))
                                                              .input(kFakeScoresName, fakeScoresTensor, kFakeScoresGradientName)
                                                              .lossName(kLossName)
                                                              .lossDataType(lossDataType)
                                       .lossWeight(lossWeight.value_or(1.0f))
                                                              .reportsRawLoss()
                                                              .build();

    lossShaperInput = rawWassersteinGANGeneratorLoss.getLoss();

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

json WassersteinGANGeneratorLoss::architectureJson() const {
    json j;
    j["factory"] = Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "wasserstein_gan_generator_loss";
    j["layer_name"] = string("layer") + to_string(getId());
    j["loss_shape"] = lossShape;
    j["loss_data_type"] = lossDataType;
    j["fake_scores_tensor"] = fakeScoresTensor.architectureJson();
    j["loss_shaper_input_tensor"] = lossShaperInput.architectureJson();
    j["loss_tensor"] = lossTensor.architectureJson();
    ThorImplementation::addLossWeightToJson(j, lossWeight);
    return j;
}

void WassersteinGANGeneratorLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in WassersteinGANGeneratorLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "wasserstein_gan_generator_loss")
        throw runtime_error("Layer type mismatch in WassersteinGANGeneratorLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["fake_scores_tensor"].at("id").get<uint64_t>();
    Tensor fakeScores = network->getApiTensorByOriginalId(originalTensorId);

    WassersteinGANGeneratorLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();

    loss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    loss.fakeScoresTensor = fakeScores;
    loss.network = network;
    loss.initialized = true;
    loss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("wasserstein_gan_generator_loss", &Thor::WassersteinGANGeneratorLoss::deserialize);
    return true;
}();
}  // namespace
