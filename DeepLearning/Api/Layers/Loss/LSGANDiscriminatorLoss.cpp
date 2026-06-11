#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/LSGANDiscriminatorLoss.h"

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
        throw runtime_error(string("Unsupported LSGANDiscriminatorLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateLSGANDiscriminatorDTypes(DataType realScoresDType, DataType fakeScoresDType) {
    validateScoreDType("real_scores", realScoresDType);
    validateScoreDType("fake_scores", fakeScoresDType);
    if (realScoresDType != fakeScoresDType)
        throw runtime_error("LSGANDiscriminatorLoss real_scores and fake_scores tensors must have the same dtype.");
}

ThorImplementation::DynamicExpression makeLSGANDiscriminatorLossExpression(DataType lossDataType, float realTarget, float fakeTarget) {
    validateScoreDType("loss", lossDataType);

    ThorImplementation::Expression realScores = ThorImplementation::Expression::input(kRealScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression fakeScores = ThorImplementation::Expression::input(kFakeScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression half(0.5);
    ThorImplementation::Expression realDiff = realScores - ThorImplementation::Expression(realTarget);
    ThorImplementation::Expression fakeDiff = fakeScores - ThorImplementation::Expression(fakeTarget);
    ThorImplementation::Expression loss = (half * ((realDiff * realDiff) + (fakeDiff * fakeDiff))).withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeLSGANDiscriminatorGradientExpression(DataType scoresDataType, float realTarget, float fakeTarget) {
    validateScoreDType("scores", scoresDataType);

    ThorImplementation::Expression realScores = ThorImplementation::Expression::input(kRealScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression fakeScores = ThorImplementation::Expression::input(kFakeScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression scale(ThorImplementation::Loss::getLossScalingFactor());
    ThorImplementation::Expression realGradient = ((realScores - ThorImplementation::Expression(realTarget)) * scale).withOutputDType(scoresDataType);
    ThorImplementation::Expression fakeGradient = ((fakeScores - ThorImplementation::Expression(fakeTarget)) * scale).withOutputDType(scoresDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kRealScoresGradientName, realGradient}, {kFakeScoresGradientName, fakeGradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void LSGANDiscriminatorLoss::buildSupportLayersAndAddToNetwork() {
    validateLSGANDiscriminatorDTypes(realScoresTensor.getDataType(), fakeScoresTensor.getDataType());
    THOR_THROW_IF_FALSE(realScoresTensor.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(realScoresTensor.getDimensions()[0] > 0);
    THOR_THROW_IF_FALSE(realScoresTensor.getDimensions() == fakeScoresTensor.getDimensions());

    MultiInputCustomLoss rawLSGANDiscriminatorLoss = MultiInputCustomLoss::Builder()
                                                         .network(*network)
                                                         .lossExpression(makeLSGANDiscriminatorLossExpression(
                                                             lossDataType, realTarget, fakeTarget))
                                                         .gradientExpression(makeLSGANDiscriminatorGradientExpression(
                                                             realScoresTensor.getDataType(), realTarget, fakeTarget))
                                                         .input(kRealScoresName, realScoresTensor, kRealScoresGradientName)
                                                         .input(kFakeScoresName, fakeScoresTensor, kFakeScoresGradientName)
                                                         .lossName(kLossName)
                                                         .lossDataType(lossDataType)
                                       .lossWeight(lossWeight.value_or(1.0f))
                                                         .reportsRawLoss()
                                                         .build();

    lossShaperInput = rawLSGANDiscriminatorLoss.getLoss();

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

json LSGANDiscriminatorLoss::architectureJson() const {
    json j;
    j["factory"] = Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "lsgan_discriminator_loss";
    j["layer_name"] = string("layer") + to_string(getId());
    j["loss_shape"] = lossShape;
    j["loss_data_type"] = lossDataType;
    j["real_target"] = realTarget;
    j["fake_target"] = fakeTarget;
    j["real_scores_tensor"] = realScoresTensor.architectureJson();
    j["fake_scores_tensor"] = fakeScoresTensor.architectureJson();
    j["loss_shaper_input_tensor"] = lossShaperInput.architectureJson();
    j["loss_tensor"] = lossTensor.architectureJson();
    ThorImplementation::addLossWeightToJson(j, lossWeight);
    return j;
}

void LSGANDiscriminatorLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in LSGANDiscriminatorLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "lsgan_discriminator_loss")
        throw runtime_error("Layer type mismatch in LSGANDiscriminatorLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["real_scores_tensor"].at("id").get<uint64_t>();
    Tensor realScores = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["fake_scores_tensor"].at("id").get<uint64_t>();
    Tensor fakeScores = network->getApiTensorByOriginalId(originalTensorId);

    LSGANDiscriminatorLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();

    loss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    loss.realTarget = j.value("real_target", 1.0f);
    loss.fakeTarget = j.value("fake_target", 0.0f);
    loss.realScoresTensor = realScores;
    loss.fakeScoresTensor = fakeScores;
    loss.network = network;
    loss.initialized = true;
    loss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("lsgan_discriminator_loss", &Thor::LSGANDiscriminatorLoss::deserialize);
    return true;
}();
}  // namespace
