#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/HingeGANDiscriminatorLoss.h"

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
        throw runtime_error(string("Unsupported HingeGANDiscriminatorLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateHingeGANDiscriminatorDTypes(DataType realScoresDType, DataType fakeScoresDType) {
    validateScoreDType("real_scores", realScoresDType);
    validateScoreDType("fake_scores", fakeScoresDType);
    if (realScoresDType != fakeScoresDType)
        throw runtime_error("HingeGANDiscriminatorLoss real_scores and fake_scores tensors must have the same dtype.");
}

ThorImplementation::DynamicExpression makeHingeGANDiscriminatorLossExpression(DataType lossDataType) {
    validateScoreDType("loss", lossDataType);

    ThorImplementation::Expression realScores = ThorImplementation::Expression::input(kRealScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression fakeScores = ThorImplementation::Expression::input(kFakeScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression one(1.0);

    ThorImplementation::Expression realHinge = (one - realScores).max(zero);
    ThorImplementation::Expression fakeHinge = (one + fakeScores).max(zero);
    ThorImplementation::Expression loss = (realHinge + fakeHinge).withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeHingeGANDiscriminatorGradientExpression(DataType scoresDataType) {
    validateScoreDType("scores", scoresDataType);

    ThorImplementation::Expression realScores = ThorImplementation::Expression::input(kRealScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression fakeScores = ThorImplementation::Expression::input(kFakeScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression one(1.0);
    ThorImplementation::Expression scale(ThorImplementation::Loss::getLossScalingFactor());

    ThorImplementation::Expression realActive = ThorImplementation::Expression::where(one - realScores > zero, one, zero);
    ThorImplementation::Expression fakeActive = ThorImplementation::Expression::where(one + fakeScores > zero, one, zero);
    ThorImplementation::Expression realGradient = (-(realActive * scale)).withOutputDType(scoresDataType);
    ThorImplementation::Expression fakeGradient = (fakeActive * scale).withOutputDType(scoresDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kRealScoresGradientName, realGradient}, {kFakeScoresGradientName, fakeGradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void HingeGANDiscriminatorLoss::buildSupportLayersAndAddToNetwork() {
    validateHingeGANDiscriminatorDTypes(realScoresTensor.getDataType(), fakeScoresTensor.getDataType());
    THOR_THROW_IF_FALSE(realScoresTensor.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(realScoresTensor.getDimensions()[0] > 0);
    THOR_THROW_IF_FALSE(realScoresTensor.getDimensions() == fakeScoresTensor.getDimensions());

    MultiInputCustomLoss rawHingeGANDiscriminatorLoss = MultiInputCustomLoss::Builder()
                                                            .network(*network)
                                                            .lossExpression(makeHingeGANDiscriminatorLossExpression(lossDataType))
                                                            .gradientExpression(
                                                                makeHingeGANDiscriminatorGradientExpression(realScoresTensor.getDataType()))
                                                            .input(kRealScoresName, realScoresTensor, kRealScoresGradientName)
                                                            .input(kFakeScoresName, fakeScoresTensor, kFakeScoresGradientName)
                                                            .lossName(kLossName)
                                                            .lossDataType(lossDataType)
                                                            .reportsRawLoss()
                                                            .build();

    lossShaperInput = rawHingeGANDiscriminatorLoss.getLoss();

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

json HingeGANDiscriminatorLoss::architectureJson() const {
    json j;
    j["factory"] = Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "hinge_gan_discriminator_loss";
    j["layer_name"] = string("layer") + to_string(getId());
    j["loss_shape"] = lossShape;
    j["loss_data_type"] = lossDataType;
    j["real_scores_tensor"] = realScoresTensor.architectureJson();
    j["fake_scores_tensor"] = fakeScoresTensor.architectureJson();
    j["loss_shaper_input_tensor"] = lossShaperInput.architectureJson();
    j["loss_tensor"] = lossTensor.architectureJson();
    return j;
}

void HingeGANDiscriminatorLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in HingeGANDiscriminatorLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "hinge_gan_discriminator_loss")
        throw runtime_error("Layer type mismatch in HingeGANDiscriminatorLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["real_scores_tensor"].at("id").get<uint64_t>();
    Tensor realScores = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["fake_scores_tensor"].at("id").get<uint64_t>();
    Tensor fakeScores = network->getApiTensorByOriginalId(originalTensorId);

    HingeGANDiscriminatorLoss hingeGANDiscriminatorLoss;
    hingeGANDiscriminatorLoss.lossShape = j.at("loss_shape").get<LossShape>();
    hingeGANDiscriminatorLoss.lossDataType = j.at("loss_data_type").get<DataType>();
    hingeGANDiscriminatorLoss.realScoresTensor = realScores;
    hingeGANDiscriminatorLoss.fakeScoresTensor = fakeScores;
    hingeGANDiscriminatorLoss.network = network;
    hingeGANDiscriminatorLoss.initialized = true;
    hingeGANDiscriminatorLoss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("hinge_gan_discriminator_loss", &Thor::HingeGANDiscriminatorLoss::deserialize);
    return true;
}();
}  // namespace
