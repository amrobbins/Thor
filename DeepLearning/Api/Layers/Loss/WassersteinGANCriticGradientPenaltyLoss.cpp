#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/WassersteinGANCriticGradientPenaltyLoss.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

#include <vector>

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kRealScoresName = "real_scores";
constexpr const char* kFakeScoresName = "fake_scores";
constexpr const char* kSampleGradientsName = "sample_gradients";
constexpr const char* kLossName = "loss";
constexpr const char* kRealScoresGradientName = "real_scores_grad";
constexpr const char* kFakeScoresGradientName = "fake_scores_grad";
constexpr const char* kSampleGradientsGradientName = "sample_gradients_grad";

void validateFloatingDType(const char* lossName, const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported ") + lossName + " " + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateDTypes(DataType realScoresDType, DataType fakeScoresDType, DataType sampleGradientsDType) {
    validateFloatingDType("WassersteinGANCriticGradientPenaltyLoss", "real_scores", realScoresDType);
    validateFloatingDType("WassersteinGANCriticGradientPenaltyLoss", "fake_scores", fakeScoresDType);
    validateFloatingDType("WassersteinGANCriticGradientPenaltyLoss", "sample_gradients", sampleGradientsDType);
    if (realScoresDType != fakeScoresDType)
        throw runtime_error("WassersteinGANCriticGradientPenaltyLoss real_scores and fake_scores tensors must have the same dtype.");
}

vector<uint64_t> getFeatureReductionAxes(const vector<uint64_t>& apiFeatureDims) {
    THOR_THROW_IF_FALSE(!apiFeatureDims.empty());
    vector<uint64_t> axes;
    axes.reserve(apiFeatureDims.size());
    for (uint64_t axis = 1; axis <= apiFeatureDims.size(); ++axis)
        axes.push_back(axis);
    return axes;
}

vector<uint64_t> getLossSqueezeAxes(const vector<uint64_t>& apiFeatureDims) {
    vector<uint64_t> axes;
    if (apiFeatureDims.size() <= 1)
        return axes;
    axes.reserve(apiFeatureDims.size() - 1);
    for (uint64_t axis = 2; axis <= apiFeatureDims.size(); ++axis)
        axes.push_back(axis);
    return axes;
}

ThorImplementation::Expression gradientPenaltyTerm(const ThorImplementation::Expression& sampleGradients,
                                                   float gradientPenaltyWeight,
                                                   float targetGradientNorm,
                                                   float eps,
                                                   const vector<uint64_t>& reductionAxes,
                                                   const vector<uint64_t>& squeezeAxes) {
    ThorImplementation::Expression squaredNorm = (sampleGradients * sampleGradients).reduce_sum(reductionAxes, squeezeAxes, DataType::FP32);
    ThorImplementation::Expression safeSquaredNorm = squaredNorm.max(ThorImplementation::Expression(eps));
    ThorImplementation::Expression norm = safeSquaredNorm.sqrt();
    ThorImplementation::Expression normDiff = norm - ThorImplementation::Expression(targetGradientNorm);
    return ThorImplementation::Expression(gradientPenaltyWeight) * normDiff * normDiff;
}

ThorImplementation::DynamicExpression makeWassersteinGANCriticGradientPenaltyLossExpression(DataType lossDataType,
                                                                                            float gradientPenaltyWeight,
                                                                                            float targetGradientNorm,
                                                                                            float eps,
                                                                                            const vector<uint64_t>& reductionAxes,
                                                                                            const vector<uint64_t>& lossSqueezeAxes) {
    validateFloatingDType("WassersteinGANCriticGradientPenaltyLoss", "loss", lossDataType);

    ThorImplementation::Expression realScores = ThorImplementation::Expression::input(kRealScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression fakeScores = ThorImplementation::Expression::input(kFakeScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression sampleGradients =
        ThorImplementation::Expression::input(kSampleGradientsName, DataType::FP32, DataType::FP32);

    ThorImplementation::Expression wasserstein = fakeScores - realScores;
    ThorImplementation::Expression penalty =
        gradientPenaltyTerm(sampleGradients, gradientPenaltyWeight, targetGradientNorm, eps, reductionAxes, lossSqueezeAxes);
    ThorImplementation::Expression loss = (wasserstein + penalty).withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeWassersteinGANCriticGradientPenaltyGradientExpression(DataType scoresDataType,
                                                                                                DataType sampleGradientsDataType,
                                                                                                float gradientPenaltyWeight,
                                                                                                float targetGradientNorm,
                                                                                                float eps,
                                                                                                const vector<uint64_t>& reductionAxes) {
    validateFloatingDType("WassersteinGANCriticGradientPenaltyLoss", "scores", scoresDataType);
    validateFloatingDType("WassersteinGANCriticGradientPenaltyLoss", "sample_gradients", sampleGradientsDataType);

    ThorImplementation::Expression realScores = ThorImplementation::Expression::input(kRealScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression fakeScores = ThorImplementation::Expression::input(kFakeScoresName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression sampleGradients =
        ThorImplementation::Expression::input(kSampleGradientsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression scale(ThorImplementation::Loss::getLossScalingFactor());
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression epsExpr(eps);

    ThorImplementation::Expression realGradient = ((realScores * zero) - scale).withOutputDType(scoresDataType);
    ThorImplementation::Expression fakeGradient = ((fakeScores * zero) + scale).withOutputDType(scoresDataType);

    ThorImplementation::Expression squaredNorm = (sampleGradients * sampleGradients).reduce_sum(reductionAxes, {}, DataType::FP32);
    ThorImplementation::Expression active = ThorImplementation::Expression::where(squaredNorm > epsExpr,
                                                                                   ThorImplementation::Expression(1.0),
                                                                                   ThorImplementation::Expression(0.0));
    ThorImplementation::Expression norm = squaredNorm.max(epsExpr).sqrt();
    ThorImplementation::Expression penaltyScale = ThorImplementation::Expression(2.0f * gradientPenaltyWeight) *
                                                  (norm - ThorImplementation::Expression(targetGradientNorm)) / norm;
    ThorImplementation::Expression sampleGradient = (active * penaltyScale * sampleGradients * scale).withOutputDType(sampleGradientsDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kRealScoresGradientName, realGradient},
                                                 {kFakeScoresGradientName, fakeGradient},
                                                 {kSampleGradientsGradientName, sampleGradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void WassersteinGANCriticGradientPenaltyLoss::buildSupportLayersAndAddToNetwork() {
    validateDTypes(realScoresTensor.getDataType(), fakeScoresTensor.getDataType(), sampleGradientsTensor.getDataType());
    THOR_THROW_IF_FALSE(realScoresTensor.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(realScoresTensor.getDimensions()[0] == 1);
    THOR_THROW_IF_FALSE(realScoresTensor.getDimensions() == fakeScoresTensor.getDimensions());
    THOR_THROW_IF_FALSE(!sampleGradientsTensor.getDimensions().empty());
    THOR_THROW_IF_FALSE(gradientPenaltyWeight >= 0.0f);
    THOR_THROW_IF_FALSE(targetGradientNorm > 0.0f);
    THOR_THROW_IF_FALSE(eps > 0.0f);

    vector<uint64_t> reductionAxes = getFeatureReductionAxes(sampleGradientsTensor.getDimensions());
    vector<uint64_t> lossSqueezeAxes = getLossSqueezeAxes(sampleGradientsTensor.getDimensions());

    MultiInputCustomLoss rawLoss = MultiInputCustomLoss::Builder()
                                       .network(*network)
                                       .lossExpression(makeWassersteinGANCriticGradientPenaltyLossExpression(
                                           lossDataType, gradientPenaltyWeight, targetGradientNorm, eps, reductionAxes, lossSqueezeAxes))
                                       .gradientExpression(makeWassersteinGANCriticGradientPenaltyGradientExpression(
                                           realScoresTensor.getDataType(), sampleGradientsTensor.getDataType(), gradientPenaltyWeight,
                                           targetGradientNorm, eps, reductionAxes))
                                       .input(kRealScoresName, realScoresTensor, kRealScoresGradientName)
                                       .input(kFakeScoresName, fakeScoresTensor, kFakeScoresGradientName)
                                       .input(kSampleGradientsName, sampleGradientsTensor, kSampleGradientsGradientName)
                                       .lossName(kLossName)
                                       .lossDataType(lossDataType)
                                       .reportsRawLoss()
                                       .build();

    lossShaperInput = rawLoss.getLoss();

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

json WassersteinGANCriticGradientPenaltyLoss::architectureJson() const {
    json j;
    j["factory"] = Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "wasserstein_gan_critic_gradient_penalty_loss";
    j["layer_name"] = string("layer") + to_string(getId());
    j["loss_shape"] = lossShape;
    j["loss_data_type"] = lossDataType;
    j["real_scores_tensor"] = realScoresTensor.architectureJson();
    j["fake_scores_tensor"] = fakeScoresTensor.architectureJson();
    j["sample_gradients_tensor"] = sampleGradientsTensor.architectureJson();
    j["gradient_penalty_weight"] = gradientPenaltyWeight;
    j["target_gradient_norm"] = targetGradientNorm;
    j["eps"] = eps;
    j["loss_shaper_input_tensor"] = lossShaperInput.architectureJson();
    j["loss_tensor"] = lossTensor.architectureJson();
    return j;
}

void WassersteinGANCriticGradientPenaltyLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in WassersteinGANCriticGradientPenaltyLoss::deserialize: " +
                            j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "wasserstein_gan_critic_gradient_penalty_loss")
        throw runtime_error("Layer type mismatch in WassersteinGANCriticGradientPenaltyLoss::deserialize: " +
                            j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["real_scores_tensor"].at("id").get<uint64_t>();
    Tensor realScores = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["fake_scores_tensor"].at("id").get<uint64_t>();
    Tensor fakeScores = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["sample_gradients_tensor"].at("id").get<uint64_t>();
    Tensor sampleGradients = network->getApiTensorByOriginalId(originalTensorId);

    WassersteinGANCriticGradientPenaltyLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();
    loss.realScoresTensor = realScores;
    loss.fakeScoresTensor = fakeScores;
    loss.sampleGradientsTensor = sampleGradients;
    loss.gradientPenaltyWeight = j.value("gradient_penalty_weight", 10.0f);
    loss.targetGradientNorm = j.value("target_gradient_norm", 1.0f);
    loss.eps = j.value("eps", 1.0e-12f);
    loss.network = network;
    loss.initialized = true;
    loss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("wasserstein_gan_critic_gradient_penalty_loss",
                               &Thor::WassersteinGANCriticGradientPenaltyLoss::deserialize);
    return true;
}();
}  // namespace
