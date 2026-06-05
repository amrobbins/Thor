#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/TverskyLoss.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

#include <vector>

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kPredictionsName = "predictions";
constexpr const char* kLabelsName = "labels";
constexpr const char* kLossName = "loss";
constexpr const char* kGradientName = "predictions_grad";
constexpr double kOverlapEpsilon = 1.0e-7;

void validateLabelsDType(DataType dtype) {
    switch (dtype) {
        case DataType::BOOLEAN:
        case DataType::UINT8:
        case DataType::UINT16:
        case DataType::UINT32:
        case DataType::FP16:
        case DataType::FP32:
            return;
        default:
            throw runtime_error("Unsupported TverskyLoss label dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validatePredictionsDType(DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error("Unsupported TverskyLoss predictions dtype: " + ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

std::vector<uint64_t> getSpatialReductionAxes(const std::vector<uint64_t>& apiFeatureDims) {
    THOR_THROW_IF_FALSE(!apiFeatureDims.empty());
    if (apiFeatureDims.size() == 1)
        return {1};

    std::vector<uint64_t> axes;
    axes.reserve(apiFeatureDims.size() - 1);
    for (uint64_t axis = 2; axis <= apiFeatureDims.size(); ++axis)
        axes.push_back(axis);
    return axes;
}

std::vector<uint64_t> getLossSqueezeAxes(const std::vector<uint64_t>& apiFeatureDims,
                                         const std::vector<uint64_t>& spatialReductionAxes) {
    if (apiFeatureDims.size() == 1)
        return {};
    return spatialReductionAxes;
}

struct TverskyTerms {
    ThorImplementation::Expression numerator;
    ThorImplementation::Expression denominator;
    ThorImplementation::Expression denominatorDerivative;
};

TverskyTerms makeTverskyTerms(const ThorImplementation::Expression& predictions,
                              const ThorImplementation::Expression& labels,
                              float alpha,
                              float beta,
                              float smooth,
                              const std::vector<uint64_t>& spatialReductionAxes,
                              const std::vector<uint64_t>& squeezeAxes) {
    ThorImplementation::Expression one(1.0);
    ThorImplementation::Expression alphaExpr(alpha);
    ThorImplementation::Expression betaExpr(beta);
    ThorImplementation::Expression smoothExpr(smooth);

    ThorImplementation::Expression truePositive = (predictions * labels).reduce_sum(spatialReductionAxes, squeezeAxes, DataType::FP32);
    ThorImplementation::Expression falsePositive = (predictions * (one - labels)).reduce_sum(spatialReductionAxes, squeezeAxes, DataType::FP32);
    ThorImplementation::Expression falseNegative = ((one - predictions) * labels).reduce_sum(spatialReductionAxes, squeezeAxes, DataType::FP32);
    ThorImplementation::Expression numerator = truePositive + smoothExpr;
    ThorImplementation::Expression denominator =
        (truePositive + (alphaExpr * falsePositive) + (betaExpr * falseNegative) + smoothExpr).max(ThorImplementation::Expression(kOverlapEpsilon));
    ThorImplementation::Expression denominatorDerivative = (alphaExpr * (one - labels)) + ((one - betaExpr) * labels);
    return {numerator, denominator, denominatorDerivative};
}

ThorImplementation::DynamicExpression makeTverskyLossExpression(DataType lossDataType,
                                                                float alpha,
                                                                float beta,
                                                                float smooth,
                                                                const std::vector<uint64_t>& spatialReductionAxes,
                                                                const std::vector<uint64_t>& lossSqueezeAxes) {
    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression one(1.0);

    TverskyTerms terms = makeTverskyTerms(predictions, labels, alpha, beta, smooth, spatialReductionAxes, lossSqueezeAxes);
    ThorImplementation::Expression loss = (one - (terms.numerator / terms.denominator)).withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeTverskyGradientExpression(DataType predictionsDataType,
                                                                    float alpha,
                                                                    float beta,
                                                                    float smooth,
                                                                    const std::vector<uint64_t>& spatialReductionAxes) {
    validatePredictionsDType(predictionsDataType);

    ThorImplementation::Expression predictions = ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression labels = ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);

    TverskyTerms terms = makeTverskyTerms(predictions, labels, alpha, beta, smooth, spatialReductionAxes, {});
    ThorImplementation::Expression gradient =
        (((terms.numerator * terms.denominatorDerivative) - (labels * terms.denominator)) / (terms.denominator * terms.denominator) *
         ThorImplementation::Expression(ThorImplementation::Loss::getLossScalingFactor()))
            .withOutputDType(predictionsDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kGradientName, gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void TverskyLoss::buildSupportLayersAndAddToNetwork() {
    validatePredictionsDType(predictionsTensor.getDataType());
    validateLabelsDType(labelsTensor.getDataType());
    THOR_THROW_IF_FALSE(alpha >= 0.0f);
    THOR_THROW_IF_FALSE(beta >= 0.0f);
    THOR_THROW_IF_FALSE(smooth >= 0.0f);

    const std::vector<uint64_t>& apiFeatureDims = predictionsTensor.getDimensions();
    std::vector<uint64_t> spatialReductionAxes = getSpatialReductionAxes(apiFeatureDims);
    std::vector<uint64_t> lossSqueezeAxes = getLossSqueezeAxes(apiFeatureDims, spatialReductionAxes);

    CustomLoss rawLoss = CustomLoss::Builder()
                             .network(*network)
                             .lossExpression(makeTverskyLossExpression(lossDataType, alpha, beta, smooth, spatialReductionAxes, lossSqueezeAxes))
                             .gradientExpression(makeTverskyGradientExpression(predictionsTensor.getDataType(), alpha, beta, smooth, spatialReductionAxes))
                             .predictions(predictionsTensor)
                             .labels(labelsTensor)
                             .predictionsName(kPredictionsName)
                             .labelsName(kLabelsName)
                             .lossName(kLossName)
                             .gradientName(kGradientName)
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

json TverskyLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["loss_shape"] = lossShape;
    j["alpha"] = alpha;
    j["beta"] = beta;
    j["smooth"] = smooth;
    return j;
}

void TverskyLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in TverskyLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "tversky_loss")
        throw runtime_error("Layer type mismatch in TverskyLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);

    TverskyLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();
    loss.alpha = j.value("alpha", 0.5f);
    loss.beta = j.value("beta", 0.5f);
    loss.smooth = j.value("smooth", 1.0f);
    loss.predictionsTensor = predictions;
    loss.labelsTensor = labels;
    loss.network = network;
    loss.initialized = true;
    loss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("tversky_loss", &Thor::TverskyLoss::deserialize);
    return true;
}();
}  // namespace
