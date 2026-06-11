#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/GaussianNLLLoss.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kMeanName = "mean";
constexpr const char* kTargetName = "target";
constexpr const char* kVarianceName = "variance";
constexpr const char* kLossName = "loss";
constexpr const char* kMeanGradientName = "mean_grad";
constexpr const char* kVarianceGradientName = "variance_grad";
constexpr double kLogTwoPi = 1.837877066409345483560659472811;

void validateFloatingDType(const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported GaussianNLLLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateGaussianDTypes(DataType meanDType, DataType targetDType, DataType varianceDType) {
    validateFloatingDType("mean", meanDType);
    validateFloatingDType("target", targetDType);
    validateFloatingDType("variance", varianceDType);
}

ThorImplementation::Expression clampedVariance(const ThorImplementation::Expression& variance, float eps) {
    return variance.max(ThorImplementation::Expression(eps));
}

ThorImplementation::DynamicExpression makeGaussianNLLLossExpression(DataType lossDataType, bool full, float eps) {
    validateFloatingDType("loss", lossDataType);

    ThorImplementation::Expression mean = ThorImplementation::Expression::input(kMeanName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression target = ThorImplementation::Expression::input(kTargetName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression variance = ThorImplementation::Expression::input(kVarianceName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression safeVariance = clampedVariance(variance, eps);
    ThorImplementation::Expression diff = mean - target;
    ThorImplementation::Expression loss = ThorImplementation::Expression(0.5) * (safeVariance.ln() + (diff * diff) / safeVariance);
    if (full)
        loss = loss + ThorImplementation::Expression(0.5 * kLogTwoPi);
    loss = loss.withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeGaussianNLLGradientExpression(DataType meanDType, DataType varianceDType, float eps) {
    validateFloatingDType("mean", meanDType);
    validateFloatingDType("variance", varianceDType);

    ThorImplementation::Expression mean = ThorImplementation::Expression::input(kMeanName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression target = ThorImplementation::Expression::input(kTargetName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression variance = ThorImplementation::Expression::input(kVarianceName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression safeVariance = clampedVariance(variance, eps);
    ThorImplementation::Expression diff = mean - target;
    ThorImplementation::Expression scale(ThorImplementation::Loss::getLossScalingFactor());

    ThorImplementation::Expression meanGradient = (diff / safeVariance * scale).withOutputDType(meanDType);
    ThorImplementation::Expression varianceGradient =
        (ThorImplementation::Expression(0.5) *
         ((ThorImplementation::Expression(1.0) / safeVariance) - ((diff * diff) / (safeVariance * safeVariance))) * scale)
            .withOutputDType(varianceDType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kMeanGradientName, meanGradient}, {kVarianceGradientName, varianceGradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void GaussianNLLLoss::buildSupportLayersAndAddToNetwork() {
    validateGaussianDTypes(predictionsTensor.getDataType(), labelsTensor.getDataType(), varianceTensor.getDataType());
    THOR_THROW_IF_FALSE(eps > 0.0f);

    MultiInputCustomLoss rawGaussianNLLLoss = MultiInputCustomLoss::Builder()
                                                  .network(*network)
                                                  .lossExpression(makeGaussianNLLLossExpression(lossDataType, full, eps))
                                                  .gradientExpression(makeGaussianNLLGradientExpression(predictionsTensor.getDataType(),
                                                                                                        varianceTensor.getDataType(),
                                                                                                        eps))
                                                  .input(kMeanName, predictionsTensor, kMeanGradientName)
                                                  .auxiliaryInput(kTargetName, labelsTensor)
                                                  .input(kVarianceName, varianceTensor, kVarianceGradientName)
                                                  .lossName(kLossName)
                                                  .lossDataType(lossDataType)
                                       .lossWeight(lossWeight.value_or(1.0f))
                                                  .reportsRawLoss()
                                                  .build();

    lossShaperInput = rawGaussianNLLLoss.getLoss();

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

json GaussianNLLLoss::architectureJson() const {
    json j = Loss::architectureJson();
    j["layer_type"] = "gaussian_nll_loss";
    j["loss_shape"] = lossShape;
    j["variance_tensor"] = varianceTensor.architectureJson();
    j["full"] = full;
    j["eps"] = eps;
    return j;
}

void GaussianNLLLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in GaussianNLLLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "gaussian_nll_loss")
        throw runtime_error("Layer type mismatch in GaussianNLLLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["predictions_tensor"].at("id").get<uint64_t>();
    Tensor predictions = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["labels_tensor"].at("id").get<uint64_t>();
    Tensor labels = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["variance_tensor"].at("id").get<uint64_t>();
    Tensor variance = network->getApiTensorByOriginalId(originalTensorId);

    GaussianNLLLoss loss;
    loss.lossShape = j.at("loss_shape").get<LossShape>();
    loss.lossDataType = j.at("loss_data_type").get<DataType>();

    loss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    loss.full = j.value("full", false);
    loss.eps = j.value("eps", 1.0e-6f);
    loss.predictionsTensor = predictions;
    loss.labelsTensor = labels;
    loss.varianceTensor = variance;
    loss.network = network;
    loss.initialized = true;
    loss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("gaussian_nll_loss", &Thor::GaussianNLLLoss::deserialize);
    return true;
}();
}  // namespace
