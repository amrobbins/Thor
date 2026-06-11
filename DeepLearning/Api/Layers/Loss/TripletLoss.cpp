#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/TripletLoss.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kAnchorName = "anchor";
constexpr const char* kPositiveName = "positive";
constexpr const char* kNegativeName = "negative";
constexpr const char* kLossName = "loss";
constexpr const char* kAnchorGradientName = "anchor_grad";
constexpr const char* kPositiveGradientName = "positive_grad";
constexpr const char* kNegativeGradientName = "negative_grad";

void validateEmbeddingDType(const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported TripletLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateTripletDTypes(DataType anchorDType, DataType positiveDType, DataType negativeDType) {
    validateEmbeddingDType("anchor", anchorDType);
    validateEmbeddingDType("positive", positiveDType);
    validateEmbeddingDType("negative", negativeDType);
    if (anchorDType != positiveDType || anchorDType != negativeDType)
        throw runtime_error("TripletLoss anchor, positive, and negative tensors must have the same dtype.");
}

ThorImplementation::Expression squaredDistance(const ThorImplementation::Expression& lhs,
                                                const ThorImplementation::Expression& rhs,
                                                float eps) {
    ThorImplementation::Expression diff = lhs - rhs;
    return ((diff * diff).reduce_sum({1}, {}) + ThorImplementation::Expression(eps));
}

ThorImplementation::DynamicExpression makeTripletLossExpression(DataType lossDataType, float margin, float eps) {
    validateEmbeddingDType("loss", lossDataType);

    ThorImplementation::Expression anchor = ThorImplementation::Expression::input(kAnchorName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression positive = ThorImplementation::Expression::input(kPositiveName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression negative = ThorImplementation::Expression::input(kNegativeName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);

    ThorImplementation::Expression dAp = squaredDistance(anchor, positive, eps).sqrt();
    ThorImplementation::Expression dAn = squaredDistance(anchor, negative, eps).sqrt();
    ThorImplementation::Expression loss = (dAp - dAn + ThorImplementation::Expression(margin)).max(zero).withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeTripletGradientExpression(DataType inputDataType, float margin, float eps) {
    validateEmbeddingDType("input", inputDataType);

    ThorImplementation::Expression anchor = ThorImplementation::Expression::input(kAnchorName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression positive = ThorImplementation::Expression::input(kPositiveName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression negative = ThorImplementation::Expression::input(kNegativeName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression scale(ThorImplementation::Loss::getLossScalingFactor());

    ThorImplementation::Expression ap = anchor - positive;
    ThorImplementation::Expression an = anchor - negative;
    ThorImplementation::Expression dAp = ((ap * ap).reduce_sum({1}, {}) + ThorImplementation::Expression(eps)).sqrt();
    ThorImplementation::Expression dAn = ((an * an).reduce_sum({1}, {}) + ThorImplementation::Expression(eps)).sqrt();
    ThorImplementation::Expression active = ThorImplementation::Expression::where(dAp - dAn + ThorImplementation::Expression(margin) > zero,
                                                                                   ThorImplementation::Expression(1.0),
                                                                                   zero);

    ThorImplementation::Expression anchorGradient = (active * ((ap / dAp) - (an / dAn)) * scale).withOutputDType(inputDataType);
    ThorImplementation::Expression positiveGradient = (active * (-(ap / dAp)) * scale).withOutputDType(inputDataType);
    ThorImplementation::Expression negativeGradient = (active * (an / dAn) * scale).withOutputDType(inputDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kAnchorGradientName, anchorGradient},
                                                 {kPositiveGradientName, positiveGradient},
                                                 {kNegativeGradientName, negativeGradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void TripletLoss::buildSupportLayersAndAddToNetwork() {
    validateTripletDTypes(anchorTensor.getDataType(), positiveTensor.getDataType(), negativeTensor.getDataType());
    THOR_THROW_IF_FALSE(anchorTensor.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(anchorTensor.getDimensions() == positiveTensor.getDimensions());
    THOR_THROW_IF_FALSE(anchorTensor.getDimensions() == negativeTensor.getDimensions());
    THOR_THROW_IF_FALSE(margin > 0.0f);
    THOR_THROW_IF_FALSE(eps > 0.0f);

    MultiInputCustomLoss rawTripletLoss = MultiInputCustomLoss::Builder()
                                              .network(*network)
                                              .lossExpression(makeTripletLossExpression(lossDataType, margin, eps))
                                              .gradientExpression(makeTripletGradientExpression(anchorTensor.getDataType(), margin, eps))
                                              .input(kAnchorName, anchorTensor, kAnchorGradientName)
                                              .input(kPositiveName, positiveTensor, kPositiveGradientName)
                                              .input(kNegativeName, negativeTensor, kNegativeGradientName)
                                              .lossName(kLossName)
                                              .lossDataType(lossDataType)
                                       .lossWeight(lossWeight.value_or(1.0f))
                                              .reportsRawLoss()
                                              .build();

    lossShaperInput = rawTripletLoss.getLoss();

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

json TripletLoss::architectureJson() const {
    json j;
    j["factory"] = Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "triplet_loss";
    j["layer_name"] = string("layer") + to_string(getId());
    j["loss_shape"] = lossShape;
    j["loss_data_type"] = lossDataType;
    j["anchor_tensor"] = anchorTensor.architectureJson();
    j["positive_tensor"] = positiveTensor.architectureJson();
    j["negative_tensor"] = negativeTensor.architectureJson();
    j["loss_shaper_input_tensor"] = lossShaperInput.architectureJson();
    j["loss_tensor"] = lossTensor.architectureJson();
    j["margin"] = margin;
    j["eps"] = eps;
    ThorImplementation::addLossWeightToJson(j, lossWeight);
    return j;
}

void TripletLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in TripletLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "triplet_loss")
        throw runtime_error("Layer type mismatch in TripletLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["anchor_tensor"].at("id").get<uint64_t>();
    Tensor anchor = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["positive_tensor"].at("id").get<uint64_t>();
    Tensor positive = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["negative_tensor"].at("id").get<uint64_t>();
    Tensor negative = network->getApiTensorByOriginalId(originalTensorId);

    TripletLoss tripletLoss;
    tripletLoss.lossShape = j.at("loss_shape").get<LossShape>();
    tripletLoss.lossDataType = j.at("loss_data_type").get<DataType>();

    tripletLoss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    tripletLoss.margin = j.value("margin", 1.0f);
    tripletLoss.eps = j.value("eps", 1.0e-6f);
    tripletLoss.anchorTensor = anchor;
    tripletLoss.positiveTensor = positive;
    tripletLoss.negativeTensor = negative;
    tripletLoss.network = network;
    tripletLoss.initialized = true;
    tripletLoss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("triplet_loss", &Thor::TripletLoss::deserialize);
    return true;
}();
}  // namespace
