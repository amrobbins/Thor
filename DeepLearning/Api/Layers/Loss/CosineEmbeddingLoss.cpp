#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Api/Layers/Loss/CosineEmbeddingLoss.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

using namespace std;
using json = nlohmann::json;

namespace Thor {
namespace {

constexpr const char* kInput1Name = "input1";
constexpr const char* kInput2Name = "input2";
constexpr const char* kTargetName = "target";
constexpr const char* kLossName = "loss";
constexpr const char* kInput1GradientName = "input1_grad";
constexpr const char* kInput2GradientName = "input2_grad";

void validateEmbeddingDType(const char* tensorName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        throw runtime_error(string("Unsupported CosineEmbeddingLoss ") + tensorName + " dtype: " +
                            ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateTargetDType(DataType dtype) {
    switch (dtype) {
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::INT64:
        case DataType::FP16:
        case DataType::FP32:
            return;
        default:
            throw runtime_error("Unsupported CosineEmbeddingLoss target dtype: " +
                                ThorImplementation::TensorDescriptor::getElementTypeName(dtype));
    }
}

void validateCosineEmbeddingDTypes(DataType input1DType, DataType input2DType, DataType targetDType) {
    validateEmbeddingDType("input1", input1DType);
    validateEmbeddingDType("input2", input2DType);
    validateTargetDType(targetDType);
    if (input1DType != input2DType)
        throw runtime_error("CosineEmbeddingLoss input1 and input2 tensors must have the same dtype.");
}

struct CosineTerms {
    ThorImplementation::Expression cosine;
    ThorImplementation::Expression input1Sq;
    ThorImplementation::Expression input2Sq;
    ThorImplementation::Expression denom;
};

CosineTerms cosineTerms(const ThorImplementation::Expression& input1,
                        const ThorImplementation::Expression& input2,
                        float eps) {
    ThorImplementation::Expression dot = (input1 * input2).reduce_sum({1}, {});
    ThorImplementation::Expression input1Sq = (input1 * input1).reduce_sum({1}, {}) + ThorImplementation::Expression(eps);
    ThorImplementation::Expression input2Sq = (input2 * input2).reduce_sum({1}, {}) + ThorImplementation::Expression(eps);
    ThorImplementation::Expression denom = (input1Sq * input2Sq).sqrt();
    ThorImplementation::Expression cosine = dot / denom;
    return CosineTerms{cosine, input1Sq, input2Sq, denom};
}

ThorImplementation::DynamicExpression makeCosineEmbeddingLossExpression(DataType lossDataType, float margin, float eps) {
    validateEmbeddingDType("loss", lossDataType);

    ThorImplementation::Expression input1 = ThorImplementation::Expression::input(kInput1Name, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression input2 = ThorImplementation::Expression::input(kInput2Name, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression target = ThorImplementation::Expression::input(kTargetName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression one(1.0);

    CosineTerms terms = cosineTerms(input1, input2, eps);
    ThorImplementation::Expression positiveLoss = one - terms.cosine;
    ThorImplementation::Expression negativeLoss = (terms.cosine - ThorImplementation::Expression(margin)).max(zero);
    ThorImplementation::Expression loss = ThorImplementation::Expression::where(target > zero, positiveLoss, negativeLoss)
                                          .withOutputDType(lossDataType);

    ThorImplementation::ExpressionDefinition definition =
        ThorImplementation::ExpressionDefinition::fromOutputs(ThorImplementation::Expression::outputs({{kLossName, loss}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

ThorImplementation::DynamicExpression makeCosineEmbeddingGradientExpression(DataType inputDataType, float margin, float eps) {
    validateEmbeddingDType("input", inputDataType);

    ThorImplementation::Expression input1 = ThorImplementation::Expression::input(kInput1Name, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression input2 = ThorImplementation::Expression::input(kInput2Name, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression target = ThorImplementation::Expression::input(kTargetName, DataType::FP32, DataType::FP32);
    ThorImplementation::Expression zero(0.0);
    ThorImplementation::Expression one(1.0);
    ThorImplementation::Expression negativeOne(-1.0);
    ThorImplementation::Expression scale(ThorImplementation::Loss::getLossScalingFactor());

    CosineTerms terms = cosineTerms(input1, input2, eps);
    ThorImplementation::Expression activeNegative = ThorImplementation::Expression::where(terms.cosine - ThorImplementation::Expression(margin) > zero,
                                                                                           one,
                                                                                           zero);
    ThorImplementation::Expression dLossDCos = ThorImplementation::Expression::where(target > zero, negativeOne, activeNegative);

    ThorImplementation::Expression dCosDInput1 = (input2 / terms.denom) - ((terms.cosine * input1) / terms.input1Sq);
    ThorImplementation::Expression dCosDInput2 = (input1 / terms.denom) - ((terms.cosine * input2) / terms.input2Sq);

    ThorImplementation::Expression input1Gradient = (dLossDCos * dCosDInput1 * scale).withOutputDType(inputDataType);
    ThorImplementation::Expression input2Gradient = (dLossDCos * dCosDInput2 * scale).withOutputDType(inputDataType);

    ThorImplementation::ExpressionDefinition definition = ThorImplementation::ExpressionDefinition::fromOutputs(
        ThorImplementation::Expression::outputs({{kInput1GradientName, input1Gradient}, {kInput2GradientName, input2Gradient}}));
    return ThorImplementation::DynamicExpression::fromExpressionDefinition(definition);
}

}  // namespace

void CosineEmbeddingLoss::buildSupportLayersAndAddToNetwork() {
    validateCosineEmbeddingDTypes(input1Tensor.getDataType(), input2Tensor.getDataType(), targetTensor.getDataType());
    THOR_THROW_IF_FALSE(input1Tensor.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(input1Tensor.getDimensions()[0] > 0);
    THOR_THROW_IF_FALSE(input1Tensor.getDimensions() == input2Tensor.getDimensions());
    THOR_THROW_IF_FALSE(targetTensor.getDimensions().size() == 1);
    THOR_THROW_IF_FALSE(targetTensor.getDimensions()[0] == 1);
    THOR_THROW_IF_FALSE(margin >= -1.0f && margin <= 1.0f);
    THOR_THROW_IF_FALSE(eps > 0.0f);

    MultiInputCustomLoss rawCosineEmbeddingLoss = MultiInputCustomLoss::Builder()
                                                      .network(*network)
                                                      .lossExpression(makeCosineEmbeddingLossExpression(lossDataType, margin, eps))
                                                      .gradientExpression(makeCosineEmbeddingGradientExpression(input1Tensor.getDataType(), margin, eps))
                                                      .input(kInput1Name, input1Tensor, kInput1GradientName)
                                                      .input(kInput2Name, input2Tensor, kInput2GradientName)
                                                      .auxiliaryInput(kTargetName, targetTensor)
                                                      .lossName(kLossName)
                                                      .lossDataType(lossDataType)
                                       .lossWeight(lossWeight.value_or(1.0f))
                                                      .reportsRawLoss()
                                                      .build();

    lossShaperInput = rawCosineEmbeddingLoss.getLoss();

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

json CosineEmbeddingLoss::architectureJson() const {
    json j;
    j["factory"] = Factory::Loss.value();
    j["version"] = getLayerVersion();
    j["layer_type"] = "cosine_embedding_loss";
    j["layer_name"] = string("layer") + to_string(getId());
    j["loss_shape"] = lossShape;
    j["loss_data_type"] = lossDataType;
    j["input1_tensor"] = input1Tensor.architectureJson();
    j["input2_tensor"] = input2Tensor.architectureJson();
    j["target_tensor"] = targetTensor.architectureJson();
    j["loss_shaper_input_tensor"] = lossShaperInput.architectureJson();
    j["loss_tensor"] = lossTensor.architectureJson();
    j["margin"] = margin;
    j["eps"] = eps;
    ThorImplementation::addLossWeightToJson(j, lossWeight);
    return j;
}

void CosineEmbeddingLoss::deserialize(const json& j, Network* network) {
    if (j.at("version").get<std::string>() != "1.0.0")
        throw runtime_error("Unsupported version in CosineEmbeddingLoss::deserialize: " + j["version"].get<std::string>());
    if (j.at("layer_type").get<std::string>() != "cosine_embedding_loss")
        throw runtime_error("Layer type mismatch in CosineEmbeddingLoss::deserialize: " + j.at("layer_type").get<std::string>());

    uint64_t originalTensorId = j["input1_tensor"].at("id").get<uint64_t>();
    Tensor input1 = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["input2_tensor"].at("id").get<uint64_t>();
    Tensor input2 = network->getApiTensorByOriginalId(originalTensorId);
    originalTensorId = j["target_tensor"].at("id").get<uint64_t>();
    Tensor target = network->getApiTensorByOriginalId(originalTensorId);

    CosineEmbeddingLoss cosineEmbeddingLoss;
    cosineEmbeddingLoss.lossShape = j.at("loss_shape").get<LossShape>();
    cosineEmbeddingLoss.lossDataType = j.at("loss_data_type").get<DataType>();

    cosineEmbeddingLoss.lossWeight = ThorImplementation::lossWeightFromJson(j);
    cosineEmbeddingLoss.margin = j.value("margin", 0.0f);
    cosineEmbeddingLoss.eps = j.value("eps", 1.0e-8f);
    cosineEmbeddingLoss.input1Tensor = input1;
    cosineEmbeddingLoss.input2Tensor = input2;
    cosineEmbeddingLoss.targetTensor = target;
    cosineEmbeddingLoss.network = network;
    cosineEmbeddingLoss.initialized = true;
    cosineEmbeddingLoss.buildSupportLayersAndAddToNetwork();
}

}  // namespace Thor

namespace {
static bool registered = []() {
    Thor::Loss::register_layer("cosine_embedding_loss", &Thor::CosineEmbeddingLoss::deserialize);
    return true;
}();
}  // namespace
