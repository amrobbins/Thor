#include "DeepLearning/Api/Layers/Loss/ListwiseLossCommon.h"

#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Layers/Loss/MultiInputCustomLoss.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <stdexcept>
#include <utility>

using namespace std;

namespace Thor {
namespace ListwiseLossCommon {

namespace {

string dtypeName(DataType dtype) { return ThorImplementation::TensorDescriptor::getElementTypeName(dtype); }

}  // namespace

void validatePredictionsDType(const string& lossName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32)
        throw runtime_error("Unsupported " + lossName + " predictions dtype: " + dtypeName(dtype));
}

void validateLabelsDType(const string& lossName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32)
        throw runtime_error("Unsupported " + lossName + " label dtype: " + dtypeName(dtype));
}

void validateLossDataType(const string& lossName, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32)
        throw runtime_error("Unsupported " + lossName + " loss dtype: " + dtypeName(dtype));
}

void validateMaskDType(const string& lossName, DataType dtype) {
    switch (dtype) {
        case DataType::BOOLEAN:
        case DataType::UINT8:
        case DataType::FP16:
        case DataType::FP32:
            return;
        default:
            throw runtime_error("Unsupported " + lossName + " mask dtype: " + dtypeName(dtype));
    }
}

void validatePositiveTemperature(const string& lossName, const string& parameterName, float temperature) {
    if (temperature <= 0.0f)
        throw runtime_error(lossName + " requires " + parameterName + " to be greater than zero.");
}

void validateFixedSizeListwiseTensors(const string& lossName,
                                      const Tensor& predictions,
                                      const Tensor& labels,
                                      const optional<Tensor>& mask) {
    validatePredictionsDType(lossName, predictions.getDataType());
    validateLabelsDType(lossName, labels.getDataType());
    if (predictions.getDimensions().size() != 1)
        throw runtime_error(lossName + " predictions must be a 1 dimensional fixed-size list score tensor.");
    if (predictions.getDimensions()[0] <= 1)
        throw runtime_error(lossName + " predictions must contain more than one document per list.");
    if (predictions.getDimensions() != labels.getDimensions())
        throw runtime_error(lossName + " labels dimensions must match predictions dimensions.");
    if (mask.has_value()) {
        validateMaskDType(lossName, mask.value().getDataType());
        if (mask.value().getDimensions() != predictions.getDimensions())
            throw runtime_error(lossName + " mask dimensions must match predictions dimensions.");
    }
}

ThorImplementation::Expression predictionsInput() {
    return ThorImplementation::Expression::input(kPredictionsName, DataType::FP32, DataType::FP32);
}

ThorImplementation::Expression labelsInput() {
    return ThorImplementation::Expression::input(kLabelsName, DataType::FP32, DataType::FP32);
}

ThorImplementation::Expression maskInput() { return ThorImplementation::Expression::input(kMaskName, DataType::FP32, DataType::FP32); }

ThorImplementation::Expression validDocumentMask() { return maskInput() > ThorImplementation::Expression(kMaskThreshold); }

ThorImplementation::Expression maskPaddedDocuments(ThorImplementation::Expression expression,
                                                   ThorImplementation::Expression validMask,
                                                   float maskedValue) {
    return ThorImplementation::Expression::where(validMask, expression, ThorImplementation::Expression(maskedValue));
}

ThorImplementation::Expression zeroPaddedDocuments(ThorImplementation::Expression expression, ThorImplementation::Expression validMask) {
    return ThorImplementation::Expression::where(validMask, expression, ThorImplementation::Expression(0.0));
}

Tensor buildRawListwiseLoss(Network& network,
                            Tensor predictions,
                            Tensor labels,
                            const optional<Tensor>& mask,
                            ThorImplementation::DynamicExpression lossExpression,
                            ThorImplementation::DynamicExpression gradientExpression,
                            DataType lossDataType) {
    validateLossDataType("listwise loss", lossDataType);
    if (mask.has_value()) {
        MultiInputCustomLoss rawLoss = MultiInputCustomLoss::Builder()
                                           .network(network)
                                           .lossExpression(std::move(lossExpression))
                                           .gradientExpression(std::move(gradientExpression))
                                           .input(kPredictionsName, predictions, kGradientName)
                                           .auxiliaryInput(kLabelsName, labels)
                                           .auxiliaryInput(kMaskName, mask.value())
                                           .lossName(kLossName)
                                           .lossDataType(lossDataType)
                                           .reportsRawLoss()
                                           .build();
        return rawLoss.getLoss();
    }

    CustomLoss rawLoss = CustomLoss::Builder()
                             .network(network)
                             .lossExpression(std::move(lossExpression))
                             .gradientExpression(std::move(gradientExpression))
                             .predictions(predictions)
                             .labels(labels)
                             .predictionsName(kPredictionsName)
                             .labelsName(kLabelsName)
                             .lossName(kLossName)
                             .gradientName(kGradientName)
                             .lossDataType(lossDataType)
                             .reportsRawLoss()
                             .build();
    return rawLoss.getLoss();
}

Tensor shapeRawListwiseLoss(Network& network, Tensor rawLoss, Loss::LossShape lossShape) {
    if (lossShape == Loss::LossShape::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(network).lossInput(rawLoss).reportsBatchLoss().build();
        return lossShaper.getLossOutput();
    }
    if (lossShape == Loss::LossShape::ELEMENTWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(network).lossInput(rawLoss).reportsElementwiseLoss().build();
        return lossShaper.getLossOutput();
    }
    if (lossShape == Loss::LossShape::CLASSWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(network).lossInput(rawLoss).reportsClasswiseLoss().build();
        return lossShaper.getLossOutput();
    }

    THOR_THROW_IF_FALSE(lossShape == Loss::LossShape::RAW);
    return rawLoss;
}

}  // namespace ListwiseLossCommon
}  // namespace Thor
