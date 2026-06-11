#pragma once

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

#include <optional>
#include <string>

namespace Thor {
namespace ListwiseLossCommon {

inline constexpr const char* kPredictionsName = "predictions";
inline constexpr const char* kLabelsName = "labels";
inline constexpr const char* kMaskName = "mask";
inline constexpr const char* kLossName = "loss";
inline constexpr const char* kGradientName = "predictions_grad";

// Fixed-size listwise masks use mask > 0.5 as valid and mask <= 0.5 as padded/ignored.
// Padded documents contribute zero loss and zero prediction gradient. A fully padded list row
// is well-defined and produces zero raw loss and zero prediction gradient.
inline constexpr float kMaskThreshold = 0.5f;
inline constexpr float kMaskedLogitValue = -1.0e20f;

void validatePredictionsDType(const std::string& lossName, DataType dtype);
void validateLabelsDType(const std::string& lossName, DataType dtype);
void validateLossDataType(const std::string& lossName, DataType dtype);
void validateMaskDType(const std::string& lossName, DataType dtype);
void validatePositiveTemperature(const std::string& lossName, const std::string& parameterName, float temperature);
void validateFixedSizeListwiseTensors(const std::string& lossName,
                                      const Tensor& predictions,
                                      const Tensor& labels,
                                      const std::optional<Tensor>& mask);

ThorImplementation::Expression predictionsInput();
ThorImplementation::Expression labelsInput();
ThorImplementation::Expression maskInput();
ThorImplementation::Expression validDocumentMask();
ThorImplementation::Expression maskPaddedDocuments(ThorImplementation::Expression expression,
                                                   ThorImplementation::Expression validMask,
                                                   float maskedValue = kMaskedLogitValue);
ThorImplementation::Expression zeroPaddedDocuments(ThorImplementation::Expression expression, ThorImplementation::Expression validMask);
Tensor buildRawListwiseLoss(Network& network,
                            Tensor predictions,
                            Tensor labels,
                            const std::optional<Tensor>& mask,
                            ThorImplementation::DynamicExpression lossExpression,
                            ThorImplementation::DynamicExpression gradientExpression,
                            DataType lossDataType,
                            std::optional<float> lossWeight);
Tensor shapeRawListwiseLoss(Network& network, Tensor rawLoss, Loss::LossShape lossShape);

}  // namespace ListwiseLossCommon
}  // namespace Thor
