#pragma once

#include <nanobind/nanobind.h>

#include <optional>
#include <string>

#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace ThorPython::Ranking::ListwiseCommon {

namespace nb = nanobind;

using DataType = ThorImplementation::DataType;
using LossShape = Thor::Loss::LossShape;

inline bool isListwiseValueDType(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::FP32; }

inline bool isListwiseMaskDType(DataType dtype) {
    return dtype == DataType::BOOLEAN || dtype == DataType::UINT8 || dtype == DataType::FP16 || dtype == DataType::FP32;
}

inline void validateReportedLossShape(LossShape reportedLossShape, const std::string& lossName) {
    if (reportedLossShape != LossShape::BATCH && reportedLossShape != LossShape::CLASSWISE &&
        reportedLossShape != LossShape::ELEMENTWISE && reportedLossShape != LossShape::RAW) {
        std::string errorMessage =
            "Invalid value " + std::to_string(static_cast<int>(reportedLossShape)) + " passed for enum reported_loss_shape to " + lossName + ".";
        throw nb::value_error(errorMessage.c_str());
    }
}

template <typename Builder>
inline void setReportedLossShape(Builder& builder, LossShape reportedLossShape) {
    if (reportedLossShape == LossShape::BATCH) {
        builder.reportsBatchLoss();
    } else if (reportedLossShape == LossShape::CLASSWISE) {
        builder.reportsPerOutputLoss();
    } else if (reportedLossShape == LossShape::ELEMENTWISE) {
        builder.reportsElementwiseLoss();
    } else {
        validateReportedLossShape(reportedLossShape, "loss");
        builder.reportsRawLoss();
    }
}

inline void validateFixedSizeListwiseTensorArguments(const std::string& lossName,
                                                    Thor::Tensor predictions,
                                                    Thor::Tensor labels,
                                                    std::optional<DataType> lossDataType,
                                                    LossShape reportedLossShape,
                                                    std::optional<Thor::Tensor> mask) {
    if (predictions.getDimensions().size() != 1 || predictions.getDimensions()[0] <= 1) {
        std::string errorMessage = lossName +
                                   ": predictions must be a 1 dimensional fixed-size list score tensor with more than one document but predictions is " +
                                   predictions.getDescriptorString();
        throw nb::value_error(errorMessage.c_str());
    }
    if (labels.getDimensions() != predictions.getDimensions()) {
        std::string errorMessage = lossName + ": labels dimensions " + labels.getDescriptorString() +
                                   " must match predictions dimensions " + predictions.getDescriptorString();
        throw nb::value_error(errorMessage.c_str());
    }
    if (mask.has_value() && mask.value().getDimensions() != predictions.getDimensions()) {
        std::string errorMessage = lossName + ": mask dimensions " + mask.value().getDescriptorString() +
                                   " must match predictions dimensions " + predictions.getDescriptorString();
        throw nb::value_error(errorMessage.c_str());
    }
    if (!isListwiseValueDType(predictions.getDataType())) {
        std::string errorMessage = lossName + ": predictions must use fp16 or fp32 dtype";
        throw nb::value_error(errorMessage.c_str());
    }
    if (!isListwiseValueDType(labels.getDataType())) {
        std::string errorMessage = lossName + ": labels must use fp16 or fp32 dtype";
        throw nb::value_error(errorMessage.c_str());
    }
    if (mask.has_value() && !isListwiseMaskDType(mask.value().getDataType())) {
        std::string errorMessage = lossName + ": mask must use bool, uint8, fp16, or fp32 dtype";
        throw nb::value_error(errorMessage.c_str());
    }

    DataType effectiveLossDataType = lossDataType.value_or(predictions.getDataType());
    if (!isListwiseValueDType(effectiveLossDataType)) {
        std::string errorMessage = lossName + ": loss_data_type must be fp16 or fp32";
        throw nb::value_error(errorMessage.c_str());
    }
    validateReportedLossShape(reportedLossShape, lossName);
}

inline void validatePositiveTemperature(const std::string& lossName, const std::string& parameterName, float temperature) {
    if (temperature <= 0.0f) {
        std::string errorMessage = lossName + ": " + parameterName + " must be greater than zero";
        throw nb::value_error(errorMessage.c_str());
    }
}

}  // namespace ThorPython::Ranking::ListwiseCommon
