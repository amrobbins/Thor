#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <optional>
#include <string>

#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Layers/Loss/RegressionLossDType.h"

namespace ThorPython::RegressionLossDType {

namespace nb = nanobind;
using DataType = ThorImplementation::DataType;

inline void validatePredictions(const std::string& lossName, const Thor::Tensor& predictions) {
    if (!ThorImplementation::RegressionLossDType::isPredictionDType(predictions.getDataType())) {
        throw nb::value_error((lossName +
                               ": predictions must use fp8_e4m3, fp8_e5m2, fp16, bf16, or fp32 dtype.")
                                  .c_str());
    }
}

inline void validateLabels(const std::string& lossName, const Thor::Tensor& labels) {
    if (!ThorImplementation::RegressionLossDType::isLabelDType(labels.getDataType())) {
        throw nb::value_error((lossName +
                               ": labels must use a Thor integer dtype, bool, fp8_e4m3, fp8_e5m2, fp16, bf16, or fp32.")
                                  .c_str());
    }
}

inline void validateExampleWeights(const std::string& lossName, const Thor::Tensor& exampleWeights) {
    if (!ThorImplementation::RegressionLossDType::isExampleWeightDType(exampleWeights.getDataType())) {
        throw nb::value_error((lossName +
                               ": example_weights must use fp8_e4m3, fp8_e5m2, fp16, bf16, or fp32 dtype.")
                                  .c_str());
    }
}

inline DataType effectiveLossDType(const std::string& lossName,
                                   DataType predictionsDType,
                                   std::optional<DataType> requestedLossDType) {
    const DataType lossDType =
        requestedLossDType.value_or(ThorImplementation::RegressionLossDType::defaultLossDType(predictionsDType));
    if (!ThorImplementation::RegressionLossDType::isLossDType(lossDType)) {
        throw nb::value_error((lossName + ": loss_data_type must be fp16 or fp32.").c_str());
    }
    return lossDType;
}

}  // namespace ThorPython::RegressionLossDType
