#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

#include <stdexcept>
#include <string>

namespace ThorImplementation::RegressionLossDType {

inline bool isPredictionDType(DataType dtype) {
    switch (dtype) {
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

inline bool isLabelDType(DataType dtype) {
    if (isPredictionDType(dtype))
        return true;

    switch (dtype) {
        case DataType::BOOLEAN:
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::INT64:
        case DataType::UINT8:
        case DataType::UINT16:
        case DataType::UINT32:
        case DataType::UINT64:
            return true;
        default:
            return false;
    }
}

inline bool isExampleWeightDType(DataType dtype) { return isPredictionDType(dtype); }

// LossShaper currently reduces through the validated FP16/FP32 cuDNN reduction
// surface. Low-precision prediction tensors are converted to FP32 inside the loss
// expression, but reduced loss storage remains FP16 or FP32.
inline bool isLossDType(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::FP32; }

inline DataType defaultLossDType(DataType predictionsDType) {
    if (!isPredictionDType(predictionsDType)) {
        throw std::runtime_error("Unsupported regression-loss predictions dtype: " +
                                 TensorDescriptor::getElementTypeName(predictionsDType));
    }
    return predictionsDType == DataType::FP16 ? DataType::FP16 : DataType::FP32;
}

inline void validatePredictionsDType(const std::string& lossName, DataType dtype) {
    if (!isPredictionDType(dtype)) {
        throw std::runtime_error("Unsupported " + lossName + " predictions dtype: " +
                                 TensorDescriptor::getElementTypeName(dtype) +
                                 ". Supported prediction dtypes are fp8_e4m3, fp8_e5m2, fp16, bf16, and fp32.");
    }
}

inline void validateLabelsDType(const std::string& lossName, DataType dtype) {
    if (!isLabelDType(dtype)) {
        throw std::runtime_error("Unsupported " + lossName + " label dtype: " + TensorDescriptor::getElementTypeName(dtype) +
                                 ". Labels may use any Thor integer dtype, bool, fp8_e4m3, fp8_e5m2, fp16, bf16, or fp32.");
    }
}

inline void validateExampleWeightDType(const std::string& lossName, DataType dtype) {
    if (!isExampleWeightDType(dtype)) {
        throw std::runtime_error("Unsupported " + lossName + " example_weights dtype: " +
                                 TensorDescriptor::getElementTypeName(dtype) +
                                 ". Supported example_weights dtypes are fp8_e4m3, fp8_e5m2, fp16, bf16, and fp32.");
    }
}

inline void validateLossDType(const std::string& lossName, DataType dtype) {
    if (!isLossDType(dtype)) {
        throw std::runtime_error("Unsupported " + lossName + " loss dtype: " + TensorDescriptor::getElementTypeName(dtype) +
                                 ". Reduced loss tensors currently support fp16 or fp32.");
    }
}

}  // namespace ThorImplementation::RegressionLossDType
