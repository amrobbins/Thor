#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

#include <stdexcept>
#include <string>

namespace ThorImplementation::ReductionMetricDType {

inline bool isValueDType(DataType dtype) {
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

inline void validateValueDType(const std::string& metricName, const std::string& inputName, DataType dtype) {
    if (isValueDType(dtype))
        return;

    throw std::runtime_error("Unsupported " + metricName + " " + inputName + " dtype: " +
                             TensorDescriptor::getElementTypeName(dtype) +
                             ". Supported dtypes are fp8_e4m3, fp8_e5m2, fp16, bf16, and fp32. "
                             "Reduction metric arithmetic and output use fp32.");
}

}  // namespace ThorImplementation::ReductionMetricDType
