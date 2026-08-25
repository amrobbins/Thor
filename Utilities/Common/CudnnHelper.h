#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

#include <cudnn.h>

namespace ThorImplementation {

class CudnnHelper {
   public:
    static cudnnDataType_t getCudnnDataType(const DataType dataType) {
        switch (dataType) {
            case ThorImplementation::DataType::FP64:
                return CUDNN_DATA_DOUBLE;
            case ThorImplementation::DataType::FP32:
                return CUDNN_DATA_FLOAT;
            case ThorImplementation::DataType::FP16:
                return CUDNN_DATA_HALF;
            case ThorImplementation::DataType::BF16:
                return CUDNN_DATA_BFLOAT16;
            case ThorImplementation::DataType::INT8:
                return CUDNN_DATA_INT8;
            case ThorImplementation::DataType::INT32:
                return CUDNN_DATA_INT32;
            case ThorImplementation::DataType::UINT8:
                return CUDNN_DATA_UINT8;
            case ThorImplementation::DataType::INT64:
                return CUDNN_DATA_INT64;
            default:
                THOR_UNREACHABLE();
        }
        THOR_UNREACHABLE();
        return CUDNN_DATA_HALF;
    }
};

}  // namespace ThorImplementation
