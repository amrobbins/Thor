#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cudnn.h>

namespace ThorImplementation {

class CudnnHelper {
   public:
    static cudnnDataType_t getCudnnDataType(const TensorDescriptor::DataType dataType) {
        switch (dataType) {
            case TensorDescriptor::DataType::FP16:
                return CUDNN_DATA_HALF;
            case TensorDescriptor::DataType::FP32:
                return CUDNN_DATA_FLOAT;
            case TensorDescriptor::DataType::FP64:
                return CUDNN_DATA_DOUBLE;
            case TensorDescriptor::DataType::INT8:
                return CUDNN_DATA_INT8;
            case TensorDescriptor::DataType::UINT8:
                return CUDNN_DATA_UINT8;
            default:
                assert(false);  // Requested data type is not supported, see above for supported data types.
        }
    }
};

}  // namespace ThorImplementation
