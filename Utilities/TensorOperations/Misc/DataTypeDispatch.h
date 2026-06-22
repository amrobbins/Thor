#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ThorImplementation::MiscTensorOperationSupport {

template <typename Fn>
decltype(auto) dispatchTensorDataType(DataType dtype, Fn&& fn) {
    switch (dtype) {
        case DataType::FP16:
            return fn.template operator()<half>();
        case DataType::FP32:
            return fn.template operator()<float>();
        case DataType::FP64:
            return fn.template operator()<double>();
        case DataType::BF16:
            return fn.template operator()<__nv_bfloat16>();
        case DataType::FP8_E4M3:
            return fn.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2:
            return fn.template operator()<__nv_fp8_e5m2>();
        case DataType::INT8:
            return fn.template operator()<int8_t>();
        case DataType::INT16:
            return fn.template operator()<int16_t>();
        case DataType::INT32:
            return fn.template operator()<int32_t>();
        case DataType::INT64:
            return fn.template operator()<int64_t>();
        case DataType::UINT8:
            return fn.template operator()<uint8_t>();
        case DataType::UINT16:
            return fn.template operator()<uint16_t>();
        case DataType::UINT32:
            return fn.template operator()<uint32_t>();
        case DataType::UINT64:
            return fn.template operator()<uint64_t>();
        case DataType::BOOLEAN:
            return fn.template operator()<bool>();
        default:
            throw std::runtime_error("Unsupported tensor dtype " + TensorDescriptor::getElementTypeName(dtype) + ".");
    }
}

}  // namespace ThorImplementation::MiscTensorOperationSupport
