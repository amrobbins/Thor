#pragma once

#include <nlohmann/json.hpp>

namespace ThorImplementation {

enum class DataType {
    FP16 = 10,
    FP32 = 11,
    FP64 = 12,
    INT8 = 13,
    INT16 = 14,
    INT32 = 15,
    INT64 = 16,
    UINT8 = 17,
    UINT16 = 18,
    UINT32 = 19,
    UINT64 = 20,
    BOOLEAN = 21,
    FP8_E4M3 = 22,
    FP8_E5M2 = 23,
    BF16 = 24,
    // Compute-only dtype token.  TF32 is not a tensor storage format in Thor; it requests
    // TensorFloat-32 cuBLAS/cuBLASLt compute for FP32 GEMM inputs/outputs.
    TF32 = 25,
};

NLOHMANN_JSON_SERIALIZE_ENUM(DataType,
                             {
                                 {DataType::BOOLEAN, "boolean"},
                                 {DataType::INT8, "int8"},
                                 {DataType::UINT8, "uint8"},
                                 {DataType::INT16, "int16"},
                                 {DataType::UINT16, "uint16"},
                                 {DataType::INT32, "int32"},
                                 {DataType::UINT32, "uint32"},
                                 {DataType::INT64, "int64"},
                                 {DataType::UINT64, "uint64"},
                                 {DataType::FP16, "fp16"},
                                 {DataType::FP32, "fp32"},
                                 {DataType::FP64, "fp64"},
                                 {DataType::BF16, "bf16"},
                                 {DataType::TF32, "tf32"},
                                 {DataType::FP8_E4M3, "fp8_e4m3"},
                                 {DataType::FP8_E5M2, "fp8_e5m2"},
                             })

}  // namespace ThorImplementation
