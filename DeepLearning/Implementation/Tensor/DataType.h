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
    // FIXME: PACKED_BOOLEAN is broken for multi-dimensional case when rows are not multiple of 8 elements
    // FIXME: to fix this I need to round each dimension to (dimensionSize+7)/8 uint8_t's and I  need to save these dimensions off
    // and use them. So that say two rows do not share bits in a single uint8_t.
    PACKED_BOOLEAN = 22,
    FP8_E4M3 = 23,
    FP8_E5M2 = 24,
    BF16 = 25,
};

NLOHMANN_JSON_SERIALIZE_ENUM(DataType,
                             {
                                 {DataType::PACKED_BOOLEAN, "packed_boolean"},
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
                                 {DataType::FP8_E4M3, "fp8_e4m3"},
                                 {DataType::FP8_E5M2, "fp8_e5m2"},
                             })

}  // namespace ThorImplementation
