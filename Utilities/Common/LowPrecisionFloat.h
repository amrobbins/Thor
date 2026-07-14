#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace ThorLowPrecision {

inline uint16_t floatToFp16Bits(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

inline uint16_t floatToBf16Bits(float value) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

inline uint8_t floatToFp8E4M3Bits(float value) {
    return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

inline uint8_t floatToFp8E5M2Bits(float value) {
    return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E5M2);
}

inline uint16_t doubleToFp16Bits(double value) {
    return __half_as_ushort(__double2half(value));
}

inline uint16_t doubleToBf16Bits(double value) {
    return __bfloat16_as_ushort(__double2bfloat16(value));
}

inline uint8_t doubleToFp8E4M3Bits(double value) {
    return __nv_cvt_double_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

inline uint8_t doubleToFp8E5M2Bits(double value) {
    return __nv_cvt_double_to_fp8(value, __NV_SATFINITE, __NV_E5M2);
}

}  // namespace ThorLowPrecision
