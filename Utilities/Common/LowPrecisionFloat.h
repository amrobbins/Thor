#pragma once

#include <cstdint>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace ThorLowPrecision {

inline __host__ __device__ __nv_fp8_e4m3 fp8E4M3FromBits(__nv_fp8_storage_t bits) {
    __nv_fp8_e4m3 value;
    value.__x = bits;
    return value;
}

inline __host__ __device__ __nv_fp8_e4m3 toFp8E4M3Satfinite(float value) {
    return fp8E4M3FromBits(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

inline __host__ __device__ __nv_fp8_e4m3 toFp8E4M3Satfinite(double value) {
    return fp8E4M3FromBits(__nv_cvt_double_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

inline __host__ __device__ __nv_fp8_e4m3 toFp8E4M3Satfinite(half value) {
    return toFp8E4M3Satfinite(__half2float(value));
}

inline __host__ __device__ __nv_fp8_e4m3 toFp8E4M3Satfinite(__nv_bfloat16 value) {
    return toFp8E4M3Satfinite(__bfloat162float(value));
}

inline __host__ __device__ __nv_fp8_e4m3 toFp8E4M3Satfinite(__nv_fp8_e4m3 value) {
    return value;
}

inline __host__ __device__ __nv_fp8_e4m3 toFp8E4M3Satfinite(__nv_fp8_e5m2 value) {
    return toFp8E4M3Satfinite(static_cast<float>(value));
}

template <typename Integer, std::enable_if_t<std::is_integral_v<Integer>, int> = 0>
inline __host__ __device__ __nv_fp8_e4m3 toFp8E4M3Satfinite(Integer value) {
    // Values in E4M3's finite range are represented exactly by double for every
    // integer source type. Larger magnitudes saturate to +/-448 as required.
    return toFp8E4M3Satfinite(static_cast<double>(value));
}

inline uint16_t floatToFp16Bits(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

inline uint16_t floatToBf16Bits(float value) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

inline uint8_t floatToFp8E4M3Bits(float value) {
    return toFp8E4M3Satfinite(value).__x;
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
    return toFp8E4M3Satfinite(value).__x;
}

inline uint8_t doubleToFp8E5M2Bits(double value) {
    return __nv_cvt_double_to_fp8(value, __NV_SATFINITE, __NV_E5M2);
}

}  // namespace ThorLowPrecision
