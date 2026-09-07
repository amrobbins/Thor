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

inline __host__ __device__ __nv_fp8_e5m2 fp8E5M2FromBits(__nv_fp8_storage_t bits) {
    __nv_fp8_e5m2 value;
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

// E5M2 represents +/-infinity. Ordinary Thor narrowing therefore uses CUDA's
// non-saturating conversion mode so finite overflow follows the destination
// format's native semantics instead of being clamped to +/-57344.
inline __host__ __device__ __nv_fp8_e5m2 toFp8E5M2Nosat(float value) {
    return fp8E5M2FromBits(__nv_cvt_float_to_fp8(value, __NV_NOSAT, __NV_E5M2));
}

inline __host__ __device__ __nv_fp8_e5m2 toFp8E5M2Nosat(double value) {
    return fp8E5M2FromBits(__nv_cvt_double_to_fp8(value, __NV_NOSAT, __NV_E5M2));
}

inline __host__ __device__ __nv_fp8_e5m2 toFp8E5M2Nosat(half value) {
    return toFp8E5M2Nosat(__half2float(value));
}

inline __host__ __device__ __nv_fp8_e5m2 toFp8E5M2Nosat(__nv_bfloat16 value) {
    return toFp8E5M2Nosat(__bfloat162float(value));
}

inline __host__ __device__ __nv_fp8_e5m2 toFp8E5M2Nosat(__nv_fp8_e4m3 value) {
    return toFp8E5M2Nosat(static_cast<float>(value));
}

inline __host__ __device__ __nv_fp8_e5m2 toFp8E5M2Nosat(__nv_fp8_e5m2 value) {
    return value;
}

template <typename Integer, std::enable_if_t<std::is_integral_v<Integer>, int> = 0>
inline __host__ __device__ __nv_fp8_e5m2 toFp8E5M2Nosat(Integer value) {
    return toFp8E5M2Nosat(static_cast<double>(value));
}

// Canonical Thor software narrowing into low-precision storage. Keep destination
// format overflow semantics here so templated kernels do not accidentally inherit
// CUDA convenience-constructor behavior.
template <typename Destination, typename Source>
inline __host__ __device__ Destination castToStorage(Source value) {
    if constexpr (std::is_same_v<Destination, __nv_fp8_e4m3>) {
        return toFp8E4M3Satfinite(value);
    } else if constexpr (std::is_same_v<Destination, __nv_fp8_e5m2>) {
        return toFp8E5M2Nosat(value);
    } else {
        return static_cast<Destination>(value);
    }
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
    return toFp8E5M2Nosat(value).__x;
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
    return toFp8E5M2Nosat(value).__x;
}

}  // namespace ThorLowPrecision
