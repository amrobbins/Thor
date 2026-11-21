#pragma once

#include <cstdint>
#include <limits>
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

// FP16 converts directly (in hardware from sm_89): the same bits as through FP32, where it is exact.
inline __host__ __device__ __nv_fp8_e4m3 toFp8E4M3Satfinite(half value) {
    return fp8E4M3FromBits(__nv_cvt_halfraw_to_fp8(static_cast<__half_raw>(value), __NV_SATFINITE, __NV_E4M3));
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

// Integers and bool convert through FP32, in hardware, which gives the same bits as through double: an integer below
// 2^24 in magnitude is exact in FP32, and one at or above it saturates either way (to +/-448, or +/-57344 for E5M2).
template <typename Integer, std::enable_if_t<std::is_integral_v<Integer>, int> = 0>
inline __host__ __device__ __nv_fp8_e4m3 toFp8E4M3Satfinite(Integer value) {
    return toFp8E4M3Satfinite(static_cast<float>(value));
}

// Canonical Thor FP8 narrowing saturates finite overflow for both E4M3 and E5M2.
// Keeping the policy here makes compiled kernels, host conversion paths, and generated
// expressions agree on overflow semantics instead of inheriting CUDA constructor defaults.
inline __host__ __device__ __nv_fp8_e5m2 toFp8E5M2Satfinite(float value) {
    return fp8E5M2FromBits(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E5M2));
}

inline __host__ __device__ __nv_fp8_e5m2 toFp8E5M2Satfinite(double value) {
    return fp8E5M2FromBits(__nv_cvt_double_to_fp8(value, __NV_SATFINITE, __NV_E5M2));
}

inline __host__ __device__ __nv_fp8_e5m2 toFp8E5M2Satfinite(half value) {
    return fp8E5M2FromBits(__nv_cvt_halfraw_to_fp8(static_cast<__half_raw>(value), __NV_SATFINITE, __NV_E5M2));
}

inline __host__ __device__ __nv_fp8_e5m2 toFp8E5M2Satfinite(__nv_bfloat16 value) {
    return toFp8E5M2Satfinite(__bfloat162float(value));
}

inline __host__ __device__ __nv_fp8_e5m2 toFp8E5M2Satfinite(__nv_fp8_e4m3 value) {
    return toFp8E5M2Satfinite(static_cast<float>(value));
}

inline __host__ __device__ __nv_fp8_e5m2 toFp8E5M2Satfinite(__nv_fp8_e5m2 value) {
    return value;
}

template <typename Integer, std::enable_if_t<std::is_integral_v<Integer>, int> = 0>
inline __host__ __device__ __nv_fp8_e5m2 toFp8E5M2Satfinite(Integer value) {
    return toFp8E5M2Satfinite(static_cast<float>(value));
}

template <typename T>
inline constexpr bool isFloatingStorageType = std::is_floating_point_v<T> || std::is_same_v<T, half> ||
                                              std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __nv_fp8_e4m3> ||
                                              std::is_same_v<T, __nv_fp8_e5m2>;

template <typename T>
inline constexpr bool isIntegerStorageType = std::is_integral_v<T> && !std::is_same_v<T, bool>;

// Thor's float -> integer rule: truncate toward zero and saturate to the integer's range (infinities included), on the
// CPU and the GPU. The GPU is PTX's cvt.rzi; 8- and 16-bit destinations take the 32-bit conversion and a clamp.
// NaN, on both sides, is what PTX documents for cvt (PTX ISA 9.0; the RTX 5090 does it, and LLVM folds the NVVM
// intrinsics to it): 0 unless the source is FP64 or the destination 64-bit, and otherwise 1 << (N - 1) of the
// conversion's N bits, i.e. the minimum for a signed type and 2^(N-1) for an unsigned one. After the clamp, FP64 NaN
// to 8 and 16 bits is the type's minimum when signed and its maximum when unsigned. The host returns the same values.
// A plain static_cast is undefined in C++ outside the range (x86 gives INT_MIN for 32-bit results, the low bits for
// narrower ones, and wraps unsigned ones), so the host tests the bounds first. The bounds are powers of two, exact in
// every float format. FP16, BF16 and FP8 go through FP32, which holds them exactly.
template <typename Integer, typename Float>
inline __host__ __device__ Integer toIntegerSaturating(Float value) {
    static_assert(isIntegerStorageType<Integer>, "an integer destination other than bool");
    static_assert(isFloatingStorageType<Float>, "a floating-point source");
    if constexpr (!std::is_floating_point_v<Float>) {
        return toIntegerSaturating<Integer>(static_cast<float>(value));
    } else {
        static_assert(std::is_same_v<Float, float> || std::is_same_v<Float, double>, "FP32 or FP64");
        constexpr int bits = 8 * static_cast<int>(sizeof(Integer));
#ifdef __CUDA_ARCH__
        constexpr bool isFloat = std::is_same_v<Float, float>;
        if constexpr (std::is_signed_v<Integer>) {
            if constexpr (bits == 64) {
                if constexpr (isFloat)
                    return static_cast<Integer>(__float2ll_rz(value));
                else
                    return static_cast<Integer>(__double2ll_rz(value));
            } else {
                int truncated;
                if constexpr (isFloat)
                    truncated = __float2int_rz(value);
                else
                    truncated = __double2int_rz(value);
                if constexpr (bits < 32) {
                    constexpr int low = -(1 << (bits - 1));
                    constexpr int high = (1 << (bits - 1)) - 1;
                    truncated = truncated < low ? low : (truncated > high ? high : truncated);
                }
                return static_cast<Integer>(truncated);
            }
        } else {
            if constexpr (bits == 64) {
                if constexpr (isFloat)
                    return static_cast<Integer>(__float2ull_rz(value));
                else
                    return static_cast<Integer>(__double2ull_rz(value));
            } else {
                unsigned int truncated;
                if constexpr (isFloat)
                    truncated = __float2uint_rz(value);
                else
                    truncated = __double2uint_rz(value);
                if constexpr (bits < 32) {
                    constexpr unsigned int high = (1u << bits) - 1u;
                    truncated = truncated > high ? high : truncated;
                }
                return static_cast<Integer>(truncated);
            }
        }
#else
        if (value != value) {
            if constexpr (std::is_same_v<Float, float> && bits < 64)
                return 0;
            else if constexpr (std::is_signed_v<Integer>)
                return std::numeric_limits<Integer>::min();
            else if constexpr (bits >= 32)
                return static_cast<Integer>(Integer(1) << (bits - 1));
            else
                return std::numeric_limits<Integer>::max();
        }
        if constexpr (std::is_signed_v<Integer>) {
            constexpr Float limit = static_cast<Float>(std::uint64_t(1) << (bits - 1));
            if (value >= limit)
                return std::numeric_limits<Integer>::max();
            if (value <= -limit)
                return std::numeric_limits<Integer>::min();
        } else {
            constexpr Float limit = static_cast<Float>(std::uint64_t(1) << (bits - 1)) * Float(2);
            if (value >= limit)
                return std::numeric_limits<Integer>::max();
            if (value <= Float(0))
                return 0;
        }
        return static_cast<Integer>(value);
#endif
    }
}

// Canonical Thor software narrowing into low-precision storage. Keep Thor's
// explicit saturation policy here so templated kernels do not accidentally inherit
// CUDA convenience-constructor behavior. Float -> integer follows toIntegerSaturating.
template <typename Destination, typename Source>
inline __host__ __device__ Destination castToStorage(Source value) {
    if constexpr (std::is_same_v<Destination, __nv_fp8_e4m3>) {
        return toFp8E4M3Satfinite(value);
    } else if constexpr (std::is_same_v<Destination, __nv_fp8_e5m2>) {
        return toFp8E5M2Satfinite(value);
    } else if constexpr (isFloatingStorageType<Source> && isIntegerStorageType<Destination>) {
        return toIntegerSaturating<Destination>(value);
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
    return toFp8E5M2Satfinite(value).__x;
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
    return toFp8E5M2Satfinite(value).__x;
}

}  // namespace ThorLowPrecision
