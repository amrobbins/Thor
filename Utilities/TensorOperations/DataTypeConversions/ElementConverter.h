#pragma once

#include "Utilities/Common/LowPrecisionFloat.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

// What the precompiled kernels that convert between element types share: the conversion of one element, by Thor's
// rules, and the accesses that move a thread's elements. TypeConverter's kernels and the converting dense concatenate
// and split use these, so a tensor converted on its way through a concatenate holds the bits that converting it first
// would give.
namespace ThorImplementation {

// A floating-point value to an integer by Thor's rule (truncate, saturate; ThorLowPrecision::toIntegerSaturating);
// to FP8 saturating finite overflow; every other pair as C++ converts the base types.
template <typename FROM_TYPE, typename TO_TYPE>
struct ElementConverter {
    __device__ __forceinline__ TO_TYPE operator()(FROM_TYPE x) const {
        if constexpr (ThorLowPrecision::isFloatingStorageType<FROM_TYPE> && ThorLowPrecision::isIntegerStorageType<TO_TYPE>)
            return ThorLowPrecision::toIntegerSaturating<TO_TYPE>(x);
        else
            return static_cast<TO_TYPE>(x);
    }
};

template <typename FROM_TYPE>
struct ElementConverter<FROM_TYPE, __nv_fp8_e4m3> {
    __device__ __forceinline__ __nv_fp8_e4m3 operator()(FROM_TYPE x) const {
        return ThorLowPrecision::toFp8E4M3Satfinite(x);
    }
};

template <typename FROM_TYPE>
struct ElementConverter<FROM_TYPE, __nv_fp8_e5m2> {
    __device__ __forceinline__ __nv_fp8_e5m2 operator()(FROM_TYPE x) const {
        return ThorLowPrecision::toFp8E5M2Satfinite(x);
    }
};

// BYTES bytes (1, 2, 4, 8, or a multiple of 16) from memory to a thread's elements, in the widest accesses that fit;
// from is on a boundary of the access (16 bytes for 16, 32 for more).
template <int BYTES>
__device__ __forceinline__ void loadConvertBytes(void *to, const void *from) {
    if constexpr (BYTES >= 16) {
        const uint4 *source = static_cast<const uint4 *>(__builtin_assume_aligned(from, BYTES >= 32 ? 32 : 16));
#pragma unroll
        for (int i = 0; i < BYTES / 16; ++i)
            static_cast<uint4 *>(to)[i] = source[i];
    } else if constexpr (BYTES == 8) {
        *static_cast<uint2 *>(to) = *static_cast<const uint2 *>(from);
    } else if constexpr (BYTES == 4) {
        *static_cast<unsigned int *>(to) = *static_cast<const unsigned int *>(from);
    } else if constexpr (BYTES == 2) {
        *static_cast<unsigned short *>(to) = *static_cast<const unsigned short *>(from);
    } else {
        static_assert(BYTES == 1, "accesses of 1 to 32 bytes");
        *static_cast<unsigned char *>(to) = *static_cast<const unsigned char *>(from);
    }
}

// The same accesses from a thread's elements to memory.
template <int BYTES>
__device__ __forceinline__ void storeConvertBytes(void *to, const void *from) {
    if constexpr (BYTES >= 16) {
        uint4 *dest = static_cast<uint4 *>(__builtin_assume_aligned(to, BYTES >= 32 ? 32 : 16));
#pragma unroll
        for (int i = 0; i < BYTES / 16; ++i)
            dest[i] = static_cast<const uint4 *>(from)[i];
    } else if constexpr (BYTES == 8) {
        *static_cast<uint2 *>(to) = *static_cast<const uint2 *>(from);
    } else if constexpr (BYTES == 4) {
        *static_cast<unsigned int *>(to) = *static_cast<const unsigned int *>(from);
    } else if constexpr (BYTES == 2) {
        *static_cast<unsigned short *>(to) = *static_cast<const unsigned short *>(from);
    } else {
        static_assert(BYTES == 1, "accesses of 1 to 32 bytes");
        *static_cast<unsigned char *>(to) = *static_cast<const unsigned char *>(from);
    }
}

}  // namespace ThorImplementation
