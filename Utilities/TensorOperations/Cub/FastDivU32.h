#pragma once

// 1138: division of a 32-bit dividend (< 2^31) by a launch-invariant 32-bit divisor with one multiply-high and one
// shift (the round-up method of Granlund and Montgomery, as in CUTLASS FastDivmod). Kernels index flat packet spaces
// with it instead of a 64-bit division per element. Usable on the host for tests.

#include <cstdint>
#include <stdexcept>

#if defined(__CUDACC__)
#define THOR_FAST_DIV_HD __host__ __device__ __forceinline__
#else
#define THOR_FAST_DIV_HD inline
#endif

namespace ThorImplementation::CubReductionInternal {

struct FastDivU32 {
    uint32_t divisor = 1;
    uint32_t multiplier = 0;  // 0 for divisor 1
    uint32_t shift = 0;

    /** Dividends must be below this bound. */
    static constexpr uint64_t MAX_DIVIDEND = uint64_t{1} << 31;

    static FastDivU32 make(uint32_t divisor) {
        if (divisor == 0) throw std::invalid_argument("FastDivU32 divisor must be nonzero.");
        FastDivU32 result;
        result.divisor = divisor;
        if (divisor == 1) return result;
        uint32_t ceil_log2 = 0;
        while ((uint64_t{1} << ceil_log2) < divisor) ++ceil_log2;
        const uint32_t p = 31 + ceil_log2;
        result.multiplier = static_cast<uint32_t>(((uint64_t{1} << p) + divisor - 1) / divisor);
        result.shift = p - 32;
        return result;
    }

    [[nodiscard]] THOR_FAST_DIV_HD uint32_t divide(uint32_t dividend) const {
        if (divisor == 1) return dividend;
#if defined(__CUDA_ARCH__)
        return __umulhi(dividend, multiplier) >> shift;
#else
        return static_cast<uint32_t>((static_cast<uint64_t>(dividend) * multiplier) >> 32) >> shift;
#endif
    }
};

}  // namespace ThorImplementation::CubReductionInternal

#undef THOR_FAST_DIV_HD
