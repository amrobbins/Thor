// 1138: FastDivU32 (multiply-high division by a launch-invariant divisor) matches integer division for every dividend
// below 2^31: small divisors exhaustively over boundary dividends, powers of two and their neighbours, and random
// divisors and dividends. Host-only (the device path is the same multiply-high via __umulhi).

#include "Utilities/TensorOperations/Cub/FastDivU32.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

using ThorImplementation::CubReductionInternal::FastDivU32;

namespace {
void expectDivides(uint32_t divisor, std::mt19937_64& rng) {
    const FastDivU32 fast = FastDivU32::make(divisor);
    constexpr uint32_t Max = static_cast<uint32_t>(FastDivU32::MAX_DIVIDEND - 1);
    std::vector<uint32_t> dividends{0, 1, divisor - 1, divisor, Max, Max - 1, Max / divisor * divisor,
                                    Max / divisor * divisor - 1};
    if (divisor < Max) dividends.push_back(divisor + 1);
    for (int i = 0; i < 64; ++i) dividends.push_back(static_cast<uint32_t>(rng() % FastDivU32::MAX_DIVIDEND));
    for (const uint32_t n : dividends) {
        if (n > Max) continue;
        ASSERT_EQ(n / divisor, fast.divide(n)) << "n=" << n << " d=" << divisor;
    }
}
}  // namespace

TEST(FastDivU32, MatchesIntegerDivision) {
    std::mt19937_64 rng(1138);
    for (uint32_t d = 1; d <= 4096; ++d) expectDivides(d, rng);
    for (uint32_t k = 1; k < 32; ++k) {
        const uint64_t p = uint64_t{1} << k;
        for (const uint64_t d : {p - 1, p, p + 1})
            if (d >= 1 && d <= 0xFFFFFFFFull) expectDivides(static_cast<uint32_t>(d), rng);
    }
    for (int i = 0; i < 2000; ++i) expectDivides(static_cast<uint32_t>(rng() % 0xFFFFFFFFull) + 1, rng);
}

TEST(FastDivU32, RejectsZero) { EXPECT_THROW((void)FastDivU32::make(0), std::invalid_argument); }
