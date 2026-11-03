// 1131 (UINT64 survey gaps 1, 2, 4): the retained FP8 selection policies reject shapes their launchers would throw on,
// so a stamp never ranks a kernel that cannot launch. Host-only.

#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8AlignedKSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8DeepRSmallKSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8OneWindowSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKLaunchLimits.h"

#include "gtest/gtest.h"

#include <cstdint>

using namespace ThorImplementation::CubReductionInternal;
using namespace ThorImplementation::CubReductionInternal::Fp8RK;

namespace {
constexpr uint64_t U32 = 0xFFFFFFFFull;
constexpr uint64_t G31 = (uint64_t{1} << 31) - 1;
}  // namespace

TEST(Fp8RKLaunchLimits, Helpers) {
    EXPECT_TRUE(fp8RKElementsFitUint32(1, 1, U32));
    EXPECT_TRUE(fp8RKElementsFitUint32(65535, 65537, 1));  // 2^32 - 1
    EXPECT_FALSE(fp8RKElementsFitUint32(65536, 65536, 1));  // 2^32
    EXPECT_FALSE(fp8RKElementsFitUint32(uint64_t{1} << 40, uint64_t{1} << 40, 2));  // no uint64 overflow
    EXPECT_TRUE(fp8RKWorkGridFits(G31, G31, G31));
    EXPECT_FALSE(fp8RKWorkGridFits(G31 + 1, 1, 1));
    EXPECT_FALSE(fp8RKWorkGridFits(1, G31 + 1, 1));
    EXPECT_FALSE(fp8RKWorkGridFits(1, 1, G31 + 1));
    EXPECT_EQ(U32 - 8, FP8_RK_PACKET_COMPLETE_MAX_ROWS);
}

// OneWindow keeps 32-bit byte offsets: no variant is offered for an input of 2^32 elements or more.
TEST(Fp8RKLaunchLimits, OneWindowRejectsFourGiBInputs) {
    const auto resident = [](Fp8OneWindowPolicy::Variant) -> uint32_t { return 4; };
    // 2^20 outputs x R 1024 x K 4 = 2^32 elements; one fewer output fits.
    EXPECT_FALSE(Fp8OneWindowPolicy::select(uint64_t{1} << 20, 1024, 4, 170, resident).has_value());
    EXPECT_TRUE(Fp8OneWindowPolicy::shortlist(uint64_t{1} << 20, 1024, 4, 170, resident).empty());
    EXPECT_TRUE(Fp8OneWindowPolicy::select((uint64_t{1} << 20) - 1, 1024, 4, 170, resident).has_value());
    EXPECT_FALSE(Fp8OneWindowPolicy::shortlist((uint64_t{1} << 20) - 1, 1024, 4, 170, resident).empty());
}

// DeepRSmallK and AlignedK launch through reductionWorkGrid: outer and shards at most 2^31 - 1.
TEST(Fp8RKLaunchLimits, GridExtentCaps) {
    EXPECT_TRUE(Fp8DeepRSmallKPolicy::legal({64}, G31, 1000, 16));
    EXPECT_FALSE(Fp8DeepRSmallKPolicy::legal({64}, G31 + 1, 1000, 16));
    EXPECT_TRUE(Fp8DeepRSmallKPolicy::legal({64}, 1, G31 * 64, 16));
    EXPECT_FALSE(Fp8DeepRSmallKPolicy::legal({64}, 1, G31 * 64 + 1, 16));  // 2^31 shards
    // The DeepRSmallK ARG twin shares the VALUE legality (gap 4).
    EXPECT_TRUE(fp8DeepRSmallKArgSupports(G31, 1000, 16, 64));
    EXPECT_FALSE(fp8DeepRSmallKArgSupports(G31 + 1, 1000, 16, 64));

    EXPECT_TRUE(Fp8AlignedKPolicy::legal({8, 64, 1}, G31, 16, 512));
    EXPECT_FALSE(Fp8AlignedKPolicy::legal({8, 64, 1}, G31 + 1, 16, 512));
    // Complete: R is the 32-bit per-shard row counter.
    EXPECT_TRUE(Fp8AlignedKPolicy::legal({8, 64, 1}, 1, U32, 512));
    EXPECT_FALSE(Fp8AlignedKPolicy::legal({8, 64, 1}, 1, U32 + 1, 512));
}
