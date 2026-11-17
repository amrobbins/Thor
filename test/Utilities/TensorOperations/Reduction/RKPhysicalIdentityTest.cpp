// The structural contract of an RK physical kernel identity: each family's fields must describe a launch that covers
// the site exactly (first-stage CTAs, packet, shards and scratch). Host-only.

#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalIdentity.h"

#include "gtest/gtest.h"

#include <cstdint>

namespace P = ThorImplementation::ReductionInternal::RK;

TEST(RKPhysicalIdentity, OneWindowCoversTheSiteWithItsCtas) {
    const P::PhysicalKernel ow{P::PhysicalFamily::OneWindow, 1, 1, 16, 0, 0, 0, 6, 16, 0};
    EXPECT_TRUE(P::structurallyValid(ow, 128, 16, 3));
    auto bad = ow;
    bad.first_stage_ctas++;
    EXPECT_FALSE(P::structurallyValid(bad, 128, 16, 3));
}

TEST(RKPhysicalIdentity, DeepRSmallKScratchHoldsEveryShard) {
    const P::PhysicalKernel dr{P::PhysicalFamily::DeepRSmallK, 128, 0, 0, 0, 0, 0, 6, 32, 4096};
    EXPECT_TRUE(P::structurallyValid(dr, 2, 2048, 32));
    auto bad = dr;
    bad.first_stage_scratch_bytes -= 4;
    EXPECT_FALSE(P::structurallyValid(bad, 2, 2048, 32));
}

TEST(RKPhysicalIdentity, AlignedKNeedsAShardCount) {
    const P::PhysicalKernel ak{P::PhysicalFamily::AlignedK, 8, 32, 4, 0, 0, 0, 6, 8, 8192};
    EXPECT_TRUE(P::structurallyValid(ak, 2, 256, 256));
    auto bad = ak;
    bad.c = 0;
    EXPECT_FALSE(P::structurallyValid(bad, 2, 256, 256));
}

TEST(RKPhysicalIdentity, RuntimeKLayoutsResidencyAndScratchAreChecked) {
    const P::PhysicalKernel packet{P::PhysicalFamily::RuntimeK,
                                   static_cast<uint64_t>(P::RuntimeLayout::PacketComplete), 2, 496, 2, 64, 1, 6, 64, 0};
    EXPECT_TRUE(P::structurallyValid(packet, 128, 64, 512));
    const P::PhysicalKernel rotated{P::PhysicalFamily::RuntimeK,
                                    static_cast<uint64_t>(P::RuntimeLayout::RotatedExact), 0, 512, 1, 256, 8, 6, 16,
                                    32768};
    EXPECT_TRUE(P::structurallyValid(rotated, 2, 2048, 512));
    auto bad = rotated;
    bad.a = 3;  // not a layout
    EXPECT_FALSE(P::structurallyValid(bad, 2, 2048, 512));
    bad = rotated;
    bad.resident_ctas_per_sm = 0;
    EXPECT_FALSE(P::structurallyValid(bad, 2, 2048, 512));
    bad = rotated;
    bad.first_stage_scratch_bytes = UINT64_MAX;
    EXPECT_FALSE(P::structurallyValid(bad, 2, 2048, 512));
}
