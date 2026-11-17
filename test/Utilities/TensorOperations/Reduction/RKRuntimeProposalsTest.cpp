// The RuntimeK physical inventory: every proposal is structurally valid, unique, legal under the RuntimeK policy and
// resident; a nonresident or empty site proposes nothing; occupancy failures propagate. Host-only (the occupancy
// callback is injected).

#include "Utilities/TensorOperations/Reduction/RK/Fp8/Fp8RuntimeKSelector.h"
#include "Utilities/TensorOperations/Reduction/RK/RKRuntimeProposals.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <set>
#include <stdexcept>
#include <tuple>

using namespace ThorImplementation::ReductionInternal::RK;
using namespace ThorImplementation::ReductionInternal::RK::Fp8;

TEST(RKRuntimeProposals, ProposalsAreValidUniqueAndLegal) {
    const std::tuple<uint64_t, uint64_t, uint64_t> sites[] = {
        {1, 512, 64}, {32, 1024, 511}, {256, 32, 8}, {7, 1024, 1024}, {16384, 65536, 512}};
    for (const auto& [outer, r, k] : sites) {
        SCOPED_TRACE(r);
        size_t occupancy_calls = 0;
        const auto proposals = enumerateRetainedRuntimeK({outer, r, k, 128}, [&](const PhysicalKernel& candidate) {
            EXPECT_EQ(candidate.family, PhysicalFamily::RuntimeK);
            ++occupancy_calls;
            return 2u;
        });
        std::set<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> seen;
        for (const auto& p : proposals) {
            EXPECT_TRUE(structurallyValid(p, outer, r, k));
            EXPECT_EQ(p.resident_ctas_per_sm, 2u);
            EXPECT_NE(p.a, 3u);  // only the three layouts
            EXPECT_TRUE(seen.emplace(p.a, p.b, p.c, p.d, p.e, p.f).second);
            const Fp8RuntimeKPolicy::Candidate candidate{
                {p.a == 0   ? Fp8RuntimeKPolicy::Layout::PacketComplete
                 : p.a == 1 ? Fp8RuntimeKPolicy::Layout::RotatedPrefix
                            : Fp8RuntimeKPolicy::Layout::RotatedExact,
                 static_cast<uint32_t>(p.b), static_cast<uint32_t>(p.c), static_cast<uint32_t>(p.d), p.e, p.f},
                p.first_stage_ctas,
                p.first_stage_scratch_bytes,
                p.resident_ctas_per_sm};
            EXPECT_TRUE(Fp8RuntimeKPolicy::legal(candidate, outer, r, k, 128));
        }
        EXPECT_GE(occupancy_calls, proposals.size());
    }
}

TEST(RKRuntimeProposals, NonresidentAndEmptySitesProposeNothing) {
    EXPECT_TRUE(enumerateRetainedRuntimeK({4, 1024, 512, 128}, [](const PhysicalKernel&) { return 0u; }).empty());
    EXPECT_TRUE(enumerateRetainedRuntimeK({0, 1024, 512, 128}, [](const PhysicalKernel&) -> uint32_t {
                    throw std::runtime_error("an empty site must not query occupancy");
                }).empty());
}

TEST(RKRuntimeProposals, OccupancyFailuresPropagate) {
    EXPECT_THROW((void)enumerateRetainedRuntimeK({4, 1024, 512, 128}, [](const PhysicalKernel&) -> uint32_t {
                     throw std::runtime_error("cuda failure");
                 }),
                 std::runtime_error);
}
