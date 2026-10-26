#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKRuntimeProposals.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RuntimeKSelector.h"

#include <cassert>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace ThorImplementation::CubReductionInternal::Fp8RK;

void testGeometry(uint64_t outer, uint64_t r, uint64_t k) {
    size_t occupancy_calls = 0;
    const auto result = enumerateRetainedRuntimeK({outer, r, k, 128},
        [&](const PhysicalKernel& candidate) {
            assert(candidate.family == PhysicalFamily::RuntimeK);
            ++occupancy_calls;
            return 2u;
        });
    std::set<std::tuple<uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t>> seen;
    for (const auto& p : result) {
        assert(structurallyValid(p, outer, r, k));
        assert(p.resident_ctas_per_sm == 2);
        assert(p.a != 3); // only the 3 retained layouts
        assert(seen.emplace(p.a,p.b,p.c,p.d,p.e,p.f).second);
        Fp8RuntimeKPolicy::Candidate c{
            {p.a == 0 ? Fp8RuntimeKPolicy::Layout::PacketComplete :
             p.a == 1 ? Fp8RuntimeKPolicy::Layout::RotatedPrefix :
                        Fp8RuntimeKPolicy::Layout::RotatedExact,
             static_cast<uint32_t>(p.b), static_cast<uint32_t>(p.c),
             static_cast<uint32_t>(p.d), p.e, p.f},
            p.first_stage_ctas, p.first_stage_scratch_bytes, p.resident_ctas_per_sm};
        assert(Fp8RuntimeKPolicy::legal(c, outer, r, k, 128));
    }
    assert(occupancy_calls >= result.size());
}

int main() {
    testGeometry(1, 512, 64);
    testGeometry(32, 1024, 511);
    testGeometry(256, 32, 8);
    testGeometry(7, 1024, 1024);
    testGeometry(16384, 65536, 512);
    // Nonresident specializations never enter the physical inventory.
    assert(enumerateRetainedRuntimeK({4, 1024, 512, 128},
        [](const PhysicalKernel&) { return 0u; }).empty());
    assert(enumerateRetainedRuntimeK({0, 1024, 512, 128},
        [](const PhysicalKernel&) { throw std::runtime_error("should not run"); return 0u; }).empty());
    bool propagated = false;
    try {
        (void) enumerateRetainedRuntimeK({4, 1024, 512, 128},
            [](const PhysicalKernel&) -> uint32_t { throw std::runtime_error("cuda failure"); });
    } catch (const std::runtime_error&) { propagated = true; }
    assert(propagated);
}
