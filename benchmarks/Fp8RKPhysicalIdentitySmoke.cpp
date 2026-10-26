// Host-only build: g++ -std=c++23 -O2 -Wall -Wextra -Werror -I. benchmarks/Fp8RKPhysicalIdentitySmoke.cpp Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalIdentity.cpp -o /tmp/fp8-rk-physical-smoke
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalIdentity.h"
#include <cassert>
#include <cstdint>

namespace P = ThorImplementation::CubReductionInternal::Fp8RK;
int main() {
    const P::PhysicalKernel ow{P::PhysicalFamily::OneWindow, 1, 1, 16, 0, 0, 0, 6, 16, 0};
    assert(P::structurallyValid(ow, 128, 16, 3));
    auto bad = ow;
    bad.first_stage_ctas++;
    assert(!P::structurallyValid(bad, 128, 16, 3));
    const P::PhysicalKernel dr{P::PhysicalFamily::DeepRSmallK, 128, 0, 0, 0, 0, 0,
                               6, 32, 4096};
    assert(P::structurallyValid(dr, 2, 2048, 32));
    bad = dr; bad.first_stage_scratch_bytes -= 4;
    assert(!P::structurallyValid(bad, 2, 2048, 32));
    const P::PhysicalKernel ak{P::PhysicalFamily::AlignedK, 8, 32, 4, 0, 0, 0,
                               6, 8, 8192};
    assert(P::structurallyValid(ak, 2, 256, 256));
    bad = ak; bad.c = 0;
    assert(!P::structurallyValid(bad, 2, 256, 256));
    const P::PhysicalKernel packet{P::PhysicalFamily::RuntimeK,
        static_cast<uint64_t>(P::RuntimeLayout::PacketComplete), 2, 496, 2, 64, 1,
        6, 64, 0};
    assert(P::structurallyValid(packet, 128, 64, 512));
    const P::PhysicalKernel rotated{P::PhysicalFamily::RuntimeK,
        static_cast<uint64_t>(P::RuntimeLayout::RotatedExact), 0, 512, 1, 256, 8,
        6, 16, 32768};
    assert(P::structurallyValid(rotated, 2, 2048, 512));
    bad = rotated; bad.a = 3;
    assert(!P::structurallyValid(bad, 2, 2048, 512));
    bad = rotated; bad.resident_ctas_per_sm = 0;
    assert(!P::structurallyValid(bad, 2, 2048, 512));
    bad = rotated; bad.first_stage_scratch_bytes = UINT64_MAX;
    assert(!P::structurallyValid(bad, 2, 2048, 512));
}
