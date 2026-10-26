#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalIdentity.h"

#include <cstdint>
#include <limits>

namespace ThorImplementation::CubReductionInternal::Fp8RK {
namespace {
uint64_t ceilDiv(uint64_t n, uint64_t d) { return n / d + (n % d != 0); }
bool safeProduct(uint64_t a, uint64_t b, uint64_t& result) {
    if (b != 0 && a > UINT64_MAX / b) return false;
    result = a * b;
    return true;
}
}

bool structurallyValid(const PhysicalKernel& p, uint64_t outer, uint64_t r, uint64_t k) {
    if (!outer || !r || !k || !p.resident_ctas_per_sm || !p.first_stage_ctas) return false;
    uint64_t input_elements = 0, scratch_elems = 0, scratch_bytes = 0, blocks = 0;
    if (!safeProduct(outer, r, input_elements) ||
        !safeProduct(input_elements, k, input_elements)) return false;
    switch (p.family) {
        case PhysicalFamily::OneWindow: {
            const bool legal_ownership = (p.a == 1 || p.a == 2 || p.a == 4 || p.a == 8) &&
                (p.b == 1 || p.b == 2 || p.b == 4 || p.b == 8) && (p.a == 1 || p.b == 1);
            if (!legal_ownership || (p.c != 4 && p.c != 8 && p.c != 16) || p.staged()) return false;
            const uint64_t lanes = 32 / p.a;
            const uint64_t capacity = lanes * p.c - (k % p.c == 0 ? 0 : p.c - 1);
            return k <= capacity && r >= p.b &&
                   p.first_stage_ctas == ceilDiv(outer, (8 / p.b) * p.a);
        }
        case PhysicalFamily::DeepRSmallK:
            if (k > 256 || p.a >= r ||
                !(p.a == 64 || p.a == 128 || p.a == 256 || p.a == 512 || p.a == 1024)) return false;
            return safeProduct(outer, ceilDiv(r, p.a), blocks) &&
                   safeProduct(blocks, k, scratch_elems) &&
                   safeProduct(scratch_elems, 4, scratch_bytes) &&
                   p.first_stage_ctas == blocks && p.first_stage_scratch_bytes == scratch_bytes;
        case PhysicalFamily::AlignedK:
            if ((p.a != 2 && p.a != 8) ||
                (p.b != 32 && p.b != 64 && p.b != 128) || !p.c || p.c > r || k % p.a) return false;
            if (p.c != 1 && (!safeProduct(outer, p.c, scratch_elems) ||
                !safeProduct(scratch_elems, k, scratch_elems) ||
                !safeProduct(scratch_elems, 4, scratch_bytes))) return false;
            return safeProduct(outer, p.c, blocks) &&
                   safeProduct(blocks, ceilDiv(k, p.a * p.b), blocks) &&
                   p.first_stage_ctas == blocks && p.first_stage_scratch_bytes == scratch_bytes;
        case PhysicalFamily::RuntimeK: {
            if (!p.c || p.c > 512 || p.d != ceilDiv(k, p.c) || !p.e || !p.f) return false;
            const RuntimeLayout layout = static_cast<RuntimeLayout>(p.a);
            if (layout == RuntimeLayout::PacketComplete) {
                if (p.f != 1 || p.e != r || p.c < 16 || p.c > 496 ||
                    !(p.b == 1 || p.b == 2 || p.b == 4 || p.b == 8) || p.staged()) return false;
            } else if (layout == RuntimeLayout::RotatedPrefix || layout == RuntimeLayout::RotatedExact) {
                if (p.b || p.e >= r || p.f != ceilDiv(r, p.e) ||
                    !(p.e == 32 || p.e == 64 || p.e == 128 || p.e == 256 || p.e == 512 || p.e == 1024) ||
                    p.c != (layout == RuntimeLayout::RotatedExact ? 512u : 497u)) return false;
                if (!safeProduct(outer, p.f, scratch_elems) ||
                    !safeProduct(scratch_elems, k, scratch_elems) ||
                    !safeProduct(scratch_elems, 4, scratch_bytes)) return false;
            } else return false; // PacketStaged is not compiled/retained.
            if (!safeProduct(outer, p.f, blocks) || !safeProduct(blocks, p.d, blocks)) return false;
            if (layout == RuntimeLayout::PacketComplete) blocks = ceilDiv(blocks, 8 / p.b);
            return p.first_stage_ctas == (blocks < 65535 ? blocks : 65535) &&
                   p.first_stage_scratch_bytes == scratch_bytes;
        }
    }
    return false;
}

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
