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

// 1192: the FP32 inventory (plans/Thor_FP32_Dense_Reduction_Plan, phase 5). The fields mean what they mean for FP8, in
// FP32 components: a 16-byte packet holds 4. OneWindow: P16 only, K <= 125 / 61 / 29 / 13 at O1 / O2 / O4 / O8 (the
// window less a 0..3 prefix; aligned K up to 128 / 64 / 32 / 16). DeepRSmallK: K <= 125, FP8's five depths. AlignedK:
// 4-, 8- or 16-byte packets (1, 2 or 4 components; K a multiple; the 4-byte packet since 1228), CTA 32 / 64 / 128, any
// 1 <= shards <= R (one shard is Complete). RuntimeK: PacketComplete with 4 <= tile <= 124, and RotatedPrefix with the
// 125-component tile (no Exact).
bool fp32StructurallyValid(const PhysicalKernel& p, uint64_t outer, uint64_t r, uint64_t k) {
    uint64_t input_elements = 0, scratch_elems = 0, scratch_bytes = 0, blocks = 0;
    if (!safeProduct(outer, r, input_elements) ||
        !safeProduct(input_elements, k, input_elements)) return false;
    switch (p.family) {
        case PhysicalFamily::OneWindow: {
            const bool legal_ownership = (p.a == 1 || p.a == 2 || p.a == 4 || p.a == 8) &&
                (p.b == 1 || p.b == 2 || p.b == 4 || p.b == 8) && (p.a == 1 || p.b == 1);
            if (!legal_ownership || p.c != 16 || p.staged()) return false;
            const uint64_t lanes = 32 / p.a;
            const uint64_t capacity = lanes * 4 - (k % 4 == 0 ? 0 : 3);
            return k <= capacity && r >= p.b &&
                   p.first_stage_ctas == ceilDiv(outer, (8 / p.b) * p.a);
        }
        case PhysicalFamily::DeepRSmallK:
            if (k > 125 || p.a >= r ||
                !(p.a == 64 || p.a == 128 || p.a == 256 || p.a == 512 || p.a == 1024)) return false;
            return safeProduct(outer, ceilDiv(r, p.a), blocks) &&
                   safeProduct(blocks, k, scratch_elems) &&
                   safeProduct(scratch_elems, 4, scratch_bytes) &&
                   p.first_stage_ctas == blocks && p.first_stage_scratch_bytes == scratch_bytes;
        case PhysicalFamily::AlignedK: {
            if ((p.a != 4 && p.a != 8 && p.a != 16) ||
                (p.b != 32 && p.b != 64 && p.b != 128) || !p.c || p.c > r || k % (p.a / 4)) return false;
            if (p.c != 1 && (!safeProduct(outer, p.c, scratch_elems) ||
                !safeProduct(scratch_elems, k, scratch_elems) ||
                !safeProduct(scratch_elems, 4, scratch_bytes))) return false;
            return safeProduct(outer, p.c, blocks) &&
                   safeProduct(blocks, ceilDiv(k, (p.a / 4) * p.b), blocks) &&
                   p.first_stage_ctas == blocks && p.first_stage_scratch_bytes == scratch_bytes;
        }
        case PhysicalFamily::RuntimeK: {
            if (!p.c || p.d != ceilDiv(k, p.c) || !p.e || !p.f) return false;
            const RuntimeLayout layout = static_cast<RuntimeLayout>(p.a);
            if (layout == RuntimeLayout::PacketComplete) {
                if (p.f != 1 || p.e != r || p.c < 4 || p.c > 124 ||
                    !(p.b == 1 || p.b == 2 || p.b == 4 || p.b == 8) || p.staged()) return false;
            } else if (layout == RuntimeLayout::RotatedPrefix) {
                if (p.b || p.e >= r || p.f != ceilDiv(r, p.e) ||
                    !(p.e == 32 || p.e == 64 || p.e == 128 || p.e == 256 || p.e == 512 || p.e == 1024) ||
                    p.c != 125u) return false;
                if (!safeProduct(outer, p.f, scratch_elems) ||
                    !safeProduct(scratch_elems, k, scratch_elems) ||
                    !safeProduct(scratch_elems, 4, scratch_bytes)) return false;
            } else return false;  // RotatedExact is FP8-only.
            if (!safeProduct(outer, p.f, blocks) || !safeProduct(blocks, p.d, blocks)) return false;
            if (layout == RuntimeLayout::PacketComplete) blocks = ceilDiv(blocks, 8 / p.b);
            return p.first_stage_ctas == (blocks < 65535 ? blocks : 65535) &&
                   p.first_stage_scratch_bytes == scratch_bytes;
        }
    }
    return false;
}

// 1246: the FP16 / BF16 inventory (plans/Thor_FP16_BF16_Dense_Reduction_Plan, phase 5), in 2-byte components: a 16-byte
// packet holds 8. OneWindow: P16 only, K <= 249 / 121 / 57 / 25 at O1 / O2 / O4 / O8 (the window less a 0..7 prefix;
// aligned K up to 256 / 128 / 64 / 32; the residency query excludes ownerships with no 2-byte unit at K). DeepRSmallK:
// K <= 249, FP8's five depths. AlignedK: 2-, 4-, 8- or 16-byte packets (1, 2, 4 or 8 components; K a multiple), CTA
// 32 / 64 / 128, any 1 <= shards <= R. RuntimeK: PacketComplete with 8 <= tile <= 248, and RotatedPrefix with the
// 249-component tile (no Exact).
bool f16StructurallyValid(const PhysicalKernel& p, uint64_t outer, uint64_t r, uint64_t k) {
    uint64_t input_elements = 0, scratch_elems = 0, scratch_bytes = 0, blocks = 0;
    if (!safeProduct(outer, r, input_elements) ||
        !safeProduct(input_elements, k, input_elements)) return false;
    switch (p.family) {
        case PhysicalFamily::OneWindow: {
            const bool legal_ownership = (p.a == 1 || p.a == 2 || p.a == 4 || p.a == 8) &&
                (p.b == 1 || p.b == 2 || p.b == 4 || p.b == 8) && (p.a == 1 || p.b == 1);
            if (!legal_ownership || p.c != 16 || p.staged()) return false;
            const uint64_t lanes = 32 / p.a;
            const uint64_t capacity = lanes * 8 - (k % 8 == 0 ? 0 : 7);
            return k <= capacity && r >= p.b &&
                   p.first_stage_ctas == ceilDiv(outer, (8 / p.b) * p.a);
        }
        case PhysicalFamily::DeepRSmallK:
            if (k > 249 || p.a >= r ||
                !(p.a == 64 || p.a == 128 || p.a == 256 || p.a == 512 || p.a == 1024)) return false;
            return safeProduct(outer, ceilDiv(r, p.a), blocks) &&
                   safeProduct(blocks, k, scratch_elems) &&
                   safeProduct(scratch_elems, 4, scratch_bytes) &&
                   p.first_stage_ctas == blocks && p.first_stage_scratch_bytes == scratch_bytes;
        case PhysicalFamily::AlignedK: {
            if ((p.a != 2 && p.a != 4 && p.a != 8 && p.a != 16) ||
                (p.b != 32 && p.b != 64 && p.b != 128) || !p.c || p.c > r || k % (p.a / 2)) return false;
            if (p.c != 1 && (!safeProduct(outer, p.c, scratch_elems) ||
                !safeProduct(scratch_elems, k, scratch_elems) ||
                !safeProduct(scratch_elems, 4, scratch_bytes))) return false;
            return safeProduct(outer, p.c, blocks) &&
                   safeProduct(blocks, ceilDiv(k, (p.a / 2) * p.b), blocks) &&
                   p.first_stage_ctas == blocks && p.first_stage_scratch_bytes == scratch_bytes;
        }
        case PhysicalFamily::RuntimeK: {
            if (!p.c || p.d != ceilDiv(k, p.c) || !p.e || !p.f) return false;
            const RuntimeLayout layout = static_cast<RuntimeLayout>(p.a);
            if (layout == RuntimeLayout::PacketComplete) {
                if (p.f != 1 || p.e != r || p.c < 8 || p.c > 248 ||
                    !(p.b == 1 || p.b == 2 || p.b == 4 || p.b == 8) || p.staged()) return false;
            } else if (layout == RuntimeLayout::RotatedPrefix) {
                if (p.b || p.e >= r || p.f != ceilDiv(r, p.e) ||
                    !(p.e == 32 || p.e == 64 || p.e == 128 || p.e == 256 || p.e == 512 || p.e == 1024) ||
                    p.c != 249u) return false;
                if (!safeProduct(outer, p.f, scratch_elems) ||
                    !safeProduct(scratch_elems, k, scratch_elems) ||
                    !safeProduct(scratch_elems, 4, scratch_bytes)) return false;
            } else return false;  // RotatedExact is FP8-only.
            if (!safeProduct(outer, p.f, blocks) || !safeProduct(blocks, p.d, blocks)) return false;
            if (layout == RuntimeLayout::PacketComplete) blocks = ceilDiv(blocks, 8 / p.b);
            return p.first_stage_ctas == (blocks < 65535 ? blocks : 65535) &&
                   p.first_stage_scratch_bytes == scratch_bytes;
        }
    }
    return false;
}
}  // namespace

bool structurallyValid(const PhysicalKernel& p, uint64_t outer, uint64_t r, uint64_t k) {
    if (!outer || !r || !k || !p.resident_ctas_per_sm || !p.first_stage_ctas) return false;
    if (p.element_bytes == 4) return fp32StructurallyValid(p, outer, r, k);
    if (p.element_bytes == 2) return f16StructurallyValid(p, outer, r, k);  // 1246
    if (p.element_bytes != 1) return false;  // 1178 / 1192 / 1246: FP8, FP16 / BF16 and FP32 have inventories
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
