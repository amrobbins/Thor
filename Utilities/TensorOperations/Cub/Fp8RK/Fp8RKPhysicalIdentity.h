#pragma once

// CUDA-independent physical-plan identity. Shared by the selector binder and
// four independently compiled family adapters. No XGBoost model, CUDA headers,
// or kernel templates here.
#include <cstdint>

namespace ThorImplementation::CubReductionInternal::Fp8RK {

enum class PhysicalFamily : uint8_t { OneWindow, DeepRSmallK, AlignedK, RuntimeK };
enum class RuntimeLayout : uint8_t { PacketComplete, RotatedPrefix, RotatedExact };

struct PhysicalKernel {
    PhysicalFamily family;
    // OneWindow: a=outputs/warp b=warps/output c=packet bytes
    // DeepRSmallK: a=rows/shard
    // AlignedK: a=packet bytes b=CTA threads c=shards/output
    // RuntimeK: a=RuntimeLayout b=warps/output c=tile components
    //           d=tile count e=rows/shard f=shards/output
    uint64_t a = 0, b = 0, c = 0, d = 0, e = 0, f = 0;
    uint32_t resident_ctas_per_sm = 0;
    uint64_t first_stage_ctas = 0;
    uint64_t first_stage_scratch_bytes = 0;
    // 1178: input element width in bytes (FP8 1, FP32 4). The same geometry fields name different kernels at another
    // width (a 16-byte packet holds 16 FP8 or 4 FP32 components), so identities of different widths never compare
    // equal. Last, so existing positional initializers keep FP8's 1.
    uint8_t element_bytes = 1;

    [[nodiscard]] bool staged() const { return first_stage_scratch_bytes != 0; }
    bool operator==(const PhysicalKernel&) const = default;
};

/** 1178: false for any element width without a retained inventory. 1192: FP8 (1) and FP32 (4) have one; the FP32
 *  rules are in Fp8RKPhysicalIdentity.cpp. */
[[nodiscard]] bool structurallyValid(const PhysicalKernel& kernel,
                                      uint64_t outer, uint64_t r, uint64_t k);

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
