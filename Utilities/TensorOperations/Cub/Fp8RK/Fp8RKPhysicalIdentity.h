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

    [[nodiscard]] bool staged() const { return first_stage_scratch_bytes != 0; }
    bool operator==(const PhysicalKernel&) const = default;
};

[[nodiscard]] bool structurallyValid(const PhysicalKernel& kernel,
                                      uint64_t outer, uint64_t r, uint64_t k);

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
