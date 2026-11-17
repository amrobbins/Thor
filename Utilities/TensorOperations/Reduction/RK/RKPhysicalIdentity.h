#pragma once

// CUDA-independent physical-plan identity. Shared by the selector binder and
// four independently compiled family adapters. No XGBoost model, CUDA headers,
// or kernel templates here.
#include <cstdint>

namespace ThorImplementation::ReductionInternal::RK {

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
    // Input element width in bytes (FP8 1, FP16 / BF16 2, FP32 4). The same geometry fields name different kernels at
    // another width (a 16-byte packet holds 16 FP8 or 4 FP32 components), so identities of different widths never
    // compare equal. Last, so existing positional initializers keep FP8's 1.
    uint8_t element_bytes = 1;

    [[nodiscard]] bool staged() const { return first_stage_scratch_bytes != 0; }
    bool operator==(const PhysicalKernel&) const = default;
};

/** False for any element width without a compiled inventory. FP8 (1), FP16 / BF16 (2) and FP32 (4) have one; the
 *  FP16 / BF16 and FP32 rules are in RKPhysicalIdentity.cpp. */
[[nodiscard]] bool structurallyValid(const PhysicalKernel& kernel,
                                      uint64_t outer, uint64_t r, uint64_t k);

/** The largest K (elements) a VALUE OneWindow pass of this ownership reads at an input base off its packet boundary.
 *  The kernel's windows follow the address, so every row group then carries a prefix of up to a packet less one
 *  element, at any K: the lane subgroup's window less that. (At an aligned base, K a multiple of the packet has no
 *  prefix and may fill the window.) */
[[nodiscard]] constexpr uint64_t oneWindowUnalignedCapacity(uint64_t outputs_per_warp, uint64_t packet_bytes,
                                                            uint64_t element_bytes) {
    const uint64_t packet_elements = packet_bytes / element_bytes;
    return (32 / outputs_per_warp) * packet_elements - (packet_elements - 1);
}

/** The input base alignment an FP8 RK VALUE pass needs at inner size `k`: the RuntimeK passes any byte address (the
 *  rotated lanes align their windows down and rotate the prefix away; PacketComplete takes its prefixes from the base
 *  address), AlignedK its packet (2 or 8 bytes), DeepRSmallK 16 bytes. OneWindow any byte address when K fits
 *  oneWindowUnalignedCapacity, else 16 bytes. */
[[nodiscard]] uint32_t rkPassInputAlignment(const PhysicalKernel& kernel, uint64_t k);

}  // namespace ThorImplementation::ReductionInternal::RK
