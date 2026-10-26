#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKRuntimeProposals.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RuntimeKSelector.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ThorImplementation::CubReductionInternal::Fp8RK {
namespace {
constexpr uint64_t MaxGrid = 65535;
uint64_t ceilDiv(uint64_t n, uint64_t d) { return n / d + (n % d != 0); }
uint32_t queriedResidency(const RuntimeProposalOccupancy& occupancy, const PhysicalKernel& p) {
    // A missing retained compiled specialization is an expected candidate
    // exclusion; CUDA/runtime failures must still propagate to the caller.
    try { return occupancy(p); }
    catch (const std::logic_error&) { return 0; }
}
bool checkedMultiply(uint64_t a, uint64_t b, uint64_t& out) {
    if (b && a > UINT64_MAX / b) return false;
    out = a * b;
    return true;
}
uint64_t cappedBlocks(uint64_t outer, uint64_t shards, uint64_t tiles, uint64_t outputs_per_block) {
    uint64_t v = 0;
    if (!checkedMultiply(outer, shards, v) || !checkedMultiply(v, tiles, v))
        throw std::overflow_error("RuntimeK proposal CTA count overflow");
    return std::min<uint64_t>(MaxGrid, ceilDiv(v, outputs_per_block));
}
struct PacketGeometry {
    uint32_t warps = 0, components = 0, tiles = 0;
    uint64_t blocks = 0;
    bool operator==(const PacketGeometry&) const = default;
};

// Retained PacketComplete generation follows the benchmark's two-wave
// adaptive-K geometry policy. PacketStaged is not compiled and never proposed.
PacketGeometry packetForWaves(const RuntimeProposalGeometry& g, uint32_t occupancy, uint32_t waves) {
    const uint64_t target = uint64_t{g.sm_count} * std::max<uint32_t>(1, std::min(waves, occupancy));
    const uint64_t min_tiles = ceilDiv(g.k, uint64_t{496});
    const uint64_t max_tiles = std::max(min_tiles, ceilDiv(g.k, uint64_t{16}));
    if (min_tiles > UINT32_MAX || max_tiles > UINT32_MAX)
        throw std::overflow_error("RuntimeK proposal tile count exceeds uint32");
    const auto make = [&](uint32_t warps, uint64_t requested_tiles) {
        const uint64_t bounded = std::clamp(requested_tiles, min_tiles, max_tiles);
        const uint32_t components = static_cast<uint32_t>(ceilDiv(g.k, bounded));
        const uint32_t tiles = static_cast<uint32_t>(ceilDiv(g.k, components));
        return PacketGeometry{warps, components, tiles,
                              cappedBlocks(g.outer, 1, tiles, 8 / warps)};
    };
    PacketGeometry best{};
    uint64_t best_overfill = UINT64_MAX;
    bool natural = false;
    for (uint32_t warps : {1u, 2u, 4u, 8u}) {
        if (warps > g.r) continue;
        const auto p = make(warps, min_tiles);
        if (p.blocks >= target && (!natural || p.blocks - target < best_overfill ||
            (p.blocks - target == best_overfill && warps < best.warps))) {
            natural = true; best = p; best_overfill = p.blocks - target;
        }
    }
    if (natural) return best;
    bool have = false, reached = false;
    uint64_t fewest_extra = UINT64_MAX, distance = UINT64_MAX;
    for (uint32_t warps : {1u, 2u, 4u, 8u}) {
        if (warps > g.r) continue;
        uint64_t numerator = 0;
        if (!checkedMultiply(target, 8 / warps, numerator))
            throw std::overflow_error("RuntimeK target block calculation overflow");
        const uint64_t requested = std::clamp(ceilDiv(numerator, g.outer), min_tiles, max_tiles);
        const auto p = make(warps, requested);
        const uint64_t extra = p.tiles > min_tiles ? p.tiles - min_tiles : 0;
        const bool hit = p.blocks >= target;
        const uint64_t delta = hit ? p.blocks - target : target - p.blocks;
        if (!have || (hit && !reached) || (hit == reached &&
            (extra < fewest_extra || (extra == fewest_extra &&
            (delta < distance || (delta == distance && warps < best.warps)))))) {
            have = true; reached = hit; fewest_extra = extra; distance = delta; best = p;
        }
    }
    return best;
}
}

std::vector<PhysicalKernel> enumerateRetainedRuntimeK(
    const RuntimeProposalGeometry& g, const RuntimeProposalOccupancy& occupancy) {
    std::vector<PhysicalKernel> out;
    if (!g.outer || !g.r || !g.k || !g.sm_count) return out;
    if (!occupancy) throw std::invalid_argument("RuntimeK proposal requires exact compiled occupancy");
    uint64_t elements = 0;
    if (!checkedMultiply(g.outer, g.r, elements) || !checkedMultiply(elements, g.k, elements))
        throw std::overflow_error("RuntimeK proposal input size overflow");
    const auto add = [&](PhysicalKernel p) {
        if (std::find_if(out.begin(), out.end(), [&](const PhysicalKernel& old) {
            return old.family == p.family && old.a == p.a && old.b == p.b &&
                   old.c == p.c && old.d == p.d && old.e == p.e && old.f == p.f;
        }) != out.end()) return;
        p.resident_ctas_per_sm = queriedResidency(occupancy, p);
        if (p.resident_ctas_per_sm && structurallyValid(p, g.outer, g.r, g.k)) out.push_back(p);
    };
    // PacketComplete's geometry depends on measured occupancy. Query the
    // physical compiled variant once; warps/tile components are runtime knobs.
    PhysicalKernel packet{};
    packet.family = PhysicalFamily::RuntimeK;
    packet.a = static_cast<uint64_t>(RuntimeLayout::PacketComplete);
    packet.b = 1; packet.c = 16; packet.d = ceilDiv(g.k, 16);
    packet.e = g.r; packet.f = 1;
    packet.first_stage_ctas = cappedBlocks(g.outer, 1, packet.d, 8);
    const uint32_t packet_residency = queriedResidency(occupancy, packet);
    if (packet_residency) {
        for (uint32_t waves : {1u, 2u}) {
            const auto p = packetForWaves(g, packet_residency, waves);
            if (!p.warps) continue;
            PhysicalKernel variant{};
            variant.family = PhysicalFamily::RuntimeK;
            variant.a = static_cast<uint64_t>(RuntimeLayout::PacketComplete);
            variant.b = p.warps; variant.c = p.components; variant.d = p.tiles;
            variant.e = g.r; variant.f = 1;
            variant.first_stage_ctas = p.blocks;
            add(variant);
        }
    }
    // All six retained depths are physically compiled. The selector, rather
    // than a hidden runtime heuristic, decides which should make the top 28.
    for (RuntimeLayout layout : {RuntimeLayout::RotatedPrefix, RuntimeLayout::RotatedExact}) {
        const uint64_t width = layout == RuntimeLayout::RotatedExact ? 512 : 497;
        const uint64_t tiles = ceilDiv(g.k, width);
        if (tiles > UINT32_MAX) continue;
        for (uint64_t rows : Fp8RuntimeKPolicy::rotated_row_depths) {
            if (rows >= g.r) continue;
            const uint64_t shards = ceilDiv(g.r, rows);
            uint64_t partials = 0;
            if (!checkedMultiply(g.outer, shards, partials) ||
                !checkedMultiply(partials, g.k, partials) ||
                !checkedMultiply(partials, 4, partials)) continue;
            PhysicalKernel variant{};
            variant.family = PhysicalFamily::RuntimeK;
            variant.a = static_cast<uint64_t>(layout);
            variant.c = width; variant.d = tiles;
            variant.e = rows; variant.f = shards;
            variant.first_stage_ctas = cappedBlocks(g.outer, shards, tiles, 1);
            variant.first_stage_scratch_bytes = partials;
            add(variant);
        }
    }
    return out;
}

} // namespace ThorImplementation::CubReductionInternal::Fp8RK
