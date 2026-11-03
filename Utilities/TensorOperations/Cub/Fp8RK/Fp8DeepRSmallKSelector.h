#pragma once

// Retained FP8 SUM DeepRSmallK local planning policy. This is the P16 compiled
// five-depth inventory, not the older production narrow-flat dispatch or a
// generalized cross-mode selector. Valid for E4M3/E5M2 with FP32 partials and
// a separately stamped production FP32 SUM continuation.
//
// The model ranks COMPLETE reductions. It combines (1) CTA staging-group
// waves, (2) the critical-path staging rounds within a CTA, and (3) FP32
// partial traffic. The constant input read volume is common to all depths.
// Keep the original coefficients unchanged for the five-depth holdout.

#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKLaunchLimits.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>
#include <utility>

namespace ThorImplementation::CubReductionInternal::Fp8RK::Fp8DeepRSmallKPolicy {

inline constexpr uint32_t packet_bytes = 16;
inline constexpr std::array<uint32_t, 5> compiled_rows = {64, 128, 256, 512, 1024};
inline constexpr uint32_t warp_count = 8;
inline constexpr uint32_t max_group_bytes = 32 * packet_bytes - (packet_bytes - 1); // 497

struct Variant {
    uint32_t rows_per_shard;
    constexpr bool operator==(const Variant&) const = default;
};

constexpr uint64_t ceilDiv(uint64_t n, uint64_t d) {
    return n / d + (n % d != 0); // d is nonzero at every call site
}

// A staged reduction requires >=2 shards, so a row depth must be strictly
// smaller than R. Do not confuse the benchmark's sampled R/K with legality.
constexpr bool legal(Variant v, uint64_t outer, uint64_t r, uint64_t k) {
    if (outer == 0 || k == 0 || k > 256 || r == 0 || r <= v.rows_per_shard) return false;
    if (std::find(compiled_rows.begin(), compiled_rows.end(), v.rows_per_shard) == compiled_rows.end())
        return false;
    if (outer > std::numeric_limits<uint64_t>::max() / r) return false;
    if (outer * r > std::numeric_limits<uint64_t>::max() / k) return false;
    const uint64_t shards = ceilDiv(r, v.rows_per_shard);
    if (outer > std::numeric_limits<uint64_t>::max() / shards) return false;
    // 1131: the launcher's reductionWorkGrid (one component tile) throws above 2^31 - 1 shards or outer.
    return fp8RKWorkGridFits(shards, 1, outer);
}

struct Cost {
    Variant variant;
    uint64_t shards_per_output;
    uint32_t rows_per_warp_group;
    uint64_t full_shard_rounds;
    double staging_group_waves;
    double critical_rounds;
    double fp32_partial_mb;
    double total_score;
};

inline Cost describe(Variant v, uint64_t outer, uint64_t r, uint64_t k,
                     uint32_t sm_count, uint32_t resident_ctas_per_sm) {
    const uint64_t rows = v.rows_per_shard;
    const uint64_t shards = ceilDiv(r, rows);
    const uint64_t group_rows = std::min<uint64_t>(rows / warp_count, max_group_bytes / k);
    const uint64_t rows_per_round = warp_count * group_rows;
    const uint64_t full_rounds = ceilDiv(rows, rows_per_round);
    const uint64_t last_rows = r - (shards - 1) * rows;
    const uint64_t last_rounds = ceilDiv(last_rows, rows_per_round);
    // The final shard can be shorter; do not bill it as a full-depth shard.
    const double all_group_rounds = static_cast<double>(outer) *
        (static_cast<double>(shards - 1) * full_rounds + last_rounds);
    const double staging_waves = std::max(static_cast<double>(full_rounds),
        all_group_rounds / (static_cast<double>(sm_count) * resident_ctas_per_sm));
    const double partial_mb = static_cast<double>(outer) * shards * k * sizeof(float) / 1e6;
    // Frozen initial candidate score from a deliberately minimal physical model.
    // Coefficients are dimensionless ranking weights, not modeled microseconds.
    const double score = staging_waves + 0.5 * full_rounds + 0.25 * partial_mb;
    return {v, shards, static_cast<uint32_t>(group_rows), full_rounds,
            staging_waves, static_cast<double>(full_rounds), partial_mb, score};
}

// The caller supplies *compiled-kernel* residency for the concrete FP8 dtype,
// K-ownership variant, P16 and depth, from the existing CUDA occupancy query.
// Skip nonresident kernels. The policy itself never creates any CUDA context.
// 1172: the single-pick select() had no callers; the ranker orders shortlist().
// Every retained compiled depth fits in a five-way stamp-time shortlist.
// Thus for this mode the full legal/resident local oracle is included.
template <typename ResidentCTAsPerSM>
std::vector<Variant> shortlist(uint64_t outer, uint64_t r, uint64_t k,
                               uint32_t sm_count, ResidentCTAsPerSM&& residency,
                               size_t limit = 5) {
    std::vector<Variant> result;
    if (!sm_count || !limit) return result;
    limit = std::min<size_t>(limit, 5);
    std::vector<std::pair<double, Variant>> ranked;
    for (uint32_t depth : compiled_rows) {
        const Variant v{depth};
        if (!legal(v, outer, r, k)) continue;
        const uint32_t resident = residency(v);
        if (!resident) continue;
        ranked.emplace_back(describe(v, outer, r, k, sm_count, resident).total_score, v);
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [score, v] : ranked) {
        static_cast<void>(score);
        if (result.size() == limit) break;
        result.push_back(v);
    }
    return result;
}

} // namespace ThorImplementation::CubReductionInternal::Fp8RK::Fp8DeepRSmallKPolicy
