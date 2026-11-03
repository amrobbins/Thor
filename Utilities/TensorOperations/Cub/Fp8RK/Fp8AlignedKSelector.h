#pragma once

// Retained AlignedK nine-combination physical inventory and geometry
// descriptors. Includes the accepted, deterministic initial AlignedK heuristic.
// Candidate pruning is a conscious inventory/performance trade-off.
// Geometry legality is independent of sampling and GPU occupancy.

#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKLaunchLimits.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <cmath>
#include <vector>
#include <utility>

namespace ThorImplementation::CubReductionInternal::Fp8RK::Fp8AlignedKPolicy {

inline constexpr std::array<uint32_t, 2> packet_bytes = {2, 8};
inline constexpr std::array<uint32_t, 3> cta_threads = {32, 64, 128};
inline constexpr std::array<uint32_t, 2> staged_target_rows = {32, 128};

struct Variant {
    uint32_t packet;
    uint32_t threads;
    uint64_t shards_per_output;  // 1 => Complete; >1 => Staged, runtime geometry.
    constexpr bool operator==(const Variant&) const = default;
};
constexpr uint64_t ceilDiv(uint64_t n, uint64_t d) { return n / d + (n % d != 0); }

// Only three Complete physical specializations are retained. The six staged
// packet/CTA pairs support both runtime shard targets (32 and 128 rows).
constexpr bool supportedComplete(uint32_t packet, uint32_t threads) {
    return (packet == 2 && threads == 32) ||
           (packet == 8 && (threads == 32 || threads == 64));
}

inline bool legal(Variant v, uint64_t outer, uint64_t r, uint64_t k) {
    if (!outer || !r || !k || !v.shards_per_output) return false;
    if (std::find(packet_bytes.begin(), packet_bytes.end(), v.packet) == packet_bytes.end() ||
        std::find(cta_threads.begin(), cta_threads.end(), v.threads) == cta_threads.end() ||
        k % v.packet) return false;
    bool valid_shards = v.shards_per_output == 1;
    for (uint32_t target : staged_target_rows) {
        if (r > target && v.shards_per_output == ceilDiv(r, target)) valid_shards = true;
    }
    if (!valid_shards) return false;
    if (v.shards_per_output == 1 && !supportedComplete(v.packet, v.threads))
        return false;
    if (outer > UINT64_MAX / r || outer * r > UINT64_MAX / k) return false;
    const uint64_t tiles = ceilDiv(k, static_cast<uint64_t>(v.threads) * v.packet);
    if (outer > UINT64_MAX / v.shards_per_output ||
        outer * v.shards_per_output > UINT64_MAX / tiles) return false;
    // Staged outputs are FP32 and the intermediate footprint must be representable.
    if (v.shards_per_output > 1 &&
        outer * v.shards_per_output > UINT64_MAX / k / sizeof(float)) return false;
    // 1131: launch limits. reductionWorkGrid(shards, tiles, outer) throws above 2^31 - 1 per extent, and
    // balancedReductionShardRows keeps each shard's rows in a 32-bit counter (Complete: R itself).
    return fp8RKWorkGridFits(v.shards_per_output, tiles, outer) && ceilDiv(r, v.shards_per_output) <= FP8_RK_MAX_UINT32;
}

inline std::vector<Variant> candidates(uint64_t outer, uint64_t r, uint64_t k) {
    std::vector<Variant> result;
    for (uint32_t p : packet_bytes) {
        if (k % p) continue;
        for (uint32_t threads : cta_threads) {
            const Variant complete{p, threads, 1};
            if (supportedComplete(p, threads) && legal(complete, outer, r, k))
                result.push_back(complete);
            for (uint32_t target : staged_target_rows) {
                if (r <= target) continue;
                const Variant staged{p, threads, ceilDiv(r, target)};
                if (legal(staged, outer, r, k) &&
                    std::find(result.begin(), result.end(), staged) == result.end())
                    result.push_back(staged);
            }
        }
    }
    return result;
}

struct Geometry {
    uint64_t component_tiles;
    uint64_t first_stage_ctas;
    uint64_t rows_per_shard;
    uint64_t intermediate_bytes;
    double cta_work_waves;
    double input_mib;
};

inline Geometry describe(Variant v, uint64_t outer, uint64_t r, uint64_t k,
                         uint32_t sm_count, uint32_t resident_ctas_per_sm) {
    const uint64_t tiles = ceilDiv(k, static_cast<uint64_t>(v.packet) * v.threads);
    const uint64_t ctas = outer * v.shards_per_output * tiles;
    const uint64_t rows = ceilDiv(r, v.shards_per_output);
    const uint64_t bytes = v.shards_per_output == 1 ? 0 :
        outer * v.shards_per_output * k * sizeof(float);
    const double parallel = static_cast<double>(sm_count) * resident_ctas_per_sm;
    return {tiles, ctas, rows, bytes,
            parallel > 0 ? std::max(1.0, static_cast<double>(ctas) / parallel)
                         : std::numeric_limits<double>::infinity(),
            static_cast<double>(outer) * r * k / (1024.0 * 1024.0)};
}


// Accepted initial AlignedK heuristic for mode-local planning. It is not a
// guarantee of near-oracle performance: final independent held-out evaluation
// of the nine legal physical combinations remains to be completed.
// Topology and packet/CTA ownership are modeled in separate steps rather than
// through an overfitted latency polynomial. No cross-mode or production
// dispatch decision is made here.
template <typename ResidentCTAsPerSM>
std::optional<Variant> select(uint64_t outer, uint64_t r, uint64_t k,
                                          uint32_t sm_count,
                                          ResidentCTAsPerSM&& residency) {
    if (!sm_count || !outer || !r || !k ||
        outer > std::numeric_limits<uint64_t>::max() / r ||
        outer * r > std::numeric_limits<uint64_t>::max() / k)
        return std::nullopt;
    const auto available = candidates(outer, r, k);
    if (available.empty()) return std::nullopt;
    // A Complete launch saves continuation overhead for short reductions
    // with enough independent output-component work.
    const long double output_components =
        static_cast<long double>(outer) * static_cast<long double>(k);
    const bool prefer_complete = r <= 32 ||
        (r <= 256 && (output_components >= 300000.0L ||
                      (k <= 64 && output_components >= 180000.0L)));
    // A staged target is a desired rows/shard, not a compiled kernel depth.
    const long double bytes = static_cast<long double>(outer) * r * k;
    const uint32_t target_rows = bytes >= 50.0L * 1024 * 1024 ? 128 : 32;
    const uint32_t packet = k >= 128 && k % 8 == 0 ? 8 : 2;
    const uint64_t packets_per_row = ceilDiv(k, packet);
    uint32_t threads = packets_per_row < 32 ? 32 :
                       packets_per_row < 80 ? 64 : 128;
    if (prefer_complete)
        threads = packet == 8 && packets_per_row >= 32 ? 64 : 32;
    const uint64_t preferred_shards = prefer_complete ? 1 : ceilDiv(r, target_rows);
    const Variant preferred{packet, threads, preferred_shards};
    // Avoid treating a nonresident compiled specialization as a selectable
    // physical kernel. Fallbacks are deterministic and remain within inventory.
    if (legal(preferred, outer, r, k) && residency(preferred) > 0)
        return preferred;
    std::optional<Variant> best;
    uint64_t best_distance = std::numeric_limits<uint64_t>::max();
    for (const auto& v : available) {
        if (residency(v) == 0) continue;
        // Preserve preferred topology/depth as the highest priority, then
        // packet and CTA ownership. The fallback is only for illegal/nonresident
        // preferred variants, never the usual performance selection path.
        const uint64_t distance =
            (v.shards_per_output != preferred_shards ? 1000000ULL : 0ULL) +
            (v.packet != packet ? 10000ULL : 0ULL) +
            (v.threads > threads ? v.threads - threads : threads - v.threads);
        if (!best || distance < best_distance) {
            best = v;
            best_distance = distance;
        }
    }
    return best;
}

// Default to five legal physical choices for cached stamp-time profiling.
// An explicit larger limit allows the general tuner to cover more retained variants.
// Retain the initial heuristic as choice zero. A deterministic distance ranking
// alone fills the list with Staged variants and misses the Complete/Staged
// crossover, so reserve alternative topology and staging-depth representatives.
template <typename ResidentCTAsPerSM>
std::vector<Variant> shortlist(uint64_t outer, uint64_t r, uint64_t k,
                               uint32_t sm_count, ResidentCTAsPerSM&& residency,
                               size_t limit = 5) {
    std::vector<Variant> result;
    if (!limit || !sm_count || !outer || !r || !k) return result;
    // Preserve the existing five-way default, but permit all legal physical
    // alternatives when a cross-family tuner explicitly requests them.
    const auto variants = candidates(outer, r, k);
    limit = std::min(limit, variants.size());
    std::vector<Variant> available;
    for (Variant v : variants)
        if (residency(v) > 0) available.push_back(v);
    if (available.empty()) return result;
    // Memoize the occupancy decision so repeated local selection never
    // triggers another device-side residency query.
    const auto resident = [&](Variant v) -> uint32_t {
        return std::find(available.begin(), available.end(), v) != available.end() ? 1u : 0u;
    };
    const auto primary = select(outer, r, k, sm_count, resident);
    if (!primary) return result;
    result.push_back(*primary);
    if (result.size() == limit) return result;

    const auto rank = [&](Variant v) -> uint64_t {
        // Geometry proximity is used only to rank *alternatives*, never to
        // override the accepted first choice.
        const uint64_t packet_penalty = v.packet == primary->packet ? 0 : 10000;
        const uint64_t threads_penalty = v.threads > primary->threads ?
            v.threads - primary->threads : primary->threads - v.threads;
        const uint64_t shards_penalty = v.shards_per_output == primary->shards_per_output ? 0 : 500;
        return packet_penalty + threads_penalty + shards_penalty;
    };
    std::stable_sort(available.begin(), available.end(), [&](Variant a, Variant b) {
        return rank(a) < rank(b);
    });
    const auto appendIf = [&](auto&& predicate) {
        for (Variant v : available) {
            if (result.size() == limit) return;
            if (predicate(v) && std::find(result.begin(), result.end(), v) == result.end()) {
                result.push_back(v);
                return;
            }
        }
    };
    const bool primary_complete = primary->shards_per_output == 1;
    // Include the other launch topology, and a second CTA ownership for it
    // when physically available (e.g. Complete CTA32 beside CTA64).
    appendIf([&](Variant v) { return (v.shards_per_output == 1) != primary_complete; });
    if (result.size() < limit) {
        const Variant first_opposite = result.size() > 1 ? result[1] : *primary;
        appendIf([&](Variant v) {
            return (v.shards_per_output == 1) != primary_complete &&
                v.threads != first_opposite.threads;
        });
    }
    // For Staged, include both runtime target-row geometries where distinct.
    appendIf([&](Variant v) {
        return v.shards_per_output > 1 &&
            std::none_of(result.begin(), result.end(), [&](Variant chosen) {
                return chosen.shards_per_output == v.shards_per_output;
            });
    });
    // Deep enumeration for cross-family autotuning should sample a different
    // packet/CTA ownership before filling more same-packet configurations.
    // This particularly protects P2/CTA64 Staged when the initial preference
    // is P8: more packet bytes do not always imply a faster reduction.
    // Preserve the established five-way local shortlist unchanged.
    if (limit > 5 && primary->shards_per_output > 1) {
        const uint32_t alternate_packet = primary->packet == 8 ? 2 : 8;
        appendIf([&](Variant v) {
            return v.packet == alternate_packet && v.threads == 64 &&
                v.shards_per_output == primary->shards_per_output;
        });
    }
    // For deep cross-family enumeration, preserve the established early picks
    // and then cover every remaining P2 Staged CTA/depth combination before
    // filling additional P8 alternatives. On the large fast census, the
    // AlignedK oracle misses beyond rank32 were all P2 Staged; a simple rank
    // #9/#10 promotion cannot reach these candidates. This is packet/CTA/depth
    // diversity, not a latency prediction or a new compiled specialization.
    // Five-way mode-local callers retain their exact existing behavior.
    if (limit > 5) {
        for (Variant v : available) {
            if (result.size() == limit) break;
            if (v.packet == 2 && v.shards_per_output > 1 &&
                std::find(result.begin(), result.end(), v) == result.end())
                result.push_back(v);
        }
    }
    for (Variant v : available) {
        if (result.size() == limit) break;
        if (std::find(result.begin(), result.end(), v) == result.end()) result.push_back(v);
    }
    return result;
}

} // namespace ThorImplementation::CubReductionInternal::Fp8RK::Fp8AlignedKPolicy
