#pragma once

// Experimental RuntimeK mode-local physical selector. No production dispatch and no
// generalized cross-mode decision. Candidates are actual executable physical plans
// with independently queried compiled-kernel occupancy. The six lean rotated
// depths are an inventory decision; the full compiled ladder remains available as
// benchmark-only oracle controls while the selector is validated.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>
#include <utility>

namespace ThorImplementation::CubReductionBenchmarking::Fp8RuntimeKPolicy {

enum class Layout : uint8_t { PacketComplete, PacketStaged, RotatedPrefix, RotatedExact };

struct Variant {
    Layout layout;
    uint32_t warps_per_output;  // 0 for rotated; 1,2,4,8 for packet
    uint32_t tile_components;
    uint32_t tile_count;
    uint64_t rows_per_shard;
    uint64_t shards_per_output;
    constexpr bool operator==(const Variant&) const = default;
};

struct Candidate {
    Variant variant;
    uint64_t first_stage_ctas; // Physical CUDA grid.x, capped at 65535.
    uint64_t scratch_bytes;     // FP32 partials for staged variants; zero for Complete.
    uint32_t resident_ctas_per_sm; // Query the exact compiled specialization.
};

inline constexpr std::array<uint64_t, 6> rotated_row_depths = {32, 64, 128, 256, 512, 1024};

inline uint64_t ceilDiv(uint64_t n, uint64_t d) {
    return n / d + static_cast<uint64_t>(n % d != 0);
}
inline bool checkedMul(uint64_t a, uint64_t b, uint64_t& result) {
    if (b && a > UINT64_MAX / b) return false;
    result = a * b;
    return true;
}

// Fail closed for unexecutable/unstamped geometry or inconsistent metadata.
inline bool legal(const Candidate& c, uint64_t outer, uint64_t r, uint64_t k,
                  uint32_t sm_count) {
    if (!outer || !r || !k || !sm_count || !c.resident_ctas_per_sm ||
        !c.variant.tile_components || !c.variant.tile_count ||
        !c.variant.rows_per_shard || !c.variant.shards_per_output) return false;
    uint64_t elements = 0, multiplied = 0;
    if (!checkedMul(outer, r, elements) || !checkedMul(elements, k, multiplied)) return false;
    const auto& v = c.variant;
    if (ceilDiv(k, v.tile_components) != v.tile_count) return false;
    uint64_t blocks = 0, scratch = 0;
    if (v.layout == Layout::RotatedExact || v.layout == Layout::RotatedPrefix) {
        if (v.warps_per_output ||
            v.tile_components != (v.layout == Layout::RotatedExact ? 512u : 497u) ||
            std::find(rotated_row_depths.begin(), rotated_row_depths.end(), v.rows_per_shard)
                == rotated_row_depths.end() || v.rows_per_shard >= r ||
            v.shards_per_output != ceilDiv(r, v.rows_per_shard)) return false;
    } else {
        if ((v.warps_per_output != 1 && v.warps_per_output != 2 &&
             v.warps_per_output != 4 && v.warps_per_output != 8) ||
            v.tile_components > 496 || v.tile_components < 16 ||
            (v.layout == Layout::PacketComplete &&
             (v.shards_per_output != 1 || v.rows_per_shard != r)) ||
            (v.layout == Layout::PacketStaged &&
             (v.shards_per_output != 2 && v.shards_per_output != 4 &&
              v.shards_per_output != 8)) ||
            v.rows_per_shard != ceilDiv(r, v.shards_per_output) ||
            v.warps_per_output > v.rows_per_shard) return false;
    }
    if (!checkedMul(outer, v.shards_per_output, blocks) ||
        !checkedMul(blocks, v.tile_count, blocks)) return false;
    if (v.layout == Layout::PacketComplete || v.layout == Layout::PacketStaged) {
        blocks = ceilDiv(blocks, 8u / v.warps_per_output);
    }
    if (c.first_stage_ctas != std::min<uint64_t>(blocks, 65535)) return false;
    if (v.layout != Layout::PacketComplete) {
        if (v.shards_per_output < 2 || v.shards_per_output > r ||
            !checkedMul(outer, v.shards_per_output, scratch) ||
            !checkedMul(scratch, k, scratch) ||
            !checkedMul(scratch, uint64_t{4}, scratch)) return false;
    }
    return c.scratch_bytes == scratch;
}

// Initial hardware-based latency proxy, not a claim of oracle selection. The
// physical operands reflect CTA supply, partial-buffer/continuation traffic,
// K-tile fragmentation and the two launch topologies. No shape lookup tables.
// Coefficients are provisional: they must be assessed on a fresh holdout.
inline double score(const Candidate& c, uint64_t outer, uint64_t r, uint64_t k,
                    uint32_t sm_count) {
    const Variant& v = c.variant;
    const double input_mib = static_cast<double>(outer) * static_cast<double>(r) *
        static_cast<double>(k) / 1048576.0; // FP8, one byte per element.
    const double waves = static_cast<double>(c.first_stage_ctas) /
        (static_cast<double>(sm_count) * c.resident_ctas_per_sm);
    const double target_waves = std::clamp(0.71 * std::pow(input_mib / 32.0, 0.10), 0.5, 5.0);
    const double wave_delta = std::log(waves / target_waves);
    double cost = (4.65 * std::max(0.0, -wave_delta) +
                   1.91 * std::max(0.0, wave_delta)) *
                  std::pow(input_mib / 32.0, 0.30);
    cost += 0.133 * (static_cast<double>(c.scratch_bytes) / 1048576.0);
    cost += 0.267 * static_cast<double>(v.tile_count - 1);
    const bool packet = v.layout == Layout::PacketComplete || v.layout == Layout::PacketStaged;
    if (packet) {
        cost += 5.68 + 2.28 * std::max(0.0, std::log2(static_cast<double>(r) / 128.0));
        cost += 2.12 * std::max(0.0, 1.0 - waves);
        if (v.layout == Layout::PacketStaged) cost += 5.26;
    } else if (v.layout == Layout::RotatedExact) {
        cost += 0.296;
        if (k % 512 == 0) cost -= 0.589;
    } else {
        cost += 0.197 * std::max(0.0, static_cast<double>(v.tile_count) * 512.0 / k - 1.0);
    }
    return cost;
}

// Bounded legal shortlist for possible *future* stamp-time profiling. This does
// not replace the deterministic choice, allocate scratch, run autotuning, or
// change production dispatch. Keep the full physical configuration per entry.
// Stable ordering makes equal-score ties deterministic across repeated stamps.
inline std::vector<Variant> shortlist(uint64_t outer, uint64_t r, uint64_t k,
                                      uint32_t sm_count, std::span<const Candidate> candidates,
                                      size_t limit = 5) {
    std::vector<Variant> result;
    if (!outer || !r || !k || !sm_count || !limit) return result;
    std::vector<std::pair<double, Variant>> ranked;
    ranked.reserve(candidates.size());
    for (const Candidate& c : candidates) {
        if (!legal(c, outer, r, k, sm_count)) continue;
        const double s = score(c, outer, r, k, sm_count);
        if (std::isfinite(s)) ranked.emplace_back(s, c.variant);
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    result.reserve(std::min(limit, ranked.size()));
    for (const auto& [cost, variant] : ranked) {
        static_cast<void>(cost);
        if (std::find(result.begin(), result.end(), variant) != result.end()) continue;
        result.push_back(variant);
        if (result.size() == limit) break;
    }
    return result;
}

inline std::optional<Variant> select(uint64_t outer, uint64_t r, uint64_t k,
                                     uint32_t sm_count, std::span<const Candidate> candidates) {
    if (!outer || !r || !k || !sm_count) return std::nullopt;
    double best_cost = std::numeric_limits<double>::infinity();
    std::optional<Variant> best;
    for (const Candidate& candidate : candidates) {
        if (!legal(candidate, outer, r, k, sm_count)) continue;
        const double cost = score(candidate, outer, r, k, sm_count);
        if (std::isfinite(cost) && cost < best_cost) {
            best_cost = cost;
            best = candidate.variant;
        }
    }
    return best;
}

} // namespace ThorImplementation::CubReductionBenchmarking::Fp8RuntimeKPolicy
