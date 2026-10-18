#pragma once

// Accepted OneWindow-local heuristic for the ten retained physical FP8 variants.
// This is a stamp-time/host policy, not production dispatch. The calibrated constants
// were frozen after 891/892 development and the 896 diagnostic comparison.
// Residency is supplied for the actual compiled kernel specialization; do not use
// the device-wide maximum CTA count as a surrogate for compiled occupancy.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>

namespace ThorImplementation::CubReductionBenchmarking::Fp8OneWindowPolicy {

struct Variant {
    uint32_t outputs_per_warp;
    uint32_t warps_per_output;
    uint32_t packet_bytes;
    constexpr bool operator==(const Variant&) const = default;
};

// The ten retained candidates. Demoted oracle controls are not selectable.
inline constexpr std::array<Variant, 10> candidates = {{{1,1,16}, {1,2,16}, {1,4,16},
    {1,8,16}, {2,1,16}, {4,1,16}, {8,1,16}, {8,1,4}, {4,1,8}, {8,1,8}}};

// Benchmark specialization availability is separate from geometry legality.
constexpr bool compiledK(uint64_t k) {
    switch (k) {
        case 1: case 2: case 3: case 4: case 5: case 7: case 8: case 9:
        case 11: case 13: case 15: case 16: case 17: case 19: case 31: case 32:
        case 63: case 64: case 127: case 128: case 129: case 255: case 256:
        case 497: case 512: return true;
        default: return false;
    }
}

constexpr bool legal(Variant v, uint64_t r, uint64_t k) {
    if (r < v.warps_per_output || k == 0 || v.outputs_per_warp == 0 ||
        (v.outputs_per_warp > 1 && v.warps_per_output != 1)) return false;
    const uint64_t window = (32u / v.outputs_per_warp) * v.packet_bytes;
    const uint64_t capacity = k % v.packet_bytes == 0 ? window : window - (v.packet_bytes - 1);
    return k <= capacity;
}

constexpr uint64_t ceilDiv(uint64_t a, uint64_t b) { return a / b + (a % b != 0); }
constexpr uint64_t ctaCount(uint64_t outer, Variant v) {
    return ceilDiv(outer, (8u / v.warps_per_output) * v.outputs_per_warp);
}

namespace Detail {
struct Score {
    Variant variant{};
    double frozen = 0;
    double waves = 0;
    double useful_lane_fraction = 0;
    double trial = 0;
    uint64_t rows_per_group = 0;
};

inline bool variantLess(Variant a, Variant b) {
    if (a.outputs_per_warp != b.outputs_per_warp) return a.outputs_per_warp < b.outputs_per_warp;
    if (a.warps_per_output != b.warps_per_output) return a.warps_per_output < b.warps_per_output;
    // Matches the existing Python stable sort of the textual variant names.
    if (a.packet_bytes == b.packet_bytes) return false;
    if (a.packet_bytes == 16) return true;
    if (b.packet_bytes == 16) return false;
    return a.packet_bytes < b.packet_bytes;
}

inline bool scoreLess(double a, double b, Variant va, Variant vb) {
    return a < b || (a == b && variantLess(va, vb));
}

inline Score describe(Variant v, uint64_t outer, uint64_t r, uint64_t k,
                      uint32_t sms, uint32_t resident_ctas_per_sm) {
    const uint64_t lanes = 32u / v.outputs_per_warp;
    const uint64_t workers = k <= lanes ? std::max<uint64_t>(1, lanes / k) : 1;
    const uint64_t components = k <= lanes ? 1 : ceilDiv(k, lanes);
    const uint64_t rows = (lanes * v.packet_bytes - (k % v.packet_bytes == 0 ? 0 : v.packet_bytes - 1)) / k;
    const uint64_t rounds = ceilDiv(r, v.warps_per_output * rows);
    const uint64_t ctas = ctaCount(outer, v);
    const double waves = std::max(1.0, static_cast<double>(ctas) /
                                    (static_cast<double>(sms) * resident_ctas_per_sm));
    const double lane_util = k > lanes ? static_cast<double>(k) / (lanes * components) :
                                               static_cast<double>(k * workers) / lanes;
    const uint64_t full = r / (v.warps_per_output * rows);
    const uint64_t tail = std::min(r % (v.warps_per_output * rows), rows);
    const uint64_t decode_iters = components * (full * ceilDiv(rows, 4 * workers) +
                                                 (tail == 0 ? 0 : ceilDiv(tail, 4 * workers)));
    const double stage_util = static_cast<double>(r) * static_cast<double>(k) /
                              (static_cast<double>(rounds) * v.warps_per_output * lanes * v.packet_bytes);
    const double avg_prefix = k % v.packet_bytes == 0 ? 0.0 : (v.packet_bytes - 1) / 2.0;
    const double packets = std::ceil((rows * static_cast<double>(k) + avg_prefix) / v.packet_bytes);
    const double packet_amp = std::max(1.0, packets * v.packet_bytes / (rows * static_cast<double>(k)));
    const double packet_lanes = std::min(1.0, v.outputs_per_warp * packets / 32.0);
    const double supplied_warps = std::min(static_cast<double>(ctas) / sms,
                                           static_cast<double>(resident_ctas_per_sm)) * 8.0 *
                                  (1.0 - .499 * (1.0 - packet_lanes));
    const double gate = 1.0 + 1.284 * std::max(0.0, 1.094 / std::max(supplied_warps, .1) - 1.0);
    const double frozen = std::pow(waves, .4018) *
        (3.9502 + rounds * (1.0 + 1.2989 * (1.0 - lane_util) +
                            .03016 * (1.0 - stage_util) + .00309 * (packet_amp - 1.0)) +
         .19559 * (static_cast<double>(decode_iters) / (1.0 + rounds)) +
         .00784 * std::log2(static_cast<double>(v.warps_per_output))) * gate;
    return {v, frozen, waves, lane_util,
            frozen + .20 * waves + 1.0 * waves * (1.0 - lane_util), rows};
}
}  // namespace Detail

// Supply compiled max-active-CTAs/SM for each concrete kernel; may be a cached
// table or an occupancy query at stamp time. A nonresident candidate is skipped.
// The two progression safeguards are the accepted conservative experimental policy:
// (1) wave correction cannot newly promote a higher-O one-row staging variant;
// (2) idle-lane correction cannot newly promote O8 and applies the same row guard.
template <typename ResidentCTAsPerSM>
std::optional<Variant> select(uint64_t outer, uint64_t r, uint64_t k,
                              uint32_t sm_count, ResidentCTAsPerSM&& residency) {
    if (outer == 0 || r == 0 || k == 0 || !compiledK(k) || sm_count == 0) return std::nullopt;
    std::array<Detail::Score, candidates.size()> options{};
    size_t count = 0;
    for (const Variant v : candidates) {
        if (!legal(v,r,k)) continue;
        const uint32_t resident = residency(v);
        if (resident == 0) continue;
        options[count++] = Detail::describe(v, outer, r, k, sm_count, resident);
    }
    if (count == 0) return std::nullopt;

    const auto best = [&](auto&& cost) -> const Detail::Score& {
        size_t index = 0;
        for (size_t i = 1; i < count; ++i) {
            if (Detail::scoreLess(cost(options[i]), cost(options[index]),
                                  options[i].variant, options[index].variant)) index = i;
        }
        return options[index];
    };
    const Detail::Score& base = best([](const auto& x) {return x.frozen;});
    const Detail::Score& wave = best([](const auto& x) {return x.frozen + .20 * x.waves;});
    const Detail::Score& current =
        wave.variant.outputs_per_warp > base.variant.outputs_per_warp && wave.rows_per_group <= 1
            ? base : wave;
    const Detail::Score& trial = best([](const auto& x) {return x.trial;});
    if (trial.variant.outputs_per_warp > current.variant.outputs_per_warp && trial.rows_per_group <= 1)
        return current.variant;
    if (trial.variant.outputs_per_warp > 4 && !(trial.variant == current.variant))
        return current.variant;
    return trial.variant;
}

} // namespace ThorImplementation::CubReductionBenchmarking::Fp8OneWindowPolicy
