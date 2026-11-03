#pragma once

// Host-only scheduling and scoring for the CUDA reduction autotuner.
// Keeping this free of CUDA types makes the timing policy independently testable.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ThorImplementation::ReductionAutotuning {

// 1141: the number of timed rounds per candidate is an option (FP8 RK: 5 for Quick, 7 for Regular, 11 for Ultra). Each
// candidate's `trim` fastest and `trim` slowest rounds are dropped (1143: 1, or 2 for Ultra) and the rest averaged; the
// smallest average time wins.
inline constexpr size_t DefaultMeasurementCount = 5;
inline constexpr size_t MinMeasurementCount = 3;
using Measurements = std::vector<double>;

struct CandidateTiming {
    Measurements elapsed_us;                   // submission order: one sample per round (empty if abandoned)
    double trimmed_mean_us = 0;                // mean time without the fastest and slowest round
    double trimmed_mean_tb_per_s = 0;          // logical input bytes / trimmed_mean_us
    // 1161: the candidate's one timed probe launch, its launches per measurement, and whether
    // the probe put it so far off the best that it was not measured (it then never wins).
    double probe_us = 0;
    size_t launches_per_measurement = 0;
    bool abandoned = false;
};

struct TuningResult {
    size_t winner_index = 0;                    // index in the caller's candidate list
    std::vector<CandidateTiming> candidates;   // same ordering as caller's candidates
};

// First round is caller order. Every subsequent round is freshly shuffled;
// avoid identical adjacent permutations even with 2 candidates.
inline std::vector<std::vector<size_t>> makeRoundOrders(size_t count, uint64_t seed,
                                                        size_t measurement_count = DefaultMeasurementCount) {
    if (count == 0) throw std::invalid_argument("Reduction autotuning needs at least one candidate");
    if (measurement_count < MinMeasurementCount)
        throw std::invalid_argument("Reduction autotuning needs at least three rounds");
    std::vector<std::vector<size_t>> rounds;
    std::vector<size_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    rounds.push_back(order);
    std::mt19937_64 rng(seed);
    for (size_t round = 1; round < measurement_count; ++round) {
        const auto previous = order;
        for (unsigned try_count = 0; try_count < 8; ++try_count) {
            std::shuffle(order.begin(), order.end(), rng);
            if (order != previous) break;
        }
        if (count > 1 && order == previous) std::rotate(order.begin(), order.begin() + 1, order.end());
        rounds.push_back(order);
    }
    return rounds;
}

// Each candidate performs the same logical reduction and therefore receives
// the same logical traffic denominator; scratch/reduction implementation traffic
// is deliberately NOT used as an efficiency bonus.
inline CandidateTiming score(const Measurements& samples, uint64_t logical_input_bytes, size_t trim = 1) {
    if (logical_input_bytes == 0) throw std::invalid_argument("Reduction autotuning needs nonzero logical input bytes");
    if (samples.size() < MinMeasurementCount)
        throw std::invalid_argument("Reduction autotuning needs at least three samples per candidate");
    if (samples.size() <= 2 * trim)
        throw std::invalid_argument("Reduction autotuning must keep at least one sample after trimming");
    CandidateTiming result;
    result.elapsed_us = samples;
    Measurements sorted = samples;
    for (double t : sorted) {
        if (!std::isfinite(t) || t <= 0.0) throw std::invalid_argument("Reduction autotuning sample must be finite and >0");
    }
    std::sort(sorted.begin(), sorted.end());
    // Drop the `trim` fastest and `trim` slowest runs of this candidate; average the survivors.
    const double survivors = static_cast<double>(sorted.size() - 2 * trim);
    for (size_t i = trim; i + trim < sorted.size(); ++i) result.trimmed_mean_us += sorted[i] / survivors;
    result.trimmed_mean_tb_per_s = static_cast<double>(logical_input_bytes) / result.trimmed_mean_us * 1e-6;
    return result;
}

inline TuningResult chooseWinner(const std::vector<Measurements>& observations,
                                uint64_t logical_input_bytes, size_t trim = 1) {
    if (observations.empty()) throw std::invalid_argument("Reduction autotuning needs at least one candidate");
    TuningResult result;
    result.candidates.reserve(observations.size());
    for (size_t i = 0; i < observations.size(); ++i) {
        result.candidates.push_back(score(observations[i], logical_input_bytes, trim));
        // Smallest trimmed mean time wins; stable tie break: earlier (higher ranked) candidate wins.
        if (result.candidates[i].trimmed_mean_us <
            result.candidates[result.winner_index].trimmed_mean_us) {
            result.winner_index = i;
        }
    }
    return result;
}

// 1161 (1162: every reduction contest): the contest times one cold launch of every candidate first, then gives each
// candidate its own launches per measurement (about target_us of GPU time per measurement, so a fast kernel batches
// enough launches to beat the ~2 us CUDA event resolution and a slow one is not repeated 32 times) and skips
// candidates whose probe is over
// abandon_ratio x max(best probe, abandon_floor_us): they cannot win, and measuring them was most of a contest's time
// (the 1160 KR confirmation sweep: 92-98% of a stamp at 64 KiB-8 MiB went to shapes 10-100x slower than the best).
// The floor keeps near-ties of kernels shorter than the event resolution from being dropped on a quantized probe.
struct ProbePlan {
    std::vector<size_t> launches;  // per candidate, 0 when abandoned
    std::vector<bool> measured;
};

inline ProbePlan planFromProbes(const std::vector<double>& probe_us, double target_us, size_t max_launches,
                                double abandon_ratio, double abandon_floor_us) {
    if (probe_us.empty()) throw std::invalid_argument("Reduction autotuning probe needs at least one candidate");
    if (!(target_us > 0.0) || max_launches == 0)
        throw std::invalid_argument("Reduction autotuning probe needs a positive target and launch cap");
    // Event time can read 0 for a launch below the resolution; treat it as one resolution step.
    constexpr double ResolutionUs = 2.048;
    double best = std::numeric_limits<double>::infinity();
    for (double t : probe_us) {
        if (!std::isfinite(t) || t < 0.0) throw std::invalid_argument("Reduction autotuning probe must be finite");
        best = std::min(best, std::max(t, ResolutionUs));
    }
    const double limit = abandon_ratio > 0.0 ? abandon_ratio * std::max(best, abandon_floor_us)
                                             : std::numeric_limits<double>::infinity();
    ProbePlan plan;
    plan.launches.resize(probe_us.size(), 0);
    plan.measured.resize(probe_us.size(), false);
    for (size_t i = 0; i < probe_us.size(); ++i) {
        const double t = std::max(probe_us[i], ResolutionUs);
        if (t > limit) continue;
        plan.measured[i] = true;
        plan.launches[i] = static_cast<size_t>(
            std::clamp(std::ceil(target_us / t), 1.0, static_cast<double>(max_launches)));
    }
    return plan;
}

// chooseWinner over the measured candidates only; abandoned ones keep their slot (abandoned = true) and never win.
inline TuningResult chooseWinner(const std::vector<Measurements>& observations, const std::vector<bool>& measured,
                                 uint64_t logical_input_bytes, size_t trim = 1) {
    if (observations.empty() || observations.size() != measured.size())
        throw std::invalid_argument("Reduction autotuning measured mask must match the candidates");
    TuningResult result;
    result.candidates.resize(observations.size());
    bool any = false;
    for (size_t i = 0; i < observations.size(); ++i) {
        if (!measured[i]) {
            result.candidates[i].abandoned = true;
            continue;
        }
        result.candidates[i] = score(observations[i], logical_input_bytes, trim);
        if (!any || result.candidates[i].trimmed_mean_us < result.candidates[result.winner_index].trimmed_mean_us)
            result.winner_index = i;
        any = true;
    }
    if (!any) throw std::invalid_argument("Reduction autotuning measured no candidate");
    return result;
}

// Each view starts on a 256-byte boundary and has at least 128 bytes of
// unused space before the next view. Merely rounding the tensor's size up to
// an alignment would NOT guarantee this padding (e.g. a 4096-byte view).
// Reuse distance is at least slot_count * slot bytes; the CUDA runner
// chooses slot_count based on L2 capacity.
// 1163: 256 bytes, cudaMalloc's alignment, so every timed launch reads an input aligned as a real tensor's is. With
// 32-byte starts a ring slot sat at a different offset mod 256 from the next (a 256 MiB input alternated 0 and 128), and
// the FP8 RK contest and sweep disagreed by up to 14% at 64-256 MiB where the losing shapes ran ~10% faster in one
// sample of five: the cue of an alignment-sensitive kernel timed at mixed alignments.
inline constexpr uint64_t TensorViewAlignmentBytes = 256;
inline constexpr uint64_t TensorViewPaddingBytes = 128;
inline uint64_t alignedStride(uint64_t bytes) {
    constexpr uint64_t alignment = TensorViewAlignmentBytes;
    constexpr uint64_t padding = TensorViewPaddingBytes;
    if (bytes == 0 ||
        bytes > std::numeric_limits<uint64_t>::max() - padding - (alignment - 1)) {
        throw std::invalid_argument("Invalid reduction autotuner tensor stride");
    }
    const uint64_t padded_bytes = bytes + padding;
    return (padded_bytes + alignment - 1) & ~(alignment - 1);
}

// 1161: slot_bytes is everything one launch touches in its slot (input + output + scratch strides), so the reuse
// distance counts all the traffic that evicts a slot's input from L2, and the rings together stay near
// reuse_distance_bytes. (Before 1161 it was the input stride alone and the output and scratch rings took the same
// slot count: a 64 KiB input with 4x as many output bytes got 12,288 slots and a 3 GiB output ring.)
inline size_t coldSlotCount(uint64_t slot_bytes, uint64_t reuse_distance_bytes) {
    if (slot_bytes == 0 || reuse_distance_bytes == 0) {
        throw std::invalid_argument("Reduction autotuner cold reuse inputs must be nonzero");
    }
    const uint64_t slots = reuse_distance_bytes / slot_bytes + 2;
    if (slots > std::numeric_limits<size_t>::max()) throw std::overflow_error("Reduction autotuner slot overflow");
    return static_cast<size_t>(slots);
}

} // namespace ThorImplementation::ReductionAutotuning
