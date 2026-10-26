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

inline constexpr size_t MeasurementCount = 5;
using Measurements = std::array<double, MeasurementCount>;

struct CandidateTiming {
    Measurements elapsed_us{};                 // submission order: one sample per round
    double trimmed_mean_us = 0;
    double trimmed_mean_tb_per_s = 0;          // mean of the middle 3 per-run throughputs
};

struct TuningResult {
    size_t winner_index = 0;                    // index in the caller's candidate list
    std::vector<CandidateTiming> candidates;   // same ordering as caller's candidates
};

// First round is caller order. Every subsequent round is freshly shuffled;
// avoid identical adjacent permutations even with 2 candidates.
inline std::vector<std::vector<size_t>> makeRoundOrders(size_t count, uint64_t seed) {
    if (count == 0) throw std::invalid_argument("Reduction autotuning needs at least one candidate");
    std::vector<std::vector<size_t>> rounds;
    std::vector<size_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    rounds.push_back(order);
    std::mt19937_64 rng(seed);
    for (size_t round = 1; round < MeasurementCount; ++round) {
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
inline CandidateTiming score(const Measurements& samples, uint64_t logical_input_bytes) {
    if (logical_input_bytes == 0) throw std::invalid_argument("Reduction autotuning needs nonzero logical input bytes");
    CandidateTiming result;
    result.elapsed_us = samples;
    Measurements sorted = samples;
    for (double t : sorted) {
        if (!std::isfinite(t) || t <= 0.0) throw std::invalid_argument("Reduction autotuning sample must be finite and >0");
    }
    std::sort(sorted.begin(), sorted.end());
    // Drop the minimum and maximum TIME of this candidate independently.
    // A time's reciprocal is throughput, so this also trims highest/lowest throughput.
    for (size_t i = 1; i + 1 < MeasurementCount; ++i) {
        result.trimmed_mean_us += sorted[i] / 3.0;
        result.trimmed_mean_tb_per_s += (static_cast<double>(logical_input_bytes) / sorted[i] * 1e-6) / 3.0;
    }
    return result;
}

inline TuningResult chooseWinner(const std::vector<Measurements>& observations,
                                uint64_t logical_input_bytes) {
    if (observations.empty()) throw std::invalid_argument("Reduction autotuning needs at least one candidate");
    TuningResult result;
    result.candidates.reserve(observations.size());
    for (size_t i = 0; i < observations.size(); ++i) {
        result.candidates.push_back(score(observations[i], logical_input_bytes));
        // Stable tie break: earlier candidate wins.
        if (result.candidates[i].trimmed_mean_tb_per_s >
            result.candidates[result.winner_index].trimmed_mean_tb_per_s) {
            result.winner_index = i;
        }
    }
    return result;
}

// Each view starts on a 32-byte boundary and has at least 128 bytes of
// unused space before the next view. Merely rounding the tensor's size up to
// an alignment would NOT guarantee this padding (e.g. a 4096-byte view).
// Reuse distance is at least slot_count * input_stride; the CUDA runner
// chooses slot_count based on L2 capacity.
inline constexpr uint64_t TensorViewAlignmentBytes = 32;
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

inline size_t coldSlotCount(uint64_t input_stride, uint64_t reuse_distance_bytes) {
    if (input_stride == 0 || reuse_distance_bytes == 0) {
        throw std::invalid_argument("Reduction autotuner cold reuse inputs must be nonzero");
    }
    const uint64_t slots = reuse_distance_bytes / input_stride + 2;
    if (slots > std::numeric_limits<size_t>::max()) throw std::overflow_error("Reduction autotuner slot overflow");
    return static_cast<size_t>(slots);
}

} // namespace ThorImplementation::ReductionAutotuning
