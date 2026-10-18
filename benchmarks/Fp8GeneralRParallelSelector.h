#pragma once

// Generalized FP8 RParallel selector, initial integration stage:
// select() retains the accepted OneWindow-only default. Completed mode-local
// policies are explicitly addressable through selectForMode(); there is NO
// cross-mode performance choice or production dispatcher integration.

#include "benchmarks/Fp8OneWindowSelector.h"
#include "benchmarks/Fp8DeepRSmallKSelector.h"
#include "benchmarks/Fp8AlignedKSelector.h"
#include "benchmarks/Fp8RuntimeKSelector.h"
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

namespace ThorImplementation::CubReductionBenchmarking::Fp8GeneralRParallelPolicy {

enum class Mode : uint8_t { OneWindow, DeepRSmallK, AlignedK, RuntimeK };
struct OneWindowConfiguration {
    Fp8OneWindowPolicy::Variant variant;
    constexpr bool operator==(const OneWindowConfiguration&) const = default;
};
struct DeepRSmallKConfiguration {
    Fp8DeepRSmallKPolicy::Variant variant; // P16 is implicit in this mode.
    constexpr bool operator==(const DeepRSmallKConfiguration&) const = default;
};
struct AlignedKConfiguration {
    Fp8AlignedKPolicy::Variant variant;
    constexpr bool operator==(const AlignedKConfiguration&) const = default;
};
struct RuntimeKConfiguration {
    Fp8RuntimeKPolicy::Variant variant;
    constexpr bool operator==(const RuntimeKConfiguration&) const = default;
};
using Configuration = std::variant<OneWindowConfiguration, DeepRSmallKConfiguration,
                                   AlignedKConfiguration, RuntimeKConfiguration>;
struct Selection {
    Mode mode;
    Configuration configuration;
};

template <typename ResidentCTAsPerSM>
std::optional<Selection> select(uint64_t outer, uint64_t r, uint64_t k,
                                uint32_t sm_count, ResidentCTAsPerSM&& residency) {
    const auto v = Fp8OneWindowPolicy::select(outer, r, k, sm_count, residency);
    if (!v) return std::nullopt; // Not a general fallback; other modes are not yet enabled.
    return Selection{Mode::OneWindow, OneWindowConfiguration{*v}};
}

// Explicit typed mode-local dispatch. Residency providers must query the actual
// compiled kernels of their respective modes. Unfinished modes fail closed,
// never silently fall back to OneWindow or another mode.
template <typename OneWindowResidency, typename DeepRSmallKResidency,
          typename AlignedKResidency>
std::optional<Selection> selectForMode(Mode mode, uint64_t outer, uint64_t r, uint64_t k,
                                      uint32_t sm_count,
                                      OneWindowResidency&& one_window_residency,
                                      DeepRSmallKResidency&& deep_r_small_k_residency,
                                      AlignedKResidency&& aligned_k_residency) {
    switch (mode) {
        case Mode::OneWindow: {
            const auto v = Fp8OneWindowPolicy::select(
                outer, r, k, sm_count, one_window_residency);
            if (!v) return std::nullopt;
            return Selection{Mode::OneWindow, OneWindowConfiguration{*v}};
        }
        case Mode::DeepRSmallK: {
            const auto v = Fp8DeepRSmallKPolicy::select(
                outer, r, k, sm_count, deep_r_small_k_residency);
            if (!v) return std::nullopt;
            return Selection{Mode::DeepRSmallK, DeepRSmallKConfiguration{*v}};
        }
        case Mode::AlignedK: {
            const auto v = Fp8AlignedKPolicy::select(
                outer, r, k, sm_count, aligned_k_residency);
            if (!v) return std::nullopt;
            return Selection{Mode::AlignedK, AlignedKConfiguration{*v}};
        }
        case Mode::RuntimeK:
            return std::nullopt; // Local policy still under development.
    }
    return std::nullopt;
}

// RuntimeK is opt-in and needs actual compiled-kernel occupancy, supplied with
// complete physical plans. The default general select() does not change.
template <typename OneWindowResidency, typename DeepRSmallKResidency,
          typename AlignedKResidency>
std::optional<Selection> selectForMode(
    Mode mode, uint64_t outer, uint64_t r, uint64_t k, uint32_t sm_count,
    OneWindowResidency&& one_window_residency,
    DeepRSmallKResidency&& deep_r_small_k_residency,
    AlignedKResidency&& aligned_k_residency,
    std::span<const Fp8RuntimeKPolicy::Candidate> runtime_candidates) {
    if (mode == Mode::RuntimeK) {
        const auto v = Fp8RuntimeKPolicy::select(outer, r, k, sm_count, runtime_candidates);
        if (!v) return std::nullopt;
        return Selection{Mode::RuntimeK, RuntimeKConfiguration{*v}};
    }
    return selectForMode(mode, outer, r, k, sm_count,
                         std::forward<OneWindowResidency>(one_window_residency),
                         std::forward<DeepRSmallKResidency>(deep_r_small_k_residency),
                         std::forward<AlignedKResidency>(aligned_k_residency));
}

inline std::optional<Selection> selectRuntimeK(
    uint64_t outer, uint64_t r, uint64_t k, uint32_t sm_count,
    std::span<const Fp8RuntimeKPolicy::Candidate> physical_candidates) {
    const auto v = Fp8RuntimeKPolicy::select(outer, r, k, sm_count, physical_candidates);
    if (!v) return std::nullopt;
    return Selection{Mode::RuntimeK, RuntimeKConfiguration{*v}};
}

// Retain the two-residency interface for existing callers. AlignedK needs an
// explicitly supplied occupancy provider; it remains unavailable here rather
// than assuming that unqueried specializations can run.
template <typename OneWindowResidency, typename DeepRSmallKResidency>
std::optional<Selection> selectForMode(Mode mode, uint64_t outer, uint64_t r, uint64_t k,
                                      uint32_t sm_count,
                                      OneWindowResidency&& one_window_residency,
                                      DeepRSmallKResidency&& deep_r_small_k_residency) {
    return selectForMode(mode, outer, r, k, sm_count,
                         std::forward<OneWindowResidency>(one_window_residency),
                         std::forward<DeepRSmallKResidency>(deep_r_small_k_residency),
                         [](Fp8AlignedKPolicy::Variant) -> uint32_t { return 0; });
}

// Explicit local helper for an AlignedK residency query over actual compiled
// packet/CTA specializations. This chooses only within the AlignedK inventory.
template <typename ResidentCTAsPerSM>
std::optional<Selection> selectAlignedK(uint64_t outer, uint64_t r, uint64_t k,
                                        uint32_t sm_count,
                                        ResidentCTAsPerSM&& residency) {
    return selectForMode(Mode::AlignedK, outer, r, k, sm_count,
                         [](Fp8OneWindowPolicy::Variant) -> uint32_t { return 0; },
                         [](Fp8DeepRSmallKPolicy::Variant) -> uint32_t { return 0; },
                         std::forward<ResidentCTAsPerSM>(residency));
}

// Backwards-compatible explicit local helper for prior benchmark callers.
template <typename ResidentCTAsPerSM>
std::optional<Selection> selectDeepRSmallK(uint64_t outer, uint64_t r, uint64_t k,
                                         uint32_t sm_count, ResidentCTAsPerSM&& residency) {
    return selectForMode(Mode::DeepRSmallK, outer, r, k, sm_count,
                         [](Fp8OneWindowPolicy::Variant) -> uint32_t { return 0; },
                         residency);
}

} // namespace ThorImplementation::CubReductionBenchmarking::Fp8GeneralRParallelPolicy
