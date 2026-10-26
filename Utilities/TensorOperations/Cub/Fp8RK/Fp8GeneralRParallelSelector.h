#pragma once

// Generalized FP8 RParallel candidate assembly for stamp-time autotuning.
// The typed cross-family shortlist below combines the four accepted local
// shortlists, with physical legality, occupancy, and explicit scratch gating.
// With RankerSemantics supplied, a frozen geometry-aware model ranks the
// executable plans (four family leaders first, score-only thereafter).
// The untyped legacy entry point retains the deterministic round-robin order.
// This is production-owned host planning policy, not CUDA dispatch.
// Legacy one-residency select()/shortlist() APIs remain OneWindow-only.

#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8OneWindowSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8DeepRSmallKSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8AlignedKSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RuntimeKSelector.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8GeometryRankerModel.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>
#include <utility>
#include <variant>

namespace ThorImplementation::CubReductionInternal::Fp8RK::Fp8GeneralRParallelPolicy {

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
    bool operator==(const Selection&) const = default;
};

// This is a physical *first-stage* candidate for constructing a stamp-time
// reduction plan. The production planner must independently plan the FP32
// continuation whenever scratch_bytes != 0 (and then final output conversion).
// The candidate does not own CUDA resources or invoke any GPU operations.
struct CandidatePlan {
    Selection selection;
    uint32_t resident_ctas_per_sm;
    uint64_t scratch_bytes;           // FP32 partials, zero for Complete
    uint64_t first_stage_work_groups; // logical CTA work, not a launch grid.x limit
    bool staged() const { return scratch_bytes != 0; }
};

// These semantics must be taken from the same concrete FP8 specialization
// for which the occupancy callbacks and RuntimeK records were built.
// They cannot be inferred from (outer, R, K) alone.
enum class Fp8Format : uint8_t { E4M3, E5M2 };
enum class Combine : uint8_t { Sum, Product, Min, Max };
enum class InputTransform : uint8_t { Identity, Abs, Square };
struct RankerSemantics {
    Fp8Format format;
    Combine combine;
    InputTransform transform;
};

struct GeneralShortlistOptions {
    // Score-only Thor1012 refit at budget 28 (full-inventory replay exact;
    // geometry-separated CV has one miss). Older GPU sweeps predate this fit;
    // independent GPU validation of the refit is still required.
    size_t max_candidates = 28;
    size_t per_family_limit = 32;
    // A real stamp-time workspace budget can rule out impractical staged plans.
    // Unlimited by default: the planner must supply its actual resource limit.
    uint64_t max_scratch_bytes = std::numeric_limits<uint64_t>::max();
    // Explicit opt-in because the old untyped API has no format/operation data.
    // With this provided, rank the complete eligible inventory using the
    // refitted embedded model, then truncate to max_candidates. No forced probes.
    std::optional<RankerSemantics> ranker_semantics = std::nullopt;
};

namespace Detail {
inline bool checkedMul(uint64_t a, uint64_t b, uint64_t& out) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b) return false;
    out = a * b;
    return true;
}
inline bool stagedBytes(uint64_t outer, uint64_t shards, uint64_t k,
                        uint64_t& out) {
    uint64_t partials;
    return checkedMul(outer, shards, partials) &&
           checkedMul(partials, k, partials) &&
           checkedMul(partials, sizeof(float), out);
}

// Matches benchmarks/experiment_fp8_geometry_ranker.py::candidate_features
// exactly (including float32 feature quantization). Only geometry/physical
// metadata, dtype, combine, transform and within-family rank are inputs; no
// measured timing or learned per-workload feedback reaches the selector.
inline std::array<float, 25> rankerFeatures(
    const CandidatePlan& plan, uint64_t outer, uint64_t r, uint64_t k,
    RankerSemantics semantics, uint32_t local_rank) {
    double packet = 0, threads = 0, shards = 0, warps = 0, tile = 0;
    double out = 0, rows = 0, mode = 0;
    switch (plan.selection.mode) {
        case Mode::OneWindow: {
            const auto v = std::get<OneWindowConfiguration>(plan.selection.configuration).variant;
            out = v.outputs_per_warp;
            warps = v.warps_per_output;
            packet = v.packet_bytes;
            threads = warps * 32;
            mode = 1;
            break;
        }
        case Mode::DeepRSmallK: {
            const auto v = std::get<DeepRSmallKConfiguration>(plan.selection.configuration).variant;
            packet = 16;
            rows = v.rows_per_shard;
            shards = Fp8DeepRSmallKPolicy::ceilDiv(r, v.rows_per_shard);
            mode = 2;
            break;
        }
        case Mode::AlignedK: {
            const auto v = std::get<AlignedKConfiguration>(plan.selection.configuration).variant;
            packet = v.packet;
            threads = v.threads;
            shards = v.shards_per_output;
            rows = Fp8AlignedKPolicy::ceilDiv(r, v.shards_per_output);
            tile = Fp8AlignedKPolicy::ceilDiv(k, uint64_t(v.packet) * v.threads);
            mode = 3;
            break;
        }
        case Mode::RuntimeK: {
            const auto v = std::get<RuntimeKConfiguration>(plan.selection.configuration).variant;
            warps = v.warps_per_output;
            threads = warps * 32;
            tile = v.tile_count;
            rows = v.rows_per_shard;
            shards = Fp8RuntimeKPolicy::ceilDiv(r, v.rows_per_shard);
            switch (v.layout) {
                case Fp8RuntimeKPolicy::Layout::PacketComplete: mode = 4; break;
                case Fp8RuntimeKPolicy::Layout::PacketStaged: mode = 5; break;
                case Fp8RuntimeKPolicy::Layout::RotatedExact: mode = 6; break;
                case Fp8RuntimeKPolicy::Layout::RotatedPrefix: mode = 7; break;
            }
            break;
        }
    }
    const double o = static_cast<double>(outer), rr = static_cast<double>(r);
    const double kk = static_cast<double>(k);
    const auto lg = [](double v) -> float { return static_cast<float>(std::log2(v)); };
    return {lg(o), lg(rr), lg(kk), lg(o*rr*kk), lg(o*kk), lg(rr/kk),
            float(semantics.format == Fp8Format::E4M3),
            float(static_cast<unsigned>(semantics.combine)),
            float(static_cast<unsigned>(semantics.transform)),
            float(static_cast<unsigned>(plan.selection.mode)), float(local_rank),
            lg(packet+1), lg(threads+1), lg(shards+1), lg(warps+1),
            lg(tile+1), lg(out+1), lg(rows+1), static_cast<float>(mode),
            lg(o*shards+1), lg(o*shards*std::max(tile,1.0)+1),
            lg(o*shards*kk*4+1), lg(o*rr*kk/1048576.0+1),
            lg(rr/std::max(rows,1.0)+1),
            lg(kk/std::max(packet*std::max(threads,1.0),1.0)+1)};
}

// Retain exactly one primary from each available physical family, preserving
// their original local-leader order. Sort every other plan by the embedded
// boosted-tree margin and break ties by pre-existing round-robin position.
inline void rankByFrozenGeometryModel(std::vector<CandidatePlan>& plans,
                                      uint64_t outer, uint64_t r, uint64_t k,
                                      RankerSemantics semantics) {
    std::array<uint32_t, 4> rank_by_family{};
    std::vector<std::pair<CandidatePlan, float>> scored;
    scored.reserve(plans.size());
    std::vector<CandidatePlan> leaders;
    leaders.reserve(4);
    for (const auto& plan : plans) {
        const auto family = static_cast<size_t>(plan.selection.mode);
        const uint32_t local_rank = ++rank_by_family[family];
        if (local_rank == 1) {
            leaders.push_back(plan);
        } else {
            const auto features = rankerFeatures(plan,outer,r,k,semantics,local_rank);
            scored.emplace_back(plan, Fp8GeometryRankerModel::margin(features));
        }
    }
    std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    plans.clear();
    plans.insert(plans.end(), leaders.begin(), leaders.end());
    for (auto& entry : scored) plans.push_back(std::move(entry.first));
}

// CUDA occupancy may require an expensive exact-specialization query; cache
// by variant so each mode-local shortlist and its plan materialization agree.
template <typename Variant, typename Callback>
class CachedResidency {
public:
    explicit CachedResidency(Callback& query) : query_(query) {}
    uint32_t operator()(Variant v) {
        for (const auto& entry : cache_)
            if (entry.first == v) return entry.second;
        const uint32_t value = query_(v);
        cache_.push_back({v, value});
        return value;
    }
private:
    Callback& query_;
    std::vector<std::pair<Variant, uint32_t>> cache_;
};
} // namespace Detail

// Cross-family *candidate generation*, not an uncalibrated scalar selector.
// Residency functions must refer to the actual retained kernel specialization
// for the requested FP8 format, input transform, combine operation and output
// path. RuntimeK requires full executable Candidate records, never guessed
// layouts or guessed occupancy. Empty/malformed inputs fail closed.
//
// First construct the COMPLETE eligible local inventory. When supplied with
// RankerSemantics, use the frozen cross-family score-only model; otherwise
// retain the legacy untyped round-robin order. Both paths preserve the family
// leaders and apply the same physical, occupancy, and scratch gating. Never
// compare incomparable local scores or treat the model margin as microseconds.
template <typename OneWindowResidency, typename DeepRSmallKResidency,
          typename AlignedKResidency>
std::vector<CandidatePlan> generalShortlist(
    uint64_t outer, uint64_t r, uint64_t k, uint32_t sm_count,
    OneWindowResidency&& one_window_residency,
    DeepRSmallKResidency&& deep_r_small_k_residency,
    AlignedKResidency&& aligned_k_residency,
    std::span<const Fp8RuntimeKPolicy::Candidate> runtime_candidates,
    GeneralShortlistOptions options = {}) {
    std::vector<CandidatePlan> result;
    if (!outer || !r || !k || !sm_count || !options.max_candidates ||
        !options.per_family_limit) return result;
    // Prevent arithmetic overflow in physical geometry and FP32 scratch.
    uint64_t elements;
    if (!Detail::checkedMul(outer, r, elements) ||
        !Detail::checkedMul(elements, k, elements)) return result;
    // The local APIs retain five-way defaults. Cross-family autotuning may
    // explicitly enumerate beyond five to avoid excluding a real family oracle.
    // Bound even pathological option values; each local inventory is smaller.
    const size_t local_limit = std::min<size_t>(32, options.per_family_limit);
    Detail::CachedResidency<Fp8OneWindowPolicy::Variant, OneWindowResidency> ow_cache(
        one_window_residency);
    Detail::CachedResidency<Fp8DeepRSmallKPolicy::Variant, DeepRSmallKResidency> dr_cache(
        deep_r_small_k_residency);
    Detail::CachedResidency<Fp8AlignedKPolicy::Variant, AlignedKResidency> ak_cache(
        aligned_k_residency);

    const auto ow = Fp8OneWindowPolicy::shortlist(outer, r, k, sm_count,
        [&](auto v) { return ow_cache(v); }, local_limit);
    // Exclude staged choices that do not fit the planner's scratch budget
    // before local ranking. Otherwise a short list could contain only
    // oversize plans while a deeper legal depth could fit.
    const auto dr = Fp8DeepRSmallKPolicy::shortlist(outer, r, k, sm_count,
        [&](auto v) {
            uint64_t bytes;
            if (!Detail::stagedBytes(outer, Fp8DeepRSmallKPolicy::ceilDiv(r, v.rows_per_shard),
                                     k, bytes) || bytes > options.max_scratch_bytes) return 0u;
            return dr_cache(v);
        }, local_limit);
    const auto ak = Fp8AlignedKPolicy::shortlist(outer, r, k, sm_count,
        [&](auto v) {
            uint64_t bytes = 0;
            if (v.shards_per_output > 1 &&
                (!Detail::stagedBytes(outer, v.shards_per_output, k, bytes) ||
                 bytes > options.max_scratch_bytes)) return 0u;
            return ak_cache(v);
        }, local_limit);
    std::vector<Fp8RuntimeKPolicy::Candidate> eligible_runtime;
    eligible_runtime.reserve(runtime_candidates.size());
    for (const auto& candidate : runtime_candidates)
        if (candidate.scratch_bytes <= options.max_scratch_bytes &&
            Fp8RuntimeKPolicy::legal(candidate, outer, r, k, sm_count))
            eligible_runtime.push_back(candidate);
    const auto rk = Fp8RuntimeKPolicy::shortlist(outer, r, k, sm_count,
        eligible_runtime, local_limit);

    // Resolve the complete RuntimeK metadata from the *same* validated input
    // record used by the mode-local selector. Repeated variants use the first
    // matching legal physical record, as in the local shortlist.
    const auto add = [&](Mode mode, Configuration cfg, uint32_t resident,
                         uint64_t scratch, uint64_t work_groups) {
        CandidatePlan plan{{mode, std::move(cfg)}, resident, scratch, work_groups};
        if (resident && scratch <= options.max_scratch_bytes &&
            std::none_of(result.begin(), result.end(), [&](const CandidatePlan& prev) {
                return prev.selection == plan.selection;
            })) result.push_back(std::move(plan));
    };
    result.reserve(ow.size() + dr.size() + ak.size() + rk.size());
    for (size_t rank = 0; rank < local_limit; ++rank) {
        if (rank < ow.size()) {
            const auto v = ow[rank];
            add(Mode::OneWindow, OneWindowConfiguration{v}, ow_cache(v), 0,
                Fp8OneWindowPolicy::ctaCount(outer, v));
        }
        if (rank < dr.size()) {
            const auto v = dr[rank];
            const uint64_t shards = Fp8DeepRSmallKPolicy::ceilDiv(r, v.rows_per_shard);
            uint64_t bytes, groups;
            if (Detail::stagedBytes(outer, shards, k, bytes) &&
                Detail::checkedMul(outer, shards, groups))
                add(Mode::DeepRSmallK, DeepRSmallKConfiguration{v}, dr_cache(v),
                    bytes, groups);
        }
        if (rank < ak.size()) {
            const auto v = ak[rank];
            uint64_t bytes = 0, groups, tiles = Fp8AlignedKPolicy::ceilDiv(
                k, uint64_t(v.packet) * v.threads);
            if ((v.shards_per_output == 1 ||
                 Detail::stagedBytes(outer, v.shards_per_output, k, bytes)) &&
                Detail::checkedMul(outer, v.shards_per_output, groups) &&
                Detail::checkedMul(groups, tiles, groups))
                add(Mode::AlignedK, AlignedKConfiguration{v}, ak_cache(v), bytes, groups);
        }
        if (rank < rk.size()) {
            const auto v = rk[rank];
            for (const auto& candidate : eligible_runtime) {
                if (candidate.variant == v) {
                    add(Mode::RuntimeK, RuntimeKConfiguration{v},
                        candidate.resident_ctas_per_sm, candidate.scratch_bytes,
                        candidate.first_stage_ctas);
                    break;
                }
            }
        }
    }
    if (options.ranker_semantics)
        Detail::rankByFrozenGeometryModel(result, outer, r, k, *options.ranker_semantics);
    if (result.size() > options.max_candidates) result.resize(options.max_candidates);
    return result;
}

// Preferred typed entry point: require the actual format, combine and input
// transform so that the same frozen model validated on GPU is always used.
// The optional semantics in GeneralShortlistOptions preserves compatibility
// with older aggregate-initialized untyped clients; production callers should
// prefer this overload instead of relying on an unset/default context.
template <typename OneWindowResidency, typename DeepRSmallKResidency,
          typename AlignedKResidency>
std::vector<CandidatePlan> generalShortlist(
    uint64_t outer, uint64_t r, uint64_t k, uint32_t sm_count,
    OneWindowResidency&& one_window_residency,
    DeepRSmallKResidency&& deep_r_small_k_residency,
    AlignedKResidency&& aligned_k_residency,
    std::span<const Fp8RuntimeKPolicy::Candidate> runtime_candidates,
    RankerSemantics semantics, GeneralShortlistOptions options = {}) {
    options.ranker_semantics = semantics;
    return generalShortlist(outer, r, k, sm_count,
        std::forward<OneWindowResidency>(one_window_residency),
        std::forward<DeepRSmallKResidency>(deep_r_small_k_residency),
        std::forward<AlignedKResidency>(aligned_k_residency),runtime_candidates,options);
}

// A typed overload of the familiar shortlist() name, available when all three
// occupancy providers and the RuntimeK physical registry are supplied.
template <typename OneWindowResidency, typename DeepRSmallKResidency,
          typename AlignedKResidency>
std::vector<Selection> shortlist(
    uint64_t outer, uint64_t r, uint64_t k, uint32_t sm_count,
    OneWindowResidency&& one_window_residency,
    DeepRSmallKResidency&& deep_r_small_k_residency,
    AlignedKResidency&& aligned_k_residency,
    std::span<const Fp8RuntimeKPolicy::Candidate> runtime_candidates,
    GeneralShortlistOptions options = {}) {
    std::vector<Selection> selections;
    for (auto& candidate : generalShortlist(outer, r, k, sm_count,
        std::forward<OneWindowResidency>(one_window_residency),
        std::forward<DeepRSmallKResidency>(deep_r_small_k_residency),
        std::forward<AlignedKResidency>(aligned_k_residency), runtime_candidates, options))
        selections.push_back(std::move(candidate.selection));
    return selections;
}

// The explicit-semantics Selection-only wrapper uses the same 28-candidate
// learned prefix, without reconstructing or independently reordering plans.
template <typename OneWindowResidency, typename DeepRSmallKResidency,
          typename AlignedKResidency>
std::vector<Selection> shortlist(
    uint64_t outer, uint64_t r, uint64_t k, uint32_t sm_count,
    OneWindowResidency&& one_window_residency,
    DeepRSmallKResidency&& deep_r_small_k_residency,
    AlignedKResidency&& aligned_k_residency,
    std::span<const Fp8RuntimeKPolicy::Candidate> runtime_candidates,
    RankerSemantics semantics, GeneralShortlistOptions options = {}) {
    options.ranker_semantics = semantics;
    return shortlist(outer,r,k,sm_count,
        std::forward<OneWindowResidency>(one_window_residency),
        std::forward<DeepRSmallKResidency>(deep_r_small_k_residency),
        std::forward<AlignedKResidency>(aligned_k_residency),runtime_candidates,options);
}

template <typename ResidentCTAsPerSM>
std::optional<Selection> select(uint64_t outer, uint64_t r, uint64_t k,
                                uint32_t sm_count, ResidentCTAsPerSM&& residency) {
    const auto v = Fp8OneWindowPolicy::select(outer, r, k, sm_count, residency);
    if (!v) return std::nullopt; // This compatibility entry point does not select other families.
    return Selection{Mode::OneWindow, OneWindowConfiguration{*v}};
}

// The shortlist API is the stamp-time autotuner's entry point. Until a real
// cross-family model exists, the default remains OneWindow-only (just like
// select()); mode-local callers use shortlistForMode() explicitly. The first
// entry is the established deterministic choice for that mode.
template <typename ResidentCTAsPerSM>
std::vector<Selection> shortlist(uint64_t outer, uint64_t r, uint64_t k,
                                 uint32_t sm_count, ResidentCTAsPerSM&& residency,
                                 size_t limit = 5) {
    std::vector<Selection> result;
    for (const auto& v : Fp8OneWindowPolicy::shortlist(
             outer, r, k, sm_count, residency, limit))
        result.push_back(Selection{Mode::OneWindow, OneWindowConfiguration{v}});
    return result;
}

template <typename OneWindowResidency, typename DeepRSmallKResidency,
          typename AlignedKResidency>
std::vector<Selection> shortlistForMode(
    Mode mode, uint64_t outer, uint64_t r, uint64_t k, uint32_t sm_count,
    OneWindowResidency&& one_window_residency,
    DeepRSmallKResidency&& deep_r_small_k_residency,
    AlignedKResidency&& aligned_k_residency,
    size_t limit = 5) {
    std::vector<Selection> result;
    switch (mode) {
        case Mode::OneWindow:
            for (auto v : Fp8OneWindowPolicy::shortlist(
                     outer, r, k, sm_count, one_window_residency, limit))
                result.push_back(Selection{mode, OneWindowConfiguration{v}});
            break;
        case Mode::DeepRSmallK:
            for (auto v : Fp8DeepRSmallKPolicy::shortlist(
                     outer, r, k, sm_count, deep_r_small_k_residency, limit))
                result.push_back(Selection{mode, DeepRSmallKConfiguration{v}});
            break;
        case Mode::AlignedK:
            for (auto v : Fp8AlignedKPolicy::shortlist(
                     outer, r, k, sm_count, aligned_k_residency, limit))
                result.push_back(Selection{mode, AlignedKConfiguration{v}});
            break;
        case Mode::RuntimeK:
            break; // Needs explicitly supplied physical plans, never invent them.
    }
    return result;
}

template <typename OneWindowResidency, typename DeepRSmallKResidency,
          typename AlignedKResidency>
std::vector<Selection> shortlistForMode(
    Mode mode, uint64_t outer, uint64_t r, uint64_t k, uint32_t sm_count,
    OneWindowResidency&& one_window_residency,
    DeepRSmallKResidency&& deep_r_small_k_residency,
    AlignedKResidency&& aligned_k_residency,
    std::span<const Fp8RuntimeKPolicy::Candidate> runtime_candidates,
    size_t limit = 5) {
    if (mode == Mode::RuntimeK) {
        std::vector<Selection> result;
        for (auto v : Fp8RuntimeKPolicy::shortlist(
                 outer, r, k, sm_count, runtime_candidates, limit))
            result.push_back(Selection{mode, RuntimeKConfiguration{v}});
        return result;
    }
    return shortlistForMode(mode, outer, r, k, sm_count,
                            std::forward<OneWindowResidency>(one_window_residency),
                            std::forward<DeepRSmallKResidency>(deep_r_small_k_residency),
                            std::forward<AlignedKResidency>(aligned_k_residency), limit);
}

template <typename OneWindowResidency, typename DeepRSmallKResidency>
std::vector<Selection> shortlistForMode(
    Mode mode, uint64_t outer, uint64_t r, uint64_t k, uint32_t sm_count,
    OneWindowResidency&& one_window_residency,
    DeepRSmallKResidency&& deep_r_small_k_residency,
    size_t limit = 5) {
    return shortlistForMode(mode, outer, r, k, sm_count,
                            std::forward<OneWindowResidency>(one_window_residency),
                            std::forward<DeepRSmallKResidency>(deep_r_small_k_residency),
                            [](Fp8AlignedKPolicy::Variant) -> uint32_t { return 0; }, limit);
}

inline std::vector<Selection> shortlistRuntimeK(
    uint64_t outer, uint64_t r, uint64_t k, uint32_t sm_count,
    std::span<const Fp8RuntimeKPolicy::Candidate> physical_candidates,
    size_t limit = 5) {
    std::vector<Selection> result;
    for (auto v : Fp8RuntimeKPolicy::shortlist(
             outer, r, k, sm_count, physical_candidates, limit))
        result.push_back(Selection{Mode::RuntimeK, RuntimeKConfiguration{v}});
    return result;
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
            return std::nullopt; // RuntimeK requires the explicit compiled-candidate overload.
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

} // namespace ThorImplementation::CubReductionInternal::Fp8RK::Fp8GeneralRParallelPolicy
