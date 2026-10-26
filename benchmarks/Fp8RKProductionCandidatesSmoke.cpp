// Host-only production/benchmark parity and legality tests:
// g++ -std=c++23 -O2 -Wall -Wextra -Werror -I. benchmarks/Fp8RKProductionCandidatesSmoke.cpp -o /tmp/fp8-rk-prod-smoke
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKProductionCandidates.h"
#include "benchmarks/Fp8GeneralRParallelSelector.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace P = ThorImplementation::CubReductionInternal::Fp8RK;
namespace B = ThorImplementation::CubReductionBenchmarking;
namespace G = P::Fp8GeneralRParallelPolicy;

static_assert(std::is_same_v<G::CandidatePlan, B::Fp8GeneralRParallelPolicy::CandidatePlan>);
static_assert(P::kFp8RKAutotuneCandidateBudget == 28);

int main() {
    // The exact same physical input will be used by both entry points; in
    // production these callbacks must instead query compiled CUDA occupancy.
    const auto ow = [](P::Fp8OneWindowPolicy::Variant v) -> uint32_t {
        return v.packet_bytes == 16 ? 6 : 0;
    };
    const auto dr = [](P::Fp8DeepRSmallKPolicy::Variant) -> uint32_t { return 4; };
    const auto ak = [](P::Fp8AlignedKPolicy::Variant v) -> uint32_t {
        return v.packet == 8 ? 3 : 0;
    };
    const P::Fp8RuntimeKPolicy::Candidate physical{
        {P::Fp8RuntimeKPolicy::Layout::RotatedPrefix, 0, 497, 1, 256, 8},
        1024, 128ULL * 8 * 256 * 4, 4};
    assert(P::Fp8RuntimeKPolicy::legal(physical, 128, 2048, 256, 170));
    const std::array runtime{physical, physical}; // Dedup identical physical plan.
    const std::array<P::Fp8RuntimeKPolicy::Candidate, 0> empty_runtime{};

    for (const auto semantics : {
             G::RankerSemantics{G::Fp8Format::E4M3, G::Combine::Sum, G::InputTransform::Identity},
             G::RankerSemantics{G::Fp8Format::E5M2, G::Combine::Max, G::InputTransform::Square},
             G::RankerSemantics{G::Fp8Format::E4M3, G::Combine::Product, G::InputTransform::Abs}}) {
        for (uint64_t budget : {uint64_t{0}, uint64_t{1 << 16},
                                std::numeric_limits<uint64_t>::max()}) {
            const P::ProductionSelectionContext ctx{128, 2048, 256, 170, semantics, budget};
            auto prod = P::selectProductionCandidates(ctx, ow, dr, ak, runtime);
            G::GeneralShortlistOptions options;
            options.max_candidates = 28;
            options.per_family_limit = 32;
            options.max_scratch_bytes = budget;
            auto bench = B::Fp8GeneralRParallelPolicy::generalShortlist(
                ctx.outer, ctx.r, ctx.k, ctx.sm_count, ow, dr, ak,
                std::span<const P::Fp8RuntimeKPolicy::Candidate>(runtime), semantics, options);
            assert(prod.size() == bench.size());
            for (size_t i = 0; i < prod.size(); ++i) {
                assert(prod[i].selection == bench[i].selection);
                assert(prod[i].resident_ctas_per_sm == bench[i].resident_ctas_per_sm);
                assert(prod[i].scratch_bytes == bench[i].scratch_bytes);
                assert(prod[i].first_stage_work_groups == bench[i].first_stage_work_groups);
            }
            assert(prod.size() <= P::kFp8RKAutotuneCandidateBudget);
            for (const auto& plan : prod) {
                assert(plan.resident_ctas_per_sm > 0);
                assert(plan.scratch_bytes <= budget);
                assert(plan.first_stage_work_groups > 0);
                for (const auto& other : prod) {
                    if (&other != &plan) assert(!(plan.selection == other.selection));
                }
            }
            if (budget == 0) {
                for (const auto& plan : prod) assert(!plan.staged());
            }
            if (budget == std::numeric_limits<uint64_t>::max()) {
                assert(prod.size() > 12);
                // Exactly one first legal leader per available retained family,
                // followed by the frozen cross-family score ordering.
                for (unsigned family = 0; family != 4; ++family)
                    assert(prod[family].selection.mode == static_cast<G::Mode>(family));
            }
        }
    }
    // In a dense legal inventory the production facade must clamp to 28,
    // without silently falling back to a smaller local-family shortlist.
    const auto full_ow = [](P::Fp8OneWindowPolicy::Variant) { return 6u; };
    const auto full_ak = [](P::Fp8AlignedKPolicy::Variant) { return 6u; };
    const auto full = P::selectProductionCandidates(
        {16, 512, 16, 170,
         {G::Fp8Format::E4M3, G::Combine::Sum, G::InputTransform::Identity},
         std::numeric_limits<uint64_t>::max()}, full_ow, dr, full_ak, empty_runtime);
    assert(full.size() == P::kFp8RKAutotuneCandidateBudget);

    const P::ProductionSelectionContext invalid{0, 2048, 256, 170,
        {G::Fp8Format::E4M3, G::Combine::Sum, G::InputTransform::Identity},
        std::numeric_limits<uint64_t>::max()};
    assert(P::selectProductionCandidates(invalid, ow, dr, ak, runtime).empty());
    // Callback-supplied physical availability, not XGBoost, controls legality.
    const auto none_ow = [](P::Fp8OneWindowPolicy::Variant) { return 0u; };
    const auto none_dr = [](P::Fp8DeepRSmallKPolicy::Variant) { return 0u; };
    const auto none_ak = [](P::Fp8AlignedKPolicy::Variant) { return 0u; };
    assert(P::selectProductionCandidates(
        {128, 2048, 256, 170, invalid.semantics,
         std::numeric_limits<uint64_t>::max()},
        none_ow, none_dr, none_ak, empty_runtime).empty());
    auto malformed = physical;
    malformed.first_stage_ctas = 1; // RuntimeK validation fails closed.
    const std::array malformed_registry{malformed};
    const auto legal_only = P::selectProductionCandidates(
        {128, 2048, 256, 170, invalid.semantics,
         std::numeric_limits<uint64_t>::max()},
        ow, dr, ak, malformed_registry);
    for (const auto& plan : legal_only)
        assert(plan.selection.mode != G::Mode::RuntimeK);
}
