// 1167: the planner's lazy best-first site selection, driven by scripted physical selections (the selector's
// proposal_override). A site's bound is the bytes left if one pass reduced it fully; sites are selected in (bound,
// KR before RK, outer before inner) order only while the bound can still beat the best candidate so far. The plan must
// equal the one that selecting every site would give. Host-only.
#include "Utilities/TensorOperations/Cub/DenseReductionPlanner.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionInternal;

namespace {

std::string key(const DenseReductionProblem& problem, DenseReductionSite site) {
    std::ostringstream out;
    out << (problem.input_state == DenseReductionInputState::OriginalInput ? 'o' : 'i');
    for (const auto& run : problem.runs) out << ',' << static_cast<int>(run.kind) << ':' << run.extent;
    out << '@' << site.run_index;
    return out.str();
}

// What each (problem, site) selection returns: the partial extent left in the site (1: complete). Unscripted sites
// complete. Every call is recorded.
struct Script {
    std::map<std::string, uint64_t> partial;
    std::vector<std::pair<std::string, DenseReducerFamily>> calls;
    // Random mode: a deterministic pseudo-random partial extent per (problem, site).
    bool random = false;
    uint64_t seed = 0;
};

uint64_t mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

uint64_t partialExtent(Script& script, const DenseReductionProblem& problem, DenseReductionSite site) {
    const std::string k = key(problem, site);
    if (!script.random) {
        const auto it = script.partial.find(k);
        return it == script.partial.end() ? 1 : it->second;
    }
    const uint64_t extent = problem.runs[site.run_index].extent;
    uint64_t h = script.seed;
    for (char c : k) h = mix(h ^ static_cast<unsigned char>(c));
    if (extent < 3 || h % 3 != 0) return 1;  // two thirds of selections complete their site
    return 2 + (h >> 8) % (extent - 2);       // a partial extent in [2, extent - 1]
}

DenseReductionCandidate scripted(const DenseReductionProblem& problem, DenseReductionSite site,
                                 DenseReducerFamily family, const DenseRKPlanningContext&, void* data) {
    auto& script = *static_cast<Script*>(data);
    script.calls.push_back({key(problem, site), family});
    DenseReductionCandidate candidate;
    candidate.family = family;
    candidate.site = site;
    candidate.physical_plan = std::make_shared<DenseReductionPhysicalPlan>();
    // Pure [R] must finish in one pass (the planner's fast path), so it always completes.
    const uint64_t extent = family == DenseReducerFamily::R ? 1 : partialExtent(script, problem, site);
    candidate.next_problem = advanceDenseReductionProblem(problem, site, extent);
    candidate.next_bytes = denseReductionProblemBytes(candidate.next_problem);
    return candidate;
}

DenseReductionProblem valueProblem(const std::vector<uint64_t>& dimensions, const std::vector<uint32_t>& axes) {
    return makeInitialDenseReductionProblem(dimensions, axes, DenseReductionAggregateKind::Value,
                                            DenseReductionOutputSpec{});
}

DensePhysicalSelectionContext scriptedContext(Script& script) {
    DensePhysicalSelectionContext context;
    context.proposal_override = scripted;
    context.proposal_override_context = &script;
    return context;
}

// The pre-1167 rule as a reference: select every site, keep the smallest next_bytes, ties KR before RK, then the
// lower run index.
int rank(DenseReducerFamily family) { return family == DenseReducerFamily::KR ? 0 : 1; }
std::vector<DenseReductionCandidate> exhaustivePlan(DenseReductionProblem current, Script& script) {
    std::vector<DenseReductionCandidate> plan;
    while (hasDenseReductionSites(current)) {
        const auto sites = enumerateDenseReductionSites(current);
        std::optional<DenseReductionCandidate> best;
        for (DenseReductionSite site : sites) {
            const DenseReducerFamily family = classifyDenseReducerFamily(current, site);
            const DenseReductionCandidate c = scripted(current, site, family, DenseRKPlanningContext{}, &script);
            const bool better = !best || c.next_bytes < best->next_bytes ||
                (c.next_bytes == best->next_bytes &&
                 (rank(c.family) < rank(best->family) ||
                  (rank(c.family) == rank(best->family) && c.site.run_index < best->site.run_index)));
            if (better) best = c;
        }
        plan.push_back(*best);
        current = best->next_problem;
    }
    return plan;
}

}  // namespace

TEST(CubDenseReductionLazyPlanner, ACompletePassOnTheBestBoundEndsTheStep) {
    // [R=8, K=3, R=5]: the RK site's bound is 3*5, the KR site's 8*3.
    Script script;
    const auto plan = planDenseReduction(valueProblem({8, 3, 5}, {0, 2}), scriptedContext(script));
    ASSERT_EQ(plan.steps.size(), 2U);
    ASSERT_EQ(plan.steps[0].considered_candidates.size(), 1U);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::RK);
    EXPECT_FALSE(plan.steps[0].site_bounds[1].selected);
    EXPECT_EQ(script.calls.size(), 2U);  // one selection per step
}

TEST(CubDenseReductionLazyPlanner, APartialPassLargerThanTheNextBoundTriesTheNextSite) {
    // RK leaves 4 partials of [3,5] (240 B), more than the KR bound (8*3*4 = 96 B): KR is selected and wins.
    Script script;
    const auto problem = valueProblem({8, 3, 5}, {0, 2});
    script.partial[key(problem, DenseReductionSite{0})] = 4;
    const auto plan = planDenseReduction(problem, scriptedContext(script));
    ASSERT_EQ(plan.steps[0].considered_candidates.size(), 2U);
    EXPECT_EQ(plan.steps[0].considered_candidates[0].next_bytes, 4U * 3U * 5U * 4U);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::KR);
    EXPECT_EQ(plan.steps[0].selected_candidate.next_bytes, 8U * 3U * 4U);
    EXPECT_TRUE(plan.steps[0].site_bounds[1].selected);
}

TEST(CubDenseReductionLazyPlanner, APartialPassSmallerThanEveryOtherBoundIsKeptWithoutTryingThem) {
    // [R=16, K=3, R=5]: RK bound 60 B, KR bound 192 B. RK leaves 2 partials (120 B), still below KR's bound.
    Script script;
    const auto problem = valueProblem({16, 3, 5}, {0, 2});
    script.partial[key(problem, DenseReductionSite{0})] = 2;
    const auto plan = planDenseReduction(problem, scriptedContext(script));
    ASSERT_EQ(plan.steps[0].considered_candidates.size(), 1U);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::RK);
    EXPECT_EQ(plan.steps[0].selected_candidate.next_bytes, 2U * 3U * 5U * 4U);
    EXPECT_FALSE(plan.steps[0].site_bounds[1].selected);
    // The next step reduces the RK partials and the KR site from the new problem.
    EXPECT_TRUE(hasDenseReductionSites(plan.steps[0].selected_candidate.next_problem));
}

TEST(CubDenseReductionLazyPlanner, APartialPassLosesToALargerBoundThatCompletes) {
    // RK leaves 8 partials... of R=16: 8*15*4 = 480 B > KR bound 192 B; KR completes (192 B) and wins.
    Script script;
    const auto problem = valueProblem({16, 3, 5}, {0, 2});
    script.partial[key(problem, DenseReductionSite{0})] = 8;
    const auto plan = planDenseReduction(problem, scriptedContext(script));
    ASSERT_EQ(plan.steps[0].considered_candidates.size(), 2U);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::KR);
}

TEST(CubDenseReductionLazyPlanner, TiesVisitKRFirstThenTheOuterSite) {
    Script script;
    // [R=5, K=3, R=5]: equal bounds; KR first, and it completes, so RK is not selected.
    const auto kr = planDenseReduction(valueProblem({5, 3, 5}, {0, 2}), scriptedContext(script));
    ASSERT_EQ(kr.steps[0].considered_candidates.size(), 1U);
    EXPECT_EQ(kr.steps[0].selected_candidate.family, DenseReducerFamily::KR);
    // [R=4, K=2, R=4, K=2]: two RK sites with equal bounds; the outer one first.
    const auto rk = planDenseReduction(valueProblem({4, 2, 4, 2}, {0, 2}), scriptedContext(script));
    ASSERT_EQ(rk.steps[0].considered_candidates.size(), 1U);
    EXPECT_EQ(rk.steps[0].selected_candidate.site.run_index, 0U);
}

TEST(CubDenseReductionLazyPlanner, ATiedSiteIsTriedWhenTheFirstIsPartial) {
    // [R=5, K=3, R=5]: KR first (tie), but it leaves 2 partials (2*5*3*4 = 120 B > RK bound 60 B): RK is selected.
    Script script;
    const auto problem = valueProblem({5, 3, 5}, {0, 2});
    script.partial[key(problem, DenseReductionSite{2})] = 2;
    const auto plan = planDenseReduction(problem, scriptedContext(script));
    ASSERT_EQ(plan.steps[0].considered_candidates.size(), 2U);
    EXPECT_EQ(plan.steps[0].selected_candidate.family, DenseReducerFamily::RK);
}

TEST(CubDenseReductionLazyPlanner, SamePlanAsSelectingEverySiteOnRandomProblems) {
    std::mt19937_64 rng(1167);
    size_t lazy_calls = 0, exhaustive_calls = 0, problems = 0;
    for (int trial = 0; trial < 600; ++trial) {
        const size_t rank_count = 2 + rng() % 5;  // 2..6 dimensions
        std::vector<uint64_t> dims;
        std::vector<uint32_t> axes;
        for (size_t d = 0; d < rank_count; ++d) {
            dims.push_back(1 + rng() % 9);
            if (rng() % 2) axes.push_back(static_cast<uint32_t>(d));
        }
        if (axes.empty()) axes.push_back(0);
        const auto problem = valueProblem(dims, axes);
        if (!hasDenseReductionSites(problem)) continue;
        Script lazy;
        lazy.random = true;
        lazy.seed = static_cast<uint64_t>(trial);
        Script reference = lazy;
        const auto plan = planDenseReduction(problem, scriptedContext(lazy));
        const auto expected = exhaustivePlan(problem, reference);
        ASSERT_EQ(plan.steps.size(), expected.size()) << "trial " << trial;
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(plan.steps[i].selected_candidate.site, expected[i].site) << "trial " << trial << " step " << i;
            EXPECT_EQ(plan.steps[i].selected_candidate.family, expected[i].family) << "trial " << trial;
            EXPECT_EQ(plan.steps[i].selected_candidate.next_problem, expected[i].next_problem) << "trial " << trial;
        }
        lazy_calls += lazy.calls.size();
        exhaustive_calls += reference.calls.size();
        ++problems;
    }
    EXPECT_GT(problems, 400U);
    EXPECT_LT(lazy_calls, exhaustive_calls);  // fewer selections (autotuning contests) for the same plans
}
