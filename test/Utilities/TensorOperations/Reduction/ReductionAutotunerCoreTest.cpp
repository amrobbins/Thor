// The contest scheduler and scoring shared by every reduction autotuner: shuffled round orders, trimmed-mean scores,
// the winner (best measured candidate), exact slice strides, cold slot counts and probe plans. Host-only. Slice
// alignment and padding for every size are in ReductionAutotunerSliceTest.

#include "Utilities/TensorOperations/Reduction/Contest/ReductionAutotunerCore.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation::ReductionAutotuning;

TEST(ReductionAutotunerCore, RoundOrdersArePermutationsThatChangeEveryRound) {
    const auto orders = makeRoundOrders(28, 1234567);
    ASSERT_EQ(orders.size(), DefaultMeasurementCount);
    std::vector<size_t> sequential(28);
    std::iota(sequential.begin(), sequential.end(), 0);
    EXPECT_EQ(orders.front(), sequential);
    for (size_t round = 0; round < DefaultMeasurementCount; ++round) {
        auto sorted = orders[round];
        std::sort(sorted.begin(), sorted.end());
        EXPECT_EQ(sorted, sequential);
        if (round) EXPECT_NE(orders[round], orders[round - 1]);
    }
    EXPECT_EQ(orders, makeRoundOrders(28, 1234567));  // deterministic per seed
    const auto twos = makeRoundOrders(2, 321);
    for (size_t round = 1; round < DefaultMeasurementCount; ++round) EXPECT_NE(twos[round], twos[round - 1]);
    EXPECT_EQ(makeRoundOrders(1, 10).size(), DefaultMeasurementCount);
    EXPECT_EQ(makeRoundOrders(12, 99, 7).size(), 7u);
}

TEST(ReductionAutotunerCore, ScoresAreTrimmedMeans) {
    // The fastest on the middle three wins, despite extreme outliers.
    const std::vector<Measurements> samples = {
        Measurements{1.0, 4.0, 4.0, 4.0, 1000.0},
        Measurements{2.0, 3.0, 3.0, 3.0, 1000.0},
        Measurements{2.0, 5.0, 5.0, 5.0, 1000.0},
    };
    const auto result = chooseWinner(samples, 1'000'000);
    EXPECT_EQ(result.winner_index, 1u);
    EXPECT_NEAR(result.candidates[1].trimmed_mean_us, 3.0, 1e-12);
    EXPECT_NEAR(result.candidates[1].trimmed_mean_tb_per_s, 1.0 / 3.0, 1e-12);
    EXPECT_EQ(chooseWinner({samples[1], samples[1]}, 1'000'000).winner_index, 0u);  // a tie goes to the first
    // Seven rounds: drop the fastest and slowest, average the middle five.
    EXPECT_NEAR(score(Measurements{9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0}, 1'000'000).trimmed_mean_us, 4.6, 1e-12);
    // Eleven rounds, trim 2 (Ultra): average the middle seven.
    EXPECT_NEAR(score(Measurements{50.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 60.0}, 1'000'000, 2)
                    .trimmed_mean_us,
                (3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0 + 9.0) / 7.0, 1e-12);
    EXPECT_THROW((void)score(Measurements{1.0, 2.0, 3.0, 4.0}, 100, 2), std::invalid_argument);  // over-trimmed
    EXPECT_THROW((void)score(Measurements{1.0, 2.0}, 100), std::invalid_argument);               // too few rounds
    EXPECT_THROW((void)score(Measurements{1, 2, 0, 4, 5}, 100), std::invalid_argument);           // a zero sample
}

TEST(ReductionAutotunerCore, SliceStridesAndColdSlots) {
    EXPECT_EQ(SliceAlignmentBytes, 256u);  // cudaMalloc's alignment
    EXPECT_EQ(alignedStride(128), 256u);
    EXPECT_EQ(alignedStride(129), 512u);
    EXPECT_EQ(alignedStride(4096), 4352u);
    EXPECT_THROW((void)alignedStride(std::numeric_limits<uint64_t>::max()), std::invalid_argument);
    EXPECT_EQ(coldSlotCount(1024, 8192), 10u);
}

// Launches per measurement are about target / probe (capped, at least 1); a probe below the ~2 us event resolution
// counts as one step; a candidate is abandoned beyond ratio x max(best, floor).
TEST(ReductionAutotunerCore, ProbePlansSizeLaunchesAndAbandonSlowCandidates) {
    // best = max(0, 2.048) = 2.048; limit = 3 x max(2.048, 8) = 24.
    const ProbePlan plan = planFromProbes({3.0, 0.0, 30.0, 25.0, 400.0}, 200.0, 32, 3.0, 8.0);
    EXPECT_EQ(plan.measured, (std::vector<bool>{true, true, false, false, false}));
    EXPECT_EQ(plan.launches[0], 32u);
    EXPECT_EQ(plan.launches[1], 32u);
    EXPECT_EQ(plan.launches[2], 0u);
    EXPECT_EQ(plan.launches[4], 0u);
    const ProbePlan large = planFromProbes({300.0, 330.0, 899.0, 901.0}, 200.0, 32, 3.0, 8.0);
    EXPECT_EQ(large.measured, (std::vector<bool>{true, true, true, false}));
    EXPECT_EQ(large.launches[0], 1u);
    EXPECT_EQ(large.launches[1], 1u);
    EXPECT_EQ(large.launches[2], 1u);
    const ProbePlan mid = planFromProbes({20.0, 21.0}, 200.0, 32, 0.0, 8.0);  // ratio 0: nothing is abandoned
    EXPECT_EQ(mid.launches[0], 10u);
    EXPECT_EQ(mid.launches[1], 10u);
    EXPECT_TRUE(mid.measured[1]);
}

TEST(ReductionAutotunerCore, TheWinnerIsTheBestMeasuredCandidate) {
    const std::vector<Measurements> observed = {Measurements{5, 5, 5}, Measurements{1, 1, 1}, Measurements{4, 4, 4}};
    const auto masked = chooseWinner(observed, {true, false, true}, 1'000'000);
    EXPECT_EQ(masked.winner_index, 2u);  // an abandoned candidate keeps its slot and never wins
    EXPECT_TRUE(masked.candidates[1].abandoned);
    EXPECT_FALSE(masked.candidates[2].abandoned);
    EXPECT_EQ(chooseWinner(observed, {true, true, true}, 1'000'000).winner_index,
              chooseWinner(observed, 1'000'000).winner_index);
    EXPECT_THROW((void)chooseWinner(observed, {false, false, false}, 1'000'000), std::invalid_argument);
}
