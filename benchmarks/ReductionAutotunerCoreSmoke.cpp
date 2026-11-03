#include "Utilities/TensorOperations/Cub/ReductionAutotunerCore.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation::ReductionAutotuning;

int main() {
    const auto orders = makeRoundOrders(28, 1234567);
    assert(orders.size() == DefaultMeasurementCount);
    std::vector<size_t> sequential(28);
    std::iota(sequential.begin(), sequential.end(), 0);
    assert(orders.front() == sequential);
    for (size_t round = 0; round < DefaultMeasurementCount; ++round) {
        auto sorted = orders[round];
        std::sort(sorted.begin(), sorted.end());
        assert(sorted == sequential);
        if (round) assert(orders[round] != orders[round - 1]);
    }
    assert(orders == makeRoundOrders(28, 1234567));
    const auto twos = makeRoundOrders(2, 321);
    for (size_t round = 1; round < DefaultMeasurementCount; ++round) {
        assert(twos[round] != twos[round - 1]);
    }
    assert(makeRoundOrders(1, 10).size() == DefaultMeasurementCount);
    assert(makeRoundOrders(12, 99, 7).size() == 7);
    // Fast candidate is fastest on the MIDDLE THREE, despite extreme outliers.
    const std::vector<Measurements> samples = {
        Measurements{1.0, 4.0, 4.0, 4.0, 1000.0},
        Measurements{2.0, 3.0, 3.0, 3.0, 1000.0},
        Measurements{2.0, 5.0, 5.0, 5.0, 1000.0},
    };
    auto result = chooseWinner(samples, 1'000'000);
    assert(result.winner_index == 1);
    assert(std::abs(result.candidates[1].trimmed_mean_us - 3.0) < 1e-12);
    assert(std::abs(result.candidates[1].trimmed_mean_tb_per_s - (1.0/3.0)) < 1e-12);
    assert(chooseWinner({samples[1], samples[1]}, 1'000'000).winner_index == 0);
    // Seven rounds: drop the fastest and slowest, average the middle five.
    const auto seven = score(Measurements{9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0}, 1'000'000);
    assert(std::abs(seven.trimmed_mean_us - 4.6) < 1e-12);
    // Eleven rounds, trim 2 (Ultra): drop the two fastest and two slowest, average the middle seven.
    const auto eleven = score(Measurements{50.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 60.0}, 1'000'000, 2);
    assert(std::abs(eleven.trimmed_mean_us - (3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0 + 9.0) / 7.0) < 1e-12);
    bool overtrim = false;
    try { (void)score(Measurements{1.0, 2.0, 3.0, 4.0}, 100, 2); } catch (const std::invalid_argument&) { overtrim = true; }
    assert(overtrim);
    bool too_few = false;
    try { (void)score(Measurements{1.0, 2.0}, 100); } catch (const std::invalid_argument&) { too_few = true; }
    assert(too_few);
    // All backing rings (input/output/scratch) use this stride: preserve
    // 256-byte view alignment AND a separate >=128-byte padding region.
    constexpr std::array<uint64_t, 10> view_bytes = {
        1, 31, 32, 33, 127, 128, 129, 4095, 4096, 4097};
    for (uint64_t bytes : view_bytes) {
        const uint64_t stride = alignedStride(bytes);
        assert(stride % TensorViewAlignmentBytes == 0);
        assert(stride - bytes >= TensorViewPaddingBytes);
        assert(stride == ((bytes + TensorViewPaddingBytes + TensorViewAlignmentBytes - 1) &
                          ~(TensorViewAlignmentBytes - 1)));
    }
    // 1163: 256-byte starts (cudaMalloc's alignment).
    assert(TensorViewAlignmentBytes == 256);
    assert(alignedStride(129) == 512);
    assert(alignedStride(4096) == 4352);
    assert(alignedStride(128) == 256);
    bool overflow_threw = false;
    try { (void)alignedStride(std::numeric_limits<uint64_t>::max()); }
    catch (const std::invalid_argument&) { overflow_threw = true; }
    assert(overflow_threw);
    assert(coldSlotCount(1024, 8192) == 10);
    // 1161: probed contests. Launches per measurement ~ target / probe (capped, at least 1); a probe below the ~2 us
    // event resolution counts as one step; abandoned beyond ratio x max(best, floor).
    {
        const ProbePlan plan = planFromProbes({3.0, 0.0, 30.0, 25.0, 400.0}, 200.0, 32, 3.0, 8.0);
        // best = max(0, 2.048) = 2.048; limit = 3 x max(2.048, 8) = 24.
        assert((plan.measured == std::vector<bool>{true, true, false, false, false}));
        assert(plan.launches[0] == 32 && plan.launches[1] == 32 && plan.launches[2] == 0 && plan.launches[4] == 0);
        const ProbePlan large = planFromProbes({300.0, 330.0, 899.0, 901.0}, 200.0, 32, 3.0, 8.0);
        assert((large.measured == std::vector<bool>{true, true, true, false}));
        assert(large.launches[0] == 1 && large.launches[1] == 1 && large.launches[2] == 1);
        const ProbePlan mid = planFromProbes({20.0, 21.0}, 200.0, 32, 0.0, 8.0);  // ratio 0: nothing abandoned
        assert(mid.launches[0] == 10 && mid.launches[1] == 10 && mid.measured[1]);
        // The winner is the best MEASURED candidate; abandoned ones keep their slot and never win.
        const std::vector<Measurements> obs = {Measurements{5, 5, 5}, Measurements{1, 1, 1}, Measurements{4, 4, 4}};
        const auto masked = chooseWinner(obs, {true, false, true}, 1'000'000);
        assert(masked.winner_index == 2 && masked.candidates[1].abandoned && !masked.candidates[2].abandoned);
        assert(chooseWinner(obs, {true, true, true}, 1'000'000).winner_index ==
               chooseWinner(obs, 1'000'000).winner_index);
        bool none = false;
        try { (void)chooseWinner(obs, {false, false, false}, 1'000'000); } catch (const std::invalid_argument&) { none = true; }
        assert(none);
    }
    bool threw = false;
    try { (void)score(Measurements{1, 2, 0, 4, 5}, 100); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    std::cout << "Reduction autotuner CPU scheduler / trimmed-throughput scoring passes\n";
}
