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
    assert(orders.size() == MeasurementCount);
    std::vector<size_t> sequential(28);
    std::iota(sequential.begin(), sequential.end(), 0);
    assert(orders.front() == sequential);
    for (size_t round = 0; round < MeasurementCount; ++round) {
        auto sorted = orders[round];
        std::sort(sorted.begin(), sorted.end());
        assert(sorted == sequential);
        if (round) assert(orders[round] != orders[round - 1]);
    }
    assert(orders == makeRoundOrders(28, 1234567));
    const auto twos = makeRoundOrders(2, 321);
    for (size_t round = 1; round < MeasurementCount; ++round) {
        assert(twos[round] != twos[round - 1]);
    }
    assert(makeRoundOrders(1, 10).size() == MeasurementCount);
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
    // All backing rings (input/output/scratch) use this stride: preserve
    // 32-byte view alignment AND a separate >=128-byte padding region.
    constexpr std::array<uint64_t, 10> view_bytes = {
        1, 31, 32, 33, 127, 128, 129, 4095, 4096, 4097};
    for (uint64_t bytes : view_bytes) {
        const uint64_t stride = alignedStride(bytes);
        assert(stride % TensorViewAlignmentBytes == 0);
        assert(stride - bytes >= TensorViewPaddingBytes);
        if (bytes % TensorViewAlignmentBytes == 0)
            assert(stride == bytes + TensorViewPaddingBytes);
    }
    assert(alignedStride(129) == 288);
    assert(alignedStride(4096) == 4224);
    bool overflow_threw = false;
    try { (void)alignedStride(std::numeric_limits<uint64_t>::max()); }
    catch (const std::invalid_argument&) { overflow_threw = true; }
    assert(overflow_threw);
    assert(coldSlotCount(1024, 8192) == 10);
    bool threw = false;
    try { (void)score(Measurements{1, 2, 0, 4, 5}, 100); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    std::cout << "Reduction autotuner CPU scheduler / trimmed-throughput scoring passes\n";
}
