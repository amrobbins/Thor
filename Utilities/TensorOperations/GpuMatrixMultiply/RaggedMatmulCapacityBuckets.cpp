#include "Utilities/TensorOperations/GpuMatrixMultiply/RaggedMatmulCapacityBuckets.h"

#include <algorithm>
#include <bit>
#include <stdexcept>

namespace ThorImplementation {
namespace {

constexpr uint64_t MIN_BUCKET_ROWS = 8;
constexpr uint64_t BUCKETING_THRESHOLD_ROWS = 16;

[[nodiscard]] uint64_t nearestPowerOfTwoToHalf(uint64_t fullCapacityRows) {
    const uint64_t halfFloor = fullCapacityRows / 2;
    const uint64_t lowerPower = std::bit_floor(halfFloor);
    const uint64_t upperPower = lowerPower * 2;

    // Compare fullCapacityRows / 2 with the arithmetic midpoint between
    // lowerPower and upperPower without using floating point:
    //
    //   fullCapacityRows / 2 ? (lowerPower + upperPower) / 2
    //   fullCapacityRows     ? 3 * lowerPower
    //
    // Choose the upper power at an exact tie.  Since fullCapacityRows >= 16
    // here, lowerPower is at least 8.  lowerPower is also at most 2^62 for a
    // uint64_t input, so 3 * lowerPower cannot overflow uint64_t.
    return fullCapacityRows < 3 * lowerPower ? lowerPower : upperPower;
}

}  // namespace

std::vector<uint64_t> makeRaggedMatmulCapacityBuckets(uint64_t fullCapacityRows) {
    if (fullCapacityRows == 0)
        throw std::invalid_argument("Ragged matmul full row capacity must be non-zero.");

    if (fullCapacityRows < BUCKETING_THRESHOLD_ROWS)
        return {fullCapacityRows};

    std::vector<uint64_t> descendingBuckets;
    uint64_t bucketRows = nearestPowerOfTwoToHalf(fullCapacityRows);
    while (bucketRows >= MIN_BUCKET_ROWS) {
        descendingBuckets.push_back(bucketRows);
        bucketRows /= 2;
    }

    std::reverse(descendingBuckets.begin(), descendingBuckets.end());
    if (descendingBuckets.empty() || descendingBuckets.back() != fullCapacityRows)
        descendingBuckets.push_back(fullCapacityRows);

    return descendingBuckets;
}

uint64_t chooseRaggedMatmulCapacityBucket(uint64_t activeRows, std::span<const uint64_t> capacityBuckets) {
    if (activeRows == 0)
        throw std::invalid_argument("Ragged matmul active row count must be non-zero when selecting a GEMM capacity bucket.");
    if (capacityBuckets.empty())
        throw std::invalid_argument("Ragged matmul capacity bucket list must be non-empty.");

    const auto bucket = std::lower_bound(capacityBuckets.begin(), capacityBuckets.end(), activeRows);
    if (bucket == capacityBuckets.end())
        throw std::invalid_argument("Ragged matmul active row count exceeds the full cached row capacity.");

    return *bucket;
}

}  // namespace ThorImplementation
