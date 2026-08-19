#include "Utilities/TensorOperations/GpuMatrixMultiply/RaggedMatmulCapacityBuckets.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ThorImplementation {
namespace {

constexpr uint64_t MIN_BUCKET_ROWS = 8;
constexpr uint64_t BUCKETING_THRESHOLD_ROWS = 16;
constexpr uint64_t DENSE_QUARTER_OCTAVE_START_ROWS = 1024;

void appendIfWithin(std::vector<uint64_t> &buckets, uint64_t candidate, uint64_t fullCapacityRows) {
    if (candidate <= fullCapacityRows && (buckets.empty() || buckets.back() != candidate)) {
        buckets.push_back(candidate);
    }
}

}  // namespace

std::vector<uint64_t> makeRaggedMatmulCapacityBuckets(uint64_t fullCapacityRows) {
    if (fullCapacityRows == 0)
        throw std::invalid_argument("Ragged matmul full row capacity must be non-zero.");

    if (fullCapacityRows < BUCKETING_THRESHOLD_ROWS)
        return {fullCapacityRows};

    std::vector<uint64_t> buckets;

    // Keep the small-capacity family simple and stable.  These buckets are cheap
    // enough that powers of two provide useful reuse without meaningful padding
    // cost.  768 fills the otherwise-large 512 -> 1024 gap.
    for (uint64_t bucketRows = MIN_BUCKET_ROWS; bucketRows <= 512 && bucketRows <= fullCapacityRows;
         bucketRows *= 2) {
        buckets.push_back(bucketRows);
    }
    appendIfWithin(buckets, 768, fullCapacityRows);
    appendIfWithin(buckets, DENSE_QUARTER_OCTAVE_START_ROWS, fullCapacityRows);

    // Starting at 1024 rows, split each power-of-two octave into quarters:
    //
    //   B, 1.25B, 1.5B, 1.75B, 2B
    //
    // This bounds the normal bucket-to-bucket growth to 25%, avoiding the near
    // 2x physical over-read/work of the old power-of-two policy for large ragged
    // GEMMs.  Use subtraction against fullCapacityRows before addition so this
    // helper remains well-defined even for synthetic uint64_t test capacities.
    uint64_t octaveBase = DENSE_QUARTER_OCTAVE_START_ROWS;
    while (octaveBase < fullCapacityRows) {
        const uint64_t quarter = octaveBase / 4;
        const uint64_t remaining = fullCapacityRows - octaveBase;
        for (uint64_t quarterStep = 1; quarterStep <= 3; ++quarterStep) {
            const uint64_t delta = quarter * quarterStep;
            if (delta > remaining)
                break;
            appendIfWithin(buckets, octaveBase + delta, fullCapacityRows);
        }

        if (octaveBase > std::numeric_limits<uint64_t>::max() / 2)
            break;
        const uint64_t nextOctave = octaveBase * 2;
        if (nextOctave > fullCapacityRows)
            break;
        appendIfWithin(buckets, nextOctave, fullCapacityRows);
        octaveBase = nextOctave;
    }

    // The exact physical allocation is always a valid final bucket.  This also
    // covers capacities that fall between canonical quarter-octave classes.
    appendIfWithin(buckets, fullCapacityRows, fullCapacityRows);
    return buckets;
}

std::vector<uint64_t> makeRaggedRmsNormCapacityBuckets(uint64_t fullCapacityRows) {
    if (fullCapacityRows == 0)
        throw std::invalid_argument("Ragged RMSNorm full row capacity must be non-zero.");

    // RMSNorm deliberately mirrors the ragged GEMM capacity classes.  The
    // stamped RMSNorm execution object owns one workspace sized to the maximum
    // requirement across the bucket family, so finer bucket granularity no
    // longer multiplies executable workspace allocations.
    return makeRaggedMatmulCapacityBuckets(fullCapacityRows);
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
