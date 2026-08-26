#include "Utilities/Expression/SqueezeAxes.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <vector>

using namespace ThorImplementation;

TEST(SqueezeAxes, ReductionUnsqueezeAccountsForScalarPlaceholder) {
    const uint64_t all_singletons = std::numeric_limits<uint64_t>::max();

    // A fully squeezed reduction is represented by Thor as shape [1], not as a
    // rank-zero tensor.  That existing singleton stands in for one of the axes
    // removed by squeeze, so only the remaining axes need to be reintroduced.
    EXPECT_EQ(normalizedReductionUnsqueezeAxes({4}, {}, {all_singletons}), (std::vector<uint64_t>{}));
    EXPECT_EQ(normalizedReductionUnsqueezeAxes({2, 3}, {}, {all_singletons}), (std::vector<uint64_t>{0}));
    EXPECT_EQ(normalizedReductionUnsqueezeAxes({2, 3, 4}, {}, {all_singletons}), (std::vector<uint64_t>{0, 1}));
}

TEST(SqueezeAxes, PartialReductionStillRestoresActualSqueezedAxes) {
    const uint64_t all_singletons = std::numeric_limits<uint64_t>::max();

    EXPECT_EQ(normalizedReductionUnsqueezeAxes({2, 3}, {1}, {all_singletons}), (std::vector<uint64_t>{1}));
    EXPECT_EQ(normalizedReductionUnsqueezeAxes({2, 3, 4}, {0, 2}, {all_singletons}), (std::vector<uint64_t>{0, 2}));
}
