// 1158: the retained FP8 KR candidate pool, its ranker features and the ranked shortlist; once a trained
// Fp8KRRankerModel.h is installed, its margins reproduce the training pipeline's fixtures exactly. Host-only.

#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRRankerModel.h"
#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRSegments.h"
#include "test/Utilities/TensorOperations/Fp8KRRankerParityFixtures.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace KR = ThorImplementation::CubReductionInternal::Fp8KR;

namespace {

const std::vector<std::pair<uint64_t, uint64_t>>& rankerSites() {
    static const std::vector<std::pair<uint64_t, uint64_t>> sites{
        {1, 1},        {1, 7},       {1, 4096},     {1, 1ULL << 30}, {3, 2},      {1000000, 5}, {64, 31},
        {17, 129},     {8, 4100},    {2, 70001},    {4, 262147},     {170, 1024}, {100000, 64}, {4096, 4096},
        {2720, 8192},  {5000, 1000}, {11298, 46814}, {600, 8166},    {1, 18509157}, {100, 1ULL << 20}};
    return sites;
}

}  // namespace

TEST(Fp8KRRanker, PoolStartsWithTheListAndIsLegalAndDistinct) {
    for (const auto& [rows, row_bytes] : rankerSites()) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << row_bytes);
        const auto listed = KR::fp8KRCandidateShapes(rows, row_bytes, 170);
        const auto pool = KR::fp8KRCandidatePool(rows, row_bytes, 170);
        ASSERT_GE(pool.size(), listed.size());
        std::set<std::string> names;
        for (size_t i = 0; i < pool.size(); ++i) {
            const auto& entry = pool[i];
            EXPECT_TRUE(KR::fp8KRSegmentShapeLegal(entry.shape));
            EXPECT_TRUE(names.insert(KR::fp8KRShapeName(entry.shape)).second);
            if (i < listed.size()) {
                EXPECT_EQ(listed[i], entry.shape);
                EXPECT_EQ(i + 1, entry.list_position);
            } else {
                EXPECT_EQ(0u, entry.list_position);
            }
        }
    }
}

TEST(Fp8KRRanker, FeaturesAreFiniteAndDescribeTheEntry) {
    const KR::Fp8KRRankerSite value_site{0, 0, 0, false};
    const KR::Fp8KRRankerSite arg_site{2, 3, 0, true};
    for (const auto& [rows, row_bytes] : rankerSites()) {
        for (const auto& entry : KR::fp8KRCandidatePool(rows, row_bytes, 170)) {
            for (const auto& site : {value_site, arg_site}) {
                const auto f = KR::fp8KRRankerFeatures(entry, site, 170);
                for (float x : f) EXPECT_TRUE(std::isfinite(x));
                EXPECT_FLOAT_EQ(std::log2(static_cast<double>(rows)), f[0]);
                EXPECT_FLOAT_EQ(std::log2(static_cast<double>(row_bytes)), f[1]);
                EXPECT_EQ(static_cast<float>(site.kind), f[17]);
                EXPECT_EQ(static_cast<float>(entry.list_position), f[22]);
            }
        }
    }
}

TEST(Fp8KRRanker, ShortlistIsAPrefixOfTheRankedPool) {
    const KR::Fp8KRRankerSite site{1, 2, 0, false};
    for (const auto& [rows, row_bytes] : rankerSites()) {
        const auto pool = KR::fp8KRCandidatePool(rows, row_bytes, 170);
        for (size_t count : {size_t{1}, size_t{3}, size_t{6}, size_t{100}}) {
            const auto shortlist = KR::fp8KRRankedShapes(rows, row_bytes, 170, site, count);
            if (!KR::fp8KRRankerTrained()) {
                auto listed = KR::fp8KRCandidateShapes(rows, row_bytes, 170);
                if (listed.size() > count) listed.resize(count);
                EXPECT_EQ(listed, shortlist);
                continue;
            }
            EXPECT_EQ(std::min(count, pool.size()), shortlist.size());
            for (const auto& shape : shortlist)
                EXPECT_TRUE(std::any_of(pool.begin(), pool.end(), [&](const auto& e) { return e.shape == shape; }));
        }
    }
}

TEST(Fp8KRRanker, ModelMatchesTrainingFixturesExactly) {
    static_assert(KR::Fp8KRRankerModel::FEATURE_COUNT == KR::FP8_KR_RANKER_FEATURES);
    if (!KR::Fp8KRRankerModel::TRAINED) {
        GTEST_SKIP() << "no trained FP8 KR ranker installed yet";
    }
    ASSERT_FALSE(KR::Fp8KRRankerParity::fixtures.empty());
    for (size_t i = 0; i < KR::Fp8KRRankerParity::fixtures.size(); ++i) {
        const auto& fixture = KR::Fp8KRRankerParity::fixtures[i];
        const float margin = KR::Fp8KRRankerModel::margin(fixture.features);
        ASSERT_EQ(std::bit_cast<uint32_t>(fixture.margin), std::bit_cast<uint32_t>(margin)) << "fixture " << i;
    }
}
