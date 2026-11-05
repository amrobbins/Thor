// 1182: the FP32 KR ranker (one ranker per dtype): the FP32 pool and its ARG restriction, the ranked shortlist with
// and without a trained model, and, once a trained Fp32KRRankerModel.h is installed, its margins against the training
// pipeline's fixtures, bit for bit. Host-only.

#include "Utilities/TensorOperations/Cub/Fp32KR/Fp32KRRankerModel.h"
#include "Utilities/TensorOperations/Cub/Fp32KR/Fp32KRStampAutotuner.h"
#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRSegments.h"
#include "test/Utilities/TensorOperations/Fp32KRRankerParityFixtures.h"

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
namespace Fp32KR = ThorImplementation::CubReductionInternal::Fp32KR;
using ThorImplementation::CubReductionTuningDepth;

namespace {

// (rows, elements): row bytes are 4 * elements.
const std::vector<std::pair<uint64_t, uint64_t>>& fp32RankerSites() {
    static const std::vector<std::pair<uint64_t, uint64_t>> sites{
        {1, 1},       {1, 7},        {1, 1024},  {1, 1ULL << 28}, {3, 2},     {1000000, 5}, {64, 31},
        {17, 129},    {8, 1025},     {2, 70001}, {4, 262147},     {170, 256}, {100000, 16}, {4096, 1024},
        {2720, 2048}, {5000, 250},   {512, 16384}, {600, 2042},   {1, 18509157}, {100, 1ULL << 18}};
    return sites;
}

const KR::Fp8KRRankerSite VALUE_SITE{0, 0, 0, false};

}  // namespace

TEST(Fp32KRRanker, PoolIsTheFp32PoolAndArgDropsOnlyIllegalShapes) {
    for (const auto& [rows, elements] : fp32RankerSites()) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << elements);
        const uint64_t row_bytes = 4 * elements;
        const auto retained = KR::retainedKRCandidatePool(rows, row_bytes, 170, 4);
        const auto value_pool = Fp32KR::fp32KRCandidatePool(rows, row_bytes, 170, VALUE_SITE);
        ASSERT_EQ(retained.size(), value_pool.size());
        std::set<std::string> names;
        for (size_t i = 0; i < value_pool.size(); ++i) {
            EXPECT_EQ(retained[i].shape, value_pool[i].shape);
            EXPECT_EQ(retained[i].list_position, value_pool[i].list_position);
            EXPECT_TRUE(KR::retainedKRShapeLegalForElement(value_pool[i].shape, 4));
            EXPECT_TRUE(names.insert(KR::fp8KRShapeName(value_pool[i].shape)).second);
        }
        const auto arg_pool =
            Fp32KR::fp32KRCandidatePool(rows, row_bytes, 170, Fp32KR::fp32KRArgRankerSite(true, true));
        size_t legal = 0;
        for (const auto& entry : retained) legal += KR::fp32KRArgShapeLegal(entry.shape) ? 1 : 0;
        EXPECT_EQ(legal, arg_pool.size());
        for (const auto& entry : arg_pool) EXPECT_TRUE(KR::fp32KRArgShapeLegal(entry.shape));
    }
}

TEST(Fp32KRRanker, FeaturesAreFiniteAndDescribeTheEntry) {
    for (const auto& [rows, elements] : fp32RankerSites()) {
        for (const auto& site : {VALUE_SITE, Fp32KR::fp32KRArgRankerSite(false, false)}) {
            for (const auto& entry : Fp32KR::fp32KRCandidatePool(rows, 4 * elements, 170, site)) {
                const auto f = KR::fp8KRRankerFeatures(entry, site, 170);
                for (float x : f) EXPECT_TRUE(std::isfinite(x));
                EXPECT_FLOAT_EQ(std::log2(static_cast<double>(rows)), f[0]);
                EXPECT_FLOAT_EQ(std::log2(static_cast<double>(4 * elements)), f[1]);
                EXPECT_EQ(0.0f, f[14]);  // row bytes mod 4: FP32 rows are whole words
                EXPECT_EQ(static_cast<float>(site.kind), f[17]);
                EXPECT_EQ(0.0f, f[20]);  // E5M2
                EXPECT_EQ(static_cast<float>(entry.list_position), f[22]);
            }
        }
    }
}

TEST(Fp32KRRanker, ShortlistIsTheListPrefixUntrainedAndARankedPoolPrefixTrained) {
    const auto arg_site = Fp32KR::fp32KRArgRankerSite(true, false);
    for (const auto& [rows, elements] : fp32RankerSites()) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << elements);
        const uint64_t row_bytes = 4 * elements;
        for (const auto& site : {VALUE_SITE, arg_site}) {
            const auto pool = Fp32KR::fp32KRCandidatePool(rows, row_bytes, 170, site);
            for (size_t count : {size_t{1}, size_t{3}, size_t{6}, size_t{100}}) {
                const auto shortlist = Fp32KR::fp32KRRankedShapes(rows, row_bytes, 170, site, count);
                if (!Fp32KR::fp32KRRankerTrained()) {
                    std::vector<KR::Fp8KRSegmentShape> listed;
                    for (const auto& entry : pool)
                        if (entry.list_position != 0 && listed.size() < count) listed.push_back(entry.shape);
                    EXPECT_EQ(listed, shortlist);
                    continue;
                }
                EXPECT_EQ(std::min(count, pool.size()), shortlist.size());
                for (const auto& shape : shortlist)
                    EXPECT_TRUE(std::any_of(pool.begin(), pool.end(), [&](const auto& e) { return e.shape == shape; }));
            }
        }
        // The depth lengths: the RK depths untrained, FP32_KR_SHORTLIST_DEPTHS trained.
        const bool trained = Fp32KR::fp32KRRankerTrained();
        EXPECT_EQ(trained ? Fp32KR::FP32_KR_SHORTLIST_DEPTHS[0] : 6u,
                  Fp32KR::fp32KRShortlistLength(CubReductionTuningDepth::Quick));
        EXPECT_EQ(trained ? Fp32KR::FP32_KR_SHORTLIST_DEPTHS[1] : 12u,
                  Fp32KR::fp32KRShortlistLength(CubReductionTuningDepth::Regular));
        EXPECT_EQ(trained ? Fp32KR::FP32_KR_SHORTLIST_DEPTHS[2] : 24u,
                  Fp32KR::fp32KRShortlistLength(CubReductionTuningDepth::Ultra));
    }
}

TEST(Fp32KRRanker, ModelMatchesTrainingFixturesExactly) {
    static_assert(Fp32KR::Fp32KRRankerModel::FEATURE_COUNT == KR::FP8_KR_RANKER_FEATURES);
    if (!Fp32KR::Fp32KRRankerModel::TRAINED) {
        GTEST_SKIP() << "no trained FP32 KR ranker installed yet";
    }
    ASSERT_FALSE(Fp32KR::Fp32KRRankerParity::fixtures.empty());
    for (size_t i = 0; i < Fp32KR::Fp32KRRankerParity::fixtures.size(); ++i) {
        const auto& fixture = Fp32KR::Fp32KRRankerParity::fixtures[i];
        const float margin = Fp32KR::Fp32KRRankerModel::margin(fixture.features);
        ASSERT_EQ(std::bit_cast<uint32_t>(fixture.margin), std::bit_cast<uint32_t>(margin)) << "fixture " << i;
    }
}
