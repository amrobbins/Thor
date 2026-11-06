// 1238: the FP16 / BF16 KR ranker (one ranker for both formats, the BF16 flag in the alternate-format feature): the
// 2-byte pool and its ARG restriction, the features, the ranked shortlist without a trained model, and, once a trained
// F16KRRankerModel.h is installed, its margins against the training pipeline's fixtures, bit for bit. Host-only.

#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/F16KR/F16KRRankerModel.h"
#include "Utilities/TensorOperations/Cub/F16KR/F16KRStampAutotuner.h"
#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRSegments.h"
#include "test/Utilities/TensorOperations/F16KRRankerParityFixtures.h"

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
namespace F16KR = ThorImplementation::CubReductionInternal::F16KR;
using ThorImplementation::CubReductionTuningDepth;
using ThorImplementation::DataType;

namespace {

// (rows, elements): row bytes are 2 * elements.
const std::vector<std::pair<uint64_t, uint64_t>>& f16RankerSites() {
    static const std::vector<std::pair<uint64_t, uint64_t>> sites{
        {1, 1},       {1, 7},      {1, 1024},    {1, 1ULL << 28}, {3, 2},        {1000000, 5}, {64, 31},
        {17, 129},    {8, 1025},   {2, 70001},   {4, 262147},     {170, 256},    {100000, 16}, {4096, 1024},
        {2720, 2048}, {5000, 250}, {512, 16384}, {600, 2042},     {1, 18509157}, {100, 1ULL << 18}, {60000, 3}};
    return sites;
}

const KR::Fp8KRRankerSite VALUE_SITE{0, 0, 0, false};

}  // namespace

TEST(F16KRRanker, PoolIsTheTwoBytePoolAndArgDropsOnlyVectorPack) {
    for (const auto& [rows, elements] : f16RankerSites()) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << elements);
        const uint64_t row_bytes = 2 * elements;
        const auto retained = KR::retainedKRCandidatePool(rows, row_bytes, 170, 2);
        const auto value_pool = F16KR::f16KRCandidatePool(rows, row_bytes, 170, VALUE_SITE);
        ASSERT_EQ(retained.size(), value_pool.size());
        std::set<std::string> names;
        for (size_t i = 0; i < value_pool.size(); ++i) {
            EXPECT_EQ(retained[i].shape, value_pool[i].shape);
            EXPECT_EQ(retained[i].list_position, value_pool[i].list_position);
            EXPECT_TRUE(KR::retainedKRShapeLegalForElement(value_pool[i].shape, 2));
            EXPECT_TRUE(names.insert(KR::fp8KRShapeName(value_pool[i].shape)).second);
        }
        const auto arg_pool =
            F16KR::f16KRCandidatePool(rows, row_bytes, 170, F16KR::f16KRArgRankerSite(true, true, true));
        size_t legal = 0;
        for (const auto& entry : retained) legal += entry.shape.vector ? 0 : 1;
        EXPECT_EQ(legal, arg_pool.size());
        for (const auto& entry : arg_pool) EXPECT_TRUE(KR::f16KRArgShapeLegal(entry.shape));
    }
}

TEST(F16KRRanker, FeaturesAreFiniteAndCarryTheBf16Flag) {
    using namespace ThorImplementation::CubReductionInternal;
    const CubReductionStageSemantics sum =
        makeValueReductionStageSemantics(ThorImplementation::CubReductionOp::Sum, DensePassRole::Complete, 8);
    EXPECT_FALSE(fp8KRValueRankerSite(sum, DataType::FP16).e5m2);
    EXPECT_TRUE(fp8KRValueRankerSite(sum, DataType::BF16).e5m2);
    EXPECT_FALSE(fp8KRValueRankerSite(sum, DataType::FP32).e5m2);
    EXPECT_TRUE(F16KR::f16KRArgRankerSite(false, false, true).e5m2);
    EXPECT_FALSE(F16KR::f16KRArgRankerSite(false, false, false).e5m2);
    for (const auto& [rows, elements] : f16RankerSites()) {
        for (const bool bf16 : {false, true}) {
            for (const auto& site : {fp8KRValueRankerSite(sum, bf16 ? DataType::BF16 : DataType::FP16),
                                     F16KR::f16KRArgRankerSite(false, true, bf16)}) {
                for (const auto& entry : F16KR::f16KRCandidatePool(rows, 2 * elements, 170, site)) {
                    const auto f = KR::fp8KRRankerFeatures(entry, site, 170);
                    for (float x : f) EXPECT_TRUE(std::isfinite(x));
                    EXPECT_FLOAT_EQ(std::log2(static_cast<double>(rows)), f[0]);
                    EXPECT_FLOAT_EQ(std::log2(static_cast<double>(2 * elements)), f[1]);
                    EXPECT_EQ(static_cast<float>((2 * elements) % 4), f[14]);
                    EXPECT_EQ(static_cast<float>(site.kind), f[17]);
                    EXPECT_EQ(bf16 ? 1.0f : 0.0f, f[20]);
                    EXPECT_EQ(static_cast<float>(entry.list_position), f[22]);
                }
            }
        }
    }
}

TEST(F16KRRanker, ShortlistIsTheListPrefixUntrained) {
    if (F16KR::f16KRRankerTrained()) {
        GTEST_SKIP() << "a trained FP16 / BF16 KR ranker is installed";
    }
    const auto arg_site = F16KR::f16KRArgRankerSite(true, false, false);
    for (const auto& [rows, elements] : f16RankerSites()) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << elements);
        const uint64_t row_bytes = 2 * elements;
        for (const auto& site : {VALUE_SITE, arg_site}) {
            const auto pool = F16KR::f16KRCandidatePool(rows, row_bytes, 170, site);
            for (size_t count : {size_t{1}, size_t{3}, size_t{6}, size_t{100}}) {
                std::vector<KR::Fp8KRSegmentShape> listed;
                for (const auto& entry : pool)
                    if (entry.list_position != 0 && listed.size() < count) listed.push_back(entry.shape);
                EXPECT_EQ(listed, F16KR::f16KRRankedShapes(rows, row_bytes, 170, site, count));
            }
        }
    }
    EXPECT_EQ(6u, F16KR::f16KRShortlistLength(CubReductionTuningDepth::Quick));
    EXPECT_EQ(12u, F16KR::f16KRShortlistLength(CubReductionTuningDepth::Regular));
    EXPECT_EQ(24u, F16KR::f16KRShortlistLength(CubReductionTuningDepth::Ultra));
}

// 1239: the trained ranker's shortlist is a prefix of the ranked pool at F16_KR_SHORTLIST_DEPTHS.
TEST(F16KRRanker, ShortlistIsARankedPoolPrefixTrained) {
    if (!F16KR::f16KRRankerTrained()) {
        GTEST_SKIP() << "no trained FP16 / BF16 KR ranker installed yet";
    }
    EXPECT_EQ(F16KR::F16_KR_SHORTLIST_DEPTHS[0], F16KR::f16KRShortlistLength(CubReductionTuningDepth::Quick));
    EXPECT_EQ(F16KR::F16_KR_SHORTLIST_DEPTHS[1], F16KR::f16KRShortlistLength(CubReductionTuningDepth::Regular));
    EXPECT_EQ(F16KR::F16_KR_SHORTLIST_DEPTHS[2], F16KR::f16KRShortlistLength(CubReductionTuningDepth::Ultra));
    const auto arg_site = F16KR::f16KRArgRankerSite(true, false, true);
    for (const auto& [rows, elements] : f16RankerSites()) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << elements);
        const uint64_t row_bytes = 2 * elements;
        for (const auto& site : {VALUE_SITE, arg_site}) {
            const auto pool = F16KR::f16KRCandidatePool(rows, row_bytes, 170, site);
            const auto all = F16KR::f16KRRankedShapes(rows, row_bytes, 170, site, pool.size());
            ASSERT_EQ(pool.size(), all.size());
            for (size_t count : {size_t{1}, size_t{4}, size_t{8}}) {
                const auto shortlist = F16KR::f16KRRankedShapes(rows, row_bytes, 170, site, count);
                ASSERT_EQ(std::min(count, pool.size()), shortlist.size());
                EXPECT_TRUE(std::equal(shortlist.begin(), shortlist.end(), all.begin()));
                for (const auto& shape : shortlist)
                    EXPECT_TRUE(std::any_of(pool.begin(), pool.end(), [&](const auto& e) { return e.shape == shape; }));
            }
        }
    }
}

TEST(F16KRRanker, ModelMatchesTrainingFixturesExactly) {
    static_assert(F16KR::F16KRRankerModel::FEATURE_COUNT == KR::FP8_KR_RANKER_FEATURES);
    if (!F16KR::F16KRRankerModel::TRAINED) {
        GTEST_SKIP() << "no trained FP16 / BF16 KR ranker installed yet";
    }
    ASSERT_FALSE(F16KR::F16KRRankerParity::fixtures.empty());
    for (size_t i = 0; i < F16KR::F16KRRankerParity::fixtures.size(); ++i) {
        const auto& fixture = F16KR::F16KRRankerParity::fixtures[i];
        const float margin = F16KR::F16KRRankerModel::margin(fixture.features);
        ASSERT_EQ(std::bit_cast<uint32_t>(fixture.margin), std::bit_cast<uint32_t>(margin)) << "fixture " << i;
    }
}
