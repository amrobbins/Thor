// The FP16 / BF16 KR ranker (one ranker for both formats, the BF16 flag in the alternate-format feature): the 2-byte
// pool and its ARG restriction, the features, the ranked shortlist without a trained model, and, once a trained
// F16KRRankerModel.h is installed, its margins against the training pipeline's fixtures, bit for bit. Host-only.

#include "Utilities/TensorOperations/Reduction/KR/F16/F16KRStampAutotuner.h"
#include "Utilities/TensorOperations/Reduction/KR/KRSegments.h"
#include "Utilities/TensorOperations/Reduction/Kernels/KernelLaunchers.h"
#include "Utilities/TensorOperations/Reduction/Kernels/StageSemantics.h"
#include "Utilities/TensorOperations/Reduction/Rankers/F16KRRankerModel.h"
#include "test/Utilities/TensorOperations/Reduction/F16KRRankerParityFixtures.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace KR = ThorImplementation::ReductionInternal::KR;
using ThorImplementation::ReductionTuningDepth;
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

const KR::KRRankerSite VALUE_SITE{0, 0, 0, false};

}  // namespace

TEST(F16KRRanker, PoolIsTheTwoBytePoolAndArgDropsOnlyVectorPack) {
    for (const auto& [rows, elements] : f16RankerSites()) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << elements);
        const uint64_t row_bytes = 2 * elements;
        const auto retained = KR::retainedKRCandidatePool(rows, row_bytes, 170, 2);
        const auto value_pool = KR::F16::f16KRCandidatePool(rows, row_bytes, 170, VALUE_SITE);
        ASSERT_EQ(retained.size(), value_pool.size());
        std::set<std::string> names;
        for (size_t i = 0; i < value_pool.size(); ++i) {
            EXPECT_EQ(retained[i].shape, value_pool[i].shape);
            EXPECT_EQ(retained[i].list_position, value_pool[i].list_position);
            EXPECT_TRUE(KR::retainedKRShapeLegalForElement(value_pool[i].shape, 2));
            EXPECT_TRUE(names.insert(KR::krShapeName(value_pool[i].shape)).second);
        }
        const auto arg_pool =
            KR::F16::f16KRCandidatePool(rows, row_bytes, 170, KR::F16::f16KRArgRankerSite(true, true, true));
        size_t legal = 0;
        for (const auto& entry : retained) legal += entry.shape.vector ? 0 : 1;
        EXPECT_EQ(legal, arg_pool.size());
        for (const auto& entry : arg_pool) EXPECT_TRUE(KR::f16KRArgShapeLegal(entry.shape));
    }
}

TEST(F16KRRanker, FeaturesAreFiniteAndCarryTheBf16Flag) {
    using namespace ThorImplementation::ReductionInternal;
    const ReductionStageSemantics sum =
        makeValueReductionStageSemantics(ThorImplementation::ReductionOp::Sum, DensePassRole::Complete, 8);
    EXPECT_FALSE(krValueRankerSite(sum, DataType::FP16).e5m2);
    EXPECT_TRUE(krValueRankerSite(sum, DataType::BF16).e5m2);
    EXPECT_FALSE(krValueRankerSite(sum, DataType::FP32).e5m2);
    EXPECT_TRUE(KR::F16::f16KRArgRankerSite(false, false, true).e5m2);
    EXPECT_FALSE(KR::F16::f16KRArgRankerSite(false, false, false).e5m2);
    for (const auto& [rows, elements] : f16RankerSites()) {
        for (const bool bf16 : {false, true}) {
            for (const auto& site : {krValueRankerSite(sum, bf16 ? DataType::BF16 : DataType::FP16),
                                     KR::F16::f16KRArgRankerSite(false, true, bf16)}) {
                for (const auto& entry : KR::F16::f16KRCandidatePool(rows, 2 * elements, 170, site)) {
                    const auto f = KR::krRankerFeatures(entry, site, 170);
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
    if (KR::F16::f16KRRankerTrained()) {
        GTEST_SKIP() << "a trained FP16 / BF16 KR ranker is installed";
    }
    const auto arg_site = KR::F16::f16KRArgRankerSite(true, false, false);
    for (const auto& [rows, elements] : f16RankerSites()) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << elements);
        const uint64_t row_bytes = 2 * elements;
        for (const auto& site : {VALUE_SITE, arg_site}) {
            const auto pool = KR::F16::f16KRCandidatePool(rows, row_bytes, 170, site);
            for (size_t count : {size_t{1}, size_t{3}, size_t{6}, size_t{100}}) {
                std::vector<KR::KRSegmentShape> listed;
                for (const auto& entry : pool)
                    if (entry.list_position != 0 && listed.size() < count) listed.push_back(entry.shape);
                EXPECT_EQ(listed, KR::F16::f16KRRankedShapes(rows, row_bytes, 170, site, count));
            }
        }
    }
    EXPECT_EQ(6u, KR::F16::f16KRShortlistLength(ReductionTuningDepth::Quick));
    EXPECT_EQ(12u, KR::F16::f16KRShortlistLength(ReductionTuningDepth::Regular));
    EXPECT_EQ(24u, KR::F16::f16KRShortlistLength(ReductionTuningDepth::Ultra));
}

// The trained ranker's shortlist is a prefix of the ranked pool at F16_KR_SHORTLIST_DEPTHS.
TEST(F16KRRanker, ShortlistIsARankedPoolPrefixTrained) {
    if (!KR::F16::f16KRRankerTrained()) {
        GTEST_SKIP() << "no trained FP16 / BF16 KR ranker installed yet";
    }
    EXPECT_EQ(KR::F16::F16_KR_SHORTLIST_DEPTHS[0], KR::F16::f16KRShortlistLength(ReductionTuningDepth::Quick));
    EXPECT_EQ(KR::F16::F16_KR_SHORTLIST_DEPTHS[1], KR::F16::f16KRShortlistLength(ReductionTuningDepth::Regular));
    EXPECT_EQ(KR::F16::F16_KR_SHORTLIST_DEPTHS[2], KR::F16::f16KRShortlistLength(ReductionTuningDepth::Ultra));
    const auto arg_site = KR::F16::f16KRArgRankerSite(true, false, true);
    for (const auto& [rows, elements] : f16RankerSites()) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << elements);
        const uint64_t row_bytes = 2 * elements;
        for (const auto& site : {VALUE_SITE, arg_site}) {
            const auto pool = KR::F16::f16KRCandidatePool(rows, row_bytes, 170, site);
            const auto all = KR::F16::f16KRRankedShapes(rows, row_bytes, 170, site, pool.size());
            ASSERT_EQ(pool.size(), all.size());
            for (size_t count : {size_t{1}, size_t{4}, size_t{8}}) {
                const auto shortlist = KR::F16::f16KRRankedShapes(rows, row_bytes, 170, site, count);
                ASSERT_EQ(std::min(count, pool.size()), shortlist.size());
                EXPECT_TRUE(std::equal(shortlist.begin(), shortlist.end(), all.begin()));
                for (const auto& shape : shortlist)
                    EXPECT_TRUE(std::any_of(pool.begin(), pool.end(), [&](const auto& e) { return e.shape == shape; }));
            }
        }
    }
}

TEST(F16KRRanker, ModelMatchesTrainingFixturesExactly) {
    static_assert(KR::F16::F16KRRankerModel::FEATURE_COUNT == KR::KR_RANKER_FEATURES);
    if (!KR::F16::F16KRRankerModel::TRAINED) {
        GTEST_SKIP() << "no trained FP16 / BF16 KR ranker installed yet";
    }
    ASSERT_FALSE(KR::F16::F16KRRankerParity::fixtures.empty());
    for (size_t i = 0; i < KR::F16::F16KRRankerParity::fixtures.size(); ++i) {
        const auto& fixture = KR::F16::F16KRRankerParity::fixtures[i];
        const float margin = KR::F16::F16KRRankerModel::margin(fixture.features);
        ASSERT_EQ(std::bit_cast<uint32_t>(fixture.margin), std::bit_cast<uint32_t>(margin)) << "fixture " << i;
    }
}
