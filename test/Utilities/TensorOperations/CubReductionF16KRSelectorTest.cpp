// 1236: the retained FP16 / BF16 R / KR VALUE selector (F16KR::selectCachedValueSite), called directly (not routed
// yet, plans/Thor_FP16_BF16_Dense_Reduction_Plan phase 8). The selection kinds, its domain and geometry (host), the
// shortlist per tuning depth, and on the GPU: the winner is a shortlisted 2-byte shape, the selection is cached under
// kind F16KR (a deeper selection answers a shallower request), FP16 and BF16 at one geometry are separate entries, and
// the winner computes the reduction. 1237: and the ARG selector (F16KR::selectCachedArgSite).

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "test/Utilities/TensorOperations/Fp8ArgReference.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Cub/F16KR/F16KRStampAutotuner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;
using namespace ThorImplementation::CubReductionInternal;
namespace KR = ThorImplementation::CubReductionInternal::Fp8KR;

namespace {

DenseReductionProblem valueProblem(const std::vector<uint64_t>& dims, const std::vector<uint32_t>& axes) {
    DenseReductionOutputSpec output;
    output.produce_value = true;
    output.value_dtype = DataType::FP32;
    return makeInitialDenseReductionProblem(dims, axes, DenseReductionAggregateKind::Value, output);
}

DenseReductionProblem argProblem(const std::vector<uint64_t>& dims, const std::vector<uint32_t>& axes) {
    DenseReductionOutputSpec output;
    output.produce_value = false;
    output.produce_index = true;
    output.index_dtype = DataType::UINT32;
    return makeInitialDenseReductionProblem(dims, axes, DenseReductionAggregateKind::Arg, output);
}

DenseReductionSite lastSite(const DenseReductionProblem& problem) {
    return enumerateDenseReductionSites(problem).back();
}

uint32_t smCount() {
    int sm_count = 0;
    EXPECT_EQ(cudaSuccess, cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    return static_cast<uint32_t>(sm_count);
}

}  // namespace

TEST(CubReductionF16KRSelector, SelectionKindsAreNamed) {
    EXPECT_EQ(std::string(denseReductionSelectionKindName(DenseReductionSelectionKind::F16RK)), "f16_rk");
    EXPECT_EQ(std::string(denseReductionSelectionKindName(DenseReductionSelectionKind::F16KR)), "f16_kr");
    DenseReductionSelectionCache cache;
    DenseReductionSelectionKey fp32;
    fp32.kind = DenseReductionSelectionKind::Fp32KR;
    fp32.site = {1, 2, 3};
    DenseReductionSelectionKey f16 = fp32;
    f16.kind = DenseReductionSelectionKind::F16KR;
    F16KR::F16KRSelection recipe;
    recipe.winner_name = "f16";
    (void)cache.getOrTune<F16KR::F16KRSelection>(f16, [&] { return recipe; });
    EXPECT_FALSE(cache.find<F16KR::F16KRSelection>(fp32).has_value());
    EXPECT_EQ(cache.tuningAttempts(DenseReductionSelectionKind::F16KR), 1U);
    EXPECT_EQ(cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR), 0U);
    EXPECT_EQ(cache.tuningAttempts(DenseReductionSelectionKind::F16RK), 0U);
}

TEST(CubReductionF16KRSelector, DomainGeometryAndShortlist) {
    // [K=64, R=512] over R: a KR site of 64 rows of 1 KiB; [4096] reduced: a pure R site.
    const DenseReductionProblem kr = valueProblem({64, 512}, {1});
    const DenseReductionProblem r = valueProblem({4096}, {0});
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        EXPECT_TRUE(F16KR::isOriginalF16KRSite(kr, lastSite(kr), dtype));
        EXPECT_TRUE(F16KR::isOriginalF16KRSite(r, lastSite(r), dtype));
    }
    for (const DataType dtype : {DataType::FP32, DataType::FP8_E4M3, DataType::FP8_E5M2})
        EXPECT_FALSE(F16KR::isOriginalF16KRSite(kr, lastSite(kr), dtype));
    // [R=512, K=64] over R is an RK site; an intermediate site reads FP32 partials (the FP32 selector's).
    const DenseReductionProblem rk = valueProblem({512, 64}, {0});
    EXPECT_FALSE(F16KR::isOriginalF16KRSite(rk, lastSite(rk), DataType::FP16));
    const DenseReductionProblem staged = advanceDenseReductionProblem(kr, lastSite(kr), 7);
    EXPECT_FALSE(F16KR::isOriginalF16KRSite(staged, lastSite(staged), DataType::FP16));

    const F16KR::F16KRSiteGeometry kr_geometry = F16KR::f16KRSiteGeometry(kr, lastSite(kr));
    EXPECT_EQ(64u, kr_geometry.rows);
    EXPECT_EQ(1024u, kr_geometry.row_bytes);
    const F16KR::F16KRSiteGeometry r_geometry = F16KR::f16KRSiteGeometry(r, lastSite(r));
    EXPECT_EQ(1u, r_geometry.rows);
    EXPECT_EQ(8192u, r_geometry.row_bytes);

    // Untrained: the shortlist is the 2-byte list's prefix at the RK depths, Quick 6, Regular 12, Ultra 24. 1239:
    // trained, a ranked prefix of the pool at F16_KR_SHORTLIST_DEPTHS.
    for (const auto& [rows, elements] : std::vector<std::pair<uint64_t, uint64_t>>{
             {1, 1 << 20}, {64, 512}, {3000, 5}, {512, 16384}, {2, 70001}, {100000, 7}}) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << elements);
        const auto list = KR::retainedKRCandidateShapes(rows, 2 * elements, 170, 2);
        const auto pool = KR::retainedKRCandidatePool(rows, 2 * elements, 170, 2);
        const bool trained = F16KR::f16KRRankerTrained();
        size_t previous = 0;
        for (const auto& [depth, count] : std::vector<std::pair<CubReductionTuningDepth, size_t>>{
                 {CubReductionTuningDepth::Quick, trained ? F16KR::F16_KR_SHORTLIST_DEPTHS[0] : 6},
                 {CubReductionTuningDepth::Regular, trained ? F16KR::F16_KR_SHORTLIST_DEPTHS[1] : 12},
                 {CubReductionTuningDepth::Ultra, trained ? F16KR::F16_KR_SHORTLIST_DEPTHS[2] : 24}}) {
            const auto shortlist = F16KR::f16KRShortlist(rows, 2 * elements, 170, KR::Fp8KRRankerSite{}, depth);
            ASSERT_EQ(std::min(count, trained ? pool.size() : list.size()), shortlist.size());
            if (!trained) EXPECT_TRUE(std::equal(shortlist.begin(), shortlist.end(), list.begin()));
            EXPECT_GE(shortlist.size(), previous);
            previous = shortlist.size();
            for (const auto& shape : shortlist) EXPECT_TRUE(KR::retainedKRShapeLegalForElement(shape, 2));
        }
    }
}

TEST(CubReductionF16KRSelector, RejectsSitesOutsideItsDomain) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const auto select = [&](const DenseReductionProblem& problem, DataType dtype, uint64_t size) {
        return F16KR::selectCachedValueSite(problem, lastSite(problem), dtype, CubReductionOp::Sum, size, 1.0f, 170,
                                            stream, CubReductionTuningDepth::Quick);
    };
    const DenseReductionProblem kr = valueProblem({64, 512}, {1});
    EXPECT_THROW(static_cast<void>(select(argProblem({64, 512}, {1}), DataType::FP16, 512)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(select(valueProblem({512, 64}, {0}), DataType::FP16, 512)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(select(kr, DataType::FP16, 0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(select(kr, DataType::FP32, 512)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(select(kr, DataType::FP8_E4M3, 512)), std::invalid_argument);
    const DenseReductionProblem staged = advanceDenseReductionProblem(kr, lastSite(kr), 7);
    EXPECT_THROW(static_cast<void>(select(staged, DataType::BF16, 512)), std::invalid_argument);
}

TEST(CubReductionF16KRSelector, SelectsCachesAndComputes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint32_t sms = smCount();
    DenseReductionSelectionCache& cache = denseReductionSelectionCache();
    cache.clear();
    std::mt19937 rng(1236);

    struct Site {
        uint64_t rows, elements;
        CubReductionOp op;
    };
    // Short rows (Pack / vector Pack), mid rows (Group / Block), long rows (Block / Split), few long rows (Split).
    const std::vector<Site> sites{{30000, 5, CubReductionOp::Sum},  {60000, 8, CubReductionOp::Max},
                                  {640, 4096, CubReductionOp::Max}, {512, 32768, CubReductionOp::Sum},
                                  {2, 70001, CubReductionOp::L1Norm}, {1, 1 << 20, CubReductionOp::Sum}};
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (const Site& s : sites) {
            SCOPED_TRACE(::testing::Message() << (dtype == DataType::FP16 ? "fp16" : "bf16") << " rows=" << s.rows
                                              << " R=" << s.elements << " op=" << static_cast<int>(s.op));
            const std::vector<uint64_t> dims =
                s.rows == 1 ? std::vector<uint64_t>{s.elements} : std::vector<uint64_t>{s.rows, s.elements};
            const DenseReductionProblem problem = valueProblem(dims, {static_cast<uint32_t>(dims.size() - 1)});
            const DenseReductionSite site = lastSite(problem);

            const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::F16KR);
            const F16KR::F16KRSelection regular = F16KR::selectCachedValueSite(
                problem, site, dtype, s.op, s.elements, 1.0f, sms, stream, CubReductionTuningDepth::Regular);
            ASSERT_TRUE(regular.shape);
            EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::F16KR));
            const auto semantics = makeValueReductionStageSemantics(s.op, DensePassRole::Complete, s.elements);
            const auto shortlist = F16KR::f16KRShortlist(s.rows, 2 * s.elements, sms,
                                                         fp8KRValueRankerSite(semantics, dtype),
                                                         CubReductionTuningDepth::Regular);
            EXPECT_EQ(shortlist.size(), regular.candidate_count);
            EXPECT_NE(shortlist.end(), std::find(shortlist.begin(), shortlist.end(), *regular.shape));
            EXPECT_EQ("f16_kr:" + KR::fp8KRShapeName(*regular.shape), regular.winner_name);

            // Cached: the same request and a shallower one tune nothing and return the same shape.
            const auto again = F16KR::selectCachedValueSite(problem, site, dtype, s.op, s.elements, 1.0f, sms, stream,
                                                            CubReductionTuningDepth::Regular);
            const auto quick = F16KR::selectCachedValueSite(problem, site, dtype, s.op, s.elements, 1.0f, sms, stream,
                                                            CubReductionTuningDepth::Quick);
            EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::F16KR));
            ASSERT_TRUE(again.shape && quick.shape);
            EXPECT_EQ(*regular.shape, *again.shape);
            EXPECT_EQ(*regular.shape, *quick.shape);

            // The winner computes the reduction (dyadic inputs, exact in both formats: SUM, |x| sum and MAX exact).
            std::vector<float> values(s.rows * s.elements);
            for (float& v : values) v = static_cast<float>(static_cast<int>(rng() % 65) - 32) / 4.0f;
            const Tensor input = makeGpuTensor(values, {s.rows, s.elements}, stream, dtype);
            Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {s.rows}));
            std::optional<Tensor> scratch;
            const uint64_t scratch_bytes = KR::fp8KRScratchBytes(*regular.shape, false);
            if (scratch_bytes != 0) scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
            launchF16KRValueSegments(semantics, input, output, *regular.shape,
                                     scratch ? scratch->getMemPtr<void>() : nullptr, scratch_bytes, 1.0f, stream);
            stream.synchronize();
            const std::vector<float> got = copyGpuTensorAsFloat(output, stream);
            for (uint64_t row = 0; row < s.rows; ++row) {
                double expected = s.op == CubReductionOp::Max ? -1e30 : 0.0;
                for (uint64_t i = 0; i < s.elements; ++i) {
                    const double x = values[row * s.elements + i];
                    expected = s.op == CubReductionOp::Max     ? std::max(expected, x)
                               : s.op == CubReductionOp::L1Norm ? expected + std::fabs(x)
                                                               : expected + x;
                }
                ASSERT_EQ(expected, static_cast<double>(got[row])) << "row " << row;
            }
        }
    }

    // FP16 and BF16 at one geometry and operation are separate entries; each is then a hit.
    const DenseReductionProblem problem = valueProblem({256, 1000}, {1});
    const DenseReductionSite site = lastSite(problem);
    uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::F16KR);
    for (const DataType dtype : {DataType::FP16, DataType::BF16, DataType::FP16, DataType::BF16})
        static_cast<void>(F16KR::selectCachedValueSite(problem, site, dtype, CubReductionOp::Sum, 1000, 1.0f, sms,
                                                       stream, CubReductionTuningDepth::Quick));
    EXPECT_EQ(before + 2, cache.tuningAttempts(DenseReductionSelectionKind::F16KR));

    // A deeper selection answers a shallower request; a shallower one does not answer a deeper request.
    before = cache.tuningAttempts(DenseReductionSelectionKind::F16KR);
    const auto ultra = F16KR::selectCachedValueSite(problem, site, DataType::FP16, CubReductionOp::Sum, 1000, 1.0f,
                                                    sms, stream, CubReductionTuningDepth::Ultra);
    EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::F16KR));
    const auto regular = F16KR::selectCachedValueSite(problem, site, DataType::FP16, CubReductionOp::Sum, 1000, 1.0f,
                                                      sms, stream, CubReductionTuningDepth::Regular);
    EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::F16KR));
    ASSERT_TRUE(ultra.shape && regular.shape);
    EXPECT_EQ(*ultra.shape, *regular.shape);
    // Another operation is another site; so is the FP32 selector's kind (untouched here).
    const uint64_t fp32_before = cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR);
    static_cast<void>(F16KR::selectCachedValueSite(problem, site, DataType::FP16, CubReductionOp::Max, 1000, 1.0f, sms,
                                                   stream, CubReductionTuningDepth::Quick));
    EXPECT_EQ(before + 2, cache.tuningAttempts(DenseReductionSelectionKind::F16KR));
    EXPECT_EQ(fp32_before, cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR));
}

TEST(CubReductionF16KRSelector, ArgSelectsCachesAndComputes) {
    REQUIRE_CUDA_DEVICE();
    namespace Reference = ThorImplementation::Fp8ArgReference;
    Stream stream(0);
    const uint32_t sms = smCount();
    DenseReductionSelectionCache& cache = denseReductionSelectionCache();
    std::mt19937 rng(1237);

    // VALUE problems, FP32 input and intermediate sites are not its sites.
    const DenseReductionProblem value = valueProblem({64, 512}, {1});
    EXPECT_THROW(static_cast<void>(F16KR::selectCachedArgSite(value, lastSite(value), DataType::FP16, false, sms,
                                                              stream, CubReductionTuningDepth::Quick)),
                 std::invalid_argument);
    const DenseReductionProblem arg = argProblem({64, 512}, {1});
    EXPECT_THROW(static_cast<void>(F16KR::selectCachedArgSite(arg, lastSite(arg), DataType::FP32, false, sms, stream,
                                                              CubReductionTuningDepth::Quick)),
                 std::invalid_argument);

    struct Site {
        uint64_t rows, elements;
        bool maximum;
    };
    const std::vector<Site> sites{{30000, 5, false}, {60000, 8, true}, {640, 4096, true}, {512, 32768, false},
                                  {2, 70001, true}, {1, 1 << 20, false}};
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (const Site& s : sites) {
            SCOPED_TRACE(::testing::Message() << (dtype == DataType::FP16 ? "fp16" : "bf16") << " rows=" << s.rows
                                              << " R=" << s.elements << " max=" << s.maximum);
            const std::vector<uint64_t> dims =
                s.rows == 1 ? std::vector<uint64_t>{s.elements} : std::vector<uint64_t>{s.rows, s.elements};
            const DenseReductionProblem problem = argProblem(dims, {static_cast<uint32_t>(dims.size() - 1)});
            const DenseReductionSite site = lastSite(problem);
            const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::F16KR);
            const auto regular = F16KR::selectCachedArgSite(problem, site, dtype, s.maximum, sms, stream,
                                                            CubReductionTuningDepth::Regular);
            ASSERT_TRUE(regular.shape);
            EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::F16KR));
            const auto shortlist = F16KR::f16KRShortlist(
                s.rows, 2 * s.elements, sms, F16KR::f16KRArgRankerSite(s.maximum, false, dtype == DataType::BF16),
                CubReductionTuningDepth::Regular);
            EXPECT_EQ(shortlist.size(), regular.candidate_count);
            EXPECT_NE(shortlist.end(), std::find(shortlist.begin(), shortlist.end(), *regular.shape));
            EXPECT_TRUE(KR::f16KRArgShapeLegal(*regular.shape));
            EXPECT_EQ("f16_arg_kr:" + KR::fp8KRShapeName(*regular.shape), regular.winner_name);
            const auto quick = F16KR::selectCachedArgSite(problem, site, dtype, s.maximum, sms, stream,
                                                          CubReductionTuningDepth::Quick);
            EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::F16KR));
            ASSERT_TRUE(quick.shape);
            EXPECT_EQ(*regular.shape, *quick.shape);
            // The other operation is another site.
            static_cast<void>(F16KR::selectCachedArgSite(problem, site, dtype, !s.maximum, sms, stream,
                                                         CubReductionTuningDepth::Quick));
            EXPECT_EQ(before + 2, cache.tuningAttempts(DenseReductionSelectionKind::F16KR));

            // The winner computes the reduction (ties included: values from a small set, exact in both formats).
            std::vector<float> values(s.rows * s.elements);
            for (float& v : values) v = static_cast<float>(static_cast<int>(rng() % 9) - 4);
            const Tensor input = makeGpuTensor(values, {s.rows, s.elements}, stream, dtype);
            Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {s.rows}));
            std::optional<Tensor> scratch;
            const uint64_t scratch_bytes = KR::fp8KRScratchBytes(*regular.shape, true);
            if (scratch_bytes != 0) scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
            launchF16KRArgSegments(input, nullptr, &index, *regular.shape,
                                   scratch ? scratch->getMemPtr<void>() : nullptr, scratch_bytes, s.maximum, 1,
                                   stream);
            stream.synchronize();
            const std::vector<uint64_t> got = copyGpuTensorAsUnsigned(index, stream);
            for (uint64_t row = 0; row < s.rows; ++row) {
                Reference::Candidate best{0, values[row * s.elements]};
                for (uint64_t i = 1; i < s.elements; ++i)
                    best = Reference::combine(best, {i, values[row * s.elements + i]}, s.maximum);
                ASSERT_EQ(best.index, got[row]) << "row " << row;
            }
        }
    }
    // FP16 and BF16 ARG at one geometry and operation are separate entries, apart from VALUE at the same geometry.
    const DenseReductionProblem problem = argProblem({256, 1000}, {1});
    const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::F16KR);
    for (const DataType dtype : {DataType::FP16, DataType::BF16, DataType::FP16, DataType::BF16})
        static_cast<void>(F16KR::selectCachedArgSite(problem, lastSite(problem), dtype, true, sms, stream,
                                                     CubReductionTuningDepth::Quick));
    EXPECT_EQ(before + 2, cache.tuningAttempts(DenseReductionSelectionKind::F16KR));
}
