// The FP32 R / KR selectors, called directly (production routes every FP32 R / KR site to them). The VALUE selector
// (KR::Fp32::selectCachedValueSite): its domain and geometry (host), the shortlist per tuning depth, and on the GPU:
// the winner is a shortlisted FP32 shape, the selection is cached under kind Fp32KR (a deeper selection answers a
// shallower request), and the winner computes the reduction. The ARG selector (KR::Fp32::selectCachedArgSite) and the
// carried-index ARG selector (KR::Fp32::selectCachedCarriedArgSite / autotuneCarriedArgSite).

#include "test/Utilities/TensorOperations/OutputSpan.h"
#include "test/Utilities/TensorOperations/Reduction/ReductionTestSupport.h"
#include "test/Utilities/TensorOperations/Reduction/Fp8ArgReference.h"
#include "Utilities/TensorOperations/Reduction/Kernels/KernelLaunchers.h"
#include "Utilities/TensorOperations/Reduction/Kernels/StageSemantics.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Reduction/KR/Fp32/Fp32KRStampAutotuner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::ReductionTestSupport;
using namespace ThorImplementation::ReductionInternal;
namespace KR = ThorImplementation::ReductionInternal::KR;

namespace {

DenseReductionProblem fp32ValueProblem(const std::vector<uint64_t>& dims, const std::vector<uint32_t>& axes) {
    DenseReductionOutputSpec output;
    output.produce_value = true;
    output.value_dtype = DataType::FP32;
    return makeInitialDenseReductionProblem(dims, axes, DenseReductionAggregateKind::Value, output);
}

DenseReductionProblem fp32ArgProblem(const std::vector<uint64_t>& dims, const std::vector<uint32_t>& axes) {
    DenseReductionOutputSpec output;
    output.produce_value = false;
    output.produce_index = true;
    output.index_dtype = DataType::UINT32;
    return makeInitialDenseReductionProblem(dims, axes, DenseReductionAggregateKind::Arg, output);
}

DenseReductionSite lastSite(const DenseReductionProblem& problem) {
    return enumerateDenseReductionSites(problem).back();
}

}  // namespace

TEST(ReducerFp32KRSelector, DomainGeometryAndShortlist) {
    // [K=64, R=512] over R: a KR site of 64 rows of 2 KiB; [4096] reduced: a pure R site.
    const DenseReductionProblem kr = fp32ValueProblem({64, 512}, {1});
    const DenseReductionProblem r = fp32ValueProblem({4096}, {0});
    EXPECT_TRUE(KR::Fp32::isOriginalFp32KRSite(kr, lastSite(kr), DataType::FP32));
    EXPECT_TRUE(KR::Fp32::isOriginalFp32KRSite(r, lastSite(r), DataType::FP32));
    EXPECT_FALSE(KR::Fp32::isOriginalFp32KRSite(kr, lastSite(kr), DataType::FP8_E4M3));
    EXPECT_FALSE(KR::Fp32::isOriginalFp32KRSite(kr, lastSite(kr), DataType::FP16));
    // [R=512, K=64] over R is an RK site, not the KR selector's.
    const DenseReductionProblem rk = fp32ValueProblem({512, 64}, {0});
    EXPECT_FALSE(KR::Fp32::isOriginalFp32KRSite(rk, lastSite(rk), DataType::FP32));

    const KR::Fp32::Fp32KRSiteGeometry kr_geometry = KR::Fp32::fp32KRSiteGeometry(kr, lastSite(kr));
    EXPECT_EQ(64u, kr_geometry.rows);
    EXPECT_EQ(2048u, kr_geometry.row_bytes);
    const KR::Fp32::Fp32KRSiteGeometry r_geometry = KR::Fp32::fp32KRSiteGeometry(r, lastSite(r));
    EXPECT_EQ(1u, r_geometry.rows);
    EXPECT_EQ(16384u, r_geometry.row_bytes);

    // The shortlist is the FP32 list's prefix: Quick 6, Regular 12, Ultra 24 (or the whole list when shorter).
    for (const auto& [rows, elements] : std::vector<std::pair<uint64_t, uint64_t>>{
             {1, 1 << 20}, {64, 512}, {3000, 5}, {512, 16384}, {2, 70001}, {100000, 64}}) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << elements);
        const auto list = KR::retainedKRCandidateShapes(rows, 4 * elements, 170, 4);
        size_t previous = 0;
        for (const auto& [depth, count] : std::vector<std::pair<ReductionTuningDepth, size_t>>{
                 {ReductionTuningDepth::Quick, 6}, {ReductionTuningDepth::Regular, 12},
                 {ReductionTuningDepth::Ultra, 24}}) {
            const auto shortlist = KR::Fp32::fp32KRShortlist(rows, 4 * elements, 170, KR::KRRankerSite{}, depth);
            if (!KR::Fp32::fp32KRRankerTrained()) {
                ASSERT_EQ(std::min(count, list.size()), shortlist.size());
                EXPECT_TRUE(std::equal(shortlist.begin(), shortlist.end(), list.begin()));
            }
            EXPECT_GE(shortlist.size(), previous);
            previous = shortlist.size();
            for (const auto& shape : shortlist) EXPECT_TRUE(KR::retainedKRShapeLegalForElement(shape, 4));
        }
    }
}

// Intermediate FP32 VALUE R / KR sites: the domain predicate, and on the GPU the selector at a final and at a non-final
// intermediate site, cached apart from the original site of the same geometry, its winner reducing the partials with
// the pass role's semantics (no input transform; the finalizer on the last pass only).
TEST(ReducerFp32KRSelector, IntermediateValueSitesSelectAndCompute) {
    // [512 rows, 4096] over the row, staged to 7 partials per row: a final intermediate KR site.
    const DenseReductionProblem original = fp32ValueProblem({512, 4096}, {1});
    const DenseReductionProblem final_intermediate = advanceDenseReductionProblem(original, lastSite(original), 7);
    ASSERT_EQ(DenseReductionInputState::IntermediateAggregate, final_intermediate.input_state);
    EXPECT_TRUE(KR::Fp32::isFp32KRValueSite(final_intermediate, lastSite(final_intermediate), DataType::FP8_E4M3));
    EXPECT_FALSE(KR::Fp32::isOriginalFp32KRSite(final_intermediate, lastSite(final_intermediate), DataType::FP32));
    EXPECT_TRUE(KR::Fp32::isFp32KRValueSite(original, lastSite(original), DataType::FP32));
    EXPECT_FALSE(KR::Fp32::isFp32KRValueSite(original, lastSite(original), DataType::FP8_E4M3));
    // [12, 40, 3000] over axes 0 and 2, the last axis staged to 9: its partials [12, 40, 9] are a non-final
    // intermediate KR site (axis 0 remains).
    const DenseReductionProblem two_axes = fp32ValueProblem({12, 40, 3000}, {0, 2});
    const DenseReductionProblem middle =
        advanceDenseReductionProblem(two_axes, enumerateDenseReductionSites(two_axes).back(), 9);
    const DenseReductionSite middle_site = enumerateDenseReductionSites(middle).back();
    ASSERT_EQ(DensePassRole::Intermediate,
              deriveDensePassRole(middle, advanceDenseReductionProblem(middle, middle_site, 1)));
    EXPECT_TRUE(KR::Fp32::isFp32KRValueSite(middle, middle_site, DataType::FP32));

    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    int sm_count = 0;
    ASSERT_EQ(cudaSuccess, cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    const uint32_t sms = static_cast<uint32_t>(sm_count);
    DenseReductionSelectionCache& cache = denseReductionSelectionCache();
    cache.clear();
    std::mt19937 rng(1207);
    struct Case {
        const DenseReductionProblem* problem;
        DenseReductionSite site;
        ReductionOp op;
        uint64_t original_size;
    };
    const std::vector<Case> cases{{&final_intermediate, lastSite(final_intermediate), ReductionOp::L2Norm, 4096},
                                  {&final_intermediate, lastSite(final_intermediate), ReductionOp::Mean, 4096},
                                  {&final_intermediate, lastSite(final_intermediate), ReductionOp::Max, 4096},
                                  {&middle, middle_site, ReductionOp::L1Norm, 12 * 3000}};
    for (const Case& c : cases) {
        SCOPED_TRACE(::testing::Message() << "op=" << static_cast<int>(c.op));
        const DenseReductionProblem& problem = *c.problem;
        const DenseReductionProblem next = advanceDenseReductionProblem(problem, c.site, 1);
        const ReductionStageSemantics semantics =
            makeValueReductionStageSemantics(c.op, deriveDensePassRole(problem, next), c.original_size);
        EXPECT_EQ(ReductionStageInputTransform::Identity, semantics.input_transform);
        const KR::Fp32::Fp32KRSiteGeometry geometry = KR::Fp32::fp32KRSiteGeometry(problem, c.site);
        const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR);
        const auto selected = KR::Fp32::selectCachedValueSite(problem, c.site, c.op, c.original_size, 2.0f, sms, stream,
                                                              ReductionTuningDepth::Regular);
        ASSERT_TRUE(selected.shape);
        EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR));
        const auto shortlist = KR::Fp32::fp32KRShortlist(geometry.rows, geometry.row_bytes, sms,
                                                         krValueRankerSite(semantics, DataType::FP32),
                                                         ReductionTuningDepth::Regular);
        EXPECT_NE(shortlist.end(), std::find(shortlist.begin(), shortlist.end(), *selected.shape));

        // The winner reduces the partials (signed: an intermediate L1Norm sums them as they are; L2Norm's partials are
        // sums of squares, so nonnegative).
        const uint64_t elements = geometry.row_bytes / 4;
        std::vector<float> values(geometry.rows * elements);
        for (float& v : values) {
            v = static_cast<float>(static_cast<int>(rng() % 65) - 32) / 4.0f;
            if (c.op == ReductionOp::L2Norm) v = std::fabs(v);
        }
        const Tensor input = makeGpuTensor(values, {geometry.rows, elements}, stream);
        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {geometry.rows}));
        std::optional<Tensor> scratch;
        const uint64_t scratch_bytes = KR::krScratchBytes(*selected.shape, false);
        if (scratch_bytes != 0) scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
        const bool final_pass = !hasDenseReductionSites(next);
        const float scale = final_pass ? 2.0f : 1.0f;
        launchFp32KRValueSegments(semantics, input, output, *selected.shape,
                                  scratch ? scratch->getMemPtr<void>() : nullptr, scratch_bytes, scale, stream);
        stream.synchronize();
        const std::vector<float> got = copyGpuTensorAsFloat(output, stream);
        for (uint64_t row = 0; row < geometry.rows; ++row) {
            double reduced = c.op == ReductionOp::Max ? -1e30 : 0.0;
            for (uint64_t i = 0; i < elements; ++i) {
                const double x = values[row * elements + i];
                reduced = c.op == ReductionOp::Max ? std::max(reduced, x) : reduced + x;
            }
            if (c.op == ReductionOp::Mean) reduced /= static_cast<double>(c.original_size);
            if (c.op == ReductionOp::L2Norm) reduced = std::sqrt(reduced);
            const double expected = scale * reduced;
            ASSERT_NEAR(expected, static_cast<double>(got[row]), 1e-6 * std::fabs(expected) + 1e-6) << "row " << row;
        }
    }
    // The original site of the final intermediate's geometry ([512, 7] original input) is a separate selection.
    const DenseReductionProblem same_geometry = fp32ValueProblem({512, 7}, {1});
    const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR);
    static_cast<void>(KR::Fp32::selectCachedValueSite(final_intermediate, lastSite(final_intermediate),
                                                      ReductionOp::Sum, 4096, 1.0f, sms, stream,
                                                      ReductionTuningDepth::Quick));
    static_cast<void>(KR::Fp32::selectCachedValueSite(same_geometry, lastSite(same_geometry), ReductionOp::Sum, 7,
                                                      1.0f, sms, stream, ReductionTuningDepth::Quick));
    EXPECT_EQ(before + 2, cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR));
}

TEST(ReducerFp32KRSelector, RejectsSitesOutsideItsDomain) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const auto select = [&](const DenseReductionProblem& problem, uint64_t size) {
        return KR::Fp32::selectCachedValueSite(problem, lastSite(problem), ReductionOp::Sum, size, 1.0f, 170, stream,
                                               ReductionTuningDepth::Quick);
    };
    EXPECT_THROW(static_cast<void>(select(fp32ArgProblem({64, 512}, {1}), 512)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(select(fp32ValueProblem({512, 64}, {0}), 512)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(select(fp32ValueProblem({64, 512}, {1}), 0)), std::invalid_argument);
}

TEST(ReducerFp32KRSelector, SelectsCachesAndComputes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    int sm_count = 0;
    ASSERT_EQ(cudaSuccess, cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    const uint32_t sms = static_cast<uint32_t>(sm_count);
    DenseReductionSelectionCache& cache = denseReductionSelectionCache();
    cache.clear();
    std::mt19937 rng(1180);

    struct Site {
        uint64_t rows, elements;
        ReductionOp op;
    };
    // Short rows (Pack), mid rows (Group / Block), long rows of 64 KiB (Block second), few long rows (Split).
    const std::vector<Site> sites{{30000, 5, ReductionOp::Sum},  {640, 4096, ReductionOp::Max},
                                  {512, 16384, ReductionOp::Sum}, {2, 70001, ReductionOp::L1Norm},
                                  {1, 1 << 20, ReductionOp::Sum}};
    for (const Site& s : sites) {
        SCOPED_TRACE(::testing::Message() << "rows=" << s.rows << " R=" << s.elements << " op="
                                          << static_cast<int>(s.op));
        const std::vector<uint64_t> dims =
            s.rows == 1 ? std::vector<uint64_t>{s.elements} : std::vector<uint64_t>{s.rows, s.elements};
        const DenseReductionProblem problem = fp32ValueProblem(dims, {static_cast<uint32_t>(dims.size() - 1)});
        const DenseReductionSite site = lastSite(problem);

        const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR);
        const KR::Fp32::Fp32KRSelection regular = KR::Fp32::selectCachedValueSite(
            problem, site, s.op, s.elements, 1.0f, sms, stream, ReductionTuningDepth::Regular);
        ASSERT_TRUE(regular.shape);
        EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR));
        const auto shortlist = KR::Fp32::fp32KRShortlist(
            s.rows, 4 * s.elements, sms,
            krValueRankerSite(makeValueReductionStageSemantics(s.op, DensePassRole::Complete, s.elements),
                              DataType::FP32),
            ReductionTuningDepth::Regular);
        EXPECT_EQ(shortlist.size(), regular.candidate_count);
        EXPECT_NE(shortlist.end(), std::find(shortlist.begin(), shortlist.end(), *regular.shape));
        EXPECT_EQ("fp32_kr:" + KR::krShapeName(*regular.shape), regular.winner_name);

        // Cached: the same request and a shallower one tune nothing and return the same shape.
        const auto again = KR::Fp32::selectCachedValueSite(problem, site, s.op, s.elements, 1.0f, sms, stream,
                                                           ReductionTuningDepth::Regular);
        const auto quick = KR::Fp32::selectCachedValueSite(problem, site, s.op, s.elements, 1.0f, sms, stream,
                                                           ReductionTuningDepth::Quick);
        EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR));
        ASSERT_TRUE(again.shape && quick.shape);
        EXPECT_EQ(*regular.shape, *again.shape);
        EXPECT_EQ(*regular.shape, *quick.shape);

        // The winner computes the reduction (dyadic inputs: SUM, |x| sum and MAX are exact).
        std::vector<float> values(s.rows * s.elements);
        for (float& v : values) v = static_cast<float>(static_cast<int>(rng() % 65) - 32) / 4.0f;
        const Tensor input = makeGpuTensor(values, {s.rows, s.elements}, stream);
        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {s.rows}));
        std::optional<Tensor> scratch;
        const uint64_t scratch_bytes = KR::krScratchBytes(*regular.shape, false);
        if (scratch_bytes != 0) scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
        launchFp32KRValueSegments(
            makeValueReductionStageSemantics(s.op, DensePassRole::Complete, s.elements), input, output,
            *regular.shape, scratch ? scratch->getMemPtr<void>() : nullptr, scratch_bytes, 1.0f, stream);
        stream.synchronize();
        const std::vector<float> got = copyGpuTensorAsFloat(output, stream);
        for (uint64_t row = 0; row < s.rows; ++row) {
            double expected = s.op == ReductionOp::Max ? -1e30 : 0.0;
            for (uint64_t i = 0; i < s.elements; ++i) {
                const double x = values[row * s.elements + i];
                expected = s.op == ReductionOp::Max     ? std::max(expected, x)
                           : s.op == ReductionOp::L1Norm ? expected + std::fabs(x)
                                                           : expected + x;
            }
            ASSERT_EQ(expected, static_cast<double>(got[row])) << "row " << row;
        }
    }

    // A deeper selection answers a shallower request; a shallower one does not answer a deeper request.
    const DenseReductionProblem problem = fp32ValueProblem({256, 1000}, {1});
    const DenseReductionSite site = lastSite(problem);
    const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR);
    const auto quick = KR::Fp32::selectCachedValueSite(problem, site, ReductionOp::Sum, 1000, 1.0f, sms, stream,
                                                       ReductionTuningDepth::Quick);
    const auto ultra = KR::Fp32::selectCachedValueSite(problem, site, ReductionOp::Sum, 1000, 1.0f, sms, stream,
                                                       ReductionTuningDepth::Ultra);
    EXPECT_EQ(before + 2, cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR));
    const auto regular = KR::Fp32::selectCachedValueSite(problem, site, ReductionOp::Sum, 1000, 1.0f, sms, stream,
                                                         ReductionTuningDepth::Regular);
    EXPECT_EQ(before + 2, cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR));
    ASSERT_TRUE(quick.shape && ultra.shape && regular.shape);
    EXPECT_EQ(*ultra.shape, *regular.shape);
    EXPECT_LE(quick.candidate_count, ultra.candidate_count);
    // Another operation is another site.
    static_cast<void>(KR::Fp32::selectCachedValueSite(problem, site, ReductionOp::Max, 1000, 1.0f, sms, stream,
                                                      ReductionTuningDepth::Quick));
    EXPECT_EQ(before + 3, cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR));
}

TEST(ReducerFp32KRSelector, ArgSelectsCachesAndComputes) {
    REQUIRE_CUDA_DEVICE();
    namespace Reference = ThorImplementation::Fp8ArgReference;
    Stream stream(0);
    int sm_count = 0;
    ASSERT_EQ(cudaSuccess, cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    const uint32_t sms = static_cast<uint32_t>(sm_count);
    DenseReductionSelectionCache& cache = denseReductionSelectionCache();
    std::mt19937 rng(1181);

    // VALUE problems are not ARG sites.
    EXPECT_THROW(static_cast<void>(KR::Fp32::selectCachedArgSite(fp32ValueProblem({64, 512}, {1}),
                                                                 lastSite(fp32ValueProblem({64, 512}, {1})), false, sms,
                                                                 stream, ReductionTuningDepth::Quick)),
                 std::invalid_argument);

    struct Site {
        uint64_t rows, elements;
        bool maximum;
    };
    const std::vector<Site> sites{{30000, 5, false}, {640, 4096, true}, {512, 16384, false}, {2, 70001, true},
                                  {1, 1 << 20, false}};
    for (const Site& s : sites) {
        SCOPED_TRACE(::testing::Message() << "rows=" << s.rows << " R=" << s.elements << " max=" << s.maximum);
        const std::vector<uint64_t> dims =
            s.rows == 1 ? std::vector<uint64_t>{s.elements} : std::vector<uint64_t>{s.rows, s.elements};
        const DenseReductionProblem problem = fp32ArgProblem(dims, {static_cast<uint32_t>(dims.size() - 1)});
        const DenseReductionSite site = lastSite(problem);
        const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR);
        const auto regular =
            KR::Fp32::selectCachedArgSite(problem, site, s.maximum, sms, stream, ReductionTuningDepth::Regular);
        ASSERT_TRUE(regular.shape);
        EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR));
        const auto shortlist = KR::Fp32::fp32KRShortlist(s.rows, 4 * s.elements, sms,
                                                         KR::Fp32::fp32KRArgRankerSite(s.maximum, false),
                                                         ReductionTuningDepth::Regular);
        EXPECT_EQ(shortlist.size(), regular.candidate_count);
        EXPECT_NE(shortlist.end(), std::find(shortlist.begin(), shortlist.end(), *regular.shape));
        EXPECT_EQ("fp32_arg_kr:" + KR::krShapeName(*regular.shape), regular.winner_name);
        const auto quick =
            KR::Fp32::selectCachedArgSite(problem, site, s.maximum, sms, stream, ReductionTuningDepth::Quick);
        EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR));
        ASSERT_TRUE(quick.shape);
        EXPECT_EQ(*regular.shape, *quick.shape);
        // The other operation is another site.
        static_cast<void>(
            KR::Fp32::selectCachedArgSite(problem, site, !s.maximum, sms, stream, ReductionTuningDepth::Quick));
        EXPECT_EQ(before + 2, cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR));

        // The winner computes the reduction (ties included: values from a small set).
        std::vector<float> values(s.rows * s.elements);
        for (float& v : values) v = static_cast<float>(static_cast<int>(rng() % 9) - 4);
        const Tensor input = makeGpuTensor(values, {s.rows, s.elements}, stream);
        Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {s.rows}));
        std::optional<Tensor> scratch;
        const uint64_t scratch_bytes = KR::krScratchBytes(*regular.shape, true);
        if (scratch_bytes != 0) scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
        launchFp32KRArgSegments(input, nullptr, outputSpan(index), *regular.shape, scratch ? scratch->getMemPtr<void>() : nullptr,
                                scratch_bytes, s.maximum, 1, stream);
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

namespace {

// An ARG problem over `axes` whose index payload is `index_dtype` (UINT64 index output, a value output).
DenseReductionProblem fp32CarriedArgProblem(const std::vector<uint64_t>& dims, const std::vector<uint32_t>& axes,
                                            DataType index_dtype) {
    DenseReductionOutputSpec output;
    output.produce_value = true;
    output.value_dtype = DataType::FP32;
    output.produce_index = true;
    output.index_dtype = DataType::UINT64;
    return makeInitialDenseReductionProblem(dims, axes, DenseReductionAggregateKind::Arg, output, index_dtype);
}

// Carried indices giving distinct composed indices per row: at stride 0 distinct values (above 2^32 for UINT64),
// otherwise values below the stride.
std::vector<uint64_t> carriedIndicesFor(uint64_t count, uint64_t stride, DataType index_dtype, std::mt19937& rng) {
    std::vector<uint64_t> carried(count);
    for (uint64_t i = 0; i < count; ++i) {
        carried[i] = stride == 0 ? ((i * 0x9E3779B1ULL + 777) & 0xFFFFFFFFULL) +
                                       (index_dtype == DataType::UINT64 ? uint64_t{1} << 40 : 0)
                                 : rng() % stride;
    }
    return carried;
}

}  // namespace

// The carried-index ARG R / KR selector: its domain; at a final site over an intermediate run (stride 0) and a
// non-final site over an original run (stride 1), with UINT32 and UINT64 carried indices: a shortlisted winner (the ARG
// pool ranked with the carried ranker site), cached under kind Fp32KR apart from the original ARG site of the same
// geometry and from the other operation, and the winner computing the reduction. Then the explicit-input autotune and
// its input checks.
TEST(ReducerFp32KRSelector, CarriedArgSitesSelectCacheAndCompute) {
    REQUIRE_CUDA_DEVICE();
    namespace Reference = ThorImplementation::Fp8ArgReference;
    Stream stream(0);
    int sm_count = 0;
    ASSERT_EQ(cudaSuccess, cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    const uint32_t sms = static_cast<uint32_t>(sm_count);
    DenseReductionSelectionCache& cache = denseReductionSelectionCache();
    cache.clear();
    std::mt19937 rng(1215);
    EXPECT_EQ(3, KR::Fp32::fp32KRArgRankerSite(false, false, true).kind);
    EXPECT_EQ(4, KR::Fp32::fp32KRArgRankerSite(true, true, true).kind);
    for (const DataType index_dtype : {DataType::UINT32, DataType::UINT64}) {
        // [640, 40000] over R, staged to 32 partials: [640, 32] is a final carried KR site (an intermediate run).
        const DenseReductionProblem staged = fp32CarriedArgProblem({640, 40000}, {1}, index_dtype);
        const DenseReductionProblem final_carried = advanceDenseReductionProblem(staged, lastSite(staged), 32);
        // [50, 64, 4096] over axes 0 and 2, the first staged to 3: R = 4096 of [3, 64, 4096] is a non-final carried KR
        // site over an original run (192 rows).
        const DenseReductionProblem two_axes = fp32CarriedArgProblem({50, 64, 4096}, {0, 2}, index_dtype);
        const DenseReductionProblem middle =
            advanceDenseReductionProblem(two_axes, enumerateDenseReductionSites(two_axes).front(), 3);
        const DenseReductionSite middle_site = lastSite(middle);
        ASSERT_EQ(DenseReducerFamily::KR, classifyDenseReducerFamily(final_carried, lastSite(final_carried)));
        ASSERT_EQ(DenseReducerFamily::KR, classifyDenseReducerFamily(middle, middle_site));
        ASSERT_EQ(DenseReductionRunKind::OriginalReduction, middle.runs[middle_site.run_index].kind);
        EXPECT_TRUE(KR::Fp32::isCarriedFp32KRArgSite(final_carried, lastSite(final_carried)));
        EXPECT_TRUE(KR::Fp32::isCarriedFp32KRArgSite(middle, middle_site));
        EXPECT_FALSE(KR::Fp32::isCarriedFp32KRArgSite(staged, lastSite(staged)));
        // Outside the domain: an original ARG site for the carried selector, a carried site for the original one.
        EXPECT_THROW(static_cast<void>(KR::Fp32::selectCachedCarriedArgSite(staged, lastSite(staged), false, sms,
                                                                            stream, ReductionTuningDepth::Quick)),
                     std::invalid_argument);
        EXPECT_THROW(static_cast<void>(KR::Fp32::selectCachedArgSite(final_carried, lastSite(final_carried), false, sms,
                                                                     stream, ReductionTuningDepth::Quick)),
                     std::invalid_argument);
        struct Case {
            const DenseReductionProblem* problem;
            DenseReductionSite site;
            bool maximum;
        };
        const std::vector<Case> cases{{&final_carried, lastSite(final_carried), true}, {&middle, middle_site, false}};
        for (const Case& c : cases) {
            const DenseReductionProblem& problem = *c.problem;
            const bool final_pass = !hasDenseReductionSites(advanceDenseReductionProblem(problem, c.site, 1));
            const KR::Fp32::Fp32KRSiteGeometry geometry = KR::Fp32::fp32KRSiteGeometry(problem, c.site);
            const uint64_t rows = geometry.rows, elements = geometry.row_bytes / 4;
            const uint64_t stride = denseArgOriginalIndexStride(problem.runs[c.site.run_index]).value_or(0);
            SCOPED_TRACE(::testing::Message() << "rows=" << rows << " R=" << elements << " index "
                                              << static_cast<int>(index_dtype) << " stride " << stride << " final "
                                              << final_pass);
            const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR);
            const auto regular = KR::Fp32::selectCachedCarriedArgSite(problem, c.site, c.maximum, sms, stream,
                                                                      ReductionTuningDepth::Regular);
            ASSERT_TRUE(regular.shape);
            EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR));
            const bool produce_value = !final_pass || problem.final_output.produce_value;
            const auto shortlist =
                KR::Fp32::fp32KRShortlist(rows, 4 * elements, sms,
                                          KR::Fp32::fp32KRArgRankerSite(c.maximum, produce_value, true),
                                          ReductionTuningDepth::Regular);
            EXPECT_EQ(shortlist.size(), regular.candidate_count);
            EXPECT_NE(shortlist.end(), std::find(shortlist.begin(), shortlist.end(), *regular.shape));
            EXPECT_EQ("fp32_carried_arg_kr:" + KR::krShapeName(*regular.shape), regular.winner_name);
            const auto quick = KR::Fp32::selectCachedCarriedArgSite(problem, c.site, c.maximum, sms, stream,
                                                                    ReductionTuningDepth::Quick);
            EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR));
            ASSERT_TRUE(quick.shape);
            EXPECT_EQ(*regular.shape, *quick.shape);
            static_cast<void>(KR::Fp32::selectCachedCarriedArgSite(problem, c.site, !c.maximum, sms, stream,
                                                                   ReductionTuningDepth::Quick));
            EXPECT_EQ(before + 2, cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR));

            // The winner computes the reduction with the site's stride (ties included: values from a small set).
            std::vector<float> values(rows * elements);
            for (float& v : values) v = static_cast<float>(static_cast<int>(rng() % 9) - 4);
            const std::vector<uint64_t> carried = carriedIndicesFor(rows * elements, stride, index_dtype, rng);
            const Tensor input = makeGpuTensor(values, {rows, elements}, stream);
            const Tensor carried_index = makeGpuUnsignedTensor(carried, {rows, elements}, stream, index_dtype);
            Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT64, {rows}));
            std::optional<Tensor> scratch;
            const uint64_t scratch_bytes = fp32KRCarriedArgScratchBytes(*regular.shape);
            if (scratch_bytes != 0) scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
            launchFp32KRCarriedArgSegments(input, carried_index, nullptr, outputSpan(index), *regular.shape,
                                           scratch ? scratch->getMemPtr<void>() : nullptr, scratch_bytes, c.maximum,
                                           stride, stream);
            stream.synchronize();
            const std::vector<uint64_t> got = copyGpuTensorAsUnsigned(index, stream);
            for (uint64_t row = 0; row < rows; ++row) {
                Reference::Candidate best{carried[row * elements], values[row * elements]};
                for (uint64_t i = 1; i < elements; ++i) {
                    best = Reference::combine(best, {carried[row * elements + i] + i * stride,
                                                     values[row * elements + i]},
                                              c.maximum);
                }
                ASSERT_EQ(best.index, got[row]) << "row " << row;
            }
        }
        // The original ARG site of the final carried site's geometry is a separate selection. A final original
        // site's key does not hold the carried index dtype (its pass reads none), so each index dtype asks for the
        // other operation: argmax first, then argmin.
        const DenseReductionProblem same_geometry = fp32CarriedArgProblem({640, 32}, {1}, index_dtype);
        const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR);
        static_cast<void>(KR::Fp32::selectCachedArgSite(same_geometry, lastSite(same_geometry),
                                                        index_dtype == DataType::UINT32, sms, stream,
                                                        ReductionTuningDepth::Quick));
        EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR));

        // The explicit-input autotune (no cache); inputs of the wrong dtype or size are rejected.
        const uint64_t count = 192 * 4096;
        std::vector<float> values(count);
        for (float& v : values) v = static_cast<float>(static_cast<int>(rng() % 9) - 4);
        const Tensor input = makeGpuTensor(values, {192, 4096}, stream);
        const Tensor carried = makeGpuUnsignedTensor(carriedIndicesFor(count, 1, index_dtype, rng), {192, 4096},
                                                     stream, index_dtype);
        const auto direct = KR::Fp32::autotuneCarriedArgSite(middle, middle_site, true, sms, input, carried, stream,
                                                             ReductionTuningDepth::Quick);
        EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::Fp32KR));
        ASSERT_TRUE(direct.shape);
        EXPECT_TRUE(KR::fp32KRArgShapeLegal(*direct.shape));
        const Tensor short_carried = makeGpuUnsignedTensor(carriedIndicesFor(count - 1, 1, index_dtype, rng),
                                                           {count - 1}, stream, index_dtype);
        EXPECT_THROW(static_cast<void>(KR::Fp32::autotuneCarriedArgSite(middle, middle_site, true, sms, input,
                                                                        short_carried, stream,
                                                                        ReductionTuningDepth::Quick)),
                     std::invalid_argument);
        EXPECT_THROW(static_cast<void>(KR::Fp32::autotuneCarriedArgSite(middle, middle_site, true, sms, carried,
                                                                        carried, stream, ReductionTuningDepth::Quick)),
                     std::invalid_argument);
    }
}
