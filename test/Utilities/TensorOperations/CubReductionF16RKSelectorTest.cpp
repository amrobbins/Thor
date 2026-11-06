// 1247: the retained FP16 / BF16 RK VALUE selector (F16RK::selectCachedValueSite / autotuneValueSite), called directly
// (not routed yet, plans/Thor_FP16_BF16_Dense_Reduction_Plan phase 8). Its domain (original FP16 / BF16 RK sites), and
// on the GPU: the winner is a shortlisted 2-byte first stage, the selection is cached under kind F16RK (a deeper
// selection answers a shallower request), FP16 and BF16 at one geometry are separate entries, an input of lower
// alignment selects only the passes that run at it (another entry), and the winner's timed program (first stage plus
// the production FP32 continuation) computes the reduction. Every pool entry computing its stage is
// CubReductionF16RK.PoolEntriesRunThroughThePhysicalAdapters (1246). 1251: the ARG selector (F16RK::selectCachedArgSite
// / autotuneArgSite): its domain, the winner is a shortlisted 2-byte ARG pass, cached under kind F16RK apart from
// VALUE and per format, an input of lower alignment selects only the passes that run at it, and the winner computes
// the value and index. Every ARG candidate computing is CubReductionF16RK.ArgCandidatesRunThroughThePassLauncher.

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Cub/F16RK/F16RKArgCandidates.h"
#include "Utilities/TensorOperations/Cub/F16RK/F16RKCandidates.h"
#include "Utilities/TensorOperations/Cub/F16RK/F16RKStampAutotuner.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKIsolatedSite.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectedContinuationBenchmark.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;
using namespace ThorImplementation::CubReductionInternal;
using Fp8RK::PhysicalKernel;

namespace {

DenseReductionProblem valueProblem(const std::vector<uint64_t>& dims, const std::vector<uint32_t>& axes) {
    DenseReductionOutputSpec output;
    output.produce_value = true;
    output.value_dtype = DataType::FP32;
    return makeInitialDenseReductionProblem(dims, axes, DenseReductionAggregateKind::Value, output);
}

DenseReductionSite lastSite(const DenseReductionProblem& problem) {
    return enumerateDenseReductionSites(problem).back();
}

// Quarters with small sums: exact in FP16, BF16 and FP32 in any order.
std::vector<float> dyadic(uint64_t count, uint64_t k) {
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    std::vector<float> values(count);
    for (uint64_t i = 0; i < count; ++i) values[i] = pattern[(i * 5 + i / k) % pattern.size()];
    return values;
}

// [outer, K] reduced over R with the semantics' combine and transform (unfinalized).
std::vector<float> reference(const std::vector<float>& x, uint64_t outer, uint64_t r, uint64_t k,
                             const CubReductionStageSemantics& semantics) {
    const bool sum = semantics.combine == CubReductionStageCombine::Sum;
    const bool maximum = semantics.combine == CubReductionStageCombine::Maximum;
    std::vector<float> out(outer * k, sum ? 0.0f
                                          : (maximum ? -std::numeric_limits<float>::infinity()
                                                     : std::numeric_limits<float>::infinity()));
    for (uint64_t o = 0; o < outer; ++o) {
        for (uint64_t row = 0; row < r; ++row) {
            for (uint64_t c = 0; c < k; ++c) {
                float v = x[(o * r + row) * k + c];
                if (semantics.input_transform == CubReductionStageInputTransform::AbsoluteValue) v = std::fabs(v);
                if (semantics.input_transform == CubReductionStageInputTransform::Square) v = v * v;
                float& acc = out[o * k + c];
                acc = sum ? acc + v : (maximum ? std::max(acc, v) : std::min(acc, v));
            }
        }
    }
    return out;
}

// Runs the selection's timed program (first stage, then the production FP32 continuation for a staged stage) on
// dyadic input and checks it against the reference (finalized for MEAN, scaled by 2).
void expectWinnerComputes(const DenseReductionProblem& problem, DenseReductionSite site, DataType dtype,
                          CubReductionOp op, const PhysicalKernel& physical, uint64_t outer, uint64_t r, uint64_t k,
                          Stream& stream) {
    const auto semantics = makeValueReductionStageSemantics(op, DensePassRole::Complete, r);
    const DenseProductionPhysicalSelectionContext context(dtype, denseRKValueOperation(op), stream, r, 2.0f, op);
    auto isolated = Fp8RK::planIsolatedRKSite(problem, site, physical, semantics);
    const auto program = Fp8RK::makeSelectedIsolatedBenchmark(problem, site, std::move(isolated), physical,
                                                              context.context(), 2.0f,
                                                              std::numeric_limits<uint64_t>::max(), stream);
    const std::vector<float> values = dyadic(outer * r * k, k);
    const Tensor input = makeGpuTensor(values, {outer, r, k}, stream, dtype);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {outer, k}));
    std::optional<Tensor> scratch;
    if (program.scratch_bytes != 0)
        scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {program.scratch_bytes}));
    program.launch(input, output, scratch ? *scratch : output, stream);
    stream.synchronize();
    const auto actual = copyGpuTensorAsFloat(output, stream);
    const auto expected = reference(values, outer, r, k, semantics);
    for (size_t i = 0; i < expected.size(); ++i) {
        const double value = op == CubReductionOp::Mean ? static_cast<double>(expected[i]) / static_cast<double>(r)
                                                        : static_cast<double>(expected[i]);
        ASSERT_NEAR(2.0 * value, actual[i], 1e-6 * std::fabs(2.0 * value) + 1e-6) << i;
    }
}

DenseReductionProblem argProblem(const std::vector<uint64_t>& dims, const std::vector<uint32_t>& axes,
                                 bool produce_value) {
    DenseReductionOutputSpec output;
    output.produce_value = produce_value;
    output.value_dtype = DataType::FP32;
    output.produce_index = true;
    output.index_dtype = DataType::UINT32;
    return makeInitialDenseReductionProblem(dims, axes, DenseReductionAggregateKind::Arg, output);
}

// Runs the ARG winner (one pass; a staged pass includes its packed continuation) on dyadic input (ties included) and
// checks the index (the lowest row of the extreme) and, when produced, the value against the reference.
void expectArgWinnerComputes(const PhysicalKernel& physical, DataType dtype, bool maximum, bool produce_value,
                             uint64_t outer, uint64_t r, uint64_t k, Stream& stream) {
    const std::vector<float> values = dyadic(outer * r * k, k);
    const Tensor input = makeGpuTensor(values, {outer, r, k}, stream, dtype);
    const auto geometry = CubReduction::analyzeGeometry({outer, r, k}, std::vector<uint32_t>{1});
    const uint64_t scratch_bytes = F16RK::f16RKArgScratchBytes(physical, outer, r, k);
    std::optional<Tensor> scratch;
    if (scratch_bytes != 0) scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
    uint64_t* packed = scratch ? static_cast<uint64_t*>(scratch->getMemPtr<void>()) : nullptr;
    Tensor value(gpuPlacement, TensorDescriptor(DataType::FP32, {outer, k}));
    Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
    F16RK::launchF16RKArgPass(physical, input, produce_value ? &value : nullptr, &index, geometry, packed, maximum, 1,
                              stream);
    stream.synchronize();
    const auto got_index = copyGpuTensorAsUnsigned(index, stream);
    const auto got_value = produce_value ? copyGpuTensorAsFloat(value, stream) : std::vector<float>{};
    for (uint64_t o = 0; o < outer; ++o) {
        for (uint64_t c = 0; c < k; ++c) {
            uint64_t best = 0;
            for (uint64_t row = 1; row < r; ++row) {
                const float v = values[(o * r + row) * k + c], b = values[(o * r + best) * k + c];
                if (maximum ? v > b : v < b) best = row;
            }
            ASSERT_EQ(best, got_index[o * k + c]) << "outer " << o << " k " << c;
            if (produce_value) {
                ASSERT_EQ(values[(o * r + best) * k + c], got_value[o * k + c]);
            }
        }
    }
}

}  // namespace

TEST(CubReductionF16RKSelector, DomainIsOriginalF16RKSites) {
    const DenseReductionProblem rk = valueProblem({4, 512, 64}, {1});
    const DenseReductionProblem kr = valueProblem({64, 512}, {1});
    EXPECT_TRUE(F16RK::isOriginalF16RKSite(rk, lastSite(rk), DataType::FP16));
    EXPECT_TRUE(F16RK::isOriginalF16RKSite(rk, lastSite(rk), DataType::BF16));
    EXPECT_FALSE(F16RK::isOriginalF16RKSite(rk, lastSite(rk), DataType::FP32));
    EXPECT_FALSE(F16RK::isOriginalF16RKSite(rk, lastSite(rk), DataType::FP8_E4M3));
    EXPECT_FALSE(F16RK::isOriginalF16RKSite(kr, lastSite(kr), DataType::FP16));
    // An intermediate site (FP32 partials of an earlier pass) belongs to the FP32 selectors.
    DenseReductionProblem intermediate = rk;
    intermediate.input_state = DenseReductionInputState::IntermediateAggregate;
    EXPECT_FALSE(F16RK::isOriginalF16RKSite(intermediate, lastSite(rk), DataType::FP16));
}

TEST(CubReductionF16RKSelector, RejectsSitesOutsideItsDomain) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const DenseReductionProblem kr = valueProblem({64, 512}, {1});
    EXPECT_THROW(static_cast<void>(F16RK::selectCachedValueSite(kr, lastSite(kr), DataType::FP16, CubReductionOp::Sum,
                                                                512, 1.0f, stream)),
                 std::invalid_argument);
    const DenseReductionProblem rk = valueProblem({4, 512, 64}, {1});
    const DenseReductionSite site = lastSite(rk);
    EXPECT_THROW(static_cast<void>(F16RK::selectCachedValueSite(rk, site, DataType::FP32, CubReductionOp::Sum, 512,
                                                                1.0f, stream)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(F16RK::selectCachedValueSite(rk, site, DataType::FP16, CubReductionOp::Sum, 0,
                                                                1.0f, stream)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(F16RK::selectCachedValueSite(rk, site, DataType::BF16, CubReductionOp::Sum, 512,
                                                                std::numeric_limits<float>::infinity(), stream)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(F16RK::selectCachedValueSite(rk, site, DataType::BF16, CubReductionOp::Sum, 512,
                                                                1.0f, stream, std::numeric_limits<uint64_t>::max(),
                                                                CubReductionTuningDepth::Quick, 1)),
                 std::invalid_argument);
    // The autotuner takes its format from the representative input.
    const Tensor fp32 = makeGpuTensor(std::vector<float>(4 * 512 * 64, 1.0f), {4, 512, 64}, stream, DataType::FP32);
    EXPECT_THROW(static_cast<void>(F16RK::autotuneValueSite(rk, site, CubReductionOp::Sum, 512, 1.0f, fp32, stream)),
                 std::invalid_argument);
}

TEST(CubReductionF16RKSelector, SelectsCachesAndComputes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint32_t sms = denseReductionDeviceIdentity(0).sm_count;
    DenseReductionSelectionCache& cache = denseReductionSelectionCache();
    cache.clear();
    const auto attempts = [&] { return cache.tuningAttempts(DenseReductionSelectionKind::F16RK); };
    // Small K with deep R (OneWindow / DeepRSmallK / rotated), mid K, large aligned K (AlignedK), large odd K
    // (RuntimeK), many outputs with short R.
    const std::vector<std::tuple<uint64_t, uint64_t, uint64_t, CubReductionOp>> sites{
        {8, 4096, 16, CubReductionOp::Sum},    {64, 512, 200, CubReductionOp::Max},
        {16, 2048, 1024, CubReductionOp::Sum}, {4, 3000, 1001, CubReductionOp::L1Norm},
        {1000, 16, 64, CubReductionOp::Mean}};
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (const auto& [outer, r, k, op] : sites) {
            SCOPED_TRACE(::testing::Message() << (dtype == DataType::FP16 ? "fp16" : "bf16") << " outer=" << outer
                                              << " R=" << r << " K=" << k << " op " << static_cast<int>(op));
            const DenseReductionProblem problem = valueProblem({outer, r, k}, {1});
            const DenseReductionSite site = lastSite(problem);
            const uint64_t before = attempts();
            const auto regular = F16RK::selectCachedValueSite(problem, site, dtype, op, r, 2.0f, stream,
                                                              std::numeric_limits<uint64_t>::max(),
                                                              CubReductionTuningDepth::Regular);
            const uint64_t tuned = attempts();
            EXPECT_EQ(before + 1, tuned);  // the continuation's FP32 sites tune under their own kinds
            EXPECT_EQ(2u, regular.physical.element_bytes);
            EXPECT_TRUE(Fp8RK::structurallyValid(regular.physical, outer, r, k));

            // The winner is one of the shortlisted kernels, and every shortlisted kernel was timed.
            const auto semantics = makeValueReductionStageSemantics(op, DensePassRole::Complete, r);
            F16RK::F16RKSite candidate_site;
            candidate_site.outer = outer;
            candidate_site.r = r;
            candidate_site.k = k;
            candidate_site.sm_count = sms;
            candidate_site.input_dtype = dtype;
            candidate_site.semantics = semantics;
            const auto pool = F16RK::f16RKCandidatePool(candidate_site, [&](const PhysicalKernel& p) {
                return F16RK::f16RKCompiledResidency(p, dtype, k, semantics);
            });
            const auto shortlist = F16RK::f16RKRankedKernels(
                pool, candidate_site, F16RK::f16RKShortlistLength(CubReductionTuningDepth::Regular));
            EXPECT_EQ(shortlist.size(), regular.candidate_count);
            EXPECT_NE(shortlist.end(), std::find(shortlist.begin(), shortlist.end(), regular.physical));

            // Cached: the same request and a shallower one tune nothing and return the same kernel.
            const auto again = F16RK::selectCachedValueSite(problem, site, dtype, op, r, 2.0f, stream);
            const auto quick = F16RK::selectCachedValueSite(problem, site, dtype, op, r, 2.0f, stream,
                                                            std::numeric_limits<uint64_t>::max(),
                                                            CubReductionTuningDepth::Quick);
            EXPECT_EQ(tuned, attempts());
            EXPECT_EQ(regular.physical, again.physical);
            EXPECT_EQ(regular.physical, quick.physical);

            expectWinnerComputes(problem, site, dtype, op, regular.physical, outer, r, k, stream);
        }
    }
    // FP16 and BF16 at one geometry are separate entries (both have element size 2); a deeper selection answers a
    // shallower request, a shallower one does not answer a deeper request; another operation is another site.
    const DenseReductionProblem problem = valueProblem({32, 1000, 48}, {1});
    const DenseReductionSite site = lastSite(problem);
    const auto max = std::numeric_limits<uint64_t>::max();
    uint64_t before = attempts();
    static_cast<void>(F16RK::selectCachedValueSite(problem, site, DataType::FP16, CubReductionOp::Sum, 1000, 1.0f,
                                                   stream, max, CubReductionTuningDepth::Quick));
    static_cast<void>(F16RK::selectCachedValueSite(problem, site, DataType::BF16, CubReductionOp::Sum, 1000, 1.0f,
                                                   stream, max, CubReductionTuningDepth::Quick));
    EXPECT_EQ(before + 2, attempts());
    before = attempts();
    const auto ultra = F16RK::selectCachedValueSite(problem, site, DataType::FP16, CubReductionOp::Sum, 1000, 1.0f,
                                                    stream, max, CubReductionTuningDepth::Ultra);
    EXPECT_EQ(before + 1, attempts());
    const auto regular = F16RK::selectCachedValueSite(problem, site, DataType::FP16, CubReductionOp::Sum, 1000, 1.0f,
                                                      stream, max, CubReductionTuningDepth::Regular);
    EXPECT_EQ(before + 1, attempts());
    EXPECT_EQ(ultra.physical, regular.physical);
    static_cast<void>(F16RK::selectCachedValueSite(problem, site, DataType::FP16, CubReductionOp::Max, 1000, 1.0f,
                                                   stream, max, CubReductionTuningDepth::Quick));
    EXPECT_EQ(before + 2, attempts());
}

// An input of lower alignment (Tensor::alignmentBytes) is another entry, and its winner runs at that alignment: at 2
// bytes, the 2-byte AlignedK packet (odd K) or the rotated RuntimeK pass; at 4 bytes, AlignedK packets of at most 4
// bytes or rotated. Each winner computes the reduction (on an aligned input; the pass is alignment-tolerant).
TEST(CubReductionF16RKSelector, AlignmentSelectsTolerantPasses) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    DenseReductionSelectionCache& cache = denseReductionSelectionCache();
    const auto attempts = [&] { return cache.tuningAttempts(DenseReductionSelectionKind::F16RK); };
    const auto max = std::numeric_limits<uint64_t>::max();
    for (const auto& [outer, r, k, alignment] :
         {std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>{8, 1000, 101, 2}, {8, 1000, 100, 4}, {4, 700, 64, 8}}) {
        SCOPED_TRACE(::testing::Message() << "K=" << k << " alignment " << alignment);
        const DenseReductionProblem problem = valueProblem({outer, r, k}, {1});
        const DenseReductionSite site = lastSite(problem);
        const uint64_t before = attempts();
        const auto aligned = F16RK::selectCachedValueSite(problem, site, DataType::BF16, CubReductionOp::Sum, r, 1.0f,
                                                          stream, max, CubReductionTuningDepth::Quick, 16);
        const auto tolerant = F16RK::selectCachedValueSite(problem, site, DataType::BF16, CubReductionOp::Sum, r, 1.0f,
                                                           stream, max, CubReductionTuningDepth::Quick, alignment);
        EXPECT_EQ(before + 2, attempts());
        EXPECT_LE(F16RK::f16RKPassInputAlignment(tolerant.physical), alignment);
        static_cast<void>(aligned);
        expectWinnerComputes(problem, site, DataType::BF16, CubReductionOp::Sum, tolerant.physical, outer, r, k,
                             stream);
    }
}

TEST(CubReductionF16RKSelector, ArgDomainAndRejections) {
    const DenseReductionProblem rk = argProblem({4, 512, 64}, {1}, true);
    EXPECT_TRUE(F16RK::isOriginalF16RKSite(rk, lastSite(rk), DataType::FP16));
    EXPECT_TRUE(F16RK::isOriginalF16RKSite(rk, lastSite(rk), DataType::BF16));
    EXPECT_FALSE(F16RK::isOriginalF16RKSite(rk, lastSite(rk), DataType::FP32));
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const DenseReductionSite site = lastSite(rk);
    const DenseReductionProblem value = valueProblem({4, 512, 64}, {1});
    EXPECT_THROW(static_cast<void>(F16RK::selectCachedArgSite(value, lastSite(value), DataType::FP16, true, 512,
                                                              stream)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(F16RK::selectCachedValueSite(rk, site, DataType::FP16, CubReductionOp::Max, 512,
                                                                1.0f, stream)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(F16RK::selectCachedArgSite(rk, site, DataType::FP32, true, 512, stream)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(F16RK::selectCachedArgSite(rk, site, DataType::FP16, true, 0, stream)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(F16RK::selectCachedArgSite(rk, site, DataType::BF16, false, 512, stream,
                                                              std::numeric_limits<uint64_t>::max(),
                                                              CubReductionTuningDepth::Quick, 1)),
                 std::invalid_argument);
    const DenseReductionProblem kr = argProblem({64, 512}, {1}, false);
    EXPECT_THROW(static_cast<void>(F16RK::selectCachedArgSite(kr, lastSite(kr), DataType::FP16, true, 512, stream)),
                 std::invalid_argument);
    const Tensor fp32 = makeGpuTensor(std::vector<float>(4 * 512 * 64, 1.0f), {4, 512, 64}, stream, DataType::FP32);
    EXPECT_THROW(static_cast<void>(F16RK::autotuneArgSite(rk, site, true, 512, fp32, stream)), std::invalid_argument);
}

TEST(CubReductionF16RKSelector, ArgSelectsCachesAndComputes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint32_t sms = denseReductionDeviceIdentity(0).sm_count;
    DenseReductionSelectionCache& cache = denseReductionSelectionCache();
    cache.clear();
    const auto attempts = [&] { return cache.tuningAttempts(DenseReductionSelectionKind::F16RK); };
    const auto max = std::numeric_limits<uint64_t>::max();
    // Small K with deep R, mid K, large aligned K, large odd K (no AlignedK twin), many outputs with short R.
    const std::vector<std::tuple<uint64_t, uint64_t, uint64_t, bool, bool>> sites{
        {8, 4096, 16, true, true},   {64, 512, 200, false, false}, {16, 2048, 1024, true, false},
        {4, 3000, 1001, false, true}, {1000, 16, 64, true, true}};
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (const auto& [outer, r, k, maximum, produce_value] : sites) {
            SCOPED_TRACE(::testing::Message() << (dtype == DataType::FP16 ? "fp16" : "bf16") << " outer=" << outer
                                              << " R=" << r << " K=" << k << " max " << maximum << " value "
                                              << produce_value);
            const DenseReductionProblem problem = argProblem({outer, r, k}, {1}, produce_value);
            const DenseReductionSite site = lastSite(problem);
            const uint64_t before = attempts();
            const auto regular = F16RK::selectCachedArgSite(problem, site, dtype, maximum, r, stream, max,
                                                            CubReductionTuningDepth::Regular);
            const uint64_t tuned = attempts();
            EXPECT_EQ(before + 1, tuned);
            EXPECT_EQ(2u, regular.physical.element_bytes);
            EXPECT_TRUE(F16RK::f16RKArgPassLegal(regular.physical, outer, r, k));

            // The winner is one of the shortlisted ARG passes, and every shortlisted pass was timed.
            auto semantics = makeValueReductionStageSemantics(maximum ? CubReductionOp::Max : CubReductionOp::Min,
                                                              DensePassRole::Complete, r);
            semantics.combine = maximum ? CubReductionStageCombine::Maximum : CubReductionStageCombine::Minimum;
            semantics.input_transform = CubReductionStageInputTransform::Identity;
            F16RK::F16RKSite candidate_site;
            candidate_site.outer = outer;
            candidate_site.r = r;
            candidate_site.k = k;
            candidate_site.sm_count = sms;
            candidate_site.input_dtype = dtype;
            candidate_site.semantics = semantics;
            const auto pool = F16RK::f16RKCandidatePool(candidate_site, [&](const PhysicalKernel& p) {
                return F16RK::f16RKCompiledResidency(p, dtype, k, semantics);
            });
            const auto args = F16RK::f16RKArgCandidates(pool, outer, r, k, max);
            const auto shortlist =
                F16RK::f16RKRankedKernels(F16RK::f16RKArgRankerPool(args), candidate_site,
                                          F16RK::f16RKShortlistLength(CubReductionTuningDepth::Regular),
                                          produce_value ? 2U : 1U);
            EXPECT_EQ(shortlist.size(), regular.candidate_count);
            EXPECT_NE(shortlist.end(), std::find(shortlist.begin(), shortlist.end(), regular.physical));
            if (k % 2 != 0) {
                EXPECT_NE(Fp8RK::PhysicalFamily::AlignedK, regular.physical.family);
            }

            // Cached: the same request and a shallower one tune nothing and return the same pass.
            const auto again = F16RK::selectCachedArgSite(problem, site, dtype, maximum, r, stream);
            const auto quick = F16RK::selectCachedArgSite(problem, site, dtype, maximum, r, stream, max,
                                                          CubReductionTuningDepth::Quick);
            EXPECT_EQ(tuned, attempts());
            EXPECT_EQ(regular.physical, again.physical);
            EXPECT_EQ(regular.physical, quick.physical);

            expectArgWinnerComputes(regular.physical, dtype, maximum, produce_value, outer, r, k, stream);
        }
    }
    // FP16 and BF16 are separate entries, as are argmin and argmax, the index-only and value-and-index requests, and
    // VALUE MAX at the same geometry.
    const uint64_t before = attempts();
    const DenseReductionProblem index_only = argProblem({32, 1000, 48}, {1}, false);
    const DenseReductionProblem both = argProblem({32, 1000, 48}, {1}, true);
    for (const DataType dtype : {DataType::FP16, DataType::BF16}) {
        for (bool maximum : {false, true}) {
            static_cast<void>(F16RK::selectCachedArgSite(index_only, lastSite(index_only), dtype, maximum, 1000,
                                                         stream, max, CubReductionTuningDepth::Quick));
        }
    }
    static_cast<void>(F16RK::selectCachedArgSite(both, lastSite(both), DataType::FP16, true, 1000, stream, max,
                                                 CubReductionTuningDepth::Quick));
    const DenseReductionProblem value = valueProblem({32, 1000, 48}, {1});
    static_cast<void>(F16RK::selectCachedValueSite(value, lastSite(value), DataType::FP16, CubReductionOp::Max, 1000,
                                                   1.0f, stream, max, CubReductionTuningDepth::Quick));
    EXPECT_EQ(before + 6, attempts());
}

// An ARG input of lower alignment is another entry, and its winner runs at that alignment: at 2 bytes only the rotated
// pass; at 4 bytes AlignedK's 4-byte packet or rotated.
TEST(CubReductionF16RKSelector, ArgAlignmentSelectsTolerantPasses) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    DenseReductionSelectionCache& cache = denseReductionSelectionCache();
    const auto attempts = [&] { return cache.tuningAttempts(DenseReductionSelectionKind::F16RK); };
    const auto max = std::numeric_limits<uint64_t>::max();
    for (const auto& [outer, r, k, alignment] :
         {std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>{8, 1000, 101, 2}, {8, 1000, 100, 4}, {4, 700, 64, 8}}) {
        SCOPED_TRACE(::testing::Message() << "K=" << k << " alignment " << alignment);
        const DenseReductionProblem problem = argProblem({outer, r, k}, {1}, true);
        const DenseReductionSite site = lastSite(problem);
        const uint64_t before = attempts();
        static_cast<void>(F16RK::selectCachedArgSite(problem, site, DataType::BF16, true, r, stream, max,
                                                     CubReductionTuningDepth::Quick, 16));
        const auto tolerant = F16RK::selectCachedArgSite(problem, site, DataType::BF16, true, r, stream, max,
                                                         CubReductionTuningDepth::Quick, alignment);
        EXPECT_EQ(before + 2, attempts());
        EXPECT_LE(F16RK::f16RKPassInputAlignment(tolerant.physical), alignment);
        expectArgWinnerComputes(tolerant.physical, DataType::BF16, true, true, outer, r, k, stream);
    }
}
