// 1215: and the carried-index ARG selector (Fp32RK::selectCachedCarriedArgSite / autotuneCarriedArgSite).
// 1192: the FP32 RK VALUE candidate pool and selector (Fp32RK/), called directly (production routes FP32 RK sites).
// On the GPU: every pool entry, with the compiled residency, computes the reduction through the shared physical
// launcher (Complete outputs scaled; staged partials combined over their shards on the host), exactly on dyadic inputs;
// the selector's winner is a shortlisted FP32 first stage, cached under kind Fp32RK (a deeper selection answers a
// shallower request), and its timed program (first stage plus the production FP32 continuation) computes the reduction.

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKArgCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKCarriedArgCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKStampAutotuner.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKIsolatedSite.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalKernels.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectedContinuationBenchmark.h"

#include <algorithm>
#include <bit>
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
using Fp8RK::PhysicalFamily;
using Fp8RK::PhysicalKernel;

namespace {

DenseReductionProblem fp32ValueProblem(const std::vector<uint64_t>& dims, const std::vector<uint32_t>& axes) {
    DenseReductionOutputSpec output;
    output.produce_value = true;
    output.value_dtype = DataType::FP32;
    return makeInitialDenseReductionProblem(dims, axes, DenseReductionAggregateKind::Value, output);
}

DenseReductionProblem fp32ArgProblem(const std::vector<uint64_t>& dims, const std::vector<uint32_t>& axes,
                                     bool produce_value) {
    DenseReductionOutputSpec output;
    output.produce_value = produce_value;
    output.value_dtype = DataType::FP32;
    output.produce_index = true;
    output.index_dtype = DataType::UINT32;
    return makeInitialDenseReductionProblem(dims, axes, DenseReductionAggregateKind::Arg, output);
}

DenseReductionSite lastSite(const DenseReductionProblem& problem) {
    return enumerateDenseReductionSites(problem).back();
}

// Dyadic inputs: quarters with small sums (additive) and sparse +-1 / 2 / 0.5 (PRODUCT), exact in any order.
std::vector<float> dyadic(uint64_t count, uint64_t k, bool product) {
    const std::array<float, 8> pattern = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    std::vector<float> values(count, 1.0f);
    for (uint64_t i = 0; i < count; ++i) {
        if (!product) values[i] = pattern[(i * 5 + i / k) % pattern.size()];
        else if (i % 97 == 3) values[i] = (i % 3 == 0) ? -1.0f : ((i % 3 == 1) ? 2.0f : 0.5f);
    }
    return values;
}

float identityOf(CubReductionStageCombine combine) {
    switch (combine) {
        case CubReductionStageCombine::Sum: return 0.0f;
        case CubReductionStageCombine::Product: return 1.0f;
        case CubReductionStageCombine::Minimum: return std::numeric_limits<float>::infinity();
        case CubReductionStageCombine::Maximum: return -std::numeric_limits<float>::infinity();
    }
    return 0.0f;
}
float combineOf(CubReductionStageCombine combine, float a, float b) {
    switch (combine) {
        case CubReductionStageCombine::Sum: return a + b;
        case CubReductionStageCombine::Product: return a * b;
        case CubReductionStageCombine::Minimum: return std::min(a, b);
        case CubReductionStageCombine::Maximum: return std::max(a, b);
    }
    return a;
}
float transformOf(CubReductionStageInputTransform transform, float v) {
    if (transform == CubReductionStageInputTransform::AbsoluteValue) return std::fabs(v);
    if (transform == CubReductionStageInputTransform::Square) return v * v;
    return v;
}

// [outer, K] reduced over R, unfinalized (the combine of the transformed inputs).
std::vector<float> reference(const std::vector<float>& x, uint64_t outer, uint64_t r, uint64_t k,
                             const CubReductionStageSemantics& semantics) {
    std::vector<float> out(outer * k, identityOf(semantics.combine));
    for (uint64_t o = 0; o < outer; ++o)
        for (uint64_t row = 0; row < r; ++row)
            for (uint64_t c = 0; c < k; ++c)
                out[o * k + c] = combineOf(semantics.combine, out[o * k + c],
                                           transformOf(semantics.input_transform, x[(o * r + row) * k + c]));
    return out;
}

uint64_t shardsOf(const PhysicalKernel& p, uint64_t r) {
    switch (p.family) {
        case PhysicalFamily::OneWindow: return 1;
        case PhysicalFamily::DeepRSmallK: return (r + p.a - 1) / p.a;
        case PhysicalFamily::AlignedK: return p.c;
        case PhysicalFamily::RuntimeK: return p.f;
    }
    return 1;
}

// Thor's ARG contract: the first NaN wins, else the minimum (maximum) with the lowest row; -0 and +0 tie.
uint64_t argReferenceRow(const std::vector<float>& x, uint64_t outer, uint64_t r, uint64_t k, uint64_t c,
                         bool maximum) {
    uint64_t best = 0;
    for (uint64_t row = 0; row < r; ++row) {
        const float v = x[(outer * r + row) * k + c];
        const float b = x[(outer * r + best) * k + c];
        if (std::isnan(b)) break;
        if (std::isnan(v) || (maximum ? v > b : v < b)) best = row;
    }
    return best;
}

// Ties, both zeros, infinities and sparse NaNs.
std::vector<float> argValues(uint64_t count, uint64_t k) {
    const std::array<float, 12> pattern = {-2.0f, 1.0f, -0.0f, 0.0f, 2.0f, -2.0f, 3.5f, -1.0f, 2.0f,
                                           std::numeric_limits<float>::infinity(),
                                           -std::numeric_limits<float>::infinity(), 0.5f};
    std::vector<float> values(count);
    for (uint64_t i = 0; i < count; ++i) {
        values[i] = pattern[(i * 7 + i / k * 3) % pattern.size()];
        if (i % 4999 == 2501) values[i] = std::numeric_limits<float>::quiet_NaN();
    }
    return values;
}

// Runs an FP32 ARG pass and checks the value bits and the index (UINT32 at stride 3) against the contract.
void expectArgPassComputes(const PhysicalKernel& identity, const Tensor& input, const std::vector<float>& values,
                           uint64_t outer, uint64_t r, uint64_t k, bool maximum, Stream& stream) {
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry({outer, r, k}, std::vector<uint32_t>{1});
    const uint64_t scratch_bytes = Fp32RK::fp32RKArgScratchBytes(identity, outer, r, k);
    std::optional<Tensor> scratch;
    if (scratch_bytes != 0) scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
    Tensor value(gpuPlacement, TensorDescriptor(DataType::FP32, {outer, k}));
    Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT32, {outer, k}));
    Fp32RK::launchFp32RKArgPass(identity, input, &value, &index, geometry,
                                scratch ? static_cast<uint64_t*>(scratch->getMemPtr<void>()) : nullptr, maximum, 3,
                                stream);
    stream.synchronize();
    const auto got_value = copyGpuTensorAsFloat(value, stream);
    const auto got_index = copyGpuTensorAsUnsigned(index, stream);
    for (uint64_t o = 0; o < outer; ++o) {
        for (uint64_t c = 0; c < k; ++c) {
            const uint64_t i = o * k + c;
            const uint64_t row = argReferenceRow(values, o, r, k, c, maximum);
            const float want = values[(o * r + row) * k + c];
            ASSERT_EQ(3 * row, got_index[i]) << "outer " << o << " k " << c;
            if (std::isnan(want)) {
                ASSERT_TRUE(std::isnan(got_value[i])) << "outer " << o << " k " << c;
            } else {
                ASSERT_EQ(std::bit_cast<uint32_t>(want), std::bit_cast<uint32_t>(got_value[i]))
                    << "outer " << o << " k " << c;
            }
        }
    }
}

}  // namespace

TEST(CubReductionFp32RKSelector, EveryPoolEntryComputesTheReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint32_t sms = denseReductionDeviceIdentity(0).sm_count;
    const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> sites{
        {3, 300, 5}, {2, 1000, 64}, {4, 200, 128}, {1, 5000, 1001}, {64, 40, 256}, {5, 129, 250}, {300, 8, 4},
        {1, 2, 3}};
    const std::vector<CubReductionOp> ops{CubReductionOp::Sum, CubReductionOp::Max, CubReductionOp::L1Norm,
                                          CubReductionOp::Product, CubReductionOp::Min};
    size_t launched = 0;
    for (const auto& [outer, r, k] : sites) {
        const std::vector<float> additive = dyadic(outer * r * k, k, false);
        const std::vector<float> multiplicative = dyadic(outer * r * k, k, true);
        const Tensor additive_input = makeGpuTensor(additive, {outer, r, k}, stream, DataType::FP32);
        const Tensor product_input = makeGpuTensor(multiplicative, {outer, r, k}, stream, DataType::FP32);
        const CubReductionGeometry geometry = CubReduction::analyzeGeometry({outer, r, k}, std::vector<uint32_t>{1});
        for (CubReductionOp op : ops) {
            CubReductionStageSemantics semantics = makeValueReductionStageSemantics(op, DensePassRole::Complete, r);
            const bool product = op == CubReductionOp::Product;
            const Tensor& input = product ? product_input : additive_input;
            const std::vector<float> expected = reference(product ? multiplicative : additive, outer, r, k, semantics);
            Fp32RK::Fp32RKSite site{outer, r, k, sms, semantics};
            const auto pool = Fp32RK::fp32RKCandidatePool(
                site, [&](const PhysicalKernel& p) { return Fp32RK::fp32RKCompiledResidency(p, k, semantics); });
            ASSERT_FALSE(pool.empty());
            for (const auto& entry : pool) {
                const PhysicalKernel& p = entry.kernel;
                SCOPED_TRACE(::testing::Message() << "outer=" << outer << " R=" << r << " K=" << k << " op "
                                                  << static_cast<int>(op) << " family " << static_cast<int>(p.family)
                                                  << " a=" << p.a << " b=" << p.b << " c=" << p.c << " e=" << p.e);
                ASSERT_TRUE(Fp8RK::structurallyValid(p, outer, r, k));
                constexpr float scale = -0.5f;
                if (!p.staged()) {
                    // The Complete launchers take the output at the geometry's output dimensions (as the timed
                    // program reshapes it).
                    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
                    Fp8RK::launchPhysicalFirstStage(p, semantics, input, output, geometry, scale, stream);
                    stream.synchronize();
                    const auto actual = copyGpuTensorAsFloat(output, stream);
                    for (size_t i = 0; i < expected.size(); ++i) ASSERT_EQ(expected[i] * scale, actual[i]) << i;
                } else {
                    const uint64_t shards = shardsOf(p, r);
                    CubReductionStageSemantics first = semantics;
                    first.finalize = CubReductionStageFinalize::Identity;
                    first.finalize_divisor = 1;
                    Tensor partials(gpuPlacement, TensorDescriptor(DataType::FP32, {outer, shards, k}));
                    Fp8RK::launchPhysicalFirstStage(p, first, input, partials, geometry, 1.0f, stream);
                    stream.synchronize();
                    const auto actual = copyGpuTensorAsFloat(partials, stream);
                    for (uint64_t o = 0; o < outer; ++o) {
                        for (uint64_t c = 0; c < k; ++c) {
                            float combined = identityOf(semantics.combine);
                            for (uint64_t s = 0; s < shards; ++s)
                                combined = combineOf(semantics.combine, combined, actual[(o * shards + s) * k + c]);
                            ASSERT_EQ(expected[o * k + c], combined) << "outer " << o << " k " << c;
                        }
                    }
                }
                ++launched;
            }
        }
    }
    EXPECT_GT(launched, 100u);
}

TEST(CubReductionFp32RKSelector, RejectsSitesOutsideItsDomain) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // A KR site ([K, R] over R), a zero original size, a non-finite scale.
    const DenseReductionProblem kr = fp32ValueProblem({64, 512}, {1});
    EXPECT_THROW(static_cast<void>(Fp32RK::selectCachedValueSite(kr, lastSite(kr), CubReductionOp::Sum, 512, 1.0f,
                                                                 stream)),
                 std::invalid_argument);
    const DenseReductionProblem rk = fp32ValueProblem({4, 512, 64}, {1});
    EXPECT_THROW(static_cast<void>(Fp32RK::selectCachedValueSite(rk, lastSite(rk), CubReductionOp::Sum, 0, 1.0f,
                                                                 stream)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(Fp32RK::selectCachedValueSite(rk, lastSite(rk), CubReductionOp::Sum, 512,
                                                                 std::numeric_limits<float>::infinity(), stream)),
                 std::invalid_argument);
}

TEST(CubReductionFp32RKSelector, SelectsCachesAndComputes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint32_t sms = denseReductionDeviceIdentity(0).sm_count;
    DenseReductionSelectionCache& cache = denseReductionSelectionCache();
    cache.clear();
    // Small K with deep R (OneWindow / DeepRSmallK / rotated), mid K, large aligned K (AlignedK), large odd K (RuntimeK).
    const std::vector<std::tuple<uint64_t, uint64_t, uint64_t, CubReductionOp>> sites{
        {8, 4096, 16, CubReductionOp::Sum},   {64, 512, 100, CubReductionOp::Max},
        {16, 2048, 1024, CubReductionOp::Sum}, {4, 3000, 1001, CubReductionOp::L1Norm},
        {1000, 16, 64, CubReductionOp::Mean}};
    for (const auto& [outer, r, k, op] : sites) {
        SCOPED_TRACE(::testing::Message() << "outer=" << outer << " R=" << r << " K=" << k << " op "
                                          << static_cast<int>(op));
        const DenseReductionProblem problem = fp32ValueProblem({outer, r, k}, {1});
        const DenseReductionSite site = lastSite(problem);
        const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK);
        const auto regular = Fp32RK::selectCachedValueSite(problem, site, op, r, 2.0f, stream,
                                                           std::numeric_limits<uint64_t>::max(),
                                                           CubReductionTuningDepth::Regular);
        // 1226: at least this site tuned; a staged candidate's routed continuation tunes its own sites too.
        const uint64_t tuned = cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK);
        EXPECT_LT(before, tuned);
        EXPECT_EQ(4u, regular.physical.element_bytes);
        EXPECT_TRUE(Fp8RK::structurallyValid(regular.physical, outer, r, k));

        // The winner is one of the shortlisted kernels, and every shortlisted kernel was timed.
        const auto semantics = makeValueReductionStageSemantics(op, DensePassRole::Complete, r);
        const Fp32RK::Fp32RKSite candidate_site{outer, r, k, sms, semantics};
        const auto pool = Fp32RK::fp32RKCandidatePool(
            candidate_site, [&](const PhysicalKernel& p) { return Fp32RK::fp32RKCompiledResidency(p, k, semantics); });
        const auto shortlist = Fp32RK::fp32RKRankedKernels(
            pool, candidate_site, Fp32RK::fp32RKShortlistLength(CubReductionTuningDepth::Regular));
        EXPECT_EQ(shortlist.size(), regular.candidate_count);
        EXPECT_NE(shortlist.end(), std::find(shortlist.begin(), shortlist.end(), regular.physical));

        // Cached: the same request and a shallower one tune nothing and return the same kernel.
        const auto again = Fp32RK::selectCachedValueSite(problem, site, op, r, 2.0f, stream);
        const auto quick = Fp32RK::selectCachedValueSite(problem, site, op, r, 2.0f, stream,
                                                         std::numeric_limits<uint64_t>::max(),
                                                         CubReductionTuningDepth::Quick);
        EXPECT_EQ(tuned, cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK));
        EXPECT_EQ(regular.physical, again.physical);
        EXPECT_EQ(regular.physical, quick.physical);

        // The winner's timed program (first stage, then the production FP32 continuation for a staged stage).
        // 1226: with the original semantics, which the routed continuation requires.
        const DenseProductionPhysicalSelectionContext context(DataType::FP32, denseRKValueOperation(op), stream, r,
                                                              2.0f, op);
        auto isolated = Fp8RK::planIsolatedRKSite(problem, site, regular.physical, semantics);
        const auto program = Fp8RK::makeSelectedIsolatedBenchmark(problem, site, std::move(isolated),
                                                                  regular.physical, context.context(), 2.0f,
                                                                  std::numeric_limits<uint64_t>::max(), stream);
        const std::vector<float> values = dyadic(outer * r * k, k, false);
        const Tensor input = makeGpuTensor(values, {outer, r, k}, stream, DataType::FP32);
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
    // A deeper selection answers a shallower request; a shallower one does not answer a deeper request; another
    // operation is another site.
    const DenseReductionProblem problem = fp32ValueProblem({32, 1000, 48}, {1});
    const DenseReductionSite site = lastSite(problem);
    const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK);
    const auto max = std::numeric_limits<uint64_t>::max();
    // 1226: counts as "tuned" / "tuned nothing" (a routed continuation tunes its own sites too).
    const auto attempts = [&] { return cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK); };
    const auto quick =
        Fp32RK::selectCachedValueSite(problem, site, CubReductionOp::Sum, 1000, 1.0f, stream, max,
                                      CubReductionTuningDepth::Quick);
    const uint64_t after_quick = attempts();
    EXPECT_LT(before, after_quick);
    const auto ultra =
        Fp32RK::selectCachedValueSite(problem, site, CubReductionOp::Sum, 1000, 1.0f, stream, max,
                                      CubReductionTuningDepth::Ultra);
    const uint64_t after_ultra = attempts();
    EXPECT_LT(after_quick, after_ultra);
    const auto regular =
        Fp32RK::selectCachedValueSite(problem, site, CubReductionOp::Sum, 1000, 1.0f, stream, max,
                                      CubReductionTuningDepth::Regular);
    EXPECT_EQ(after_ultra, attempts());
    EXPECT_EQ(ultra.physical, regular.physical);
    EXPECT_LE(quick.candidate_count, ultra.candidate_count);
    static_cast<void>(Fp32RK::selectCachedValueSite(problem, site, CubReductionOp::Max, 1000, 1.0f, stream, max,
                                                    CubReductionTuningDepth::Quick));
    EXPECT_LT(after_ultra, attempts());
}

// 1204: every FP32 RK ARG candidate (the twins of the VALUE MIN / MAX pool and the ARG-only rotated passes) computes
// argmin / argmax exactly through launchFp32RKArgPass, and every family appears over the sites.
TEST(CubReductionFp32RKSelector, EveryArgCandidateComputesTheReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint32_t sms = denseReductionDeviceIdentity(0).sm_count;
    const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> sites{
        {3, 300, 5}, {2, 1000, 64}, {4, 200, 128}, {1, 5000, 1001}, {64, 40, 256}, {5, 129, 120}, {300, 8, 4},
        {1, 2, 3},   {2, 3000, 16}};
    std::array<size_t, 4> families{};
    size_t launched = 0;
    for (const auto& [outer, r, k] : sites) {
        const std::vector<float> values = argValues(outer * r * k, k);
        const Tensor input = makeGpuTensor(values, {outer, r, k}, stream, DataType::FP32);
        for (bool maximum : {false, true}) {
            auto semantics = makeValueReductionStageSemantics(maximum ? CubReductionOp::Max : CubReductionOp::Min,
                                                              DensePassRole::Complete, r);
            const Fp32RK::Fp32RKSite site{outer, r, k, sms, semantics};
            const auto pool = Fp32RK::fp32RKCandidatePool(
                site, [&](const PhysicalKernel& p) { return Fp32RK::fp32RKCompiledResidency(p, k, semantics); });
            const auto candidates =
                Fp32RK::fp32RKArgCandidates(pool, outer, r, k, std::numeric_limits<uint64_t>::max());
            ASSERT_FALSE(candidates.empty());
            for (const auto& candidate : candidates) {
                const PhysicalKernel& p = candidate.identity;
                SCOPED_TRACE(::testing::Message() << "outer=" << outer << " R=" << r << " K=" << k << " max "
                                                  << maximum << " family " << static_cast<int>(p.family)
                                                  << " a=" << p.a << " b=" << p.b << " c=" << p.c << " e=" << p.e);
                expectArgPassComputes(p, input, values, outer, r, k, maximum, stream);
                ++families[static_cast<size_t>(p.family)];
                ++launched;
            }
        }
    }
    for (size_t family = 0; family < families.size(); ++family) EXPECT_GT(families[family], 0u) << family;
    EXPECT_GT(launched, 100u);
}

// 1204: the FP32 RK ARG selector: its domain, a shortlisted winner, caching (kind Fp32RK; argmin and argmax are
// separate sites, and separate from VALUE), and the winner computing the reduction.
TEST(CubReductionFp32RKSelector, ArgSelectsCachesAndComputes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint32_t sms = denseReductionDeviceIdentity(0).sm_count;
    DenseReductionSelectionCache& cache = denseReductionSelectionCache();
    cache.clear();
    // A VALUE problem and a zero original size are outside the domain.
    const DenseReductionProblem value = fp32ValueProblem({4, 512, 64}, {1});
    EXPECT_THROW(static_cast<void>(Fp32RK::selectCachedArgSite(value, lastSite(value), false, 512, stream)),
                 std::invalid_argument);
    const DenseReductionProblem zero = fp32ArgProblem({4, 512, 64}, {1}, false);
    EXPECT_THROW(static_cast<void>(Fp32RK::selectCachedArgSite(zero, lastSite(zero), false, 0, stream)),
                 std::invalid_argument);

    struct Site {
        uint64_t outer, r, k;
        bool maximum, produce_value;
    };
    const std::vector<Site> sites{{8, 4096, 16, false, true}, {64, 512, 100, true, false},
                                  {16, 2048, 1024, false, true}, {4, 3000, 1001, true, true},
                                  {1000, 16, 64, false, false}};
    for (const Site& s : sites) {
        SCOPED_TRACE(::testing::Message() << "outer=" << s.outer << " R=" << s.r << " K=" << s.k << " max "
                                          << s.maximum << " value " << s.produce_value);
        const DenseReductionProblem problem = fp32ArgProblem({s.outer, s.r, s.k}, {1}, s.produce_value);
        const DenseReductionSite site = lastSite(problem);
        const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK);
        const auto regular = Fp32RK::selectCachedArgSite(problem, site, s.maximum, s.r, stream,
                                                         std::numeric_limits<uint64_t>::max(),
                                                         CubReductionTuningDepth::Regular);
        // 1226: at least this site tuned; a staged candidate's routed continuation tunes its own sites too.
        const uint64_t tuned = cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK);
        EXPECT_LT(before, tuned);
        ASSERT_NE(0u, regular.candidate_count);
        EXPECT_EQ(4u, regular.physical.element_bytes);
        EXPECT_TRUE(Fp32RK::fp32RKArgPassLegal(regular.physical, s.outer, s.r, s.k));

        // The winner is one of the shortlisted passes, and every shortlisted pass was timed.
        auto semantics = makeValueReductionStageSemantics(s.maximum ? CubReductionOp::Max : CubReductionOp::Min,
                                                          DensePassRole::Complete, s.r);
        const Fp32RK::Fp32RKSite candidate_site{s.outer, s.r, s.k, sms, semantics};
        const auto pool = Fp32RK::fp32RKCandidatePool(candidate_site, [&](const PhysicalKernel& p) {
            return Fp32RK::fp32RKCompiledResidency(p, s.k, semantics);
        });
        const auto generated =
            Fp32RK::fp32RKArgCandidates(pool, s.outer, s.r, s.k, std::numeric_limits<uint64_t>::max());
        const auto shortlist = Fp32RK::fp32RKRankedKernels(
            Fp32RK::fp32RKArgRankerPool(generated), candidate_site,
            Fp32RK::fp32RKShortlistLength(CubReductionTuningDepth::Regular), s.produce_value ? 2U : 1U);
        EXPECT_EQ(shortlist.size(), regular.candidate_count);
        EXPECT_NE(shortlist.end(), std::find(shortlist.begin(), shortlist.end(), regular.physical));

        // Cached: the same request and a shallower one tune nothing; the other operation is another site.
        const auto quick = Fp32RK::selectCachedArgSite(problem, site, s.maximum, s.r, stream,
                                                       std::numeric_limits<uint64_t>::max(),
                                                       CubReductionTuningDepth::Quick);
        EXPECT_EQ(tuned, cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK));
        EXPECT_EQ(regular.physical, quick.physical);
        static_cast<void>(Fp32RK::selectCachedArgSite(problem, site, !s.maximum, s.r, stream,
                                                      std::numeric_limits<uint64_t>::max(),
                                                      CubReductionTuningDepth::Quick));
        EXPECT_LT(tuned, cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK));

        // The winner computes the reduction.
        const std::vector<float> values = argValues(s.outer * s.r * s.k, s.k);
        const Tensor input = makeGpuTensor(values, {s.outer, s.r, s.k}, stream, DataType::FP32);
        expectArgPassComputes(regular.physical, input, values, s.outer, s.r, s.k, s.maximum, stream);
    }
    // VALUE MIN at an ARG site's shape is a separate selection.
    const DenseReductionProblem value_min = fp32ValueProblem({8, 4096, 16}, {1});
    const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK);
    static_cast<void>(Fp32RK::selectCachedValueSite(value_min, lastSite(value_min), CubReductionOp::Min, 4096, 1.0f,
                                                    stream, std::numeric_limits<uint64_t>::max(),
                                                    CubReductionTuningDepth::Quick));
    EXPECT_LT(before, cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK));  // 1226: and its continuations
}

// 1207: intermediate FP32 VALUE RK sites (phase 8): the selector at a final and at a non-final intermediate site,
// cached apart from the original site of the same geometry, and its winner's timed program (first stage plus, for a
// staged winner, the production continuation) reducing the partials with the pass role's semantics: no input
// transform, the finalizer and output scale on the last pass only.
TEST(CubReductionFp32RKSelector, IntermediateValueSitesSelectAndCompute) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    DenseReductionSelectionCache& cache = denseReductionSelectionCache();
    cache.clear();
    // [16, 3000, 64] over R, staged to 47 partials: [16, 47, 64] is a final intermediate RK site.
    const DenseReductionProblem original = fp32ValueProblem({16, 3000, 64}, {1});
    const DenseReductionProblem final_intermediate = advanceDenseReductionProblem(original, lastSite(original), 47);
    ASSERT_EQ(DenseReductionInputState::IntermediateAggregate, final_intermediate.input_state);
    // [8, 600, 30, 5000] over axes 1 and 3, the last staged to 3: R = 600 of [8, 600, 30 * 3] is a non-final
    // intermediate RK site (the 3 partials remain).
    const DenseReductionProblem two_axes = fp32ValueProblem({8, 600, 30, 5000}, {1, 3});
    const DenseReductionProblem middle =
        advanceDenseReductionProblem(two_axes, enumerateDenseReductionSites(two_axes).back(), 3);
    const DenseReductionSite middle_site = enumerateDenseReductionSites(middle).front();
    ASSERT_EQ(DenseReducerFamily::RK, classifyDenseReducerFamily(middle, middle_site));
    ASSERT_EQ(DensePassRole::Intermediate,
              deriveDensePassRole(middle, advanceDenseReductionProblem(middle, middle_site, 1)));
    struct Case {
        const DenseReductionProblem* problem;
        DenseReductionSite site;
        CubReductionOp op;
        uint64_t original_size;
    };
    const std::vector<Case> cases{{&final_intermediate, lastSite(final_intermediate), CubReductionOp::Mean, 3000},
                                  {&final_intermediate, lastSite(final_intermediate), CubReductionOp::L1Norm, 3000},
                                  {&final_intermediate, lastSite(final_intermediate), CubReductionOp::Min, 3000},
                                  {&middle, middle_site, CubReductionOp::SumSquares, 600 * 5000}};
    for (const Case& c : cases) {
        const DenseReductionProblem& problem = *c.problem;
        const DenseReductionProblem next = advanceDenseReductionProblem(problem, c.site, 1);
        const bool final_pass = !hasDenseReductionSites(next);
        const float scale = final_pass ? 2.0f : 1.0f;
        uint64_t outer = 1, r = problem.runs[c.site.run_index].extent, k = 1;
        for (size_t i = 0; i < problem.runs.size(); ++i) {
            if (i < c.site.run_index) outer *= problem.runs[i].extent;
            if (i > c.site.run_index) k *= problem.runs[i].extent;
        }
        SCOPED_TRACE(::testing::Message() << "outer=" << outer << " R=" << r << " K=" << k << " op "
                                          << static_cast<int>(c.op));
        const auto semantics = makeValueReductionStageSemantics(c.op, deriveDensePassRole(problem, next),
                                                                c.original_size);
        EXPECT_EQ(CubReductionStageInputTransform::Identity, semantics.input_transform);
        const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK);
        const auto selected = Fp32RK::selectCachedValueSite(problem, c.site, c.op, c.original_size, scale, stream,
                                                            std::numeric_limits<uint64_t>::max(),
                                                            CubReductionTuningDepth::Regular);
        EXPECT_LT(before, cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK));  // 1226: and its continuations
        EXPECT_EQ(4u, selected.physical.element_bytes);
        EXPECT_TRUE(Fp8RK::structurallyValid(selected.physical, outer, r, k));

        const DenseProductionPhysicalSelectionContext context(DataType::FP32, denseRKValueOperation(c.op), stream,
                                                              c.original_size, scale, c.op);
        auto isolated = Fp8RK::planIsolatedRKSite(problem, c.site, selected.physical, semantics);
        const auto program = Fp8RK::makeSelectedIsolatedBenchmark(problem, c.site, std::move(isolated),
                                                                  selected.physical, context.context(), scale,
                                                                  std::numeric_limits<uint64_t>::max(), stream);
        // Signed partials: an intermediate L1Norm / SumSquares sums them as they are.
        const std::vector<float> values = dyadic(outer * r * k, k, false);
        const Tensor input = makeGpuTensor(values, {outer, r, k}, stream, DataType::FP32);
        Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {outer, k}));
        std::optional<Tensor> scratch;
        if (program.scratch_bytes != 0)
            scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {program.scratch_bytes}));
        program.launch(input, output, scratch ? *scratch : output, stream);
        stream.synchronize();
        const auto actual = copyGpuTensorAsFloat(output, stream);
        const auto reduced = reference(values, outer, r, k, semantics);
        for (size_t i = 0; i < reduced.size(); ++i) {
            const double value = c.op == CubReductionOp::Mean
                                     ? static_cast<double>(reduced[i]) / static_cast<double>(c.original_size)
                                     : static_cast<double>(reduced[i]);
            const double expected = scale * value;
            ASSERT_NEAR(expected, actual[i], 1e-6 * std::fabs(expected) + 1e-6) << i;
        }
    }
    // The original site of the final intermediate's geometry ([16, 47, 64] original input) is a separate selection.
    const DenseReductionProblem same_geometry = fp32ValueProblem({16, 47, 64}, {1});
    const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK);
    static_cast<void>(Fp32RK::selectCachedValueSite(final_intermediate, lastSite(final_intermediate),
                                                    CubReductionOp::Sum, 3000, 1.0f, stream,
                                                    std::numeric_limits<uint64_t>::max(),
                                                    CubReductionTuningDepth::Quick));
    const uint64_t after_intermediate = cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK);
    EXPECT_LT(before, after_intermediate);
    static_cast<void>(Fp32RK::selectCachedValueSite(same_geometry, lastSite(same_geometry), CubReductionOp::Sum, 47,
                                                    1.0f, stream, std::numeric_limits<uint64_t>::max(),
                                                    CubReductionTuningDepth::Quick));
    EXPECT_LT(after_intermediate, cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK));
}

// 1211: every carried-index ARG candidate computes argmin / argmax over values with a UINT32 or UINT64 carried index
// (not monotone in the row; composed with row strides 0 and 7) through launchFp32RKCarriedArgPass, exactly.
TEST(CubReductionFp32RKSelector, EveryCarriedArgCandidateComputesTheReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> sites{
        {3, 300, 5}, {2, 1000, 64}, {4, 200, 128}, {1, 5000, 1001}, {64, 40, 256}, {1, 2, 3}, {2, 3000, 16}};
    const auto beats = [](float a, uint64_t ia, float b, uint64_t ib, bool maximum) {
        if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && (!std::isnan(b) || ia < ib);
        if (a == b) return ia < ib;
        return maximum ? a > b : a < b;
    };
    size_t launched = 0;
    for (const auto& [outer, r, k] : sites) {
        const uint64_t count = outer * r * k;
        const std::vector<float> values = argValues(count, k);
        const Tensor input = makeGpuTensor(values, {outer, r, k}, stream, DataType::FP32);
        const CubReductionGeometry geometry = CubReduction::analyzeGeometry({outer, r, k}, std::vector<uint32_t>{1});
        for (DataType index_dtype : {DataType::UINT32, DataType::UINT64}) {
            std::vector<uint64_t> carried(count);
            uint64_t h = 0x9E3779B97F4A7C15ULL;
            const uint64_t limit = index_dtype == DataType::UINT32 ? 1000000 : (uint64_t{1} << 40);
            for (uint64_t i = 0; i < count; ++i) {
                h ^= h >> 31;
                h *= 0xBF58476D1CE4E5B9ULL;
                h ^= h >> 29;
                carried[i] = (h + i) % limit;
            }
            const Tensor carried_index = makeGpuUnsignedTensor(carried, {outer, r, k}, stream, index_dtype);
            const auto candidates =
                Fp32RK::fp32RKCarriedArgCandidates(outer, r, k, std::numeric_limits<uint64_t>::max());
            ASSERT_FALSE(candidates.empty());
            for (bool maximum : {false, true}) {
                for (uint64_t stride : {0u, 7u}) {
                    for (const auto& candidate : candidates) {
                        const PhysicalKernel& p = candidate.identity;
                        SCOPED_TRACE(::testing::Message()
                                     << "outer=" << outer << " R=" << r << " K=" << k << " index "
                                     << static_cast<int>(index_dtype) << " max " << maximum << " stride " << stride
                                     << " family " << static_cast<int>(p.family) << " a=" << p.a << " b=" << p.b
                                     << " c=" << p.c << " e=" << p.e);
                        std::optional<Tensor> scratch;
                        if (candidate.scratch_bytes != 0)
                            scratch = Tensor(gpuPlacement,
                                             TensorDescriptor(DataType::UINT8, {candidate.scratch_bytes}));
                        Tensor value(gpuPlacement, TensorDescriptor(DataType::FP32, {outer, k}));
                        Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT64, {outer, k}));
                        Fp32RK::launchFp32RKCarriedArgPass(p, input, carried_index, &value, &index, geometry,
                                                           scratch ? scratch->getMemPtr<void>() : nullptr, maximum,
                                                           stride, stream);
                        stream.synchronize();
                        const auto got_value = copyGpuTensorAsFloat(value, stream);
                        const auto got_index = copyGpuTensorAsUnsigned(index, stream);
                        for (uint64_t o = 0; o < outer; ++o) {
                            for (uint64_t c = 0; c < k; ++c) {
                                uint64_t best = 0, best_index = carried[o * r * k + c];
                                for (uint64_t row = 1; row < r; ++row) {
                                    const uint64_t e = (o * r + row) * k + c;
                                    const uint64_t composed = carried[e] + row * stride;
                                    if (beats(values[e], composed, values[(o * r + best) * k + c], best_index,
                                              maximum)) {
                                        best = row;
                                        best_index = composed;
                                    }
                                }
                                const uint64_t i = o * k + c;
                                const float want = values[(o * r + best) * k + c];
                                ASSERT_EQ(best_index, got_index[i]) << "outer " << o << " k " << c;
                                if (std::isnan(want)) {
                                    ASSERT_TRUE(std::isnan(got_value[i])) << "outer " << o << " k " << c;
                                } else {
                                    ASSERT_EQ(std::bit_cast<uint32_t>(want), std::bit_cast<uint32_t>(got_value[i]))
                                        << "outer " << o << " k " << c;
                                }
                            }
                        }
                        ++launched;
                    }
                }
            }
        }
    }
    EXPECT_GT(launched, 100u);
}

namespace {

// 1215: an ARG problem over `axes` whose index payload is `index_dtype`.
DenseReductionProblem fp32CarriedArgProblem(const std::vector<uint64_t>& dims, const std::vector<uint32_t>& axes,
                                            bool produce_value, DataType index_dtype) {
    DenseReductionOutputSpec output;
    output.produce_value = produce_value;
    output.value_dtype = DataType::FP32;
    output.produce_index = true;
    output.index_dtype = DataType::UINT64;
    return makeInitialDenseReductionProblem(dims, axes, DenseReductionAggregateKind::Arg, output, index_dtype);
}

// Carried indices giving distinct composed indices per output (as an earlier pass's partials do): at stride 0 distinct
// values (above 2^32 for UINT64), otherwise values below the stride.
std::vector<uint64_t> carriedIndicesFor(uint64_t count, uint64_t stride, DataType index_dtype) {
    std::vector<uint64_t> carried(count);
    uint64_t h = 0x2545F4914F6CDD1DULL;
    for (uint64_t i = 0; i < count; ++i) {
        h ^= h >> 31;
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 29;
        carried[i] = stride == 0 ? ((i * 0x9E3779B1ULL + 12345) & 0xFFFFFFFFULL) +
                                       (index_dtype == DataType::UINT64 ? uint64_t{1} << 40 : 0)
                                 : h % stride;
    }
    return carried;
}

// Runs a carried RK ARG pass and checks the value bits and the composed index (UINT64 output) exactly.
void expectCarriedArgPassComputes(const PhysicalKernel& identity, uint64_t outer, uint64_t r, uint64_t k,
                                  DataType index_dtype, bool maximum, uint64_t stride, Stream& stream) {
    const uint64_t count = outer * r * k;
    const std::vector<float> values = argValues(count, k);
    const std::vector<uint64_t> carried = carriedIndicesFor(count, stride, index_dtype);
    const Tensor input = makeGpuTensor(values, {outer, r, k}, stream, DataType::FP32);
    const Tensor carried_index = makeGpuUnsignedTensor(carried, {outer, r, k}, stream, index_dtype);
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry({outer, r, k}, std::vector<uint32_t>{1});
    const uint64_t scratch_bytes = Fp32RK::fp32RKCarriedArgScratchBytes(identity, outer, r, k);
    std::optional<Tensor> scratch;
    if (scratch_bytes != 0) scratch = Tensor(gpuPlacement, TensorDescriptor(DataType::UINT8, {scratch_bytes}));
    Tensor value(gpuPlacement, TensorDescriptor(DataType::FP32, {outer, k}));
    Tensor index(gpuPlacement, TensorDescriptor(DataType::UINT64, {outer, k}));
    Fp32RK::launchFp32RKCarriedArgPass(identity, input, carried_index, &value, &index, geometry,
                                       scratch ? scratch->getMemPtr<void>() : nullptr, maximum, stride, stream);
    stream.synchronize();
    const auto got_value = copyGpuTensorAsFloat(value, stream);
    const auto got_index = copyGpuTensorAsUnsigned(index, stream);
    const auto beats = [&](float a, uint64_t ia, float b, uint64_t ib) {
        if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && (!std::isnan(b) || ia < ib);
        if (a == b) return ia < ib;
        return maximum ? a > b : a < b;
    };
    for (uint64_t o = 0; o < outer; ++o) {
        for (uint64_t c = 0; c < k; ++c) {
            uint64_t best = 0, best_index = carried[o * r * k + c];
            for (uint64_t row = 1; row < r; ++row) {
                const uint64_t e = (o * r + row) * k + c;
                const uint64_t composed = carried[e] + row * stride;
                if (beats(values[e], composed, values[(o * r + best) * k + c], best_index)) {
                    best = row;
                    best_index = composed;
                }
            }
            const uint64_t i = o * k + c;
            const float want = values[(o * r + best) * k + c];
            ASSERT_EQ(best_index, got_index[i]) << "outer " << o << " k " << c;
            if (std::isnan(want)) {
                ASSERT_TRUE(std::isnan(got_value[i])) << "outer " << o << " k " << c;
            } else {
                ASSERT_EQ(std::bit_cast<uint32_t>(want), std::bit_cast<uint32_t>(got_value[i]))
                    << "outer " << o << " k " << c;
            }
        }
    }
}

}  // namespace

// 1215: the carried-index ARG RK selector (phase 8): its domain; at a final site over an intermediate run (stride 0)
// and a non-final site over an original run (its original-domain stride), with UINT32 and UINT64 carried indices: a
// shortlisted winner (the carried candidates ranked with the carried aggregate), cached under kind Fp32RK apart from
// the original ARG site and from the other operation, and the winner computing the reduction. Then the explicit-input
// autotune and its input checks.
TEST(CubReductionFp32RKSelector, CarriedArgSitesSelectCacheAndCompute) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const uint32_t sms = denseReductionDeviceIdentity(0).sm_count;
    DenseReductionSelectionCache& cache = denseReductionSelectionCache();
    cache.clear();
    for (const DataType index_dtype : {DataType::UINT32, DataType::UINT64}) {
        // [16, 3000, 64] over R, staged to 47 partials: [16, 47, 64] is a final carried site (an intermediate run).
        const DenseReductionProblem staged = fp32CarriedArgProblem({16, 3000, 64}, {1}, true, index_dtype);
        const DenseReductionProblem final_carried = advanceDenseReductionProblem(staged, lastSite(staged), 47);
        // [8, 600, 30, 5000] over axes 1 and 3, the last staged to 3: R = 600 of [8, 600, 30 * 3] is a non-final
        // carried site over an original run.
        const DenseReductionProblem two_axes = fp32CarriedArgProblem({8, 600, 30, 5000}, {1, 3}, false, index_dtype);
        const DenseReductionProblem middle =
            advanceDenseReductionProblem(two_axes, enumerateDenseReductionSites(two_axes).back(), 3);
        const DenseReductionSite middle_site = enumerateDenseReductionSites(middle).front();
        ASSERT_EQ(DenseReductionInputState::IntermediateAggregate, final_carried.input_state);
        ASSERT_EQ(DenseReductionRunKind::IntermediateReduction,
                  final_carried.runs[lastSite(final_carried).run_index].kind);
        ASSERT_EQ(DenseReductionRunKind::OriginalReduction, middle.runs[middle_site.run_index].kind);
        ASSERT_EQ(DenseReducerFamily::RK, classifyDenseReducerFamily(middle, middle_site));
        // Outside the domain: an original ARG site for the carried selector, a carried site for the original one.
        EXPECT_THROW(static_cast<void>(Fp32RK::selectCachedCarriedArgSite(staged, lastSite(staged), false, 3000,
                                                                          stream)),
                     std::invalid_argument);
        EXPECT_THROW(static_cast<void>(Fp32RK::selectCachedArgSite(final_carried, lastSite(final_carried), false,
                                                                   3000, stream)),
                     std::invalid_argument);
        struct Case {
            const DenseReductionProblem* problem;
            DenseReductionSite site;
            uint64_t original_size;
            bool maximum;
        };
        const std::vector<Case> cases{{&final_carried, lastSite(final_carried), 3000, false},
                                      {&middle, middle_site, 600 * 5000, true}};
        for (const Case& c : cases) {
            const DenseReductionProblem& problem = *c.problem;
            const bool final_pass = !hasDenseReductionSites(advanceDenseReductionProblem(problem, c.site, 1));
            uint64_t outer = 1, r = problem.runs[c.site.run_index].extent, k = 1;
            for (size_t i = 0; i < problem.runs.size(); ++i) {
                if (i < c.site.run_index) outer *= problem.runs[i].extent;
                if (i > c.site.run_index) k *= problem.runs[i].extent;
            }
            const uint64_t stride = denseArgOriginalIndexStride(problem.runs[c.site.run_index]).value_or(0);
            SCOPED_TRACE(::testing::Message() << "outer=" << outer << " R=" << r << " K=" << k << " index "
                                              << static_cast<int>(index_dtype) << " stride " << stride << " final "
                                              << final_pass);
            const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK);
            const auto regular = Fp32RK::selectCachedCarriedArgSite(problem, c.site, c.maximum, c.original_size,
                                                                    stream, std::numeric_limits<uint64_t>::max(),
                                                                    CubReductionTuningDepth::Regular);
            EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK));
            ASSERT_NE(0u, regular.candidate_count);
            EXPECT_EQ(4u, regular.physical.element_bytes);
            EXPECT_TRUE(Fp32RK::fp32RKCarriedArgPassLegal(regular.physical, outer, r, k));

            // The winner is one of the shortlisted carried passes, and every shortlisted pass was timed.
            auto semantics = makeValueReductionStageSemantics(c.maximum ? CubReductionOp::Max : CubReductionOp::Min,
                                                              DensePassRole::Complete, r);
            semantics.combine = c.maximum ? CubReductionStageCombine::Maximum : CubReductionStageCombine::Minimum;
            semantics.input_transform = CubReductionStageInputTransform::Identity;
            const Fp32RK::Fp32RKSite candidate_site{outer, r, k, sms, semantics};
            const bool produce_value = !final_pass || problem.final_output.produce_value;
            const auto shortlist = Fp32RK::fp32RKRankedKernels(
                Fp32RK::fp32RKArgRankerPool(
                    Fp32RK::fp32RKCarriedArgCandidates(outer, r, k, std::numeric_limits<uint64_t>::max())),
                candidate_site, Fp32RK::fp32RKShortlistLength(CubReductionTuningDepth::Regular),
                produce_value ? 4U : 3U);
            EXPECT_EQ(shortlist.size(), regular.candidate_count);
            EXPECT_NE(shortlist.end(), std::find(shortlist.begin(), shortlist.end(), regular.physical));

            // Cached: the same request and a shallower one tune nothing; the other operation is another site.
            const auto quick = Fp32RK::selectCachedCarriedArgSite(problem, c.site, c.maximum, c.original_size, stream,
                                                                  std::numeric_limits<uint64_t>::max(),
                                                                  CubReductionTuningDepth::Quick);
            EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK));
            EXPECT_EQ(regular.physical, quick.physical);
            static_cast<void>(Fp32RK::selectCachedCarriedArgSite(problem, c.site, !c.maximum, c.original_size, stream,
                                                                 std::numeric_limits<uint64_t>::max(),
                                                                 CubReductionTuningDepth::Quick));
            EXPECT_EQ(before + 2, cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK));

            // The winner computes the reduction with the site's stride.
            expectCarriedArgPassComputes(regular.physical, outer, r, k, index_dtype, c.maximum, stride, stream);
        }
        // The original ARG site of the final carried site's geometry is a separate selection.
        const DenseReductionProblem same_geometry = fp32CarriedArgProblem({16, 47, 64}, {1}, true, index_dtype);
        const uint64_t before = cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK);
        static_cast<void>(Fp32RK::selectCachedArgSite(same_geometry, lastSite(same_geometry), false, 47, stream,
                                                      std::numeric_limits<uint64_t>::max(),
                                                      CubReductionTuningDepth::Quick));
        EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK));

        // The explicit-input autotune (no cache) over real values and carried indices; inputs of the wrong dtype or
        // size are rejected.
        const uint64_t count = 8 * 600 * 90;
        const Tensor values = makeGpuTensor(argValues(count, 90), {8, 600, 90}, stream, DataType::FP32);
        const Tensor carried = makeGpuUnsignedTensor(carriedIndicesFor(count, 150000, index_dtype), {8, 600, 90},
                                                     stream, index_dtype);
        const auto direct = Fp32RK::autotuneCarriedArgSite(middle, middle_site, false, 600 * 5000, values, carried,
                                                           stream, std::numeric_limits<uint64_t>::max(),
                                                           CubReductionTuningDepth::Quick);
        EXPECT_EQ(before + 1, cache.tuningAttempts(DenseReductionSelectionKind::Fp32RK));
        ASSERT_NE(0u, direct.candidate_count);
        EXPECT_TRUE(Fp32RK::fp32RKCarriedArgPassLegal(direct.physical, 8, 600, 90));
        const DataType other = index_dtype == DataType::UINT32 ? DataType::UINT64 : DataType::UINT32;
        const Tensor wrong_dtype =
            makeGpuUnsignedTensor(carriedIndicesFor(count, 150000, other), {8, 600, 90}, stream, other);
        EXPECT_THROW(static_cast<void>(Fp32RK::autotuneCarriedArgSite(middle, middle_site, false, 600 * 5000, values,
                                                                      wrong_dtype, stream)),
                     std::invalid_argument);
        EXPECT_THROW(static_cast<void>(Fp32RK::autotuneCarriedArgSite(middle, middle_site, false, 600 * 5000, carried,
                                                                      carried, stream)),
                     std::invalid_argument);
    }
}
