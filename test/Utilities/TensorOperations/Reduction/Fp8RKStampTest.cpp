// The contract of a stamped FP8 RK VALUE reduction, on the GPU:
// - selection happens at stamp time only: the first stamp of a site tunes once, an identical stamp is a cache hit with
//   the same kernel, running a stamp never consults the cache, and a stamp outlives a cache clear;
// - every stamp owns its intermediates and scratch, and its workspace equals what its passes own and what its
//   diagnostics report, and what a descriptor-only query returns;
// - a runtime output scale replaces the stamp's scale and applies once, on the final pass;
// - MEAN and an FP16 output run on the FP8 passes;
// - an injected plan (a chosen first pass, then the production FP32 selector) materializes like a tuned one;
// - the descriptor and Tensor selection requests share one cached recipe.

#include "Utilities/TensorOperations/Reduction/Planner/DensePhysicalPassExecution.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Reduction/RK/Fp8/Fp8RKStampAutotuner.h"
#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/Reducer.h"
#include "test/Utilities/TensorOperations/Reduction/ReductionTestSupport.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation {
// Test-only access to a stamp's passes, and the stamp constructor for an injected plan (a friend of
// StampedReducer).
struct Fp8RKStampProbe {
    static const std::vector<ReductionInternal::DenseExecutablePass>& passes(const StampedReducer& stamp) {
        return stamp.dense_executable_plan.passes;
    }
    static bool ownsStampScratch(const StampedReducer& stamp) { return stamp.temp_storage.isInitialized(); }
    static std::shared_ptr<StampedReducer> stamp(const Tensor& input, const Tensor& output,
                                                 const std::vector<uint32_t>& axes,
                                                 const ReductionInternal::DenseReductionPlan& plan,
                                                 uint64_t total_reduction, Stream& stream, ReductionOp op) {
        ReductionInternal::DenseExecutablePlan passes = ReductionInternal::materializeDenseValueExecutablePlan(
            op, input, output, plan, total_reduction, 1.0f, stream);
        return std::shared_ptr<StampedReducer>(new StampedReducer(
            op, Reducer::analyzeGeometry(input.getDimensions(), axes), input, output, std::move(passes),
            "injected physical plan", true, true, total_reduction, 1.0f, stream));
    }
};
}  // namespace ThorImplementation

using namespace ThorImplementation;
using namespace ThorImplementation::ReductionTestSupport;
namespace C = ThorImplementation::ReductionInternal;
namespace F = ThorImplementation::ReductionInternal::RK;

namespace {

constexpr uint64_t O = 16, R = 512, K = 64;
constexpr DataType Fp8Formats[] = {DataType::FP8_E4M3, DataType::FP8_E5M2};

C::DenseReductionSelectionCache& cache() { return C::denseReductionSelectionCache(); }
uint64_t rkAttempts() { return cache().tuningAttempts(C::DenseReductionSelectionKind::Fp8RK); }

Tensor filled(DataType dtype, const std::vector<uint64_t>& dims, float value, Stream& stream) {
    Tensor tensor(gpuPlacement, TensorDescriptor(dtype, dims));
    tensor.fill(value, stream);
    return tensor;
}

void expectAll(const Tensor& output, float expected, Stream& stream, float tolerance = 1e-3f) {
    for (const float value : copyGpuTensorAsFloat(output, stream)) {
        ASSERT_TRUE(std::isfinite(value));
        ASSERT_NEAR(value, expected, tolerance);
    }
}

const F::PhysicalKernel& firstKernel(const StampedReducer& stamp) {
    const auto& passes = Fp8RKStampProbe::passes(stamp);
    if (passes.empty()) throw std::logic_error("the stamp has no passes");
    const auto* first = dynamic_cast<const C::RKPhysicalPlan*>(passes.front().physical_plan.get());
    if (first == nullptr) throw std::logic_error("the first pass is not a retained FP8 RK pass");
    return first->kernel;
}

// The stamp's workspace is what its passes own (scratch allocated exactly when queried) and what its diagnostics
// report.
void expectOwnedWorkspace(const StampedReducer& stamp) {
    ASSERT_TRUE(stamp.usesDenseExecutablePlan());
    const auto& passes = Fp8RKStampProbe::passes(stamp);
    size_t owned = 0;
    for (const auto& pass : passes) {
        EXPECT_EQ(pass.temp_storage.isInitialized(), pass.temp_storage_bytes != 0);
        EXPECT_TRUE(pass.intermediate_bytes != 0 || pass.output.isInitialized());
        owned += pass.workspaceBytes();
    }
    EXPECT_EQ(owned, stamp.getWorkspaceSizeInBytes());
    const auto diagnostics = stamp.getDensePhysicalPassDiagnostics();
    ASSERT_EQ(diagnostics.size(), passes.size());
    size_t described = 0;
    for (size_t i = 0; i < diagnostics.size(); ++i) {
        const auto& pass = diagnostics[i];
        described += pass.intermediate_bytes + pass.scratch_bytes;
        EXPECT_FALSE(pass.family.empty());
        EXPECT_FALSE(pass.input_dimensions.empty());
        EXPECT_FALSE(pass.output_dimensions.empty());
        if (pass.intermediate_bytes != 0) EXPECT_EQ(pass.output_dtype, DataType::FP32);
        if (i + 1 < diagnostics.size()) EXPECT_EQ(pass.output_dtype, diagnostics[i + 1].input_dtype);
    }
    EXPECT_EQ(described, owned);
}

// Two live stamps share no mutable buffer, whatever their pass counts.
void expectNoSharedBuffers(const StampedReducer& a, const StampedReducer& b) {
    const auto& left = Fp8RKStampProbe::passes(a);
    const auto& right = Fp8RKStampProbe::passes(b);
    EXPECT_NE(left.data(), right.data());
    for (const auto& l : left) {
        for (const auto& r : right) {
            if (l.intermediate_bytes != 0 && r.intermediate_bytes != 0)
                EXPECT_NE(l.output.getMemPtr<void>(), r.output.getMemPtr<void>());
            if (l.temp_storage_bytes != 0 && r.temp_storage_bytes != 0)
                EXPECT_NE(l.temp_storage.getMemPtr<void>(), r.temp_storage.getMemPtr<void>());
        }
    }
}

}  // namespace

TEST(Fp8RKStamp, SelectionHappensAtStampTimeOnly) {
    REQUIRE_CUDA_DEVICE();
    for (const DataType dtype : Fp8Formats) {
        SCOPED_TRACE(static_cast<int>(dtype));
        Stream stream_a(0), stream_b(0);
        Tensor input_a = filled(dtype, {O, R, K}, 1.0f, stream_a);
        Tensor input_b = filled(dtype, {O, R, K}, -0.5f, stream_b);
        Tensor output_a(gpuPlacement, TensorDescriptor(DataType::FP32, {O, K}));
        Tensor output_b(gpuPlacement, TensorDescriptor(DataType::FP32, {O, K}));
        stream_a.synchronize();  // nothing else on the GPU during the contest
        stream_b.synchronize();
        const Reducer sum(ReductionOp::Sum, {1}, DataType::FP32);

        cache().clear();
        const uint64_t before = rkAttempts();
        auto a = sum.stamp(input_a, output_a, stream_a);
        // One FP8 RK contest. The cache also holds the FP32 selections that contest tuned for its staged candidates'
        // continuations, so its size is not 1.
        EXPECT_EQ(rkAttempts(), before + 1);
        EXPECT_EQ(a->getModernRKStrategyChain().find("retained_fp8_rk:"), 0u);
        auto b = sum.stamp(input_b, output_b, stream_b);
        EXPECT_EQ(rkAttempts(), before + 1);  // a hit: the same kernel and workspace
        EXPECT_TRUE(firstKernel(*a) == firstKernel(*b));
        EXPECT_EQ(a->getModernRKStrategyChain(), b->getModernRKStrategyChain());
        EXPECT_EQ(a->getWorkspaceSizeInBytes(), b->getWorkspaceSizeInBytes());
        expectOwnedWorkspace(*a);
        expectOwnedWorkspace(*b);
        expectNoSharedBuffers(*a, *b);

        // Running live stamps, concurrently on two streams, neither reads nor refills a cleared cache.
        cache().clear();
        const uint64_t before_runs = rkAttempts();
        auto run_a = std::async(std::launch::async, [&] {
            for (int i = 0; i < 16; ++i) a->runOn(stream_a);
            return copyGpuTensorAsFloat(output_a, stream_a);
        });
        auto run_b = std::async(std::launch::async, [&] {
            for (int i = 0; i < 16; ++i) b->runOn(stream_b);
            return copyGpuTensorAsFloat(output_b, stream_b);
        });
        for (const float v : run_a.get()) EXPECT_NEAR(v, static_cast<float>(R), 1e-3f);
        for (const float v : run_b.get()) EXPECT_NEAR(v, -0.5f * static_cast<float>(R), 1e-3f);
        EXPECT_EQ(cache().size(), 0u);
        EXPECT_EQ(rkAttempts(), before_runs);

        // A new stamp after the clear tunes afresh (it may pick another kernel); the old stamps keep working, and
        // dropping one leaves the others intact.
        Tensor output_c(gpuPlacement, TensorDescriptor(DataType::FP32, {O, K}));
        auto c = sum.stamp(input_a, output_c, stream_a);
        EXPECT_EQ(rkAttempts(), before_runs + 1);
        expectNoSharedBuffers(*a, *c);
        expectNoSharedBuffers(*b, *c);
        b->runOn(stream_b);
        c->runOn(stream_a);
        expectAll(output_b, -0.5f * static_cast<float>(R), stream_b);
        expectAll(output_c, static_cast<float>(R), stream_a);
        const uint64_t settled = rkAttempts();
        b.reset();
        a->runOn(stream_a);
        expectAll(output_a, static_cast<float>(R), stream_a);
        // An identical stamp hits the refilled cache.
        auto d = sum.stamp(input_a, output_c, stream_a);
        EXPECT_EQ(rkAttempts(), settled);
        d->runOn(stream_a);
        expectAll(output_c, static_cast<float>(R), stream_a);
    }
}

TEST(Fp8RKStamp, QueryDiagnosticsAndRuntimeScaleAgreeWithTheStamp) {
    REQUIRE_CUDA_DEVICE();
    for (const DataType dtype : Fp8Formats) {
        for (const bool multi_axis : {false, true}) {
            SCOPED_TRACE(std::to_string(static_cast<int>(dtype)) + (multi_axis ? " multi-axis" : " one axis"));
            Stream stream(0), other(0);
            const std::vector<uint64_t> shape =
                multi_axis ? std::vector<uint64_t>{512, 2, 64} : std::vector<uint64_t>{16, 512, 64};
            const std::vector<uint32_t> axes = multi_axis ? std::vector<uint32_t>{0, 2} : std::vector<uint32_t>{1};
            const std::vector<uint64_t> output_shape =
                multi_axis ? std::vector<uint64_t>{1, 2, 1} : std::vector<uint64_t>{16, 1, 64};
            const float expected = multi_axis ? 512.0f * 64.0f * 0.25f : 512.0f * 0.25f;
            Tensor input = filled(dtype, shape, 0.25f, stream);
            Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, output_shape));
            stream.synchronize();

            const Reducer sum(ReductionOp::Sum, axes, DataType::FP32);
            const size_t queried = sum.queryWorkspaceSizeInBytes(TensorDescriptor(dtype, shape), stream);
            auto stamp = sum.stamp(input, output, stream);
            EXPECT_EQ(stamp->getWorkspaceSizeInBytes(), queried);
            expectOwnedWorkspace(*stamp);
            const auto passes = stamp->getDensePhysicalPassDiagnostics();
            ASSERT_FALSE(passes.empty());
            if (multi_axis) {
                EXPECT_GE(passes.size(), 2u);
            } else {
                EXPECT_EQ(passes.front().family.find("retained_fp8_rk:"), 0u);
                EXPECT_EQ(passes.front().scratch_bytes, 0u);
            }

            // The runtime scale replaces the stamp's and applies once, on another stream too; no run selects.
            cache().clear();
            const uint64_t attempts = rkAttempts();
            for (int i = 0; i < 3; ++i) stamp->runOn(other);
            expectAll(output, expected, other, 1e-2f);
            stamp->runOn(other, -0.5f);
            expectAll(output, -0.5f * expected, other, 1e-2f);
            stamp->runOn(other);
            expectAll(output, expected, other, 1e-2f);
            EXPECT_EQ(cache().size(), 0u);
            EXPECT_EQ(rkAttempts(), attempts);

            const Reducer scaled_sum(ReductionOp::Sum, axes, DataType::FP32, 2.0f);
            auto scaled = scaled_sum.stamp(input, output, stream);
            EXPECT_EQ(scaled_sum.queryWorkspaceSizeInBytes(TensorDescriptor(dtype, shape), stream),
                      scaled->getWorkspaceSizeInBytes());
            scaled->runOn(other);
            expectAll(output, 2.0f * expected, other, 1e-2f);
            scaled->runOn(other, -1.5f);
            expectAll(output, -1.5f * expected, other, 1e-2f);
        }
    }
}

TEST(Fp8RKStamp, MeanAndFp16OutputRunOnTheRetainedPasses) {
    REQUIRE_CUDA_DEVICE();
    for (const DataType dtype : Fp8Formats) {
        SCOPED_TRACE(static_cast<int>(dtype));
        Stream stream(0);
        Tensor input = filled(dtype, {O, R, K}, 0.25f, stream);
        Tensor mean_output(gpuPlacement, TensorDescriptor(DataType::FP32, {O, 1, K}));
        Tensor half_output(gpuPlacement, TensorDescriptor(DataType::FP16, {O, 1, K}));
        auto mean = Reducer(ReductionOp::Mean, {1}, DataType::FP32).stamp(input, mean_output, stream);
        auto half = Reducer(ReductionOp::Sum, {1}, DataType::FP16).stamp(input, half_output, stream);
        EXPECT_EQ(mean->getModernRKStrategyChain().find("retained_fp8_rk:"), 0u);
        EXPECT_EQ(half->getModernRKStrategyChain().find("retained_fp8_rk:"), 0u);
        mean->runOn(stream);
        expectAll(mean_output, 0.25f, stream);  // finalized once, by the original R
        half->runOn(stream);
        expectAll(half_output, 0.25f * static_cast<float>(R), stream, 0.01f);  // converted once, at the end
    }
}

// A chosen first pass, with every other site from the production selector, stamps and runs like a tuned plan: each
// stamp owns its intermediates and scratch, the first pass writes FP32 with no scratch, and running selects nothing.
TEST(Fp8RKStamp, AnInjectedFirstPassMaterializesLikeATunedOne) {
    REQUIRE_CUDA_DEVICE();
    struct Choice {
        C::DensePhysicalSelectionContext production;
        F::PhysicalKernel first;
        unsigned retained = 0;
        unsigned other = 0;
        static C::DenseReductionCandidate propose(const C::DenseReductionProblem& problem, C::DenseReductionSite site,
                                                  C::DenseReducerFamily family, const C::DenseRKPlanningContext&,
                                                  void* opaque) {
            auto& choice = *static_cast<Choice*>(opaque);
            if (problem.input_state == C::DenseReductionInputState::OriginalInput &&
                family == C::DenseReducerFamily::RK) {
                ++choice.retained;
                return C::makeRetainedFp8RKPhysicalCandidate(problem, site, choice.first, choice.production);
            }
            ++choice.other;
            return C::selectDensePhysicalPass(problem, site, choice.production);
        }
    };
    enum class First { Complete, Staged, PacketComplete };
    struct Case {
        First first;
        bool multi_axis;
        ReductionOp op;
    };
    std::vector<Case> cases;
    for (const bool multi_axis : {false, true}) {
        cases.push_back({First::Complete, multi_axis, ReductionOp::Sum});
        cases.push_back({First::Staged, multi_axis, ReductionOp::Sum});
        for (const ReductionOp op : {ReductionOp::Mean, ReductionOp::Product, ReductionOp::L2Norm,
                                     ReductionOp::Min, ReductionOp::Max, ReductionOp::L1Norm,
                                     ReductionOp::SumSquares}) {
            cases.push_back({First::Staged, multi_axis, op});
        }
    }
    cases.push_back({First::PacketComplete, false, ReductionOp::Sum});

    for (const DataType dtype : Fp8Formats) {
        for (const Case& c : cases) {
            SCOPED_TRACE(std::to_string(static_cast<int>(dtype)) + " first " +
                         std::to_string(static_cast<int>(c.first)) + (c.multi_axis ? " multi-axis" : " one axis") +
                         " op " + std::to_string(static_cast<int>(c.op)));
            Stream stream(0), other(0);
            const bool staged = c.first == First::Staged;
            const uint64_t r = staged ? 2048 : (c.multi_axis ? 64 : 16);
            const std::vector<uint64_t> dims = c.multi_axis ? std::vector<uint64_t>{r, 2, 16}
                                                            : std::vector<uint64_t>{r, 32};
            const std::vector<uint32_t> axes = c.multi_axis ? std::vector<uint32_t>{0, 2} : std::vector<uint32_t>{0};
            const uint64_t total = r * (c.multi_axis ? 16 : 1);
            C::DenseReductionOutputSpec final_output;
            final_output.value_dtype = DataType::FP32;
            const auto problem =
                C::makeInitialDenseReductionProblem(dims, axes, C::DenseReductionAggregateKind::Value, final_output);
            const C::DenseProductionPhysicalSelectionContext production(dtype, C::denseRKValueOperation(c.op), stream,
                                                                        total, 1.0f, c.op);
            const uint64_t shards = r / 128 + (r % 128 != 0);
            F::PhysicalKernel first{F::PhysicalFamily::OneWindow, 1, 1, 16, 0, 0, 0, 6, 1, 0};
            if (staged) first = {F::PhysicalFamily::DeepRSmallK, 128, 0, 0, 0, 0, 0, 6, shards, shards * 32 * 4};
            if (c.first == First::PacketComplete) {
                first = {F::PhysicalFamily::RuntimeK, static_cast<uint64_t>(F::RuntimeLayout::PacketComplete), 1, 32,
                         1, r, 1, 6, 1, 0};
            }
            Choice choice{production.context(), first};
            C::DensePhysicalSelectionContext injected = choice.production;
            injected.proposal_override = Choice::propose;
            injected.proposal_override_context = &choice;
            const auto plan = C::planDenseReduction(problem, injected);
            ASSERT_FALSE(plan.steps.empty());
            const auto* retained = dynamic_cast<const C::RKPhysicalPlan*>(
                plan.steps.front().selected_candidate.physical_plan.get());
            ASSERT_NE(retained, nullptr);
            EXPECT_EQ(choice.retained, 1u);
            if (staged) {
                ASSERT_GE(plan.steps.size(), 2u);
                EXPECT_EQ(retained->output_dimensions, (std::vector<uint64_t>{1, 16, 32}));
                EXPECT_TRUE(plan.steps[1].input_problem == plan.steps[0].selected_candidate.next_problem);
                if (c.multi_axis) EXPECT_EQ(plan.steps[1].selected_candidate.family, C::DenseReducerFamily::KR);
            }
            EXPECT_FALSE(C::hasDenseReductionSites(plan.final_problem));

            Tensor input = filled(dtype, dims, 1.0f, stream);
            stream.synchronize();
            const auto geometry = Reducer::analyzeGeometry(dims, axes);
            Tensor result_a(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
            Tensor result_b(gpuPlacement, TensorDescriptor(DataType::FP32, geometry.output_dimensions));
            auto stamp_a = Fp8RKStampProbe::stamp(input, result_a, axes, plan, total, stream, c.op);
            auto stamp_b = Fp8RKStampProbe::stamp(input, result_b, axes, plan, total, stream, c.op);
            EXPECT_EQ(stamp_a->getDenseExecutablePassCount(), plan.steps.size());  // no hidden passes
            EXPECT_FALSE(Fp8RKStampProbe::ownsStampScratch(*stamp_a));
            EXPECT_FALSE(Fp8RKStampProbe::ownsStampScratch(*stamp_b));
            expectOwnedWorkspace(*stamp_a);
            expectNoSharedBuffers(*stamp_a, *stamp_b);
            const auto& passes = Fp8RKStampProbe::passes(*stamp_a);
            EXPECT_EQ(passes[0].temp_storage_bytes, 0u);
            EXPECT_EQ(passes[0].output.getDataType(), DataType::FP32);
            if (staged) {
                EXPECT_EQ(passes[0].output.getDimensions(), retained->output_dimensions);
                EXPECT_EQ(passes[0].intermediate_bytes, 16u * 32u * sizeof(float));
            }
            // The continuation is a separate pass that reads the FP32 partials (an RK continuation is the same RK plan
            // type, at element width 4).
            if (passes.size() > 1) EXPECT_EQ(passes[1].input.getDataType(), DataType::FP32);

            const unsigned proposals = choice.retained + choice.other;
            stamp_a->runOn(stream);
            stamp_b->runOn(other);
            EXPECT_EQ(choice.retained + choice.other, proposals);
            // All ones: MEAN divides by the original domain once; L2 takes its root after every FP32 pass.
            const float expected = c.op == ReductionOp::Sum || c.op == ReductionOp::L1Norm ||
                                           c.op == ReductionOp::SumSquares
                                       ? static_cast<float>(total)
                                   : c.op == ReductionOp::L2Norm ? std::sqrt(static_cast<float>(total))
                                                                     : 1.0f;
            expectAll(result_a, expected, stream, 0.1f);
            expectAll(result_b, expected, other, 0.1f);
            stamp_a->runOn(stream, -0.5f);
            expectAll(result_a, -0.5f * expected, stream, 0.1f);
        }
    }
}

TEST(Fp8RKStamp, DescriptorAndTensorRequestsShareOneRecipe) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t budget = uint64_t{256} * 1024 * 1024;
    C::DenseReductionOutputSpec final_output;
    final_output.value_dtype = DataType::FP32;
    const auto problem =
        C::makeInitialDenseReductionProblem({O, R, K}, {1}, C::DenseReductionAggregateKind::Value, final_output);
    const auto site = C::enumerateDenseReductionSites(problem).at(0);
    for (const DataType dtype : Fp8Formats) {
        SCOPED_TRACE(static_cast<int>(dtype));
        const C::DenseProductionPhysicalSelectionContext context(dtype, C::DenseRKValueOperation::Sum, stream, R, 1.0f,
                                                                 ReductionOp::Sum, budget);
        EXPECT_TRUE(context.context().original_value_operation == ReductionOp::Sum);
        EXPECT_EQ(context.context().original_reduction_size, R);
        EXPECT_EQ(context.context().output_scale, 1.0f);
        EXPECT_EQ(context.context().max_workspace_bytes, budget);

        cache().clear();
        const uint64_t before = rkAttempts();
        const auto miss = F::Fp8::selectCachedIsolatedRKSite(problem, site, ReductionOp::Sum, R, 1.0f, dtype, stream,
                                                             budget);
        EXPECT_EQ(rkAttempts(), before + 1);
        EXPECT_GT(miss.candidate_count, 0u);
        EXPECT_LE(miss.candidate_count, 28u);
        EXPECT_LT(miss.winner_index, miss.candidate_count);
        const auto hit = F::Fp8::selectCachedIsolatedRKSite(problem, site, ReductionOp::Sum, R, 1.0f, dtype, stream,
                                                            budget);
        EXPECT_TRUE(hit.physical == miss.physical);
        Tensor input = filled(dtype, {O, R, K}, 1.0f, stream);
        const auto tensor_hit = F::Fp8::selectCachedIsolatedRKSite(problem, site, ReductionOp::Sum, R, 1.0f, input,
                                                                   stream, budget);
        EXPECT_TRUE(tensor_hit.physical == miss.physical);
        EXPECT_EQ(tensor_hit.winner_index, miss.winner_index);
        EXPECT_EQ(rkAttempts(), before + 1);

        // Invalid requests are rejected even with a related key cached.
        EXPECT_THROW((void)F::Fp8::selectCachedIsolatedRKSite(problem, site, ReductionOp::Sum, 0, 1.0f, dtype, stream,
                                                              budget),
                     std::invalid_argument);
        EXPECT_THROW((void)F::Fp8::selectCachedIsolatedRKSite(problem, site, ReductionOp::Sum, R,
                                                              std::numeric_limits<float>::quiet_NaN(), dtype, stream,
                                                              budget),
                     std::invalid_argument);
        EXPECT_THROW((void)F::Fp8::selectCachedIsolatedRKSite(problem, site, ReductionOp::Sum, R, 1.0f, DataType::FP32,
                                                              stream, budget),
                     std::invalid_argument);
    }
}
