// The ambient stamp-time tuning depth and the depth rules of the FP8 selectors in the selection cache.
//
// Host: a reduction constructed without an explicit depth takes this thread's innermost ScopedReductionTuningDepth,
// else the process default (Regular unless set); an explicit depth always wins; scopes nest, restore and stay on their
// own thread.
// GPU: for both FP8 selectors (RK, and R / KR) in the one dense reduction selection cache, a deeper cached selection
// answers a shallower request without a new contest, and a shallower cached selection does not answer a deeper one. The shapes are unusual primes so no other
// test in the process has tuned them.

#include "test/Utilities/TensorOperations/Reduction/ReductionTestSupport.h"
#include "Utilities/TensorOperations/Reduction/ArgReducer.h"
#include "Utilities/TensorOperations/Reduction/Reducer.h"
#include "Utilities/TensorOperations/Reduction/ReductionTuningDepth.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionSelectionCache.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::ReductionTestSupport;
namespace Internal = ThorImplementation::ReductionInternal;

namespace {

// Restores the process default whatever a test sets it to.
struct DefaultDepthGuard {
    ReductionTuningDepth saved = defaultReductionTuningDepth();
    ~DefaultDepthGuard() { setDefaultReductionTuningDepth(saved); }
};

ReductionTuningDepth valueDepth(std::optional<ReductionTuningDepth> depth = std::nullopt) {
    return Reducer(ReductionOp::Sum, 0U, DataType::FP32, 1.0f, depth).getTuningDepth();
}

ReductionTuningDepth argDepth(std::optional<ReductionTuningDepth> depth = std::nullopt) {
    return ArgReducer(ArgReductionOp::ArgMax, 0U, ArgReductionOutputOptions{}, depth).getTuningDepth();
}

}  // namespace

TEST(ReductionTuningDepth, DefaultIsRegularAndExplicitDepthWins) {
    DefaultDepthGuard guard;
    setDefaultReductionTuningDepth(ReductionTuningDepth::Regular);
    EXPECT_EQ(currentReductionTuningDepth(), ReductionTuningDepth::Regular);
    EXPECT_EQ(valueDepth(), ReductionTuningDepth::Regular);
    EXPECT_EQ(argDepth(), ReductionTuningDepth::Regular);
    EXPECT_EQ(valueDepth(ReductionTuningDepth::Quick), ReductionTuningDepth::Quick);
    EXPECT_EQ(argDepth(ReductionTuningDepth::Ultra), ReductionTuningDepth::Ultra);
    // The axes-vector constructors resolve the same way.
    EXPECT_EQ(Reducer(ReductionOp::Max, std::vector<uint32_t>{0, 1}).getTuningDepth(),
              ReductionTuningDepth::Regular);
    EXPECT_EQ(ArgReducer(ArgReductionOp::ArgMin, std::vector<uint32_t>{1}).getTuningDepth(),
              ReductionTuningDepth::Regular);
}

TEST(ReductionTuningDepth, ProcessDefaultAppliesToReductionsWithoutADepth) {
    DefaultDepthGuard guard;
    setDefaultReductionTuningDepth(ReductionTuningDepth::Ultra);
    EXPECT_EQ(defaultReductionTuningDepth(), ReductionTuningDepth::Ultra);
    EXPECT_EQ(valueDepth(), ReductionTuningDepth::Ultra);
    EXPECT_EQ(argDepth(), ReductionTuningDepth::Ultra);
    EXPECT_EQ(valueDepth(ReductionTuningDepth::Quick), ReductionTuningDepth::Quick);
    // The depth is resolved at construction: a later change does not reach an existing reduction.
    const Reducer made_at_ultra(ReductionOp::Sum, 0U);
    setDefaultReductionTuningDepth(ReductionTuningDepth::Quick);
    EXPECT_EQ(made_at_ultra.getTuningDepth(), ReductionTuningDepth::Ultra);
    EXPECT_EQ(valueDepth(), ReductionTuningDepth::Quick);
}

TEST(ReductionTuningDepth, ScopesNestAndRestore) {
    DefaultDepthGuard guard;
    setDefaultReductionTuningDepth(ReductionTuningDepth::Regular);
    {
        ScopedReductionTuningDepth quick(ReductionTuningDepth::Quick);
        EXPECT_EQ(valueDepth(), ReductionTuningDepth::Quick);
        {
            ScopedReductionTuningDepth ultra(ReductionTuningDepth::Ultra);
            EXPECT_EQ(valueDepth(), ReductionTuningDepth::Ultra);
            EXPECT_EQ(argDepth(), ReductionTuningDepth::Ultra);
            EXPECT_EQ(valueDepth(ReductionTuningDepth::Quick), ReductionTuningDepth::Quick);
        }
        EXPECT_EQ(valueDepth(), ReductionTuningDepth::Quick);
        // A scope overrides the process default, whatever it is set to.
        setDefaultReductionTuningDepth(ReductionTuningDepth::Ultra);
        EXPECT_EQ(valueDepth(), ReductionTuningDepth::Quick);
    }
    EXPECT_EQ(valueDepth(), ReductionTuningDepth::Ultra);
    EXPECT_EQ(defaultReductionTuningDepth(), ReductionTuningDepth::Ultra);
}

TEST(ReductionTuningDepth, ScopesStayOnTheirThread) {
    DefaultDepthGuard guard;
    setDefaultReductionTuningDepth(ReductionTuningDepth::Regular);
    ScopedReductionTuningDepth ultra(ReductionTuningDepth::Ultra);
    ReductionTuningDepth seen_default = ReductionTuningDepth::Ultra;
    ReductionTuningDepth seen_scoped = ReductionTuningDepth::Ultra;
    ReductionTuningDepth seen_after = ReductionTuningDepth::Ultra;
    std::thread worker([&] {
        seen_default = valueDepth();  // the other thread's scope does not apply: the process default does
        {
            ScopedReductionTuningDepth quick(ReductionTuningDepth::Quick);
            seen_scoped = argDepth();
        }
        seen_after = valueDepth();
    });
    worker.join();
    EXPECT_EQ(seen_default, ReductionTuningDepth::Regular);
    EXPECT_EQ(seen_scoped, ReductionTuningDepth::Quick);
    EXPECT_EQ(seen_after, ReductionTuningDepth::Regular);
    EXPECT_EQ(valueDepth(), ReductionTuningDepth::Ultra);  // the worker's scope did not reach this thread
}

#if THOR_CUB_ENABLE_FP8_TYPES

namespace {

struct Attempts {
    uint64_t rk = 0, kr = 0;
};

Attempts attempts() {
    const Internal::DenseReductionSelectionCache& cache = Internal::denseReductionSelectionCache();
    return {cache.tuningAttempts(Internal::DenseReductionSelectionKind::Fp8RK),
            cache.tuningAttempts(Internal::DenseReductionSelectionKind::Fp8KR)};
}

// Sums an all-ones FP8 E4M3 tensor over `axis` at `depth` (nullopt: the ambient depth), checks the result and returns
// the contests the stamp ran in each cache.
Attempts stampSum(const std::vector<uint64_t>& dims, uint32_t axis, std::optional<ReductionTuningDepth> depth,
                  Stream& stream) {
    uint64_t elements = 1;
    for (uint64_t d : dims) elements *= d;
    Tensor input = makeGpuTensor(std::vector<float>(elements, 1.0f), dims, stream, DataType::FP8_E4M3);
    const Attempts before = attempts();
    auto stamped = Reducer(ReductionOp::Sum, axis, DataType::FP32, 1.0f, depth).stamp(input, stream);
    const Attempts after = attempts();
    stamped->run();
    stream.synchronize();
    const std::vector<float> got = copyGpuTensorAsFloat(stamped->getOutputTensor(), stream);
    EXPECT_EQ(got.size(), elements / dims[axis]);
    for (float v : got) EXPECT_EQ(v, static_cast<float>(dims[axis]));
    return {after.rk - before.rk, after.kr - before.kr};
}

uint64_t contests(const Attempts& a, bool rk) { return rk ? a.rk : a.kr; }

// Ultra, then Quick and Regular: one contest at most for the first stamp's cache, none afterwards. Then on a second
// shape Quick, then Ultra: the Ultra stamp runs a contest of its own.
void checkDepthRules(const std::vector<uint64_t>& deeper_first, const std::vector<uint64_t>& shallower_first,
                     uint32_t axis, bool rk, Stream& stream) {
    EXPECT_GE(contests(stampSum(deeper_first, axis, ReductionTuningDepth::Ultra, stream), rk), 1U);
    const Attempts quick = stampSum(deeper_first, axis, ReductionTuningDepth::Quick, stream);
    EXPECT_EQ(quick.rk, 0U);
    EXPECT_EQ(quick.kr, 0U);
    const Attempts regular = stampSum(deeper_first, axis, ReductionTuningDepth::Regular, stream);
    EXPECT_EQ(regular.rk, 0U);
    EXPECT_EQ(regular.kr, 0U);
    const Attempts again = stampSum(deeper_first, axis, ReductionTuningDepth::Ultra, stream);
    EXPECT_EQ(again.rk, 0U);
    EXPECT_EQ(again.kr, 0U);

    EXPECT_GE(contests(stampSum(shallower_first, axis, ReductionTuningDepth::Quick, stream), rk), 1U);
    EXPECT_GE(contests(stampSum(shallower_first, axis, ReductionTuningDepth::Ultra, stream), rk), 1U);
    const Attempts quick_after_ultra = stampSum(shallower_first, axis, ReductionTuningDepth::Quick, stream);
    EXPECT_EQ(quick_after_ultra.rk, 0U);
    EXPECT_EQ(quick_after_ultra.kr, 0U);
}

}  // namespace

TEST(ReductionTuningDepth, RKCacheDeeperAnswersShallower) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // [R, K] reduced over R: one original FP8 RK site.
    checkDepthRules({6151, 41}, {6163, 43}, 0, /*rk=*/true, stream);
}

TEST(ReductionTuningDepth, KRCacheDeeperAnswersShallower) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // [K, R] reduced over R: one original FP8 KR site.
    checkDepthRules({47, 6173}, {53, 6197}, 1, /*rk=*/false, stream);
}

TEST(ReductionTuningDepth, AmbientScopeReachesTheSelectionCaches) {
    REQUIRE_CUDA_DEVICE();
    DefaultDepthGuard guard;
    Stream stream(0);
    {
        // A reduction without a depth, as the framework constructs it, tunes at the scope's depth...
        ScopedReductionTuningDepth ultra(ReductionTuningDepth::Ultra);
        EXPECT_GE(stampSum({6211, 59}, 0, std::nullopt, stream).rk, 1U);
        EXPECT_GE(stampSum({61, 6217}, 1, std::nullopt, stream).kr, 1U);
    }
    // ...so later requests at any depth are answered from the cache.
    for (const auto depth : {ReductionTuningDepth::Quick, ReductionTuningDepth::Regular,
                             ReductionTuningDepth::Ultra}) {
        const Attempts rk = stampSum({6211, 59}, 0, depth, stream);
        const Attempts kr = stampSum({61, 6217}, 1, depth, stream);
        EXPECT_EQ(rk.rk + rk.kr + kr.rk + kr.kr, 0U) << "depth rank " << reductionTuningDepthRank(depth);
    }
    // A Quick process default tunes a new shape at Quick; an Ultra request for it then runs its own contest.
    setDefaultReductionTuningDepth(ReductionTuningDepth::Quick);
    EXPECT_GE(stampSum({6221, 67}, 0, std::nullopt, stream).rk, 1U);
    EXPECT_GE(stampSum({6221, 67}, 0, ReductionTuningDepth::Ultra, stream).rk, 1U);
}

#endif  // THOR_CUB_ENABLE_FP8_TYPES
