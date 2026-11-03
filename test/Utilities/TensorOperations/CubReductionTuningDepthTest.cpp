// 1169: the ambient stamp-time tuning depth and the depth rules of the FP8 selectors in the selection cache.
//
// Host: a reduction constructed without an explicit depth takes this thread's innermost ScopedCubReductionTuningDepth,
// else the process default (Regular unless set); an explicit depth always wins; scopes nest, restore and stay on their
// own thread.
// GPU: for both FP8 selectors (RK, and R / KR) in the one dense reduction selection cache, a deeper cached selection
// answers a shallower request without a new contest, and a shallower cached selection does not answer a deeper one. The shapes are unusual primes so no other
// test in the process has tuned them.

#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/CubReductionTuningDepth.h"
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;
namespace Internal = ThorImplementation::CubReductionInternal;

namespace {

// Restores the process default whatever a test sets it to.
struct DefaultDepthGuard {
    CubReductionTuningDepth saved = defaultCubReductionTuningDepth();
    ~DefaultDepthGuard() { setDefaultCubReductionTuningDepth(saved); }
};

CubReductionTuningDepth valueDepth(std::optional<CubReductionTuningDepth> depth = std::nullopt) {
    return CubReduction(CubReductionOp::Sum, 0U, DataType::FP32, 1.0f, depth).getTuningDepth();
}

CubReductionTuningDepth argDepth(std::optional<CubReductionTuningDepth> depth = std::nullopt) {
    return CubArgReduction(CubArgReductionOp::ArgMax, 0U, CubArgReductionOutputOptions{}, depth).getTuningDepth();
}

}  // namespace

TEST(CubReductionTuningDepth, DefaultIsRegularAndExplicitDepthWins) {
    DefaultDepthGuard guard;
    setDefaultCubReductionTuningDepth(CubReductionTuningDepth::Regular);
    EXPECT_EQ(currentCubReductionTuningDepth(), CubReductionTuningDepth::Regular);
    EXPECT_EQ(valueDepth(), CubReductionTuningDepth::Regular);
    EXPECT_EQ(argDepth(), CubReductionTuningDepth::Regular);
    EXPECT_EQ(valueDepth(CubReductionTuningDepth::Quick), CubReductionTuningDepth::Quick);
    EXPECT_EQ(argDepth(CubReductionTuningDepth::Ultra), CubReductionTuningDepth::Ultra);
    // The axes-vector constructors resolve the same way.
    EXPECT_EQ(CubReduction(CubReductionOp::Max, std::vector<uint32_t>{0, 1}).getTuningDepth(),
              CubReductionTuningDepth::Regular);
    EXPECT_EQ(CubArgReduction(CubArgReductionOp::ArgMin, std::vector<uint32_t>{1}).getTuningDepth(),
              CubReductionTuningDepth::Regular);
}

TEST(CubReductionTuningDepth, ProcessDefaultAppliesToReductionsWithoutADepth) {
    DefaultDepthGuard guard;
    setDefaultCubReductionTuningDepth(CubReductionTuningDepth::Ultra);
    EXPECT_EQ(defaultCubReductionTuningDepth(), CubReductionTuningDepth::Ultra);
    EXPECT_EQ(valueDepth(), CubReductionTuningDepth::Ultra);
    EXPECT_EQ(argDepth(), CubReductionTuningDepth::Ultra);
    EXPECT_EQ(valueDepth(CubReductionTuningDepth::Quick), CubReductionTuningDepth::Quick);
    // The depth is resolved at construction: a later change does not reach an existing reduction.
    const CubReduction made_at_ultra(CubReductionOp::Sum, 0U);
    setDefaultCubReductionTuningDepth(CubReductionTuningDepth::Quick);
    EXPECT_EQ(made_at_ultra.getTuningDepth(), CubReductionTuningDepth::Ultra);
    EXPECT_EQ(valueDepth(), CubReductionTuningDepth::Quick);
}

TEST(CubReductionTuningDepth, ScopesNestAndRestore) {
    DefaultDepthGuard guard;
    setDefaultCubReductionTuningDepth(CubReductionTuningDepth::Regular);
    {
        ScopedCubReductionTuningDepth quick(CubReductionTuningDepth::Quick);
        EXPECT_EQ(valueDepth(), CubReductionTuningDepth::Quick);
        {
            ScopedCubReductionTuningDepth ultra(CubReductionTuningDepth::Ultra);
            EXPECT_EQ(valueDepth(), CubReductionTuningDepth::Ultra);
            EXPECT_EQ(argDepth(), CubReductionTuningDepth::Ultra);
            EXPECT_EQ(valueDepth(CubReductionTuningDepth::Quick), CubReductionTuningDepth::Quick);
        }
        EXPECT_EQ(valueDepth(), CubReductionTuningDepth::Quick);
        // A scope overrides the process default, whatever it is set to.
        setDefaultCubReductionTuningDepth(CubReductionTuningDepth::Ultra);
        EXPECT_EQ(valueDepth(), CubReductionTuningDepth::Quick);
    }
    EXPECT_EQ(valueDepth(), CubReductionTuningDepth::Ultra);
    EXPECT_EQ(defaultCubReductionTuningDepth(), CubReductionTuningDepth::Ultra);
}

TEST(CubReductionTuningDepth, ScopesStayOnTheirThread) {
    DefaultDepthGuard guard;
    setDefaultCubReductionTuningDepth(CubReductionTuningDepth::Regular);
    ScopedCubReductionTuningDepth ultra(CubReductionTuningDepth::Ultra);
    CubReductionTuningDepth seen_default = CubReductionTuningDepth::Ultra;
    CubReductionTuningDepth seen_scoped = CubReductionTuningDepth::Ultra;
    CubReductionTuningDepth seen_after = CubReductionTuningDepth::Ultra;
    std::thread worker([&] {
        seen_default = valueDepth();  // the other thread's scope does not apply: the process default does
        {
            ScopedCubReductionTuningDepth quick(CubReductionTuningDepth::Quick);
            seen_scoped = argDepth();
        }
        seen_after = valueDepth();
    });
    worker.join();
    EXPECT_EQ(seen_default, CubReductionTuningDepth::Regular);
    EXPECT_EQ(seen_scoped, CubReductionTuningDepth::Quick);
    EXPECT_EQ(seen_after, CubReductionTuningDepth::Regular);
    EXPECT_EQ(valueDepth(), CubReductionTuningDepth::Ultra);  // the worker's scope did not reach this thread
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
Attempts stampSum(const std::vector<uint64_t>& dims, uint32_t axis, std::optional<CubReductionTuningDepth> depth,
                  Stream& stream) {
    uint64_t elements = 1;
    for (uint64_t d : dims) elements *= d;
    Tensor input = makeGpuTensor(std::vector<float>(elements, 1.0f), dims, stream, DataType::FP8_E4M3);
    const Attempts before = attempts();
    auto stamped = CubReduction(CubReductionOp::Sum, axis, DataType::FP32, 1.0f, depth).stamp(input, stream);
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
    EXPECT_GE(contests(stampSum(deeper_first, axis, CubReductionTuningDepth::Ultra, stream), rk), 1U);
    const Attempts quick = stampSum(deeper_first, axis, CubReductionTuningDepth::Quick, stream);
    EXPECT_EQ(quick.rk, 0U);
    EXPECT_EQ(quick.kr, 0U);
    const Attempts regular = stampSum(deeper_first, axis, CubReductionTuningDepth::Regular, stream);
    EXPECT_EQ(regular.rk, 0U);
    EXPECT_EQ(regular.kr, 0U);
    const Attempts again = stampSum(deeper_first, axis, CubReductionTuningDepth::Ultra, stream);
    EXPECT_EQ(again.rk, 0U);
    EXPECT_EQ(again.kr, 0U);

    EXPECT_GE(contests(stampSum(shallower_first, axis, CubReductionTuningDepth::Quick, stream), rk), 1U);
    EXPECT_GE(contests(stampSum(shallower_first, axis, CubReductionTuningDepth::Ultra, stream), rk), 1U);
    const Attempts quick_after_ultra = stampSum(shallower_first, axis, CubReductionTuningDepth::Quick, stream);
    EXPECT_EQ(quick_after_ultra.rk, 0U);
    EXPECT_EQ(quick_after_ultra.kr, 0U);
}

}  // namespace

TEST(CubReductionTuningDepth, RKCacheDeeperAnswersShallower) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // [R, K] reduced over R: one original FP8 RK site.
    checkDepthRules({6151, 41}, {6163, 43}, 0, /*rk=*/true, stream);
}

TEST(CubReductionTuningDepth, KRCacheDeeperAnswersShallower) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // [K, R] reduced over R: one original FP8 KR site.
    checkDepthRules({47, 6173}, {53, 6197}, 1, /*rk=*/false, stream);
}

TEST(CubReductionTuningDepth, AmbientScopeReachesTheSelectionCaches) {
    REQUIRE_CUDA_DEVICE();
    DefaultDepthGuard guard;
    Stream stream(0);
    {
        // A reduction without a depth, as the framework constructs it, tunes at the scope's depth...
        ScopedCubReductionTuningDepth ultra(CubReductionTuningDepth::Ultra);
        EXPECT_GE(stampSum({6211, 59}, 0, std::nullopt, stream).rk, 1U);
        EXPECT_GE(stampSum({61, 6217}, 1, std::nullopt, stream).kr, 1U);
    }
    // ...so later requests at any depth are answered from the cache.
    for (const auto depth : {CubReductionTuningDepth::Quick, CubReductionTuningDepth::Regular,
                             CubReductionTuningDepth::Ultra}) {
        const Attempts rk = stampSum({6211, 59}, 0, depth, stream);
        const Attempts kr = stampSum({61, 6217}, 1, depth, stream);
        EXPECT_EQ(rk.rk + rk.kr + kr.rk + kr.kr, 0U) << "depth rank " << cubReductionTuningDepthRank(depth);
    }
    // A Quick process default tunes a new shape at Quick; an Ultra request for it then runs its own contest.
    setDefaultCubReductionTuningDepth(CubReductionTuningDepth::Quick);
    EXPECT_GE(stampSum({6221, 67}, 0, std::nullopt, stream).rk, 1U);
    EXPECT_GE(stampSum({6221, 67}, 0, CubReductionTuningDepth::Ultra, stream).rk, 1U);
}

#endif  // THOR_CUB_ENABLE_FP8_TYPES
