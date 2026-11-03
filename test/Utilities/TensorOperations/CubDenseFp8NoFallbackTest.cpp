// 1173: original FP8 sites have no generic (CUB / direct) fallback. Every original FP8 R, KR and RK site, VALUE and
// ARG, is served by the retained FP8 selectors; anything they cannot serve is a planning error. Sites over an FP32
// intermediate (after an FP8 pass) and non-FP8 inputs keep the ordinary selector. Host-only: every rejection happens
// before any tuning.

#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPhysicalSelector.h"
#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionInternal;

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

// An FP8 context without stamp-time tuning information (no stream).
DensePhysicalSelectionContext fp8Context(std::optional<CubReductionOp> op, uint64_t reduction_size) {
    DensePhysicalSelectionContext context;
    context.rk.original_input_dtype = DataType::FP8_E4M3;
    context.original_value_operation = op;
    context.original_reduction_size = reduction_size;
    return context;
}

void expectRejected(const DenseReductionProblem& problem, DenseReductionSite site,
                    const DensePhysicalSelectionContext& context, const std::string& message_part) {
    try {
        (void)selectDensePhysicalPass(problem, site, context);
    } catch (const std::logic_error& error) {
        EXPECT_NE(std::string(error.what()).find(message_part), std::string::npos) << error.what();
        return;
    }
    ADD_FAILURE() << "an original FP8 site was planned without the retained FP8 selectors";
}

}  // namespace

TEST(CubDenseFp8NoFallback, OriginalFp8KRAndRSitesNeedTheRetainedSelector) {
    // [K=64, R=512] reduced over R: a KR site; [4096] reduced: a pure R site.
    expectRejected(valueProblem({64, 512}, {1}), DenseReductionSite{1}, fp8Context(CubReductionOp::Sum, 512),
                   "Retained FP8 R/KR");
    expectRejected(valueProblem({4096}, {0}), DenseReductionSite{0}, fp8Context(CubReductionOp::Max, 4096),
                   "Retained FP8 R/KR");
    expectRejected(argProblem({64, 512}, {1}), DenseReductionSite{1}, fp8Context(CubReductionOp::Max, 512),
                   "Retained FP8 R/KR");
    // Without the original operation either.
    expectRejected(valueProblem({64, 512}, {1}), DenseReductionSite{1}, fp8Context(std::nullopt, 512),
                   "Retained FP8 R/KR");
}

TEST(CubDenseFp8NoFallback, OriginalFp8RKArgNeedsTheRetainedSelector) {
    // [O=16, R=512, K=64] reduced over R: an RK ARG site.
    expectRejected(argProblem({16, 512, 64}, {1}), DenseReductionSite{1}, fp8Context(CubReductionOp::Min, 512),
                   "Retained FP8 ARG");
    // No operation, or a VALUE operation, is not a fallback either.
    expectRejected(argProblem({16, 512, 64}, {1}), DenseReductionSite{1}, fp8Context(std::nullopt, 512),
                   "Retained FP8 ARG");
    expectRejected(argProblem({16, 512, 64}, {1}), DenseReductionSite{1}, fp8Context(CubReductionOp::Sum, 512),
                   "Retained FP8 ARG");
}

TEST(CubDenseFp8NoFallback, OriginalFp8RKValueNeedsTheRetainedSelector) {
    expectRejected(valueProblem({16, 512, 64}, {1}), DenseReductionSite{1}, fp8Context(CubReductionOp::Sum, 512),
                   "Retained FP8 RK");
}

TEST(CubDenseFp8NoFallback, Fp32IntermediatesAndOtherDTypesKeepTheOrdinarySelector) {
    // A KR site over an FP32 intermediate (an FP8 pass already reduced part of R): the ordinary selector plans it.
    const DenseReductionProblem first = valueProblem({64, 512}, {1});
    const DenseReductionProblem staged = advanceDenseReductionProblem(first, DenseReductionSite{1}, 8);
    ASSERT_EQ(staged.input_state, DenseReductionInputState::IntermediateAggregate);
    ASSERT_TRUE(hasDenseReductionSites(staged));
    const auto sites = enumerateDenseReductionSites(staged);
    ASSERT_EQ(sites.size(), 1U);
    EXPECT_NO_THROW((void)selectDensePhysicalPass(staged, sites.front(), fp8Context(CubReductionOp::Sum, 512)));

    // FP32 input: unchanged.
    DensePhysicalSelectionContext fp32 = fp8Context(CubReductionOp::Sum, 512);
    fp32.rk.original_input_dtype = DataType::FP32;
    EXPECT_NO_THROW((void)selectDensePhysicalPass(first, DenseReductionSite{1}, fp32));
}
