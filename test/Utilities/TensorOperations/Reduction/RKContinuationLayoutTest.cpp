// An RK first stage's continuation layout (whether an FP32 continuation follows, its partials and input shape, the
// workspace lower bound) and the contest scratch layout built from it. Host-only.

#include "Utilities/TensorOperations/Reduction/RK/RKContinuationLayout.h"

#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace P = ThorImplementation::ReductionInternal::RK;

namespace {
const P::PhysicalKernel complete{P::PhysicalFamily::OneWindow, 1, 1, 16, 0, 0, 0, 6, 16, 0};
const P::PhysicalKernel deep{P::PhysicalFamily::DeepRSmallK, 128, 0, 0, 0, 0, 0, 6, 32, 4096};
const P::PhysicalKernel aligned{P::PhysicalFamily::AlignedK, 8, 32, 4, 0, 0, 0, 6, 8, 8192};
const P::PhysicalKernel rotated{P::PhysicalFamily::RuntimeK, static_cast<uint64_t>(P::RuntimeLayout::RotatedExact), 0,
                                512, 1, 256, 8, 6, 16, 32768};
}  // namespace

TEST(RKContinuationLayout, ACompleteKernelHasNoContinuationOrScratch) {
    auto layout = P::describeContinuationLayout(complete, 128, 16, 3, true);
    EXPECT_FALSE(layout.has_fp32_continuation);
    EXPECT_EQ(layout.minimum_workspace_bytes, 0u);
    EXPECT_EQ(P::makeAutotuneScratchLayout(layout, 0).total_bytes, 0u);
    EXPECT_THROW((void)P::makeAutotuneScratchLayout(layout, 1), std::invalid_argument);
    layout = P::describeContinuationLayout(complete, 128, 16, 3, false);
    EXPECT_FALSE(layout.has_fp32_continuation);
    EXPECT_EQ(layout.retained_result_bytes, 1536u);
}

TEST(RKContinuationLayout, AStagedKernelDescribesItsPartialsAndContinuation) {
    auto layout = P::describeContinuationLayout(deep, 2, 2048, 32, true);
    EXPECT_TRUE(layout.has_fp32_continuation);
    EXPECT_EQ(layout.shards, 16u);
    EXPECT_EQ(layout.partial_bytes, 4096u);
    EXPECT_EQ(layout.continuation_input_dimensions, (std::array<uint64_t, 3>{2, 16, 32}));
    EXPECT_EQ(layout.continuation_reduction_axis, 1u);
    EXPECT_EQ(layout.minimum_workspace_bytes, 4096u);
    const auto workspace = P::makeAutotuneScratchLayout(layout, 17);
    EXPECT_EQ(workspace.partial_offset_bytes, 0u);
    EXPECT_EQ(workspace.partial_bytes, 4096u);
    EXPECT_EQ(workspace.continuation_offset_bytes, 4096u);
    EXPECT_EQ(workspace.total_bytes, 4113u);
    layout = P::describeContinuationLayout(deep, 2, 2048, 32, false);
    EXPECT_EQ(layout.retained_result_bytes, 256u);
    EXPECT_EQ(layout.minimum_workspace_bytes, 4096u);

    layout = P::describeContinuationLayout(aligned, 2, 256, 256, false);
    EXPECT_EQ(layout.shards, 4u);
    EXPECT_EQ(layout.partial_bytes, 8192u);
    EXPECT_EQ(layout.retained_result_bytes, 2048u);
    EXPECT_EQ(P::makeAutotuneScratchLayout(layout, 3).total_bytes, 8195u);

    layout = P::describeContinuationLayout(rotated, 2, 2048, 512, true);
    EXPECT_EQ(layout.shards, 8u);
    EXPECT_EQ(layout.partial_bytes, 32768u);
    EXPECT_EQ(layout.retained_result_bytes, 0u);
}

TEST(RKContinuationLayout, ContinuationScratchIsAlignedAndOverflowIsRejected) {
    P::ContinuationLayout odd;
    odd.has_fp32_continuation = true;
    odd.partial_bytes = 12;
    const auto workspace = P::makeAutotuneScratchLayout(odd, 7);
    EXPECT_EQ(workspace.continuation_offset_bytes, 32u);
    EXPECT_EQ(workspace.total_bytes, 39u);
    odd.partial_bytes = UINT64_MAX - 2;
    EXPECT_THROW((void)P::makeAutotuneScratchLayout(odd, 8), std::overflow_error);

    auto bad = rotated;
    bad.first_stage_scratch_bytes = UINT64_MAX;
    EXPECT_THROW((void)P::describeContinuationLayout(bad, 2, 2048, 512, true), std::invalid_argument);
}
