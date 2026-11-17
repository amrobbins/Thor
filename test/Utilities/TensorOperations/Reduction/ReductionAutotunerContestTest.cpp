// The reduction autotuner's contest on the GPU: ring slots are 256-byte aligned with a 128-byte guard after each
// view, and a contest between a fast complete kernel, a staged kernel with an FP32 continuation and a deliberately
// slow kernel never picks the slow or an abandoned one, with abandonment on or off.

#include "Utilities/TensorOperations/Reduction/Contest/ReductionAutotuner.h"
#include "test/Utilities/TensorOperations/Reduction/ReductionTestSupport.h"
#include "test/Utilities/TensorOperations/Reduction/ReductionAutotunerDevice.h"

#include "gtest/gtest.h"

#include <cstdint>

using namespace ThorImplementation;
using namespace ThorImplementation::ReductionAutotuning;
using ThorImplementation::ReductionTestSupport::gpuPlacement;

TEST(ReductionAutotunerContest, RingSlotsAreAlignedAndGuarded) {
    REQUIRE_CUDA_DEVICE();
    for (const TensorDescriptor& descriptor : {TensorDescriptor(DataType::UINT8, {4096}),
                                               TensorDescriptor(DataType::FP32, {129}),
                                               TensorDescriptor(DataType::UINT8, {257})}) {
        detail::TensorRing ring(gpuPlacement, descriptor, 3);
        const uint64_t logical_bytes = descriptor.getArraySizeInBytes();
        uintptr_t previous = 0;
        for (size_t i = 0; i < 3; ++i) {
            const uintptr_t address = reinterpret_cast<uintptr_t>(ring.at(i).getMemPtr<void>());
            EXPECT_EQ(address % SliceAlignmentBytes, 0u);
            if (i != 0) EXPECT_GE(address - previous, logical_bytes + SlicePaddingBytes);
            previous = address;
        }
    }
}

TEST(ReductionAutotunerContest, TheSlowOrAnAbandonedCandidateNeverWins) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input(gpuPlacement, TensorDescriptor(DataType::UINT8, {ReductionAutotunerDevice::InputBytes}));
    input.memsetAsync(stream, 1);
    const TensorDescriptor output(DataType::FP32, {ReductionAutotunerDevice::Partials});
    const auto candidates = ReductionAutotunerDevice::candidates();
    Options options;
    options.random_seed = 20261023;
    const RunResult shipped = autotune(candidates, input, output, stream, options);
    EXPECT_NE(shipped.winner_index, 2u);
    EXPECT_FALSE(shipped.candidates[shipped.winner_index].abandoned);
    // With abandonment off every candidate is measured, and the slow one still loses.
    Options measure_all = options;
    measure_all.abandon_ratio = 0.0;
    const RunResult measured = autotune(candidates, input, output, stream, measure_all);
    EXPECT_EQ(measured.abandoned_count, 0u);
    EXPECT_NE(measured.winner_index, 2u);
}
