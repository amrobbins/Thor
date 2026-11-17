#include "Utilities/TensorOperations/Reduction/Arg/Fp32CarriedArgContestInput.h"
#include "Utilities/TensorOperations/Reduction/Contest/ReductionAutotunerCore.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

using namespace ThorImplementation;

namespace {

// Every autotuning memory range starts on a 256-byte boundary and has at least 128 bytes before the next range's
// byte offset.
TEST(ReductionAutotunerSlices, EverySliceIsAlignedAndPaddedBeforeTheNext) {
    for (uint64_t bytes : std::vector<uint64_t>{1, 4, 127, 128, 129, 255, 256, 257, 384, 4096, 4097, 1u << 20}) {
        const uint64_t next = ReductionAutotuning::nextSliceOffset(bytes);
        EXPECT_EQ(next, ReductionAutotuning::alignedStride(bytes)) << bytes;
        EXPECT_EQ(next % ReductionAutotuning::SliceAlignmentBytes, 0u) << bytes;
        EXPECT_GE(next - bytes, ReductionAutotuning::SlicePaddingBytes) << bytes;
        EXPECT_LT(next - bytes, ReductionAutotuning::SlicePaddingBytes + ReductionAutotuning::SliceAlignmentBytes) << bytes;
    }
    // No slice: the next one starts at the beginning.
    EXPECT_EQ(ReductionAutotuning::nextSliceOffset(0), 0u);
    EXPECT_EQ(ReductionAutotuning::SliceAlignmentBytes, 256u);
    EXPECT_EQ(ReductionAutotuning::SlicePaddingBytes, 128u);
}

TEST(ReductionAutotunerSlices, CarriedContestInputPadsTheValuesBeforeTheCarriedIndex) {
    // 64 FP32 values are 256 bytes: before, the index started right after them, at 256; now 128 bytes later, rounded
    // up to 256.
    const ReductionInternal::Fp32CarriedArgContestInput wide =
        ReductionInternal::fp32CarriedArgContestInput(64, DataType::UINT32);
    EXPECT_EQ(wide.index_offset, 512u);
    EXPECT_EQ(wide.bytes, 512u + 64u * 4u);

    const ReductionInternal::Fp32CarriedArgContestInput narrow =
        ReductionInternal::fp32CarriedArgContestInput(32, DataType::UINT64);
    EXPECT_EQ(narrow.index_offset, 256u);
    EXPECT_EQ(narrow.bytes, 256u + 32u * 8u);

    // The slices of a contest slot are spans of its bytes.
    Tensor slot(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::UINT8, {wide.bytes}));
    const StridedSpan values = wide.values(slot, {8, 8});
    const StridedSpan carried = wide.carriedIndex(slot, {8, 8});
    EXPECT_EQ(values.getDataType(), DataType::FP32);
    EXPECT_EQ(carried.getDataType(), DataType::UINT32);
    EXPECT_EQ(values.getMemPtr<uint8_t>(), slot.getMemPtr<uint8_t>());
    EXPECT_EQ(carried.getMemPtr<uint8_t>(), slot.getMemPtr<uint8_t>() + 512);
    EXPECT_EQ(carried.getDimensions(), (std::vector<uint64_t>{8, 8}));
}

}  // namespace
