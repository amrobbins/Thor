#include "DeepLearning/Implementation/Tensor/StridedSpan.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

namespace {

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);

TEST(StridedSpan, AWholeTensorConvertsWithItsOwnLayout) {
    Tensor tensor(cpuPlacement, TensorDescriptor(DataType::FP32, {3, 4}));
    const StridedSpan span = tensor;
    EXPECT_TRUE(span.getSourceTensor() == tensor);
    EXPECT_EQ(span.getStorage().offset, 0u);
    EXPECT_EQ(span.getStorage().count, 12u);
    EXPECT_EQ(span.getDimensions(), (std::vector<uint64_t>{3, 4}));
    EXPECT_EQ(span.getStridesElements(), (std::vector<uint64_t>{4, 1}));
    EXPECT_TRUE(span.isDenseContiguous());
    EXPECT_EQ(span.getMemPtr<void>(), tensor.getMemPtr<void>());
    EXPECT_EQ(span.getDescriptor(), tensor.getDescriptor());
}

TEST(StridedSpan, AGluHalfIsAStridedSpanOfTheProjection) {
    // A [T, 2H] projection with T = 3, H = 4: the gate half is [T, H] with row stride 2H, starting H elements in.
    Tensor projection(cpuPlacement, TensorDescriptor(DataType::FP32, {3, 8}));
    float* base = projection.getMemPtr<float>();
    const StridedSpan gate(projection, 4, {3, 4}, {8, 1});
    EXPECT_FALSE(gate.isDenseContiguous());
    EXPECT_EQ(gate.getTotalNumElements(), 12u);
    EXPECT_EQ(gate.getStorage().count, 2u * 8u + 3u + 1u);  // through the last row's last element
    EXPECT_EQ(gate.getMemPtr<void>(), static_cast<void*>(base + 4));
    EXPECT_EQ(projection.getDimensions(), (std::vector<uint64_t>{3, 8}));

    // An offset-only slice is dense (it can be read as a dense tensor from its first element).
    EXPECT_TRUE(StridedSpan(projection, 8, {2, 8}, {8, 1}).isDenseContiguous());
}

TEST(StridedSpan, RejectsLayoutsOutsideTheSourceOrMalformed) {
    Tensor projection(cpuPlacement, TensorDescriptor(DataType::FP32, {3, 8}));
    EXPECT_THROW(StridedSpan(projection, 5, {3, 4}, {8, 1}), std::invalid_argument);  // last element is 24
    EXPECT_THROW(StridedSpan(projection, 0, {3, 4}, {8}), std::invalid_argument);
    EXPECT_THROW(StridedSpan(projection, 0, {}, {}), std::invalid_argument);
    EXPECT_THROW(StridedSpan(projection, 0, {3, 0}, {8, 1}), std::invalid_argument);
    EXPECT_NO_THROW(StridedSpan(projection, 4, {3, 4}, {8, 1}));
}

TEST(StridedSpan, ReadsThroughTheSameAccessorsAsATensor) {
    Tensor projection(cpuPlacement, TensorDescriptor(DataType::FP32, {3, 8}));
    float* base = projection.getMemPtr<float>();
    const StridedSpan rows(projection, 8, {2, 8}, {8, 1});
    EXPECT_TRUE(rows.isInitialized());
    EXPECT_EQ(rows.getDimensions(), (std::vector<uint64_t>{2, 8}));
    EXPECT_EQ(rows.getStridesElements(), (std::vector<uint64_t>{8, 1}));
    EXPECT_EQ(rows.getArraySizeInBytes(), 16u * sizeof(float));
    EXPECT_EQ(rows.getMemPtr<float>(), base + 8);
    EXPECT_EQ(rows.getDescriptor(), TensorDescriptor(DataType::FP32, {2, 8}));
    EXPECT_TRUE(rows.getSourceTensor() == projection);
    EXPECT_FALSE(rows.isWholeTensor());
    EXPECT_TRUE(StridedSpan(projection).isWholeTensor());
    EXPECT_GE(rows.alignmentBytes(16), 4u);

    // A dense span reshapes like a tensor; a strided one does not.
    StridedSpan reshaped = rows;
    reshaped.reshape({16});
    EXPECT_EQ(reshaped.getDimensions(), (std::vector<uint64_t>{16}));
    EXPECT_EQ(reshaped.getMemPtr<float>(), base + 8);
    StridedSpan gate(projection, 4, {3, 4}, {8, 1});
    EXPECT_THROW(gate.reshape({12}), std::invalid_argument);
    EXPECT_THROW(reshaped.reshape({15}), std::invalid_argument);

    EXPECT_FALSE(StridedSpan().isInitialized());
}

TEST(StridedSpan, BytesAsReadsPartOfADenseSpansBytesAsAnotherType) {
    // An autotuner arena slot: bytes of a UINT8 tensor read as typed spans, with no tensor view.
    Tensor arena(cpuPlacement, TensorDescriptor(DataType::UINT8, {1024}));
    uint8_t* base = arena.getMemPtr<uint8_t>();
    const StridedSpan slot = StridedSpan(arena).bytesAs(256, DataType::FP32, {4, 8});
    EXPECT_EQ(slot.getDataType(), DataType::FP32);
    EXPECT_EQ(slot.getDimensions(), (std::vector<uint64_t>{4, 8}));
    EXPECT_EQ(slot.getStridesElements(), (std::vector<uint64_t>{8, 1}));
    EXPECT_EQ(slot.getArraySizeInBytes(), 128u);
    EXPECT_EQ(slot.getMemPtr<uint8_t>(), base + 256);
    EXPECT_TRUE(slot.isDenseContiguous());
    EXPECT_FALSE(slot.isWholeTensor());
    EXPECT_TRUE(slot.getSourceTensor() == arena);

    // A slice of a slice counts from the slice's first byte; a reshape keeps the dtype.
    StridedSpan index = slot.bytesAs(64, DataType::UINT32, {8});
    EXPECT_EQ(index.getMemPtr<uint8_t>(), base + 320);
    EXPECT_EQ(index.getDataType(), DataType::UINT32);
    index.reshape({2, 4});
    EXPECT_EQ(index.getDataType(), DataType::UINT32);
    EXPECT_EQ(index.getMemPtr<uint8_t>(), base + 320);

    EXPECT_THROW((void)StridedSpan(arena).bytesAs(2, DataType::FP32, {4}), std::invalid_argument);       // misaligned
    EXPECT_THROW((void)StridedSpan(arena).bytesAs(1020, DataType::FP32, {2}), std::invalid_argument);    // past the end
    EXPECT_THROW((void)slot.bytesAs(0, DataType::FP32, {33}), std::invalid_argument);                    // past the slot
    const StridedSpan strided(arena, 0, {4, 4}, {8, 1});
    EXPECT_THROW((void)strided.bytesAs(0, DataType::UINT8, {4}), std::invalid_argument);                 // not dense
}

TEST(StridedSpan, WithLayoutReadsADenseSpansElementsUnderAnotherLayout) {
    // A transposed FP32 [3, 2] over bytes of a UINT8 arena: bytesAs, then a layout over those elements.
    Tensor arena(cpuPlacement, TensorDescriptor(DataType::UINT8, {512}));
    uint8_t* base = arena.getMemPtr<uint8_t>();
    const StridedSpan flat = StridedSpan(arena).bytesAs(256, DataType::FP32, {6});
    const StridedSpan transposed = flat.withLayout({3, 2}, {1, 3});
    EXPECT_EQ(transposed.getDataType(), DataType::FP32);
    EXPECT_EQ(transposed.getDimensions(), (std::vector<uint64_t>{3, 2}));
    EXPECT_EQ(transposed.getStridesElements(), (std::vector<uint64_t>{1, 3}));
    EXPECT_FALSE(transposed.isDenseContiguous());
    EXPECT_EQ(transposed.getMemPtr<uint8_t>(), base + 256);

    EXPECT_THROW((void)flat.withLayout({3, 3}, {3, 1}), std::invalid_argument);    // reaches 9 of 6 elements
    EXPECT_THROW((void)flat.withLayout({3, 2}, {1}), std::invalid_argument);       // rank mismatch
    EXPECT_THROW((void)transposed.withLayout({6}, {1}), std::invalid_argument);    // not dense
}

// 1365: asDense reads a compact permutation's elements as dense, in storage order (the view reductions of
// plans/Thor_View_Reduction_Plan_2026-11-08.md phase 1).
TEST(StridedSpan, IsCompactPermutationAcceptsExactlyThePermutedDenseLayouts) {
    Tensor storage(cpuPlacement, TensorDescriptor(DataType::FP32, {4096}));
    EXPECT_TRUE(StridedSpan(storage, 0, {6, 4}, {1, 6}).isCompactPermutation());                 // a transpose
    EXPECT_TRUE(StridedSpan(storage, 0, {7, 6, 40}, {1, 280, 7}).isCompactPermutation());         // rank 3
    EXPECT_TRUE(StridedSpan(storage, 0, {4, 6}, {6, 1}).isCompactPermutation());                  // dense itself
    EXPECT_TRUE(StridedSpan(storage, 5, {3, 1, 4}, {1, 1000, 3}).isCompactPermutation());         // extent-one axis
    EXPECT_FALSE(StridedSpan(storage, 0, {4, 4}, {8, 1}).isCompactPermutation());                 // row gaps
    EXPECT_FALSE(StridedSpan(storage, 0, {2, 2}, {1, 1}).isCompactPermutation());                 // overlap
    EXPECT_FALSE(StridedSpan(storage, 0, {5}, {3}).isCompactPermutation());                       // strided rank-1
    EXPECT_FALSE(StridedSpan(storage, 0, {5, 2}, {0, 1}).isCompactPermutation());                 // broadcast
}

TEST(StridedSpan, AsDenseReadsACompactPermutationsStorageInOrder) {
    // A [7, 6, 40] permutation of a physical [6, 40, 7] that starts 3 elements (12 bytes) into its tensor.
    Tensor storage(cpuPlacement, TensorDescriptor(DataType::FP16, {3 + 1680}));
    const StridedSpan view(storage, 3, {7, 6, 40}, {1, 280, 7});
    const StridedSpan flat = view.asDense({1680});
    EXPECT_TRUE(flat.isDenseContiguous());
    EXPECT_EQ(flat.getDimensions(), (std::vector<uint64_t>{1680}));
    EXPECT_EQ(flat.getDataType(), DataType::FP16);
    EXPECT_EQ(flat.getMemPtr<void>(), view.getMemPtr<void>());
    EXPECT_EQ(flat.getStorage().offset, 3u);
    EXPECT_EQ(flat.getStorage().count, 1680u);
    const StridedSpan rows = view.asDense({240, 7});
    EXPECT_EQ(rows.getStridesElements(), (std::vector<uint64_t>{7, 1}));
    EXPECT_EQ(rows.getMemPtr<void>(), view.getMemPtr<void>());

    EXPECT_THROW((void)view.asDense({1681}), std::invalid_argument);                                  // count
    EXPECT_THROW((void)view.asDense({}), std::invalid_argument);                                      // rank 0
    EXPECT_THROW((void)StridedSpan(storage, 0, {4, 4}, {8, 1}).asDense({16}), std::invalid_argument);  // gaps
    EXPECT_THROW((void)StridedSpan().asDense({1}), std::invalid_argument);                            // empty
}

}  // namespace
