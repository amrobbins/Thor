#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

TEST(CubReduction, MultiAxisContiguousSuffixUsesFixedSegments) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                        {2, 2, 3},
                                        stream);

    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Sum, std::vector<uint32_t>{1, 2}, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(stamped->getPath(), CubReductionPath::ContiguousFixedSegment);
    EXPECT_EQ(stamped->getGeometry().reduction_size, 6U);
    EXPECT_EQ(stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 1, 1}));

    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {21.0f, 57.0f});
}

TEST(CubReduction, MultiAxisDisjointUsesLogicalIndexMappingAndLeadingAxesUseTiledReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                        {2, 3, 2},
                                        stream);

    std::shared_ptr<StampedCubReduction> disjoint =
        CubReduction(CubReductionOp::Sum, std::vector<uint32_t>{0, 2}, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(disjoint->getPath(), CubReductionPath::StridedFixedSegment);
    EXPECT_EQ(disjoint->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1, 3, 1}));
    disjoint->run();

    std::shared_ptr<StampedCubReduction> leading =
        CubReduction(CubReductionOp::Sum, std::vector<uint32_t>{0, 1}, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(leading->getPath(), CubReductionPath::TiledFixedSegment);
    EXPECT_EQ(leading->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1, 1, 2}));
    leading->run();
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat(disjoint->getOutputTensor(), stream), {18.0f, 26.0f, 34.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat(leading->getOutputTensor(), stream), {36.0f, 42.0f});
}

TEST(CubReduction, MultiAxisAllAxesUsesDeviceTransformReduce) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                        {2, 3, 2},
                                        stream);

    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Mean, std::vector<uint32_t>{0, 1, 2}, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(stamped->getPath(), CubReductionPath::DeviceTransformReduce);
    EXPECT_EQ(stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1, 1, 1}));
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {6.5f});
}

TEST(CubReduction, PreallocatedOutputAcceptsAnySingletonEquivalentShape) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                        {2, 3, 2},
                                        stream);
    CubReduction reduction(CubReductionOp::Sum, std::vector<uint32_t>{0, 2}, DataType::FP32);

    Tensor keep_dimensions(gpuPlacement, TensorDescriptor(DataType::FP32, {1, 3, 1}));
    Tensor squeezed(gpuPlacement, TensorDescriptor(DataType::FP32, {3}));
    Tensor alternate_singletons(gpuPlacement, TensorDescriptor(DataType::FP32, {3, 1}));
    Tensor wrong_shape(gpuPlacement, TensorDescriptor(DataType::FP32, {1, 1, 3, 2}));

    std::shared_ptr<StampedCubReduction> keep_stamped = reduction.stamp(input, keep_dimensions, stream);
    std::shared_ptr<StampedCubReduction> squeezed_stamped = reduction.stamp(input, squeezed, stream);
    std::shared_ptr<StampedCubReduction> alternate_stamped = reduction.stamp(input, alternate_singletons, stream);
    EXPECT_EQ(keep_stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1, 3, 1}));
    EXPECT_EQ(squeezed_stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{3}));
    EXPECT_EQ(alternate_stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{3, 1}));
    EXPECT_THROW(static_cast<void>(reduction.stamp(input, wrong_shape, stream)), std::invalid_argument);

    keep_stamped->run();
    squeezed_stamped->run();
    alternate_stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(keep_dimensions, stream), {18.0f, 26.0f, 34.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat(squeezed, stream), {18.0f, 26.0f, 34.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat(alternate_singletons, stream), {18.0f, 26.0f, 34.0f});
}

TEST(CubReduction, SqueezedScalarOutputUsesOneElementShape) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, stream);
    Tensor scalar_output(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));

    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Product, std::vector<uint32_t>{0, 1}, DataType::FP32)
            .stamp(input, scalar_output, stream);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(scalar_output, stream), {24.0f});
}

TEST(CubReduction, RankNineStridedReductionUsesStampedDynamicMetadata) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    std::vector<float> values(32);
    for (uint64_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i);
    }
    Tensor input = makeGpuTensor(values, {2, 1, 2, 1, 2, 1, 2, 1, 2}, stream);
    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Sum, std::vector<uint32_t>{0, 2, 4, 6}, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(stamped->getPath(), CubReductionPath::StridedFixedSegment);
    EXPECT_EQ(stamped->getGeometry().rank, 9U);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {240.0f, 256.0f});
}

TEST(CubReduction, PermutedViewUsesProductionTiledPathWithoutMaterialization) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor storage = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                   {2, 3, 2},
                                   stream);
    // Logical [k,i,j] view of dense physical [i,j,k]. The production planner recognizes
    // [outer=i,reduction=j,inner=k] and shared-transposes finalized retained values into dense [k,i].
    Tensor permuted = storage.aliasView({2, 2, 3}, {1, 6, 2}, 0);

    auto stamped = CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(permuted, stream);
    ASSERT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
    ASSERT_TRUE(stamped->getGeometry().permutation_aware_tiled_geometry.has_value());
    EXPECT_TRUE(stamped->getGeometry().tiled_output_permuted);
    EXPECT_TRUE(stamped->getGeometry().tiled_output_shared_transpose);
    EXPECT_EQ(stamped->getGeometry().outer_size, 2U);
    EXPECT_EQ(stamped->getGeometry().reduction_size, 3U);
    EXPECT_EQ(stamped->getGeometry().inner_size, 2U);
    EXPECT_EQ(stamped->getGeometry().tiled_output_outer_stride, 1U);
    EXPECT_EQ(stamped->getGeometry().tiled_output_inner_stride, 2U);
    EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), 1U);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {9.0f, 27.0f, 12.0f, 30.0f});
}

TEST(CubReduction, DensePhysicalPermutationCanKeepNaturalRetainedOutputOrder) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor storage = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                   {2, 3, 2},
                                   stream);
    // Logical [j,i,k] over the same dense [i,j,k] storage. Reducing logical j leaves the
    // natural physical [i,k] retained order, so the tuned kernels keep their normal dense stores.
    Tensor permuted = storage.aliasView({3, 2, 2}, {2, 6, 1}, 0);

    auto stamped = CubReduction(CubReductionOp::Sum, 0, DataType::FP32).stamp(permuted, stream);
    ASSERT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
    ASSERT_TRUE(stamped->getGeometry().permutation_aware_tiled_geometry.has_value());
    EXPECT_FALSE(stamped->getGeometry().tiled_output_permuted);
    EXPECT_FALSE(stamped->getGeometry().tiled_output_shared_transpose);
    EXPECT_EQ(stamped->getGeometry().tiled_output_outer_stride, 2U);
    EXPECT_EQ(stamped->getGeometry().tiled_output_inner_stride, 1U);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {9.0f, 12.0f, 27.0f, 30.0f});
}

TEST(CubReduction, PermutationAwareTiledPathHonorsViewStorageOffset) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor storage = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f},
                                   {3, 3, 2},
                                   stream);
    // Skip the first physical i-row. getMemPtr() is already offset to the visible dense [2,3,2] region.
    Tensor permuted = storage.aliasView({2, 2, 3}, {1, 6, 2}, 6);
    ASSERT_EQ(permuted.getStorageElementOffset(), 6U);

    auto stamped = CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(permuted, stream);
    ASSERT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
    EXPECT_TRUE(stamped->getGeometry().tiled_output_permuted);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {27.0f, 45.0f, 30.0f, 48.0f});
}

TEST(CubReduction, PermutationAwareTiledPathHandlesRetainedHeightAbove65535) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 65537;
    constexpr uint64_t reduction_size = 2;
    constexpr uint64_t inner_size = 3;
    std::vector<float> values(outer_size * reduction_size * inner_size);
    std::vector<float> expected(inner_size * outer_size, 0.0f);
    for (uint64_t outer = 0; outer < outer_size; ++outer) {
        for (uint64_t reduction = 0; reduction < reduction_size; ++reduction) {
            for (uint64_t inner = 0; inner < inner_size; ++inner) {
                const float value = static_cast<float>((outer % 17) + reduction * 3 + inner);
                values[(outer * reduction_size + reduction) * inner_size + inner] = value;
                expected[inner * outer_size + outer] += value;
            }
        }
    }

    Tensor storage = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream);
    Tensor permuted = storage.aliasView(
        {inner_size, outer_size, reduction_size},
        {1, reduction_size * inner_size, inner_size},
        0);

    auto stamped = CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(permuted, stream);
    ASSERT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
    ASSERT_TRUE(stamped->getGeometry().permutation_aware_tiled_geometry.has_value());
    EXPECT_TRUE(stamped->getGeometry().tiled_output_permuted);
    EXPECT_EQ(stamped->getGeometry().outer_size, outer_size);
    EXPECT_EQ(stamped->getGeometry().inner_size, inner_size);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
}

TEST(CubReduction, PhysicallyContiguousDisjointLogicalAxesUsePermutationAwareTiledExecution) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 2;
    constexpr uint64_t reduction_a = 3;
    constexpr uint64_t reduction_b = 5;
    constexpr uint64_t inner_size = 7;
    std::vector<float> values(outer_size * reduction_a * reduction_b * inner_size);
    std::vector<float> expected(outer_size * inner_size, 0.0f);
    for (uint64_t outer = 0; outer < outer_size; ++outer) {
        for (uint64_t a = 0; a < reduction_a; ++a) {
            for (uint64_t b = 0; b < reduction_b; ++b) {
                for (uint64_t inner = 0; inner < inner_size; ++inner) {
                    const float value = static_cast<float>(outer * 100 + a * 10 + b + inner * 0.25);
                    values[((outer * reduction_a + a) * reduction_b + b) * inner_size + inner] = value;
                    expected[outer * inner_size + inner] += value;
                }
            }
        }
    }

    Tensor storage = makeGpuTensor(values, {outer_size, reduction_a, reduction_b, inner_size}, stream);
    // Physical order [logical 1, logical 0, logical 2, logical 3]. Reduced logical axes {0,2} are
    // disjoint in the view but contiguous in physical storage and can be flattened into one reduction domain.
    Tensor permuted = storage.aliasView(
        {reduction_a, outer_size, reduction_b, inner_size},
        {reduction_b * inner_size, reduction_a * reduction_b * inner_size, inner_size, 1},
        0);

    auto stamped = CubReduction(CubReductionOp::Sum, std::vector<uint32_t>{0, 2}, DataType::FP32)
                       .stamp(permuted, stream);
    ASSERT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
    EXPECT_FALSE(stamped->getGeometry().reduced_axes_are_contiguous);
    ASSERT_TRUE(stamped->getGeometry().permutation_aware_tiled_geometry.has_value());
    EXPECT_FALSE(stamped->getGeometry().tiled_output_permuted);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
}

TEST(CubReduction, RankOneAffineViewUsesDeviceTransformReduceWithoutGeneralIndexMetadata) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor storage = makeGpuTensor({1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f},
                                   {3, 3},
                                   stream);
    Tensor diagonal = storage.aliasView({3}, {4}, 0);

    auto stamped = CubReduction(CubReductionOp::Sum, 0, DataType::FP32).stamp(diagonal, stream);
    EXPECT_EQ(stamped->getPath(), CubReductionPath::DeviceTransformReduce);
    EXPECT_TRUE(stamped->getGeometry().device_transform_uses_affine_stride);
    EXPECT_EQ(stamped->getGeometry().affine_input_stride, 4U);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {15.0f});
}
