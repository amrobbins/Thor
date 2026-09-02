#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

TEST(CubReduction, DefinesExplicitFp32EmptyReductionValues) {
    EXPECT_FLOAT_EQ(CubReduction::getFp32EmptyReductionValue(CubReductionOp::Sum), 0.0f);
    EXPECT_FLOAT_EQ(CubReduction::getFp32EmptyReductionValue(CubReductionOp::Product), 1.0f);
    EXPECT_FLOAT_EQ(CubReduction::getFp32EmptyReductionValue(CubReductionOp::Mean), 0.0f);
    EXPECT_FLOAT_EQ(CubReduction::getFp32EmptyReductionValue(CubReductionOp::L1Norm), 0.0f);
    EXPECT_FLOAT_EQ(CubReduction::getFp32EmptyReductionValue(CubReductionOp::L2Norm), 0.0f);
    EXPECT_FLOAT_EQ(CubReduction::getFp32EmptyReductionValue(CubReductionOp::SumSquares), 0.0f);
    EXPECT_EQ(CubReduction::getFp32EmptyReductionValue(CubReductionOp::Min),
              std::numeric_limits<float>::infinity());
    EXPECT_EQ(CubReduction::getFp32EmptyReductionValue(CubReductionOp::Max),
              -std::numeric_limits<float>::infinity());
}

TEST(CubReduction, WholeTensorPathSupportsEveryValueOperation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({-2.0f, 3.0f, -4.0f}, {3}, stream);

    expectOperations(input,
                     0,
                     {{CubReductionOp::Sum, {-3.0f}, 0.0f},
                      {CubReductionOp::Product, {24.0f}, 0.0f},
                      {CubReductionOp::Mean, {-1.0f}, 0.0f},
                      {CubReductionOp::Min, {-4.0f}, 0.0f},
                      {CubReductionOp::Max, {3.0f}, 0.0f},
                      {CubReductionOp::L1Norm, {9.0f}, 0.0f},
                      {CubReductionOp::L2Norm, {std::sqrt(29.0f)}, 1.0e-5f},
                      {CubReductionOp::SumSquares, {29.0f}, 0.0f}},
                     stream);
}

TEST(CubReduction, ContiguousFixedSegmentPathSupportsEveryValueOperation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f}, {2, 3}, stream);

    expectOperations(input,
                     1,
                     {{CubReductionOp::Sum, {-2.0f, 5.0f}, 0.0f},
                      {CubReductionOp::Product, {6.0f, -120.0f}, 0.0f},
                      {CubReductionOp::Mean, {-2.0f / 3.0f, 5.0f / 3.0f}, 1.0e-6f},
                      {CubReductionOp::Min, {-3.0f, -5.0f}, 0.0f},
                      {CubReductionOp::Max, {2.0f, 6.0f}, 0.0f},
                      {CubReductionOp::L1Norm, {6.0f, 15.0f}, 0.0f},
                      {CubReductionOp::L2Norm, {std::sqrt(14.0f), std::sqrt(77.0f)}, 1.0e-5f},
                      {CubReductionOp::SumSquares, {14.0f, 77.0f}, 0.0f}},
                     stream);
}

TEST(CubReduction, TiledFixedSegmentPathSupportsEveryValueOperation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor(
        {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f, -9.0f, 10.0f, -11.0f, 12.0f},
        {2, 3, 2},
        stream);

    expectOperations(input,
                     1,
                     {{CubReductionOp::Sum, {-9.0f, 12.0f, -27.0f, 30.0f}, 0.0f},
                      {CubReductionOp::Product, {-15.0f, 48.0f, -693.0f, 960.0f}, 0.0f},
                      {CubReductionOp::Mean, {-3.0f, 4.0f, -9.0f, 10.0f}, 0.0f},
                      {CubReductionOp::Min, {-5.0f, 2.0f, -11.0f, 8.0f}, 0.0f},
                      {CubReductionOp::Max, {-1.0f, 6.0f, -7.0f, 12.0f}, 0.0f},
                      {CubReductionOp::L1Norm, {9.0f, 12.0f, 27.0f, 30.0f}, 0.0f},
                      {CubReductionOp::L2Norm,
                       {std::sqrt(35.0f), std::sqrt(56.0f), std::sqrt(251.0f), std::sqrt(308.0f)},
                       1.0e-5f},
                      {CubReductionOp::SumSquares, {35.0f, 56.0f, 251.0f, 308.0f}, 0.0f}},
                     stream);
}

TEST(CubReduction, PermutationAwareTiledPathSupportsEveryValueOperation) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor storage = makeGpuTensor(
        {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f, -9.0f, 10.0f, -11.0f, 12.0f},
        {2, 3, 2},
        stream);
    Tensor permuted = storage.aliasView({2, 2, 3}, {1, 6, 2}, 0);

    expectOperations(permuted,
                     2,
                     {{CubReductionOp::Sum, {-9.0f, -27.0f, 12.0f, 30.0f}, 0.0f},
                      {CubReductionOp::Product, {-15.0f, -693.0f, 48.0f, 960.0f}, 0.0f},
                      {CubReductionOp::Mean, {-3.0f, -9.0f, 4.0f, 10.0f}, 0.0f},
                      {CubReductionOp::Min, {-5.0f, -11.0f, 2.0f, 8.0f}, 0.0f},
                      {CubReductionOp::Max, {-1.0f, -7.0f, 6.0f, 12.0f}, 0.0f},
                      {CubReductionOp::L1Norm, {9.0f, 27.0f, 12.0f, 30.0f}, 0.0f},
                      {CubReductionOp::L2Norm,
                       {std::sqrt(35.0f), std::sqrt(251.0f), std::sqrt(56.0f), std::sqrt(308.0f)},
                       1.0e-5f},
                      {CubReductionOp::SumSquares, {35.0f, 251.0f, 56.0f, 308.0f}, 0.0f}},
                     stream);
}

TEST(CubReduction, TiledFixedSegmentCanCoalescePermutedDenseOutputWithoutIntermediate) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer = 35;
    constexpr uint64_t reduction = 3;
    constexpr uint64_t inner = 37;
    std::vector<float> values(outer * reduction * inner);
    std::vector<float> expected(inner * outer, 0.0f);
    for (uint64_t i = 0; i < outer; ++i) {
        for (uint64_t j = 0; j < reduction; ++j) {
            for (uint64_t k = 0; k < inner; ++k) {
                const float value = static_cast<float>((i % 7) * 0.5 + j * 2.0 + (k % 11) * 0.25);
                values[(i * reduction + j) * inner + k] = value;
                expected[k * outer + i] += value;
            }
        }
    }

    Tensor storage = makeGpuTensor(values, {outer, reduction, inner}, stream);
    Tensor permuted = storage.aliasView(
        {inner, outer, reduction}, {1, reduction * inner, inner}, 0);
    Tensor squeezed_output(gpuPlacement, TensorDescriptor(DataType::FP32, {inner, outer}));
    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(permuted, squeezed_output, stream);

    ASSERT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
    EXPECT_TRUE(stamped->getGeometry().tiled_output_permuted);
    EXPECT_TRUE(stamped->getGeometry().tiled_output_shared_transpose);
    EXPECT_EQ(stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{inner, outer}));
    EXPECT_EQ(stamped->getOutputTensor().getMemPtr<void>(), squeezed_output.getMemPtr<void>());

    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
}

TEST(CubReduction, PermutationAwareExactWarpWidthHandlesShortReduction) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer = 11;
    constexpr uint64_t reduction = 2;
    constexpr uint64_t inner = 32;
    std::vector<float> values(outer * reduction * inner);
    std::vector<float> expected(inner * outer, 0.0f);
    for (uint64_t i = 0; i < outer; ++i) {
        for (uint64_t j = 0; j < reduction; ++j) {
            for (uint64_t k = 0; k < inner; ++k) {
                const float value = static_cast<float>(static_cast<int>((i * 3 + j * 5 + k * 7) % 13) - 6);
                values[(i * reduction + j) * inner + k] = value;
                expected[k * outer + i] += value;
            }
        }
    }

    Tensor storage = makeGpuTensor(values, {outer, reduction, inner}, stream);
    Tensor permuted = storage.aliasView(
        {inner, outer, reduction}, {1, reduction * inner, inner}, 0);
    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(permuted, stream);

    ASSERT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
    ASSERT_TRUE(stamped->getGeometry().tiled_output_shared_transpose);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
}

TEST(CubReduction, SharedTransposePermutedTiledWriterSupportsStorageDTypesAndTailTiles) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer = 513;  // enough retained tiles to exercise one-warp/component-tile production geometry
    constexpr uint64_t reduction = 2;
    constexpr uint64_t inner = 65;   // both component and outer dimensions have partial final tiles
    std::vector<float> values(outer * reduction * inner);
    std::vector<float> expected(inner * outer, 0.0f);
    for (uint64_t i = 0; i < outer; ++i) {
        for (uint64_t j = 0; j < reduction; ++j) {
            for (uint64_t k = 0; k < inner; ++k) {
                const float value = static_cast<float>(static_cast<int>((i + j * 3 + k * 5) % 9) - 4) * 0.5f;
                values[(i * reduction + j) * inner + k] = value;
                expected[k * outer + i] += value;
            }
        }
    }

    std::vector<DataType> input_dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};
#if THOR_CUB_ENABLE_FP8_TYPES
    input_dtypes.push_back(DataType::FP8_E4M3);
    input_dtypes.push_back(DataType::FP8_E5M2);
#endif
#if THOR_CUB_ENABLE_64BIT_TYPES
    input_dtypes.push_back(DataType::FP64);
#endif
    for (DataType dtype : input_dtypes) {
        SCOPED_TRACE(static_cast<int>(dtype));
        Tensor storage = makeGpuTensor(values, {outer, reduction, inner}, stream, dtype);
        Tensor input = storage.aliasView(
            {inner, outer, reduction}, {1, reduction * inner, inner}, 0);
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(input, stream);
        ASSERT_TRUE(stamped->getGeometry().tiled_output_shared_transpose);
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected, 1.0e-4f);
    }
}

TEST(CubReduction, SharedTransposePermutedTiledWriterCoversRetainedWidthRegimes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer = 9;
    constexpr uint64_t reduction = 3;
    for (uint64_t inner : {2ULL,
                           31ULL,
                           32ULL,
                           33ULL,
                           64ULL,
                           65ULL,
                           128ULL,
                           129ULL,
                           256ULL,
                           257ULL,
                           511ULL,
                           512ULL,
                           513ULL,
                           1024ULL,
                           2048ULL,
                           4096ULL,
                           4097ULL,
                           8193ULL}) {
        SCOPED_TRACE(inner);
        std::vector<float> values(outer * reduction * inner);
        std::vector<float> expected(inner * outer, 0.0f);
        for (uint64_t i = 0; i < outer; ++i) {
            for (uint64_t j = 0; j < reduction; ++j) {
                for (uint64_t k = 0; k < inner; ++k) {
                    const float value = static_cast<float>(static_cast<int>((i * 7 + j * 5 + k * 3) % 17) - 8)
                                        * 0.125f;
                    values[(i * reduction + j) * inner + k] = value;
                    expected[k * outer + i] += value;
                }
            }
        }

        Tensor storage = makeGpuTensor(values, {outer, reduction, inner}, stream);
        Tensor permuted = storage.aliasView(
            {inner, outer, reduction}, {1, reduction * inner, inner}, 0);
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(permuted, stream);
        ASSERT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
        ASSERT_TRUE(stamped->getGeometry().tiled_output_shared_transpose);

        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected, 1.0e-5f);
    }
}

TEST(CubReduction, SharedTransposePermutedTiledWriterHonorsDenseViewStorageOffset) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer = 9;
    constexpr uint64_t reduction = 5;
    constexpr uint64_t inner = 35;
    constexpr uint64_t offset = 17;
    const uint64_t visible_elements = outer * reduction * inner;
    std::vector<float> storage_values(offset + visible_elements + 11, -999.0f);
    std::vector<float> expected(inner * outer, 0.0f);
    for (uint64_t i = 0; i < outer; ++i) {
        for (uint64_t j = 0; j < reduction; ++j) {
            for (uint64_t k = 0; k < inner; ++k) {
                const float value = static_cast<float>(static_cast<int>((i * 3 + j * 7 + k) % 11) - 5);
                storage_values[offset + (i * reduction + j) * inner + k] = value;
                expected[k * outer + i] += value;
            }
        }
    }

    Tensor storage = makeGpuTensor(storage_values, {storage_values.size()}, stream);
    Tensor input = storage.aliasView(
        {inner, outer, reduction}, {1, reduction * inner, inner}, offset);
    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(input, stream);
    ASSERT_TRUE(stamped->getGeometry().tiled_output_shared_transpose);

    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
}

TEST(CubReduction, SharedTransposePermutedTiledWriterGridStridesPast65535OuterRows) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer = 65537;
    constexpr uint64_t reduction = 2;
    constexpr uint64_t inner = 33;
    std::vector<float> values(outer * reduction * inner, 1.0f);
    Tensor storage = makeGpuTensor(values, {outer, reduction, inner}, stream);
    Tensor input = storage.aliasView(
        {inner, outer, reduction}, {1, reduction * inner, inner}, 0);
    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(input, stream);
    ASSERT_TRUE(stamped->getGeometry().tiled_output_shared_transpose);

    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
                          std::vector<float>(inner * outer, 2.0f));
}

TEST(CubReduction, TiledFixedSegmentCoversWidthPoliciesAndAsyncStaging) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 2;
    constexpr uint64_t reduction_size = 65;
    for (uint64_t inner_size :
         {2ULL, 3ULL, 4ULL, 5ULL, 8ULL, 9ULL, 16ULL, 17ULL, 31ULL, 32ULL, 33ULL, 63ULL, 64ULL, 65ULL, 127ULL, 128ULL, 129ULL, 255ULL, 256ULL, 257ULL, 511ULL, 512ULL, 513ULL, 1023ULL, 1024ULL, 1025ULL, 2047ULL, 2048ULL, 2049ULL, 4095ULL, 4096ULL, 4097ULL}) {
        SCOPED_TRACE(inner_size);
        std::vector<float> values(outer_size * reduction_size * inner_size);
        std::vector<float> expected(outer_size * inner_size, 0.0f);
        for (uint64_t outer = 0; outer < outer_size; ++outer) {
            for (uint64_t row = 0; row < reduction_size; ++row) {
                for (uint64_t component = 0; component < inner_size; ++component) {
                    const int value = static_cast<int>((outer * 3 + row * 5 + component * 7) % 11) - 5;
                    const uint64_t input_index = (outer * reduction_size + row) * inner_size + component;
                    values[input_index] = static_cast<float>(value);
                    expected[outer * inner_size + component] += static_cast<float>(value);
                }
            }
        }

        Tensor input = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream);
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 1, DataType::FP32).stamp(input, stream);
        EXPECT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
        EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), 1U);
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
    }
}

TEST(CubReduction, PermutationAwareTiledPathCoversTunedWidthPolicies) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 3;
    constexpr uint64_t reduction_size = 65;
    for (uint64_t inner_size :
         {2ULL, 3ULL, 5ULL, 17ULL, 32ULL, 33ULL, 63ULL, 64ULL, 65ULL, 127ULL, 128ULL, 129ULL, 255ULL, 256ULL,
          257ULL, 511ULL, 512ULL, 513ULL, 1024ULL, 2048ULL, 4096ULL, 4097ULL}) {
        SCOPED_TRACE(inner_size);
        std::vector<float> values(outer_size * reduction_size * inner_size);
        std::vector<float> expected(inner_size * outer_size, 0.0f);
        for (uint64_t outer = 0; outer < outer_size; ++outer) {
            for (uint64_t row = 0; row < reduction_size; ++row) {
                for (uint64_t component = 0; component < inner_size; ++component) {
                    const int value = static_cast<int>((outer * 3 + row * 5 + component * 7) % 11) - 5;
                    values[(outer * reduction_size + row) * inner_size + component] = static_cast<float>(value);
                    expected[component * outer_size + outer] += static_cast<float>(value);
                }
            }
        }

        Tensor storage = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream);
        Tensor permuted = storage.aliasView(
            {inner_size, outer_size, reduction_size},
            {1, reduction_size * inner_size, inner_size},
            0);
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 2, DataType::FP32).stamp(permuted, stream);
        ASSERT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
        ASSERT_TRUE(stamped->getGeometry().permutation_aware_tiled_geometry.has_value());
        EXPECT_TRUE(stamped->getGeometry().tiled_output_permuted);
        EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), 1U);
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
    }
}

TEST(CubReduction, TiledFixedSegmentBlockShardingScalesBeyondOneBlockWidth) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t outer_size = 1;
    constexpr uint64_t reduction_size = 7;
    for (uint64_t inner_size : {4097ULL, 6144ULL, 8191ULL, 8192ULL, 8193ULL, 16384ULL, 16385ULL, 65536ULL, 65537ULL}) {
        SCOPED_TRACE(inner_size);
        std::vector<float> values(outer_size * reduction_size * inner_size);
        std::vector<float> expected(outer_size * inner_size, 0.0f);
        for (uint64_t row = 0; row < reduction_size; ++row) {
            for (uint64_t component = 0; component < inner_size; ++component) {
                const int value = static_cast<int>((row * 5 + component * 7) % 11) - 5;
                values[row * inner_size + component] = static_cast<float>(value);
                expected[component] += static_cast<float>(value);
            }
        }

        Tensor input = makeGpuTensor(values, {outer_size, reduction_size, inner_size}, stream);
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 1, DataType::FP32).stamp(input, stream);
        EXPECT_EQ(stamped->getPath(), CubReductionPath::TiledFixedSegment);
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
    }
}

TEST(CubReduction, NewOperationsDefaultOutputStorageToInputDtype) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f}, {2, 3}, stream, DataType::BF16);

    const std::vector<OperationExpectation> expectations = {
        {CubReductionOp::Product, {6.0f, -6.0f}, 0.0f},
        {CubReductionOp::Mean, {2.0f, -2.0f}, 0.0f},
        {CubReductionOp::L1Norm, {6.0f, 6.0f}, 0.0f},
        {CubReductionOp::L2Norm, {std::sqrt(14.0f), std::sqrt(14.0f)}, 2.0e-2f},
        {CubReductionOp::SumSquares, {14.0f, 14.0f}, 2.0e-2f},
    };

    for (const OperationExpectation& expectation : expectations) {
        SCOPED_TRACE(static_cast<int>(expectation.op));
        std::shared_ptr<StampedCubReduction> stamped = CubReduction(expectation.op, 1).stamp(input, stream);
        EXPECT_EQ(stamped->getOutputDataType(), DataType::BF16);
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream),
                              expectation.expected,
                              expectation.tolerance);
    }
}

TEST(CubReduction, MeanAndL2FinalizeInFp32BeforeFp16StorageConversion) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor mean_input = makeGpuTensor({40000.0f, 40000.0f, -40000.0f, -40000.0f}, {2, 2}, stream);
    std::shared_ptr<StampedCubReduction> mean =
        CubReduction(CubReductionOp::Mean, 1, DataType::FP16).stamp(mean_input, stream);
    mean->run();

    Tensor l2_input = makeGpuTensor({300.0f, 400.0f, 500.0f, 1200.0f}, {2, 2}, stream);
    std::shared_ptr<StampedCubReduction> l2 =
        CubReduction(CubReductionOp::L2Norm, 1, DataType::FP16).stamp(l2_input, stream);
    l2->run();
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat(mean->getOutputTensor(), stream), {40000.0f, -40000.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat(l2->getOutputTensor(), stream), {500.0f, 1300.0f});
}

TEST(CubReduction, MinimumAndMaximumPropagateNanAcrossAllPaths) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const float nan = std::numeric_limits<float>::quiet_NaN();

    Tensor whole = makeGpuTensor({1.0f, nan, -2.0f}, {3}, stream);
    Tensor contiguous = makeGpuTensor({nan, 1.0f, 2.0f, 3.0f, nan, 4.0f}, {2, 3}, stream);
    Tensor tiled = makeGpuTensor(
        {1.0f, 2.0f, nan, nan, 5.0f, 6.0f, 7.0f, 8.0f, nan, nan, 11.0f, 12.0f},
        {2, 3, 2},
        stream);

    for (CubReductionOp op : {CubReductionOp::Min, CubReductionOp::Max}) {
        SCOPED_TRACE(static_cast<int>(op));
        expectFloatVectorNear(executeFp32Output(whole, op, 0, stream), {nan});
        expectFloatVectorNear(executeFp32Output(contiguous, op, 1, stream), {nan, nan});
        expectFloatVectorNear(executeFp32Output(tiled, op, 1, stream), {nan, nan, nan, nan});
    }
}

TEST(CubReduction, ArithmeticAndNormOperationsNaturallyPropagateNan) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Tensor input = makeGpuTensor({1.0f, nan, 3.0f}, {3}, stream);

    for (CubReductionOp op : {CubReductionOp::Sum,
                              CubReductionOp::Product,
                              CubReductionOp::Mean,
                              CubReductionOp::L1Norm,
                              CubReductionOp::L2Norm,
                              CubReductionOp::SumSquares}) {
        SCOPED_TRACE(static_cast<int>(op));
        expectFloatVectorNear(executeFp32Output(input, op, 0, stream), {nan});
    }
}

TEST(CubReduction, HandlesPositiveAndNegativeInfinity) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const float infinity = std::numeric_limits<float>::infinity();
    Tensor input = makeGpuTensor({-infinity, 1.0f, infinity}, {3}, stream);

    expectFloatVectorNear(executeFp32Output(input, CubReductionOp::Min, 0, stream), {-infinity});
    expectFloatVectorNear(executeFp32Output(input, CubReductionOp::Max, 0, stream), {infinity});
    expectFloatVectorNear(executeFp32Output(input, CubReductionOp::L1Norm, 0, stream), {infinity});
    expectFloatVectorNear(executeFp32Output(input, CubReductionOp::L2Norm, 0, stream), {infinity});
}

TEST(CubReduction, StampedReductionAcceptsRuntimeOutputScale) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f}, {3}, stream);
    std::shared_ptr<StampedCubReduction> reduction =
        CubReduction(CubReductionOp::Sum, 0, DataType::FP32, 1.0f).stamp(input, stream);

    reduction->run(0.5f);
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(reduction->getOutputTensor(), stream), {3.0f});

    reduction->runOn(stream, 2.0f);
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(reduction->getOutputTensor(), stream), {12.0f});
}
