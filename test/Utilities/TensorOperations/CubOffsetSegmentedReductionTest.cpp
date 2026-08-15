#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

namespace {

std::vector<float> executeSegmented(const Tensor& input,
                                    const Tensor& offsets,
                                    CubReductionOp op,
                                    Stream& stream,
                                    DataType output_dtype = DataType::FP32) {
    std::shared_ptr<StampedCubSegmentedReduction> stamped =
        CubSegmentedReduction(op, output_dtype).stamp(input, offsets, stream);
    EXPECT_EQ(stamped->getPath(), CubReductionPath::OffsetSegmented);
    EXPECT_EQ(stamped->getAccumulatorDataType(), DataType::FP32);
    stamped->run();
    stream.synchronize();
    return copyGpuTensorAsFloat(stamped->getOutputTensor(), stream);
}

}  // namespace

TEST(CubSegmentedReduction, SumMeanMinAndMaxSupportEmptyAndSkewedSegments) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, -2.0f, 4.0f, 5.0f, 7.0f, -1.0f, 8.0f, 99.0f, 100.0f},
                                 {9},
                                 stream);
    Tensor offsets = makeGpuUnsignedTensor({0, 3, 3, 5, 7}, {5}, stream);

    expectFloatVectorNear(executeSegmented(input, offsets, CubReductionOp::Sum, stream),
                          {3.0f, 0.0f, 12.0f, 7.0f});
    expectFloatVectorNear(executeSegmented(input, offsets, CubReductionOp::Mean, stream),
                          {1.0f, 0.0f, 6.0f, 3.5f});
    expectFloatVectorNear(executeSegmented(input, offsets, CubReductionOp::Min, stream),
                          {-2.0f, std::numeric_limits<float>::infinity(), 5.0f, -1.0f});
    expectFloatVectorNear(executeSegmented(input, offsets, CubReductionOp::Max, stream),
                          {4.0f, -std::numeric_limits<float>::infinity(), 7.0f, 8.0f});
}

TEST(CubSegmentedReduction, ConvertsLowPrecisionInputAndAccumulatesInFp32) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<DataType> input_dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};
#if THOR_CUB_ENABLE_FP8_TYPES
    input_dtypes.insert(input_dtypes.begin(), {DataType::FP8_E4M3, DataType::FP8_E5M2});
#endif
#if THOR_CUB_ENABLE_64BIT_TYPES
    input_dtypes.push_back(DataType::FP64);
#endif
    for (DataType dtype : input_dtypes) {
        SCOPED_TRACE(static_cast<int>(dtype));
        Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f}, {4}, stream, dtype);
        Tensor offsets = makeGpuUnsignedTensor({0, 2, 4}, {3}, stream);
        std::shared_ptr<StampedCubSegmentedReduction> stamped =
            CubSegmentedReduction(CubReductionOp::Sum, DataType::FP32).stamp(input, offsets, stream);
        EXPECT_EQ(stamped->getInputDataType(), dtype);
        EXPECT_EQ(stamped->getOutputDataType(), DataType::FP32);
        EXPECT_EQ(stamped->getAccumulatorDataType(), DataType::FP32);
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {3.0f, 7.0f});
    }
}

#if THOR_CUB_ENABLE_64BIT_SEGMENT_OFFSETS
TEST(CubSegmentedReduction, SupportsUint64SegmentOffsets) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f}, {4}, stream);
    Tensor offsets = makeGpuUnsignedTensor({0, 1, 4}, {3}, stream, DataType::UINT64);

    std::shared_ptr<StampedCubSegmentedReduction> stamped =
        CubSegmentedReduction(CubReductionOp::Sum, DataType::FP32).stamp(input, offsets, stream);
    EXPECT_EQ(stamped->getOffsetDataType(), DataType::UINT64);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {1.0f, 9.0f});
}
#endif

TEST(CubSegmentedArgReduction, GlobalWinnerIndicesMatchDenseTieAndNanPolicyForBothOffsetDTypes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    Tensor input = makeGpuTensor({2.0f, -1.0f, -1.0f, nan, 5.0f, nan, 4.0f, 4.0f, 99.0f}, {9}, stream);
    const std::vector<uint64_t> offsets_values{0, 3, 3, 6, 8};

    for (DataType offsets_dtype : {DataType::UINT32, DataType::UINT64}) {
        SCOPED_TRACE(static_cast<int>(offsets_dtype));
        Tensor offsets = makeGpuUnsignedTensor(offsets_values, {5}, stream, offsets_dtype);

        auto argmin = CubSegmentedArgReduction(CubArgReductionOp::ArgMin).stamp(input, offsets, stream);
        argmin->run();
        stream.synchronize();
        EXPECT_EQ(copyGpuTensorAsUnsigned(argmin->getIndexOutputTensor(), stream),
                  (std::vector<uint64_t>{1ULL, std::numeric_limits<uint64_t>::max(), 3ULL, 6ULL}));

        auto argmax = CubSegmentedArgReduction(CubArgReductionOp::ArgMax).stamp(input, offsets, stream);
        argmax->run();
        stream.synchronize();
        EXPECT_EQ(copyGpuTensorAsUnsigned(argmax->getIndexOutputTensor(), stream),
                  (std::vector<uint64_t>{0ULL, std::numeric_limits<uint64_t>::max(), 3ULL, 6ULL}));
    }
}


TEST(CubSegmentedArgReduction, VectorValuesProduceGlobalPackedWinnerIndicesAndEmptySentinels) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    Tensor input = makeGpuTensor({2.0f, 7.0f, 5.0f,
                                  -1.0f, 7.0f, 6.0f,
                                  -1.0f, 4.0f, nan,
                                  6.0f, 1.0f, 9.0f,
                                  5.0f, 1.0f, 8.0f},
                                 {5, 3},
                                 stream);

    for (DataType offsets_dtype : {DataType::UINT32, DataType::UINT64}) {
        SCOPED_TRACE(static_cast<int>(offsets_dtype));
        Tensor offsets = makeGpuUnsignedTensor({0, 3, 3, 5}, {4}, stream, offsets_dtype);

        auto argmin = CubSegmentedArgReduction(CubArgReductionOp::ArgMin, DataType::UINT64).stamp(input, offsets, stream);
        EXPECT_EQ(argmin->getIndexOutputTensor().getDimensions(), (std::vector<uint64_t>{3, 3}));
        EXPECT_EQ(argmin->getWorkspaceSizeInBytes(), 1U);
        argmin->run();
        stream.synchronize();
        EXPECT_EQ(copyGpuTensorAsUnsigned(argmin->getIndexOutputTensor(), stream),
                  (std::vector<uint64_t>{3ULL, 7ULL, 8ULL,
                                         std::numeric_limits<uint64_t>::max(),
                                         std::numeric_limits<uint64_t>::max(),
                                         std::numeric_limits<uint64_t>::max(),
                                         12ULL, 10ULL, 14ULL}));

        Tensor uint32_winners(gpuPlacement, TensorDescriptor(DataType::UINT32, {3, 3}));
        auto argmax = CubSegmentedArgReduction(CubArgReductionOp::ArgMax, DataType::UINT32)
                          .stamp(input, uint32_winners, offsets, stream);
        argmax->run();
        stream.synchronize();
        EXPECT_EQ(copyGpuTensorAsUnsigned(uint32_winners, stream),
                  (std::vector<uint64_t>{0ULL, 1ULL, 8ULL,
                                         static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                                         static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                                         static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                                         9ULL, 10ULL, 11ULL}));
    }
}

TEST(CubSegmentedArgReduction, VectorValuesConvertLowPrecisionInputAndCompareInFp32) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<DataType> input_dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};
#if THOR_CUB_ENABLE_FP8_TYPES
    input_dtypes.insert(input_dtypes.begin(), {DataType::FP8_E4M3, DataType::FP8_E5M2});
#endif
#if THOR_CUB_ENABLE_64BIT_TYPES
    input_dtypes.push_back(DataType::FP64);
#endif
    for (DataType dtype : input_dtypes) {
        SCOPED_TRACE(static_cast<int>(dtype));
        Tensor input = makeGpuTensor({1.0f, 10.0f,
                                      2.0f, 5.0f,
                                      3.0f, 8.0f,
                                      4.0f, 6.0f},
                                     {4, 2},
                                     stream,
                                     dtype);
        Tensor offsets = makeGpuUnsignedTensor({0, 2, 4}, {3}, stream);
        auto stamped = CubSegmentedArgReduction(CubArgReductionOp::ArgMin, DataType::UINT32).stamp(input, offsets, stream);
        stamped->run();
        stream.synchronize();
        EXPECT_EQ(copyGpuTensorAsUnsigned(stamped->getIndexOutputTensor(), stream),
                  (std::vector<uint64_t>{0ULL, 3ULL, 4ULL, 7ULL}));
    }
}

TEST(CubSegmentedArgReduction, VectorWideTrailingValuesUseMultipleComponentTiles) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t width = 257;
    std::vector<float> values(3 * width);
    for (uint64_t component = 0; component < width; ++component) {
        values[component] = static_cast<float>(component);
        values[width + component] = 1000.0f + static_cast<float>(component);
        values[2 * width + component] = -1000.0f + static_cast<float>(component);
    }
    Tensor input = makeGpuTensor(values, {3, width}, stream);
    Tensor offsets = makeGpuUnsignedTensor({0, 2, 3}, {3}, stream);
    auto stamped = CubSegmentedArgReduction(CubArgReductionOp::ArgMax, DataType::UINT32).stamp(input, offsets, stream);
    stamped->run();
    stream.synchronize();

    const std::vector<uint64_t> actual = copyGpuTensorAsUnsigned(stamped->getIndexOutputTensor(), stream);
    ASSERT_EQ(actual.size(), 2 * width);
    for (uint64_t component = 0; component < width; ++component) {
        EXPECT_EQ(actual[component], width + component);
        EXPECT_EQ(actual[width + component], 2 * width + component);
    }
}

TEST(CubSegmentedArgReduction, VectorStampedReductionReusesOutputAndDynamicOffsets) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 10.0f,
                                  2.0f, 20.0f,
                                  3.0f, 30.0f,
                                  4.0f, 40.0f},
                                 {4, 2},
                                 stream);
    Tensor offsets = makeGpuUnsignedTensor({0, 2, 4}, {3}, stream);
    Tensor winners(gpuPlacement, TensorDescriptor(DataType::UINT32, {2, 2}));
    auto stamped = CubSegmentedArgReduction(CubArgReductionOp::ArgMin, DataType::UINT32)
                       .stamp(input, winners, offsets, stream);
    const void* winner_storage = winners.getMemPtr<void>();
    EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), 1U);

    stamped->run();
    stream.synchronize();
    EXPECT_EQ(copyGpuTensorAsUnsigned(winners, stream), (std::vector<uint64_t>{0ULL, 1ULL, 4ULL, 5ULL}));

    overwriteGpuUnsignedTensor(offsets, {0, 1, 4}, stream);
    stamped->run();
    stream.synchronize();
    EXPECT_EQ(copyGpuTensorAsUnsigned(winners, stream), (std::vector<uint64_t>{0ULL, 1ULL, 2ULL, 3ULL}));
    EXPECT_EQ(winners.getMemPtr<void>(), winner_storage);
}

TEST(CubSegmentedReduction, RuntimeOffsetsAreNotReadUntilExecution) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f}, {4}, stream);
    // Deliberately invalid contents model a network-input allocation before the
    // first runtime row partition has been copied into it. Graph stamping must
    // inspect only the descriptor, not these transient bytes.
    Tensor offsets = makeGpuUnsignedTensor({7, 7, 7}, {3}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {2}));
    auto stamped = CubSegmentedReduction(CubReductionOp::Sum, DataType::FP32)
                       .stampRuntimeOffsets(input, output, offsets, stream);

    overwriteGpuUnsignedTensor(offsets, {0, 2, 4}, stream);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(output, stream), {3.0f, 7.0f});
}

TEST(CubSegmentedArgReduction, RuntimeOffsetsAreNotReadUntilExecution) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({4.0f, 1.0f, 9.0f, 2.0f}, {4}, stream);
    Tensor offsets = makeGpuUnsignedTensor({9, 9, 9}, {3}, stream);
    Tensor winners(gpuPlacement, TensorDescriptor(DataType::UINT64, {2}));
    auto stamped = CubSegmentedArgReduction(CubArgReductionOp::ArgMin, DataType::UINT64)
                       .stampRuntimeOffsets(input, winners, offsets, stream);

    overwriteGpuUnsignedTensor(offsets, {0, 2, 4}, stream);
    stamped->run();
    stream.synchronize();
    EXPECT_EQ(copyGpuTensorAsUnsigned(winners, stream), (std::vector<uint64_t>{1ULL, 3ULL}));
}

TEST(CubSegmentedReduction, ReusesOutputWorkspaceAndDynamicOffsetsAcrossExecutions) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {6}, stream);
    Tensor offsets = makeGpuUnsignedTensor({0, 2, 2, 6}, {4}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {3}));

    std::shared_ptr<StampedCubSegmentedReduction> stamped =
        CubSegmentedReduction(CubReductionOp::Mean, DataType::FP32).stamp(input, output, offsets, stream);
    const void* output_storage = output.getMemPtr<void>();
    const size_t workspace_bytes = stamped->getWorkspaceSizeInBytes();

    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(output, stream), {1.5f, 0.0f, 4.5f});

    overwriteGpuUnsignedTensor(offsets, {0, 1, 4, 6}, stream);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(output, stream), {1.0f, 3.0f, 5.5f});
    EXPECT_EQ(output.getMemPtr<void>(), output_storage);
    EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), workspace_bytes);
}


TEST(CubSegmentedReduction, VectorValuesPreserveTrailingShapeAndEmptySegmentSemantics) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                  10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f,
                                  -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f,
                                  7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                                  2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f},
                                 {5, 2, 3},
                                 stream);
    Tensor offsets = makeGpuUnsignedTensor({0, 2, 2, 5}, {4}, stream);

    auto run = [&](CubReductionOp op) {
        auto stamped = CubSegmentedReduction(op, DataType::FP32).stamp(input, offsets, stream);
        EXPECT_EQ(stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{3, 2, 3}));
        EXPECT_EQ(stamped->getPath(), CubReductionPath::OffsetSegmented);
        stamped->run();
        stream.synchronize();
        return copyGpuTensorAsFloat(stamped->getOutputTensor(), stream);
    };

    expectFloatVectorNear(run(CubReductionOp::Sum),
                          {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f,
                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                           8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f});
    expectFloatVectorNear(run(CubReductionOp::Mean),
                          {5.5f, 11.0f, 16.5f, 22.0f, 27.5f, 33.0f,
                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                           8.0f / 3.0f, 3.0f, 10.0f / 3.0f, 11.0f / 3.0f, 4.0f, 13.0f / 3.0f},
                          1.0e-5f);
    expectFloatVectorNear(run(CubReductionOp::Min),
                          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                           std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                           -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f});
    expectFloatVectorNear(run(CubReductionOp::Max),
                          {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f,
                           -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                           -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                           -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                           7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
}

TEST(CubSegmentedReduction, VectorValuesConvertLowPrecisionInputAndAccumulateInFp32) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<DataType> input_dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};
#if THOR_CUB_ENABLE_FP8_TYPES
    input_dtypes.insert(input_dtypes.begin(), {DataType::FP8_E4M3, DataType::FP8_E5M2});
#endif
#if THOR_CUB_ENABLE_64BIT_TYPES
    input_dtypes.push_back(DataType::FP64);
#endif
    for (DataType dtype : input_dtypes) {
        SCOPED_TRACE(static_cast<int>(dtype));
        // Keep source values exactly representable in every enabled storage format, including FP8 E5M2.
        // This test is for storage->FP32 conversion and FP32 accumulation, not quantization error.
        Tensor input = makeGpuTensor({1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 24.0f, 4.0f, 40.0f},
                                     {4, 2},
                                     stream,
                                     dtype);
        Tensor offsets = makeGpuUnsignedTensor({0, 2, 4}, {3}, stream);
        auto stamped = CubSegmentedReduction(CubReductionOp::Sum, DataType::FP32).stamp(input, offsets, stream);
        EXPECT_EQ(stamped->getInputDataType(), dtype);
        EXPECT_EQ(stamped->getOutputDataType(), DataType::FP32);
        EXPECT_EQ(stamped->getAccumulatorDataType(), DataType::FP32);
        EXPECT_EQ(stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 2}));
        stamped->run();
        stream.synchronize();
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {3.0f, 30.0f, 7.0f, 64.0f});
    }
}

TEST(CubSegmentedReduction, VectorWideTrailingValuesUseMultipleComponentTiles) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t width = 257;
    std::vector<float> values(3 * width);
    for (uint64_t row = 0; row < 3; ++row) {
        for (uint64_t component = 0; component < width; ++component) {
            values[row * width + component] = static_cast<float>(100 * row + component);
        }
    }
    Tensor input = makeGpuTensor(values, {3, width}, stream);
    Tensor offsets = makeGpuUnsignedTensor({0, 2, 3}, {3}, stream);
    auto stamped = CubSegmentedReduction(CubReductionOp::Sum, DataType::FP32).stamp(input, offsets, stream);
    stamped->run();
    stream.synchronize();

    std::vector<float> expected(2 * width);
    for (uint64_t component = 0; component < width; ++component) {
        expected[component] = 100.0f + 2.0f * static_cast<float>(component);
        expected[width + component] = 200.0f + static_cast<float>(component);
    }
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), expected);
}

#if THOR_CUB_ENABLE_64BIT_SEGMENT_OFFSETS
TEST(CubSegmentedReduction, VectorValuesSupportUint64SegmentOffsets) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f}, {3, 2}, stream);
    Tensor offsets = makeGpuUnsignedTensor({0, 1, 3}, {3}, stream, DataType::UINT64);
    auto stamped = CubSegmentedReduction(CubReductionOp::Sum, DataType::FP32).stamp(input, offsets, stream);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {1.0f, 10.0f, 5.0f, 50.0f});
}
#endif

TEST(CubSegmentedReduction, VectorMinAndMaxPropagateNanPerComponent) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Tensor input = makeGpuTensor({3.0f, 4.0f, 5.0f,
                                  2.0f, nan, 7.0f,
                                  1.0f, 6.0f, nan},
                                 {3, 3},
                                 stream);
    Tensor offsets = makeGpuUnsignedTensor({0, 3}, {2}, stream);

    for (CubReductionOp op : {CubReductionOp::Min, CubReductionOp::Max}) {
        auto stamped = CubSegmentedReduction(op, DataType::FP32).stamp(input, offsets, stream);
        stamped->run();
        stream.synchronize();
        const std::vector<float> result = copyGpuTensorAsFloat(stamped->getOutputTensor(), stream);
        ASSERT_EQ(result.size(), 3U);
        EXPECT_FLOAT_EQ(result[0], op == CubReductionOp::Min ? 1.0f : 3.0f);
        EXPECT_TRUE(std::isnan(result[1]));
        EXPECT_TRUE(std::isnan(result[2]));
    }
}

TEST(CubSegmentedReduction, VectorStampedReductionReusesOutputAndDynamicOffsets) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f,
                                  4.0f, 40.0f, 5.0f, 50.0f, 6.0f, 60.0f},
                                 {6, 2},
                                 stream);
    Tensor offsets = makeGpuUnsignedTensor({0, 2, 2, 6}, {4}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {3, 2}));

    auto stamped = CubSegmentedReduction(CubReductionOp::Mean, DataType::FP32).stamp(input, output, offsets, stream);
    const void* output_storage = output.getMemPtr<void>();
    EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), 1U);

    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(output, stream), {1.5f, 15.0f, 0.0f, 0.0f, 4.5f, 45.0f});

    overwriteGpuUnsignedTensor(offsets, {0, 1, 4, 6}, stream);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(output, stream), {1.0f, 10.0f, 3.0f, 30.0f, 5.5f, 55.0f});
    EXPECT_EQ(output.getMemPtr<void>(), output_storage);
}

TEST(CubSegmentedReduction, ValidatesTensorAndOperationContractsAtStampTime) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f}, {4}, stream);
    Tensor offsets = makeGpuUnsignedTensor({0, 2, 4}, {3}, stream);

    EXPECT_THROW(static_cast<void>(CubSegmentedReduction(CubReductionOp::Product)), std::invalid_argument);

    Tensor rank_two_input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, stream);
    Tensor vector_row_out_of_bounds = makeGpuUnsignedTensor({0, 1, 3}, {3}, stream);
    EXPECT_THROW(static_cast<void>(CubSegmentedReduction(CubReductionOp::Sum).stamp(
                     rank_two_input, vector_row_out_of_bounds, stream)),
                 std::invalid_argument);

    Tensor floating_offsets = makeGpuTensor({0.0f, 2.0f, 4.0f}, {3}, stream);
    EXPECT_THROW(static_cast<void>(CubSegmentedReduction(CubReductionOp::Sum).stamp(
                     input, floating_offsets, stream)),
                 std::invalid_argument);

    Tensor short_offsets = makeGpuUnsignedTensor({0}, {1}, stream);
    EXPECT_THROW(static_cast<void>(CubSegmentedReduction(CubReductionOp::Sum).stamp(
                     input, short_offsets, stream)),
                 std::invalid_argument);

    Tensor nonzero_start = makeGpuUnsignedTensor({1, 2, 4}, {3}, stream);
    EXPECT_THROW(static_cast<void>(CubSegmentedReduction(CubReductionOp::Sum).stamp(
                     input, nonzero_start, stream)),
                 std::invalid_argument);

    Tensor nonmonotonic = makeGpuUnsignedTensor({0, 3, 2}, {3}, stream);
    EXPECT_THROW(static_cast<void>(CubSegmentedReduction(CubReductionOp::Sum).stamp(
                     input, nonmonotonic, stream)),
                 std::invalid_argument);

    Tensor out_of_bounds = makeGpuUnsignedTensor({0, 2, 5}, {3}, stream);
    EXPECT_THROW(static_cast<void>(CubSegmentedReduction(CubReductionOp::Sum).stamp(
                     input, out_of_bounds, stream)),
                 std::invalid_argument);

    Tensor vector_offsets = makeGpuUnsignedTensor({0, 1, 2}, {3}, stream);
    Tensor wrong_vector_output(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 1, 2}));
    EXPECT_THROW(static_cast<void>(CubSegmentedReduction(CubReductionOp::Sum, DataType::FP32).stamp(
                     rank_two_input, wrong_vector_output, vector_offsets, stream)),
                 std::invalid_argument);

    Tensor wrong_output(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2}));
    EXPECT_THROW(static_cast<void>(CubSegmentedReduction(CubReductionOp::Sum, DataType::FP32).stamp(
                     input, wrong_output, offsets, stream)),
                 std::invalid_argument);

    Tensor wrong_arg_output(gpuPlacement, TensorDescriptor(DataType::UINT32, {2}));
    EXPECT_THROW(static_cast<void>(CubSegmentedArgReduction(CubArgReductionOp::ArgMin, DataType::UINT32).stamp(
                     rank_two_input, wrong_arg_output, vector_offsets, stream)),
                 std::invalid_argument);
}
