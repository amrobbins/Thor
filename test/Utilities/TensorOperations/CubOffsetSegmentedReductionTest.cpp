#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

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

#if THOR_CUB_ENABLE_64BIT_TYPES
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

TEST(CubSegmentedReduction, ValidatesTensorAndOperationContractsAtStampTime) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f}, {4}, stream);
    Tensor offsets = makeGpuUnsignedTensor({0, 2, 4}, {3}, stream);

    EXPECT_THROW(static_cast<void>(CubSegmentedReduction(CubReductionOp::Product)), std::invalid_argument);

    Tensor rank_two_input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, stream);
    EXPECT_THROW(static_cast<void>(CubSegmentedReduction(CubReductionOp::Sum).stamp(
                     rank_two_input, offsets, stream)),
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

    Tensor wrong_output(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2}));
    EXPECT_THROW(static_cast<void>(CubSegmentedReduction(CubReductionOp::Sum, DataType::FP32).stamp(
                     input, wrong_output, offsets, stream)),
                 std::invalid_argument);
}
