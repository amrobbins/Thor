#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

TEST(CubReduction, EverySupportedInputStorageDtypeAccumulatesInFp32) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    expectInputStorageAccumulatesInFp32(DataType::FP32, stream);
    expectInputStorageAccumulatesInFp32(DataType::FP16, stream);
    expectInputStorageAccumulatesInFp32(DataType::BF16, stream);
#if THOR_CUB_ENABLE_FP8_TYPES
    expectInputStorageAccumulatesInFp32(DataType::FP8_E4M3, stream);
    expectInputStorageAccumulatesInFp32(DataType::FP8_E5M2, stream);
#endif
#if THOR_CUB_ENABLE_64BIT_TYPES
    expectInputStorageAccumulatesInFp32(DataType::FP64, stream);
#endif
}

TEST(CubReduction, WholeTensorBf16SumAccumulatesInFp32AndDefaultsOutputToInputDtype) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    std::vector<float> values(257, 1.0f);
    values[0] = 256.0f;
    Tensor input = makeGpuTensor(values, {257}, stream, DataType::BF16);

    CubReduction reduction(CubReductionOp::Sum, 0);
    std::shared_ptr<StampedCubReduction> stamped = reduction.stamp(input, stream);

    EXPECT_EQ(stamped->getPath(), CubReductionPath::DeviceTransformReduce);
    EXPECT_EQ(stamped->getInputDataType(), DataType::BF16);
    EXPECT_EQ(stamped->getOutputDataType(), DataType::BF16);
    EXPECT_EQ(stamped->getAccumulatorDataType(), DataType::FP32);
    EXPECT_EQ(stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1}));
    EXPECT_GT(stamped->getWorkspaceSizeInBytes(), 0U);

    stamped->run();
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {512.0f});
}

TEST(CubReduction, ReusesStampedOutputAndWorkspaceAcrossExecutions) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, stream, DataType::BF16);
    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::L1Norm, 1, DataType::FP32).stamp(input, stream);
    const uint64_t output_tensor_id = stamped->getOutputTensor().getTensorId();
    const size_t workspace_bytes = stamped->getWorkspaceSizeInBytes();

    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {3.0f, 7.0f});

    Tensor replacement = makeGpuTensor({-10.0f, 20.0f, -30.0f, 40.0f}, {2, 2}, stream, DataType::BF16);
    input.copyFromAsync(replacement, stream);
    stamped->run();
    stream.synchronize();

    EXPECT_EQ(stamped->getOutputTensor().getTensorId(), output_tensor_id);
    EXPECT_EQ(stamped->getWorkspaceSizeInBytes(), workspace_bytes);
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {30.0f, 70.0f});
}

TEST(CubReduction, ValidatesPreallocatedOutputContract) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, stream, DataType::BF16);
    CubReduction reduction(CubReductionOp::Mean, 1, DataType::FP32);

    Tensor correct_output(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 1}));
    std::shared_ptr<StampedCubReduction> stamped = reduction.stamp(input, correct_output, stream);
    EXPECT_EQ(stamped->getOutputTensor().getTensorId(), correct_output.getTensorId());

    Tensor wrong_dtype(gpuPlacement, TensorDescriptor(DataType::BF16, {2, 1}));
    EXPECT_THROW(static_cast<void>(reduction.stamp(input, wrong_dtype, stream)), std::invalid_argument);

    Tensor wrong_shape(gpuPlacement, TensorDescriptor(DataType::FP32, {1, 3}));
    EXPECT_THROW(static_cast<void>(reduction.stamp(input, wrong_shape, stream)), std::invalid_argument);

    Tensor scalar = makeGpuTensor({1.0f}, {1}, stream, DataType::BF16);
    EXPECT_THROW(static_cast<void>(CubReduction(CubReductionOp::Sum, 0).stamp(scalar, scalar, stream)),
                 std::invalid_argument);
}

#if THOR_CUB_ENABLE_FP8_TYPES
TEST(CubReduction, Fp8OutputsUseExplicitSaturatingFp32Conversion) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor({40000.0f, 40000.0f, -40000.0f, -40000.0f}, {2, 2}, stream);

    std::shared_ptr<StampedCubReduction> e4m3 =
        CubReduction(CubReductionOp::Sum, 1, DataType::FP8_E4M3).stamp(input, stream);
    std::shared_ptr<StampedCubReduction> e5m2 =
        CubReduction(CubReductionOp::Sum, 1, DataType::FP8_E5M2).stamp(input, stream);

    e4m3->run();
    e5m2->run();
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat(e4m3->getOutputTensor(), stream), {448.0f, -448.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat(e5m2->getOutputTensor(), stream), {57344.0f, -57344.0f});
}
#endif

TEST(CubReduction, RuntimeOutputIteratorSupportsEveryConfiguredOutputDtype) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f}, {1, 3}, stream, DataType::FP32);

    std::vector<DataType> output_dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};
#if THOR_CUB_ENABLE_FP8_TYPES
    output_dtypes.push_back(DataType::FP8_E4M3);
    output_dtypes.push_back(DataType::FP8_E5M2);
#endif
#if THOR_CUB_ENABLE_64BIT_TYPES
    output_dtypes.push_back(DataType::FP64);
#endif

    for (DataType output_dtype : output_dtypes) {
        SCOPED_TRACE(static_cast<int>(output_dtype));
        std::shared_ptr<StampedCubReduction> stamped =
            CubReduction(CubReductionOp::Sum, 1, output_dtype).stamp(input, stream);
        stamped->run();
        stream.synchronize();
        EXPECT_EQ(stamped->getOutputDataType(), output_dtype);
        expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {6.0f});
    }
}

TEST(CubReduction, AppliesRuntimeOutputScaleInFp32) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, stream);
    CubReduction reduction(CubReductionOp::Sum, 0, DataType::FP32, 0.25f);
    std::shared_ptr<StampedCubReduction> stamped = reduction.stamp(input, stream);

    EXPECT_FLOAT_EQ(reduction.getOutputScale(), 0.25f);
    EXPECT_FLOAT_EQ(stamped->getOutputScale(), 0.25f);
    stamped->run();
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {1.25f, 1.75f, 2.25f});
}

TEST(CubReduction, QueryOnlyWorkspaceMatchesStampedWorkspace) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, stream, DataType::BF16);
    CubReduction reduction(CubReductionOp::Sum, 0, DataType::FP32, 0.5f);

    const size_t queried = reduction.queryWorkspaceSizeInBytes(input.getDescriptor(), stream);
    std::shared_ptr<StampedCubReduction> stamped = reduction.stamp(input, stream);
    EXPECT_EQ(queried, stamped->getWorkspaceSizeInBytes());
}

TEST(CubReduction, RejectsNonFiniteOutputScale) {
    EXPECT_THROW(static_cast<void>(CubReduction(CubReductionOp::Sum,
                                                0,
                                                DataType::FP32,
                                                std::numeric_limits<float>::infinity())),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(CubReduction(CubReductionOp::Sum,
                                                0,
                                                DataType::FP32,
                                                std::numeric_limits<float>::quiet_NaN())),
                 std::invalid_argument);
}
