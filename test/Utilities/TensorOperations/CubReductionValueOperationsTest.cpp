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
                      {CubReductionOp::L2Norm, {std::sqrt(29.0f)}, 1.0e-5f}},
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
                      {CubReductionOp::L2Norm, {std::sqrt(14.0f), std::sqrt(77.0f)}, 1.0e-5f}},
                     stream);
}

TEST(CubReduction, StridedFixedSegmentPathSupportsEveryValueOperation) {
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
                       1.0e-5f}},
                     stream);
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
    Tensor strided = makeGpuTensor(
        {1.0f, 2.0f, nan, nan, 5.0f, 6.0f, 7.0f, 8.0f, nan, nan, 11.0f, 12.0f},
        {2, 3, 2},
        stream);

    for (CubReductionOp op : {CubReductionOp::Min, CubReductionOp::Max}) {
        SCOPED_TRACE(static_cast<int>(op));
        expectFloatVectorNear(executeFp32Output(whole, op, 0, stream), {nan});
        expectFloatVectorNear(executeFp32Output(contiguous, op, 1, stream), {nan, nan});
        expectFloatVectorNear(executeFp32Output(strided, op, 1, stream), {nan, nan, nan, nan});
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
                              CubReductionOp::L2Norm}) {
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
