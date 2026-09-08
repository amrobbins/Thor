#include "Utilities/TensorOperations/Ragged/RaggedWeightedReduction.h"
#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

namespace {

struct DTypePair {
    DataType values;
    DataType weights;
};

}  // namespace

TEST(RaggedWeightedReduction, ActivePrefixHonorsPoisonedCapacityAndStorageDTypes) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::array<DTypePair, 5> dtypePairs{{
        {DataType::FP32, DataType::FP32},
        {DataType::FP16, DataType::BF16},
        {DataType::BF16, DataType::FP16},
        {DataType::FP8_E4M3, DataType::FP16},
        {DataType::FP8_E5M2, DataType::BF16},
    }};

    constexpr uint64_t maxTotalValues = 9;
    constexpr uint64_t elementsPerValue = 2;
    constexpr uint64_t activeValues = 2;
    constexpr float workspaceSentinel = -12345.0f;
    const std::vector<float> hostValues{
        1.0f, 10.0f, 2.0f, 20.0f,
        1000.0f, 2000.0f, 3000.0f, 4000.0f, 5000.0f, 6000.0f,
        7000.0f, 8000.0f, 9000.0f, 10000.0f,
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
    };
    const std::vector<float> hostWeights{
        1.0f, 2.0f, 3.0f, 4.0f,
        50.0f, 60.0f, 70.0f, 80.0f, 90.0f, 100.0f,
        110.0f, 120.0f, 130.0f, 140.0f,
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
    };

    for (const DTypePair dtypes : dtypePairs) {
        if (!CubSegmentedReduction::isInputDataTypeSupported(dtypes.values) ||
            !CubSegmentedReduction::isInputDataTypeSupported(dtypes.weights)) {
            continue;
        }
        Tensor values = makeGpuTensor(hostValues, {maxTotalValues, elementsPerValue}, stream, dtypes.values);
        Tensor weights = makeGpuTensor(hostWeights, {maxTotalValues, elementsPerValue}, stream, dtypes.weights);
        Tensor partialStatistics(
            gpuPlacement,
            raggedWeightedMeanStatisticsWorkspaceDescriptor(maxTotalValues, elementsPerValue));
        Tensor numerator(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
        Tensor denominator(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
        partialStatistics.fill(workspaceSentinel, stream);

        raggedWeightedMeanStatistics(values,
                                     weights,
                                     partialStatistics,
                                     numerator,
                                     denominator,
                                     activeValues,
                                     maxTotalValues,
                                     elementsPerValue,
                                     stream);
        stream.synchronize();

        // Compute the reference from the actually quantized storage so FP8,
        // FP16, and BF16 conversion differences do not weaken the test.
        const std::vector<float> storedValues = copyGpuTensorAsFloat(values, stream);
        const std::vector<float> storedWeights = copyGpuTensorAsFloat(weights, stream);
        float expectedNumerator = 0.0f;
        float expectedDenominator = 0.0f;
        for (size_t i = 0; i < activeValues * elementsPerValue; ++i) {
            expectedNumerator += storedValues[i] * storedWeights[i];
            expectedDenominator += storedWeights[i];
        }

        const std::vector<float> actualNumerator = copyGpuTensorAsFloat(numerator, stream);
        const std::vector<float> actualDenominator = copyGpuTensorAsFloat(denominator, stream);
        ASSERT_EQ(actualNumerator.size(), 1U);
        ASSERT_EQ(actualDenominator.size(), 1U);
        EXPECT_NEAR(actualNumerator.front(), expectedNumerator, 1.0e-4f);
        EXPECT_NEAR(actualDenominator.front(), expectedDenominator, 1.0e-4f);
        EXPECT_TRUE(std::isfinite(actualNumerator.front()));
        EXPECT_TRUE(std::isfinite(actualDenominator.front()));

        // A one-CTA active prefix finalizes directly into the output scalars;
        // the capacity-sized partial workspace is not touched at all.
        const std::vector<float> workspace = copyGpuTensorAsFloat(partialStatistics, stream);
        for (float value : workspace) EXPECT_FLOAT_EQ(value, workspaceSentinel);
    }
}

TEST(RaggedWeightedReduction, ZeroWeightCanonicalizesNumeratorEvenWhenIgnoredValuesAreNan) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    Tensor values = makeGpuTensor(
        {std::numeric_limits<float>::quiet_NaN(),
         std::numeric_limits<float>::quiet_NaN(),
         3.0f,
         4.0f,
         100.0f,
         200.0f},
        {3, 2},
        stream,
        DataType::FP32);
    Tensor weights =
        makeGpuTensor({0.0f, 0.0f, 0.0f, 0.0f, 9.0f, 9.0f}, {3, 2}, stream, DataType::FP32);
    Tensor partialStatistics(
        gpuPlacement, raggedWeightedMeanStatisticsWorkspaceDescriptor(3, 2));
    Tensor numerator(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor denominator(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));

    // Only row 0 is valid: its active weights are all zero and its values may
    // contain NaN because the batch has no WeightedMean contribution.
    raggedWeightedMeanStatistics(
        values, weights, partialStatistics, numerator, denominator, 2, 3, 2, stream);
    stream.synchronize();

    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(numerator, stream).front(), 0.0f);
    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(denominator, stream).front(), 0.0f);
}


TEST(RaggedWeightedReduction, RuntimePartialCountUsesActivePrefixNotCapacity) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t maxTotalValues = 8192;
    constexpr uint64_t elementsPerValue = 2;
    constexpr uint64_t activeValues = 5000;
    constexpr uint64_t activeScalars = activeValues * elementsPerValue;
    constexpr uint32_t blockSize = 256;
    constexpr uint32_t targetItemsPerThread = 8;
    constexpr uint32_t expectedRuntimePartials =
        static_cast<uint32_t>((activeScalars + blockSize * targetItemsPerThread - 1) /
                              (blockSize * targetItemsPerThread));
    constexpr float workspaceSentinel = -24680.0f;

    const TensorDescriptor workspaceDescriptor =
        raggedWeightedMeanStatisticsWorkspaceDescriptor(maxTotalValues, elementsPerValue);
    ASSERT_EQ(workspaceDescriptor.getDimensions().size(), 2U);
    ASSERT_GT(workspaceDescriptor.getDimensions().front(), expectedRuntimePartials);
    EXPECT_EQ(workspaceDescriptor.getDimensions()[1], 2U);

    std::vector<float> hostValues(maxTotalValues * elementsPerValue,
                                  std::numeric_limits<float>::quiet_NaN());
    std::vector<float> hostWeights(maxTotalValues * elementsPerValue,
                                   std::numeric_limits<float>::quiet_NaN());
    for (uint64_t i = 0; i < activeScalars; ++i) {
        hostValues[i] = static_cast<float>((i % 7) + 1);
        hostWeights[i] = static_cast<float>((i % 3) + 1);
    }

    Tensor values = makeGpuTensor(
        hostValues, {maxTotalValues, elementsPerValue}, stream, DataType::FP32);
    Tensor weights = makeGpuTensor(
        hostWeights, {maxTotalValues, elementsPerValue}, stream, DataType::FP32);
    Tensor partialStatistics(gpuPlacement, workspaceDescriptor);
    Tensor numerator(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor denominator(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    partialStatistics.fill(workspaceSentinel, stream);

    raggedWeightedMeanStatistics(values,
                                 weights,
                                 partialStatistics,
                                 numerator,
                                 denominator,
                                 activeValues,
                                 maxTotalValues,
                                 elementsPerValue,
                                 stream);
    stream.synchronize();

    float expectedNumerator = 0.0f;
    float expectedDenominator = 0.0f;
    for (uint64_t i = 0; i < activeScalars; ++i) {
        expectedNumerator += hostValues[i] * hostWeights[i];
        expectedDenominator += hostWeights[i];
    }

    EXPECT_NEAR(copyGpuTensorAsFloat(numerator, stream).front(), expectedNumerator, 1.0e-2f);
    EXPECT_NEAR(copyGpuTensorAsFloat(denominator, stream).front(), expectedDenominator, 1.0e-2f);
    EXPECT_TRUE(std::isfinite(copyGpuTensorAsFloat(numerator, stream).front()));
    EXPECT_TRUE(std::isfinite(copyGpuTensorAsFloat(denominator, stream).front()));

    // Capacity reserves more partial slots than this active prefix needs. Only
    // the runtime prefix may be written by first-pass CTAs; the rest must retain
    // its sentinel value.
    const std::vector<float> workspace = copyGpuTensorAsFloat(partialStatistics, stream);
    ASSERT_EQ(workspace.size(), workspaceDescriptor.getTotalNumElements());
    ASSERT_GT(workspace.size(), static_cast<size_t>(expectedRuntimePartials * 2));
    EXPECT_NE(workspace.front(), workspaceSentinel);
    for (size_t i = expectedRuntimePartials * 2; i < workspace.size(); ++i)
        EXPECT_FLOAT_EQ(workspace[i], workspaceSentinel);
}

TEST(RaggedWeightedReduction, EmptyActivePrefixFinalizesDirectlyWithoutTouchingWorkspace) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t maxTotalValues = 8192;
    constexpr uint64_t elementsPerValue = 2;
    constexpr float workspaceSentinel = -31415.0f;
    std::vector<float> poison(maxTotalValues * elementsPerValue,
                              std::numeric_limits<float>::quiet_NaN());
    Tensor values = makeGpuTensor(
        poison, {maxTotalValues, elementsPerValue}, stream, DataType::FP32);
    Tensor weights = makeGpuTensor(
        poison, {maxTotalValues, elementsPerValue}, stream, DataType::FP32);
    Tensor partialStatistics(
        gpuPlacement,
        raggedWeightedMeanStatisticsWorkspaceDescriptor(maxTotalValues, elementsPerValue));
    Tensor numerator(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    Tensor denominator(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));
    partialStatistics.fill(workspaceSentinel, stream);

    raggedWeightedMeanStatistics(values,
                                 weights,
                                 partialStatistics,
                                 numerator,
                                 denominator,
                                 /*active_value_count=*/0,
                                 maxTotalValues,
                                 elementsPerValue,
                                 stream);
    stream.synchronize();

    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(numerator, stream).front(), 0.0f);
    EXPECT_FLOAT_EQ(copyGpuTensorAsFloat(denominator, stream).front(), 0.0f);
    const std::vector<float> workspace = copyGpuTensorAsFloat(partialStatistics, stream);
    for (float value : workspace) EXPECT_FLOAT_EQ(value, workspaceSentinel);
}
